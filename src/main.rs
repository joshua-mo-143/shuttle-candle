use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use axum::{
    routing::{get, post},
    Router,
};
use serde::de::Deserializer;
use serde::Deserialize;
use std::collections::HashSet;

mod llm;
use llm::TokenOutputStream;

use anyhow::{Error as E, Result};

use candle_transformers::models::mistral::{Config, Model as Mistral};

use hf_hub::api::sync::ApiRepo;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
pub struct Prompt {
    prompt: String,
}

#[derive(Clone)]
pub struct AppState {
    model: Mistral,
    device: Device,
    tokenizer: Tokenizer,
}

impl From<(Mistral, Device, Tokenizer)> for AppState {
    fn from(e: (Mistral, Device, Tokenizer)) -> Self {
        Self {
            model: e.0,
            device: e.1,
            tokenizer: e.2,
        }
    }
}

impl From<AppState> for TextGeneration {
    fn from(e: AppState) -> Self {
        Self::new(
            e.model,
            e.tokenizer,
            299792458, // seed RNG
            Some(0.),  // temperature
            None,      // top_p - Nucleus sampling probability stuff
            1.1,       // repeat penalty
            64,        // context size to consider for the repeat penalty
            &e.device,
        )
    }
}

async fn hello_world() -> &'static str {
    "Hello, world!"
}

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(api_token) = std::env::var("HF_TOKEN") else {
        return Err(anyhow::anyhow!("Error getting HF_TOKEN env var"))
    };
    let state = initialise_model(api_token)?;

    let router = Router::new()
        .route("/", get(hello_world))
        .route("/prompt", post(run_pipeline))
        .with_state(state);

    let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await.unwrap();

    axum::serve(tcp_listener, router).await.unwrap();

    Ok(())
}

async fn run_pipeline(
    State(state): State<AppState>,
    Json(Prompt { prompt }): Json<Prompt>,
) -> impl IntoResponse {
    let ai_gen = TextGeneration::from(state);
    ai_gen.run(prompt, 20)
}

struct TextGeneration {
    model: Mistral,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Mistral,
        tokenizer: Tokenizer,
        seed: u64,
        _temp: Option<f64>,
        _top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, Some(0.0), None);

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(mut self, prompt: String, sample_len: usize) -> String {
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        println!("Got tokens!");

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => panic!("cannot find the </s> token"),
        };

        let mut string = String::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self.model.forward(&input, start_pos).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                println!("Found a token!");
                string.push_str(&t);
            }
        }

        string
    }
}

fn get_repo(token: String) -> Result<ApiRepo> {
    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let model_id = "mistralai/Mistral-7B-v0.1".to_string();

   Ok(api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "26bca36bde8333b5d7f72e9ed20ccda6a618af24".to_string(),
    )))
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;

    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

fn initialise_model(token: String) -> Result<AppState> {
    let repo = get_repo(token)?;
    let tokenizer = get_tokenizer(&repo)?;
    let device = Device::Cpu;
    let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;

    let config = Config::config_7b_v0_1(false);

    let model = {
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        Mistral::new(&config, vb)?
    };

    Ok((model, device, tokenizer).into())
}

#[derive(Debug, Deserialize)]
struct Weightmaps {
  #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

 // Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(deserializer: D) -> Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: Weightmaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(pathbufs)
}
