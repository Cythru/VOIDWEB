//! Oracle HTTP Server
//!
//! OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming.
//! Built on Axum + Hyper — zero Python, zero GIL.
//!
//! Endpoints:
//!   POST /v1/chat/completions   — chat inference (stream or non-stream)
//!   GET  /v1/models             — list loaded models
//!   GET  /health                — liveness probe

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{State, Json},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
    Router,
};
use serde::{Serialize, Deserialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, warn, error};
use tracing_subscriber::EnvFilter;

use engine::Engine;
use scheduler::{SamplingParams, RequestId};

// ── App State ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    engine: Arc<tokio::sync::Mutex<Engine>>,
    model_id: String,
}

// ── OpenAI-compatible types ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    model:       Option<String>,
    messages:    Vec<ChatMessage>,
    max_tokens:  Option<u32>,
    temperature: Option<f32>,
    top_p:       Option<f32>,
    top_k:       Option<u32>,
    stream:      Option<bool>,
    #[serde(default)]
    stop:        Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role:    String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id:      String,
    object:  &'static str,
    created: u64,
    model:   String,
    choices: Vec<Choice>,
    usage:   Usage,
}

#[derive(Serialize)]
struct Choice {
    index:         u32,
    message:       ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct StreamChunk {
    id:      String,
    object:  &'static str,
    created: u64,
    model:   String,
    choices: Vec<StreamChoice>,
}

#[derive(Serialize)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role:    Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize, Default)]
struct Usage {
    prompt_tokens:     u32,
    completion_tokens: u32,
    total_tokens:      u32,
}

#[derive(Serialize)]
struct ModelInfo {
    id:      String,
    object:  &'static str,
    created: u64,
    owned_by: String,
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("oracle_server=info".parse()?)
            .add_directive("tower_http=warn".parse()?))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1)
        .cloned()
        .unwrap_or_else(|| "/models/default".into());
    let kernel_lib = args.get(2)
        .cloned()
        .unwrap_or_else(|| "./libkernels.so".into());
    let host = args.get(3).cloned().unwrap_or_else(|| "0.0.0.0".into());
    let port: u16 = args.get(4).and_then(|p| p.parse().ok()).unwrap_or(8000);

    info!("Oracle inference server starting — model={model_path} kernel_lib={kernel_lib}");

    // Load model config from the model directory.
    let config_json = std::fs::read_to_string(format!("{model_path}/config.json"))
        .unwrap_or_else(|_| "{}".into());
    let config_val: serde_json::Value = serde_json::from_str(&config_json)?;

    let quant = quantization::detect_quant_scheme(&config_val);
    let model_id = config_val["model_type"].as_str().unwrap_or("oracle").to_string();

    let cfg = engine::ModelConfig {
        model_type:     model_id.clone(),
        hidden_size:    config_val["hidden_size"].as_u64().unwrap_or(4096) as usize,
        num_heads:      config_val["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_kv_heads:   config_val["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        num_layers:     config_val["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        head_dim:       0, // computed
        vocab_size:     config_val["vocab_size"].as_u64().unwrap_or(32000) as usize,
        max_seq_len:    config_val["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
        rope_theta:     config_val["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        rms_norm_eps:   config_val["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        quant_scheme:   quant,
        weights_path:   model_path.clone().into(),
        tokenizer_path: None,
    };

    let engine = Engine::load(cfg, std::path::Path::new(&kernel_lib))?;
    let state = AppState {
        engine:   Arc::new(tokio::sync::Mutex::new(engine)),
        model_id: model_id.clone(),
    };

    // Spawn the continuous-batching inference loop.
    {
        let eng = state.engine.clone();
        tokio::spawn(async move {
            loop {
                let result = {
                    let mut e = eng.lock().await;
                    e.step()
                };
                match result {
                    Ok(toks) if toks.is_empty() => {
                        tokio::time::sleep(Duration::from_micros(100)).await;
                    }
                    Ok(_) => {}
                    Err(e) => warn!("Engine step error: {e}"),
                }
            }
        });
    }

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::new(host.parse()?, port);
    info!("Listening on http://{addr}");
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn health() -> &'static str { "ok" }

async fn list_models(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": s.model_id,
            "object": "model",
            "created": unix_now(),
            "owned_by": "oracle"
        }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req):    Json<ChatRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(false);
    let params = SamplingParams {
        temperature:        req.temperature.unwrap_or(0.7),
        top_p:              req.top_p.unwrap_or(0.95),
        top_k:              req.top_k.unwrap_or(50),
        max_new_tokens:     req.max_tokens.unwrap_or(2048),
        repetition_penalty: 1.1,
        stop_tokens:        vec![],
    };

    // Build a prompt string from messages, then tokenize.
    let prompt = build_prompt(&req.messages);
    let prompt_tokens = {
        let eng = state.engine.lock().await;
        eng.tokenizer.encode(&prompt, true)
    };
    let prompt_len = prompt_tokens.len() as u32;

    // Submit to scheduler.
    let req_id = {
        let eng = state.engine.lock().await;
        eng.scheduler.lock().add_request(prompt_tokens, params)
    };

    if stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
        let eng = state.engine.clone();
        let model_id = state.model_id.clone();

        tokio::spawn(async move {
            stream_tokens(eng, req_id, model_id, tx, prompt_len).await;
        });

        let stream = ReceiverStream::new(rx);
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        // Non-streaming: wait for completion.
        let output = wait_for_completion(state.engine.clone(), req_id).await;
        let completion_tokens = output.len() as u32;
        let text = {
            let eng = state.engine.lock().await;
            eng.tokenizer.decode(&output)
        };
        Json(ChatResponse {
            id:      format!("chatcmpl-{req_id}"),
            object:  "chat.completion",
            created: unix_now(),
            model:   state.model_id,
            choices: vec![Choice {
                index:         0,
                message:       ChatMessage { role: "assistant".into(), content: text },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_len + completion_tokens,
            },
        }).into_response()
    }
}

// ── Streaming helper ──────────────────────────────────────────────────────────

async fn stream_tokens(
    engine:       Arc<tokio::sync::Mutex<Engine>>,
    req_id:       RequestId,
    model_id:     String,
    tx:           mpsc::Sender<Result<Event, Infallible>>,
    prompt_len:   u32,
) {
    // First chunk: role delta.
    let first = StreamChunk {
        id:      format!("chatcmpl-{req_id}"),
        object:  "chat.completion.chunk",
        created: unix_now(),
        model:   model_id.clone(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta { role: Some("assistant".into()), content: None },
            finish_reason: None,
        }],
    };
    let _ = tx.send(Ok(Event::default().data(serde_json::to_string(&first).unwrap()))).await;

    let mut completion_tokens = 0u32;
    let poll_interval = Duration::from_micros(200);

    loop {
        tokio::time::sleep(poll_interval).await;

        let (finished, new_tokens) = {
            let eng = engine.lock().await;
            let finished_list = eng.scheduler.lock().drain_finished();
            let found = finished_list.into_iter().find(|(id, _)| *id == req_id);
            (found.is_some(), found.map(|(_, toks)| toks))
        };

        if finished {
            if let Some(toks) = new_tokens {
                let text = {
                    let eng = engine.lock().await;
                    eng.tokenizer.decode(&toks)
                };
                let chunk = make_content_chunk(req_id, &model_id, &text, Some("stop"));
                let _ = tx.send(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))).await;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            break;
        }
    }
}

fn make_content_chunk(req_id: RequestId, model: &str, text: &str, finish: Option<&str>) -> StreamChunk {
    StreamChunk {
        id:      format!("chatcmpl-{req_id}"),
        object:  "chat.completion.chunk",
        created: unix_now(),
        model:   model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta { role: None, content: Some(text.to_string()) },
            finish_reason: finish.map(String::from),
        }],
    }
}

async fn wait_for_completion(engine: Arc<tokio::sync::Mutex<Engine>>, req_id: RequestId) -> Vec<u32> {
    loop {
        tokio::time::sleep(Duration::from_millis(5)).await;
        let finished = {
            let eng = engine.lock().await;
            let list = eng.scheduler.lock().drain_finished();
            list.into_iter().find(|(id, _)| *id == req_id)
        };
        if let Some((_, toks)) = finished { return toks; }
    }
}

// ── Prompt builder ─────────────────────────────────────────────────────────────

fn build_prompt(messages: &[ChatMessage]) -> String {
    // ChatML format (used by Qwen, Llama 3, etc.)
    let mut s = String::new();
    for msg in messages {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
