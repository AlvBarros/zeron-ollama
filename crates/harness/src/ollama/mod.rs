//! Native Ollama driver — connects directly to Ollama's HTTP API.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::timeout;
use uuid::Uuid;

use zeron_proto::{
    AgentEvent, DoneStatus, HarnessId, Model, ModelOption, ModelOptionChoice, ReasoningLevel,
    RunRequest, SlashCommand, SteeringMode,
};

use crate::{Harness, HarnessError, RunControls};

const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const HEALTH_POLL: Duration = Duration::from_millis(500);
const CALL_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_STALL_BOUND: Duration = Duration::from_secs(60);

const REASONING_LEVELS: &[ReasoningLevel] = &[
    ReasoningLevel::Low,
    ReasoningLevel::Medium,
    ReasoningLevel::High,
    ReasoningLevel::XHigh,
    ReasoningLevel::Max,
];

fn default_ollama_url() -> String {
    std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn default_model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaModel {
    name: String,
    modified_at: String,
    size: i64,
    digest: String,
    details: Option<OllamaModelDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaModelDetails {
    format: String,
    family: String,
    families: Option<Vec<String>>,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: Option<OllamaOptions>,
}

/// Transcripts keyed by the session id handed back in `Done`. Ollama's chat
/// endpoint is stateless, so every turn has to resend the whole conversation.
/// Kept in memory and mirrored to disk (like Pi's session store) so a restart
/// or a different host process still finds the conversation.
fn sessions() -> &'static Mutex<HashMap<String, Vec<OllamaMessage>>> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Vec<OllamaMessage>>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `$ZERON_DATA_DIR/ollama-sessions`, else `~/.zeron/ollama-sessions`.
fn sessions_dir() -> PathBuf {
    std::env::var_os("ZERON_DATA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::executable::home_or_current_dir().join(".zeron"))
        .join("ollama-sessions")
}

fn session_file(session_id: &str) -> PathBuf {
    use sha2::{Digest, Sha256};
    sessions_dir().join(format!("{:x}.json", Sha256::digest(session_id.as_bytes())))
}

fn load_history(session_id: &str) -> Vec<OllamaMessage> {
    if let Some(h) = sessions().lock().unwrap_or_else(|e| e.into_inner()).get(session_id) {
        return h.clone();
    }
    let history: Vec<OllamaMessage> = std::fs::read(session_file(session_id))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    sessions()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(session_id.to_string(), history.clone());
    history
}

fn remember_turn(session_id: &str, prompt: String, reply: String) {
    let mut history = load_history(session_id);
    history.push(OllamaMessage { role: "user".to_string(), content: prompt });
    history.push(OllamaMessage { role: "assistant".to_string(), content: reply });
    let path = session_file(session_id);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
        let temp = dir.join(format!("{}.tmp", Uuid::new_v4()));
        if std::fs::write(&temp, serde_json::to_vec(&history).unwrap_or_default()).is_ok()
            && std::fs::rename(&temp, &path).is_err()
        {
            let _ = std::fs::remove_file(&temp);
        }
    }
    sessions()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(session_id.to_string(), history);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Default)]
struct OllamaOptions {
    temperature: Option<f32>,
    num_predict: Option<i32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    total_duration: Option<i64>,
    load_duration: Option<i64>,
    prompt_eval_count: Option<i32>,
    eval_count: Option<i32>,
    eval_duration: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    model: String,
    created_at: String,
    message: OllamaMessage,
    done: bool,
    total_duration: Option<i64>,
    eval_count: Option<i32>,
}

pub struct OllamaHarness {
    base_url: String,
    default_model: String,
    client: Client,
    models_cache: crate::catalog::Catalog,
    commands_cache: tokio::sync::OnceCell<Vec<SlashCommand>>,
}

impl Default for OllamaHarness {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
            default_model: default_model(),
            client: Client::builder()
                .timeout(CALL_TIMEOUT)
                .build()
                .expect("reqwest client"),
            models_cache: crate::catalog::Catalog::default(),
            commands_cache: tokio::sync::OnceCell::new(),
        }
    }
}

impl OllamaHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    async fn health_check(&self) -> Result<(), HarnessError> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = timeout(CALL_TIMEOUT, self.client.get(&url).send()).await
            .map_err(|_| HarnessError::Protocol("health check timeout".into()))?
            .map_err(|e| HarnessError::Protocol(format!("health check failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(HarnessError::Protocol(format!("Ollama returned {}", resp.status())));
        }
        Ok(())
    }

    async fn fetch_models(&self) -> Result<Vec<Model>, HarnessError> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = timeout(CALL_TIMEOUT, self.client.get(&url).send()).await
            .map_err(|_| HarnessError::Protocol("fetch models timeout".into()))?
            .map_err(|e| HarnessError::Protocol(format!("fetch models failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(HarnessError::Protocol(format!("Ollama returned {}", resp.status())));
        }

        let data: OllamaTagsResponse = resp.json().await
            .map_err(|e| HarnessError::Protocol(format!("parse models failed: {e}")))?;

        let models = data.models.into_iter().map(|m| {
            let id = m.name.clone();
            let label = m.name.clone();
            let description = m.details.as_ref().map(|d| {
                format!("{} • {} • {}", d.family, d.parameter_size, d.quantization_level)
            });
Model {
                    id,
                    label,
                    description,
                    reasoning_levels: REASONING_LEVELS.to_vec(),
                    options: vec![
                        ModelOption {
                            id: "temperature".into(),
                            label: "Temperature".into(),
                            choices: vec![
                                ModelOptionChoice { id: "0.0".into(), label: "0.0 (deterministic)".into() },
                                ModelOptionChoice { id: "0.3".into(), label: "0.3".into() },
                                ModelOptionChoice { id: "0.7".into(), label: "0.7".into() },
                                ModelOptionChoice { id: "1.0".into(), label: "1.0".into() },
                            ],
                            default_choice: "0.7".into(),
                        },
                        ModelOption {
                            id: "num_predict".into(),
                            label: "Max Tokens".into(),
                            choices: vec![
                                ModelOptionChoice { id: "-1".into(), label: "Unlimited".into() },
                                ModelOptionChoice { id: "1024".into(), label: "1024".into() },
                                ModelOptionChoice { id: "2048".into(), label: "2048".into() },
                                ModelOptionChoice { id: "4096".into(), label: "4096".into() },
                            ],
                            default_choice: "-1".into(),
                        },
                    ],
                }
        }).collect();

        Ok(models)
    }
}

#[async_trait]
impl Harness for OllamaHarness {
    fn id(&self) -> HarnessId {
        HarnessId::Ollama
    }

    fn display_name(&self) -> &str {
        "Ollama"
    }

    fn supports_steering(&self) -> bool {
        false
    }

    fn steering_mode(&self) -> SteeringMode {
        SteeringMode::TurnBoundary
    }

    fn reasoning_levels(&self) -> &[ReasoningLevel] {
        REASONING_LEVELS
    }

    fn installed(&self) -> bool {
        true
    }

    fn executable_path(&self) -> Option<PathBuf> {
        None
    }

    fn deterministic_turn_end(&self) -> bool {
        true
    }

    fn authoritative_prompt_end(&self) -> bool {
        true
    }

    async fn model_catalog(&self, force: bool) -> Result<crate::ModelCatalog, HarnessError> {
        let context_key = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(self.base_url.as_bytes());
            hasher.update(b"|");
            hasher.update(self.default_model.as_bytes());
            hasher.finalize().into()
        };
        self.models_cache
            .get_with_timeout(
                force,
                DEFAULT_STARTUP_TIMEOUT * 2,
                || Ok(context_key),
                || self.fetch_models(),
            )
            .await
    }

    async fn models(&self) -> Result<Vec<Model>, HarnessError> {
        self.model_catalog(true).await.map(|c| c.models)
    }

    async fn commands(&self) -> Result<Vec<SlashCommand>, HarnessError> {
        self.commands_cache
            .get_or_try_init(|| async { Ok(Vec::new()) })
            .await
            .cloned()
    }

    async fn run(
        &self,
        request: RunRequest,
        controls: RunControls,
    ) -> Result<BoxStream<'static, Result<AgentEvent, HarnessError>>, HarnessError> {
        let cwd = if request.cwd.is_empty() { None } else { Some(request.cwd.clone()) };
        let model = request.model.as_deref().unwrap_or(&self.default_model);
        let prompt = zeron_proto::invocation::harness_prompt(&request.prompt, self.id());
        let session_id = request.resume.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let history = load_history(&session_id);

        let (event_tx, event_rx) = mpsc::channel::<Result<AgentEvent, HarnessError>>(64);

        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let model_name = model.to_string();
        let interrupt = controls.interrupt.clone();
        let session_id_clone = session_id.clone();

        tokio::spawn(async move {
            let mut messages = history;
            messages.push(OllamaMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            });

            let req = OllamaChatRequest {
                model: model_name.clone(),
                messages,
                stream: true,
                options: None,
            };

            let url = format!("{}/api/chat", base_url);
            let resp = match timeout(CALL_TIMEOUT, client.post(&url).json(&req).send()).await {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => {
                    let _ = event_tx.send(Err(HarnessError::Protocol(format!("request failed: {e}")))).await;
                    let _ = event_tx.send(Ok(AgentEvent::Done {
                        status: DoneStatus::Errored,
                        result: None,
                        error: Some(format!("request failed: {e}")),
                        session_id: Some(session_id_clone),
                    })).await;
                    return;
                }
                Err(_) => {
                    let _ = event_tx.send(Err(HarnessError::Protocol("request timeout".into()))).await;
                    let _ = event_tx.send(Ok(AgentEvent::Done {
                        status: DoneStatus::Errored,
                        result: None,
                        error: Some("request timeout".into()),
                        session_id: Some(session_id_clone),
                    })).await;
                    return;
                }
            };

            if !resp.status().is_success() {
                let err_text = resp.text().await.unwrap_or_default();
                let _ = event_tx.send(Err(HarnessError::Protocol(format!("Ollama error: {err_text}")))).await;
                let _ = event_tx.send(Ok(AgentEvent::Done {
                    status: DoneStatus::Errored,
                    result: None,
                    error: Some(format!("Ollama error: {err_text}")),
                    session_id: Some(session_id_clone),
                })).await;
                return;
            }

            let mut stream = resp.bytes_stream();
            let mut full_content = String::new();

            while let Some(chunk_result) = stream.next().await {
                if interrupt.is_cancelled() {
                    remember_turn(&session_id_clone, prompt.clone(), full_content.clone());
                    let _ = event_tx.send(Ok(AgentEvent::Done {
                        status: DoneStatus::Interrupted,
                        result: Some(full_content),
                        error: None,
                        session_id: Some(session_id_clone),
                    })).await;
                    return;
                }

                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = event_tx.send(Err(HarnessError::Protocol(format!("stream error: {e}")))).await;
                        let _ = event_tx.send(Ok(AgentEvent::Done {
                            status: DoneStatus::Errored,
                            result: Some(full_content),
                            error: Some(format!("stream error: {e}")),
                            session_id: Some(session_id_clone),
                        })).await;
                        return;
                    }
                };

                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if let Ok(chunk_data) = serde_json::from_str::<OllamaStreamChunk>(line) {
                        if !chunk_data.message.content.is_empty() {
                            full_content.push_str(&chunk_data.message.content);
                            let _ = event_tx.send(Ok(AgentEvent::TextDelta {
                                text: chunk_data.message.content,
                            })).await;
                        }
                        if chunk_data.done {
                            remember_turn(&session_id_clone, prompt.clone(), full_content.clone());
                            let _ = event_tx.send(Ok(AgentEvent::Done {
                                status: DoneStatus::Completed,
                                result: Some(full_content),
                                error: None,
                                session_id: Some(session_id_clone),
                            })).await;
                            return;
                        }
                    }
                }
            }

            remember_turn(&session_id_clone, prompt, full_content.clone());
            let _ = event_tx.send(Ok(AgentEvent::Done {
                status: DoneStatus::Completed,
                result: Some(full_content),
                error: None,
                session_id: Some(session_id_clone),
            })).await;
        });

        Ok(futures::stream::unfold(event_rx, |mut rx| async move {
            rx.recv().await.map(|ev| (ev, rx))
        }).boxed())
    }
}