//! Native Ollama driver — connects directly to Ollama's HTTP API.

mod tools;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use zeron_proto::{
    AgentEvent, DoneStatus, HarnessId, Model, ModelOption, ModelOptionChoice, ReasoningLevel,
    RunRequest, SandboxLevel, SlashCommand, SteeringMode, UserInputQuestion,
};

use crate::{Harness, HarnessError, RunControls};

use tools::ToolOutput;

const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const HEALTH_POLL: Duration = Duration::from_millis(500);
const CALL_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_STALL_BOUND: Duration = Duration::from_secs(60);
/// Model round-trips allowed in one turn before the loop gives up.
const MAX_TOOL_STEPS: usize = 25;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
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

/// Append one finished turn (user prompt, tool calls and results, replies).
fn remember_turn(session_id: &str, turn: Vec<OllamaMessage>) {
    let mut history = load_history(session_id);
    history.extend(turn);
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
    #[serde(default)]
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
}

impl OllamaMessage {
    fn text(role: &str, content: String) -> Self {
        Self { role: role.to_string(), content, tool_calls: None, tool_name: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaFunctionCall {
    name: String,
    arguments: Value,
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

/// Everything one turn needs, detached from the harness so the turn can run
/// on its own task.
struct TurnJob {
    base_url: String,
    client: Client,
    model: String,
    prompt: String,
    session_id: String,
    history: Vec<OllamaMessage>,
    cwd: PathBuf,
    sandbox: SandboxLevel,
    auto_approve: bool,
    tools: Vec<Value>,
}

/// Why one model round-trip failed. `ToolsUnsupported` lets the loop retry
/// the same step without tools for models that cannot call them.
enum StepError {
    ToolsUnsupported,
    Failed(String),
}

/// One model round-trip: the text it produced and the tool calls it asked for.
struct Step {
    tool_calls: Vec<OllamaToolCall>,
    interrupted: bool,
}

/// Drive one turn to completion: stream model steps, run the tools they
/// request, and repeat until the model answers without tool calls.
async fn run_turn(
    job: TurnJob,
    controls: RunControls,
    events: mpsc::Sender<Result<AgentEvent, HarnessError>>,
) {
    let session_id = job.session_id.clone();
    let mut messages = job.history.clone();
    let turn_start = messages.len();
    messages.push(OllamaMessage::text("user", job.prompt.clone()));

    let mut reply = String::new();
    match agent_loop(&job, &controls, &events, &mut messages, &mut reply).await {
        Ok(status) => {
            remember_turn(&session_id, messages[turn_start..].to_vec());
            let _ = events.send(Ok(AgentEvent::Done {
                status,
                result: Some(reply),
                error: None,
                session_id: Some(session_id),
            })).await;
        }
        Err(message) => {
            let _ = events.send(Err(HarnessError::Protocol(message.clone()))).await;
            let _ = events.send(Ok(AgentEvent::Done {
                status: DoneStatus::Errored,
                result: None,
                error: Some(message),
                session_id: Some(session_id),
            })).await;
        }
    }
}

async fn agent_loop(
    job: &TurnJob,
    controls: &RunControls,
    events: &mpsc::Sender<Result<AgentEvent, HarnessError>>,
    messages: &mut Vec<OllamaMessage>,
    reply: &mut String,
) -> Result<DoneStatus, String> {
    let mut tools = job.tools.clone();
    for _ in 0..MAX_TOOL_STEPS {
        let step = loop {
            match stream_step(job, &tools, &controls.interrupt, events, messages, reply).await {
                Ok(step) => break step,
                Err(StepError::ToolsUnsupported) => tools.clear(),
                Err(StepError::Failed(message)) => return Err(message),
            }
        };
        if step.interrupted {
            return Ok(DoneStatus::Interrupted);
        }
        if step.tool_calls.is_empty() {
            return Ok(DoneStatus::Completed);
        }
        for call in step.tool_calls {
            run_tool_call(job, controls, events, messages, call).await;
        }
        if controls.interrupt.is_cancelled() {
            return Ok(DoneStatus::Interrupted);
        }
    }
    Err(format!("stopped after {MAX_TOOL_STEPS} tool steps without a final answer"))
}

/// Stream one model response. Appends the assistant message to `messages`
/// and its text to `reply`.
async fn stream_step(
    job: &TurnJob,
    tools: &[Value],
    interrupt: &CancellationToken,
    events: &mpsc::Sender<Result<AgentEvent, HarnessError>>,
    messages: &mut Vec<OllamaMessage>,
    reply: &mut String,
) -> Result<Step, StepError> {
    let request = OllamaChatRequest {
        model: job.model.clone(),
        messages: messages.clone(),
        stream: true,
        options: None,
        tools: (!tools.is_empty()).then(|| tools.to_vec()),
    };
    let url = format!("{}/api/chat", job.base_url);
    let resp = match timeout(CALL_TIMEOUT, job.client.post(&url).json(&request).send()).await {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => return Err(StepError::Failed(format!("request failed: {e}"))),
        Err(_) => return Err(StepError::Failed("request timeout".into())),
    };
    if !resp.status().is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        if !tools.is_empty() && err_text.contains("does not support tools") {
            return Err(StepError::ToolsUnsupported);
        }
        return Err(StepError::Failed(format!("Ollama error: {err_text}")));
    }

    let mut stream = resp.bytes_stream();
    // Network chunks can split a JSON line; buffer bytes until a newline.
    let mut pending: Vec<u8> = Vec::new();
    let mut content = String::new();
    let mut tool_calls: Vec<OllamaToolCall> = Vec::new();
    let mut done = false;
    let mut interrupted = false;
    'read: while let Some(chunk) = stream.next().await {
        if interrupt.is_cancelled() {
            interrupted = true;
            break;
        }
        let chunk = chunk.map_err(|e| StepError::Failed(format!("stream error: {e}")))?;
        pending.extend_from_slice(&chunk);
        while let Some(end) = pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = pending.drain(..=end).collect();
            if absorb_line(&line, &mut content, &mut tool_calls, events).await {
                done = true;
                break 'read;
            }
        }
    }
    if !done && !interrupted && !pending.is_empty() {
        absorb_line(&pending, &mut content, &mut tool_calls, events).await;
    }

    // An interrupted step keeps its text but drops tool calls it never got
    // to run, so the saved history never holds calls without results.
    if interrupted {
        tool_calls.clear();
    }
    messages.push(OllamaMessage {
        role: "assistant".to_string(),
        content: content.clone(),
        tool_calls: (!tool_calls.is_empty()).then(|| tool_calls.clone()),
        tool_name: None,
    });
    reply.push_str(&content);
    Ok(Step { tool_calls, interrupted })
}

/// Handle one complete JSON line of the stream. Returns true on the final chunk.
async fn absorb_line(
    line: &[u8],
    content: &mut String,
    tool_calls: &mut Vec<OllamaToolCall>,
    events: &mpsc::Sender<Result<AgentEvent, HarnessError>>,
) -> bool {
    let text = String::from_utf8_lossy(line);
    let text = text.trim();
    if text.is_empty() {
        return false;
    }
    let Ok(chunk) = serde_json::from_str::<OllamaStreamChunk>(text) else {
        return false;
    };
    if !chunk.message.content.is_empty() {
        content.push_str(&chunk.message.content);
        let _ = events.send(Ok(AgentEvent::TextDelta {
            text: chunk.message.content,
        })).await;
    }
    if let Some(calls) = chunk.message.tool_calls {
        tool_calls.extend(calls);
    }
    chunk.done
}

/// Run one tool call the model asked for, asking first when it changes state,
/// and record the result in the transcript.
async fn run_tool_call(
    job: &TurnJob,
    controls: &RunControls,
    events: &mpsc::Sender<Result<AgentEvent, HarnessError>>,
    messages: &mut Vec<OllamaMessage>,
    call: OllamaToolCall,
) {
    let name = call.function.name;
    let args = call.function.arguments;
    let id = Uuid::new_v4().to_string();
    let _ = events.send(Ok(AgentEvent::ToolCall {
        id: id.clone(),
        call: tools::describe(&name, &args),
    })).await;

    let output = if controls.interrupt.is_cancelled() {
        ToolOutput { text: "not run: the turn was interrupted".into(), is_error: true }
    } else if tools::needs_approval(&name) && !job.auto_approve && !approve(controls, &name, &args).await {
        ToolOutput { text: "the user denied this tool call".into(), is_error: true }
    } else {
        tools::execute(&name, &args, &job.cwd, job.sandbox, &controls.interrupt).await
    };

    let _ = events.send(Ok(AgentEvent::ToolResult {
        id,
        is_error: output.is_error,
        output: Some(output.text.clone()),
        diff: None,
    })).await;
    messages.push(OllamaMessage {
        role: "tool".to_string(),
        content: output.text,
        tool_calls: None,
        tool_name: Some(name),
    });
}

/// Ask the user to allow a state-changing call. Anything but an explicit
/// "Allow" (including a dropped request) counts as a denial.
async fn approve(controls: &RunControls, name: &str, args: &Value) -> bool {
    let question = UserInputQuestion {
        id: "approve-tool".to_string(),
        header: format!("Allow {name}?"),
        question: format!("{name} {args}"),
        options: vec!["Allow".to_string(), "Deny".to_string()],
        multi_select: false,
        prefill: None,
        multiline: false,
    };
    let answers = (controls.request_input)(vec![question]).await;
    answers.is_ok_and(|answers| {
        answers.iter().any(|a| a.labels.iter().any(|label| label == "Allow"))
    })
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
        let model = request.model.as_deref().unwrap_or(&self.default_model);
        let prompt = zeron_proto::invocation::harness_prompt(&request.prompt, self.id());
        let session_id = request.resume.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

        let job = TurnJob {
            base_url: self.base_url.clone(),
            client: self.client.clone(),
            model: model.to_string(),
            prompt,
            history: load_history(&session_id),
            session_id,
            cwd: PathBuf::from(&request.cwd),
            sandbox: request.sandbox,
            auto_approve: request.auto_approve,
            tools: tools::specs(&request.cwd, request.sandbox),
        };

        let (event_tx, event_rx) = mpsc::channel::<Result<AgentEvent, HarnessError>>(64);
        // The turn owns `controls`, so the execution lease it holds stays
        // alive until the turn has finished.
        tokio::spawn(run_turn(job, controls, event_tx));

        Ok(futures::stream::unfold(event_rx, |mut rx| async move {
            rx.recv().await.map(|ev| (ev, rx))
        }).boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use zeron_proto::{ToolCall, UserInputAnswer};

    /// Serves one scripted reply per request. Each reply is written in pieces
    /// so the stream parser also sees JSON lines split across TCP reads.
    /// Returns the base URL and the request bodies the server received.
    async fn fake_ollama(replies: Vec<Vec<&'static str>>) -> (String, Arc<Mutex<Vec<Value>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&bodies);
        tokio::spawn(async move {
            for reply in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let body = read_request_body(&mut socket).await;
                seen.lock().unwrap().push(serde_json::from_slice(&body).unwrap());
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n")
                    .await
                    .unwrap();
                for piece in reply {
                    socket.write_all(piece.as_bytes()).await.unwrap();
                    socket.flush().await.unwrap();
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                socket.shutdown().await.ok();
            }
        });
        (url, bodies)
    }

    async fn read_request_body(socket: &mut TcpStream) -> Vec<u8> {
        let mut buf = Vec::new();
        let header_end = loop {
            let mut chunk = [0u8; 1024];
            let n = socket.read(&mut chunk).await.unwrap();
            buf.extend_from_slice(&chunk[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let headers = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
        let length: usize = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .map(|v| v.trim().parse().unwrap())
            .unwrap_or(0);
        while buf.len() < header_end + length {
            let mut chunk = [0u8; 1024];
            let n = socket.read(&mut chunk).await.unwrap();
            buf.extend_from_slice(&chunk[..n]);
        }
        buf[header_end..header_end + length].to_vec()
    }

    /// Controls whose approval prompts are all answered with `answer`.
    fn controls(answer: &'static str) -> RunControls {
        let (_steer_tx, steering) = mpsc::channel(1);
        RunControls {
            execution_lease: None,
            request_input: Box::new(move |questions| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                let answers = questions
                    .iter()
                    .map(|q| UserInputAnswer {
                        question_id: q.id.clone(),
                        labels: vec![answer.to_string()],
                    })
                    .collect();
                let _ = tx.send(answers);
                rx
            }),
            steering,
            interrupt: CancellationToken::new(),
        }
    }

    fn job(base_url: String, cwd: &Path, sandbox: SandboxLevel, auto_approve: bool) -> TurnJob {
        TurnJob {
            base_url,
            client: Client::new(),
            model: "test-model".to_string(),
            prompt: "do the task".to_string(),
            session_id: Uuid::new_v4().to_string(),
            history: Vec::new(),
            cwd: cwd.to_path_buf(),
            sandbox,
            auto_approve,
            tools: tools::specs(cwd.to_str().unwrap(), sandbox),
        }
    }

    /// Run a turn to completion and return its events. Removes the persisted
    /// session so tests leave no transcripts behind.
    async fn run(job: TurnJob, controls: RunControls) -> Vec<AgentEvent> {
        let session_id = job.session_id.clone();
        let (tx, mut rx) = mpsc::channel(64);
        tokio::spawn(run_turn(job, controls, tx));
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event.unwrap_or_else(|e| panic!("turn failed: {e}")));
        }
        let _ = std::fs::remove_file(session_file(&session_id));
        sessions().lock().unwrap().remove(&session_id);
        events
    }

    fn text_deltas(events: &[AgentEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn tool_call_round_trip_feeds_the_result_back() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "hello").unwrap();
        let (base_url, bodies) = fake_ollama(vec![
            vec![
                "{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"read_file\",\"arguments\":{\"path\":\"notes.txt\"}}}]},\"done\":false}\n",
                "{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
            ],
            vec![
                "{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"The file says \"},\"done\":false}\n{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"hel",
                "lo.\"},\"done\":false}\n",
                "{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
            ],
        ])
        .await;

        let events = run(job(base_url, dir.path(), SandboxLevel::WorkspaceWrite, false), controls("Allow")).await;

        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolCall { call: ToolCall::ReadFile { path }, .. } if path == "notes.txt"
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolResult { output: Some(out), is_error: false, .. } if out == "hello"
        )));
        assert_eq!(text_deltas(&events), "The file says hello.");
        assert!(matches!(
            events.last(),
            Some(AgentEvent::Done { status: DoneStatus::Completed, result: Some(r), .. }) if r == "The file says hello."
        ));

        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        let messages = bodies[1]["messages"].as_array().unwrap();
        assert!(messages.iter().any(|m| {
            m["role"] == "tool" && m["content"] == "hello" && m["tool_name"] == "read_file"
        }));
    }

    #[tokio::test]
    async fn denied_write_leaves_the_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let (base_url, bodies) = fake_ollama(vec![
            vec![
                "{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"out.txt\",\"content\":\"x\"}}}]},\"done\":true}\n",
            ],
            vec![
                "{\"model\":\"m\",\"created_at\":\"t\",\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"done\":true}\n",
            ],
        ])
        .await;

        let events = run(job(base_url, dir.path(), SandboxLevel::WorkspaceWrite, false), controls("Deny")).await;

        assert!(!dir.path().join("out.txt").exists());
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolResult { is_error: true, output: Some(out), .. } if out == "the user denied this tool call"
        )));
        let bodies = bodies.lock().unwrap();
        let messages = bodies[1]["messages"].as_array().unwrap();
        assert!(messages.iter().any(|m| m["role"] == "tool" && m["tool_name"] == "write_file"));
    }
}
