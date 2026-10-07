//! Local Ollama models driven through Claude Code's own agent loop, via
//! Ollama's Anthropic-compatible Messages API
//! (<https://docs.ollama.com/api/anthropic-compatibility>,
//! <https://docs.ollama.com/integrations/claude-code>).
//!
//! Catalog rows are namespaced `ollama/<name>` so a run can tell them apart
//! from Anthropic model ids; the prefix is stripped before `--model`. The
//! endpoint override rides ONLY the spawned child's environment for those
//! runs — the user's real Anthropic credentials stay untouched for every
//! other model and are never sent to the Ollama endpoint.

use std::time::Duration;

use serde_json::Value;
use zeron_proto::Model;

use crate::process::Command;

pub(crate) const MODEL_PREFIX: &str = "ollama/";

/// Discovery must never hold up the Claude picker when Ollama is down.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(3);

/// `OLLAMA_BASE_URL` (shared with the native Ollama harness), else the
/// documented local server.
pub(crate) fn default_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL")
        .ok()
        .map(|v| v.trim().trim_end_matches('/').to_owned())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "http://localhost:11434".into())
}

/// The Ollama model name behind a catalog id, if it is Ollama-backed.
pub(crate) fn model_name(id: &str) -> Option<&str> {
    id.strip_prefix(MODEL_PREFIX)
        .filter(|name| !name.is_empty())
}

/// Point this child at Ollama, per the integration docs:
/// `ANTHROPIC_AUTH_TOKEN=ollama` (required but ignored),
/// `ANTHROPIC_API_KEY=""`, `ANTHROPIC_BASE_URL=<ollama>`.
///
/// Beyond the docs: Claude Code also issues background and subagent requests
/// under its default Haiku/Sonnet/Opus ids, which Ollama would reject, so
/// those aliases resolve to the selected model too. Provider switches and the
/// OAuth token env are cleared so neither can redirect the run or leak the
/// user's credentials to the Ollama endpoint.
pub(crate) fn apply_env(cmd: &mut Command, base_url: &str, model: &str) {
    cmd.env("ANTHROPIC_BASE_URL", base_url)
        .env("ANTHROPIC_AUTH_TOKEN", "ollama")
        .env("ANTHROPIC_API_KEY", "");
    for key in [
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ] {
        cmd.env(key, model);
    }
    for key in [
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ] {
        cmd.env_remove(key);
    }
}

/// Catalog rows from an `/api/tags` reply. Claude Code cannot run without
/// tools, so a model whose reported `capabilities` lack `tools` (completion-
/// only or embedding models) is left out. Older servers that omit
/// `capabilities` list every model.
pub(crate) fn parse_tags(response: &Value) -> Vec<Model> {
    let mut models = Vec::new();
    for entry in response
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(name) = ["name", "model"]
            .iter()
            .find_map(|k| entry.get(*k).and_then(Value::as_str))
            .map(str::trim)
            .filter(|n| !n.is_empty())
        else {
            continue;
        };
        let capabilities: Option<Vec<&str>> = entry
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|caps| caps.iter().filter_map(Value::as_str).collect());
        if capabilities
            .as_ref()
            .is_some_and(|caps| !caps.contains(&"tools"))
        {
            continue;
        }
        let id = format!("{MODEL_PREFIX}{name}");
        if models.iter().any(|m: &Model| m.id == id) {
            continue;
        }
        let details = entry.get("details");
        let detail = |key: &str| {
            details
                .and_then(|d| d.get(key))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty())
        };
        let description = std::iter::once("Ollama")
            .chain(
                entry
                    .get("remote_host")
                    .and_then(Value::as_str)
                    .map(|_| "cloud"),
            )
            .chain(
                ["family", "parameter_size", "quantization_level"]
                    .map(detail)
                    .into_iter()
                    .flatten(),
            )
            .collect::<Vec<_>>()
            .join(" · ");
        let thinking = capabilities
            .as_ref()
            .is_some_and(|caps| caps.contains(&"thinking"));
        models.push(Model {
            id,
            label: name.to_owned(),
            description: Some(description),
            // `--effort` is Anthropic-specific; Ollama accepts but does not
            // enforce thinking budgets.
            reasoning_levels: vec![],
            options: if thinking {
                vec![super::catalog::toggle("thinking", "Thinking")]
            } else {
                vec![]
            },
        });
    }
    models
}

/// Live `/api/tags` discovery against `base_url`.
pub(crate) async fn discover(base_url: &str) -> Result<Vec<Model>, crate::HarnessError> {
    let client = reqwest::Client::builder()
        .timeout(DISCOVERY_TIMEOUT)
        .build()
        .map_err(|e| crate::HarnessError::Protocol(format!("ollama client: {e}")))?;
    let response = client
        .get(format!("{base_url}/api/tags"))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| crate::HarnessError::Protocol(format!("ollama /api/tags: {e}")))?;
    let body: Value = response
        .json()
        .await
        .map_err(|e| crate::HarnessError::Protocol(format!("ollama /api/tags: {e}")))?;
    Ok(parse_tags(&body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_name_requires_prefix() {
        assert_eq!(model_name("ollama/qwen3.8:latest"), Some("qwen3.8:latest"));
        assert_eq!(model_name("ollama/hf.co/a/b:Q4"), Some("hf.co/a/b:Q4"));
        assert_eq!(model_name("ollama/"), None);
        assert_eq!(model_name("claude-opus-5-5"), None);
    }

    #[test]
    fn tags_keep_tool_capable_models_only() {
        let models = parse_tags(&json!({"models": [
            {"name": "qwen3.8:latest", "details": {"family": "qwen35", "parameter_size": "27.3B",
                "quantization_level": "Q4_K_M"}, "capabilities": ["completion", "tools", "thinking"]},
            {"name": "deepseek-coder-v2:16b", "capabilities": ["completion", "insert"]},
            {"name": "nomic-embed-text", "capabilities": ["embedding"]},
            {"name": "glm-5.3-flash:cloud", "remote_host": "https://ollama.com",
                "details": {"family": ""}, "capabilities": ["tools"]},
            {"name": "legacy:7b"},
            {"name": "legacy:7b"},
            {"name": "  "}
        ]}));
        let ids: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "ollama/qwen3.8:latest",
                "ollama/glm-5.3-flash:cloud",
                "ollama/legacy:7b"
            ]
        );
        assert_eq!(models[0].label, "qwen3.8:latest");
        assert_eq!(
            models[0].description.as_deref(),
            Some("Ollama · qwen35 · 27.3B · Q4_K_M")
        );
        assert_eq!(models[0].options[0].id, "thinking");
        assert!(models[0].reasoning_levels.is_empty());
        assert_eq!(models[1].description.as_deref(), Some("Ollama · cloud"));
        assert!(models[1].options.is_empty());
        assert!(parse_tags(&json!({})).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn env_targets_ollama_and_drops_anthropic_credentials() {
        let mut cmd = Command::new("claude");
        apply_env(&mut cmd, "http://127.0.0.1:11434", "qwen3.8:latest");
        let env: std::collections::HashMap<_, _> = cmd
            .as_std()
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_str().unwrap().to_owned(),
                    v.map(|v| v.to_str().unwrap().to_owned()),
                )
            })
            .collect();
        let set = |k: &str| env.get(k).cloned().flatten();
        assert_eq!(
            set("ANTHROPIC_BASE_URL").as_deref(),
            Some("http://127.0.0.1:11434")
        );
        assert_eq!(set("ANTHROPIC_AUTH_TOKEN").as_deref(), Some("ollama"));
        assert_eq!(set("ANTHROPIC_API_KEY").as_deref(), Some(""));
        assert_eq!(
            set("ANTHROPIC_DEFAULT_HAIKU_MODEL").as_deref(),
            Some("qwen3.8:latest")
        );
        assert_eq!(env.get("CLAUDE_CODE_OAUTH_TOKEN"), Some(&None));
        assert_eq!(env.get("CLAUDE_CODE_USE_BEDROCK"), Some(&None));
    }
}
