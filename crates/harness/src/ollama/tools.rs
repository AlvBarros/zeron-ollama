//! Local tools offered to Ollama models: file access and shell commands.
//!
//! Paths are confined to the run's working directory unless the sandbox is
//! `danger-full-access`. Commands are not confined by the sandbox: they run in
//! the working directory with the user's own shell and permissions.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use zeron_proto::{SandboxLevel, ToolCall};

use crate::process::{Command, Stdio};

/// Tool output beyond this many bytes is cut off before it reaches the model.
const MAX_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_SEARCH_MATCHES: usize = 200;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
/// Directories the search tool never descends into.
const SKIPPED_DIRS: &[&str] = &["node_modules", "target"];

pub const READ_FILE: &str = "read_file";
pub const LIST_DIR: &str = "list_dir";
pub const SEARCH: &str = "search";
pub const WRITE_FILE: &str = "write_file";
pub const EDIT_FILE: &str = "edit_file";
pub const RUN_COMMAND: &str = "run_command";

/// Tools a model may call in this run. Empty when there is no working
/// directory to confine them to.
pub fn specs(cwd: &str, sandbox: SandboxLevel) -> Vec<Value> {
    if cwd.is_empty() {
        return Vec::new();
    }
    let mut specs = vec![
        spec(READ_FILE, "Read a UTF-8 text file.", json!({
            "path": {"type": "string", "description": "File path, relative to the working directory."},
        }), &["path"]),
        spec(LIST_DIR, "List the entries of a directory. Directories end with '/'.", json!({
            "path": {"type": "string", "description": "Directory path, relative to the working directory."},
        }), &["path"]),
        spec(SEARCH, "Find lines containing a literal text pattern under a directory.", json!({
            "pattern": {"type": "string", "description": "Literal text to find (not a regex)."},
            "path": {"type": "string", "description": "Directory to search. Defaults to the working directory."},
        }), &["pattern"]),
    ];
    if sandbox != SandboxLevel::ReadOnly {
        specs.push(spec(WRITE_FILE, "Create or overwrite a file with the given content.", json!({
            "path": {"type": "string", "description": "File path, relative to the working directory."},
            "content": {"type": "string", "description": "Full file content."},
        }), &["path", "content"]));
        specs.push(spec(EDIT_FILE, "Replace one exact occurrence of old_string with new_string in a file.", json!({
            "path": {"type": "string", "description": "File path, relative to the working directory."},
            "old_string": {"type": "string", "description": "Exact text to replace; must occur exactly once."},
            "new_string": {"type": "string", "description": "Replacement text."},
        }), &["path", "old_string", "new_string"]));
        specs.push(spec(RUN_COMMAND, "Run a shell command in the working directory and return its output.", json!({
            "command": {"type": "string", "description": "Shell command to run."},
        }), &["command"]));
    }
    specs
}

/// Whether a call changes files or runs a command, and so needs the user's
/// approval unless the run is auto-approved.
pub fn needs_approval(name: &str) -> bool {
    matches!(name, WRITE_FILE | EDIT_FILE | RUN_COMMAND)
}

/// The UI view of a call, shown before it runs.
pub fn describe(name: &str, args: &Value) -> ToolCall {
    let text = |key: &str| args.get(key).and_then(Value::as_str).map(str::to_owned);
    let path = || text("path").unwrap_or_default();
    match name {
        READ_FILE => ToolCall::ReadFile { path: path() },
        WRITE_FILE => ToolCall::WriteFile { path: path(), content: text("content") },
        EDIT_FILE => ToolCall::EditFile {
            path: path(),
            old_string: text("old_string"),
            new_string: text("new_string"),
        },
        SEARCH => ToolCall::Search {
            pattern: text("pattern").unwrap_or_default(),
            path: text("path"),
        },
        RUN_COMMAND => ToolCall::Exec { command: text("command").unwrap_or_default() },
        _ => ToolCall::Unknown { name: name.to_owned(), input: Some(args.clone()) },
    }
}

pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
}

/// Run one tool call. Errors are returned to the model as text rather than
/// failing the turn, so it can recover.
pub async fn execute(
    name: &str,
    args: &Value,
    cwd: &Path,
    sandbox: SandboxLevel,
    interrupt: &CancellationToken,
) -> ToolOutput {
    if sandbox == SandboxLevel::ReadOnly && needs_approval(name) {
        return ToolOutput { text: format!("{name} is not allowed in a read-only run"), is_error: true };
    }
    let result = match name {
        READ_FILE => read_file(args, cwd, sandbox),
        LIST_DIR => list_dir(args, cwd, sandbox),
        SEARCH => search(args, cwd, sandbox),
        WRITE_FILE => write_file(args, cwd, sandbox),
        EDIT_FILE => edit_file(args, cwd, sandbox),
        RUN_COMMAND => run_command(args, cwd, sandbox, interrupt).await,
        _ => Err(format!("unknown tool: {name}")),
    };
    match result {
        Ok(text) => ToolOutput { text: truncate(text), is_error: false },
        Err(text) => ToolOutput { text, is_error: true },
    }
}

fn read_file(args: &Value, cwd: &Path, sandbox: SandboxLevel) -> Result<String, String> {
    let path = resolve_path(cwd, arg(args, "path")?, sandbox)?;
    std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))
}

fn list_dir(args: &Value, cwd: &Path, sandbox: SandboxLevel) -> Result<String, String> {
    let path = resolve_path(cwd, arg(args, "path")?, sandbox)?;
    let entries = std::fs::read_dir(&path).map_err(|e| format!("list {}: {e}", path.display()))?;
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().is_ok_and(|t| t.is_dir()) { format!("{name}/") } else { name }
        })
        .collect();
    names.sort();
    Ok(names.join("\n"))
}

fn search(args: &Value, cwd: &Path, sandbox: SandboxLevel) -> Result<String, String> {
    let pattern = arg(args, "pattern")?;
    if pattern.is_empty() {
        return Err("pattern must not be empty".into());
    }
    let root = match args.get("path").and_then(Value::as_str) {
        Some(path) => resolve_path(cwd, path, sandbox)?,
        None => cwd.to_path_buf(),
    };
    let mut matches = Vec::new();
    let mut pending = vec![root.clone()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                if !name.starts_with('.') && !SKIPPED_DIRS.contains(&name.as_str()) {
                    pending.push(path);
                }
                continue;
            }
            // Unreadable or binary files are skipped rather than failing the search.
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let shown = path.strip_prefix(cwd).unwrap_or(&path).display().to_string();
            for (number, line) in text.lines().enumerate() {
                if line.contains(pattern) {
                    matches.push(format!("{shown}:{}: {line}", number + 1));
                    if matches.len() == MAX_SEARCH_MATCHES {
                        matches.push(format!("[stopped after {MAX_SEARCH_MATCHES} matches]"));
                        return Ok(matches.join("\n"));
                    }
                }
            }
        }
    }
    if matches.is_empty() {
        return Ok(format!("no matches for {pattern:?}"));
    }
    Ok(matches.join("\n"))
}

fn write_file(args: &Value, cwd: &Path, sandbox: SandboxLevel) -> Result<String, String> {
    let path = resolve_path(cwd, arg(args, "path")?, sandbox)?;
    let content = arg(args, "content")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(&path, content).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(format!("wrote {} bytes to {}", content.len(), path.display()))
}

fn edit_file(args: &Value, cwd: &Path, sandbox: SandboxLevel) -> Result<String, String> {
    let path = resolve_path(cwd, arg(args, "path")?, sandbox)?;
    let old = arg(args, "old_string")?;
    let new = arg(args, "new_string")?;
    if old.is_empty() {
        return Err("old_string must not be empty".into());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    match text.matches(old).count() {
        0 => Err(format!("old_string not found in {}", path.display())),
        1 => {
            std::fs::write(&path, text.replacen(old, new, 1))
                .map_err(|e| format!("write {}: {e}", path.display()))?;
            Ok(format!("edited {}", path.display()))
        }
        n => Err(format!("old_string occurs {n} times in {}; make it unique", path.display())),
    }
}

async fn run_command(
    args: &Value,
    cwd: &Path,
    sandbox: SandboxLevel,
    interrupt: &CancellationToken,
) -> Result<String, String> {
    if sandbox == SandboxLevel::ReadOnly {
        return Err("commands are not allowed in a read-only run".into());
    }
    let command = arg(args, "command")?;
    let (shell, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
    let child = Command::new(shell)
        .arg(flag)
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn command: {e}"))?;
    let output = tokio::select! {
        output = child.wait_with_output() => output.map_err(|e| format!("run command: {e}"))?,
        _ = interrupt.cancelled() => return Err("command interrupted".into()),
        _ = tokio::time::sleep(COMMAND_TIMEOUT) => {
            return Err(format!("command timed out after {}s", COMMAND_TIMEOUT.as_secs()));
        }
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&stderr);
    }
    let code = output.status.code().map_or("signal".to_string(), |c| c.to_string());
    text.push_str(&format!("\n[exit {code}]"));
    if output.status.success() { Ok(text) } else { Err(text) }
}

fn arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{key}`"))
}

/// Resolve a tool path against `cwd`, refusing anything outside it unless the
/// sandbox allows full access. Symlinks are followed before the check.
pub fn resolve_path(cwd: &Path, raw: &str, sandbox: SandboxLevel) -> Result<PathBuf, String> {
    let joined = if Path::new(raw).is_absolute() { PathBuf::from(raw) } else { cwd.join(raw) };
    if sandbox == SandboxLevel::DangerFullAccess {
        return Ok(joined);
    }
    let root = cwd
        .canonicalize()
        .map_err(|e| format!("working directory {}: {e}", cwd.display()))?;
    let resolved = canonicalize_existing_prefix(&lexical_normalize(&joined))
        .map_err(|e| format!("resolve {raw}: {e}"))?;
    if resolved.starts_with(&root) {
        Ok(resolved)
    } else {
        Err(format!("path is outside the working directory: {raw}"))
    }
}

/// Lexically drop `.` and resolve `..`, so a path that does not exist yet
/// cannot escape through a `..` segment.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Canonicalize the longest existing ancestor (following symlinks), then
/// re-append the components that do not exist yet.
fn canonicalize_existing_prefix(path: &Path) -> std::io::Result<PathBuf> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        match current.canonicalize() {
            Ok(base) => return Ok(missing.iter().rev().fold(base, |acc, part| acc.join(part))),
            Err(err) => match (current.parent(), current.file_name()) {
                (Some(parent), Some(name)) => {
                    missing.push(name.to_owned());
                    current = parent;
                }
                _ => return Err(err),
            },
        }
    }
}

fn truncate(mut text: String) -> String {
    if text.len() <= MAX_OUTPUT_BYTES {
        return text;
    }
    let mut cut = MAX_OUTPUT_BYTES;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text.truncate(cut);
    text.push_str("\n[output truncated]");
    text
}

fn spec(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cwd() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn read_only_runs_offer_no_mutating_tools() {
        let dir = cwd();
        let names: Vec<String> = specs(dir.path().to_str().unwrap(), SandboxLevel::ReadOnly)
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, [READ_FILE, LIST_DIR, SEARCH]);
    }

    #[test]
    fn no_tools_without_a_working_directory() {
        assert!(specs("", SandboxLevel::DangerFullAccess).is_empty());
    }

    #[test]
    fn paths_outside_the_working_directory_are_refused() {
        let dir = cwd();
        let root = dir.path();
        assert!(resolve_path(root, "../escape.txt", SandboxLevel::WorkspaceWrite).is_err());
        assert!(resolve_path(root, "/etc/passwd", SandboxLevel::WorkspaceWrite).is_err());
        assert!(resolve_path(root, "a/../../escape", SandboxLevel::WorkspaceWrite).is_err());
        assert!(resolve_path(root, "/etc/passwd", SandboxLevel::DangerFullAccess).is_ok());
    }

    #[test]
    fn new_files_inside_the_working_directory_are_allowed() {
        let dir = cwd();
        let path = resolve_path(dir.path(), "src/new/file.rs", SandboxLevel::WorkspaceWrite).unwrap();
        assert!(path.starts_with(dir.path().canonicalize().unwrap()));
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_pointing_outside_are_refused() {
        let dir = cwd();
        let outside = cwd();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
        assert!(resolve_path(dir.path(), "link/secret", SandboxLevel::WorkspaceWrite).is_err());
    }

    #[tokio::test]
    async fn edit_requires_a_unique_match() {
        let dir = cwd();
        std::fs::write(dir.path().join("f.txt"), "one two one").unwrap();
        let interrupt = CancellationToken::new();
        let args = json!({"path": "f.txt", "old_string": "one", "new_string": "1"});
        let out = execute(EDIT_FILE, &args, dir.path(), SandboxLevel::WorkspaceWrite, &interrupt).await;
        assert!(out.is_error);
        assert!(out.text.contains("occurs 2 times"));

        let args = json!({"path": "f.txt", "old_string": "two", "new_string": "2"});
        let out = execute(EDIT_FILE, &args, dir.path(), SandboxLevel::WorkspaceWrite, &interrupt).await;
        assert!(!out.is_error);
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).unwrap(), "one 2 one");
    }

    #[tokio::test]
    async fn search_reports_file_and_line() {
        let dir = cwd();
        std::fs::write(dir.path().join("a.txt"), "alpha\nbeta needle\n").unwrap();
        let interrupt = CancellationToken::new();
        let args = json!({"pattern": "needle"});
        let out = execute(SEARCH, &args, dir.path(), SandboxLevel::ReadOnly, &interrupt).await;
        assert_eq!(out.text, "a.txt:2: beta needle");
    }

    #[test]
    fn long_output_is_truncated_on_a_char_boundary() {
        let text = "é".repeat(MAX_OUTPUT_BYTES);
        let out = truncate(text);
        assert!(out.ends_with("[output truncated]"));
    }

    #[tokio::test]
    async fn commands_are_refused_when_read_only() {
        let dir = cwd();
        let interrupt = CancellationToken::new();
        let args = json!({"command": "echo hi"});
        let out = execute(RUN_COMMAND, &args, dir.path(), SandboxLevel::ReadOnly, &interrupt).await;
        assert!(out.is_error);
    }
}
