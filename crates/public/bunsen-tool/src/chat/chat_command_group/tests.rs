use clap::Parser;

use super::*;

/// `chat`'s commands alone, as `bunsen-tool chat` parses them.
#[derive(clap::Parser, Debug)]
struct Args {
    #[clap(subcommand)]
    command: ChatSubCommand,
}

/// Thinking is Qwen3's: its family and the family-agnostic commands
/// take the flags, nanochat's does not.
#[test]
fn test_family_flags() {
    let parse = |args: &[&str]| Args::try_parse_from([&["chat"], args].concat());
    assert!(parse(&["run", "--think", "hi"]).is_ok());
    assert!(parse(&["family", "qwen3", "run", "--no-think", "hi"]).is_ok());
    assert!(parse(&["family", "nanochat", "run", "hi"]).is_ok());
    assert!(parse(&["family", "gemma", "run", "hi"]).is_ok());
    assert!(parse(&["family", "qwen3.5", "run", "--think", "hi"]).is_ok());
    assert!(parse(&["family", "gemma", "run", "--think", "hi"]).is_err());
    assert!(parse(&["family", "nanochat", "run", "--think", "hi"]).is_err());
    assert!(parse(&["family", "nanochat", "repl", "--show-thinking"]).is_err());
    assert!(parse(&["repl", "--no-think"]).is_ok());
    assert!(parse(&["chat"]).is_err(), "the conversation is `repl`");
}
