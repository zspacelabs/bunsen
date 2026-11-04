//! A checkpoint's chat template.
use std::{
    collections::BTreeMap,
    path::Path,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};
use minijinja::Environment;
use serde::Serialize;

/// One turn of a conversation, as a chat template reads it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,

    /// What was said; for an assistant, its answer.
    pub content: String,

    /// An assistant's thinking, when it is kept apart from the answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// A turn of `role`.
    pub fn new(
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    /// A system prompt.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// An assistant turn.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// The same turn, with its thinking kept apart.
    pub fn with_reasoning(
        mut self,
        reasoning: impl Into<String>,
    ) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }
}

/// The template's name in its environment.
const NAME: &str = "chat";

/// A checkpoint's chat template: the Jinja its `tokenizer_config.json`
/// carries, which renders a conversation as the text the model was tuned
/// on.
///
/// Rendered by `minijinja` as `transformers`' `apply_chat_template` renders
/// it: blocks trimmed and left-stripped, Python's string methods
/// (`startswith`, `split`, `strip`, ...) available, an undefined variable
/// falsy, `raise_exception` failing the render. A template sees the
/// tokenizer's special tokens (`bos_token`, `eos_token`, `pad_token`,
/// `unk_token`) as `transformers` passes them: Gemma's opens with
/// `{{ bos_token }}`. A Qwen3 template reads `enable_thinking` (absent unless
/// given) and `add_generation_prompt`; it renders no tools here.
#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
    special_tokens: BTreeMap<String, String>,
}

/// A `tokenizer_config.json`'s special tokens, by their template names: each
/// a string, or an added token's `content`.
fn special_tokens(config: &serde_json::Value) -> BTreeMap<String, String> {
    ["bos_token", "eos_token", "pad_token", "unk_token"]
        .into_iter()
        .filter_map(|key| {
            let value = config.get(key)?;
            let content = value
                .as_str()
                .or_else(|| value.get("content").and_then(serde_json::Value::as_str))?;
            Some((key.to_string(), content.to_string()))
        })
        .collect()
}

impl ChatTemplate {
    /// The template of a `tokenizer_config.json` file.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when it cannot be
    /// read; as [`from_tokenizer_config`](Self::from_tokenizer_config).
    pub fn from_file(tokenizer_config_json: &Path) -> BunsenResult<Self> {
        let json = std::fs::read_to_string(tokenizer_config_json)
            .map_err(sys_at("read", tokenizer_config_json))?;
        Self::from_tokenizer_config(&json).map_err(|e| e.context(tokenizer_config_json.display()))
    }

    /// The template in a file of its own (`chat_template.jinja`), its special
    /// tokens those of a `tokenizer_config.json`.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when a file cannot
    /// be read; [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the config does not parse or the template does not compile.
    pub fn from_jinja_file(
        template: &Path,
        tokenizer_config_json: &Path,
    ) -> BunsenResult<Self> {
        let source = std::fs::read_to_string(template).map_err(sys_at("read", template))?;
        let json = std::fs::read_to_string(tokenizer_config_json)
            .map_err(sys_at("read", tokenizer_config_json))?;
        let value = Self::parse_config(&json)?;
        Ok(Self::new(&source)
            .map_err(|e| e.context(template.display()))?
            .with_special_tokens(special_tokens(&value)))
    }

    fn parse_config(json: &str) -> BunsenResult<serde_json::Value> {
        serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e))
    }

    /// The template of a `tokenizer_config.json`, with its special tokens.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse, has no `chat_template`, or the template does
    /// not compile.
    pub fn from_tokenizer_config(json: &str) -> BunsenResult<Self> {
        let value = Self::parse_config(json)?;
        let source = value
            .get("chat_template")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                BunsenError::invalid_resource("a tokenizer_config.json without a chat_template")
            })?;
        Ok(Self::new(source)?.with_special_tokens(special_tokens(&value)))
    }

    /// This template, seeing `special_tokens` (`bos_token` and its kin) by
    /// name.
    pub fn with_special_tokens(
        self,
        special_tokens: BTreeMap<String, String>,
    ) -> Self {
        Self {
            special_tokens,
            ..self
        }
    }

    /// A template from its Jinja source.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not compile.
    pub fn new(source: &str) -> BunsenResult<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template_owned(NAME, source.to_string())
            .map_err(|e| BunsenError::invalid_resource("a chat template").with_cause(e))?;
        Ok(Self {
            env,
            special_tokens: BTreeMap::new(),
        })
    }

    /// Renders a conversation.
    ///
    /// # Arguments
    /// * `messages` - the conversation so far.
    /// * `add_generation_prompt` - whether to open the assistant's next turn.
    /// * `enable_thinking` - Qwen3's switch; `None` leaves it undefined, as a
    ///   caller who does not pass it does.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when the template fails on the conversation.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> BunsenResult<String> {
        let template = self.env.get_template(NAME).expect("added at construction");
        let base = minijinja::context! {
            messages,
            add_generation_prompt,
            ..minijinja::Value::from_serialize(&self.special_tokens)
        };
        let context = match enable_thinking {
            Some(enable_thinking) => minijinja::context! { enable_thinking, ..base },
            None => base,
        };
        template
            .render(context)
            .map_err(|e| BunsenError::invalid_resource("a chat template's render").with_cause(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3's thinking switch and its multi-turn rule, in miniature: the
    /// shape the real template has.
    const TOY: &str = r#"{%- for message in messages %}
{%- if message.role == "assistant" and '</think>' in message.content %}
{{- '<|im_start|>assistant\n' + message.content.split('</think>')[-1].lstrip('\n') + '<|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}"#;

    #[test]
    fn test_render() {
        let template = ChatTemplate::new(TOY).unwrap();
        let messages = [
            ChatMessage::user("hi"),
            ChatMessage::assistant("<think>\nhm\n</think>\n\nhello"),
            ChatMessage::user("again"),
        ];
        let text = template.render(&messages, true, None).unwrap();
        assert_eq!(
            text,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n\
             <|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n"
        );
        let off = template.render(&messages[..1], true, Some(false)).unwrap();
        assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let on = template.render(&messages[..1], true, Some(true)).unwrap();
        assert!(on.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_from_tokenizer_config() {
        let json = serde_json::json!({ "chat_template": TOY }).to_string();
        ChatTemplate::from_tokenizer_config(&json).unwrap();
        assert!(ChatTemplate::from_tokenizer_config("{}").is_err());
    }

    /// Gemma 2's template, as its `tokenizer_config.json` has it: the special
    /// tokens reach it, a model's turn is `model`, and a system turn is
    /// refused through `raise_exception`.
    const GEMMA2: &str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}";

    #[test]
    fn test_special_tokens_and_raise_exception() {
        let json = serde_json::json!({
            "chat_template": GEMMA2,
            "bos_token": "<bos>",
            "eos_token": {"content": "<eos>", "special": true},
        })
        .to_string();
        let template = ChatTemplate::from_tokenizer_config(&json).unwrap();
        let text = template
            .render(
                &[ChatMessage::user(" hi "), ChatMessage::assistant("hello")],
                true,
                None,
            )
            .unwrap();
        assert_eq!(
            text,
            "<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nhello<end_of_turn>\n\
             <start_of_turn>model\n"
        );
        let err = template
            .render(&[ChatMessage::system("be brief")], true, None)
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("System role not supported"),
            "{err:?}"
        );
    }
}
