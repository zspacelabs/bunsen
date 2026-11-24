//! The Hugging Face Hub's credentials: a user's token, sent to the Hub alone.
//!
//! A gated repo (a dataset whose terms must be accepted, such as `ImageNet`'s
//! mirrors) answers 401 or 403 to a request without a token. A request to
//! the Hub ([`is_hub_url`]) carries the user's token when there is one
//! ([`hub_token`]), found as `huggingface_hub` finds it. The token goes to
//! the Hub's own hosts only: the HTTP client drops the header on every
//! redirect (`ureq`'s `RedirectAuthHeaders::Never`), so the CDN a download
//! is redirected to never sees it. It is never logged or shown.

use std::path::{
    Path,
    PathBuf,
};

/// The Hub's hosts: a request to one of them, or to a subdomain of one,
/// carries the token.
const HUB_HOSTS: [&str; 2] = ["huggingface.co", "hf.co"];

/// Whether `url` is an `https` URL of the Hugging Face Hub
/// (`huggingface.co`, `hf.co`, or a subdomain of either).
///
/// ```
/// use bunsen_fetch::is_hub_url;
///
/// assert!(is_hub_url("https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/x.tar"));
/// assert!(!is_hub_url("http://huggingface.co/x"));
/// assert!(!is_hub_url("https://huggingface.co.example.com/x"));
/// assert!(!is_hub_url("https://download.pytorch.org/models/x.pth"));
/// ```
pub fn is_hub_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit('@').next().unwrap_or("");
    let host = host.split(':').next().unwrap_or("").to_ascii_lowercase();
    HUB_HOSTS
        .iter()
        .any(|hub| host == *hub || host.ends_with(&format!(".{hub}")))
}

/// The user's Hub token, found as `huggingface_hub` finds it, or `None`:
///
/// 1. none at all when `HF_HUB_DISABLE_IMPLICIT_TOKEN` is set;
/// 2. `HF_TOKEN`, else the older `HUGGING_FACE_HUB_TOKEN`;
/// 3. the file `HF_TOKEN_PATH` names, else `$HF_HOME/token`, else
///    `$XDG_CACHE_HOME/huggingface/token`, else `~/.cache/huggingface/token`
///    (what `hf auth login` writes).
///
/// A value is trimmed; an empty one is none.
pub fn hub_token() -> Option<String> {
    hub_token_from(
        |name| std::env::var(name).ok(),
        |path| std::fs::read_to_string(path).ok(),
    )
}

/// [`hub_token`], over an environment and a file reader.
fn hub_token_from(
    env: impl Fn(&str) -> Option<String>,
    read: impl Fn(&Path) -> Option<String>,
) -> Option<String> {
    let nonempty = |s: String| {
        let s = s.trim().to_string();
        (!s.is_empty()).then_some(s)
    };
    if env("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_some_and(|v| !v.is_empty()) {
        return None;
    }
    if let Some(token) = env("HF_TOKEN")
        .or_else(|| env("HUGGING_FACE_HUB_TOKEN"))
        .and_then(nonempty)
    {
        return Some(token);
    }
    let path = env("HF_TOKEN_PATH").map(PathBuf::from).or_else(|| {
        env("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| env("XDG_CACHE_HOME").map(|c| PathBuf::from(c).join("huggingface")))
            .or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface")))
            .map(|home| home.join("token"))
    })?;
    read(&path).and_then(nonempty)
}

/// The `Authorization` header value a request to `url` carries: `Bearer` and
/// the user's token, for a Hub URL when there is a token; else `None`.
pub fn hub_authorization(url: &str) -> Option<String> {
    is_hub_url(url)
        .then(hub_token)
        .flatten()
        .map(|token| format!("Bearer {token}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// [`hub_token_from`] over a fixed environment and files.
    fn token(
        env: &[(&str, &str)],
        files: &[(&str, &str)],
    ) -> Option<String> {
        let env: HashMap<_, _> = env.iter().copied().collect();
        let files: HashMap<_, _> = files.iter().copied().collect();
        hub_token_from(
            |k| env.get(k).map(|v| v.to_string()),
            |p| files.get(p.to_str().unwrap()).map(|v| v.to_string()),
        )
    }

    #[test]
    fn test_hub_hosts() {
        assert!(is_hub_url("https://hf.co/x"));
        assert!(is_hub_url("https://cdn-lfs.huggingface.co/x"));
        assert!(is_hub_url("https://user@huggingface.co:443/x"));
        assert!(is_hub_url("https://HuggingFace.co/x"));
        assert!(!is_hub_url("https://nothuggingface.co/x"));
        assert!(!is_hub_url(
            "https://raw.githubusercontent.com/huggingface.co/x"
        ));
    }

    /// The environment wins, then the token file, in `huggingface_hub`'s
    /// order; empty values are none; the opt-out turns all of it off.
    #[test]
    fn test_token_lookup_order() {
        let home = [("HOME", "/h")];
        let file = [("/h/.cache/huggingface/token", "file-token\n")];
        assert_eq!(token(&home, &file).as_deref(), Some("file-token"));
        assert_eq!(
            token(&[("HOME", "/h"), ("HF_TOKEN", " env-token ")], &file).as_deref(),
            Some("env-token")
        );
        assert_eq!(
            token(&[("HOME", "/h"), ("HUGGING_FACE_HUB_TOKEN", "old")], &file).as_deref(),
            Some("old")
        );
        assert_eq!(
            token(&[("HOME", "/h"), ("HF_TOKEN", "")], &file).as_deref(),
            Some("file-token"),
            "an empty variable is none"
        );
        assert_eq!(
            token(&[("HF_HOME", "/hf")], &[("/hf/token", "hf-home")]).as_deref(),
            Some("hf-home")
        );
        assert_eq!(
            token(
                &[("XDG_CACHE_HOME", "/x")],
                &[("/x/huggingface/token", "xdg")]
            )
            .as_deref(),
            Some("xdg")
        );
        assert_eq!(
            token(
                &[("HF_TOKEN_PATH", "/t"), ("HF_HOME", "/hf")],
                &[("/t", "path"), ("/hf/token", "no")]
            )
            .as_deref(),
            Some("path")
        );
        assert_eq!(
            token(
                &[("HOME", "/h"), ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")],
                &file
            ),
            None
        );
        assert_eq!(token(&home, &[]), None);
    }
}
