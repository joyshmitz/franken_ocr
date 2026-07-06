//! Model artifact distribution — `focr pull` + first-run auto-download.
//!
//! `focr` ships without the 6.67 GB weights; it fetches them on demand in the
//! optimal on-disk format (an int8 `.focrq`, ~3.9 GB; see `quant::convert`) plus
//! the `tokenizer.json` sidecar the loader resolves next to the model. A small
//! JSON [`Manifest`] lists mirror URLs (GitHub Releases + Hugging Face) and the
//! sha256 of every byte; the downloader verifies each part AND the reassembled
//! whole, then installs into `~/.cache/franken_ocr/models/` (already a model
//! search dir), so once cached, INFERENCE is fully offline.
//!
//! HTTP is asupersync's native, capability-gated stack over rustls + Mozilla
//! webpki roots (feature `tls-webpki-roots`) — no `reqwest`/`ureq`/`hyper`.
//! Redirects (GitHub 302 → S3, HF → CDN) are followed automatically; the body
//! is streamed frame-by-frame so a 2 GB part never sits in RAM.

use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::h1::HttpError;
use asupersync::http::{Body, Client, ClientError, Method};
use asupersync::runtime::RuntimeBuilder;

use crate::error::{FocrError, FocrResult};

/// The default quant level `focr pull` (and the first-run prompt) fetches. int8
/// is the validated, byte-identical-to-load-time format; int4 is deferred until
/// it has its own CER validation (see the model-distribution plan).
pub const DEFAULT_QUANT: &str = "int8";

/// Environment override for the manifest source (a local path or an `http(s)`
/// URL). Takes precedence over [`DEFAULT_MANIFEST_URL`].
pub const MANIFEST_URL_ENV: &str = "FOCR_MANIFEST_URL";

/// The built-in manifest source — the small JSON checked into the franken_ocr
/// repo, which lists the mirror URLs + sha256s for the large artifacts. A user
/// may override it with `--manifest <path|url>` or [`MANIFEST_URL_ENV`].
pub const DEFAULT_MANIFEST_URL: &str =
    "https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/models/manifest.json";

/// The supported manifest schema version. The reader rejects anything newer so
/// an old binary fails loudly rather than misreading a future layout.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// The committed repo manifest, embedded at build time — the same file
/// [`DEFAULT_MANIFEST_URL`] serves from `main`. Lets `focr models` report
/// pull availability fully OFFLINE; `focr pull` still fetches the live
/// source so a newer committed manifest always wins at pull time.
pub const BUILTIN_MANIFEST_JSON: &str = include_str!("../models/manifest.json");

/// Parse the embedded repo manifest (see [`BUILTIN_MANIFEST_JSON`]).
///
/// # Errors
/// [`FocrError::FormatMismatch`] if the committed manifest is malformed or
/// newer than this binary understands — a build-time file, so a failure here
/// is a packaging bug, and the schema round-trip test catches it.
pub fn builtin_manifest() -> FocrResult<Manifest> {
    parse_manifest(BUILTIN_MANIFEST_JSON.as_bytes())
}

/// The download manifest: every artifact, its mirrors, and its sha256s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest layout version (see [`MANIFEST_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Model short name (e.g. `"unlimited-ocr"`).
    pub model: String,
    /// License notice that must travel with the redistributed weights.
    #[serde(default)]
    pub license_notice: String,
    /// Per-quant artifacts, keyed by quant tag (`"int8"`, …). Describes the
    /// primary [`model`](Self::model) (default `unlimited-ocr`); old binaries
    /// read only these top-level fields, so they stay in place for back-compat.
    pub quants: BTreeMap<String, QuantEntry>,
    /// The tokenizer sidecar for the primary model, installed beside the `.focrq`.
    pub tokenizer: RemoteFile,
    /// Additional models, keyed by model id (e.g. `"got-ocr2"`), selectable via
    /// `focr pull <model>`. The primary model above is NOT duplicated here. A
    /// binary predating multi-model manifests simply ignores this field.
    #[serde(default)]
    pub models: BTreeMap<String, ModelEntry>,
}

/// The artifacts for one non-primary model (its quants + tokenizer sidecar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// License notice that must travel with this model's redistributed weights.
    #[serde(default)]
    pub license_notice: String,
    /// Per-quant artifacts, keyed by quant tag (`"int8"`, …).
    pub quants: BTreeMap<String, QuantEntry>,
    /// The tokenizer sidecar (e.g. `qwen.tiktoken`), installed beside the `.focrq`.
    pub tokenizer: RemoteFile,
    /// Additional runtime-required sidecars beyond [`tokenizer`](Self::tokenizer)
    /// — e.g. OneChart's `merges.txt` + `added_tokens.json` beside its
    /// `vocab.json`, or TrOMR's remaining three tokenizer tables. Installed
    /// beside the `.focrq` like the tokenizer. Additive field: binaries
    /// predating it ignore it (serde default), so the committed manifest stays
    /// parseable by every released `focr`.
    #[serde(default)]
    pub sidecars: Vec<RemoteFile>,
}

/// The artifacts for one quant level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantEntry {
    /// The quantized weights blob.
    pub focrq: RemoteFile,
}

/// A logical file split into one or more ordered, sha256-pinned parts. The
/// concatenation of the parts IS the file (a GitHub-friendly split that HF can
/// mirror part-for-part); a single-part file is the unsplit case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteFile {
    /// Install filename in the cache dir (e.g. `unlimited-ocr.int8.focrq`).
    pub filename: String,
    /// Total size of the reassembled file, in bytes.
    pub size: u64,
    /// Lowercase-hex sha256 of the reassembled file.
    pub sha256: String,
    /// Ordered parts; concatenation = the file.
    pub parts: Vec<RemotePart>,
}

/// One sha256-pinned part with its mirror URLs (tried in order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotePart {
    /// Size of this part, in bytes.
    pub size: u64,
    /// Lowercase-hex sha256 of this part.
    pub sha256: String,
    /// Mirror URLs for this part, tried in order until one verifies.
    pub urls: Vec<String>,
}

/// Lowercase-hex encode a 32-byte digest.
fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap_or('0'));
    }
    s
}

/// True iff `expected` (lowercase-hex sha256) equals the digest of `data`.
fn sha256_hex_matches(data: &[u8], expected: &str) -> bool {
    let mut h = Sha256::new();
    h.update(data);
    let got: [u8; 32] = h.finalize().into();
    hex32(&got) == expected.trim().to_ascii_lowercase()
}

/// The per-user cache root for franken_ocr artifacts, resolved per platform:
/// `%LOCALAPPDATA%\franken_ocr` on Windows (falling back to
/// `%USERPROFILE%\.cache\franken_ocr`), and `$HOME/.cache/franken_ocr`
/// everywhere else. Returns `None` only when the platform's home/appdata
/// environment is entirely unset.
#[must_use]
pub fn cache_root() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return Some(PathBuf::from(local).join("franken_ocr"));
        }
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            return Some(PathBuf::from(profile).join(".cache").join("franken_ocr"));
        }
        None
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache").join("franken_ocr"))
    }
}

/// The cache directory new artifacts install into: `%LOCALAPPDATA%\franken_ocr\models`
/// on Windows, `~/.cache/franken_ocr/models` elsewhere (the first user-cache entry
/// of the model search path).
pub fn cache_models_dir() -> FocrResult<PathBuf> {
    cache_root().map(|root| root.join("models")).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "cannot resolve a user cache directory (set HOME, or LOCALAPPDATA/USERPROFILE on Windows)"
        ))
    })
}

/// Is `source` an `http(s)` URL (vs. a local filesystem path)?
fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

/// Resolve the manifest source string: explicit `arg`, else [`MANIFEST_URL_ENV`],
/// else [`DEFAULT_MANIFEST_URL`].
pub fn resolve_manifest_source(arg: Option<&str>) -> String {
    if let Some(a) = arg {
        return a.to_string();
    }
    if let Ok(env) = std::env::var(MANIFEST_URL_ENV)
        && !env.trim().is_empty()
    {
        return env;
    }
    DEFAULT_MANIFEST_URL.to_string()
}

/// Parse + validate a manifest from raw JSON bytes.
pub fn parse_manifest(bytes: &[u8]) -> FocrResult<Manifest> {
    let manifest: Manifest = serde_json::from_slice(bytes)
        .map_err(|e| FocrError::FormatMismatch(format!("manifest JSON parse: {e}")))?;
    if manifest.schema_version > MANIFEST_SCHEMA_VERSION {
        return Err(FocrError::FormatMismatch(format!(
            "manifest schema_version {} is newer than this binary supports ({}) — update focr",
            manifest.schema_version, MANIFEST_SCHEMA_VERSION
        )));
    }
    Ok(manifest)
}

// ── HTTP download (asupersync native stack) ─────────────────────────────────

/// Distinct download failures, surfaced with context up the `FocrError` chain.
#[derive(Debug)]
enum DownloadError {
    Request(ClientError),
    Body(HttpError),
    UnexpectedStatus(u16),
    Io(std::io::Error),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(e) => write!(f, "request failed: {e}"),
            Self::Body(e) => write!(f, "body stream failed: {e}"),
            Self::UnexpectedStatus(s) => write!(f, "unexpected HTTP status {s}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

/// Build the streaming client. The streaming body still enforces
/// `max_body_size`, and the runtime-default client leaves it at the 16 MiB codec
/// default — far below a 2 GB part — so we raise it explicitly to 8 GiB.
fn streaming_client() -> Client {
    Client::builder()
        .max_body_size(8 * 1024 * 1024 * 1024)
        .build()
}

/// Stream one URL's body into `sink`, invoking it with each chunk. Returns the
/// total bytes written. Redirects/TLS are handled inside asupersync.
async fn stream_url<S: FnMut(&[u8]) -> Result<(), std::io::Error>>(
    cx: &Cx,
    client: &Client,
    url: &str,
    mut sink: S,
) -> Result<u64, DownloadError> {
    let resp = client
        .request_streaming(cx, Method::Get, url, Vec::new(), Vec::new())
        .await
        .map_err(DownloadError::Request)?;
    if !(200..=299).contains(&resp.head.status) {
        return Err(DownloadError::UnexpectedStatus(resp.head.status));
    }
    let mut total: u64 = 0;
    let mut body = resp.body;
    while let Some(frame) =
        std::future::poll_fn(|task_cx| Pin::new(&mut body).poll_frame(task_cx)).await
    {
        let frame = frame.map_err(DownloadError::Body)?;
        if let Some(mut chunk) = frame.into_data() {
            while chunk.has_remaining() {
                let bytes = chunk.chunk();
                sink(bytes).map_err(DownloadError::Io)?;
                let n = bytes.len();
                total += n as u64;
                chunk.advance(n);
            }
        }
    }
    Ok(total)
}

/// Download a small resource (a manifest) fully into memory.
async fn fetch_bytes(cx: &Cx, url: &str) -> Result<Vec<u8>, DownloadError> {
    let client = streaming_client();
    let mut buf = Vec::new();
    stream_url(cx, &client, url, |chunk| {
        buf.extend_from_slice(chunk);
        Ok(())
    })
    .await?;
    Ok(buf)
}

/// Download + verify one [`RemoteFile`] into `dest` (a `.tmp` path), then return.
/// Each part is streamed straight into `dest` while a per-part AND a whole-file
/// hasher run; a failed mirror rolls `dest` + the whole-file hash back to the
/// part boundary and tries the next URL. The whole-file sha256 is checked last.
/// Per-part progress goes to stderr (stdout stays clean for any JSON consumer);
/// `quiet` silences it.
async fn download_remote_file(
    cx: &Cx,
    file: &RemoteFile,
    dest: &Path,
    quiet: bool,
) -> FocrResult<()> {
    let client = streaming_client();
    let mut out = std::fs::File::create(dest)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("create {}: {e}", dest.display())))?;
    let mut full = Sha256::new();
    let mut committed: u64 = 0;

    for (i, part) in file.parts.iter().enumerate() {
        let full_checkpoint = full.clone();
        let mut part_ok = false;
        let mut last_err: Option<String> = None;

        for url in &part.urls {
            // Roll back to the part boundary before each attempt.
            out.set_len(committed)
                .and_then(|()| out.seek(SeekFrom::Start(committed)).map(|_| ()))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("seek {}: {e}", dest.display())))?;
            full = full_checkpoint.clone();
            let mut part_hash = Sha256::new();

            if !quiet {
                eprintln!(
                    "  part {}/{} ({:.1} MB) <- {url}",
                    i + 1,
                    file.parts.len(),
                    part.size as f64 / 1.0e6
                );
            }
            let res = stream_url(cx, &client, url, |chunk| {
                out.write_all(chunk)?;
                part_hash.update(chunk);
                full.update(chunk);
                Ok(())
            })
            .await;

            match res {
                Ok(n) if n == part.size => {
                    let got: [u8; 32] = part_hash.finalize().into();
                    if hex32(&got) == part.sha256.trim().to_ascii_lowercase() {
                        part_ok = true;
                        committed += part.size;
                        break;
                    }
                    last_err = Some(format!("part {} sha256 mismatch from {url}", i + 1));
                }
                Ok(n) => {
                    last_err = Some(format!("part {} size {n} != expected {}", i + 1, part.size))
                }
                Err(e) => last_err = Some(format!("{e} (from {url})")),
            }
        }

        if !part_ok {
            return Err(FocrError::Other(anyhow::anyhow!(
                "all mirrors failed for part {} of {}: {}",
                i + 1,
                file.filename,
                last_err.unwrap_or_else(|| "no urls".into())
            )));
        }
    }

    out.flush()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("flush {}: {e}", dest.display())))?;
    let got: [u8; 32] = full.finalize().into();
    if hex32(&got) != file.sha256.trim().to_ascii_lowercase() {
        return Err(FocrError::FormatMismatch(format!(
            "reassembled {} sha256 {} != manifest {}",
            file.filename,
            hex32(&got),
            file.sha256
        )));
    }
    if committed != file.size {
        return Err(FocrError::FormatMismatch(format!(
            "reassembled {} size {committed} != manifest {}",
            file.filename, file.size
        )));
    }
    Ok(())
}

/// Is `path` already a byte-perfect copy of `file` (so the download is skippable)?
fn already_cached(path: &Path, file: &RemoteFile) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.len() != file.size {
        return false;
    }
    // Size matched; confirm by hashing (cheap relative to a multi-GB download).
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    sha256_hex_matches(&bytes, &file.sha256)
}

/// The outcome of a [`pull`]: where the model + tokenizer landed.
#[derive(Debug, Clone)]
pub struct PullOutcome {
    /// The installed `.focrq` path (a model search dir resolves it by name).
    pub focrq_path: PathBuf,
    /// The installed `tokenizer.json` path (sibling of the `.focrq`).
    pub tokenizer_path: PathBuf,
    /// Additional installed sidecar paths (siblings of the `.focrq`).
    pub sidecar_paths: Vec<PathBuf>,
    /// Quant level pulled (the manifest's actual tag — see the sole-quant
    /// fallback in [`pull`]).
    pub quant: String,
    /// The pulled model's license notice from the manifest (empty when the
    /// manifest entry carries none; the caller may substitute the primary
    /// model's built-in notice).
    pub license_notice: String,
    /// True iff every artifact was already cached (nothing downloaded).
    pub from_cache: bool,
}

/// Pick the quant entry for a pull. Exact match wins; otherwise, when the
/// model publishes EXACTLY ONE quant, fall back to it — the CLI default is
/// `int8`, TrOMR ships f32-only (int8 gated behind an unrun lossless
/// proof), and failing `focr pull tromr` over a default flag serves no one.
/// The returned tag is the ACTUAL quant (callers report it; `pull` prints a
/// visible note when it differs from the request).
///
/// # Errors
/// [`FocrError::Usage`] when the quant is absent and the fallback is
/// ambiguous (zero or several published quants).
fn select_quant<'m>(
    quants: &'m BTreeMap<String, QuantEntry>,
    requested: &str,
) -> FocrResult<(String, &'m QuantEntry)> {
    match quants.get(requested) {
        Some(entry) => Ok((requested.to_owned(), entry)),
        None if quants.len() == 1 => {
            let (tag, entry) = quants.iter().next().expect("len==1 map has an entry");
            Ok((tag.clone(), entry))
        }
        None => Err(FocrError::Usage(format!(
            "manifest has no quant '{requested}' (available: {})",
            quants.keys().cloned().collect::<Vec<_>>().join(", ")
        ))),
    }
}

/// Fetch (or confirm-cached) the `quant` weights + tokenizer described by the
/// manifest at `manifest_source` (path or URL) into the model cache. `progress`
/// receives human status lines. Network only happens here — once it returns Ok,
/// the model loads offline.
pub fn pull(
    model: Option<&str>,
    quant: &str,
    manifest_source: &str,
    quiet: bool,
    mut progress: impl FnMut(&str),
) -> FocrResult<PullOutcome> {
    let runtime = RuntimeBuilder::new()
        .build()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("runtime build: {e}")))?;

    // 1. Load the manifest (local file or remote URL).
    let manifest_bytes = if is_url(manifest_source) {
        progress(&format!("fetching manifest {manifest_source}"));
        let url = manifest_source.to_string();
        runtime.block_on(async move {
            let cx = Cx::current().ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!("runtime did not install an ambient Cx"))
            })?;
            fetch_bytes(&cx, &url)
                .await
                .map_err(|e| FocrError::Other(anyhow::anyhow!("manifest fetch: {e}")))
        })?
    } else {
        std::fs::read(manifest_source)
            .map_err(|e| FocrError::ModelNotFound(format!("manifest {manifest_source}: {e}")))?
    };
    let manifest = parse_manifest(&manifest_bytes)?;

    // Select the model: the primary top-level model (default, and the only one
    // old binaries know) unless `model` names a distinct entry in `models`.
    // Non-primary models install into their own `<cache>/<model-id>/` subdir:
    // sidecar filenames are NOT unique across models (smolvlm2 ships a
    // `tokenizer.json` that would clobber unlimited-ocr's in a flat cache),
    // and the loaders resolve sidecars beside the artifact, so isolation per
    // model is both necessary and sufficient. The primary model stays flat —
    // the layout every released binary already knows.
    static NO_SIDECARS: Vec<RemoteFile> = Vec::new();
    let (quants, tokenizer, sidecars, subdir, license_notice): (
        &BTreeMap<String, QuantEntry>,
        &RemoteFile,
        &[RemoteFile],
        Option<&str>,
        &str,
    ) = match model {
        None => (
            &manifest.quants,
            &manifest.tokenizer,
            &NO_SIDECARS,
            None,
            manifest.license_notice.as_str(),
        ),
        Some(m) if m == manifest.model => (
            &manifest.quants,
            &manifest.tokenizer,
            &NO_SIDECARS,
            None,
            manifest.license_notice.as_str(),
        ),
        Some(m) => {
            let entry = manifest.models.get(m).ok_or_else(|| {
                let mut avail = vec![manifest.model.clone()];
                avail.extend(manifest.models.keys().cloned());
                FocrError::Usage(format!(
                    "manifest has no model '{m}' (available: {})",
                    avail.join(", ")
                ))
            })?;
            (
                &entry.quants,
                &entry.tokenizer,
                &entry.sidecars,
                Some(m),
                entry.license_notice.as_str(),
            )
        }
    };

    let (quant_used, quant_entry) = select_quant(quants, quant)?;
    if quant_used != quant {
        progress(&format!(
            "manifest has no quant '{quant}' for this model; using the sole \
             published quant '{quant_used}'"
        ));
    }

    let mut dir = cache_models_dir()?;
    if let Some(sub) = subdir {
        dir = dir.join(sub);
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("create cache {}: {e}", dir.display())))?;
    let focrq_path = dir.join(&quant_entry.focrq.filename);
    let tokenizer_path = dir.join(&tokenizer.filename);

    // 2. Download each artifact unless already byte-perfect in the cache.
    let focrq_cached = already_cached(&focrq_path, &quant_entry.focrq);
    let tok_cached = already_cached(&tokenizer_path, tokenizer);
    let mut from_cache = focrq_cached && tok_cached;

    if !focrq_cached {
        install_file(
            &runtime,
            &quant_entry.focrq,
            &focrq_path,
            quiet,
            &mut progress,
        )?;
    } else {
        progress(&format!("cached: {}", focrq_path.display()));
    }
    if !tok_cached {
        install_file(&runtime, tokenizer, &tokenizer_path, quiet, &mut progress)?;
    } else {
        progress(&format!("cached: {}", tokenizer_path.display()));
    }
    let mut sidecar_paths = Vec::with_capacity(sidecars.len());
    for sidecar in sidecars {
        let path = dir.join(&sidecar.filename);
        if already_cached(&path, sidecar) {
            progress(&format!("cached: {}", path.display()));
        } else {
            install_file(&runtime, sidecar, &path, quiet, &mut progress)?;
            from_cache = false;
        }
        sidecar_paths.push(path);
    }

    Ok(PullOutcome {
        focrq_path,
        tokenizer_path,
        sidecar_paths,
        quant: quant_used,
        license_notice: license_notice.to_owned(),
        from_cache,
    })
}

/// Download `file` to a `.partial` sibling, verify, then atomically rename into
/// place. `quiet` suppresses the per-part stderr progress.
fn install_file(
    runtime: &asupersync::runtime::Runtime,
    file: &RemoteFile,
    final_path: &Path,
    quiet: bool,
    progress: &mut impl FnMut(&str),
) -> FocrResult<()> {
    let tmp = final_path.with_extension("partial");
    let tmp_for_async = tmp.clone();
    let file_owned = file.clone();
    progress(&format!(
        "downloading {} ({:.2} GB, {} part(s))",
        file.filename,
        file.size as f64 / 1.0e9,
        file.parts.len()
    ));
    let download = runtime.block_on(async move {
        let cx = Cx::current().ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!("runtime did not install an ambient Cx"))
        })?;
        download_remote_file(&cx, &file_owned, &tmp_for_async, quiet).await
    });
    if let Err(e) = download {
        // Don't leave a half-written `.partial` littering the cache; the next
        // attempt would truncate it anyway, but a failed pull should be tidy.
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, final_path).map_err(|e| {
        FocrError::Other(anyhow::anyhow!(
            "install {} -> {}: {e}",
            tmp.display(),
            final_path.display()
        ))
    })?;
    progress(&format!("installed {}", final_path.display()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex32_is_lowercase_64() {
        let mut h = Sha256::new();
        h.update(b"franken_ocr");
        let d: [u8; 32] = h.finalize().into();
        let hex = hex32(&d);
        assert_eq!(hex.len(), 64);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn sha256_matches_known_vector() {
        // sha256("") = e3b0c442...
        assert!(sha256_hex_matches(
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ));
        assert!(sha256_hex_matches(
            b"abc",
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD" // uppercase ok
        ));
        assert!(!sha256_hex_matches(b"abc", "00".repeat(32).as_str()));
    }

    #[test]
    fn manifest_round_trips_and_rejects_future_schema() {
        let m = Manifest {
            schema_version: 1,
            model: "unlimited-ocr".into(),
            license_notice: "MIT (Baidu)".into(),
            quants: BTreeMap::from([(
                "int8".to_string(),
                QuantEntry {
                    focrq: RemoteFile {
                        filename: "unlimited-ocr.int8.focrq".into(),
                        size: 3914093440,
                        sha256: "ab".repeat(32),
                        parts: vec![RemotePart {
                            size: 3914093440,
                            sha256: "ab".repeat(32),
                            urls: vec!["https://example/part0".into()],
                        }],
                    },
                },
            )]),
            tokenizer: RemoteFile {
                filename: "tokenizer.json".into(),
                size: 9979544,
                sha256: "cd".repeat(32),
                parts: vec![RemotePart {
                    size: 9979544,
                    sha256: "cd".repeat(32),
                    urls: vec!["https://example/tok".into()],
                }],
            },
            models: BTreeMap::from([(
                "got-ocr2".to_string(),
                ModelEntry {
                    license_notice: "Apache-2.0 (GOT-OCR2.0)".into(),
                    quants: BTreeMap::from([(
                        "int8".to_string(),
                        QuantEntry {
                            focrq: RemoteFile {
                                filename: "got-ocr2.int8.focrq".into(),
                                size: 813877416,
                                sha256: "ef".repeat(32),
                                parts: vec![RemotePart {
                                    size: 813877416,
                                    sha256: "ef".repeat(32),
                                    urls: vec!["https://example/got".into()],
                                }],
                            },
                        },
                    )]),
                    tokenizer: RemoteFile {
                        filename: "qwen.tiktoken".into(),
                        size: 2561218,
                        sha256: "12".repeat(32),
                        parts: vec![RemotePart {
                            size: 2561218,
                            sha256: "12".repeat(32),
                            urls: vec!["https://example/qwen".into()],
                        }],
                    },
                    sidecars: Vec::new(),
                },
            )]),
        };
        let json = serde_json::to_vec(&m).expect("serialize");
        let back = parse_manifest(&json).expect("parse");
        assert_eq!(back.model, "unlimited-ocr");
        assert_eq!(back.quants["int8"].focrq.size, 3914093440);
        // The secondary model resolves by id with its own tokenizer filename.
        assert_eq!(back.models["got-ocr2"].tokenizer.filename, "qwen.tiktoken");
        assert_eq!(back.models["got-ocr2"].quants["int8"].focrq.size, 813877416);

        // A newer schema is rejected loudly.
        let future = br#"{"schema_version":999,"model":"x","quants":{},"tokenizer":{"filename":"t","size":0,"sha256":"","parts":[]}}"#;
        assert!(matches!(
            parse_manifest(future),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    #[test]
    fn manifest_source_resolution_precedence() {
        // Explicit arg wins.
        assert_eq!(resolve_manifest_source(Some("/tmp/m.json")), "/tmp/m.json");
        // Default when nothing set (env not set in this unit context).
        // (We avoid mutating process env in tests; just assert the default const
        // is what falls through.)
        assert!(DEFAULT_MANIFEST_URL.starts_with("https://"));
    }

    #[test]
    fn is_url_discriminates() {
        assert!(is_url("https://x/y"));
        assert!(is_url("http://x/y"));
        assert!(!is_url("/local/path"));
        assert!(!is_url("models/manifest.json"));
    }

    #[test]
    fn already_cached_matches_only_on_size_and_hash() {
        let dir = std::env::temp_dir().join(format!("focr_dist_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("blob.bin");
        std::fs::write(&p, b"hello franken").unwrap();
        let mut h = Sha256::new();
        h.update(b"hello franken");
        let d: [u8; 32] = h.finalize().into();
        let good = RemoteFile {
            filename: "blob.bin".into(),
            size: 13,
            sha256: hex32(&d),
            parts: vec![],
        };
        assert!(already_cached(&p, &good));
        let wrong_size = RemoteFile {
            size: 99,
            ..good.clone()
        };
        assert!(!already_cached(&p, &wrong_size));
        let wrong_hash = RemoteFile {
            sha256: "00".repeat(32),
            ..good.clone()
        };
        assert!(!already_cached(&p, &wrong_hash));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// bd-av64.7: the COMMITTED manifest is embedded and must lint clean —
    /// every sha 64-hex lowercase, every size positive, every URL https,
    /// part sizes summing to the whole, filenames unique per install dir —
    /// and it must publish the full runtime-ready zoo.
    #[test]
    fn builtin_manifest_publishes_the_zoo_and_lints_clean() {
        let m = builtin_manifest().expect("embedded manifest parses");
        assert_eq!(m.model, "unlimited-ocr");
        for id in ["got-ocr2", "smolvlm2", "onechart", "tromr"] {
            assert!(m.models.contains_key(id), "manifest missing model {id}");
        }
        // Runtime-required sidecar sets (the anti-broken-pull contract; the
        // loaders resolve these beside the artifact — see native_engine).
        let tromr = &m.models["tromr"];
        assert_eq!(
            tromr.quants.keys().cloned().collect::<Vec<_>>(),
            vec!["f32".to_string()],
            "tromr publishes f32 only (int8 gated behind a lossless proof)"
        );
        assert_eq!(tromr.tokenizer.filename, "tokenizer_rhythm.json");
        let mut tromr_sidecars: Vec<&str> =
            tromr.sidecars.iter().map(|f| f.filename.as_str()).collect();
        tromr_sidecars.sort_unstable();
        assert_eq!(
            tromr_sidecars,
            vec![
                "tokenizer_lift.json",
                "tokenizer_note.json",
                "tokenizer_pitch.json"
            ]
        );
        let onechart = &m.models["onechart"];
        assert_eq!(onechart.tokenizer.filename, "vocab.json");
        let mut oc_sidecars: Vec<&str> = onechart
            .sidecars
            .iter()
            .map(|f| f.filename.as_str())
            .collect();
        oc_sidecars.sort_unstable();
        assert_eq!(oc_sidecars, vec!["added_tokens.json", "merges.txt"]);
        assert_eq!(m.models["smolvlm2"].tokenizer.filename, "tokenizer.json");
        assert!(m.models["smolvlm2"].sidecars.is_empty());

        // The lint, over every file of every model.
        let lint = |file: &RemoteFile, ctx: &str| {
            assert!(file.size > 0, "{ctx}: zero size");
            assert_eq!(file.sha256.len(), 64, "{ctx}: sha length");
            assert!(
                file.sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{ctx}: sha not lowercase hex"
            );
            assert!(!file.parts.is_empty(), "{ctx}: no parts");
            assert_eq!(
                file.parts.iter().map(|p| p.size).sum::<u64>(),
                file.size,
                "{ctx}: part sizes do not sum to the whole"
            );
            for part in &file.parts {
                assert!(!part.urls.is_empty(), "{ctx}: part without urls");
                for url in &part.urls {
                    assert!(url.starts_with("https://"), "{ctx}: non-https url {url}");
                }
            }
        };
        let mut primary: Vec<&str> = vec![m.tokenizer.filename.as_str()];
        lint(&m.tokenizer, "primary tokenizer");
        for (tag, q) in &m.quants {
            lint(&q.focrq, &format!("primary quant {tag}"));
            primary.push(q.focrq.filename.as_str());
        }
        primary.sort_unstable();
        primary.dedup();
        assert_eq!(
            primary.len(),
            1 + m.quants.len(),
            "primary filename collision"
        );
        for (id, entry) in &m.models {
            let mut names: Vec<&str> = vec![entry.tokenizer.filename.as_str()];
            lint(&entry.tokenizer, &format!("{id} tokenizer"));
            assert!(
                !entry.license_notice.is_empty(),
                "{id}: empty license notice"
            );
            for (tag, q) in &entry.quants {
                lint(&q.focrq, &format!("{id} quant {tag}"));
                names.push(q.focrq.filename.as_str());
            }
            for sc in &entry.sidecars {
                lint(sc, &format!("{id} sidecar {}", sc.filename));
                names.push(sc.filename.as_str());
            }
            let total = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(
                names.len(),
                total,
                "{id}: filename collision in its install dir"
            );
        }
    }

    /// bd-av64.7: exact quant wins; a sole published quant is the fallback
    /// (TrOMR is f32-only while the CLI default is int8); several published
    /// quants + a miss stays a loud Usage error.
    #[test]
    fn select_quant_exact_sole_and_ambiguous() {
        let file = RemoteFile {
            filename: "x.focrq".into(),
            size: 1,
            sha256: "ab".repeat(32),
            parts: vec![],
        };
        let entry = QuantEntry { focrq: file };
        let sole = BTreeMap::from([("f32".to_string(), entry.clone())]);
        let (tag, _) = select_quant(&sole, "int8").expect("sole quant falls back");
        assert_eq!(tag, "f32");
        let (tag, _) = select_quant(&sole, "f32").expect("exact match");
        assert_eq!(tag, "f32");
        let two = BTreeMap::from([
            ("int8".to_string(), entry.clone()),
            ("int4".to_string(), entry.clone()),
        ]);
        let (tag, _) = select_quant(&two, "int8").expect("exact among several");
        assert_eq!(tag, "int8");
        assert!(
            select_quant(&two, "f32").is_err(),
            "ambiguous miss must error"
        );
        assert!(select_quant(&BTreeMap::new(), "int8").is_err());
    }

    /// Back-compat: a ModelEntry WITHOUT the (new) sidecars field parses with
    /// an empty list — the shape every pre-sidecar manifest entry has.
    #[test]
    fn model_entry_without_sidecars_parses_empty() {
        let json = r#"{
            "license_notice": "x",
            "quants": {"int8": {"focrq": {"filename": "a.focrq", "size": 1,
                "sha256": "00", "parts": []}}},
            "tokenizer": {"filename": "t.json", "size": 1, "sha256": "00", "parts": []}
        }"#;
        let entry: ModelEntry = serde_json::from_str(json).expect("parses");
        assert!(entry.sidecars.is_empty());
    }
}
