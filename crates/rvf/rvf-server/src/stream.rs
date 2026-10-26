//! Verified streaming delivery (ADR-353): `GET /v1/stream/descriptor`,
//! `GET /v1/stream/body` and the TCP message pair `0x05` / `0x85`.
//!
//! The server holds an existing `.rvf` body in memory, a descriptor generated
//! with `StreamDescriptorBuilder` and signed with a **server-held key read from
//! a configured path** (64 hex characters or 32 raw bytes). The server never
//! generates or stores keys. The body is served as the stage-ordered frame
//! stream with `Range: bytes=start-[end]` for resume.
//!
//! Abuse controls: a semaphore bounds concurrent streams (429 when full);
//! only a single `bytes=a-[b]` range is accepted (multi-range, suffix ranges,
//! malformed or unsatisfiable ranges get 416); one response is capped at
//! `max_range_bytes`; TCP replies are capped at `tcp_max_response`.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use ed25519_dalek::SigningKey;
use rvf_runtime::progressive_stream::build::{descriptor_for, BuildParams};
use rvf_runtime::progressive_stream::plan::StreamPlan;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Stream server configuration.
#[derive(Clone, Debug)]
pub struct StreamServerConfig {
    /// The `.rvf` body to serve.
    pub body_path: PathBuf,
    /// Path of the descriptor signing key (hex or raw 32 bytes).
    pub key_path: PathBuf,
    /// Descriptor `chunk_log2`: frame size is `1 << chunk_log2` (`0xFF` = one
    /// frame per segment).
    pub chunk_log2: u8,
    /// Descriptor epoch.
    pub epoch: u64,
    /// Concurrent body streams allowed.
    pub max_concurrent_streams: usize,
    /// Largest response for one range request, bytes.
    pub max_range_bytes: u64,
    /// Largest TCP stream reply payload, bytes.
    pub tcp_max_response: u32,
}

impl StreamServerConfig {
    /// Defaults: 64 KiB frames, epoch 1, 8 streams, 256 MiB per range, 1 MiB
    /// per TCP reply.
    pub fn new(body_path: PathBuf, key_path: PathBuf) -> Self {
        Self {
            body_path,
            key_path,
            chunk_log2: 16,
            epoch: 1,
            max_concurrent_streams: 8,
            max_range_bytes: 256 << 20,
            tcp_max_response: 1 << 20,
        }
    }
}

/// Loaded stream server state.
pub struct StreamServer {
    body: Arc<Vec<u8>>,
    plan: Arc<StreamPlan>,
    permits: Arc<Semaphore>,
    cfg: StreamServerConfig,
}

fn read_key(path: &std::path::Path) -> Result<SigningKey, String> {
    let raw = std::fs::read(path).map_err(|e| format!("key file {}: {e}", path.display()))?;
    let bytes: [u8; 32] = if raw.len() == 32 {
        raw.try_into().expect("32")
    } else {
        let t = String::from_utf8_lossy(&raw);
        let t = t.trim();
        if t.len() != 64 || !t.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("key file must hold 32 raw bytes or 64 hex characters".into());
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(&t[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())?;
        }
        out
    };
    Ok(SigningKey::from_bytes(&bytes))
}

impl StreamServer {
    /// Read the body and key, build the staged descriptor and frame plan.
    pub fn load(cfg: StreamServerConfig) -> Result<Arc<Self>, String> {
        let key = read_key(&cfg.key_path)?;
        let p = BuildParams {
            chunk_log2: cfg.chunk_log2,
            epoch: cfg.epoch,
            ..Default::default()
        };
        // The artifact's file_id: the signed root page's FileIdentity.
        let tail = std::fs::read(&cfg.body_path).map_err(|e| format!("body: {e}"))?;
        let mut file_id = [0u8; 16];
        if tail.len() > 4096 && tail[tail.len() - 4096..][..4] == *b"0MVR" {
            file_id.copy_from_slice(&tail[tail.len() - 4096 + 0xF00..][..16]);
        }
        let (body, bundle) = descriptor_for(&cfg.body_path, file_id, key, &p)
            .map_err(|e| format!("descriptor: {e}"))?;
        let plan = StreamPlan::new(bundle, &body);
        Ok(Arc::new(Self {
            permits: Arc::new(Semaphore::new(cfg.max_concurrent_streams.max(1))),
            body: Arc::new(body),
            plan: Arc::new(plan),
            cfg,
        }))
    }

    /// The signed descriptor bytes.
    pub fn descriptor(&self) -> &[u8] {
        &self.plan.bundle().descriptor
    }

    /// Total bytes of the frame stream.
    pub fn stream_len(&self) -> u64 {
        self.plan.total_len()
    }

    /// Configuration.
    pub fn config(&self) -> &StreamServerConfig {
        &self.cfg
    }

    /// Copy `[start, end)` of the frame stream.
    pub fn read_range(&self, start: u64, end: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity((end.saturating_sub(start)).min(1 << 24) as usize);
        self.plan.write_range(&self.body, start, end, &mut |b| {
            out.extend_from_slice(b);
            true
        });
        out
    }

    /// Take one stream permit (held until dropped), if any is free.
    pub fn try_permit(&self) -> Option<OwnedSemaphorePermit> {
        self.permits.clone().try_acquire_owned().ok()
    }
}

/// Router for the stream endpoints.
pub fn router(server: Arc<StreamServer>) -> Router {
    Router::new()
        .route("/v1/stream/descriptor", get(descriptor))
        .route("/v1/stream/body", get(body))
        .with_state(server)
}

async fn descriptor(State(s): State<Arc<StreamServer>>) -> Response {
    let mut h = HeaderMap::new();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    h.insert(
        "x-rvf-descriptor-sha256",
        HeaderValue::from_str(&hex(&s.plan.bundle().descriptor_hash)).expect("hex"),
    );
    (StatusCode::OK, h, s.descriptor().to_vec()).into_response()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Parse `bytes=a-` / `bytes=a-b`. Anything else is unsatisfiable.
fn parse_range(v: &str, total: u64) -> Result<(u64, u64), ()> {
    let spec = v.trim().strip_prefix("bytes=").ok_or(())?;
    if spec.contains(',') {
        return Err(()); // multi-range
    }
    let (a, b) = spec.split_once('-').ok_or(())?;
    if a.is_empty() {
        return Err(()); // suffix range
    }
    let start: u64 = a.parse().map_err(|_| ())?;
    let last = if b.is_empty() {
        total.saturating_sub(1)
    } else {
        b.parse::<u64>()
            .map_err(|_| ())?
            .min(total.saturating_sub(1))
    };
    if start >= total || last < start {
        return Err(());
    }
    Ok((start, last + 1))
}

async fn body(State(s): State<Arc<StreamServer>>, headers: HeaderMap) -> Response {
    let total = s.stream_len();
    let (start, mut end, partial) = match headers.get(header::RANGE) {
        None => (0, total, false),
        Some(v) => match v.to_str().ok().and_then(|v| parse_range(v, total).ok()) {
            Some((a, z)) => (a, z, true),
            None => {
                let mut h = HeaderMap::new();
                h.insert(
                    header::CONTENT_RANGE,
                    HeaderValue::from_str(&format!("bytes */{total}")).expect("hdr"),
                );
                return (StatusCode::RANGE_NOT_SATISFIABLE, h).into_response();
            }
        },
    };
    if partial {
        end = end.min(start + s.cfg.max_range_bytes);
    }
    let Some(permit) = s.try_permit() else {
        return (StatusCode::TOO_MANY_REQUESTS, "too many concurrent streams").into_response();
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(4);
    let srv = s.clone();
    let held = Arc::new(permit);
    tokio::task::spawn_blocking(move || {
        let mut acc: Vec<u8> = Vec::with_capacity(128 << 10);
        let mut open = true;
        srv.plan.write_range(&srv.body, start, end, &mut |piece| {
            acc.extend_from_slice(piece);
            if acc.len() >= 64 << 10 {
                open = tx
                    .blocking_send(Ok(Bytes::from(std::mem::take(&mut acc))))
                    .is_ok();
            }
            open
        });
        if open && !acc.is_empty() {
            let _ = tx.blocking_send(Ok(Bytes::from(acc)));
        }
    });
    // The permit lives as long as the response body: released when the body
    // finishes or the connection (and so the stream) is dropped.
    let stream = futures_util::stream::unfold((rx, held), |(mut rx, held)| async move {
        rx.recv().await.map(|v| (v, (rx, held)))
    });
    let mut h = HeaderMap::new();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    h.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&(end - start).to_string()).expect("len"),
    );
    let status = if partial {
        h.insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {}-{}/{}", start, end - 1, total)).expect("hdr"),
        );
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    (status, h, Body::from_stream(stream)).into_response()
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn range_parser_rejects_abuse() {
        assert_eq!(parse_range("bytes=0-", 100), Ok((0, 100)));
        assert_eq!(parse_range("bytes=10-19", 100), Ok((10, 20)));
        assert_eq!(parse_range("bytes=90-5000", 100), Ok((90, 100)));
        for bad in [
            "bytes=0-1,5-6",
            "bytes=-5",
            "bytes=100-",
            "bytes=20-10",
            "bytes=a-b",
            "items=0-1",
            "bytes=0-1-2",
            "bytes=18446744073709551616-",
        ] {
            assert!(parse_range(bad, 100).is_err(), "{bad}");
        }
    }
}
