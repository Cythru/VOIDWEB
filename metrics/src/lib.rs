//! metrics — Prometheus-compatible counters + live benchmark stats
//!
//! Tracks per-request and aggregate performance:
//!   • TTFT       — time-to-first-token (µs)
//!   • ITL        — inter-token latency (µs / token)
//!   • Throughput — tokens / second
//!   • Cache hit rate — radix cache prefix reuse %
//!   • Active requests, queue depth, KV cache utilisation
//!
//! Exposes a /metrics HTTP endpoint (Prometheus text format) via `serve()`.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

// ── Atomic counters ────────────────────────────────────────────────────────────

pub struct Counters {
    // Request lifecycle
    pub requests_total:    AtomicU64,
    pub requests_active:   AtomicUsize,
    pub requests_queued:   AtomicUsize,
    pub tokens_generated:  AtomicU64,
    pub tokens_prefill:    AtomicU64,

    // Latency accumulators (µs, use running sum / count for mean)
    pub ttft_sum_us:       AtomicU64,
    pub ttft_count:        AtomicU64,
    pub itl_sum_us:        AtomicU64,
    pub itl_count:         AtomicU64,

    // Cache
    pub cache_hits:        AtomicU64,
    pub cache_misses:      AtomicU64,
    pub cache_blocks_used: AtomicUsize,
    pub cache_blocks_total:AtomicUsize,
}

impl Default for Counters {
    fn default() -> Self {
        macro_rules! zero { () => { AtomicU64::new(0) } }
        macro_rules! uzero { () => { AtomicUsize::new(0) } }
        Self {
            requests_total:     zero!(),
            requests_active:    uzero!(),
            requests_queued:    uzero!(),
            tokens_generated:   zero!(),
            tokens_prefill:     zero!(),
            ttft_sum_us:        zero!(),
            ttft_count:         zero!(),
            itl_sum_us:         zero!(),
            itl_count:          zero!(),
            cache_hits:         zero!(),
            cache_misses:       zero!(),
            cache_blocks_used:  uzero!(),
            cache_blocks_total: uzero!(),
        }
    }
}

impl Counters {
    /// Record a TTFT measurement.
    pub fn record_ttft(&self, dur: Duration) {
        self.ttft_sum_us.fetch_add(dur.as_micros() as u64, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an ITL measurement.
    pub fn record_itl(&self, dur: Duration) {
        self.itl_sum_us.fetch_add(dur.as_micros() as u64, Ordering::Relaxed);
        self.itl_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Mean TTFT in milliseconds.
    pub fn mean_ttft_ms(&self) -> f64 {
        let count = self.ttft_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.ttft_sum_us.load(Ordering::Relaxed) as f64 / count as f64 / 1000.0
    }

    /// Mean ITL in milliseconds.
    pub fn mean_itl_ms(&self) -> f64 {
        let count = self.itl_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.itl_sum_us.load(Ordering::Relaxed) as f64 / count as f64 / 1000.0
    }

    /// Cache hit rate as a fraction 0.0–1.0.
    pub fn cache_hit_rate(&self) -> f64 {
        let hits   = self.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f64;
        let total  = hits + misses;
        if total == 0.0 { 0.0 } else { hits / total }
    }
}

// ── Global metrics handle ─────────────────────────────────────────────────────

lazy_static::lazy_static! {
    pub static ref METRICS: Arc<Counters> = Arc::new(Counters::default());
}

/// Convenience accessor.
pub fn metrics() -> Arc<Counters> {
    Arc::clone(&*METRICS)
}

// ── Prometheus text format ─────────────────────────────────────────────────────

pub fn prometheus_text() -> String {
    let m      = metrics();
    let ts     = SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    format!(
r#"# HELP oracle_requests_total Total inference requests received
# TYPE oracle_requests_total counter
oracle_requests_total {requests_total}

# HELP oracle_requests_active Requests currently being processed
# TYPE oracle_requests_active gauge
oracle_requests_active {requests_active}

# HELP oracle_tokens_generated_total Total tokens generated
# TYPE oracle_tokens_generated_total counter
oracle_tokens_generated_total {tokens_generated}

# HELP oracle_ttft_ms_mean Mean time-to-first-token in milliseconds
# TYPE oracle_ttft_ms_mean gauge
oracle_ttft_ms_mean {ttft_ms:.3}

# HELP oracle_itl_ms_mean Mean inter-token latency in milliseconds
# TYPE oracle_itl_ms_mean gauge
oracle_itl_ms_mean {itl_ms:.3}

# HELP oracle_cache_hit_rate Radix KV-cache prefix hit rate (0.0–1.0)
# TYPE oracle_cache_hit_rate gauge
oracle_cache_hit_rate {cache_hit_rate:.4}

# HELP oracle_cache_blocks_used KV-cache blocks currently in use
# TYPE oracle_cache_blocks_used gauge
oracle_cache_blocks_used {cache_blocks_used}
"#,
        requests_total   = m.requests_total.load(Ordering::Relaxed),
        requests_active  = m.requests_active.load(Ordering::Relaxed),
        tokens_generated = m.tokens_generated.load(Ordering::Relaxed),
        ttft_ms          = m.mean_ttft_ms(),
        itl_ms           = m.mean_itl_ms(),
        cache_hit_rate   = m.cache_hit_rate(),
        cache_blocks_used= m.cache_blocks_used.load(Ordering::Relaxed),
    )
}
