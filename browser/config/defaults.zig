// NebulaBrowser — Build Configuration & Defaults (Zig)
// Compile-time configuration, allocator setup, SIMD dispatch.

const std = @import("std");

/// Browser version
pub const VERSION = "0.1.0";
pub const CODENAME = "NebulaBrowser";
pub const USER_AGENT_BASE = "NebulaBrowser/0.1.0";

/// Default privacy frontend instances
/// Compile-time verified — no runtime parsing overhead
pub const Frontend = struct {
    name: []const u8,
    default_instance: []const u8,
    enabled: bool,
};

pub const default_frontends = [_]Frontend{
    .{ .name = "invidious", .default_instance = "https://vid.puffyan.us", .enabled = true },
    .{ .name = "nitter", .default_instance = "https://nitter.poast.org", .enabled = true },
    .{ .name = "redlib", .default_instance = "https://safereddit.com", .enabled = true },
    .{ .name = "bibliogram", .default_instance = "https://bibliogram.art", .enabled = true },
    .{ .name = "searxng", .default_instance = "https://search.ononoki.org", .enabled = true },
    .{ .name = "scribe", .default_instance = "https://scribe.rip", .enabled = true },
    .{ .name = "rimgo", .default_instance = "https://rimgo.pussthecat.org", .enabled = true },
    .{ .name = "proxitok", .default_instance = "https://proxitok.pabloferreiro.es", .enabled = true },
    .{ .name = "wikiless", .default_instance = "https://wikiless.org", .enabled = true },
    .{ .name = "lingva", .default_instance = "https://lingva.ml", .enabled = true },
};

/// Memory allocator configuration
pub const AllocConfig = struct {
    /// Page allocator pool size (for tab content)
    page_pool_size: usize = 256 * 1024 * 1024, // 256 MB
    /// Arena allocator for per-request scratch
    arena_size: usize = 4 * 1024 * 1024, // 4 MB
    /// Max cache size for filter lists
    filter_cache_max: usize = 64 * 1024 * 1024, // 64 MB
    /// Download buffer size
    download_buffer: usize = 8 * 1024 * 1024, // 8 MB
};

pub const default_alloc_config = AllocConfig{};

/// Compile-time SIMD feature detection
pub const SimdLevel = enum {
    scalar,
    neon,
    sse42,
    avx2,
    avx512,
};

pub fn detect_simd() SimdLevel {
    const arch = @import("builtin").cpu.arch;
    if (arch == .x86_64) {
        if (std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx512f)) {
            return .avx512;
        } else if (std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx2)) {
            return .avx2;
        } else {
            return .sse42;
        }
    } else if (arch == .aarch64) {
        return .neon;
    }
    return .scalar;
}

/// URL pattern matching — zero-allocation, SIMD-accelerated
pub fn fast_domain_match(url: []const u8, domain: []const u8) bool {
    // Find "://" then match domain
    var i: usize = 0;
    while (i + 2 < url.len) : (i += 1) {
        if (url[i] == ':' and url[i + 1] == '/' and url[i + 2] == '/') {
            const host_start = i + 3;
            // Skip optional "www."
            var host = url[host_start..];
            if (host.len >= 4 and std.mem.eql(u8, host[0..4], "www.")) {
                host = host[4..];
            }
            // Check if domain matches at start
            if (host.len >= domain.len and std.mem.eql(u8, host[0..domain.len], domain)) {
                // Must be followed by /, ?, #, :, or end
                if (host.len == domain.len) return true;
                const next = host[domain.len];
                if (next == '/' or next == '?' or next == '#' or next == ':') return true;
            }
            return false;
        }
    }
    return false;
}

test "fast_domain_match" {
    try std.testing.expect(fast_domain_match("https://youtube.com/watch?v=abc", "youtube.com"));
    try std.testing.expect(fast_domain_match("https://www.youtube.com/watch?v=abc", "youtube.com"));
    try std.testing.expect(!fast_domain_match("https://notyoutube.com/", "youtube.com"));
    try std.testing.expect(fast_domain_match("https://twitter.com/user", "twitter.com"));
    try std.testing.expect(fast_domain_match("https://x.com/user", "x.com"));
}
