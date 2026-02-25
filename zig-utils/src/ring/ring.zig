//! ring.zig — Oracle Lock-Free SPSC Ring Buffer
//!
//! Single-Producer / Single-Consumer ring buffer.
//! Used to pass completed token IDs from the engine thread
//! to the HTTP server's SSE sender thread with zero copies.
//!
//! Suckless: fixed-size, no heap allocation after init, power-of-2 capacity.
//! Cache-line padded producer/consumer indices to avoid false sharing.

const std = @import("std");
const assert = std.debug.assert;
const AtomicOrder = std.builtin.AtomicOrder;
const Atomic = std.atomic.Value;

/// Cache line size (x86-64 / ARM64).
const CACHE_LINE = 64;

pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    comptime assert(std.math.isPowerOfTwo(capacity));
    const MASK = capacity - 1;

    return struct {
        const Self = @This();

        // Producer index on its own cache line.
        head:    Atomic(usize) align(CACHE_LINE) = Atomic(usize).init(0),
        _pad0:   [CACHE_LINE - @sizeOf(usize)]u8 = undefined,

        // Consumer index on its own cache line.
        tail:    Atomic(usize) align(CACHE_LINE) = Atomic(usize).init(0),
        _pad1:   [CACHE_LINE - @sizeOf(usize)]u8 = undefined,

        // Data storage.
        data:    [capacity]T = undefined,

        /// Push one item (producer side).  Returns false if full.
        pub fn push(self: *Self, item: T) bool {
            const h = self.head.load(.monotonic);
            const next = (h + 1) & MASK;
            if (next == self.tail.load(.acquire)) return false; // full
            self.data[h & MASK] = item;
            self.head.store(h + 1, .release);
            return true;
        }

        /// Pop one item (consumer side).  Returns null if empty.
        pub fn pop(self: *Self) ?T {
            const t = self.tail.load(.monotonic);
            if (t == self.head.load(.acquire)) return null; // empty
            const item = self.data[t & MASK];
            self.tail.store(t + 1, .release);
            return item;
        }

        /// Push a batch; returns number actually pushed.
        pub fn pushBatch(self: *Self, items: []const T) usize {
            var n: usize = 0;
            for (items) |item| {
                if (!self.push(item)) break;
                n += 1;
            }
            return n;
        }

        /// Pop up to `out.len` items; returns slice of filled portion.
        pub fn popBatch(self: *Self, out: []T) []T {
            var n: usize = 0;
            while (n < out.len) {
                out[n] = self.pop() orelse break;
                n += 1;
            }
            return out[0..n];
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.head.load(.acquire) == self.tail.load(.acquire);
        }

        pub fn isFull(self: *const Self) bool {
            const h = self.head.load(.acquire);
            const next = (h + 1) & MASK;
            return next == self.tail.load(.acquire);
        }

        pub fn len(self: *const Self) usize {
            const h = self.head.load(.acquire);
            const t = self.tail.load(.acquire);
            return (h - t) & MASK;
        }
    };
}

// ── Oracle-specific ring types ────────────────────────────────────────────────
/// Token output ring: engine → SSE sender.  4096 slots per sequence.
pub const TokenRing = RingBuffer(u32, 4096);

/// Request ID ring: HTTP handler → scheduler.
pub const ReqRing = RingBuffer(u64, 512);

// ── Tests ──────────────────────────────────────────────────────────────────────
test "spsc ring push/pop" {
    var ring = RingBuffer(u32, 8){};
    try std.testing.expect(ring.isEmpty());
    try std.testing.expect(ring.push(1));
    try std.testing.expect(ring.push(2));
    try std.testing.expect(ring.push(3));
    try std.testing.expect(ring.len() == 3);
    try std.testing.expect(ring.pop().? == 1);
    try std.testing.expect(ring.pop().? == 2);
    ring.push(4) |> _ = &;
    try std.testing.expect(ring.len() == 2);
}

test "ring batch ops" {
    var ring = RingBuffer(u32, 16){};
    const src = [_]u32{10, 20, 30, 40, 50};
    const pushed = ring.pushBatch(&src);
    try std.testing.expect(pushed == 5);
    var dst: [8]u32 = undefined;
    const got = ring.popBatch(&dst);
    try std.testing.expect(got.len == 5);
    try std.testing.expect(got[0] == 10);
    try std.testing.expect(got[4] == 50);
}
