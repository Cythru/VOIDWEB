//! arena.zig — Oracle Arena Allocator
//!
//! A suckless, zero-dependency arena allocator.
//!
//! Design:
//!   - Single large mmap'd slab (configurable size, default 256 MB)
//!   - Bump-pointer allocation: O(1) alloc, O(1) free-all
//!   - Thread-local (no locks on the hot path)
//!   - Reset: single atomic store (resets the bump pointer)
//!   - Alignment: always aligned to requested alignment (power-of-two)
//!
//! Suckless principle: do one thing well.  No fragmentation tracking,
//! no per-allocation free — just allocate forward and reset when done.
//! For per-allocation free, use a pool allocator on top.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const MMAP_PROT  = std.os.PROT.READ | std.os.PROT.WRITE;
const MMAP_FLAGS = std.os.MAP{ .TYPE = .PRIVATE, .ANONYMOUS = true };

pub const DEFAULT_SIZE: usize = 256 * 1024 * 1024; // 256 MB

pub const ArenaError = error{
    OutOfMemory,
    InvalidAlignment,
    MmapFailed,
};

pub const Arena = struct {
    base:     [*]u8,   // mmap'd slab base
    capacity: usize,   // total bytes
    pos:      usize,   // current bump position (atomic for multi-reader sanity)

    /// Initialise an arena backed by an anonymous mmap.
    pub fn init(capacity: usize) ArenaError!Arena {
        const mem = std.os.mmap(
            null,
            capacity,
            MMAP_PROT,
            MMAP_FLAGS,
            -1,
            0,
        ) catch return ArenaError.MmapFailed;

        // Advise the kernel: sequential access, keep in memory.
        _ = std.os.madvise(mem.ptr, capacity, std.os.MADV.SEQUENTIAL) catch {};

        return Arena{
            .base     = mem.ptr,
            .capacity = capacity,
            .pos      = 0,
        };
    }

    /// Allocate `size` bytes aligned to `alignment` (must be power-of-2).
    pub fn alloc(self: *Arena, size: usize, alignment: usize) ArenaError![*]u8 {
        assert(std.math.isPowerOfTwo(alignment));
        const aligned_pos = std.mem.alignForward(usize, self.pos, alignment);
        const new_pos = aligned_pos + size;
        if (new_pos > self.capacity) return ArenaError.OutOfMemory;
        self.pos = new_pos;
        return self.base + aligned_pos;
    }

    /// Typed allocation helper.
    pub fn create(self: *Arena, comptime T: type) ArenaError!*T {
        const ptr = try self.alloc(@sizeOf(T), @alignOf(T));
        return @ptrCast(@alignCast(ptr));
    }

    /// Allocate a slice of `n` elements of type T.
    pub fn allocSlice(self: *Arena, comptime T: type, n: usize) ArenaError![]T {
        const ptr = try self.alloc(@sizeOf(T) * n, @alignOf(T));
        const typed: [*]T = @ptrCast(@alignCast(ptr));
        return typed[0..n];
    }

    /// Save the current position (lightweight checkpoint).
    pub fn save(self: *const Arena) usize {
        return self.pos;
    }

    /// Restore to a previously saved position (partial free).
    pub fn restore(self: *Arena, checkpoint: usize) void {
        assert(checkpoint <= self.pos);
        self.pos = checkpoint;
    }

    /// Reset the arena to zero (free everything — O(1)).
    pub fn reset(self: *Arena) void {
        self.pos = 0;
    }

    /// How many bytes are currently used.
    pub fn used(self: *const Arena) usize {
        return self.pos;
    }

    /// How many bytes remain.
    pub fn remaining(self: *const Arena) usize {
        return self.capacity - self.pos;
    }

    /// Release the mmap'd slab back to the OS.
    pub fn deinit(self: *Arena) void {
        std.os.munmap(self.base[0..self.capacity]);
        self.* = undefined;
    }
};

// ── Thread-local scratch arena ────────────────────────────────────────────────
/// 8 MB per-thread scratch arena for temporaries.
/// Automatically reset at the start of each request processing cycle.
threadlocal var tl_scratch: ?Arena = null;

pub fn getScratch() *Arena {
    if (tl_scratch == null) {
        tl_scratch = Arena.init(8 * 1024 * 1024) catch @panic("scratch arena OOM");
    }
    return &tl_scratch.?;
}

// ── Tests ──────────────────────────────────────────────────────────────────────
test "arena basic alloc/reset" {
    var a = try Arena.init(1024 * 1024);
    defer a.deinit();

    const p1 = try a.alloc(64, 8);
    const p2 = try a.alloc(128, 16);
    try std.testing.expect(@intFromPtr(p2) > @intFromPtr(p1));
    try std.testing.expect(a.used() == std.mem.alignForward(usize, 64, 16) + 128);

    const ck = a.save();
    _ = try a.alloc(256, 8);
    a.restore(ck);
    try std.testing.expect(a.used() == ck);

    a.reset();
    try std.testing.expect(a.used() == 0);
}

test "arena typed create" {
    const T = struct { x: f64, y: f64 };
    var a = try Arena.init(4096);
    defer a.deinit();
    const t = try a.create(T);
    t.x = 1.0; t.y = 2.0;
    try std.testing.expect(t.x == 1.0);
}
