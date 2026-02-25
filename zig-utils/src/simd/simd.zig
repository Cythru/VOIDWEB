//! simd.zig — Oracle SIMD Utilities
//!
//! Zig exposes SIMD vectors natively through `@Vector` — no intrinsics required.
//! The compiler maps these to AVX2 / AVX-512 / NEON / SVE automatically.
//!
//! Provides:
//!   - Vectorised dot product (f32 × N)
//!   - Vectorised softmax (f32 × N)
//!   - Vectorised argmax (f32 × N)
//!   - BF16 ↔ F32 conversion helpers
//!   - SIMD memory fill / copy

const std = @import("std");
const builtin = @import("builtin");

// Vector width — auto-selected at compile time.
// 16 × f32 = 512-bit (AVX-512 / SVE2 / NEON x4)
const VEC_WIDTH: comptime_int = switch (builtin.cpu.arch) {
    .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 16 else 8,
    .aarch64 => 4,
    else => 4,
};

const F32x = @Vector(VEC_WIDTH, f32);
const I32x = @Vector(VEC_WIDTH, i32);
const U16x = @Vector(VEC_WIDTH, u16);

// ── Dot product ───────────────────────────────────────────────────────────────

/// SIMD dot product of two f32 slices (must be same length).
pub fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var acc: F32x = @splat(0.0);
    var i: usize = 0;
    while (i + VEC_WIDTH <= n) : (i += VEC_WIDTH) {
        const va: F32x = a[i..][0..VEC_WIDTH].*;
        const vb: F32x = b[i..][0..VEC_WIDTH].*;
        acc += va * vb;
    }
    // Horizontal sum.
    var sum: f32 = @reduce(.Add, acc);
    // Scalar tail.
    while (i < n) : (i += 1) sum += a[i] * b[i];
    return sum;
}

// ── Argmax ────────────────────────────────────────────────────────────────────

pub fn argmaxF32(v: []const f32) usize {
    var best_val: f32 = -std.math.inf(f32);
    var best_idx: usize = 0;
    var i: usize = 0;
    // SIMD chunk: find maximum in each lane.
    const max_vec: F32x = @splat(-std.math.inf(f32));
    var mv = max_vec;
    var mi: @Vector(VEC_WIDTH, u32) = @splat(0);
    while (i + VEC_WIDTH <= v.len) : (i += VEC_WIDTH) {
        const vv: F32x = v[i..][0..VEC_WIDTH].*;
        const mask = vv > mv;
        mv = @select(f32, mask, vv, mv);
        // Track indices.
        var idx_vec: @Vector(VEC_WIDTH, u32) = undefined;
        comptime var k = 0;
        inline while (k < VEC_WIDTH) : (k += 1) {
            idx_vec[k] = @intCast(i + k);
        }
        mi = @select(u32, mask, idx_vec, mi);
    }
    // Reduce SIMD result.
    comptime var k = 0;
    inline while (k < VEC_WIDTH) : (k += 1) {
        if (mv[k] > best_val) { best_val = mv[k]; best_idx = mi[k]; }
    }
    // Scalar tail.
    while (i < v.len) : (i += 1) {
        if (v[i] > best_val) { best_val = v[i]; best_idx = i; }
    }
    return best_idx;
}

// ── In-place softmax ──────────────────────────────────────────────────────────

pub fn softmaxF32(v: []f32) void {
    // 1. Find max.
    var vmax: f32 = -std.math.inf(f32);
    var i: usize = 0;
    while (i + VEC_WIDTH <= v.len) : (i += VEC_WIDTH) {
        const vv: F32x = v[i..][0..VEC_WIDTH].*;
        vmax = @max(vmax, @reduce(.Max, vv));
    }
    while (i < v.len) : (i += 1) vmax = @max(vmax, v[i]);

    // 2. Exp(x - max) and sum.
    const vmax_vec: F32x = @splat(vmax);
    var sum_vec: F32x = @splat(0.0);
    i = 0;
    while (i + VEC_WIDTH <= v.len) : (i += VEC_WIDTH) {
        var vv: F32x = v[i..][0..VEC_WIDTH].*;
        vv = @exp(vv - vmax_vec);
        v[i..][0..VEC_WIDTH].* = vv;
        sum_vec += vv;
    }
    var sum: f32 = @reduce(.Add, sum_vec);
    while (i < v.len) : (i += 1) {
        v[i] = @exp(v[i] - vmax);
        sum += v[i];
    }

    // 3. Normalise.
    const inv_sum_vec: F32x = @splat(1.0 / sum);
    i = 0;
    while (i + VEC_WIDTH <= v.len) : (i += VEC_WIDTH) {
        v[i..][0..VEC_WIDTH].* = @as(F32x, v[i..][0..VEC_WIDTH].*) * inv_sum_vec;
    }
    while (i < v.len) : (i += 1) v[i] /= sum;
}

// ── BF16 ↔ F32 ────────────────────────────────────────────────────────────────

/// Convert a BF16 u16 value to f32 (no-op bit shift).
pub inline fn bf16ToF32(v: u16) f32 {
    const bits: u32 = @as(u32, v) << 16;
    return @bitCast(bits);
}

/// Convert f32 to BF16 (round-to-nearest-even).
pub inline fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const rounding: u32 = 0x7fff + ((bits >> 16) & 1);
    return @intCast((bits + rounding) >> 16);
}

/// Convert a slice of BF16 (u16) values to f32 in-place (output buf must be same len).
pub fn bf16SliceToF32(src: []const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, 0..) |v, i| dst[i] = bf16ToF32(v);
}

// ── Memory helpers ────────────────────────────────────────────────────────────

pub fn fillF32(dst: []f32, val: f32) void {
    const vv: F32x = @splat(val);
    var i: usize = 0;
    while (i + VEC_WIDTH <= dst.len) : (i += VEC_WIDTH) dst[i..][0..VEC_WIDTH].* = vv;
    while (i < dst.len) : (i += 1) dst[i] = val;
}

// ── Tests ──────────────────────────────────────────────────────────────────────
test "dotF32" {
    const a = [_]f32{1, 2, 3, 4};
    const b = [_]f32{4, 3, 2, 1};
    const r = dotF32(&a, &b);
    try std.testing.expectApproxEqAbs(r, 20.0, 1e-5);
}

test "argmaxF32" {
    const v = [_]f32{0.1, 0.9, 0.3, 0.7};
    try std.testing.expect(argmaxF32(&v) == 1);
}

test "softmaxF32" {
    var v = [_]f32{1.0, 2.0, 3.0};
    softmaxF32(&v);
    var sum: f32 = 0;
    for (v) |x| sum += x;
    try std.testing.expectApproxEqAbs(sum, 1.0, 1e-6);
}

test "bf16 roundtrip" {
    const f: f32 = 3.14159;
    const b = f32ToBf16(f);
    const back = bf16ToF32(b);
    try std.testing.expectApproxEqAbs(f, back, 0.01);
}
