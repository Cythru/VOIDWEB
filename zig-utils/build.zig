//! build.zig — Oracle Zig utilities build script
//!
//! Compiles arena, ring, simd, hash modules into a static library
//! for linking into the Rust/C++ layers.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Static library ────────────────────────────────────────────────────────
    const lib = b.addStaticLibrary(.{
        .name     = "zig_utils",
        .root_source_file = b.path("src/root.zig"),
        .target   = target,
        .optimize = optimize,
    });

    // Expose as C ABI for Rust FFI.
    lib.bundle_compiler_rt = true;
    lib.link_libc = true;

    b.installArtifact(lib);

    // ── Header generation (for C consumers) ──────────────────────────────────
    // If we add @export'd C functions, zig can emit the header:
    // const header = lib.getEmittedH();
    // b.installFile(header, "include/zig_utils.h");

    // ── Tests ─────────────────────────────────────────────────────────────────
    const test_arena = b.addTest(.{
        .root_source_file = b.path("src/alloc/arena.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    const test_ring = b.addTest(.{
        .root_source_file = b.path("src/ring/ring.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    const test_simd = b.addTest(.{
        .root_source_file = b.path("src/simd/simd.zig"),
        .target           = target,
        .optimize         = optimize,
    });

    const run_arena = b.addRunArtifact(test_arena);
    const run_ring  = b.addRunArtifact(test_ring);
    const run_simd  = b.addRunArtifact(test_simd);

    const test_step = b.step("test", "Run all Zig unit tests");
    test_step.dependOn(&run_arena.step);
    test_step.dependOn(&run_ring.step);
    test_step.dependOn(&run_simd.step);
}
