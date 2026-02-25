// VOIDWEB — Engine Launcher
//
// Starts and stops all inference engines for benchmarking.
// Replaces start_engines.sh with a proper native binary.
//
// Ports:
//   OracleInference   :8000
//   Oracle+RadixCache :8005
//   vLLM              :8001
//   TensorRT-LLM      :8002
//   SGLang            :8003
//   llama.cpp         :8004
//
// Build:  zig build-exe launcher.zig
//
// Usage:
//   ./launcher                          start all engines
//   ./launcher oracle vllm              start specific engines
//   ./launcher stop                     kill all engine processes
//   MODEL=Qwen/Qwen2.5-7B ./launcher

const std = @import("std");

// ── Engine definitions ────────────────────────────────────────────────────────

const Engine = struct {
    key:  []const u8,
    name: []const u8,
    port: u16,
};

const ENGINES = [_]Engine{
    .{ .key = "oracle",       .name = "OracleInference",   .port = 8000 },
    .{ .key = "oracle_radix", .name = "Oracle+RadixCache",  .port = 8005 },
    .{ .key = "vllm",         .name = "vLLM",               .port = 8001 },
    .{ .key = "tensorrt",     .name = "TensorRT-LLM",       .port = 8002 },
    .{ .key = "sglang",       .name = "SGLang",             .port = 8003 },
    .{ .key = "llamacpp",     .name = "llama.cpp",          .port = 8004 },
};

// ── Config from environment ───────────────────────────────────────────────────

const Config = struct {
    model:       []const u8 = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    gpu_util:    []const u8 = "0.78",
    max_len:     []const u8 = "4096",
    dtype:       []const u8 = "bfloat16",
    oracle_bin:  []const u8 = "./target/release/oracle-server",
    vllm_py:     []const u8 = "python3",
    sglang_py:   []const u8 = "python3",
    llamacpp:    []const u8 = "llama-server",
};

fn get_env(allocator: std.mem.Allocator, key: []const u8, default: []const u8) []const u8 {
    return std.process.getEnvVarOwned(allocator, key) catch default;
}

// ── Port check ────────────────────────────────────────────────────────────────

fn port_in_use(port: u16) bool {
    const addr = std.net.Address.initIp4(.{ 127, 0, 0, 1 }, port);
    const stream = std.net.tcpConnectToAddress(addr) catch return false;
    stream.close();
    return true;
}

fn wait_ready(name: []const u8, port: u16, timeout_s: u32) bool {
    std.debug.print("  Waiting for {s} to come up on :{d}...", .{ name, port });
    var elapsed: u32 = 0;
    while (elapsed < timeout_s) : (elapsed += 2) {
        std.time.sleep(2 * std.time.ns_per_s);
        std.debug.print(".", .{});
        if (port_in_use(port)) {
            std.debug.print(" ready ({d}s)\n", .{elapsed + 2});
            return true;
        }
    }
    std.debug.print(" TIMEOUT after {d}s\n", .{timeout_s});
    return false;
}

// ── Process helpers ───────────────────────────────────────────────────────────

fn start_bg(allocator: std.mem.Allocator, name: []const u8, argv: []const []const u8) !void {
    const log_path = try std.fmt.allocPrint(allocator, "/tmp/bench_{s}.log", .{name});
    defer allocator.free(log_path);

    std.debug.print("  Starting {s} → {s}\n", .{ name, log_path });

    const log_file = try std.fs.createFileAbsolute(log_path, .{});
    defer log_file.close();

    var child = std.process.Child.init(argv, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    _ = try child.spawn();

    // Write PID file
    const pid_path = try std.fmt.allocPrint(allocator, "/tmp/bench_{s}.pid", .{name});
    defer allocator.free(pid_path);
    const pid_file = try std.fs.createFileAbsolute(pid_path, .{});
    defer pid_file.close();
    try pid_file.writer().print("{d}\n", .{child.id});
}

fn stop_all() !void {
    const ports = [_]u16{ 8000, 8001, 8002, 8003, 8004, 8005 };
    std.debug.print("Stopping all benchmark engines...\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    for (ports) |port| {
        const pid_path = try std.fmt.allocPrint(allocator, "/tmp/bench_{d}.pid", .{port});
        defer allocator.free(pid_path);

        // Try reading PID file named by port (fallback: lsof)
        const pid_file = std.fs.openFileAbsolute(pid_path, .{}) catch continue;
        defer pid_file.close();

        var buf: [32]u8 = undefined;
        const n = try pid_file.read(&buf);
        const pid_str = std.mem.trimRight(u8, buf[0..n], "\n\r ");
        const pid = std.fmt.parseInt(std.posix.pid_t, pid_str, 10) catch continue;
        std.posix.kill(pid, std.posix.SIG.TERM) catch {};
        std.debug.print("  Sent SIGTERM to pid {d} (port {d})\n", .{ pid, port });
    }
}

// ── Engine launchers ──────────────────────────────────────────────────────────

fn launch_oracle(allocator: std.mem.Allocator, cfg: Config, radix: bool) !void {
    const port: u16 = if (radix) 8005 else 8000;
    const name = if (radix) "oracle_radix" else "oracle";
    const display = if (radix) "Oracle+RadixCache" else "OracleInference";

    if (!std.fs.accessAbsolute(cfg.oracle_bin, .{}) catch false and
        std.fs.cwd().access(cfg.oracle_bin, .{}) catch true)
    {
        std.debug.print("  [skip] oracle binary not found at {s}\n", .{cfg.oracle_bin});
        std.debug.print("         build with: cargo build --release\n", .{});
        return;
    }

    var argv = std.ArrayList([]const u8).init(allocator);
    defer argv.deinit();
    try argv.appendSlice(&.{
        cfg.oracle_bin,
        "--model",    cfg.model,
        "--port",     try std.fmt.allocPrint(allocator, "{d}", .{port}),
        "--host",     "0.0.0.0",
        "--dtype",    cfg.dtype,
        "--gpu-util", cfg.gpu_util,
        "--max-len",  cfg.max_len,
    });
    if (radix) try argv.append("--radix-cache");

    try start_bg(allocator, name, argv.items);
    _ = wait_ready(display, port, 120);
}

fn launch_vllm(allocator: std.mem.Allocator, cfg: Config) !void {
    // Check vLLM importable
    const check = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ cfg.vllm_py, "-c", "import vllm" },
    }) catch {
        std.debug.print("  [skip] vLLM not installed\n", .{});
        return;
    };
    allocator.free(check.stdout);
    allocator.free(check.stderr);
    if (check.term != .Exited or check.term.Exited != 0) {
        std.debug.print("  [skip] vLLM not installed (pip install vllm)\n", .{});
        return;
    }

    const argv = [_][]const u8{
        cfg.vllm_py, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                    cfg.model,
        "--port",                     "8001",
        "--host",                     "0.0.0.0",
        "--dtype",                    cfg.dtype,
        "--gpu-memory-utilization",   cfg.gpu_util,
        "--max-model-len",            cfg.max_len,
        "--enable-prefix-caching",
        "--disable-log-requests",
    };
    try start_bg(allocator, "vllm", &argv);
    _ = wait_ready("vLLM", 8001, 120);
}

fn launch_sglang(allocator: std.mem.Allocator, cfg: Config) !void {
    const check = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ cfg.sglang_py, "-c", "import sglang" },
    }) catch {
        std.debug.print("  [skip] SGLang not installed\n", .{});
        return;
    };
    allocator.free(check.stdout);
    allocator.free(check.stderr);
    if (check.term != .Exited or check.term.Exited != 0) {
        std.debug.print("  [skip] SGLang not installed (pip install sglang)\n", .{});
        return;
    }

    const argv = [_][]const u8{
        cfg.sglang_py, "-m", "sglang.launch_server",
        "--model-path",          cfg.model,
        "--port",                "8003",
        "--host",                "0.0.0.0",
        "--dtype",               cfg.dtype,
        "--mem-fraction-static", cfg.gpu_util,
        "--context-length",      cfg.max_len,
        "--enable-torch-compile",
        "--disable-radix-cache",
    };
    try start_bg(allocator, "sglang", &argv);
    _ = wait_ready("SGLang", 8003, 180);
}

fn launch_llamacpp(allocator: std.mem.Allocator, cfg: Config) !void {
    const gguf = std.process.getEnvVarOwned(allocator, "GGUF") catch "";
    if (gguf.len == 0) {
        std.debug.print("  [skip] llama.cpp — set GGUF=/path/to/model.gguf\n", .{});
        return;
    }

    const argv = [_][]const u8{
        cfg.llamacpp,
        "--model",      gguf,
        "--port",       "8004",
        "--host",       "0.0.0.0",
        "--n-gpu-layers", "999",
        "--ctx-size",   cfg.max_len,
        "--parallel",   "4",
        "--threads",    "8",
    };
    try start_bg(allocator, "llamacpp", &argv);
    _ = wait_ready("llama.cpp", 8004, 120);
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // stop mode
    if (args.len > 1 and std.mem.eql(u8, args[1], "stop")) {
        try stop_all();
        return;
    }

    const cfg = Config{
        .model      = get_env(allocator, "MODEL",       Config{}.model),
        .gpu_util   = get_env(allocator, "GPU_UTIL",    Config{}.gpu_util),
        .max_len    = get_env(allocator, "MAX_LEN",     Config{}.max_len),
        .dtype      = get_env(allocator, "DTYPE",       Config{}.dtype),
        .oracle_bin = get_env(allocator, "ORACLE_BIN",  Config{}.oracle_bin),
        .vllm_py    = get_env(allocator, "VLLM_PYTHON", Config{}.vllm_py),
        .sglang_py  = get_env(allocator, "SGLANG_PYTHON", Config{}.sglang_py),
        .llamacpp   = get_env(allocator, "LLAMACPP_BIN",  Config{}.llamacpp),
    };

    // determine which engines to start
    var selected = std.ArrayList([]const u8).init(allocator);
    defer selected.deinit();

    if (args.len <= 1) {
        for (ENGINES) |e| try selected.append(e.key);
    } else {
        for (args[1..]) |a| try selected.append(a);
    }

    std.debug.print("Starting engines: ", .{});
    for (selected.items) |e| std.debug.print("{s} ", .{e});
    std.debug.print("\nModel: {s}\n\n", .{cfg.model});

    for (selected.items) |key| {
        if (std.mem.eql(u8, key, "oracle")) {
            try launch_oracle(allocator, cfg, false);
        } else if (std.mem.eql(u8, key, "oracle_radix")) {
            try launch_oracle(allocator, cfg, true);
        } else if (std.mem.eql(u8, key, "vllm")) {
            try launch_vllm(allocator, cfg);
        } else if (std.mem.eql(u8, key, "sglang")) {
            try launch_sglang(allocator, cfg);
        } else if (std.mem.eql(u8, key, "llamacpp")) {
            try launch_llamacpp(allocator, cfg);
        } else if (std.mem.eql(u8, key, "tensorrt")) {
            std.debug.print("  [tensorrt] build TRT engine first — see bench/README\n", .{});
        } else {
            std.debug.print("  [unknown engine: {s}]\n", .{key});
        }
    }

    std.debug.print("\nAll engines up. Run:\n", .{});
    std.debug.print("  ./bench_runner --engines", .{});
    for (selected.items) |e| std.debug.print(" {s}", .{e});
    std.debug.print("\n\nStop all:  ./launcher stop\n", .{});
}
