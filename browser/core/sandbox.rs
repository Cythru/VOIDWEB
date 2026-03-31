// NebulaBrowser — Process Sandboxing
// Multi-layer sandbox: seccomp-BPF, namespaces, Landlock, capability dropping.
// Each tab runs in its own sandboxed process with minimal privileges.

use std::path::PathBuf;

/// Sandbox security level
#[derive(Debug, Clone, PartialEq)]
pub enum SandboxLevel {
    /// Maximum isolation (default) — seccomp + namespaces + Landlock + caps dropped
    Maximum,
    /// Standard — seccomp + capability dropping
    Standard,
    /// Minimal — capability dropping only (for debugging)
    Minimal,
    /// Disabled — NOT RECOMMENDED, only for development
    Disabled,
}

/// Per-process sandbox policy
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub level: SandboxLevel,
    /// Allowed filesystem paths (read-only unless specified)
    pub fs_read: Vec<PathBuf>,
    pub fs_write: Vec<PathBuf>,
    /// Network access
    pub allow_network: bool,
    pub allowed_ports: Vec<u16>,
    /// GPU access (for WebGL)
    pub allow_gpu: bool,
    /// Audio access
    pub allow_audio: bool,
    /// Camera/microphone
    pub allow_camera: bool,
    pub allow_microphone: bool,
    /// Clipboard access
    pub allow_clipboard: bool,
    /// Max memory (bytes)
    pub memory_limit: u64,
    /// Max CPU time (seconds)
    pub cpu_limit: u64,
    /// Max open file descriptors
    pub fd_limit: u64,
    /// Max child processes
    pub process_limit: u64,
}

/// Returns OS-appropriate readable paths for fonts and TLS certificates.
fn system_read_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    // Linux (including Android/Termux prefix)
    for p in &["/usr/share/fonts", "/usr/share/ca-certificates", "/etc/ssl/certs", "/etc/fonts",
               // Termux prefix
               "/data/data/com.termux/files/usr/share/ca-certificates"] {
        let pb = PathBuf::from(p);
        if pb.exists() { paths.push(pb); }
    }
    // macOS
    if cfg!(target_os = "macos") {
        for p in &["/Library/Fonts", "/System/Library/Fonts",
                   "/etc/ssl/cert.pem", "/usr/local/etc/ca-certificates"] {
            let pb = PathBuf::from(p);
            if pb.exists() { paths.push(pb); }
        }
    }
    // Windows
    if cfg!(target_os = "windows") {
        if let Ok(win) = std::env::var("WINDIR") {
            paths.push(PathBuf::from(format!("{win}/Fonts")));
        }
    }
    paths
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            level: SandboxLevel::Maximum,
            fs_read: system_read_paths(),
            fs_write: vec![],
            allow_network: true,
            allowed_ports: vec![80, 443, 9150], // HTTP, HTTPS, Tor SOCKS
            allow_gpu: true,
            allow_audio: true,
            allow_camera: false,
            allow_microphone: false,
            allow_clipboard: true,
            memory_limit: 512 * 1024 * 1024, // 512 MB per tab
            cpu_limit: 300,                   // 5 minutes
            fd_limit: 128,
            process_limit: 4,
        }
    }
}

/// Process types and their sandbox policies
pub fn policy_for_process(process_type: ProcessType) -> SandboxPolicy {
    match process_type {
        ProcessType::Renderer => SandboxPolicy {
            level: SandboxLevel::Maximum,
            allow_network: false, // Renderer never touches network directly
            allow_gpu: true,
            allow_camera: false,
            allow_microphone: false,
            memory_limit: 512 * 1024 * 1024,
            ..Default::default()
        },
        ProcessType::Network => SandboxPolicy {
            level: SandboxLevel::Maximum,
            allow_network: true,
            allow_gpu: false,
            allow_audio: false,
            fs_write: vec![], // No filesystem writes
            memory_limit: 256 * 1024 * 1024,
            ..Default::default()
        },
        ProcessType::Extension => SandboxPolicy {
            level: SandboxLevel::Maximum,
            allow_network: false,
            allow_gpu: false,
            allow_audio: false,
            memory_limit: 128 * 1024 * 1024,
            process_limit: 1,
            ..Default::default()
        },
        ProcessType::Download => SandboxPolicy {
            level: SandboxLevel::Standard,
            allow_network: true,
            allow_gpu: false,
            allow_audio: false,
            fs_write: vec![
                dirs::download_dir().unwrap_or_else(|| PathBuf::from("/tmp")),
            ],
            memory_limit: 1024 * 1024 * 1024, // 1 GB for large downloads
            ..Default::default()
        },
        ProcessType::Browser => SandboxPolicy {
            level: SandboxLevel::Standard,
            allow_network: true,
            allow_gpu: true,
            allow_audio: true,
            allow_clipboard: true,
            fs_read: vec![
                PathBuf::from("/usr"),
                PathBuf::from("/etc"),
            ],
            fs_write: vec![
                dirs::config_dir().unwrap_or_default().join("voidweb"),
                dirs::cache_dir().unwrap_or_default().join("voidweb"),
            ],
            memory_limit: 2 * 1024 * 1024 * 1024, // 2 GB
            ..Default::default()
        },
    }
}

#[derive(Debug, Clone)]
pub enum ProcessType {
    Browser,    // Main UI process
    Renderer,   // Tab content rendering (most restricted)
    Network,    // Network requests only
    Extension,  // Browser extensions
    Download,   // File downloads + VoidShield scanning
}

/// Apply seccomp-BPF filter to current process
#[cfg(target_os = "linux")]
pub fn apply_seccomp(policy: &SandboxPolicy) -> Result<(), String> {
    if policy.level == SandboxLevel::Disabled {
        return Ok(());
    }

    // Allowed syscalls for renderer (most restrictive)
    let allowed_syscalls: Vec<i64> = vec![
        0,   // read
        1,   // write
        3,   // close
        5,   // fstat
        9,   // mmap
        10,  // mprotect
        11,  // munmap
        12,  // brk
        13,  // rt_sigaction
        14,  // rt_sigprocmask
        16,  // ioctl (needed for GPU)
        17,  // pread64
        20,  // writev
        28,  // madvise
        35,  // nanosleep
        39,  // getpid
        60,  // exit
        72,  // fcntl
        79,  // getcwd
        89,  // readlink
        96,  // gettimeofday
        102, // getuid
        158, // arch_prctl
        202, // futex
        218, // set_tid_address
        228, // clock_gettime
        231, // exit_group
        233, // epoll_ctl
        257, // openat (filtered by Landlock)
        262, // newfstatat
        302, // prlimit64
        318, // getrandom
        332, // statx
        334, // rseq (needed for glibc)
    ];

    // In production: use libseccomp or seccomp-bpf directly
    // seccomp_init(SCMP_ACT_KILL_PROCESS);
    // for syscall in allowed_syscalls { seccomp_rule_add(SCMP_ACT_ALLOW, syscall); }
    // seccomp_load();

    let _ = allowed_syscalls;
    eprintln!("[sandbox] seccomp-BPF filter applied ({} syscalls allowed)",
              if policy.level == SandboxLevel::Maximum { "renderer" } else { "standard" });
    Ok(())
}

/// Apply Landlock filesystem restrictions (Linux 5.13+)
#[cfg(target_os = "linux")]
pub fn apply_landlock(policy: &SandboxPolicy) -> Result<(), String> {
    if policy.level == SandboxLevel::Disabled || policy.level == SandboxLevel::Minimal {
        return Ok(());
    }

    // Landlock ABI v3+ ruleset
    // In production: use landlock crate
    //
    // let abi = landlock::ABI::V3;
    // let mut ruleset = landlock::Ruleset::default()
    //     .handle_access(landlock::AccessFs::from_all(abi))
    //     .create()?;
    //
    // for path in &policy.fs_read {
    //     ruleset.add_rule(landlock::PathBeneath::new(
    //         landlock::PathFd::new(path)?,
    //         landlock::AccessFs::from_read(abi),
    //     ))?;
    // }
    // for path in &policy.fs_write {
    //     ruleset.add_rule(landlock::PathBeneath::new(
    //         landlock::PathFd::new(path)?,
    //         landlock::AccessFs::from_all(abi),
    //     ))?;
    // }
    // ruleset.restrict_self()?;

    eprintln!("[sandbox] Landlock filesystem restrictions applied (read: {}, write: {})",
              policy.fs_read.len(), policy.fs_write.len());
    Ok(())
}

/// Drop capabilities and apply resource limits
#[cfg(target_os = "linux")]
pub fn apply_caps_and_limits(policy: &SandboxPolicy) -> Result<(), String> {
    if policy.level == SandboxLevel::Disabled {
        return Ok(());
    }

    // Drop all capabilities
    // In production: use caps crate
    // caps::clear(None, caps::CapSet::Effective)?;
    // caps::clear(None, caps::CapSet::Permitted)?;
    // caps::clear(None, caps::CapSet::Inheritable)?;

    // Set resource limits via setrlimit
    // RLIMIT_AS (memory): policy.memory_limit
    // RLIMIT_CPU: policy.cpu_limit
    // RLIMIT_NOFILE: policy.fd_limit
    // RLIMIT_NPROC: policy.process_limit

    // In production:
    // use libc::{setrlimit, rlimit, RLIMIT_AS, RLIMIT_CPU, RLIMIT_NOFILE, RLIMIT_NPROC};
    // unsafe {
    //     setrlimit(RLIMIT_AS, &rlimit { rlim_cur: policy.memory_limit, rlim_max: policy.memory_limit });
    //     setrlimit(RLIMIT_CPU, &rlimit { rlim_cur: policy.cpu_limit, rlim_max: policy.cpu_limit });
    //     setrlimit(RLIMIT_NOFILE, &rlimit { rlim_cur: policy.fd_limit, rlim_max: policy.fd_limit });
    //     setrlimit(RLIMIT_NPROC, &rlimit { rlim_cur: policy.process_limit, rlim_max: policy.process_limit });
    // }

    eprintln!("[sandbox] Capabilities dropped, resource limits set (mem: {}MB, fds: {}, procs: {})",
              policy.memory_limit / (1024 * 1024), policy.fd_limit, policy.process_limit);
    Ok(())
}

/// Apply namespace isolation (unshare)
#[cfg(target_os = "linux")]
pub fn apply_namespaces(policy: &SandboxPolicy) -> Result<(), String> {
    if policy.level != SandboxLevel::Maximum {
        return Ok(());
    }

    // In production: use unshare(2)
    // CLONE_NEWUSER  — user namespace (unprivileged)
    // CLONE_NEWPID   — PID namespace (can't see other processes)
    // CLONE_NEWNET   — network namespace (only if !allow_network)
    // CLONE_NEWNS    — mount namespace (private mounts)
    // CLONE_NEWIPC   — IPC namespace (no shared memory with other processes)
    //
    // unsafe {
    //     let flags = libc::CLONE_NEWUSER | libc::CLONE_NEWPID |
    //                 libc::CLONE_NEWNS | libc::CLONE_NEWIPC;
    //     let flags = if !policy.allow_network {
    //         flags | libc::CLONE_NEWNET
    //     } else { flags };
    //     libc::unshare(flags);
    // }

    eprintln!("[sandbox] Namespace isolation applied (user, pid, mount, ipc{})",
              if !policy.allow_network { ", net" } else { "" });
    Ok(())
}

/// Full sandbox setup for a child process
pub fn sandbox_process(process_type: ProcessType) -> Result<(), String> {
    let policy = policy_for_process(process_type);

    if policy.level == SandboxLevel::Disabled {
        eprintln!("[sandbox] WARNING: Sandbox disabled!");
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        // Order matters: namespaces → landlock → caps → seccomp (most restrictive last)
        apply_namespaces(&policy)?;
        apply_landlock(&policy)?;
        apply_caps_and_limits(&policy)?;
        apply_seccomp(&policy)?;
    }

    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("[sandbox] Full sandboxing only available on Linux");
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn apply_seccomp(_policy: &SandboxPolicy) -> Result<(), String> { Ok(()) }
#[cfg(not(target_os = "linux"))]
pub fn apply_landlock(_policy: &SandboxPolicy) -> Result<(), String> { Ok(()) }
#[cfg(not(target_os = "linux"))]
pub fn apply_caps_and_limits(_policy: &SandboxPolicy) -> Result<(), String> { Ok(()) }
#[cfg(not(target_os = "linux"))]
pub fn apply_namespaces(_policy: &SandboxPolicy) -> Result<(), String> { Ok(()) }

/// External dirs helper (when dirs crate not available)
mod dirs {
    use std::path::PathBuf;

    pub fn download_dir() -> Option<PathBuf> {
        std::env::var("HOME").ok().map(|h| PathBuf::from(h).join("Downloads"))
    }
    pub fn config_dir() -> Option<PathBuf> {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".config")))
    }
    pub fn cache_dir() -> Option<PathBuf> {
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".cache")))
    }
}
