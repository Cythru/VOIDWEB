// VOIDWEB — Tor Integration via Arti
// All traffic optionally routed through Tor for anonymity.
// Per-tab Tor circuits, .onion support, bridge/pluggable transport support.

use std::path::PathBuf;

/// Tor connection mode
#[derive(Debug, Clone, PartialEq)]
pub enum TorMode {
    /// No Tor — direct connection (fastest)
    Disabled,
    /// Route all traffic through Tor (default for privacy)
    AlwaysOn,
    /// Only use Tor for .onion addresses
    OnionOnly,
    /// Per-tab: user can toggle Tor per tab
    PerTab,
}

/// Bridge configuration for censored networks
#[derive(Debug, Clone)]
pub enum BridgeType {
    None,
    Obfs4 { bridge_line: String },
    Snowflake,
    Meek { url: String },
    WebTunnel { url: String },
}

/// Tor proxy configuration
pub struct TorConfig {
    pub mode: TorMode,
    pub socks_port: u16,
    pub control_port: u16,
    pub data_dir: PathBuf,
    pub bridge: BridgeType,
    pub enforce_https: bool,
    pub isolate_by_tab: bool,       // Separate circuit per tab
    pub new_circuit_on_domain: bool, // New circuit for each domain
    pub exit_country: Option<String>, // Preferred exit node country
}

impl Default for TorConfig {
    fn default() -> Self {
        Self {
            mode: TorMode::AlwaysOn,
            socks_port: 9150,
            control_port: 9151,
            data_dir: dirs_config().join("tor"),
            bridge: BridgeType::None,
            enforce_https: true,
            isolate_by_tab: true,
            new_circuit_on_domain: true,
            exit_country: None,
        }
    }
}

fn dirs_config() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("voidweb")
}

/// Circuit information for a tab
#[derive(Debug, Clone)]
pub struct CircuitInfo {
    pub circuit_id: u64,
    pub entry_node: String,
    pub middle_node: String,
    pub exit_node: String,
    pub exit_country: String,
    pub latency_ms: u32,
    pub established_at: std::time::SystemTime,
}

/// Tor proxy manager
pub struct TorProxy {
    config: TorConfig,
    running: bool,
    circuits: Vec<CircuitInfo>,
    stats: TorStats,
}

#[derive(Debug, Default)]
pub struct TorStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub circuits_created: u64,
    pub circuits_failed: u64,
    pub onion_sites_visited: u64,
}

impl TorProxy {
    pub fn new(config: TorConfig) -> Self {
        Self {
            config,
            running: false,
            circuits: Vec::new(),
            stats: TorStats::default(),
        }
    }

    /// Start the Tor proxy (via arti or system tor)
    pub fn start(&mut self) -> Result<(), String> {
        if self.config.mode == TorMode::Disabled {
            return Ok(());
        }

        // Create data directory
        std::fs::create_dir_all(&self.config.data_dir)
            .map_err(|e| format!("Failed to create tor dir: {}", e))?;

        // Write torrc
        let torrc = self.generate_torrc();
        let torrc_path = self.config.data_dir.join("torrc");
        std::fs::write(&torrc_path, &torrc)
            .map_err(|e| format!("Failed to write torrc: {}", e))?;

        // Try arti first (Rust Tor implementation), fall back to system tor
        let arti_result = std::process::Command::new("arti")
            .arg("proxy")
            .arg("-c")
            .arg(&torrc_path)
            .spawn();

        match arti_result {
            Ok(_child) => {
                self.running = true;
                eprintln!("[tor] Started arti proxy on SOCKS5 port {}", self.config.socks_port);
                Ok(())
            }
            Err(_) => {
                // Fall back to system tor
                let tor_result = std::process::Command::new("tor")
                    .arg("-f")
                    .arg(&torrc_path)
                    .spawn();

                match tor_result {
                    Ok(_child) => {
                        self.running = true;
                        eprintln!("[tor] Started system tor on SOCKS5 port {}", self.config.socks_port);
                        Ok(())
                    }
                    Err(e) => Err(format!(
                        "Neither arti nor tor found. Install one:\n  cargo install arti\n  sudo pacman -S tor\nError: {}",
                        e
                    )),
                }
            }
        }
    }

    /// Stop the Tor proxy
    pub fn stop(&mut self) {
        if self.running {
            // Kill tor/arti process
            let _ = std::process::Command::new("pkill").arg("arti").output();
            let _ = std::process::Command::new("pkill").arg("-x").arg("tor").output();
            self.running = false;
            eprintln!("[tor] Proxy stopped");
        }
    }

    /// Get SOCKS5 proxy URL for HTTP client configuration
    pub fn socks_url(&self) -> Option<String> {
        if self.running || self.config.mode != TorMode::Disabled {
            Some(format!("socks5://127.0.0.1:{}", self.config.socks_port))
        } else {
            None
        }
    }

    /// Request a new circuit (new identity)
    pub fn new_circuit(&mut self) -> Result<(), String> {
        self.stats.circuits_created += 1;
        // Send NEWNYM signal via control port
        let signal = format!(
            "AUTHENTICATE\r\nSIGNAL NEWNYM\r\nQUIT\r\n"
        );
        // TCP connect to control port and send signal
        use std::io::Write;
        let mut stream = std::net::TcpStream::connect(
            format!("127.0.0.1:{}", self.config.control_port)
        ).map_err(|e| format!("Failed to connect to control port: {}", e))?;
        stream.write_all(signal.as_bytes())
            .map_err(|e| format!("Failed to send NEWNYM: {}", e))?;
        Ok(())
    }

    /// Check if URL is a .onion address
    pub fn is_onion(url: &str) -> bool {
        url.contains(".onion")
    }

    /// Should this URL go through Tor?
    pub fn should_proxy(&self, url: &str) -> bool {
        match self.config.mode {
            TorMode::Disabled => false,
            TorMode::AlwaysOn => true,
            TorMode::OnionOnly => Self::is_onion(url),
            TorMode::PerTab => true, // Controlled at tab level
        }
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn stats(&self) -> &TorStats {
        &self.stats
    }

    fn generate_torrc(&self) -> String {
        let mut torrc = format!(
            "SocksPort {}\nControlPort {}\nDataDirectory {}\n",
            self.config.socks_port,
            self.config.control_port,
            self.config.data_dir.display()
        );

        if self.config.enforce_https {
            torrc.push_str("# HTTPS enforcement handled at browser level\n");
        }

        if let Some(ref country) = self.config.exit_country {
            torrc.push_str(&format!("ExitNodes {{{}}}\n", country));
        }

        match &self.config.bridge {
            BridgeType::None => {}
            BridgeType::Obfs4 { bridge_line } => {
                torrc.push_str("UseBridges 1\n");
                torrc.push_str(&format!("Bridge obfs4 {}\n", bridge_line));
                torrc.push_str("ClientTransportPlugin obfs4 exec /usr/bin/obfs4proxy\n");
            }
            BridgeType::Snowflake => {
                torrc.push_str("UseBridges 1\n");
                torrc.push_str("ClientTransportPlugin snowflake exec /usr/bin/snowflake-client\n");
                torrc.push_str("Bridge snowflake 192.0.2.3:80 2B280B23E1107BB62ABFC40DDCC8824814F80A72 fingerprint=2B280B23E1107BB62ABFC40DDCC8824814F80A72 url=https://snowflake-broker.torproject.net.global.prod.fastly.net/ front=cdn.sstatic.net ice=stun:stun.l.google.com:19302,stun:stun.antisip.com:3478,stun:stun.bluesip.net:3478\n");
            }
            BridgeType::Meek { url } => {
                torrc.push_str("UseBridges 1\n");
                torrc.push_str(&format!("Bridge meek_lite {} url={}\n", "0.0.2.0:2", url));
            }
            BridgeType::WebTunnel { url } => {
                torrc.push_str("UseBridges 1\n");
                torrc.push_str(&format!("Bridge webtunnel {}\n", url));
            }
        }

        torrc
    }
}
