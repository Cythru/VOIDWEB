// NebulaBrowser — Rust Library Root
// All Rust modules for the browser's security-critical components.

pub mod adblock {
    #[path = "adblock/adblock.rs"]
    pub mod engine;
}

pub mod shield {
    #[path = "shield/malware_scanner.rs"]
    pub mod malware_scanner;
}

pub mod tor {
    #[path = "tor/tor_proxy.rs"]
    pub mod tor_proxy;
}

pub mod net {
    #[path = "net/privacy_net.rs"]
    pub mod privacy_net;
}

pub mod auth {
    #[path = "auth/authenticator.rs"]
    pub mod authenticator;
}

pub mod core {
    #[path = "core/sandbox.rs"]
    pub mod sandbox;
}
