#!/bin/bash
# ╔═══════════════════════════════════════╗
# ║  NebulaBrowser Launcher               ║
# ║  Firefox-based, privacy-first         ║
# ╚═══════════════════════════════════════╝

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE_SRC="$SCRIPT_DIR/firefox-profile"
PROFILE_DIR="$HOME/.nebula-browser"

# Create profile directory if needed
if [ ! -d "$PROFILE_DIR" ]; then
    echo "[nebula] Creating profile at $PROFILE_DIR ..."
    mkdir -p "$PROFILE_DIR"
fi

# Always sync user.js (overrides) from source
cp -f "$PROFILE_SRC/user.js" "$PROFILE_DIR/user.js"

# Sync chrome directory (custom CSS)
mkdir -p "$PROFILE_DIR/chrome"
cp -f "$PROFILE_SRC/chrome/userChrome.css" "$PROFILE_DIR/chrome/userChrome.css"
# Enable userChrome.css loading
if ! grep -q "toolkit.legacyUserProfileCustomizations.stylesheets" "$PROFILE_DIR/user.js" 2>/dev/null; then
    echo 'user_pref("toolkit.legacyUserProfileCustomizations.stylesheets", true);' >> "$PROFILE_DIR/user.js"
fi

# Install extension (as temporary unsigned for development)
EXT_DIR="$PROFILE_SRC/extensions/nebula-privacy"

# Check for Tor and enable SOCKS proxy if running
if pgrep -x "tor" > /dev/null 2>&1 || pgrep -x "arti" > /dev/null 2>&1; then
    echo "[nebula] Tor detected — routing traffic through SOCKS5 127.0.0.1:9150"
    # Append proxy settings
    cat >> "$PROFILE_DIR/user.js" <<'TOREOF'
user_pref("network.proxy.type", 1);
user_pref("network.proxy.socks", "127.0.0.1");
user_pref("network.proxy.socks_port", 9150);
user_pref("network.proxy.socks_version", 5);
user_pref("network.proxy.socks_remote_dns", true);
user_pref("network.proxy.no_proxies_on", "");
TOREOF
else
    echo "[nebula] Tor not detected — direct connection (DNS encrypted via DoH)"
fi

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║           NebulaBrowser v0.1.0               ║"
echo "  ║   Powered by Gecko (Firefox 148)             ║"
echo "  ║                                              ║"
echo "  ║   ✓ DNS-over-HTTPS (Cloudflare)              ║"
echo "  ║   ✓ Encrypted Client Hello (ECH)             ║"
echo "  ║   ✓ Anti-fingerprinting (Win11 spoof)        ║"
echo "  ║   ✓ Tracking param stripping                 ║"
echo "  ║   ✓ Privacy frontend redirects               ║"
echo "  ║   ✓ WebRTC leak prevention                   ║"
echo "  ║   ✓ HTTPS-Only mode                          ║"
echo "  ║   ✓ All telemetry killed                     ║"
if pgrep -x "tor" > /dev/null 2>&1 || pgrep -x "arti" > /dev/null 2>&1; then
echo "  ║   ✓ Tor SOCKS5 proxy active                  ║"
fi
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# Launch Firefox with our hardened profile
# --no-remote allows running alongside normal Firefox
exec firefox \
    --profile "$PROFILE_DIR" \
    --no-remote \
    --class NebulaBrowser \
    "$@"
