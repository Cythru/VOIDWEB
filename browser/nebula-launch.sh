#!/bin/bash
# ╔═══════════════════════════════════════╗
# ║  NebulaBrowser Launcher               ║
# ║  Firefox/Gecko — zero Google           ║
# ╚═══════════════════════════════════════╝

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE_SRC="$SCRIPT_DIR/firefox-profile"
PROFILE_DIR="$HOME/.nebula-browser"

# Create profile directory if needed
if [ ! -d "$PROFILE_DIR" ]; then
    echo "[nebula] First launch — creating profile at $PROFILE_DIR ..."
    mkdir -p "$PROFILE_DIR"
fi

# Sync user.js with path substitution
sed "s|__PROFILE_DIR__|$PROFILE_DIR|g" "$PROFILE_SRC/user.js" > "$PROFILE_DIR/user.js"

# Sync chrome directory (custom theme CSS)
mkdir -p "$PROFILE_DIR/chrome"
cp -f "$PROFILE_SRC/chrome/userChrome.css" "$PROFILE_DIR/chrome/userChrome.css"

# Copy new tab page
cp -f "$PROFILE_SRC/newtab.html" "$PROFILE_DIR/newtab.html"

# Copy extension for manual loading
mkdir -p "$PROFILE_DIR/extensions-src"
cp -rf "$PROFILE_SRC/extensions/nebula-privacy" "$PROFILE_DIR/extensions-src/"

# Check for Tor and enable SOCKS proxy if running
TOR_ACTIVE=false
if pgrep -x "tor" > /dev/null 2>&1 || pgrep -x "arti" > /dev/null 2>&1; then
    TOR_ACTIVE=true
    cat >> "$PROFILE_DIR/user.js" <<'TOREOF'
user_pref("network.proxy.type", 1);
user_pref("network.proxy.socks", "127.0.0.1");
user_pref("network.proxy.socks_port", 9150);
user_pref("network.proxy.socks_version", 5);
user_pref("network.proxy.socks_remote_dns", true);
user_pref("network.proxy.no_proxies_on", "");
TOREOF
fi

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║                                              ║"
echo "  ║     ◈  N E B U L A  B R O W S E R  ◈        ║"
echo "  ║         v0.1.0 — Gecko Engine                ║"
echo "  ║                                              ║"
echo "  ║   ◉ DNS-over-HTTPS          ◉ ECH Active     ║"
echo "  ║   ◉ Win11 Fingerprint       ◉ WebRTC Blocked ║"
echo "  ║   ◉ Tracking Stripped       ◉ HTTPS-Only     ║"
echo "  ║   ◉ Frontend Redirects      ◉ Telemetry Dead ║"
if [ "$TOR_ACTIVE" = true ]; then
echo "  ║   ◉ Tor SOCKS5 Proxy        ◉ Censorship-Proof║"
else
echo "  ║   ○ Tor Offline             ◉ DoH Encrypted  ║"
fi
echo "  ║                                              ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""
echo "  [tip] Load extension: about:debugging → This Firefox"
echo "        → Load Temporary Add-on → select:"
echo "        $PROFILE_DIR/extensions-src/nebula-privacy/manifest.json"
echo ""

# Launch — desktop vs Android/Termux
if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
    # Android: Firefox profile can't be passed via CLI — profile is synced above
    echo "  [android] Profile synced to: $PROFILE_DIR"
    echo "  [android] Import manually: Firefox → about:config"
    echo "             or copy user.js to your Firefox profile folder."
    echo ""
    # Try launching Firefox for Android (Fenix or legacy)
    if command -v am &>/dev/null; then
        am start -a android.intent.action.VIEW \
            -d "about:blank" \
            -n org.mozilla.fenix/org.mozilla.fenix.HomeActivity 2>/dev/null \
        || am start -a android.intent.action.VIEW \
            -d "about:blank" \
            -n org.mozilla.firefox/org.mozilla.gecko.BrowserApp 2>/dev/null \
        || echo "  [android] Install Firefox (Fenix) from F-Droid, then re-run."
    else
        echo "  [android] Open Firefox manually — privacy profile is staged."
    fi
else
    # Desktop — launch Firefox with our hardened profile
    exec firefox \
        --profile "$PROFILE_DIR" \
        --no-remote \
        --class NebulaBrowser \
        "$@"
fi
