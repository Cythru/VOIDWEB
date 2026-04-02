#!/usr/bin/env bash
# NebulaBrowser — Installer
# Suckless: pick your modules. Build only what you want.
# Runs on: Android (Termux), Linux, macOS, Windows (WSL/Git Bash)
# Usage: bash install.sh

set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
R='\033[0m'; B='\033[1m'; DIM='\033[2m'; C='\033[36m'; G='\033[32m'
Y='\033[33m'; RED='\033[31m'; M='\033[35m'

# ── platform detect ───────────────────────────────────────────────────────────
detect_platform() {
    if [[ -d /data/data/com.termux ]]; then echo "android"
    elif [[ "$(uname)" == "Darwin" ]]; then echo "macos"
    elif grep -qi microsoft /proc/version 2>/dev/null; then echo "wsl"
    else echo "linux"
    fi
}
PLATFORM=$(detect_platform)

# ── logo ─────────────────────────────────────────────────────────────────────
clear
echo -e "${C}${B}"
cat << 'EOF'
  ███╗   ██╗███████╗██████╗ ██╗   ██╗██╗      █████╗
  ████╗  ██║██╔════╝██╔══██╗██║   ██║██║     ██╔══██╗
  ██╔██╗ ██║█████╗  ██████╔╝██║   ██║██║     ███████║
  ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║██║     ██╔══██║
  ██║ ╚████║███████╗██████╔╝╚██████╔╝███████╗██║  ██║
  ╚═╝  ╚═══╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
EOF
echo -e "${R}"
echo -e "  ${DIM}Firefox · Privacy · Open Source · No Chrome · No Compromise${R}"
echo -e "  ${DIM}Platform: ${PLATFORM}${R}"
echo ""

# ── module registry ───────────────────────────────────────────────────────────
# Format: "key|label|description|default(on/off)|required(true/false)"
declare -a MODULES=(
    "core|Core Engine|GeckoView (Firefox) + tab manager. Required.|on|true"
    "hardening|Hardening|LibreWolf-level user.js, arkenfox prefs, no telemetry.|on|false"
    "spoof|Crowd-Blend Spoofing|Chrome/Win11 UA, canvas noise, letterboxing. Blend in.|on|false"
    "voidblock|VoidBlock|Built-in ad/tracker blocker. uBlock-level filter lists.|on|false"
    "containers|Containers|Firefox Multi-Account Containers. Per-container Tor circuits.|on|false"
    "tor|Tor Integration|Route containers through Tor. Requires Orbot on Android.|off|false"
    "sandbox|Sandboxing|Seccomp-BPF, process isolation, per-tab namespaces.|on|false"
    "voidshield-ai|VoidShield AI|1B model security auditor. URL/script/download scanning.|off|false"
    "terminal|Terminal Mode|Browse in terminal. Firefox headless + text renderer.|off|false"
)

# Track selection state
declare -A SELECTED
for entry in "${MODULES[@]}"; do
    IFS='|' read -r key label desc default required <<< "$entry"
    SELECTED[$key]=$default
done

# ── module selection UI ───────────────────────────────────────────────────────
print_modules() {
    echo -e "  ${B}Select modules:${R}  ${DIM}↑↓ navigate  space toggle  enter confirm${R}"
    echo ""
    local i=0
    for entry in "${MODULES[@]}"; do
        IFS='|' read -r key label desc default required <<< "$entry"
        local state="${SELECTED[$key]}"
        local prefix="  "
        if [[ $i -eq $CURSOR ]]; then prefix="${C}${B}▶ ${R}"; fi
        if [[ "$state" == "on" ]]; then
            echo -e "${prefix}${G}[✓]${R} ${B}${label}${R}  ${DIM}${desc}${R}"
        else
            echo -e "${prefix}${DIM}[ ] ${label}${R}  ${DIM}${desc}${R}"
        fi
        if [[ "$required" == "true" ]]; then
            echo -e "    ${DIM}(required)${R}"
        fi
        ((i++))
    done
    echo ""
}

# Simple non-interactive fallback (for pipes/CI)
if [[ ! -t 0 ]] || [[ "${1:-}" == "--all" ]]; then
    echo -e "  ${DIM}Non-interactive mode — installing with defaults${R}\n"
else
    # Interactive selection
    CURSOR=0
    NMODULES=${#MODULES[@]}

    tput civis 2>/dev/null || true
    trap 'tput cnorm 2>/dev/null; echo ""' EXIT

    while true; do
        clear
        echo -e "${C}${B}"
        echo "  NebulaBrowser — Module Selection"
        echo -e "${R}"
        print_modules

        # Read input
        IFS= read -r -s -n1 key 2>/dev/null || key=""
        if [[ "$key" == $'\x1b' ]]; then
            read -r -s -n2 key2 2>/dev/null || key2=""
            key="${key}${key2}"
        fi

        case "$key" in
            $'\x1b[A'|k) # up
                ((CURSOR--)) || CURSOR=$((NMODULES-1))
                [[ $CURSOR -lt 0 ]] && CURSOR=$((NMODULES-1))
                ;;
            $'\x1b[B'|j) # down
                ((CURSOR++)) || CURSOR=0
                [[ $CURSOR -ge $NMODULES ]] && CURSOR=0
                ;;
            ' ') # toggle
                entry="${MODULES[$CURSOR]}"
                IFS='|' read -r mkey label desc default required <<< "$entry"
                if [[ "$required" == "true" ]]; then
                    : # can't toggle required
                elif [[ "${SELECTED[$mkey]}" == "on" ]]; then
                    SELECTED[$mkey]="off"
                else
                    SELECTED[$mkey]="on"
                fi
                ;;
            ''|$'\n') # confirm
                break
                ;;
            q|Q)
                echo -e "\n  ${DIM}Cancelled.${R}"
                exit 0
                ;;
        esac
    done
    tput cnorm 2>/dev/null || true
fi

# ── summary ───────────────────────────────────────────────────────────────────
clear
echo -e "${C}${B}  NebulaBrowser — Build Plan${R}\n"
for entry in "${MODULES[@]}"; do
    IFS='|' read -r key label desc default required <<< "$entry"
    if [[ "${SELECTED[$key]}" == "on" ]]; then
        echo -e "  ${G}✓${R}  ${label}"
    else
        echo -e "  ${DIM}✗  ${label}${R}"
    fi
done
echo ""
echo -e "  ${DIM}Platform: ${PLATFORM}${R}"
echo ""
read -rp "  Build? [Y/n] " confirm
[[ "${confirm:-Y}" =~ ^[Nn] ]] && echo -e "\n  ${DIM}Cancelled.${R}" && exit 0

# ── dependency install ────────────────────────────────────────────────────────
echo -e "\n  ${DIM}Installing dependencies...${R}"
case "$PLATFORM" in
    android)
        pkg install openjdk-21 gradle kotlin -y 2>&1 | grep -E "Setting up|already" | sed 's/^/  /'
        ;;
    linux)
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y openjdk-21-jdk gradle 2>&1 | tail -3 | sed 's/^/  /'
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm jdk-openjdk gradle 2>&1 | tail -3 | sed 's/^/  /'
        fi
        ;;
    macos)
        brew install openjdk gradle 2>&1 | tail -3 | sed 's/^/  /'
        ;;
esac

# ── write enabled-modules.txt ─────────────────────────────────────────────────
ENABLED=()
for entry in "${MODULES[@]}"; do
    IFS='|' read -r key label desc default required <<< "$entry"
    [[ "${SELECTED[$key]}" == "on" ]] && ENABLED+=("$key")
done
printf '%s\n' "${ENABLED[@]}" > enabled-modules.txt
echo -e "  ${DIM}modules locked: ${ENABLED[*]}${R}"

# ── generate settings.gradle from selected modules ────────────────────────────
{
    echo "rootProject.name = 'nebula-browser'"
    echo "include ':app'"
    for mod in "${ENABLED[@]}"; do
        [[ "$mod" != "core" ]] && echo "include ':modules:${mod}'"
    done
} > settings.gradle
echo -e "  ${DIM}settings.gradle written${R}"

# ── build ─────────────────────────────────────────────────────────────────────
echo -e "\n  ${C}${B}Building...${R}"
if command -v gradle &>/dev/null; then
    gradle assembleRelease 2>&1 | grep -E "BUILD|error:|warning:|:assemble" | sed 's/^/  /' || {
        echo -e "  ${Y}Build needs Android SDK — see README for setup${R}"
    }
fi

# ── install APK ───────────────────────────────────────────────────────────────
APK=$(find . -name "*.apk" 2>/dev/null | head -1)
if [[ -n "$APK" ]] && [[ "$PLATFORM" == "android" ]]; then
    echo -e "\n  ${G}${B}Installing APK...${R}"
    am start -a android.intent.action.VIEW \
        -d "file://${APK}" \
        -t "application/vnd.android.package-archive" 2>/dev/null \
    && echo -e "  ${G}Done — check your notifications${R}" \
    || echo -e "  ${Y}APK at: ${APK}${R}"
elif [[ -n "$APK" ]]; then
    echo -e "\n  ${G}APK ready: ${APK}${R}"
else
    echo -e "\n  ${DIM}APK build pending Android SDK setup.${R}"
    echo -e "  ${DIM}Run: bash install.sh again after SDK is installed.${R}"
fi

echo -e "\n  ${C}${B}NebulaBrowser — done.${R}\n"
