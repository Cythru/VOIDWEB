#!/bin/bash
# NebulaBrowser Performance Benchmark
# Tests: DNS resolution, TLS handshake, page load, throughput

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║    NebulaBrowser Performance Benchmark       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# Colors
G='\033[0;32m'
Y='\033[0;33m'
C='\033[0;36m'
P='\033[0;35m'
R='\033[0m'

# ─────────────────────────────────────────
# 1. DNS Resolution Speed (DoH vs plaintext)
# ─────────────────────────────────────────
echo -e "${P}═══ DNS Resolution (DoH) ═══${R}"

dns_bench() {
    local domain=$1
    local start=$(date +%s%N)
    curl -s -o /dev/null -w "" "https://mozilla.cloudflare-dns.com/dns-query?name=${domain}&type=A" \
        -H "Accept: application/dns-json" 2>/dev/null
    local end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))
    echo -e "  ${C}${domain}${R} → ${G}${ms}ms${R}"
    echo $ms
}

echo "  Resolving via DoH (Cloudflare):"
dns_total=0
for domain in google.com github.com youtube.com reddit.com wikipedia.org; do
    ms=$(dns_bench "$domain" | tail -1)
    dns_total=$((dns_total + ms))
done
dns_avg=$((dns_total / 5))
echo -e "  ${Y}Average: ${dns_avg}ms${R}"
echo ""

# ─────────────────────────────────────────
# 2. TLS Handshake Speed
# ─────────────────────────────────────────
echo -e "${P}═══ TLS Handshake Speed ═══${R}"

tls_bench() {
    local host=$1
    local timing=$(curl -s -o /dev/null -w "%{time_appconnect}" "https://${host}" 2>/dev/null)
    local ms=$(echo "$timing * 1000" | bc 2>/dev/null || echo "0")
    printf "  ${C}%-25s${R} → ${G}%.0fms${R}\n" "$host" "$ms"
    echo "$ms"
}

tls_total=0
for host in google.com github.com cloudflare.com mozilla.org duckduckgo.com; do
    ms=$(tls_bench "$host" | tail -1)
    tls_total=$(echo "$tls_total + $ms" | bc 2>/dev/null || echo "0")
done
tls_avg=$(echo "$tls_total / 5" | bc 2>/dev/null || echo "0")
printf "  ${Y}Average: %.0fms${R}\n" "$tls_avg"
echo ""

# ─────────────────────────────────────────
# 3. Page Load Speed (TTFB + full download)
# ─────────────────────────────────────────
echo -e "${P}═══ Page Load Speed ═══${R}"

page_bench() {
    local url=$1
    local label=$2
    local result=$(curl -s -o /dev/null -w "dns:%{time_namelookup} connect:%{time_connect} tls:%{time_appconnect} ttfb:%{time_starttransfer} total:%{time_total} size:%{size_download}" "$url" 2>/dev/null)

    local ttfb=$(echo "$result" | grep -oP 'ttfb:\K[0-9.]+')
    local total=$(echo "$result" | grep -oP 'total:\K[0-9.]+')
    local size=$(echo "$result" | grep -oP 'size:\K[0-9]+')
    local size_kb=$((size / 1024))

    local ttfb_ms=$(echo "$ttfb * 1000" | bc 2>/dev/null || echo "0")
    local total_ms=$(echo "$total * 1000" | bc 2>/dev/null || echo "0")

    printf "  ${C}%-20s${R} TTFB: ${G}%4.0fms${R}  Total: ${G}%5.0fms${R}  Size: %dKB\n" "$label" "$ttfb_ms" "$total_ms" "$size_kb"
}

page_bench "https://www.google.com" "Google"
page_bench "https://github.com" "GitHub"
page_bench "https://en.wikipedia.org/wiki/Main_Page" "Wikipedia"
page_bench "https://www.cloudflare.com" "Cloudflare"
page_bench "https://duckduckgo.com" "DuckDuckGo"
page_bench "https://searx.be" "SearXNG"
page_bench "https://vid.puffyan.us" "Invidious"
page_bench "https://nitter.poast.org" "Nitter"
echo ""

# ─────────────────────────────────────────
# 4. Download Throughput
# ─────────────────────────────────────────
echo -e "${P}═══ Download Throughput ═══${R}"

# 10MB test file from Cloudflare
echo -n "  10MB download (Cloudflare): "
dl_result=$(curl -s -o /dev/null -w "%{speed_download} %{time_total}" \
    "https://speed.cloudflare.com/__down?bytes=10000000" 2>/dev/null)
dl_speed=$(echo "$dl_result" | awk '{printf "%.1f", $1/1048576}')
dl_time=$(echo "$dl_result" | awk '{printf "%.1f", $2}')
echo -e "${G}${dl_speed} MB/s${R} (${dl_time}s)"

# 1MB test
echo -n "  1MB download (Cloudflare):  "
dl_result=$(curl -s -o /dev/null -w "%{speed_download} %{time_total}" \
    "https://speed.cloudflare.com/__down?bytes=1000000" 2>/dev/null)
dl_speed=$(echo "$dl_result" | awk '{printf "%.1f", $1/1048576}')
dl_time=$(echo "$dl_result" | awk '{printf "%.2f", $2}')
echo -e "${G}${dl_speed} MB/s${R} (${dl_time}s)"
echo ""

# ─────────────────────────────────────────
# 5. HTTP/3 QUIC Support Check
# ─────────────────────────────────────────
echo -e "${P}═══ Protocol Support ═══${R}"

check_h3() {
    local host=$1
    local proto=$(curl -s -o /dev/null -w "%{http_version}" --http3-only "https://${host}" 2>/dev/null)
    if [ "$proto" = "3" ]; then
        echo -e "  ${C}${host}${R} → ${G}HTTP/3 (QUIC) ✓${R}"
    else
        proto=$(curl -s -o /dev/null -w "%{http_version}" "https://${host}" 2>/dev/null)
        echo -e "  ${C}${host}${R} → HTTP/${proto}"
    fi
}

check_h3 "www.google.com"
check_h3 "cloudflare.com"
check_h3 "www.facebook.com"
check_h3 "github.com"
echo ""

# ─────────────────────────────────────────
# 6. Privacy Frontend Availability
# ─────────────────────────────────────────
echo -e "${P}═══ Privacy Frontend Status ═══${R}"

check_frontend() {
    local name=$1
    local url=$2
    local code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null)
    if [ "$code" = "200" ] || [ "$code" = "301" ] || [ "$code" = "302" ]; then
        echo -e "  ${G}✓${R} ${C}${name}${R} (${url}) — ${G}UP${R}"
    else
        echo -e "  ${R}✗${R} ${C}${name}${R} (${url}) — ${Y}DOWN (${code})${R}"
    fi
}

check_frontend "Invidious" "https://vid.puffyan.us"
check_frontend "Nitter" "https://nitter.poast.org"
check_frontend "Redlib" "https://safereddit.com"
check_frontend "SearXNG" "https://search.ononoki.org"
check_frontend "Scribe" "https://scribe.rip"
check_frontend "Wikiless" "https://wikiless.org"
check_frontend "Lingva" "https://lingva.ml"
echo ""

# ─────────────────────────────────────────
# 7. Concurrent Request Performance
# ─────────────────────────────────────────
echo -e "${P}═══ Concurrent Requests (10 parallel) ═══${R}"

concurrent_start=$(date +%s%N)
for i in $(seq 1 10); do
    curl -s -o /dev/null "https://www.cloudflare.com" &
done
wait
concurrent_end=$(date +%s%N)
concurrent_ms=$(( (concurrent_end - concurrent_start) / 1000000 ))
echo -e "  10 parallel HTTPS requests: ${G}${concurrent_ms}ms${R} total"
echo -e "  Average per request: ${G}$((concurrent_ms / 10))ms${R}"
echo ""

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
echo -e "${P}═══════════════════════════════════════${R}"
echo -e "${P}  NebulaBrowser Benchmark Complete${R}"
echo -e "${P}═══════════════════════════════════════${R}"
echo ""
