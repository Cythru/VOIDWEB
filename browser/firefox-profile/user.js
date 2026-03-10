// ╔═══════════════════════════════════════════════════════════════╗
// ║  NebulaBrowser — Hardened Firefox Profile                    ║
// ║  Privacy-first, censorship-proof, anti-fingerprint           ║
// ║  Based on Firefox 148 / Gecko engine                         ║
// ╚═══════════════════════════════════════════════════════════════╝

// =============================================================
// CENSORSHIP RESISTANCE
// =============================================================

// DNS-over-HTTPS (encrypted DNS — ISPs can't see or block queries)
user_pref("network.trr.mode", 3);  // 3 = DoH ONLY, no fallback to plaintext
user_pref("network.trr.uri", "https://mozilla.cloudflare-dns.com/dns-query");
user_pref("network.trr.custom_uri", "https://mozilla.cloudflare-dns.com/dns-query");
user_pref("network.trr.bootstrapAddr", "1.1.1.1");
// Backup DoH resolvers
user_pref("network.trr.resolvers", '[{"url":"https://mozilla.cloudflare-dns.com/dns-query"},{"url":"https://dns.quad9.net/dns-query"},{"url":"https://dns.google/dns-query"}]');

// Encrypted Client Hello (ECH) — hides which site you're visiting from ISP
user_pref("network.dns.echconfig.enabled", true);
user_pref("network.dns.use_https_rr_as_altsvc", true);

// ESNI (predecessor to ECH, still useful)
user_pref("network.security.esni.enabled", true);

// Disable OCSP (certificate status leaks which sites you visit)
user_pref("security.OCSP.enabled", 0);
user_pref("security.OCSP.require", false);

// Disable safe browsing (phones home to Google with every URL)
user_pref("browser.safebrowsing.malware.enabled", false);
user_pref("browser.safebrowsing.phishing.enabled", false);
user_pref("browser.safebrowsing.downloads.enabled", false);
user_pref("browser.safebrowsing.downloads.remote.enabled", false);
user_pref("browser.safebrowsing.downloads.remote.url", "");
user_pref("browser.safebrowsing.provider.google4.dataSharing.enabled", false);
user_pref("browser.safebrowsing.provider.google4.updateURL", "");
user_pref("browser.safebrowsing.provider.google4.reportURL", "");
user_pref("browser.safebrowsing.provider.google.updateURL", "");
user_pref("browser.safebrowsing.provider.google.reportURL", "");

// HTTPS-Only mode (block all plaintext HTTP — prevents injection/sniffing)
user_pref("dom.security.https_only_mode", true);
user_pref("dom.security.https_only_mode_ever_enabled", true);
user_pref("dom.security.https_only_mode_send_http_background_request", false);

// Disable captive portal detection (leaks network info)
user_pref("network.captive-portal-service.enabled", false);
user_pref("captivedetect.canonicalURL", "");

// Disable connectivity checks
user_pref("network.connectivity-service.enabled", false);

// =============================================================
// TOR / PROXY SUPPORT
// =============================================================

// SOCKS5 proxy (for Tor — configure to 127.0.0.1:9150 when Tor is active)
// Uncomment when using Tor:
// user_pref("network.proxy.type", 1);
// user_pref("network.proxy.socks", "127.0.0.1");
// user_pref("network.proxy.socks_port", 9150);
// user_pref("network.proxy.socks_version", 5);
// user_pref("network.proxy.socks_remote_dns", true);  // DNS through Tor too
// user_pref("network.proxy.no_proxies_on", "");  // Proxy everything

// Disable WebRTC (leaks real IP even through VPN/Tor)
user_pref("media.peerconnection.enabled", false);
user_pref("media.peerconnection.ice.default_address_only", true);
user_pref("media.peerconnection.ice.no_host", true);
user_pref("media.peerconnection.ice.proxy_only_if_behind_proxy", true);
user_pref("media.peerconnection.turn.disable", true);

// =============================================================
// ANTI-FINGERPRINTING — Windows 11 Spoof
// =============================================================

// Firefox's built-in fingerprint resistance (Tor Browser level)
user_pref("privacy.resistFingerprinting", true);
user_pref("privacy.resistFingerprinting.letterboxing", false);  // Disable letterboxing (annoying)

// Spoof OS to Windows
user_pref("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0");
user_pref("general.platform.override", "Win32");
user_pref("general.oscpu.override", "Windows NT 10.0; Win64; x64");
user_pref("general.appversion.override", "5.0 (Windows)");
user_pref("general.buildID.override", "20181001000000");

// Spoof timezone to US Eastern
user_pref("privacy.resistFingerprinting.spoofOsLocale", "en-US");

// Spoof screen to 1920x1080 @ 60Hz
user_pref("privacy.resistFingerprinting.target_video_res", 1080);

// Canvas fingerprint protection
user_pref("privacy.resistFingerprinting.autoDeclineNoUserInputCanvasPrompts", true);

// Font fingerprint — only expose common fonts
user_pref("browser.display.use_document_fonts", 1);
user_pref("layout.css.font-visibility.resistFingerprinting", 1);

// WebGL fingerprint protection
user_pref("webgl.disabled", false);  // Keep WebGL but spoof renderer
user_pref("webgl.renderer-string-override", "ANGLE (Intel, Intel(R) UHD Graphics 730 Direct3D11 vs_5_0 ps_5_0, D3D11)");
user_pref("webgl.vendor-string-override", "Google Inc. (Intel)");

// Disable battery API (fingerprinting vector)
user_pref("dom.battery.enabled", false);

// Disable gamepad API (fingerprinting vector)
user_pref("dom.gamepad.enabled", false);

// Disable sensor APIs (fingerprinting)
user_pref("device.sensors.enabled", false);

// Disable speech synthesis (fingerprinting via available voices)
user_pref("media.webspeech.synth.enabled", false);

// =============================================================
// TRACKING / AD BLOCKING
// =============================================================

// Enhanced Tracking Protection — strict mode
user_pref("browser.contentblocking.category", "strict");
user_pref("privacy.trackingprotection.enabled", true);
user_pref("privacy.trackingprotection.socialtracking.enabled", true);
user_pref("privacy.trackingprotection.cryptomining.enabled", true);
user_pref("privacy.trackingprotection.fingerprinting.enabled", true);

// Block third-party cookies
user_pref("network.cookie.cookieBehavior", 5);  // 5 = dFPI (dynamic First Party Isolation)
user_pref("privacy.partition.always_partition_third_party_non_cookie_storage", true);
user_pref("privacy.partition.always_partition_third_party_non_cookie_storage.exempt_sessionstorage", false);

// First-Party Isolation (Tor Browser style)
user_pref("privacy.firstparty.isolate", true);

// Strip tracking params from URLs
user_pref("privacy.query_stripping.enabled", true);
user_pref("privacy.query_stripping.enabled.pbmode", true);
user_pref("privacy.query_stripping.strip_list", "utm_source utm_medium utm_campaign utm_term utm_content utm_name utm_cid utm_reader utm_viz_id utm_pubreferrer utm_swu fbclid gclid gclsrc dclid msclkid mc_cid mc_eid yclid _openstat twclid ttclid igshid ref_src ref_url");

// Disable prefetching (leaks browsing intent)
user_pref("network.prefetch-next", false);
user_pref("network.dns.disablePrefetch", true);
user_pref("network.dns.disablePrefetchFromHTTPS", true);
user_pref("network.predictor.enabled", false);
user_pref("network.predictor.enable-prefetch", false);
user_pref("network.http.speculative-parallel-limit", 0);

// Disable hyperlink auditing (ping tracking)
user_pref("browser.send_pings", false);
user_pref("browser.send_pings.require_same_host", true);

// =============================================================
// TELEMETRY — KILL IT ALL
// =============================================================

user_pref("toolkit.telemetry.unified", false);
user_pref("toolkit.telemetry.enabled", false);
user_pref("toolkit.telemetry.server", "data:,");
user_pref("toolkit.telemetry.archive.enabled", false);
user_pref("toolkit.telemetry.newProfilePing.enabled", false);
user_pref("toolkit.telemetry.shutdownPingSender.enabled", false);
user_pref("toolkit.telemetry.updatePing.enabled", false);
user_pref("toolkit.telemetry.bhrPing.enabled", false);
user_pref("toolkit.telemetry.firstShutdownPing.enabled", false);
user_pref("toolkit.telemetry.coverage.opt-out", true);
user_pref("toolkit.coverage.opt-out", true);
user_pref("toolkit.coverage.endpoint.base", "");
user_pref("datareporting.healthreport.uploadEnabled", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("app.shield.optoutstudies.enabled", false);
user_pref("app.normandy.enabled", false);
user_pref("app.normandy.api_url", "");
user_pref("breakpad.reportURL", "");
user_pref("browser.tabs.crashReporting.sendReport", false);
user_pref("browser.crashReports.unsubmittedCheck.autoSubmit2", false);
user_pref("browser.crashReports.unsubmittedCheck.enabled", false);

// Disable experiments
user_pref("messaging-system.rsexperimentloader.enabled", false);

// Disable Pocket
user_pref("extensions.pocket.enabled", false);
user_pref("extensions.pocket.api", "");
user_pref("extensions.pocket.oAuthConsumerKey", "");
user_pref("extensions.pocket.site", "");

// Disable Firefox accounts
user_pref("identity.fxaccounts.enabled", false);

// Disable Discovery/recommendations
user_pref("browser.discovery.enabled", false);
user_pref("browser.newtabpage.activity-stream.feeds.topsites", false);
user_pref("browser.newtabpage.activity-stream.feeds.section.topstories", false);
user_pref("browser.newtabpage.activity-stream.showSponsored", false);
user_pref("browser.newtabpage.activity-stream.showSponsoredTopSites", false);
user_pref("browser.newtabpage.activity-stream.asrouter.userprefs.cfr.addons", false);
user_pref("browser.newtabpage.activity-stream.asrouter.userprefs.cfr.features", false);

// =============================================================
// SECURITY HARDENING
// =============================================================

// Disable DRM (Widevine CDM) — optional, enable if you need Netflix
user_pref("media.eme.enabled", false);
user_pref("media.gmp-widevinecdm.enabled", false);

// Disable WebAssembly JIT (potential exploit vector — enable if needed for speed)
// user_pref("javascript.options.wasm", false);

// Enable process isolation
user_pref("fission.autostart", true);

// Strict SSL/TLS
user_pref("security.ssl.require_safe_negotiation", true);
user_pref("security.tls.version.min", 3);  // TLS 1.2 minimum
user_pref("security.tls.enable_0rtt_data", false);
user_pref("security.ssl.treat_unsafe_negotiation_as_broken", true);
user_pref("browser.xul.error_pages.expert_bad_cert", true);
user_pref("security.cert_pinning.enforcement_level", 2);  // Strict pinning

// Disable dangerous features
user_pref("dom.disable_beforeunload", true);
user_pref("dom.disable_open_during_load", true);
user_pref("dom.popup_allowed_events", "click dblclick mousedown pointerdown");

// Disable service workers (tracking/persistence mechanism)
// user_pref("dom.serviceWorkers.enabled", false);  // Breaks some sites

// Disable push notifications
user_pref("dom.push.enabled", false);
user_pref("dom.push.connection.enabled", false);

// Disable Web Notifications
user_pref("dom.webnotifications.enabled", false);

// =============================================================
// CENSORSHIP-PROOF DNS FALLBACKS
// =============================================================

// If primary DoH fails, try alternatives
// These are hardcoded so ISP DNS blocking can't prevent resolution
user_pref("network.trr.excluded-domains", "");  // Don't exclude anything from DoH
user_pref("network.trr.builtin-excluded-domains", "");

// Disable GeoIP lookups (used for censorship targeting)
user_pref("browser.search.geoip.url", "");
user_pref("browser.search.region", "US");
user_pref("browser.search.geoSpecificDefaults", false);

// =============================================================
// UI / UX
// =============================================================

// Dark mode
user_pref("ui.systemUsesDarkTheme", 1);
user_pref("browser.in-content.dark-mode", true);
user_pref("devtools.theme", "dark");
user_pref("layout.css.prefers-color-scheme.content-override", 0);  // Dark

// Compact toolbar
user_pref("browser.compactmode.show", true);
user_pref("browser.uidensity", 1);  // Compact

// Enable userChrome.css (REQUIRED for NebulaBrowser theme)
user_pref("toolkit.legacyUserProfileCustomizations.stylesheets", true);

// Homepage — NebulaBrowser custom new tab
user_pref("browser.startup.homepage", "file://__PROFILE_DIR__/newtab.html");
user_pref("browser.startup.page", 1);  // Homepage
user_pref("browser.newtabpage.enabled", false);
user_pref("browser.newtab.url", "file://__PROFILE_DIR__/newtab.html");

// Disable default Firefox new tab
user_pref("browser.newtabpage.activity-stream.enabled", false);
user_pref("browser.newtabpage.activity-stream.feeds.topsites", false);
user_pref("browser.newtabpage.activity-stream.feeds.section.highlights", false);
user_pref("browser.newtabpage.activity-stream.feeds.snippets", false);

// Disable annoying defaults
user_pref("browser.aboutConfig.showWarning", false);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("browser.warnOnQuit", false);
user_pref("general.warnOnAboutConfig", false);

// Disable default bookmarks/import
user_pref("browser.bookmarks.restore_default_bookmarks", false);
user_pref("browser.places.importBookmarksHTML", false);

// =============================================================
// PERFORMANCE
// =============================================================

// HTTP/3 QUIC support
user_pref("network.http.http3.enabled", true);

// Faster TLS
user_pref("network.ssl_tokens_cache_capacity", 32768);

// Session restore off (privacy + speed)
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.sessionstore.max_tabs_undo", 0);
user_pref("browser.sessionstore.max_windows_undo", 0);

// Disk cache off (RAM only — faster + no forensic traces)
user_pref("browser.cache.disk.enable", false);
user_pref("browser.cache.memory.enable", true);
user_pref("browser.cache.memory.capacity", 524288);  // 512MB RAM cache

// GPU acceleration
user_pref("layers.acceleration.force-enabled", true);
user_pref("gfx.webrender.all", true);
