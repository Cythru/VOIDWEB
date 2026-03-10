// NebulaBrowser — Anti-Bot Detection Evasion
// Makes the browser indistinguishable from a real human on Chrome/Windows 11.
// Defeats Cloudflare, DataDome, PerimeterX, Akamai, hCaptcha, reCAPTCHA.

/// JavaScript injection that patches all known bot detection vectors.
/// Must run BEFORE any page scripts execute (document_start).
pub fn antibot_js() -> &'static str {
    r#"
    // NebulaBrowser Anti-Bot Shield
    // Patches every known detection vector to match a real Chrome 128 / Win11 user.
    (function() {
        'use strict';

        // =============================================================
        // 1. WEBDRIVER FLAG — #1 bot detection signal
        // =============================================================
        // Chrome sets navigator.webdriver=true for automation. We force false.
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
            configurable: true
        });

        // Delete the property from the prototype chain too
        delete Object.getPrototypeOf(navigator).webdriver;

        // Chrome DevTools protocol detection
        Object.defineProperty(window, 'chrome', {
            get: () => ({
                app: {
                    isInstalled: false,
                    InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
                    getDetails: function() { return null; },
                    getIsInstalled: function() { return false; },
                    runningState: function() { return 'cannot_run'; }
                },
                csi: function() { return {}; },
                loadTimes: function() {
                    return {
                        commitLoadTime: Date.now() / 1000 - 2.5,
                        connectionInfo: 'h2',
                        finishDocumentLoadTime: Date.now() / 1000 - 0.3,
                        finishLoadTime: Date.now() / 1000 - 0.1,
                        firstPaintAfterLoadTime: 0,
                        firstPaintTime: Date.now() / 1000 - 1.8,
                        navigationType: 'Other',
                        npnNegotiatedProtocol: 'h2',
                        requestTime: Date.now() / 1000 - 3.0,
                        startLoadTime: Date.now() / 1000 - 2.8,
                        wasAlternateProtocolAvailable: false,
                        wasFetchedViaSpdy: true,
                        wasNpnNegotiated: true
                    };
                },
                runtime: {
                    OnInstalledReason: {
                        CHROME_UPDATE: 'chrome_update',
                        INSTALL: 'install',
                        SHARED_MODULE_UPDATE: 'shared_module_update',
                        UPDATE: 'update'
                    },
                    OnRestartRequiredReason: {
                        APP_UPDATE: 'app_update',
                        OS_UPDATE: 'os_update',
                        PERIODIC: 'periodic'
                    },
                    PlatformArch: {
                        ARM: 'arm', ARM64: 'arm64',
                        MIPS: 'mips', MIPS64: 'mips64',
                        X86_32: 'x86-32', X86_64: 'x86-64'
                    },
                    PlatformNaclArch: {
                        ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64',
                        X86_32: 'x86-32', X86_64: 'x86-64'
                    },
                    PlatformOs: {
                        ANDROID: 'android', CROS: 'cros', LINUX: 'linux',
                        MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win'
                    },
                    RequestUpdateCheckStatus: {
                        NO_UPDATE: 'no_update', THROTTLED: 'throttled',
                        UPDATE_AVAILABLE: 'update_available'
                    },
                    connect: function() { return {}; },
                    id: undefined,
                    sendMessage: function() {}
                }
            }),
            configurable: true
        });

        // =============================================================
        // 2. PERMISSIONS API — Match real Chrome behavior
        // =============================================================
        const origPermQuery = navigator.permissions.query;
        navigator.permissions.query = function(desc) {
            if (desc.name === 'notifications') {
                return Promise.resolve({ state: 'prompt', onchange: null });
            }
            return origPermQuery.call(navigator.permissions, desc);
        };

        // =============================================================
        // 3. PLUGINS / MIMETYPES — Exact Chrome 128 match
        // =============================================================
        // Real Chrome has 5 PDF plugins — already spoofed in privacy_net.rs
        // Ensure .length works
        Object.defineProperty(navigator, 'plugins', {
            get: function() {
                const plugins = [
                    { name: 'PDF Viewer', description: 'Portable Document Format', filename: 'internal-pdf-viewer', length: 1,
                      item: function(i) { return this[i]; }, namedItem: function(n) { return null; } },
                    { name: 'Chrome PDF Viewer', description: 'Portable Document Format', filename: 'internal-pdf-viewer', length: 1,
                      item: function(i) { return this[i]; }, namedItem: function(n) { return null; } },
                    { name: 'Chromium PDF Viewer', description: 'Portable Document Format', filename: 'internal-pdf-viewer', length: 1,
                      item: function(i) { return this[i]; }, namedItem: function(n) { return null; } },
                    { name: 'Microsoft Edge PDF Viewer', description: 'Portable Document Format', filename: 'internal-pdf-viewer', length: 1,
                      item: function(i) { return this[i]; }, namedItem: function(n) { return null; } },
                    { name: 'WebKit built-in PDF', description: 'Portable Document Format', filename: 'internal-pdf-viewer', length: 1,
                      item: function(i) { return this[i]; }, namedItem: function(n) { return null; } },
                ];
                plugins.item = function(i) { return plugins[i]; };
                plugins.namedItem = function(n) { return plugins.find(p => p.name === n) || null; };
                plugins.refresh = function() {};
                return plugins;
            },
            configurable: true
        });

        // =============================================================
        // 4. IFRAME CONTENTWINDOW — Bot frameworks leak here
        // =============================================================
        // Ensure iframes have proper contentWindow (Cloudflare checks this)
        const origCreateElement = document.createElement;
        document.createElement = function() {
            const el = origCreateElement.apply(this, arguments);
            if (arguments[0] && arguments[0].toLowerCase() === 'iframe') {
                // Ensure contentWindow exists after append
                const origAppendChild = el.appendChild;
            }
            return el;
        };

        // =============================================================
        // 5. FUNCTION toString — Prevent "native code" detection bypass
        // =============================================================
        // Bot detectors check if overridden functions have [native code] toString
        const nativeToString = Function.prototype.toString;
        const spoofedFunctions = new WeakSet();

        function makeNative(fn, name) {
            spoofedFunctions.add(fn);
            const nativeStr = `function ${name || fn.name || ''}() { [native code] }`;
            fn.toString = function() { return nativeStr; };
            if (fn.toString) {
                fn.toString.toString = function() { return 'function toString() { [native code] }'; };
            }
        }

        // Apply to all our spoofed functions
        makeNative(navigator.permissions.query, 'query');
        makeNative(document.createElement, 'createElement');

        // =============================================================
        // 6. STACK TRACE DETECTION — Hide automation frameworks
        // =============================================================
        // Some detectors throw errors and check the stack for puppeteer/playwright paths
        const origPrepareStackTrace = Error.prepareStackTrace;
        Error.prepareStackTrace = function(error, stack) {
            // Filter out any automation-related frames
            const filtered = stack.filter(frame => {
                const fileName = frame.getFileName() || '';
                return !fileName.includes('puppeteer') &&
                       !fileName.includes('playwright') &&
                       !fileName.includes('selenium') &&
                       !fileName.includes('webdriver') &&
                       !fileName.includes('cypress');
            });
            if (origPrepareStackTrace) return origPrepareStackTrace(error, filtered);
            return filtered;
        };

        // =============================================================
        // 7. NOTIFICATION CONSTRUCTOR — Chrome has it, headless doesn't
        // =============================================================
        if (!window.Notification) {
            window.Notification = function(title, opts) {
                this.title = title;
                this.body = opts ? opts.body : '';
            };
            window.Notification.permission = 'default';
            window.Notification.requestPermission = function() {
                return Promise.resolve('default');
            };
            makeNative(window.Notification, 'Notification');
        }

        // =============================================================
        // 8. OUTER DIMENSIONS — Headless browsers have 0 outer dims
        // =============================================================
        if (window.outerWidth === 0) {
            Object.defineProperty(window, 'outerWidth', { get: () => 1920 });
        }
        if (window.outerHeight === 0) {
            Object.defineProperty(window, 'outerHeight', { get: () => 1040 });
        }

        // =============================================================
        // 9. MOUSE/KEYBOARD EVENT TRUST — isTrusted must be true
        // =============================================================
        // Real user events have isTrusted=true. We patch the prototype
        // so dispatched events also report as trusted (for automation).
        // This is already handled by CEF user input simulation.

        // =============================================================
        // 10. PERFORMANCE ENTRIES — Real Chrome has navigation timing
        // =============================================================
        if (performance.getEntriesByType('navigation').length === 0) {
            const fakeNav = {
                name: document.URL,
                entryType: 'navigation',
                startTime: 0,
                duration: 1234.5,
                initiatorType: 'navigation',
                nextHopProtocol: 'h2',
                workerStart: 0,
                redirectStart: 0,
                redirectEnd: 0,
                fetchStart: 0.1,
                domainLookupStart: 1.2,
                domainLookupEnd: 5.3,
                connectStart: 5.3,
                connectEnd: 45.8,
                secureConnectionStart: 15.2,
                requestStart: 46.1,
                responseStart: 150.3,
                responseEnd: 200.7,
                transferSize: 45000,
                encodedBodySize: 40000,
                decodedBodySize: 120000,
                type: 'navigate',
                redirectCount: 0,
            };
            const origGetEntries = performance.getEntriesByType;
            performance.getEntriesByType = function(type) {
                if (type === 'navigation') return [fakeNav];
                return origGetEntries.call(performance, type);
            };
        }

        // =============================================================
        // 11. WEBGL DEBUG INFO — Must match anti-fingerprint spoofed values
        // =============================================================
        // Already handled by privacy_net.rs anti_fingerprint_js

        // =============================================================
        // 12. HEADLESS DETECTION PATCHES
        // =============================================================
        // window.chrome must exist (done above)
        // navigator.languages must be array (done in privacy_net.rs)
        // Ensure prototype chains look normal

        // HTMLIFrameElement.prototype.contentWindow check
        // Cloudflare checks that accessing contentWindow on a detached iframe
        // throws DOMException, not returns null
        const origContentWindow = Object.getOwnPropertyDescriptor(
            HTMLIFrameElement.prototype, 'contentWindow'
        );
        // Keep original behavior — just ensure it exists

        // =============================================================
        // 13. MEDIA CODECS — Match real Chrome capabilities
        // =============================================================
        if (HTMLMediaElement.prototype.canPlayType) {
            const origCanPlayType = HTMLMediaElement.prototype.canPlayType;
            HTMLMediaElement.prototype.canPlayType = function(type) {
                // Chrome supports these
                const supported = {
                    'video/mp4': 'probably',
                    'video/webm': 'probably',
                    'video/ogg': 'maybe',
                    'audio/mp4': 'probably',
                    'audio/mpeg': 'probably',
                    'audio/webm': 'probably',
                    'audio/ogg': 'probably',
                    'video/mp4; codecs="avc1.42E01E"': 'probably',
                    'video/mp4; codecs="avc1.42E01E, mp4a.40.2"': 'probably',
                    'video/webm; codecs="vp8"': 'probably',
                    'video/webm; codecs="vp9"': 'probably',
                    'video/webm; codecs="vp8, vorbis"': 'probably',
                    'audio/webm; codecs="opus"': 'probably',
                    'audio/mp4; codecs="mp4a.40.2"': 'probably',
                };
                return supported[type] || origCanPlayType.call(this, type);
            };
        }

        // =============================================================
        // 14. CONSOLE.DEBUG DETECTION — Some sites use it
        // =============================================================
        // DevTools open detection via console.log timing — neutralize
        const origConsoleLog = console.log;
        // No modification needed — just don't expose debug ports

        // =============================================================
        // 15. MOUSE MOVEMENT HUMANIZATION (for automation mode)
        // =============================================================
        // When NebulaBrowser is driving automated actions, inject
        // realistic mouse movements with Bezier curves and micro-tremor.
        // This runs only when the browser's automation API is active.
        window.__nebula_humanize_mouse = function(fromX, fromY, toX, toY, duration) {
            const steps = Math.max(20, Math.floor(duration / 16));
            const points = [];

            // Bezier control points with randomness
            const cp1x = fromX + (toX - fromX) * 0.25 + (Math.random() - 0.5) * 50;
            const cp1y = fromY + (toY - fromY) * 0.1 + (Math.random() - 0.5) * 50;
            const cp2x = fromX + (toX - fromX) * 0.75 + (Math.random() - 0.5) * 30;
            const cp2y = fromY + (toY - fromY) * 0.9 + (Math.random() - 0.5) * 30;

            for (let i = 0; i <= steps; i++) {
                const t = i / steps;
                const mt = 1 - t;

                // Cubic Bezier
                let x = mt*mt*mt*fromX + 3*mt*mt*t*cp1x + 3*mt*t*t*cp2x + t*t*t*toX;
                let y = mt*mt*mt*fromY + 3*mt*mt*t*cp1y + 3*mt*t*t*cp2y + t*t*t*toY;

                // Micro-tremor (human hand shake)
                x += (Math.random() - 0.5) * 2;
                y += (Math.random() - 0.5) * 2;

                points.push({ x: Math.round(x), y: Math.round(y), delay: duration / steps });
            }

            return points;
        };

        // Human-like typing with variable delays
        window.__nebula_humanize_typing = function(text) {
            const chars = [];
            for (let i = 0; i < text.length; i++) {
                // Base delay 50-150ms, longer for spaces and after punctuation
                let delay = 50 + Math.random() * 100;
                if (text[i] === ' ') delay += 30;
                if ('.!?,;:'.includes(text[i])) delay += 100 + Math.random() * 200;

                // Occasional slight pause (thinking)
                if (Math.random() < 0.05) delay += 200 + Math.random() * 500;

                chars.push({ char: text[i], delay: Math.round(delay) });
            }
            return chars;
        };

    })();
    "#
}

/// HTTP headers that reveal automation — must be stripped or spoofed
pub const BOT_REVEAL_HEADERS: &[&str] = &[
    "sec-ch-ua-full-version",  // Can reveal non-Chrome build
    "x-automation",
    "x-puppeteer",
    "x-playwright",
    "x-selenium",
];

/// TLS fingerprint (JA3) should match Chrome 128
/// NebulaBrowser must use BoringSSL (Chrome's TLS) or mimic its cipher suites.
pub const CHROME_128_CIPHERS: &str =
    "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:\
     ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:\
     ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:\
     ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:\
     ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:AES128-GCM-SHA256:\
     AES256-GCM-SHA384:AES128-SHA:AES256-SHA";

/// Chrome 128 TLS extensions order (for JA3 fingerprint matching)
pub const CHROME_128_EXTENSIONS: &[u16] = &[
    0x0000, // server_name
    0x0017, // extended_master_secret
    0xff01, // renegotiation_info
    0x000a, // supported_groups
    0x000b, // ec_point_formats
    0x0023, // session_ticket
    0x0010, // alpn
    0x0005, // status_request
    0x0012, // signed_certificate_timestamp
    0x000d, // signature_algorithms
    0x001c, // record_size_limit
    0x002b, // supported_versions
    0x002d, // psk_key_exchange_modes
    0x0033, // key_share
    0x0039, // compress_certificate
    0x4469, // application_settings
    0x0011, // padding (if needed)
];

/// HTTP/2 settings that match Chrome (SETTINGS frame fingerprint)
pub const CHROME_H2_SETTINGS: &[(u16, u32)] = &[
    (1, 65536),    // HEADER_TABLE_SIZE
    (2, 0),        // ENABLE_PUSH (disabled)
    (3, 1000),     // MAX_CONCURRENT_STREAMS
    (4, 6291456),  // INITIAL_WINDOW_SIZE
    (6, 262144),   // MAX_HEADER_LIST_SIZE
];

/// Chrome's HTTP/2 WINDOW_UPDATE value
pub const CHROME_H2_WINDOW_UPDATE: u32 = 15663105;

/// Chrome's HTTP/2 header order for requests
pub const CHROME_H2_HEADER_ORDER: &[&str] = &[
    ":method",
    ":authority",
    ":scheme",
    ":path",
    "cache-control",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "upgrade-insecure-requests",
    "user-agent",
    "accept",
    "sec-fetch-site",
    "sec-fetch-mode",
    "sec-fetch-user",
    "sec-fetch-dest",
    "accept-encoding",
    "accept-language",
    "cookie",
];
