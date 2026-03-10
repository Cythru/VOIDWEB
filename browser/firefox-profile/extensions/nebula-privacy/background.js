// NebulaBrowser — Background Extension
// Handles: privacy frontend redirects, tracking param stripping,
// header spoofing, censorship circumvention

// =============================================================
// PRIVACY FRONTEND REDIRECTS
// =============================================================

const FRONTEND_REDIRECTS = {
    // YouTube → Invidious
    youtube: {
        enabled: true,
        patterns: [
            /^https?:\/\/(www\.)?youtube\.com/,
            /^https?:\/\/youtu\.be/,
            /^https?:\/\/m\.youtube\.com/,
        ],
        instances: [
            'https://yewtu.be',
            'https://invidious.fdn.fr',
            'https://inv.tux.pizza',
            'https://invidious.protokolla.fi',
        ],
        rewrite: (url, instance) => {
            const u = new URL(url);
            // youtu.be/XXX → instance/watch?v=XXX
            if (u.hostname === 'youtu.be') {
                return `${instance}/watch?v=${u.pathname.slice(1)}`;
            }
            return `${instance}${u.pathname}${u.search}`;
        }
    },

    // Twitter/X → Nitter
    twitter: {
        enabled: true,
        patterns: [
            /^https?:\/\/(www\.)?(twitter|x)\.com/,
            /^https?:\/\/mobile\.(twitter|x)\.com/,
        ],
        instances: [
            'https://nitter.privacydev.net',
            'https://nitter.1d4.us',
            'https://nitter.poast.org',
        ],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}${u.search}`;
        }
    },

    // Reddit → Redlib
    reddit: {
        enabled: true,
        patterns: [
            /^https?:\/\/(www\.|old\.|new\.)?reddit\.com/,
        ],
        instances: [
            'https://safereddit.com',
            'https://redlib.catsarch.com',
            'https://libreddit.kavin.rocks',
        ],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}${u.search}`;
        }
    },

    // Instagram → Bibliogram
    instagram: {
        enabled: true,
        patterns: [/^https?:\/\/(www\.)?instagram\.com/],
        instances: ['https://bibliogram.art'],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}`;
        }
    },

    // Google Search → SearXNG
    google: {
        enabled: true,
        patterns: [/^https?:\/\/(www\.)?google\.com\/search/],
        instances: [
            'https://searx.be',
            'https://search.sapti.me',
            'https://searx.tiekoetter.com',
        ],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}/search${u.search}`;
        }
    },

    // Medium → Scribe
    medium: {
        enabled: true,
        patterns: [/^https?:\/\/(www\.)?medium\.com/],
        instances: ['https://scribe.rip'],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}`;
        }
    },

    // Imgur → Rimgo
    imgur: {
        enabled: true,
        patterns: [/^https?:\/\/(www\.|i\.)?imgur\.com/],
        instances: ['https://rimgo.pussthecat.org'],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}`;
        }
    },

    // TikTok → ProxiTok
    tiktok: {
        enabled: true,
        patterns: [/^https?:\/\/(www\.)?tiktok\.com/],
        instances: ['https://proxitok.pabloferreiro.es'],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}`;
        }
    },

    // Wikipedia → Wikiless
    wikipedia: {
        enabled: true,
        patterns: [/^https?:\/\/(\w+\.)?wikipedia\.org/],
        instances: ['https://wikiless.org'],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}${u.search}`;
        }
    },

    // Google Translate → Lingva
    translate: {
        enabled: true,
        patterns: [/^https?:\/\/translate\.google\.com/],
        instances: ['https://lingva.ml'],
        rewrite: (url, instance) => {
            const u = new URL(url);
            return `${instance}${u.pathname}${u.search}`;
        }
    },
};

// =============================================================
// TRACKING PARAMETER STRIPPING
// =============================================================

const TRACKING_PARAMS = [
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'utm_name', 'utm_cid', 'utm_reader', 'utm_viz_id', 'utm_pubreferrer', 'utm_swu',
    'fbclid', 'gclid', 'gclsrc', 'dclid',
    'msclkid', 'mc_cid', 'mc_eid',
    'yclid', 'twclid', 'ttclid',
    '_ga', '_gl', '_hsenc', '_hsmi',
    'ref', 'ref_', 'ref_src', 'ref_url',
    'igshid', 's_cid',
    'oly_anon_id', 'oly_enc_id',
    'vero_id', 'wickedid',
    '__s', 'ss_source', 'ss_campaign_id',
    'rb_clickid', 's_kwcid',
    'mkt_tok', 'elqTrackId',
    'spm', 'scm', '_openstat',
];

function stripTrackingParams(url) {
    try {
        const u = new URL(url);
        let changed = false;
        for (const param of TRACKING_PARAMS) {
            if (u.searchParams.has(param)) {
                u.searchParams.delete(param);
                changed = true;
            }
        }
        return changed ? u.toString() : null;
    } catch {
        return null;
    }
}

// =============================================================
// REQUEST INTERCEPTION
// =============================================================

// Redirect to privacy frontends
browser.webRequest.onBeforeRequest.addListener(
    (details) => {
        const url = details.url;

        // Strip tracking params
        const stripped = stripTrackingParams(url);
        if (stripped && stripped !== url) {
            return { redirectUrl: stripped };
        }

        // Privacy frontend redirects
        for (const [service, config] of Object.entries(FRONTEND_REDIRECTS)) {
            if (!config.enabled) continue;
            for (const pattern of config.patterns) {
                if (pattern.test(url)) {
                    const instance = config.instances[0]; // Use first instance
                    const redirected = config.rewrite(url, instance);
                    if (redirected && redirected !== url) {
                        console.log(`[nebula] ${service}: ${url} → ${redirected}`);
                        return { redirectUrl: redirected };
                    }
                }
            }
        }

        return {};
    },
    { urls: ['<all_urls>'], types: ['main_frame'] },
    ['blocking']
);

// Spoof headers on all requests
browser.webRequest.onBeforeSendHeaders.addListener(
    (details) => {
        const headers = details.requestHeaders;

        // Remove tracking headers
        const removeHeaders = [
            'x-forwarded-for', 'x-real-ip', 'x-client-ip',
            'cf-connecting-ip', 'via', 'forwarded',
        ];

        const filtered = headers.filter(h =>
            !removeHeaders.includes(h.name.toLowerCase())
        );

        // Set/override headers for Windows 11 Chrome spoof
        const setHeader = (name, value) => {
            const idx = filtered.findIndex(h => h.name.toLowerCase() === name.toLowerCase());
            if (idx >= 0) filtered[idx].value = value;
            else filtered.push({ name, value });
        };

        setHeader('Sec-CH-UA', '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"');
        setHeader('Sec-CH-UA-Mobile', '?0');
        setHeader('Sec-CH-UA-Platform', '"Windows"');
        setHeader('Sec-GPC', '1');
        setHeader('DNT', '1');

        return { requestHeaders: filtered };
    },
    { urls: ['<all_urls>'] },
    ['blocking', 'requestHeaders']
);

// =============================================================
// CENSORSHIP CIRCUMVENTION — Instance Health Checking
// =============================================================

// If a frontend instance is blocked/down, try the next one
async function checkInstanceHealth(url) {
    try {
        const resp = await fetch(url, { method: 'HEAD', mode: 'no-cors' });
        return resp.ok || resp.type === 'opaque';  // opaque = CORS blocked but alive
    } catch {
        return false;
    }
}

// Rotate to healthy instances on failure
browser.webRequest.onErrorOccurred.addListener(
    async (details) => {
        if (details.type !== 'main_frame') return;

        for (const [service, config] of Object.entries(FRONTEND_REDIRECTS)) {
            if (!config.enabled) continue;
            const currentInstance = config.instances[0];
            if (details.url.startsWith(currentInstance)) {
                // Current instance failed — try next
                console.log(`[nebula] ${service} instance down: ${currentInstance}`);
                for (let i = 1; i < config.instances.length; i++) {
                    const alt = config.instances[i];
                    const healthy = await checkInstanceHealth(alt);
                    if (healthy) {
                        console.log(`[nebula] ${service} failover → ${alt}`);
                        // Move healthy instance to front
                        config.instances.splice(i, 1);
                        config.instances.unshift(alt);
                        // Retry navigation
                        const retryUrl = details.url.replace(currentInstance, alt);
                        browser.tabs.update(details.tabId, { url: retryUrl });
                        return;
                    }
                }
                console.log(`[nebula] ${service}: all instances down, loading original`);
            }
        }
    },
    { urls: ['<all_urls>'] }
);

console.log('[NebulaBrowser] Privacy shield loaded — redirects, tracking stripped, headers spoofed');
