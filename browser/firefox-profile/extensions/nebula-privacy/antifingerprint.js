// NebulaBrowser — Content Script Anti-Fingerprinting
// Injected into every page at document_start.
// Spoofs all info to match a generic Windows 11 Chrome 128 desktop.

(function() {
    'use strict';

    function spoof(obj, prop, value) {
        try {
            Object.defineProperty(obj, prop, {
                get: () => value, configurable: true, enumerable: true
            });
        } catch(e) {}
    }

    // Navigator — Windows 11 desktop
    spoof(navigator, 'platform', 'Win32');
    spoof(navigator, 'hardwareConcurrency', 8);
    spoof(navigator, 'deviceMemory', 16);
    spoof(navigator, 'maxTouchPoints', 0);
    spoof(navigator, 'webdriver', false);
    spoof(navigator, 'vendor', 'Google Inc.');
    spoof(navigator, 'languages', Object.freeze(['en-US', 'en']));

    // Screen — 1080p 60Hz
    spoof(screen, 'width', 1920);
    spoof(screen, 'height', 1080);
    spoof(screen, 'availWidth', 1920);
    spoof(screen, 'availHeight', 1040);
    spoof(screen, 'colorDepth', 24);
    spoof(screen, 'pixelDepth', 24);
    spoof(window, 'devicePixelRatio', 1);
    spoof(window, 'outerWidth', 1920);
    spoof(window, 'outerHeight', 1040);

    // Timezone — US Eastern
    Date.prototype.getTimezoneOffset = function() { return 300; };
    const origResolvedOptions = Intl.DateTimeFormat.prototype.resolvedOptions;
    Intl.DateTimeFormat.prototype.resolvedOptions = function() {
        const o = origResolvedOptions.call(this);
        o.timeZone = 'America/New_York';
        o.locale = 'en-US';
        return o;
    };

    // Geolocation — blocked
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition = (s, e) => { if(e) e({code:1, message:'denied'}); };
        navigator.geolocation.watchPosition = (s, e) => { if(e) e({code:1, message:'denied'}); return 0; };
    }

    // WebGL — Intel UHD 730
    function spoofGL(proto) {
        const orig = proto.getParameter;
        proto.getParameter = function(p) {
            if (p === 37445) return 'Google Inc. (Intel)';
            if (p === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics 730 Direct3D11 vs_5_0 ps_5_0, D3D11)';
            return orig.call(this, p);
        };
        const origPrecision = proto.getShaderPrecisionFormat;
        proto.getShaderPrecisionFormat = function() {
            return { rangeMin: 127, rangeMax: 127, precision: 23 };
        };
    }
    if (window.WebGLRenderingContext) spoofGL(WebGLRenderingContext.prototype);
    if (window.WebGL2RenderingContext) spoofGL(WebGL2RenderingContext.prototype);

    // Canvas — seeded noise
    let seed = Math.floor(Math.random() * 0xFFFFFFFF);
    function noise(i) { let x=(i+seed)*2654435761; x=((x>>16)^x)*0x45d9f3b; return (x&3)-1; }

    const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function() {
        try {
            const ctx = this.getContext('2d');
            if (ctx && this.width > 0 && this.height > 0) {
                const d = ctx.getImageData(0,0,this.width,this.height);
                for (let i=0; i<d.data.length; i+=4) {
                    d.data[i] = Math.max(0,Math.min(255, d.data[i]+noise(i)));
                    d.data[i+1] = Math.max(0,Math.min(255, d.data[i+1]+noise(i+1)));
                }
                ctx.putImageData(d,0,0);
            }
        } catch(e) {}
        return origToDataURL.apply(this, arguments);
    };

    // Audio — perturbation
    if (window.OfflineAudioContext) {
        const origRender = OfflineAudioContext.prototype.startRendering;
        OfflineAudioContext.prototype.startRendering = function() {
            return origRender.call(this).then(buf => {
                const d = buf.getChannelData(0);
                for (let i=0; i<d.length; i+=100) d[i] += noise(i)*0.00001;
                return buf;
            });
        };
    }

    // WebRTC — blocked
    if (window.RTCPeerConnection) {
        const Fake = function() {
            return { createDataChannel:()=>({}), createOffer:()=>Promise.resolve({}),
                     setLocalDescription:()=>Promise.resolve(), close:()=>{},
                     addEventListener:()=>{}, removeEventListener:()=>{},
                     onicecandidate:null, localDescription:null };
        };
        Fake.generateCertificate = () => Promise.resolve({});
        window.RTCPeerConnection = Fake;
        window.webkitRTCPeerConnection = Fake;
    }

    // Battery — desktop (always charging)
    if (navigator.getBattery) {
        navigator.getBattery = () => Promise.resolve({
            charging:true, chargingTime:0, dischargingTime:Infinity, level:1.0,
            addEventListener:()=>{}, removeEventListener:()=>{}
        });
    }

    // Performance — clamp timing precision
    const origNow = performance.now;
    performance.now = function() { return Math.round(origNow.call(performance)*10)/10; };

    // Storage estimation — generic 500GB
    if (navigator.storage && navigator.storage.estimate) {
        navigator.storage.estimate = () => Promise.resolve({quota:268435456000, usage:52428800});
    }

    // Media devices — audio only, no camera
    if (navigator.mediaDevices) {
        navigator.mediaDevices.enumerateDevices = () => Promise.resolve([
            {deviceId:'',kind:'audioinput',label:'',groupId:'default'},
            {deviceId:'',kind:'audiooutput',label:'',groupId:'default'},
        ]);
    }

    // Client Hints API
    if (navigator.userAgentData) {
        spoof(navigator, 'userAgentData', Object.freeze({
            brands: Object.freeze([
                {brand:'Chromium',version:'128'}, {brand:'Not;A=Brand',version:'24'},
                {brand:'Google Chrome',version:'128'}
            ]),
            mobile: false, platform: 'Windows',
            getHighEntropyValues: () => Promise.resolve({
                architecture:'x86', bitness:'64', mobile:false, model:'',
                platform:'Windows', platformVersion:'15.0.0',
                uaFullVersion:'128.0.6613.120', wow64:false
            }),
            toJSON: function() { return {brands:this.brands, mobile:false, platform:'Windows'}; }
        }));
    }

    // matchMedia — 60Hz, light mode (Windows 11 default)
    const origMM = window.matchMedia;
    window.matchMedia = function(q) {
        if (q==='(prefers-color-scheme: dark)') return {matches:false, media:q, addEventListener:()=>{}, removeEventListener:()=>{}};
        if (q==='(prefers-color-scheme: light)') return {matches:true, media:q, addEventListener:()=>{}, removeEventListener:()=>{}};
        if (q==='(prefers-reduced-motion: reduce)') return {matches:false, media:q, addEventListener:()=>{}, removeEventListener:()=>{}};
        return origMM.call(window, q);
    };

})();
