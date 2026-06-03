function isAllowedSite() {

    const hostname = location.hostname;

    return WHITELIST.some(domain =>
        hostname === domain ||
        hostname.endsWith("." + domain)
    );
}

function normalizeUrl(url) {

    try {
        return new URL(url).href;
    } catch {
        return null;
    }
}

function isValidImage(url) {

    if (!url) {
        return false;
    }

    if (url.startsWith("data:")) {
        return false;
    }

    const blocked = [
        ".svg",
        "icon",
        "logo"
    ];

    return !blocked.some(v =>
        url.toLowerCase().includes(v)
    );
}