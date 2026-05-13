function normalizeUrl(url) {
    try {
        return new URL(url).href;
    } catch {
        return null;
    }
}

function isValidImage(url) {
    if (!url) return false;

    if (url.startsWith("data:")) {
        return false;
    }

    const blocked = [
        ".svg"
    ];

    for (const item of blocked) {
        if (url.includes(item)) {
            return false;
        }
    }

    return true;
}