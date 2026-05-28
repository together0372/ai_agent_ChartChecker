console.log("뉴스 차트 탐지기 로드됨");

const observed = new Set();

function isAllowedSite() {
    const hostname = location.hostname;
    return WHITELIST.some(domain =>
        hostname.includes(domain)
    );
}

async function sendToServer(imageUrl) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: {
            url: imageUrl,
            page: location.href,
            site: location.hostname,
            title: document.title
        }
    }, (response) => {
        console.log("분석 결과:", response);
    });
}

function isSvgChart(img, url) {
    return (
        url.includes(".svg") ||
        img.closest("svg")
    );
}

function hasChartContext(img) {

    const container =
        img.closest("figure") ||
        img.parentElement;

    if (!container) return false;

    const text = container.innerText.toLowerCase();

    const keywords = [
        "chart",
        "graph",
        "data",
        "statistics",
        "%",
        "survey",
        "increase",
        "decrease",
        "source"
    ];

    return keywords.some(word =>
        text.includes(word)
    );
}

function isInsideBadContainer(img) {
    return img.closest(
        ".ad, .ads, .advertisement, .banner, .sidebar, " +
        ".recommended, .related, .outbrain, .taboola, " +
        ".promo, .thumbnail, nav, aside, footer, header"
    );
}

function looksLikeAd(url) {
    const lower = url.toLowerCase();

    return [
        "ads",
        "advertisement",
        "banner",
        "promo",
        "sponsor",
        "thumbnail",
        "icon",
        "logo"
    ].some(word => lower.includes(word));
}

function isTooSmall(img) {
    const area = img.naturalWidth * img.naturalHeight;

    return (
        img.naturalWidth < 400 ||
        img.naturalHeight < 250 ||
        area < 180000
    );
}

function isDisplayedTooSmall(img) {
    return (
        img.clientWidth < 250 ||
        img.clientHeight < 150
    );
}

function hasBadAspectRatio(img) {
    const ratio = img.naturalWidth / img.naturalHeight;

    return (
        ratio > 4.5 || ratio < 0.45
    );
}

function collectImages() {

    // Find the actual article body first
    const article =
        document.querySelector("article") ||
        document.querySelector('[role="main"]') ||
        document.querySelector(".article-body");

    if (!article) return;

    // Only search inside article
    const imgs = article.querySelectorAll("img");

    imgs.forEach(img => {

        // Remove ads / sidebars / recommended sections
        if (isInsideBadContainer(img)) return;

        const src =
            img.src ||
            img.dataset.src ||
            img.getAttribute("data-src");

        if (!src) return;

        const normalized = normalizeUrl(src);

        if (!normalized) return;

        if (!hasChartContext(img)) return;

        if (!isValidImage(normalized)) return;

        if (observed.has(normalized)) return;

        if (isInsideBadContainer(img)) return;

        if (!img.complete || img.naturalWidth === 0) return;

        if (isTooSmall(img)) return;

        if (isDisplayedTooSmall(img)) return;

        if (hasBadAspectRatio(img)) return;

        if (looksLikeAd(normalized)) return;

        observed.add(normalized);

        console.log("차트 후보 이미지:", normalized);
        sendToServer(normalized);
    });
}

function main() {
    console.log("isAllowedSite():", isAllowedSite());

    if (!isAllowedSite()) {
        console.log("허용되지 않은 사이트");
        return; // 함수 종료
    }

    console.log("뉴스 차트 탐지기 시작");

    collectImages();

    const observer = new MutationObserver(() => {
    clearTimeout(window.__chartObserverTimeout);

    window.__chartObserverTimeout = setTimeout(() => {
        collectImages();
    }, 300);
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    let scrollTimeout;

    window.addEventListener("scroll", () => {
        clearTimeout(scrollTimeout);

        scrollTimeout = setTimeout(() => {
            collectImages();
        }, 300);
    });
}

// 실행 시작
main();
