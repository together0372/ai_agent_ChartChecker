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
            site: location.hostname
        }
    }, (response) => {
        console.log("분석 결과:", response);
    });
}

function collectImages() {
    const imgs = document.querySelectorAll(
        "article img, .article img, #container img"
    );

    imgs.forEach(img => {
        const src =
            img.src ||
            img.dataset.src ||
            img.getAttribute("data-src");

        if (!src) return;

        const normalized = normalizeUrl(src);
        if (!normalized) return;

        if (!isValidImage(normalized)) return;

        if (observed.has(normalized)) return;

        if (img.naturalWidth < 400) return;
        if (img.naturalHeight < 250) return;

        observed.add(normalized);

        console.log("이미지 발견:", normalized);
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
        collectImages();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    window.addEventListener("scroll", () => {
        collectImages();
    });
}

// 실행 시작
main();
