console.log("Auto Image Saver loaded");

let currentUrl = location.href;
let observedImages = new Set();

function isAllowedSite() {
    const hostname = location.hostname;
    return WHITELIST.some(domain =>
        hostname.includes(domain)
    );
}

function collectImages() {
    const imgs = document.querySelectorAll("img");
    console.log("이미지 개수:", imgs.length);

    imgs.forEach((img) => {
        const src =
            img.src ||
            img.dataset.src ||
            img.dataset.lazySrc ||
            img.getAttribute("data-src");

        if (!src) return;

        const normalized = normalizeUrl(src);
        if (!normalized) return;

        if (!isValidImage(normalized)) return;

        if (observedImages.has(normalized)) return;

        const width = img.naturalWidth;
        const height = img.naturalHeight;
        if (width < 200 || height < 200) return;

        observedImages.add(normalized);

        console.log("다운로드 요청:", normalized);

        chrome.runtime.sendMessage({
            type: "DOWNLOAD_IMAGE",
            url: normalized
        });
    });
}

function detectUrlChange() {
    if (location.href !== currentUrl) {
        console.log("페이지 변경 감지:", location.href);
        currentUrl = location.href;
        setTimeout(() => {
            collectImages();
        }, 1500);
    }
}

function main() {
    console.log("isAllowedSite():", isAllowedSite());

    if (!isAllowedSite()) {
        console.log("허용되지 않은 사이트");
        return; // 여기서 함수 종료
    }

    // 허용된 사이트일 경우만 실행
    collectImages();

    const observer = new MutationObserver(() => {
        detectUrlChange();
        collectImages();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    window.addEventListener("load", () => {
        collectImages();
    });

    window.addEventListener("scroll", () => {
        collectImages();
    });

    setInterval(() => {
        detectUrlChange();
    }, 2000);
}

// 실행 시작
main();
