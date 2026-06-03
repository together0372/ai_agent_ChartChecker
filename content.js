const observed = new Set();

// 💡 1. 백그라운드 서버와 통신하는 함수 (실수로 빠졌던 부분 복구!)
function sendToServer(imageUrl, imgElement, loadingUI, wrapper) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: { url: imageUrl, page: location.href, site: location.hostname }
    }, (response) => {
        // 서버의 답장이 오면 배지 삭제
        if (loadingUI) loadingUI.remove();

        if (chrome.runtime.lastError) {
            console.error("🚨 통신 에러:", chrome.runtime.lastError.message);
            return;
        }

        // 왜곡된 타겟 차트라면 퀴즈 팝업 UI 호출
        if (response && response.is_misleading === true) {
            if (typeof showQuizOverlay === "function") {
                showQuizOverlay(imgElement, response, wrapper);
            } else {
                console.error("🚨 showQuizOverlay 함수를 찾을 수 없습니다 (ui.js 확인 필요).");
            }
        }
    });
}

// 💡 2. 스크롤 감지기 (Intersection Observer) 설정
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.2 // 이미지가 화면에 20% 이상 보일 때 작동!
};

const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            
            // 한 번 감지했으니 더 이상 감시하지 않음
            observer.unobserve(img);
            
            console.log("👀 스크롤 감지! 화면에 등장:", img.src);
            
            // ui.js의 함수를 가져와서 뱃지 부착
            const wrapper = wrapImageForUI(img);
            const loadingUI = createLoadingOverlay();
            wrapper.appendChild(loadingUI);
            
            // 서버로 전송하고 5초/10초 타이머 시작!
            sendToServer(img.dataset.normalizedSrc, img, loadingUI, wrapper);
        }
    });
}, observerOptions);

function processImage(img) {
    const src = img.src || img.dataset.src || img.getAttribute("data-src");
    if (!src) return;

    const imgClass = (img.className || "").toLowerCase();
    const srcLower = src.toLowerCase();
    const adKeywords = ['icon', 'logo', 'btn']; 
    if (adKeywords.some(keyword => imgClass.includes(keyword) || srcLower.includes(keyword))) return;

    const normalized = src;
    if (observed.has(normalized)) return;

    if (img.width < 100 || img.height < 100) return;

    observed.add(normalized);
    img.dataset.normalizedSrc = normalized;
    
    // 이미지를 관찰 대상에 추가
    imageObserver.observe(img);
}

function collectImages() {
    const imgs = document.querySelectorAll("img");
    imgs.forEach(img => {
        if (img.complete) {
            processImage(img);
        } else {
            img.addEventListener('load', () => processImage(img));
        }
    });
}

function main() {
    collectImages();
    const mutationObserver = new MutationObserver(() => collectImages());
    mutationObserver.observe(document.body, { childList: true, subtree: true });
}

main();