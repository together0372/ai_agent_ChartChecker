const observed = new Set();
let activeAnalysisCount = 0; // 💡 동시에 분석 중인 차트 개수를 세는 글로벌 카운터

// 💡 1. 백그라운드 서버와 통신하는 함수
function sendToServer(imageUrl, imgElement, loadingUI, wrapper) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: { url: imageUrl, page: location.href, site: location.hostname }
    }, (response) => {
        if (loadingUI) loadingUI.remove();

        // 답장이 올 때마다 분석 중인 카운트 감소
        activeAnalysisCount--;

        if (chrome.runtime.lastError) {
            console.error("🚨 통신 에러:", chrome.runtime.lastError.message);
            if (activeAnalysisCount <= 0) {
                activeAnalysisCount = 0;
                updateFloatingWidget('silent');
            }
            return;
        }

        if (response && response.is_misleading === true) {
            // 🚨 왜곡 차트 감지 시 즉시 경고 띄움
            updateFloatingWidget('warning', imgElement);
            
            if (typeof showQuizOverlay === "function") {
                showQuizOverlay(imgElement, response, wrapper);
            }
            
            // 사용자가 퀴즈 내부의 버튼(O/X)을 누르는 순간 강제 침묵 모드로 전환
            wrapper.addEventListener('click', function(event) {
                if (event.target.closest('button')) {
                    updateFloatingWidget('force_silent');
                }
            }, { once: true });
        } else {
            // ✅ 정상 차트의 경우, 현재 분석 중인 다른 차트가 아예 없을 때만 말풍선을 닫음
            if (activeAnalysisCount <= 0) {
                activeAnalysisCount = 0;
                updateFloatingWidget('silent');
            }
        }
    });
}

// 💡 2. 스크롤 감지기 (Intersection Observer) 설정
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.2
};

const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            observer.unobserve(img);
            
            const wrapper = wrapImageForUI(img);
            const loadingUI = createLoadingOverlay();
            wrapper.appendChild(loadingUI);
            
            // 🚀 새로운 차트 감지 시 카운트 증가 및 분석 중 상태 호출
            activeAnalysisCount++;
            updateFloatingWidget('analyzing');
            
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
    if (typeof createFloatingWidget === "function") {
        createFloatingWidget();
    }
    
    collectImages();
    const mutationObserver = new MutationObserver(() => collectImages());
    mutationObserver.observe(document.body, { childList: true, subtree: true });
}

main();
