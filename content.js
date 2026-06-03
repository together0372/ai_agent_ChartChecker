console.log("🚀 ChartQuiz MVP 모드 로드 완료!");
const observed = new Set();

function sendToServer(imageUrl, imgElement, loadingUI, wrapper) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: { url: imageUrl, page: location.href, site: location.hostname }
    }, (response) => {
        // Mock 서버의 답장이 오면 무조건 배지 삭제
        if (loadingUI) loadingUI.remove();

        if (chrome.runtime.lastError) {
            console.error("🚨 통신 에러:", chrome.runtime.lastError.message);
            return;
        }

        // 왜곡된 타겟 차트라면 UI 호출 (ui.js에 있는 함수 사용)
        if (response && response.is_misleading === true) {
            showQuizOverlay(imgElement, response, wrapper);
        }
    });
}

// ✅ 이미지를 실제로 검사하고 뱃지를 붙이는 전담 함수
function processImage(img) {
    const src = img.src || img.dataset.src || img.getAttribute("data-src");
    if (!src) return;

    // URL이나 클래스에 'thumb', 'sns' 등이 있어도 정상 차트일 수 있으니 필터 대폭 완화!
    const imgClass = (img.className || "").toLowerCase();
    const srcLower = src.toLowerCase();
    const adKeywords = ['icon', 'logo', 'btn']; // 확실한 것만 놔두고 전부 제거
    if (adKeywords.some(keyword => imgClass.includes(keyword) || srcLower.includes(keyword))) return;

    const normalized = src;
    if (observed.has(normalized)) return;

    // 💡 핵심: 이제 이미지가 로드된 상태이므로 정확한 width/height 측정이 가능함!
    if (img.width < 100 || img.height < 100) return;

    observed.add(normalized);
    console.log("✅ 이미지 정찰 완료:", normalized);
    
    // ui.js의 함수를 가져와서 뱃지 부착
    const wrapper = wrapImageForUI(img);
    const loadingUI = createLoadingOverlay();
    wrapper.appendChild(loadingUI);
    
    sendToServer(normalized, img, loadingUI, wrapper);
}

// ✅ 문서를 뒤져서 이미지 수집 시작 (로딩 대기 로직 추가)
function collectImages() {
    // 복잡한 컨테이너 조건 다 지우고, 페이지 전체 이미지를 싹 다 검사!
    const imgs = document.querySelectorAll("img");

    imgs.forEach(img => {
        if (img.complete) {
            // 이미 다운로드가 끝나서 화면에 보인다면 즉시 처리
            processImage(img);
        } else {
            // 아직 다운로드 중이라면, 다 그려진 직후(load)에 처리하도록 예약!
            img.addEventListener('load', () => processImage(img));
        }
    });
}

function main() {
    collectImages();
    const observer = new MutationObserver(() => collectImages());
    observer.observe(document.body, { childList: true, subtree: true });
}

main();