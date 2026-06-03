console.log("뉴스 차트 탐지기 v2 로드됨");

const observed = new Set();

// ────────────────────────────────────────────────
// CSS 애니메이션 (스피너용, 1회만 삽입)
// ────────────────────────────────────────────────
(function injectSpinStyle() {
    if (document.getElementById("cd-spin-style")) return;
    const style = document.createElement("style");
    style.id = "cd-spin-style";
    style.textContent = "@keyframes cd-spin { to { transform: rotate(360deg); } }";
    document.head.appendChild(style);
})();

// ────────────────────────────────────────────────
// 공통: 이미지 래퍼 (없으면 생성, 있으면 재사용)
// ────────────────────────────────────────────────
function wrapImage(imgElement) {
    if (imgElement.parentNode && imgElement.parentNode.classList.contains("chart-detector-wrapper")) {
        return imgElement.parentNode;
    }
    const wrapper = document.createElement("div");
    wrapper.classList.add("chart-detector-wrapper");
    wrapper.style.cssText = "position:relative; display:inline-block;";
    imgElement.parentNode.insertBefore(wrapper, imgElement);
    wrapper.appendChild(imgElement);
    return wrapper;
}

// ────────────────────────────────────────────────
// 로딩 스피너 표시 / 제거
// ────────────────────────────────────────────────
function showLoadingSpinner(imgElement) {
    const wrapper = wrapImage(imgElement);
    if (wrapper.querySelector(".cd-spinner")) return; // 이미 있으면 스킵

    const spinner = document.createElement("div");
    spinner.classList.add("cd-spinner");
    spinner.style.cssText = `
        position: absolute; top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.38);
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        z-index: 999997;
        border-radius: 4px;
        pointer-events: none;
    `;
    spinner.innerHTML = `
        <div style="
            width: 38px; height: 38px;
            border: 4px solid rgba(255,255,255,0.25);
            border-top: 4px solid #ffffff;
            border-radius: 50%;
            animation: cd-spin 0.75s linear infinite;
        "></div>
        <p style="
            margin: 10px 0 0;
            color: white;
            font-size: 12px;
            font-family: 'Apple SD Gothic Neo', '맑은 고딕', sans-serif;
            letter-spacing: 0.5px;
        ">AI 분석 중...</p>
    `;
    wrapper.appendChild(spinner);
}

function removeLoadingSpinner(imgElement) {
    const wrapper = imgElement.parentNode;
    if (!wrapper || !wrapper.classList.contains("chart-detector-wrapper")) return;
    const spinner = wrapper.querySelector(".cd-spinner");
    if (spinner) spinner.remove();
}

// ────────────────────────────────────────────────
// 서버 통신
// ────────────────────────────────────────────────
function sendToServer(imageUrl, imgElement) {
    showLoadingSpinner(imgElement); // 분석 시작 → 스피너 ON

    chrome.runtime.sendMessage({
        action: "analyze",
        data: {
            url:   imageUrl,
            page:  location.href,
            site:  location.hostname,
            title: document.title,
        }
    }, (response) => {
        removeLoadingSpinner(imgElement); // 응답 도착 → 스피너 OFF

        if (chrome.runtime.lastError) {
            console.error("익스텐션 통신 에러:", chrome.runtime.lastError.message);
            return;
        }

        console.log("서버 응답:", response);

        if (response && response.is_misleading === true && imgElement) {
            showDistortionOverlay(imgElement, response);
        }
    });
}

// ────────────────────────────────────────────────
// 이미지 수집
// ────────────────────────────────────────────────
function collectImages() {
    const imgs = document.querySelectorAll("img");

    imgs.forEach(img => {
        const src = img.src || img.dataset.src || img.getAttribute("data-src");
        if (!src) return;

        const w = img.naturalWidth  || img.clientWidth;
        const h = img.naturalHeight || img.clientHeight;
        if (w < 200 || h < 200) return;

        const normalized = normalizeUrl(src);
        if (!normalized || !isValidImage(normalized)) return;
        if (observed.has(normalized)) return;

        observed.add(normalized);
        console.log("이미지 발견:", normalized);
        sendToServer(normalized, img);
    });
}

// ────────────────────────────────────────────────
// 오류 탐지 오버레이 (오류 탭 + OX 퀴즈 탭)
// ────────────────────────────────────────────────
function showDistortionOverlay(imgElement, data) {

    // 이미 배지가 있으면 스킵
    const wrapper = imgElement.parentNode;
    if (wrapper && wrapper.querySelector(".cd-badge")) return;

    // wrapImage로 래퍼 재사용 (스피너가 썼던 래퍼 그대로 사용)
    const w = wrapImage(imgElement);

    // ── 경고 배지 ───────────────────────────────────────────
    const badge = document.createElement("div");
    badge.classList.add("cd-badge");
    badge.style.cssText = `
        position: absolute; top: 8px; right: 8px;
        background: rgba(255,71,87,0.92); color: white;
        font-size: 13px; font-weight: bold;
        padding: 4px 10px; border-radius: 20px;
        cursor: pointer; z-index: 999998;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        user-select: none;
        font-family: sans-serif;
    `;
    badge.innerText = "⚠️ 차트 오류 감지";
    badge.addEventListener("click", () => modal.style.display = "flex");
    w.appendChild(badge);

    // ── 모달 오버레이 ──────────────────────────────────────
    const modal = document.createElement("div");
    modal.style.cssText = `
        display: none;
        position: fixed; top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.65);
        justify-content: center; align-items: center;
        z-index: 999999;
        font-family: 'Apple SD Gothic Neo', '맑은 고딕', sans-serif;
    `;

    // ── 오류 목록 HTML ─────────────────────────────────────
    const errors = data.visual_errors || [];
    const errorListHtml = errors.length > 0
        ? errors.map(e => `<li style="margin: 6px 0; padding: 8px 12px; background:#fff5f5; border-left:3px solid #ff4757; border-radius:4px; font-size:14px;">${e}</li>`).join("")
        : `<li style="color:#888; font-size:14px;">구체적인 오류 목록 없음</li>`;

    const confidenceText = data.confidence ? ` (신뢰도: ${data.confidence})` : "";

    // ── OX 퀴즈 HTML ─────────────────────────────────────
    const question    = data.quiz_question    || "이 차트에는 오해를 유발하는 요소가 있다?";
    const quizAnswer  = data.quiz_answer      ? data.quiz_answer.replace(/[^OX]/g, "") : "O";
    const explanation = data.quiz_explanation || "AI가 해설을 생성하지 못했습니다.";

    // ── 모달 컨텐츠 ────────────────────────────────────────
    modal.innerHTML = `
    <div style="background:white; border-radius:16px; width:90%; max-width:480px; max-height:85vh;
                overflow-y:auto; box-shadow:0 8px 32px rgba(0,0,0,0.4);">

        <!-- 헤더 -->
        <div style="background:linear-gradient(135deg,#ff4757,#ff6b81); padding:18px 20px; border-radius:16px 16px 0 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin:0; color:white; font-size:18px;">⚠️ AI 차트 왜곡 감지</h2>
                <span id="cd-close" style="color:white; font-size:22px; cursor:pointer; line-height:1;">✕</span>
            </div>
            <p style="margin:6px 0 0; color:rgba(255,255,255,0.9); font-size:13px;">
                판정: <b style="color:white;">${data.verdict || "오류"}${confidenceText}</b>
            </p>
        </div>

        <!-- 탭 버튼 -->
        <div style="display:flex; border-bottom:2px solid #f0f0f0;">
            <button id="cd-tab-errors" style="
                flex:1; padding:12px; border:none; background:white; cursor:pointer;
                font-size:14px; font-weight:bold; color:#ff4757;
                border-bottom:3px solid #ff4757; margin-bottom:-2px;">
                ⚠️ 오류 목록
            </button>
            <button id="cd-tab-quiz" style="
                flex:1; padding:12px; border:none; background:white; cursor:pointer;
                font-size:14px; font-weight:bold; color:#747d8c;
                border-bottom:3px solid transparent; margin-bottom:-2px;">
                ❓ OX 퀴즈
            </button>
        </div>

        <!-- 오류 목록 패널 -->
        <div id="cd-panel-errors" style="padding:20px; display:block;">
            <p style="margin:0 0 12px; font-size:13px; color:#555;">
                이 차트에서 다음과 같은 <b style="color:#ff4757;">${errors.length}가지 왜곡 기법</b>이 발견되었습니다.
            </p>
            <ul style="margin:0; padding:0; list-style:none;">
                ${errorListHtml}
            </ul>
        </div>

        <!-- OX 퀴즈 패널 -->
        <div id="cd-panel-quiz" style="padding:20px; display:none;">
            <!-- 질문 -->
            <p style="font-size:15px; font-weight:bold; color:#2d3436; margin:0 0 20px; line-height:1.5; word-break:keep-all;">
                ${question}
            </p>

            <!-- 버튼 -->
            <div id="cd-quiz-btns" style="display:flex; gap:12px;">
                <button id="cd-btn-o" style="
                    flex:1; padding:14px; font-size:20px; font-weight:bold;
                    background:#2ed573; color:white; border:none; border-radius:10px; cursor:pointer;">
                    O
                </button>
                <button id="cd-btn-x" style="
                    flex:1; padding:14px; font-size:20px; font-weight:bold;
                    background:#ff4757; color:white; border:none; border-radius:10px; cursor:pointer;">
                    X
                </button>
            </div>

            <!-- 결과 (초기 숨김) -->
            <div id="cd-quiz-result" style="display:none; margin-top:18px;">
                <p id="cd-result-text" style="font-size:22px; font-weight:bold; text-align:center; margin:0 0 12px;"></p>
                <div style="background:#f8f9fa; border-radius:10px; padding:14px;">
                    <p style="margin:0; font-size:14px; color:#555; line-height:1.6; word-break:keep-all;">
                        💡 <b>AI 해설:</b><br>${explanation}
                    </p>
                </div>
                <button id="cd-retry" style="
                    margin-top:12px; width:100%; padding:10px;
                    background:#dfe6e9; color:#2d3436; border:none; border-radius:8px;
                    cursor:pointer; font-size:14px; font-weight:bold;">
                    다시 풀기
                </button>
            </div>
        </div>

    </div>`;

    document.body.appendChild(modal);

    // ── 이벤트 바인딩 ──────────────────────────────────────
    // 닫기
    modal.querySelector("#cd-close").addEventListener("click", () => {
        modal.style.display = "none";
    });
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.style.display = "none";
    });

    // 탭 전환
    const tabErrors   = modal.querySelector("#cd-tab-errors");
    const tabQuiz     = modal.querySelector("#cd-tab-quiz");
    const panelErrors = modal.querySelector("#cd-panel-errors");
    const panelQuiz   = modal.querySelector("#cd-panel-quiz");

    tabErrors.addEventListener("click", () => {
        panelErrors.style.display = "block";
        panelQuiz.style.display   = "none";
        tabErrors.style.color              = "#ff4757";
        tabErrors.style.borderBottomColor  = "#ff4757";
        tabQuiz.style.color                = "#747d8c";
        tabQuiz.style.borderBottomColor    = "transparent";
    });

    tabQuiz.addEventListener("click", () => {
        panelErrors.style.display = "none";
        panelQuiz.style.display   = "block";
        tabQuiz.style.color                = "#ff4757";
        tabQuiz.style.borderBottomColor    = "#ff4757";
        tabErrors.style.color              = "#747d8c";
        tabErrors.style.borderBottomColor  = "transparent";
    });

    // OX 퀴즈 답변
    const quizBtns   = modal.querySelector("#cd-quiz-btns");
    const quizResult = modal.querySelector("#cd-quiz-result");
    const resultText = modal.querySelector("#cd-result-text");

    function handleAnswer(userAnswer) {
        quizBtns.style.display   = "none";
        quizResult.style.display = "block";

        if (userAnswer === quizAnswer) {
            resultText.innerText   = "🎉 정답입니다!";
            resultText.style.color = "#2ed573";
        } else {
            resultText.innerText   = "앗, 속으셨네요! 😅";
            resultText.style.color = "#ff4757";
        }
    }

    modal.querySelector("#cd-btn-o").addEventListener("click", () => handleAnswer("O"));
    modal.querySelector("#cd-btn-x").addEventListener("click", () => handleAnswer("X"));

    // 다시 풀기
    modal.querySelector("#cd-retry").addEventListener("click", () => {
        quizBtns.style.display   = "flex";
        quizResult.style.display = "none";
    });
}

// ────────────────────────────────────────────────
// 메인 진입점
// ────────────────────────────────────────────────
function main() {
    if (!isAllowedSite()) {
        console.log("허용되지 않은 사이트");
        return;
    }

    console.log("뉴스 차트 탐지기 v2 시작");

    collectImages();

    // DOM 변경 감지
    const observer = new MutationObserver(() => {
        clearTimeout(window.__chartObserverTimeout);
        window.__chartObserverTimeout = setTimeout(collectImages, 300);
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // 스크롤 감지
    let scrollTimeout;
    window.addEventListener("scroll", () => {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(collectImages, 300);
    });
}

main();
