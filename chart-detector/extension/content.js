console.log("뉴스 차트 탐지기 로드됨");

const observed = new Set();

// 🌟 서버로 전송하고 결과를 받는 함수 (로딩 딜레이 적용)
function sendToServer(imageUrl, imgElement, loadingUI, wrapper) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: {
            url: imageUrl,
            page: location.href,
            site: location.hostname
        }
    }, (response) => {
        // [핵심 변경] CNN이 0.1초 만에 응답하더라도, 사용자가 볼 수 있게 최소 1초는 로딩 바를 띄워둡니다.
        setTimeout(() => {
            if (loadingUI) loadingUI.remove(); // 1초 뒤에 스피너 삭제

            if (chrome.runtime.lastError) {
                console.error("🚨 익스텐션 내부 통신 에러:", chrome.runtime.lastError.message);
                return;
            }

            console.log("🔥 서버 응답 도착! (content.js):", response);

            // 서버에서 왜곡된 차트라고 판정하면 팝업 띄우기
            if (response && response.is_misleading === true) {
                console.log("✅ 팝업 띄우기 조건 만족! showQuizOverlay 실행!");
                showQuizOverlay(imgElement, response, wrapper);
            }
        }, 1000); // 1000ms (1초) 대기
    });
}

function collectImages() {
    const bodySelectors = [
        "#articleBody", "#article_body", ".article_body", 
        "#dic_area", "#news_body_id", ".news_view", 
        ".article-view", "[itemprop='articleBody']", 
        ".news_contents", ".news-content"
    ];

    let targetContainer = null;
    for (let selector of bodySelectors) {
        const el = document.querySelector(selector);
        if (el) { targetContainer = el; break; }
    }
    targetContainer = targetContainer || document.querySelector("article") || document.body;

    const imgs = targetContainer.querySelectorAll("img");

    imgs.forEach(img => {
        const src = img.src || img.dataset.src || img.getAttribute("data-src");
        if (!src) return;

        const imgClass = (img.className || "").toLowerCase();
        const srcLower = src.toLowerCase();
        const adKeywords = ['ad', 'banner', 'sponsor', 'icon', 'logo', 'sns', 'btn', 'thumb'];
        
        const isAd = adKeywords.some(keyword => imgClass.includes(keyword) || srcLower.includes(keyword));
        if (isAd) return;

        const normalized = typeof normalizeUrl === "function" ? normalizeUrl(src) : src;
        if (!normalized) return;

        if (observed.has(normalized)) return;

        if (img.naturalWidth > 0 && img.naturalWidth < 400) return;
        if (img.naturalHeight > 0 && img.naturalHeight < 250) return;

        observed.add(normalized);
        console.log("✅ 이미지 발견, 서버로 전송:", normalized);

        // =================================================================
        // 🌟 [UI 강화] 절대 다른 요소에 가려지지 않도록 최상단(Z-index) 설정
        // =================================================================
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';
        wrapper.style.width = 'fit-content';
        wrapper.style.margin = 'auto'; // 이미지 중앙 정렬 깨짐 방지
        img.parentNode.insertBefore(wrapper, img);
        wrapper.appendChild(img);

        const loadingUI = document.createElement('div');
        loadingUI.style.position = 'absolute';
        loadingUI.style.top = '15px';
        loadingUI.style.right = '15px';
        loadingUI.style.backgroundColor = 'rgba(0, 0, 0, 0.85)';
        loadingUI.style.color = '#ffffff';
        loadingUI.style.padding = '8px 16px';
        loadingUI.style.borderRadius = '30px';
        loadingUI.style.fontSize = '14px';
        loadingUI.style.fontWeight = 'bold';
        loadingUI.style.zIndex = '2147483647'; // 👑 크롬이 허용하는 가장 높은 우선순위 값
        loadingUI.style.display = 'flex';
        loadingUI.style.alignItems = 'center';
        loadingUI.style.gap = '8px';
        loadingUI.style.backdropFilter = 'blur(6px)';
        loadingUI.style.boxShadow = '0 4px 12px rgba(0,0,0,0.5)';
        loadingUI.innerHTML = `<span class="ai-spinner">⚙️</span> AI 분석 중...`;
        wrapper.appendChild(loadingUI);

        // 스피너 CSS 애니메이션 강제 주입
        if (!document.getElementById('chart-spinner-style')) {
            const style = document.createElement('style');
            style.id = 'chart-spinner-style';
            style.innerHTML = `
                @keyframes spin { 100% { transform: rotate(360deg); } } 
                .ai-spinner { display: inline-block; animation: spin 2s linear infinite; font-size: 16px; }
            `;
            document.head.appendChild(style);
        }

        // 전송
        sendToServer(normalized, img, loadingUI, wrapper);
    });
}

function main() {
    if (typeof isAllowedSite === "function" && !isAllowedSite()) return;

    console.log("뉴스 차트 탐지기 시작");
    collectImages();

    const observer = new MutationObserver(() => collectImages());
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("scroll", () => collectImages());
}

// 팝업 UI 화면 렌더링
function showQuizOverlay(imgElement, quizData, wrapper) {
    const confidence = quizData.confidence || "중간";
    let trafficColor = "#ffa502"; 
    if (confidence === "높음") { trafficColor = "#ff4757"; } 
    else if (confidence === "낮음") { trafficColor = "#2ed573"; }

    const introBanner = document.createElement('div');
    introBanner.style.position = 'absolute';
    introBanner.style.bottom = '15px'; 
    introBanner.style.left = '50%';
    introBanner.style.transform = 'translateX(-50%)'; 
    introBanner.style.width = '90%';
    introBanner.style.maxWidth = '380px'; 
    introBanner.style.backgroundColor = 'rgba(255, 255, 255, 0.95)'; 
    introBanner.style.padding = '12px 15px';
    introBanner.style.borderRadius = '10px';
    introBanner.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.3)'; 
    introBanner.style.zIndex = '2147483646';
    introBanner.style.fontFamily = 'sans-serif';
    introBanner.style.backdropFilter = 'blur(4px)'; 

    introBanner.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px;">👀</span>
                    <span style="font-size: 14px; font-weight: bold; color: #333; line-height: 1.3; word-break: keep-all;">
                        숨겨진 진실 퀴즈가 있습니다!<br>풀어보시겠어요?
                    </span>
                </div>
                <div style="display: flex; gap: 6px;">
                    <button id="btn-start-quiz" style="padding: 8px 14px; font-size: 14px; font-weight: bold; cursor: pointer; background: #2ed573; color: white; border: none; border-radius: 6px;">O</button>
                    <button id="btn-skip-quiz" style="padding: 8px 14px; font-size: 14px; font-weight: bold; cursor: pointer; background: #a4b0be; color: white; border: none; border-radius: 6px;">X</button>
                </div>
            </div>
            <button id="btn-show-answer" style="width: 100%; padding: 8px; font-size: 13px; font-weight: bold; cursor: pointer; background: #f1f2f6; color: #57606f; border: 1px solid #dfe4ea; border-radius: 6px;">
                퀴즈 건너뛰고 바로 해설 보기 💡
            </button>
        </div>
    `;
    wrapper.appendChild(introBanner);

    const quizOverlay = document.createElement('div');
    quizOverlay.style.position = 'absolute';
    quizOverlay.style.top = '0';
    quizOverlay.style.left = '0';
    quizOverlay.style.width = '100%';
    quizOverlay.style.height = '100%';
    quizOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    quizOverlay.style.display = 'none'; 
    quizOverlay.style.justifyContent = 'center';
    quizOverlay.style.alignItems = 'center';
    quizOverlay.style.zIndex = '2147483647';

    quizOverlay.innerHTML = `
        <div style="background: white; padding: 25px; border-radius: 12px; text-align: center; max-width: 85%; box-shadow: 0 4px 20px rgba(0,0,0,0.5); font-family: sans-serif;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #333; font-size: 20px;">👀 숨겨진 진실 퀴즈</h3>
                <div title="AI 왜곡 확신도: ${confidence}" style="width: 16px; height: 16px; border-radius: 50%; background-color: ${trafficColor}; box-shadow: 0 0 8px ${trafficColor};"></div>
            </div>
            <p style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 25px; line-height: 1.4; word-break: keep-all;">
                ${quizData.quiz_question || "이 차트에는 과장되거나 생략된 정보가 있다?"}
            </p>
            <div id="quiz-buttons">
                <button id="btn-o" style="padding: 12px 35px; font-size: 18px; font-weight: bold; cursor: pointer; background: #2ed573; color: white; border: none; border-radius: 8px; margin-right: 15px;">O (그렇다)</button>
                <button id="btn-x" style="padding: 12px 35px; font-size: 18px; font-weight: bold; cursor: pointer; background: #ff4757; color: white; border: none; border-radius: 8px;">X (아니다)</button>
            </div>
            <div id="quiz-result" style="display: none; margin-top: 20px; border-top: 2px solid #eee; padding-top: 20px;">
                <p id="result-text" style="font-size: 22px; font-weight: bold; margin: 0 0 10px 0;"></p>
                <p style="font-size: 15px; color: #555; margin: 0 0 20px 0; line-height: 1.6; word-break: keep-all; text-align: left; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    💡 <b>AI 해설:</b><br>${quizData.quiz_explanation || "AI가 퀴즈 해설을 생성하지 못했습니다."}
                </p>
                <button id="btn-close" style="padding: 10px 25px; font-size: 15px; font-weight: bold; cursor: pointer; background: #dfe6e9; color: #2d3436; border: none; border-radius: 5px;">퀴즈 닫고 원본 보기</button>
            </div>
        </div>
    `;
    wrapper.appendChild(quizOverlay);

    introBanner.querySelector('#btn-start-quiz').addEventListener('click', () => { introBanner.style.display = 'none'; quizOverlay.style.display = 'flex'; });
    introBanner.querySelector('#btn-skip-quiz').addEventListener('click', () => { introBanner.remove(); });
    introBanner.querySelector('#btn-show-answer').addEventListener('click', () => {
        introBanner.style.display = 'none'; quizOverlay.style.display = 'flex'; 
        quizOverlay.querySelector('#quiz-buttons').style.display = 'none';
        quizOverlay.querySelector('#quiz-result').style.display = 'block';
        quizOverlay.querySelector('#result-text').innerText = "🔍 숨겨진 차트의 진실";
        quizOverlay.querySelector('#result-text').style.color = "#333";
    });

    const btnO = quizOverlay.querySelector('#btn-o');
    const btnX = quizOverlay.querySelector('#btn-x');
    const quizResult = quizOverlay.querySelector('#quiz-result');
    const resultText = quizOverlay.querySelector('#result-text');

    function handleAnswer(userAnswer) {
        const correctAnswer = quizData.quiz_answer ? quizData.quiz_answer.replace(/[^OX]/g, '') : "O";
        quizOverlay.querySelector('#quiz-buttons').style.display = 'none';
        quizResult.style.display = 'block';

        if (userAnswer === correctAnswer) { resultText.innerText = "🎉 정답입니다!"; resultText.style.color = "#2ed573"; } 
        else { resultText.innerText = "앗, 속으셨네요! 😅"; resultText.style.color = "#ff4757"; }
    }

    btnO.addEventListener('click', () => handleAnswer("O"));
    btnX.addEventListener('click', () => handleAnswer("X"));
    quizOverlay.querySelector('#btn-close').addEventListener('click', () => { introBanner.remove(); quizOverlay.remove(); });
}

main();