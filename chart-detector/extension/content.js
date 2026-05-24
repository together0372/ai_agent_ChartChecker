console.log("뉴스 차트 탐지기 로드됨");

const observed = new Set();

// 서버로 URL과 img 요소(Element)를 함께 보냅니다.
function sendToServer(imageUrl, imgElement) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: {
            url: imageUrl,
            page: location.href,
            site: location.hostname
        }
    }, (response) => {
        // ⭐ 크롬 통신이 중간에 끊겼는지 확인하는 에러 감지기
        if (chrome.runtime.lastError) {
            console.error("🚨 익스텐션 내부 통신 에러 (또는 타임아웃):", chrome.runtime.lastError.message);
            return;
        }

        console.log("🔥 서버 응답 도착! (content.js):", response);

        // 서버에서 왜곡된 차트라고 판정(is_misleading: true)하면 무조건 팝업을 띄웁니다!
        if (response && response.is_misleading === true) {
            console.log("✅ 팝업 띄우기 조건 만족! showQuizOverlay 실행!");
            showQuizOverlay(imgElement, response);
        } else {
            console.log("❌ 정상 차트이거나 데이터가 부족하여 팝업을 띄우지 않습니다.");
        }
    });
}

function collectImages() {
    const imgs = document.querySelectorAll(
        "article img, .article img, #container img"
    );

    imgs.forEach(img => {
        const src = img.src || img.dataset.src || img.getAttribute("data-src");
        if (!src) return;

        const normalized = normalizeUrl(src); // utils.js에 있는 함수 사용
        if (!normalized || !isValidImage(normalized)) return;

        if (observed.has(normalized)) return;

        if (img.naturalWidth < 400 || img.naturalHeight < 250) return;

        observed.add(normalized);
        console.log("이미지 발견, 서버로 전송:", normalized);
        
        // 발견한 이미지 엘리먼트를 같이 넘겨줍니다.
        sendToServer(normalized, img);
    });
}

function main() {
    if (!isAllowedSite()) {
        console.log("허용되지 않은 사이트");
        return;
    }

    console.log("뉴스 차트 탐지기 시작");
    collectImages();

    const observer = new MutationObserver(() => collectImages());
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("scroll", () => collectImages());
}

// 팝업 UI를 화면에 그리는 핵심 함수
function showQuizOverlay(imgElement, quizData) {
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    
    // 원본 이미지를 감싸기
    imgElement.parentNode.insertBefore(wrapper, imgElement);
    wrapper.appendChild(imgElement);

    const overlay = document.createElement('div');
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.zIndex = '999999'; // 화면 최상단으로 끌어올림

    // 퀴즈 데이터가 비어있을 경우를 대비한 기본값
    const question = quizData.quiz_question || "이 차트에는 과장되거나 생략된 정보가 있다?";
    const explanation = quizData.quiz_explanation || "AI가 퀴즈 해설을 생성하지 못했습니다.";
    
    overlay.innerHTML = `
        <div style="background: white; padding: 25px; border-radius: 12px; text-align: center; max-width: 85%; box-shadow: 0 4px 20px rgba(0,0,0,0.5); font-family: sans-serif;">
            <h3 style="margin: 0 0 15px 0; color: #ff4757; font-size: 20px;">👀 숨겨진 진실 퀴즈</h3>
            <p style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 25px; line-height: 1.4; word-break: keep-all;">
                ${question}
            </p>
            
            <div id="quiz-buttons">
                <button id="btn-o" style="padding: 12px 35px; font-size: 18px; font-weight: bold; cursor: pointer; background: #2ed573; color: white; border: none; border-radius: 8px; margin-right: 15px;">O (그렇다)</button>
                <button id="btn-x" style="padding: 12px 35px; font-size: 18px; font-weight: bold; cursor: pointer; background: #ff4757; color: white; border: none; border-radius: 8px;">X (아니다)</button>
            </div>

            <div id="quiz-result" style="display: none; margin-top: 20px; border-top: 2px solid #eee; padding-top: 20px;">
                <p id="result-text" style="font-size: 22px; font-weight: bold; margin: 0 0 10px 0;"></p>
                <p style="font-size: 15px; color: #555; margin: 0 0 20px 0; line-height: 1.6; word-break: keep-all; text-align: left; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    💡 <b>AI 해설:</b><br>${explanation}
                </p>
                <button id="btn-close" style="padding: 10px 25px; font-size: 15px; font-weight: bold; cursor: pointer; background: #dfe6e9; color: #2d3436; border: none; border-radius: 5px;">퀴즈 닫고 원본 보기</button>
            </div>
        </div>
    `;

    wrapper.appendChild(overlay);

    // 이벤트 리스너 추가
    const btnO = overlay.querySelector('#btn-o');
    const btnX = overlay.querySelector('#btn-x');
    const quizButtons = overlay.querySelector('#quiz-buttons');
    const quizResult = overlay.querySelector('#quiz-result');
    const resultText = overlay.querySelector('#result-text');
    const btnClose = overlay.querySelector('#btn-close');

    function handleAnswer(userAnswer) {
        // 정답 판별 (AI가 준 정답이 없으면 일단 O를 누르면 맞다고 처리)
        const correctAnswer = quizData.quiz_answer ? quizData.quiz_answer.replace(/[^OX]/g, '') : "O";
        
        quizButtons.style.display = 'none';
        quizResult.style.display = 'block';

        if (userAnswer === correctAnswer) {
            resultText.innerText = "🎉 정답입니다!";
            resultText.style.color = "#2ed573";
        } else {
            resultText.innerText = "앗, 속으셨네요! 😅";
            resultText.style.color = "#ff4757";
        }
    }

    btnO.addEventListener('click', () => handleAnswer("O"));
    btnX.addEventListener('click', () => handleAnswer("X"));
    btnClose.addEventListener('click', () => overlay.remove());
}

main();