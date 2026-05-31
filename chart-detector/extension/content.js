console.log("뉴스 차트 탐지기 로드됨");

const observed = new Set();

// 서버로 URL과 img 요소(Element)를 함께 보냅니다.
function sendToServer(imageUrl, imgElement) {
    chrome.runtime.sendMessage({
        action: "analyze",
        data: {
            url: imageUrl,
            page: location.href,
            site: location.hostname,
            title: document.title
        }
    }, (response) => {
        if (chrome.runtime.lastError) {
            console.error("익스텐션 통신 에러:", chrome.runtime.lastError.message);
            return;
        }

        console.log("서버 응답:", response);

        if (response && response.is_misleading === true && imgElement) {
            showQuizOverlay(imgElement, response);
        }
    });
}

function collectImages() {
    const imgs = document.querySelectorAll("img");

    imgs.forEach(img => {
        const src =
            img.src ||
            img.dataset.src ||
            img.getAttribute("data-src");

        if (!src) return;

        const w = img.naturalWidth || img.clientWidth;
        const h = img.naturalHeight || img.clientHeight;
        if (w < 200 || h < 200) return;

        const normalized = normalizeUrl(src);
        if (!normalized) return;

        if (observed.has(normalized)) return;

        observed.add(normalized);

        console.log("이미지 발견:", normalized);
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

function showQuizOverlay(imgElement, quizData) {
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';

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
    overlay.style.zIndex = '999999';

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

    const btnO = overlay.querySelector('#btn-o');
    const btnX = overlay.querySelector('#btn-x');
    const quizButtons = overlay.querySelector('#quiz-buttons');
    const quizResult = overlay.querySelector('#quiz-result');
    const resultText = overlay.querySelector('#result-text');
    const btnClose = overlay.querySelector('#btn-close');

    function handleAnswer(userAnswer) {
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
