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
        if (chrome.runtime.lastError) {
            console.error("🚨 익스텐션 내부 통신 에러:", chrome.runtime.lastError.message);
            return;
        }

        console.log("🔥 서버 응답 도착! (content.js):", response);

        // 서버에서 왜곡된 차트라고 판정하면 무조건 팝업 띄우기
        if (response && response.is_misleading === true) {
            console.log("✅ 팝업 띄우기 조건 만족! showQuizOverlay 실행!");
            showQuizOverlay(imgElement, response);
        }
    });
}

// 우리가 작성했던 심플하고 확실한 '본문 타겟팅 + 가벼운 광고 차단' 로직
function collectImages() {
    // 1️⃣ 기사 본문을 감싸는 핵심 영역 찾기
    const bodySelectors = [
        "#articleBody", "#article_body", ".article_body", 
        "#dic_area", "#news_body_id", ".news_view", 
        ".article-view", "[itemprop='articleBody']", 
        ".news_contents", ".news-content"
    ];

    let targetContainer = null;
    for (let selector of bodySelectors) {
        const el = document.querySelector(selector);
        if (el) {
            targetContainer = el;
            break;
        }
    }
    targetContainer = targetContainer || document.querySelector("article") || document.body;

    const imgs = targetContainer.querySelectorAll("img");

    imgs.forEach(img => {
        const src = img.src || img.dataset.src || img.getAttribute("data-src");
        if (!src) return;

        // 2️⃣ 가벼운 광고/배너/아이콘 필터링
        const imgClass = (img.className || "").toLowerCase();
        const srcLower = src.toLowerCase();
        const adKeywords = ['ad', 'banner', 'sponsor', 'icon', 'logo', 'sns', 'btn', 'thumb'];
        
        const isAd = adKeywords.some(keyword => imgClass.includes(keyword) || srcLower.includes(keyword));
        if (isAd) return;

        const normalized = typeof normalizeUrl === "function" ? normalizeUrl(src) : src;
        if (!normalized) return;

        // 이미 서버로 보낸 이미지는 패스
        if (observed.has(normalized)) return;

        // 3️⃣ 최소 크기 필터 (단, 값이 0일 때는 지연 로딩 중일 수 있으므로 차단하지 않음)
        if (img.naturalWidth > 0 && img.naturalWidth < 400) return;
        if (img.naturalHeight > 0 && img.naturalHeight < 250) return;

        observed.add(normalized);
        console.log("✅ 이미지 발견, 서버로 전송:", normalized);
        
        sendToServer(normalized, img);
    });
}

function main() {
    if (typeof isAllowedSite === "function" && !isAllowedSite()) {
        return;
    }

    console.log("뉴스 차트 탐지기 시작");
    collectImages();

    // 심플하고 반응이 빠른 옵저버 유지
    const observer = new MutationObserver(() => collectImages());
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("scroll", () => collectImages());
}

// 팝업 UI를 화면에 그리는 핵심 함수 (하단 미니 배너 + 전체 퀴즈 화면 + 해설 바로보기)
function showQuizOverlay(imgElement, quizData) {
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    
    imgElement.parentNode.insertBefore(wrapper, imgElement);
    wrapper.appendChild(imgElement);

    const confidence = quizData.confidence || "중간";
    let trafficColor = "#ffa502"; 
    if (confidence === "높음") {
        trafficColor = "#ff4757"; 
    } else if (confidence === "낮음") {
        trafficColor = "#2ed573"; 
    }

    // 1️⃣ 하단 미니 제안 배너
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
    introBanner.style.zIndex = '999998';
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
                    <button id="btn-start-quiz" style="padding: 8px 14px; font-size: 14px; font-weight: bold; cursor: pointer; background: #2ed573; color: white; border: none; border-radius: 6px; transition: 0.2s;">O</button>
                    <button id="btn-skip-quiz" style="padding: 8px 14px; font-size: 14px; font-weight: bold; cursor: pointer; background: #a4b0be; color: white; border: none; border-radius: 6px; transition: 0.2s;">X</button>
                </div>
            </div>
            <button id="btn-show-answer" style="width: 100%; padding: 8px; font-size: 13px; font-weight: bold; cursor: pointer; background: #f1f2f6; color: #57606f; border: 1px solid #dfe4ea; border-radius: 6px; transition: 0.2s;">
                퀴즈 건너뛰고 바로 해설 보기 💡
            </button>
        </div>
    `;
    wrapper.appendChild(introBanner);

    // 2️⃣ 전체 퀴즈 UI
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
    quizOverlay.style.zIndex = '999999';

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

    // 이벤트 리스너 모음
    const btnO = quizOverlay.querySelector('#btn-o');
    const btnX = quizOverlay.querySelector('#btn-x');
    const quizButtons = quizOverlay.querySelector('#quiz-buttons');
    const quizResult = quizOverlay.querySelector('#quiz-result');
    const resultText = quizOverlay.querySelector('#result-text');
    const btnClose = quizOverlay.querySelector('#btn-close');

    introBanner.querySelector('#btn-start-quiz').addEventListener('click', () => {
        introBanner.style.display = 'none'; 
        quizOverlay.style.display = 'flex'; 
    });
    
    introBanner.querySelector('#btn-skip-quiz').addEventListener('click', () => {
        introBanner.remove(); 
    });

    introBanner.querySelector('#btn-show-answer').addEventListener('click', () => {
        introBanner.style.display = 'none'; 
        quizOverlay.style.display = 'flex'; 
        quizButtons.style.display = 'none'; 
        quizResult.style.display = 'block'; 
        resultText.innerText = "🔍 숨겨진 차트의 진실";
        resultText.style.color = "#333";
    });

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
    btnClose.addEventListener('click', () => {
        introBanner.remove();
        quizOverlay.remove();
    });

    const showAnswerBtn = introBanner.querySelector('#btn-show-answer');
    showAnswerBtn.addEventListener('mouseover', () => showAnswerBtn.style.backgroundColor = '#dfe4ea');
    showAnswerBtn.addEventListener('mouseout', () => showAnswerBtn.style.backgroundColor = '#f1f2f6');
}

main();