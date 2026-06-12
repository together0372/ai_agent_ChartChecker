function wrapImageForUI(imgElement) {
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    imgElement.parentNode.insertBefore(wrapper, imgElement);
    wrapper.appendChild(imgElement);
    return wrapper;
}

function createLoadingOverlay() {
    const loadingUI = document.createElement('div');
    loadingUI.style.position = 'absolute';
    loadingUI.style.top = '10px';
    loadingUI.style.right = '10px';
    loadingUI.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    loadingUI.style.color = '#fff';
    loadingUI.style.padding = '8px 15px';
    loadingUI.style.borderRadius = '20px';
    loadingUI.style.fontFamily = 'sans-serif';
    loadingUI.style.fontSize = '13px';
    loadingUI.style.fontWeight = 'bold';
    loadingUI.style.zIndex = '999999';
    loadingUI.innerHTML = '⚙️ AI 분석 중...';
    return loadingUI;
}

function showQuizOverlay(imgElement, quizData, wrapper) {
    const confidence = quizData.confidence || "중간";
    let trafficColor = "#ffa502"; 
    if (confidence === "높음") trafficColor = "#ff4757"; 
    else if (confidence === "낮음") trafficColor = "#2ed573"; 

    // 하단 미니 배너
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
    introBanner.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px;">👀</span>
                    <span style="font-size: 14px; font-weight: bold; color: #333; line-height: 1.3;">숨겨진 진실 퀴즈가 있습니다!<br>풀어보시겠어요?</span>
                </div>
                <div style="display: flex; gap: 6px;">
                    <button id="btn-start-quiz" style="padding: 8px 14px; font-weight: bold; cursor: pointer; background: #2ed573; color: white; border: none; border-radius: 6px;">O</button>
                    <button id="btn-skip-quiz" style="padding: 8px 14px; font-weight: bold; cursor: pointer; background: #a4b0be; color: white; border: none; border-radius: 6px;">X</button>
                </div>
            </div>
        </div>
    `;
    wrapper.appendChild(introBanner);

    // 전체 퀴즈 오버레이
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
        <div style="background: white; padding: 25px; border-radius: 12px; text-align: center; max-width: 85%; font-family: sans-serif;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #333; font-size: 20px;">👀 숨겨진 진실 퀴즈</h3>
                <div title="확신도: ${confidence}" style="width: 16px; height: 16px; border-radius: 50%; background-color: ${trafficColor};"></div>
            </div>
            <p style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 25px; word-break: keep-all;">${quizData.quiz_question}</p>
            <div id="quiz-buttons">
                <button id="btn-o" style="padding: 12px 35px; font-size: 18px; font-weight: bold; cursor: pointer; background: #2ed573; color: white; border: none; border-radius: 8px; margin-right: 15px;">O</button>
                <button id="btn-x" style="padding: 12px 35px; font-size: 18px; font-weight: bold; cursor: pointer; background: #ff4757; color: white; border: none; border-radius: 8px;">X</button>
            </div>
            <div id="quiz-result" style="display: none; margin-top: 20px; border-top: 2px solid #eee; padding-top: 20px;">
                <p id="result-text" style="font-size: 22px; font-weight: bold; margin: 0 0 10px 0;"></p>
                <p style="font-size: 15px; color: #555; margin: 0 0 20px 0; line-height: 1.6; text-align: left; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    💡 <b>AI 해설:</b><br>${quizData.quiz_explanation}
                </p>
                <button id="btn-close" style="padding: 10px 25px; font-size: 15px; font-weight: bold; cursor: pointer; background: #dfe6e9; color: #2d3436; border: none; border-radius: 5px;">닫기</button>
            </div>
        </div>
    `;
    wrapper.appendChild(quizOverlay);

    // 이벤트 연결
    introBanner.querySelector('#btn-start-quiz').onclick = () => { introBanner.style.display = 'none'; quizOverlay.style.display = 'flex'; };
    introBanner.querySelector('#btn-skip-quiz').onclick = () => introBanner.remove();
    quizOverlay.querySelector('#btn-close').onclick = () => { introBanner.remove(); quizOverlay.remove(); };
    
    const handleAnswer = (userAnswer) => {
        const correctAnswer = quizData.quiz_answer.replace(/[^OX]/g, '');
        quizOverlay.querySelector('#quiz-buttons').style.display = 'none';
        quizOverlay.querySelector('#quiz-result').style.display = 'block';
        const resultText = quizOverlay.querySelector('#result-text');
        
        if (userAnswer === correctAnswer) {
            resultText.innerText = "🎉 정답입니다!";
            resultText.style.color = "#2ed573";
        } else {
            resultText.innerText = "앗, 속으셨네요! 😅";
            resultText.style.color = "#ff4757";
        }
    };
    
    quizOverlay.querySelector('#btn-o').onclick = () => handleAnswer("O");
    quizOverlay.querySelector('#btn-x').onclick = () => handleAnswer("X");
}

// ==========================================
// 🤖 둥둥 떠다니는 AI 에이전트 위젯 생성 및 상태 관리
// ==========================================

function createFloatingWidget() {
    if (document.getElementById('chartquiz-floating-widget')) return;

    const widget = document.createElement('div');
    widget.id = 'chartquiz-floating-widget';
    widget.dataset.state = 'silent'; // 💡 현재 상태를 기억할 숨겨진 태그 추가
    
    widget.style.cssText = `
        position: fixed;
        right: 20px;
        bottom: 20px;
        z-index: 999999;
        display: flex;
        align-items: flex-end;
        gap: 10px;
        font-family: 'Noto Sans KR', sans-serif;
        pointer-events: none;
    `;

    widget.innerHTML = `
        <div id="chartquiz-bubble" style="
            background: #2c3e50;
            color: white;
            padding: 10px 15px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            opacity: 0.9;
            display: none;
        "></div>
        <div id="chartquiz-bot" style="
            font-size: 35px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            padding: 5px;
            transition: transform 0.3s;
        ">🤖</div>
    `;

    document.body.appendChild(widget);
}

function updateFloatingWidget(state, imgElement = null) {
    const widget = document.getElementById('chartquiz-floating-widget');
    if (!widget) return;
    
    const bubble = document.getElementById('chartquiz-bubble');
    const bot = document.getElementById('chartquiz-bot');
    if (!bubble || !bot) return;

    const currentState = widget.dataset.state; // 현재 상태 확인

    if (state === 'analyzing') {
        // 💡 이미 경고가 뜬 상태라면 다른 차트가 분석 중이어도 경고를 가리지 않음
        if (currentState === 'warning') return;
        
        widget.dataset.state = 'analyzing';
        bubble.innerText = "🤔 화면 내 차트를 분석하고 있습니다...";
        bubble.style.background = "#f39c12";
        bubble.style.display = 'flex';
        bot.style.transform = "rotate(15deg) scale(1.1)";
        
    } else if (state === 'warning' && imgElement) {
        // 🚨 경고 상태 강제 고정
        widget.dataset.state = 'warning';
        const rect = imgElement.getBoundingClientRect();
        let direction = "이 화면에";
        
        if (rect.bottom < 0) {
            direction = "화면 위쪽에 ⬆️";
        } else if (rect.top > window.innerHeight) {
            direction = "화면 아래쪽에 ⬇️";
        }

        bubble.innerText = `🚨 경고! ${direction} 왜곡된 차트 발견!`;
        bubble.style.background = "#e74c3c";
        bubble.style.display = 'flex';
        bot.style.transform = "rotate(-15deg) scale(1.2)";
        
    } else if (state === 'silent') {
        // 💡 일반적인 분석 완료 신호는 현재 켜져 있는 '경고'를 끌 수 없음
        if (currentState === 'warning') return;
        
        widget.dataset.state = 'silent';
        bubble.innerText = '';
        bubble.style.display = 'none';
        bot.style.transform = "rotate(0deg) scale(1)";
        
    } else if (state === 'force_silent') {
        // 🎯 퀴즈 상호작용이 시작되었을 때만 경고를 강제로 해제
        widget.dataset.state = 'silent';
        bubble.innerText = '';
        bubble.style.display = 'none';
        bot.style.transform = "rotate(0deg) scale(1)";
    }
}
