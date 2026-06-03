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