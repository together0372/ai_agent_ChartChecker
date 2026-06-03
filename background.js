import { MOCK_DB } from './mockData.js';

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyze") {
        console.log("🤖 [Mock 서버] 분석 요청 수신:", request.data.url);

        // MOCK_DB에서 현재 이미지 URL에 포함된 키워드가 있는지 검사
        const matchedData = MOCK_DB.find(mock => request.data.url.includes(mock.keyword));

        if (matchedData) {
            console.log(`🎯 [Mock 서버] 타겟 차트 발견 (${matchedData.keyword})! 3초 후 퀴즈 전송`);
            
            setTimeout(() => {
                sendResponse({
                    success: true,
                    is_chart: true,
                    is_misleading: matchedData.is_misleading,
                    confidence: matchedData.confidence,
                    quiz_question: matchedData.quiz_question,
                    quiz_answer: matchedData.quiz_answer,
                    quiz_explanation: matchedData.quiz_explanation
                });
            }, 10000); 
        } else {
            // 타겟이 아닌 일반 이미지는 1초 뒤 조용히 패스 (팝업 안 뜸)
            setTimeout(() => {
                sendResponse({ success: true, is_chart: true, is_misleading: false });
            }, 5000);
        }
        
        return true; // 비동기 응답 필수!
    }
});