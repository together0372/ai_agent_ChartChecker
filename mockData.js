export const MOCK_DB = [
    // 🚨 1번 구역: 동아일보 (역순 배치)
    {
        keyword: "99360936", 
        is_misleading: true,
        confidence: "높음",
        quiz_question: "이 차트는 가로축(X축)의 연도가 최신부터 과거 순으로 '역순' 배치되어 있다? (O/X)",
        quiz_answer: "O",
        quiz_explanation: "최근 연도를 왼쪽에 두는 역순 배치는 하락하는 데이터를 우상향하는 것처럼 착각하게 만듭니다."
    },
    
    // 🚨 2번 구역: 빅카인즈 산업활동동향 (날짜 오기재 및 장식 요소 가림)
    {
        keyword: "04100078", 
        is_misleading: true,
        confidence: "높음",
        quiz_question: "이 차트는 X축의 연도가 '2025.3'이어야 할 곳에 '2024.3'으로 잘못 표기되어 시계열에 혼란을 주고 있다? (O/X)",
        quiz_answer: "O",
        quiz_explanation: "데이터의 실제 시점과 다른 연도를 축에 표기하는 심각한 오류가 있습니다. 또한, 차트 중앙을 가리는 화물선 이미지(Chartjunk)가 핵심 데이터 구간을 가려 정확한 정보 판독을 방해하고 있습니다."
    },

    // 🚨 5번 구역: 국민일보 문재인 대통령 지지율 (축 절단)
    {
        keyword: "201709082334", // HTML의 mock_id와 동일
        is_misleading: true,
        confidence: "매우 높음",
        quiz_question: "이 차트의 세로축(Y축)은 0%가 아닌 65% 부근부터 시작하여, 실제 3~4%p의 하락 폭을 시각적으로 과장하고 있다? (O/X)",
        quiz_answer: "O",
        quiz_explanation: "Y축의 시작점을 자르는 '축 절단(Truncated Axis)' 기법입니다. 0부터 시작하지 않으면 작은 변화도 마치 폭락한 것처럼 보이게 만들어 시청자의 착시를 강하게 유발합니다."
    },

    // 🚨 6번 구역: 면세점 품목별 판매 규모 (막대 비례 왜곡 및 배경 편향)
    {
        keyword: "14907_15803_210", // HTML의 mock_id와 동일
        is_misleading: true,
        confidence: "높음",
        quiz_question: "이 수평 막대 차트는 실제 데이터 수치와 막대의 길이가 비례하지 않으며, 배경 사진으로 인해 특정 품목에 시각적 편향을 주고 있다? (O/X)",
        quiz_answer: "O",
        quiz_explanation: "막대의 길이가 실제 수치에 맞지 않는 '비례 왜곡(Non-Proportional Bars)' 현상이 있습니다. 또한 화장품 매장 실사 배경(Background Photo Bias)을 사용하여 1위 항목을 과도하게 강조하고 정보의 객관성을 해치고 있습니다."
    }
];