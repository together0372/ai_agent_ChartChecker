🗺️ ChartQuiz 코드 구조 및 동작 원리 안내서 (v2.0)
이 문서는 ChartQuiz 프로젝트의 최종 아키텍처와 핵심 파일들의 역할을 설명합니다.

🌊 전체 시스템 흐름도 (데이터의 여정)
[프론트] 타겟팅: content.js가 웹페이지 내 기사 본문 영역을 분석하여 차트 이미지를 탐지합니다.

[프론트 ➔ 백엔드] 호출: content.js가 background.js에게 분석 작업을 무전(Message)으로 요청하고, background.js는 이를 받아 FastAPI 서버(127.0.0.1:8000)로 전송합니다. (이 과정은 30초 타임아웃을 방지하기 위한 비동기 이벤트 기반으로 작동합니다.)

[백엔드] 2중 지능형 필터링:

1차 필터(CNN): cnn.py가 이미지를 0.1초 만에 분석하여 '차트가 아니면' 즉시 드랍합니다.

2차 필터(멀티에이전트): 진짜 차트라면 main.py가 LangGraph 워크플로우를 가동합니다.

[백엔드 ➔ 프론트] UI 렌더링: background.js가 역으로 content.js에게 결과를 쏴주면, content.js가 차트 위에 팝업 UI를 생성합니다.

🖥️ 프론트엔드 (크롬 익스텐션) - /extension/
content.js (UI 렌더링 및 통신 제어)

핵심: sendToServer 함수가 이제 chrome.runtime.sendMessage를 통해 background.js와 통신합니다. 서버에서 보내주는 왜곡 데이터(is_misleading)를 받아 showQuizOverlay를 통해 화면에 팝업을 그립니다.

background.js (비동기 통신병)

핵심: 브라우저와 서버 사이의 통신 중개자입니다. fetch API를 사용하여 30초 이상 걸릴 수 있는 AI 연산을 안전하게 기다린 뒤, 분석이 완료되면 chrome.tabs.sendMessage를 통해 결과물을 content.js로 전달합니다.

⚙️ 백엔드 & AI 에이전트 - /server/
main.py (시스템 게이트웨이)

핵심: FastAPI 서버의 엔트리포인트입니다. cnn.py로 1차 필터링 후, 통과한 이미지만 workflow.py의 LangGraph 엔진으로 넘깁니다. 퀴즈 생성 로직(quiz_generator)과 결합하여 최종 응답을 JSON으로 구성합니다.

workflow.py (AI 워크플로우 엔진)

핵심: ChartClassifierAgent와 DistortionDetectorAgent를 노드로 연결합니다. LangGraph를 통해 데이터가 노드를 이동할 때마다 ChartCheckState를 갱신하며 수사망을 좁혀갑니다.

state.py (상태 관리 데이터베이스)

핵심: 에이전트 간 주고받는 모든 데이터의 집합소입니다. chart_image_path부터 분석 증거(visual_errors), 최종 판정(verdict), 퀴즈 정보(quiz_question)까지 모든 흐름을 관리합니다.

/agents/core/ (수사관 에이전트들)

agent.py: 멀티에이전트 시스템의 중심. observer와 debate 노드를 통제합니다.

distortion_detector.py: 차트의 왜곡(Y축 절단, 이중 축 등)을 파헤치는 전문가 에이전트입니다.

lc_tools.py & math_tools.py: AI가 텍스트만 읽는 게 아니라, 실제 픽셀과 데이터를 수학적으로 검증할 때 사용하는 도구 모음입니다.

💡 개발자를 위한 팁
UI 수정: 로딩 배지나 퀴즈 디자인은 content.js의 showQuizOverlay 함수를 수정하세요.

AI 판별력 수정: 1차 판별 모델은 cnn.py, 왜곡 분석 로직은 distortion_detector.py를 살펴보시면 됩니다.

통신 디버깅: 서버 응답이 이상하다면 main.py의 로그를, 프론트에서 통신이 안 된다면 F12 콘솔의 Unchecked runtime.lastError 메시지를 확인하세요.
