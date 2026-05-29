📊 ChartQuiz (뉴스 차트 왜곡 탐지기)
HCI 프로젝트: AI를 활용하여 뉴스 기사 속 왜곡된 차트를 탐지하고, 독자의 인지적 충격(Cognitive Dissonance)을 유도하는 인터랙티브 크롬 익스텐션입니다.

✨ 기획 의도 및 핵심 시나리오 (UX)
단순히 "이 차트는 왜곡되었습니다"라고 알려주는 것을 넘어, [Trigger → Engage → Visual Reveal] 패턴을 통해 사용자가 능동적으로 정보의 진실을 파악하도록 설계되었습니다.

Trigger (탐지): 기사 스크롤 중 차트를 발견하면 시야를 가리지 않는 하단 미니 배너 팝업을 제공합니다.

Engage (참여): 사용자가 스스로 O/X 퀴즈를 풀거나, 퀴즈를 건너뛰고 해설만 볼 수 있는 선택권(Agency)을 부여합니다.

Visual Reveal (충격 및 학습): AI의 확신도를 나타내는 '신호등 UI'와 함께, 시각적 착각과 실제 팩트를 대조하는 날카로운 해설을 제공합니다.

🛠 기술 스택 (Tech Stack)
Frontend: JavaScript (Chrome Extension Manifest V3), HTML/CSS (DOM Manipulation)

Backend: Python, FastAPI, Uvicorn

AI Agent: LangGraph, LangChain, Ollama (Local LLM)

Computer Vision: OpenCV, Tesseract OCR (1차 차트 필터링)

🚀 설치 및 실행 방법 (Getting Started)
프로젝트를 실행하려면 1. AI 모델 세팅, 2. 백엔드 서버 구동, 3. 크롬 익스텐션 설치 세 가지 단계가 필요합니다.

1. AI 환경 세팅 (Ollama)
로컬에서 AI 에이전트를 돌리기 위해 Ollama가 설치되어 있어야 합니다.

Ollama 공식 홈페이지에서 다운로드 및 설치를 진행합니다.

터미널을 열고 프로젝트에서 사용하는 모델을 다운로드합니다.

Bash
# 사용 중인 모델에 맞게 명령어를 입력하세요.
ollama pull gemma4:e4b-it-q4_K_M  
2. 백엔드 서버 구동 (FastAPI)
파이썬 가상환경을 세팅하고 의존성 패키지를 설치한 뒤 서버를 실행합니다.

VS Code 등에서 chart-detector/server 폴더로 이동합니다.

터미널을 열고 아래 명령어를 순서대로 실행합니다.

Bash
# 1. 가상환경 생성 및 활성화 (Windows 기준)
python -m venv venv
.\venv\Scripts\activate

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. FastAPI 서버 실행
uvicorn main:app --reload
💡 터미널에 Application startup complete.가 뜨면 백엔드 준비 완료입니다. 터미널 창은 끄지 말고 계속 켜두세요!

3. 크롬 익스텐션(Frontend) 설치
크롬 브라우저를 열고 주소창에 chrome://extensions/를 입력합니다.

우측 상단의 [개발자 모드] 토글을 켭니다.

좌측 상단의 [압축해제된 확장 프로그램 로드] 버튼을 클릭합니다.

프로젝트 폴더 내의 extension 폴더를 통째로 선택하여 로드합니다.

💡 테스트 및 사용 방법
화이트리스트에 등록된 뉴스 사이트(예: SBS 뉴스, KBS 뉴스 등)에 접속합니다.

키보드 F12를 눌러 개발자 도구의 [Console] 탭을 열어두면 통신 상태를 실시간으로 확인할 수 있습니다.

기사를 읽으며 스크롤을 내려 차트 이미지가 화면에 나타나게 합니다.

백엔드 서버가 차트를 분석하는 동안 잠시 대기합니다. (컴퓨터 사양에 따라 10~30초 소요)

분석이 완료되면 화면 하단에 "👀 숨겨진 진실 퀴즈가 있습니다!"라는 미니 배너가 등장합니다.

[O] 버튼을 눌러 퀴즈를 풀거나, [해설 보기] 버튼을 눌러 AI의 분석 결과를 확인하세요!

📁 주요 폴더 구조 및 역할
/server/main.py : 프론트엔드와 통신하는 FastAPI 메인 웹 서버

/server/workflow.py : LangGraph 기반의 AI 에이전트 실행 파이프라인

/server/agents/ : 왜곡 탐지기(distortion_detector.py)와 퀴즈 생성기(quiz_generator.py) 코드가 분리되어 있는 폴더

/extension/content.js : 브라우저 화면에서 차트를 찾고 팝업 UI를 그려주는 핵심 프론트엔드 코드

/extension/background.js : 서버와 익스텐션 간의 비동기 통신(Fetch)을 담당하는 서비스 워커

🚨 트러블슈팅 (Troubleshooting)
Q. 코드를 수정했는데 브라우저 화면이 안 바뀝니다.

A. content.js나 background.js를 수정한 경우, 반드시 chrome://extensions/ 페이지에서 새로고침(🔄) 버튼을 누른 뒤 테스트할 뉴스 기사 페이지도 새로고침(F5) 해야 합니다.

Q. 터미널에 ModuleNotFoundError가 뜹니다.

A. 파이썬 가상환경(venv)이 활성화되어 있는지 확인하고, 현재 위치한 경로가 chart-detector/server 폴더 내부인지 확인하세요.

Q. F12 콘솔에 Failed to fetch 에러가 뜹니다.

A. 파이썬 백엔드 서버(uvicorn)가 꺼져 있거나 에러로 멈춰있는 상태입니다. 백엔드 터미널을 확인하고 서버를 재시작해 주세요.
