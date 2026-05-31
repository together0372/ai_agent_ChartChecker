# 📊 ChartQuiz: 뉴스 차트 왜곡 탐지 및 퀴즈 생성 시스템

Chrome Extension과 FastAPI 서버를 연동하여, 뉴스 사이트를 탐색할 때 차트 이미지를 자동으로 수집하고 **CNN(1차) + 멀티에이전트 LLM(2차)** 기반으로 시각적 왜곡을 탐지하여 사용자에게 교육용 O/X 퀴즈를 띄워주는 시스템입니다.

---

## 🚀 시스템 아키텍처 (동작 흐름)

```text
[프론트엔드 (Chrome Extension)]
  1. 기사 스크롤 중 본문 이미지 탐지 (광고/배너 필터링)
  2. FastAPI 서버로 이미지 URL 전송 및 로딩 스피너 표출 (⚙️ AI 분석 중...)

[백엔드 (FastAPI + PyTorch + LangGraph)]
  3. CNN 모델(MobileNetV3): 이미지가 실제 "차트"인지 1차 판별 (아니면 즉시 드랍)
  4. LangGraph 멀티에이전트: 진짜 차트인 경우, 도구(수치 추출/수학 검증)를 활용해 왜곡 분석
  5. 데이터 저널리스트 에이전트: 왜곡 기반 O/X 퀴즈 및 해설 생성

[프론트엔드 (Chrome Extension)]
  6. 서버 응답을 받아 차트 위에 "👀 숨겨진 진실 퀴즈" 팝업 UI 동적 렌더링

  🛠️ 설치 및 세팅
1. Ollama 설치 및 로컬 LLM 모델 다운로드
우선 Ollama 공식 홈페이지에서 프로그램을 설치한 후, 아래 명령어로 모델을 다운로드합니다.
ollama pull qwen3.5:9b

2. 백엔드 패키지 설치
Python 3.11 이상의 환경(가상환경 권장)에서 필수 패키지를 설치합니다.
cd chart-detector/server
pip install -r requirements.txt

3. 크롬 익스텐션 로드 (프론트엔드)
1) 크롬 주소창에 chrome://extensions/ 입력 후 접속

2) 우측 상단 [개발자 모드] 활성화

3) 좌측 상단 [압축해제된 확장 프로그램 로드] 클릭

4) 프로젝트의 chart-detector/extension 폴더 선택

🏃‍♂️ 사용법
1. FastAPI 서버 실행
VS Code 터미널에서 백엔드 서버를 가동합니다.
cd chart-detector/server
uvicorn main:app --reload

2. 차트 탐지 테스트
화이트리스트에 등록된 뉴스 사이트(SBS, KBS 등) 기사에 접속합니다.

기사 본문의 차트 이미지가 화면에 나타나면, 우측 상단에 ⚙️ AI 분석 중... 로딩 뱃지가 뜹니다.

분석이 완료되면 차트 위에 O/X 퀴즈 팝업이 생성됩니다.

🔍 탐지 가능한 오류 유형 
Y축 절단    Y축을 0이 아닌 값에서 시작해 차이를 과장
이중 축    조작 무관한 두 변수를 이중 Y축으로 설정해 상관관계처럼 눈속임
선택적    강조 특정 값만 색상·주석으로 부각해 결론을 유도
비대칭 눈금    양방향 막대 차트의 좌우 축 스케일 불일치
파이 각도 왜곡    파이/도넛 슬라이스 각도가 실제 데이터 비율과 불일치
데이터 공백    X축 일부 구간(연도 등)을 임의로 생략
로그 축 미표기    로그 스케일임을 표시하지 않아 시각적 오해 유발

⚙️ 주요 설정
server/main.py 및 server/workflow.py 내부에서 아래 항목들을 변경할 수 있습니다.

LLM 모델 변경: llm = ChatOllama(model="gemma4:e4b-it-q4_K_M", ...)

CNN 신뢰도 임계값: cnn.py 내부의 THRESHOLD (기본값 0.45)