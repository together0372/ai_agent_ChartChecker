# 뉴스 차트 자동 수집 프로젝트

## 실행 방법

Anaconda Prompt 또는 터미널에서 가상환경을 실행한 뒤:

```bash
conda activate chartdetector
```

server 폴더로 이동 후:

```bash
uvicorn main:app --reload
```

를 실행한다. 이후 Chrome에서:

```text
chrome://extensions
```

접속 → 개발자 모드 활성화 → `압축해제된 확장 프로그램 로드` 클릭 → extension 폴더 선택. 그 다음 뉴스 사이트에 접속하면 이미지가 자동 분석되며 차트 이미지는:

```text
server/downloads/
```

폴더에 자동 저장된다.

## 프로젝트 요약

이 프로젝트는:

```text
Chrome Extension + FastAPI + OpenCV + OCR
```

구조를 이용해서 뉴스 사이트를 탐색할 때 자동으로:

1. 페이지 이동 감지
2. 이미지 수집
3. 차트 여부 분석
4. 차트 이미지 자동 저장

을 수행하는 시스템이다.

---

## 동작 흐름

```text
뉴스 페이지 접속
↓
Chrome Extension이 이미지 탐색
↓
background.js가 FastAPI 서버로 전송
↓
detector.py가 차트 여부 분석
↓
차트면 downloads 폴더에 저장
```

---

# 프로젝트 구조

```text
chart-detector/
├── extension/
│   ├── manifest.json
│   ├── background.js
│   ├── whitelist.js
│   ├── content.js
│   └── utils.js
│
├── server/
│   ├── main.py
│   ├── detector.py
│   ├── requirements.txt
│   ├── downloads/
│   └── temp/
```

---

## 사용 기술

### Chrome Extension

* 페이지 이동 감지
* 이미지 URL 수집
* whitelist 기반 사이트 제한

### FastAPI

* 이미지 분석 요청 처리
* detector.py 실행

### OpenCV

* 선(line) 탐지
* 막대그래프 구조 탐지
* 원형그래프 탐지

### Tesseract OCR

* 숫자
* 퍼센트
* 경제/통계 키워드

---

## 차트 판별 방식

현재 detector.py는 아래 요소들을 종합해서 점수를 계산한다.

* OCR 키워드
* 숫자 비율
* 수평/수직선 개수
* 막대그래프 형태
* 원형그래프 형태

그리고 score 기준 이상이면 차트 이미지로 판정한다.

---
