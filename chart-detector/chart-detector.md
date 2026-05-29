# 뉴스 차트 자동 수집 프로젝트

## 프로젝트 요약

Chrome Extension이 뉴스 사이트를 탐색할 때 이미지를 수집하고, FastAPI 서버에서 CNN + 룰 기반 이중 분류로 차트 여부를 판별해 자동 저장하는 시스템이다.

```
Chrome Extension + FastAPI + MobileNetV3-Small(CNN) + OpenCV(Rule)
```

---

## 동작 흐름

```
뉴스 페이지 접속
↓
Chrome Extension이 이미지 URL 탐색 (화이트리스트 사이트만)
↓
background.js → FastAPI /analyze 엔드포인트로 전송
↓
CNN 분류 (MobileNetV3-Small, confidence >= 0.45)  ←  최종 판단
룰 기반 분석 (OpenCV, score >= 3)                 ←  비교 참고용
↓
차트로 판정되면 server/downloads/{사이트명}/ 에 저장
```

---

## 프로젝트 구조

```
chart-detector/
├── extension/
│   ├── manifest.json       # Chrome Extension Manifest V3
│   ├── background.js       # 이미지 URL 수집 및 서버 전송
│   ├── content.js          # 페이지 이동 감지
│   ├── whitelist.js        # 허용 사이트 목록
│   └── utils.js
│
└── server/
    ├── main.py             # FastAPI 서버 (POST /analyze)
    ├── cnn.py              # CNN 분류기 (MobileNetV3-Small)
    ├── detector.py         # 룰 기반 분석기 (OpenCV)
    ├── train_cnn.py        # CNN 모델 학습 스크립트
    ├── test_cnn.py
    ├── test_rule.py
    ├── requirements.txt
    ├── model/
    │   └── chart_classifier.pth   # 학습된 모델 가중치
    ├── downloads/          # 저장된 차트 이미지 (사이트별 폴더)
    └── temp/               # 분석 중 임시 파일
```

---

## 실행 방법

### 1. 환경 설치

```bash
conda activate chartdetector
cd chart-detector/server
pip install -r requirements.txt
```

> PyTorch CPU 전용 설치 시:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 2. 서버 실행

```bash
uvicorn main:app --reload
```

### 3. Chrome Extension 로드

1. Chrome에서 `chrome://extensions` 접속
2. 개발자 모드 활성화
3. `압축해제된 확장 프로그램 로드` 클릭 → `extension/` 폴더 선택

이후 화이트리스트에 등록된 뉴스 사이트를 탐색하면 차트 이미지가 자동으로 `server/downloads/` 에 저장된다.

---

## 모델 학습

### 데이터 준비

```
raw_data/
├── charts/               # 차트 이미지 (Positive) — 사이트별 하위 폴더
└── not_charts/
    └── raw_data/
        └── coco_val/
            └── val2017/  # 일반 이미지 (Negative, COCO val2017)
```

### 학습 실행

```bash
cd chart-detector/server
python train_cnn.py                        # 기본값 (10 epoch)
python train_cnn.py --epochs 20 --lr 5e-5  # 옵션 지정
```

- 베이스 모델: MobileNetV3-Small (ImageNet 사전학습)
- 최적 Recall 기준으로 `model/chart_classifier.pth` 자동 저장
- 클래스 불균형은 WeightedRandomSampler로 처리

---

## 사용 기술

### Chrome Extension (Manifest V3)
- 페이지 이동 감지 및 이미지 URL 수집
- 화이트리스트 기반 사이트 필터링

### FastAPI
- `POST /analyze` — 이미지 URL을 받아 CNN + 룰 기반 분석 결과 반환
- 서버 시작 시 CNN 모델 사전 로드

### CNN 분류기 (`cnn.py`)
- 모델: MobileNetV3-Small (fine-tuned)
- 입력: 224×224 center-crop, ImageNet 정규화
- 임계값: confidence ≥ 0.45 → 차트 판정
- 최종 판단 담당

### 룰 기반 분석기 (`detector.py`)
- Canny 엣지 검출
- HoughLinesP 기반 수평/수직선 탐지
- 컨투어 분석 기반 막대그래프 구조 탐지
- HoughCircles 기반 원형그래프 탐지
- 점수 합산 (≥ 3점 → 차트), CNN 결과와 비교 참고용

---

## API 응답 예시

```json
{
  "success": true,
  "is_chart": true,
  "cnn_confidence": 0.8731,
  "rule_score": 5,
  "rule_is_chart": true,
  "agree": true,
  "saved": "downloads/kbs_co_kr/abc123.jpg"
}
```

| 필드 | 설명 |
|------|------|
| `is_chart` | CNN 최종 판정 결과 |
| `cnn_confidence` | CNN 신뢰도 (0~1) |
| `rule_score` | 룰 기반 점수 |
| `rule_is_chart` | 룰 기반 판정 결과 |
| `agree` | CNN과 룰 기반 판정 일치 여부 |
| `saved` | 저장된 파일 경로 (차트인 경우) |
