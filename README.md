# HandPose Recognition (1인칭 실시간 손동작 인식)

이 프로젝트는 **1인칭 시점 손동작 영상**과 **MediaPipe 손 랜드마크**를 함께 활용하여  
**3D CNN (영상 기반) + LSTM (랜드마크 기반) + Attention 융합 모델**을 학습하고,  
실시간으로 손동작을 인식하는 프레임워크입니다.

---

## 📂 프로젝트 구조
handpose-recognition/
│── README.md # 프로젝트 설명
│── requirements.txt # 필요한 라이브러리 목록
│── config.py # 공통 설정
│
├── data/ # 데이터 저장 폴더
│ └── raw/ # 원본 영상 데이터 (클래스별 폴더)
│
├── dataset/
│ └── data_loader.py # 데이터 로딩 & 전처리
│
├── models/
│ ├── attention_layer.py # Attention 레이어 정의
│ ├── multimodal_model.py # (2D CNN + LSTM) 모델
│ └── multimodal_model_3d.py # (3D CNN + LSTM) 모델
│
├── utils/
│ ├── mediapipe_utils.py # MediaPipe 헬퍼 함수
│ ├── visualization.py # 시각화 함수 (학습곡선, 혼동행렬)
│
├── train.py # 모델 학습 스크립트
├── evaluate.py # 모델 평가 (리포트 & confusion matrix 저장)
└── inference.py # 실시간 웹캠 추론

