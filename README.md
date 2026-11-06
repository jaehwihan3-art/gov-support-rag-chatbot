# 정부 지원사업 가이드 RAG 챗봇

소상공인·초기 창업자가 자신에게 맞는 정부 지원사업을 찾고, 지원 자격을 확인할 수 있도록 돕는 RAG 챗봇 프로젝트입니다.

## 프로젝트 개요
- 대상: 소상공인, 초기 창업자
- 기능: 지원사업 가이드, 지원 자격 검증
- 데이터 범위: 대전 + 전국 (창업·소상공인 분야)
- 데이터 출처: 기업마당, K-Startup
- 진행 방식: 1인 프로젝트

## 폴더 구조
```
govchat/
├── data/                  # 수집한 공고 데이터
├── collect_bizinfo.py     # 기업마당 공고 수집
├── collect_kstartup.py    # K-Startup 공고 수집
├── merge_and_download.py  # 수집 결과 병합 및 저장
├── eval_candidates.csv    # 평가용 정답 후보
├── env.example            # 환경변수 예시
└── docs/                  # 제안서
```

## 환경 설정
1. `env.example`을 참고해 프로젝트 루트에 `.env` 파일을 만듭니다.
2. 아래 키를 발급받아 값을 채웁니다. (`.env`는 git에 올라가지 않습니다)
   - `KSTARTUP_KEY`: 공공데이터포털 일반 인증키
   - `BIZINFO_KEY`: 기업마당 인증키

## 실행 방법
```bash
python collect_bizinfo.py
python collect_kstartup.py
python merge_and_download.py
```

## 진행 계획
1. 공고 수집
2. 정답표 작성
3. 300쌍 정답 분포 확인
4. 기준선(고치기 전 점수) 기록

## 문서
- 제안서: `docs/` 폴더 참고