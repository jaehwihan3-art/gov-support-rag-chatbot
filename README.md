# 정부 지원사업 가이드 RAG 챗봇

소상공인·초기 창업자가 자신에게 맞는 정부 지원사업을 찾고, 지원 자격을 확인할 수 있도록 돕는 RAG 챗봇 프로젝트입니다.

## 프로젝트 개요
- 대상: 소상공인, 초기 창업자
- 기능: 지원사업 가이드, 지원 자격 검증 (원문 근거 제시)
- 데이터 범위: 대전 + 전국 (창업·소상공인 분야)
- 데이터 출처: 기업마당(API), K-Startup(API + 상세페이지 스크래핑)
- 진행 방식: 1인 프로젝트

## 진행 상황
| 단계 | 상태 | 산출물 |
|---|---|---|
| 1. 주제 확정 | 완료 | 제안서(`docs/`), 저장소 |
| 2. 데이터 구축 | 완료 | 공고 208개 수집, 172개 사용 (총 1,003,397자) — `data/data_summary.md`, `data/data_inventory.csv`, `data/data_inventory_files.csv` |
| 3. 파싱·정제·청킹 | 완료 | 정제 규칙서 `docs/cleaning_rules.md`, 청크 통계표 `data/chunks/chunk_stats.md`, 수동 검사표 `data/chunks/chunk_review.xlsx`, 검사 요약 `data/chunks/chunk_review_summary.md` |
| 4. 평가셋 | 완료 | 자격 판정 300쌍 (`data/answer_key/`, `data/pairs/`) + 질의응답 30문항 (`data/eval/qa_eval.csv`), 결과 `docs/qa_eval_report.md` — A2 hit@5 84%, 답변 정확도 61.7%(LLM)/63.3%(사람), LLM·사람 채점 카파 0.44 |
| 5. 오답노트 | 완료 | `docs/error_notes.md` — 판정 오류 74건 원인 11유형(추출 65%, 질문 생성 26%), 질의응답 오류 유형, 개선 우선순위 / 집계 `build_error_notes.py` → `data/error_notes/` |
| 6. 개선 | 완료 | 실험 로그 `docs/experiment_log.md`, 그래프 `docs/figures/` — 자격 판정 v0 → v7: 개발셋 55.4% → 79.6%, 홀드아웃 55.0% → 78.3% / 질의응답(G2 채점) 58.3% → 83.3% |
| 7. 서비스화 | 완료 | Streamlit `app.py` (공고 질문·자격 확인·로그 탭), 로직 `service.py`, SQLite 로그 `data/app_logs.db`, 로그 분석 `analyze_logs.py` → `docs/service_log_report.md` |
| 8. 최종 리포트 | 예정 | |

## 폴더 구조
```
govchat/
├── collect_bizinfo.py         # 기업마당 공고 목록 수집 (API)
├── collect_kstartup.py        # K-Startup 공고 목록 수집 (API)
├── merge_and_download.py      # 두 목록 합치기(중복 제거) + 기업마당 공고문 내려받기
├── collect_kstartup_files.py  # K-Startup 상세페이지·첨부파일 내려받기 (스크래핑)
├── build_corpus.py            # 텍스트 추출, 스캔본·깨진 문서·서식·중복·범위 밖 거르기, 데이터 목록표
├── update_data.py             # 위 2단계 과정을 한 번에 다시 실행
├── build_chunks.py            # 정제(R1~R6) + 청킹 2종(A1 고정 길이, A2 조항 단위) + 통계표·검사표
├── gen_pairs.py, gen_rules.json   # 정답표 × 가상 프로필 → 자격 판정 평가 쌍 300개
├── finalize_keys.py           # 정답표 초안 → 확정본
├── baseline.py                # 판정 파이프라인 (버전 v0~v6)
├── evaluate.py                # 판정 결과 채점 (4분류, FPR, 물러섬 비율)
├── compare_versions.py        # 버전별 핵심 지표 비교표
├── make_qa_draft.py           # 4단계 질의응답 평가셋 30문항 + 근거 검사
├── run_qa_eval.py             # 검색 → 답변 → 채점 (hit@k, 정확도, 인용, 무응답)
├── compare_graders.py         # LLM 채점 vs 사람 채점 (일치율, 카파)
├── build_error_notes.py       # 5단계 오류 원인 유형 집계
├── eval_retrieval.py          # 6단계 검색 방식 비교 (dense·bm25·hybrid)
├── plot_experiments.py        # 6단계 실험 그래프 -> docs/figures/
├── app.py                     # 7단계 Streamlit 화면
├── service.py                 # 7단계 서비스 로직 (질의응답·자격 확인·DB 로그)
├── analyze_logs.py            # 7단계 서비스 로그 분석
├── data/
│   ├── data_inventory*.csv, data_summary.md   # 2단계 데이터 목록표
│   ├── answer_key/            # 공고 30개 정답표 (원문 대조 확정)
│   ├── pairs/                 # 평가 쌍 300개 (개발 240 / 홀드아웃 60)
│   ├── chunks/                # 3단계 청크 통계·검사표
│   └── baseline/              # 버전별 판정 결과
├── docs/                      # 제안서, 정제 규칙서
└── env.example                # 환경변수 예시
```
`data/files`(원본 파일), `data/raw`, `data/corpus`, 청크 jsonl, 시뮬레이터 캐시는 용량 때문에 올리지 않습니다. 스크립트로 다시 만들 수 있습니다.

## 환경 설정
1. `env.example`을 참고해 프로젝트 루트에 `.env` 파일을 만듭니다. (`.env`는 git에 올라가지 않습니다)
   - `KSTARTUP_KEY`: 공공데이터포털 일반 인증키
   - `BIZINFO_KEY`: 기업마당 인증키 (등록한 IP에서만 호출됨)
   - `OPENAI_API_KEY`: 판정 파이프라인용
2. `pip install requests beautifulsoup4 pypdf openai numpy matplotlib streamlit pandas`

## 실행 방법
```bash
# 2단계: 데이터 수집·목록표 (공고가 갱신되면 다시 실행)
python update_data.py
# 3단계: 정제·청킹
python build_chunks.py
# 판정 평가 (개발셋)
python gen_pairs.py
python baseline.py --version v5 --split dev --sim-model gpt-4.1-mini
python evaluate.py data/pairs/pairs_dev.csv data/baseline/v5/predictions_dev.csv --out data/baseline/v5/eval_dev.txt
python compare_versions.py
# 7단계 서비스 실행 (브라우저가 열림)
streamlit run app.py
python analyze_logs.py
# 4단계 질의응답 평가
python run_qa_eval.py --strategy A2 --k 5
python compare_graders.py --name A2_k5
```


## 서비스 (7단계)
- **공고 질문:** hybrid 검색(임베딩 + BM25) → 근거 문장 인용·전제 확인 후 답변(Q1S) → 인용 검증 가드. 답마다 근거 원문과 공고명·쪽수를 보여 준다. 사업을 찾는 질문은 여러 공고에서 찾아 목록으로 답한다. 자격을 묻는 질문에는 단정하지 않고 자격 확인 탭으로 안내한다.
- **자격 확인:** 공고 30개 중 하나를 고르고 내 정보를 입력한 뒤, 공고의 제외 사유·필수 자격 질문에 예/아니오/모름으로 답하면 판정 v7로 신청 가능 / 조건부 가능 / 신청 불가와 걸린 조건(원문 근거)을 보여 준다.
- **로그:** 모든 질문·답변·인용·검색 결과·응답 시간·토큰·피드백, 판정 입력·결과가 `data/app_logs.db`에 쌓인다 (개인 정보가 있을 수 있어 git에 올리지 않음).
- 참고용 서비스다. 최종 자격은 공고 기관에 확인해야 한다.
