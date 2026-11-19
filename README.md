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
| 6. 개선 | 진행 중 | 실험 로그 `docs/experiment_log.md`, 그래프 `docs/figures/` — 자격 판정 v0 55.4% → v7 79.6%, 질의응답(G2 채점) 58.3% → 83.3% |
| 7. 서비스화 | 예정 | |
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
2. `pip install requests beautifulsoup4 pypdf openai`

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
# 4단계 질의응답 평가
python run_qa_eval.py --strategy A2 --k 5
python compare_graders.py --name A2_k5
```
