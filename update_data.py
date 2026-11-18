"""
데이터 전체 갱신 (2단계: 공고가 바뀌면 이 파일 하나만 다시 실행)

순서
  1) collect_kstartup.py          K-Startup 공고 목록 (API)
  2) collect_bizinfo.py           기업마당 공고 목록 (API)
  3) merge_and_download.py        두 목록 합치기(중복 제거) + 기업마당 공고문 내려받기
  4) collect_kstartup_files.py    기업마당에 없는 K-Startup 공고의 상세페이지·첨부파일 내려받기 (스크래핑)
  5) build_corpus.py              텍스트 추출, 스캔본·깨진 문서·서식·중복 거르기, 데이터 목록표 만들기

사용법 (govchat 폴더에서):
    python update_data.py
    python update_data.py --skip-collect      # 이미 받은 파일로 5)만 다시
API 키는 .env 에 (KSTARTUP_KEY, BIZINFO_KEY). 기업마당은 등록한 IP에서만 호출됨.
"""
import argparse, subprocess, sys

STEPS = [
    ["collect_kstartup.py"],
    ["collect_bizinfo.py"],
    ["merge_and_download.py"],
    ["collect_kstartup_files.py", "--only", "data/kstartup_manual.csv"],
    ["build_corpus.py"],
]

ap = argparse.ArgumentParser()
ap.add_argument("--skip-collect", action="store_true")
a = ap.parse_args()
for step in (STEPS[-1:] if a.skip_collect else STEPS):
    print(f"\n===== {' '.join(step)} =====")
    r = subprocess.run([sys.executable] + step)
    if r.returncode != 0:
        sys.exit(f"'{step[0]}' 단계에서 멈춤 (위 메시지 확인). 고친 뒤 다시 실행하면 이미 받은 파일은 건너뜁니다.")
print("\n완료: data/data_summary.md 를 확인하세요.")
