# 평가셋 30개 공고의 파일이 모두 있는지 확인한다.
# 사용법: python check_files.py   (govchat 폴더에서 실행, eval_candidates.csv는 data/ 안에)
import csv, os, zipfile, glob

CAND = "data/eval_candidates.csv"
BASE = "data/files"

ok = 0
rows = list(csv.DictReader(open(CAND, encoding="utf-8-sig")))
for r in rows:
    nid = r["notice_id"]
    folder = os.path.join(BASE, nid)
    # 일괄 다운로드 zip이 들어 있으면 같은 폴더에 풀어 준다
    for z in glob.glob(os.path.join(folder, "*.zip")):
        with zipfile.ZipFile(z) as f:
            for info in f.infolist():
                name = info.filename
                try:  # 한글 파일명 깨짐 복구
                    name = name.encode("cp437").decode("cp949")
                except Exception:
                    pass
                target = os.path.join(folder, os.path.basename(name))
                if name.endswith("/") or os.path.exists(target):
                    continue
                with open(target, "wb") as out:
                    out.write(f.read(info))
        os.remove(z)
        print(f"  {nid}: zip 풀기 완료")
    files = [f for f in os.listdir(folder)] if os.path.isdir(folder) else []
    body = [f for f in files if f.startswith("본문_")]
    if not files:
        state = "없음  <- 받아야 함"
    elif not body:
        state = f"{len(files)}개, 본문_ 표시 없음  <- 공고문 이름 앞에 '본문_' 붙이기"
    else:
        state = f"{len(files)}개 OK"
        ok += 1
    print(f"{r['순번']:>2} {nid:<12} {state}   {r['title'][:30]}")

print(f"\n준비 완료: {ok}/{len(rows)}개")
