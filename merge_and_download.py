"""
두 출처 공고 목록 합치기(중복 제거) + 기업마당 공고문 파일 내려받기

먼저 collect_kstartup.py, collect_bizinfo.py 를 실행해 두 CSV를 만들어 둔 뒤 실행:
    python merge_and_download.py              # 합치기 + 파일 내려받기
    python merge_and_download.py --no-download  # 합치기만

결과
  data/notices_merged.csv           최종 공고 목록 (중복 제거, 출처·지역·파일 여부·이미 본 공고 표시)
  data/dup_review.csv               같은 공고인지 애매한 쌍 (사람이 확인)
  data/files/<notice_id>/...        기업마당 공고문·첨부파일
  data/download_log.csv             내려받기 성공/실패 기록
  data/kstartup_manual.csv          기업마당에 없는 K-Startup 공고 (평가셋에 고르면 상세페이지에서 직접 받기)

주의: 기업마당 파일 링크에는 임시 토큰이 있어 시간이 지나면 만료될 수 있음.
      실패가 많으면 collect_bizinfo.py 를 다시 실행해 링크를 새로 받은 뒤 이 파일을 다시 실행.
"""
import argparse, csv, json, re, time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

DATA = Path("data")
SAME, MAYBE = 0.90, 0.80   # 제목 유사도: 이상이면 같은 공고 / 사이면 사람이 확인

# Toy Test·3장 실측에 쓴 공고 (홀드아웃에서 제외해야 함). 제목에 이 단어가 있으면 표시
SEEN_BEFORE = ["웰컴 스테이", "웰컴스테이", "착한가격업소", "기술교류네트워크", "딥테크 스튜디오",
               "액셀러레이터 연계", "투자연계 창업도약", "해외실증", "푸드테크", "신용보증", "소상공인 정책자금"]


def read(path):
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 수집 코드를 먼저 실행하세요.")
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def norm(t):
    t = re.sub(r"\[[^\]]*\]|「|」|『|』", "", t)
    t = re.sub(r"(재공고|수정|변경|연장|추가|모집|공고|참여기업|참가기업|신청|안내)", "", t)
    return re.sub(r"[^가-힣A-Za-z0-9]", "", t).lower()


def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|]', "_", s).strip()
    return s[:150] or "file"


def file_pairs(row):
    """기업마당 file_fields(JSON)에서 (URL, 파일명) 목록. 여러 파일은 '@'로 이어져 있음."""
    if not row.get("file_fields"):
        return []
    f = json.loads(row["file_fields"])
    pairs = []
    for url_key, name_key, prefix in (("printFlpthNm", "printFileNm", "본문_"), ("flpthNm", "fileNm", "")):
        urls = [u for u in f.get(url_key, "").split("@") if u]
        names = [n for n in f.get(name_key, "").split("@") if n]
        for i, u in enumerate(urls):
            name = names[i] if i < len(names) else f"{url_key}_{i + 1}"
            pairs.append((u, prefix + name))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    bi = read(DATA / "notices_bizinfo.csv")
    ks = read(DATA / "notices_kstartup.csv")
    bi_norm = [(norm(r["title"]), r) for r in bi]

    merged, review, manual = [], [], []
    for r in bi:
        merged.append({
            "notice_id": r["notice_id"], "source": "기업마당", "title": r["title"],
            "region": r["region_guess"], "region_basis": r.get("region_basis", ""),
            "period": r["period"], "ministry": r["ministry"], "org": r["org"], "field": r["field"],
            "detail_url": r["detail_url"], "also_in": "", "files": len(file_pairs(r)),
        })
    by_id = {m["notice_id"]: m for m in merged}

    for k in ks:
        nk = norm(k["title"])
        best, score = None, 0.0
        for nb, b in bi_norm:
            s = SequenceMatcher(None, nk, nb).ratio()
            if s > score:
                best, score = b, s
        if best and score >= SAME:
            by_id[best["notice_id"]]["also_in"] = k["notice_id"]     # 같은 공고 → 기업마당 쪽만 남김
            continue
        if best and score >= MAYBE:
            review.append({"kstartup_id": k["notice_id"], "kstartup_title": k["title"],
                           "bizinfo_id": best["notice_id"], "bizinfo_title": best["title"], "similarity": round(score, 2)})
        merged.append({
            "notice_id": k["notice_id"], "source": "K-Startup", "title": k["title"],
            "region": k["region"] or "전국", "region_basis": "API 지역칸",
            "period": f'{k["start"]} ~ {k["end"]}', "ministry": k["sprv_inst"], "org": k["org"], "field": k["field"],
            "detail_url": k["detail_url"], "also_in": "", "files": 0,
        })
        manual.append({"notice_id": k["notice_id"], "title": k["title"], "detail_url": k["detail_url"]})

    for m in merged:
        m["seen_before"] = "Y" if any(w in m["title"] for w in SEEN_BEFORE) else ""

    def write(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"저장: {path} ({len(rows)}건)")

    write(DATA / "notices_merged.csv", merged)
    write(DATA / "dup_review.csv", review)
    write(DATA / "kstartup_manual.csv", manual)
    dup = sum(bool(m["also_in"]) for m in merged)
    print(f"  기업마당 {len(bi)} + K-Startup {len(ks)} → 중복 {dup}건 제거 → {len(merged)}건 / 이미 본 공고 표시 {sum(bool(m['seen_before']) for m in merged)}건")

    if a.no_download:
        return
    import requests
    log, today = [], datetime.now().strftime("%Y-%m-%d")
    for r in bi:
        folder = DATA / "files" / safe_name(r["notice_id"])
        for url, name in file_pairs(r):
            path = folder / safe_name(name)
            if path.exists() and path.stat().st_size > 0:
                log.append({"notice_id": r["notice_id"], "file": name, "result": "이미 있음", "collected_at": ""}); continue
            try:
                res = requests.get(url, timeout=60)
                ctype = res.headers.get("Content-Type", "")
                if res.status_code != 200 or ("text/html" in ctype and len(res.content) < 20000):
                    raise RuntimeError(f"HTTP {res.status_code}, {ctype}")
                folder.mkdir(parents=True, exist_ok=True)
                path.write_bytes(res.content)
                log.append({"notice_id": r["notice_id"], "file": name, "result": f"성공 {len(res.content) // 1024}KB", "collected_at": today})
            except Exception as e:
                log.append({"notice_id": r["notice_id"], "file": name, "result": f"실패: {e}", "collected_at": today})
            time.sleep(0.5)
    write(DATA / "download_log.csv", log)
    ok = sum(l["result"].startswith(("성공", "이미")) for l in log)
    print(f"  파일 {ok}/{len(log)}개 확보" + ("" if ok == len(log) else " — 실패가 있으면 collect_bizinfo.py 재실행 후 다시 시도"))


if __name__ == "__main__":
    main()
