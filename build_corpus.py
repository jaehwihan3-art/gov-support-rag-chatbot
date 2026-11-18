"""
2단계 데이터 구축: 내려받은 공고 파일 -> 텍스트 코퍼스 + 데이터 목록표

data/files/<공고ID>/ 의 파일을 모두 읽어서
  - 텍스트를 뽑고(pdf·hwp·hwpx·docx·txt), 스캔본·깨진 문서·서식 파일·중복 공고를 걸러내고
  - 쓸 공고의 본문 텍스트를 data/corpus/<공고ID>.txt 로 저장하고
  - 표로 정리한다 (엑셀에서 바로 열림).
여러 번 실행해도 되고, 수집 스크립트를 다시 돌린 뒤 이 파일만 다시 실행하면 목록이 갱신된다.

사용법 (govchat 폴더에서):
    pip install pypdf
    python build_corpus.py

결과
  data/data_inventory.csv         공고 단위 목록: 출처, 제목, 수집일, 본문 파일, 글자 수, 사용/제외, 제외 사유, 평가셋 여부
  data/data_inventory_files.csv   파일 단위 목록: 형식, 쪽수, 글자 수, 한글 비율, 판정
  data/data_summary.md            요약: 출처별 문서 수, 총 글자 수, 사용/제외 수와 사유
  data/corpus/<공고ID>.txt        사용하는 공고의 본문 텍스트
"""
import csv, hashlib, os, re, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import baseline as B          # 텍스트 추출 함수(pdf·hwp·hwpx·docx)를 baseline.py와 같은 것으로 씀

DATA = Path("data")
FILES, CORPUS = DATA / "files", DATA / "corpus"
DOC_EXT = {".pdf", ".hwp", ".hwpx", ".docx", ".txt"}
FORM = re.compile(r"서식|신청서|양식|동의서|서약서|확인서|계획서|체크리스트|명단|리스트|위임장|붙임\s*\d*\s*[_.)]?\s*(신청|제출)")
MIN_CHARS = 300            # 이보다 짧으면 본문으로 보기 어려움
SCAN_CHARS_PER_PAGE = 100  # PDF 쪽당 글자가 이보다 적으면 스캔본(이미지) 의심
MIN_HANGUL = 0.30          # 글자 중 한글 비율이 이보다 낮으면 깨진 문서 의심
OUT_OF_SCOPE = {"행사ㆍ네트워크", "행사·네트워크"}   # K-Startup 분류 중 지원사업이 아닌 것 (행사·설명회 공지)


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if p.exists() else []


def unzip_all(folder):
    """zip 첨부를 풀어 둠 (한글 파일명 cp437 -> cp949 복구)"""
    for z in folder.glob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    try:
                        name = name.encode("cp437").decode("cp949")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    out = folder / Path(name).name
                    if not out.exists():
                        out.write_bytes(zf.read(info))
        except zipfile.BadZipFile:
            pass


def pdf_pages(path):
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def judge_file(path, role):
    """(판정, 사유, 글자 수, 쪽수, 한글 비율, 텍스트)"""
    ext = path.suffix.lower()
    if ext == ".zip":
        return "제외", "압축 파일(풀어서 따로 판정)", 0, None, None, ""
    if ext not in DOC_EXT:
        return "제외", f"읽지 않는 형식({ext or '확장자 없음'})", 0, None, None, ""
    try:
        text = B.clean(B.file_text(str(path)))
    except Exception as e:
        return "제외", f"파싱 실패({type(e).__name__})", 0, None, None, ""
    body = re.sub(r"\[p\.\d+\]", "", text)
    letters = [c for c in body if c.isalpha()]
    hangul = sum("가" <= c <= "힣" for c in letters) / len(letters) if letters else 0
    n = len(re.sub(r"\s", "", body))
    pages = pdf_pages(path) if ext == ".pdf" else None
    if ext == ".pdf" and pages and n / pages < SCAN_CHARS_PER_PAGE:
        return "제외", f"스캔본 의심(쪽당 {n // pages}자)", n, pages, hangul, text
    if n < MIN_CHARS:
        return "제외", f"글자 너무 적음({n}자)", n, pages, hangul, text
    if hangul < MIN_HANGUL:
        return "제외", f"깨진 문서 의심(한글 비율 {hangul:.0%})", n, pages, hangul, text
    if role == "첨부" and FORM.search(path.name):
        return "제외", "서식·신청 양식", n, pages, hangul, text
    return ("사용" if role == "본문" else "보조"), "", n, pages, hangul, text


def collected_dates():
    d = {}
    for log in ("download_log.csv", "download_log_kstartup.csv"):
        for r in read_csv(DATA / log):
            if r.get("collected_at"):
                d[r["notice_id"]] = max(d.get(r["notice_id"], ""), r["collected_at"])
    return d


def main():
    meta = {r["notice_id"]: r for r in read_csv(DATA / "notices_merged.csv")}
    for r in read_csv(DATA / "notices_kstartup.csv"):
        meta.setdefault(r["notice_id"], {"source": "K-Startup", "title": r.get("title", "")})
        meta[r["notice_id"]]["field"] = meta[r["notice_id"]].get("field") or r.get("field", "")
    evalset = {r["notice_id"] for r in read_csv(DATA / "eval_candidates.csv")}
    dates = collected_dates()
    CORPUS.mkdir(parents=True, exist_ok=True)
    for old in CORPUS.glob("*.txt"):                 # 지난 실행 결과를 지우고 새로 만듦 (제외로 바뀐 공고가 남지 않게)
        old.unlink()

    file_rows, notice_rows, seen_hash = [], [], {}
    folders = sorted(p for p in FILES.iterdir() if p.is_dir()) if FILES.exists() else []
    for folder in folders:
        nid = folder.name
        unzip_all(folder)
        files = sorted(p for p in folder.iterdir() if p.is_file())
        judged = []
        for p in files:
            role = "본문" if p.name.startswith("본문_") else "첨부"
            st, why, n, pages, hg, text = judge_file(p, role)
            judged.append((p, role, st, why, n, text))
            file_rows.append({"notice_id": nid, "file": p.name, "role": role, "format": p.suffix.lower().lstrip("."),
                              "pages": pages or "", "chars": n, "hangul_ratio": f"{hg:.2f}" if hg is not None else "",
                              "status": st, "reason": why})
        used = [j for j in judged if j[2] == "사용"]
        if used and all(j[0].name == "본문_상세페이지.txt" for j in used):
            # K-Startup: 상세페이지 글은 요약이라 짧음 -> 이름에 '공고'가 있는 첨부(공고문)가 있으면 함께 씀
            extra = sorted((j for j in judged if j[2] == "보조" and "공고" in j[0].name), key=lambda j: -j[4])
            used = used + extra[:1]
        if not used:   # 본문_ 표시가 없거나 본문이 못 쓰는 파일이면, 이름에 '공고'가 있는 첨부 중 가장 긴 것을 본문으로
            cand = sorted((j for j in judged if j[2] == "보조" and "공고" in j[0].name), key=lambda j: -j[4])
            used = cand[:1]
        m = meta.get(nid, {})
        src = m.get("source") or ("K-Startup" if nid.startswith("KS-") else "기업마당")
        date = dates.get(nid) or (datetime.fromtimestamp(max(p.stat().st_mtime for p in files)).strftime("%Y-%m-%d") if files else "")
        row = {"notice_id": nid, "source": src, "title": m.get("title", ""), "collected_at": date,
               "files": len(files), "body_files": " | ".join(j[0].name for j in used), "chars": sum(j[4] for j in used),
               "status": "사용", "reason": "", "eval_set": "Y" if nid in evalset else ""}
        if m.get("field") in OUT_OF_SCOPE and nid not in evalset:
            row.update(status="제외", reason=f"범위 밖({m.get('field')}: 지원사업 아님)")
        elif not files:
            row.update(status="제외", reason="파일 없음")
        elif not used:
            reasons = Counter(j[3] for j in judged if j[3])
            row.update(status="제외", reason="쓸 수 있는 본문 없음: " + ", ".join(f"{k}({v})" for k, v in reasons.most_common(3)))
        else:
            text = "\n\n".join(j[5] for j in used)
            h = hashlib.md5(re.sub(r"\s", "", text)[:5000].encode()).hexdigest()
            if h in seen_hash:
                row.update(status="제외", reason=f"중복(같은 본문: {seen_hash[h]})")
            else:
                seen_hash[h] = nid
                (CORPUS / f"{nid}.txt").write_text(text, encoding="utf-8")
        notice_rows.append(row)

    def write(path, rows):
        if rows:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(DATA / "data_inventory.csv", notice_rows)
    write(DATA / "data_inventory_files.csv", file_rows)

    use = [r for r in notice_rows if r["status"] == "사용"]
    L = [f"# 데이터 목록 요약 ({datetime.now():%Y-%m-%d %H:%M} 기준)\n",
         "| 출처 | 공고 수 | 사용 | 제외 | 사용 공고 총 글자 수 | 평균 글자 수 | 평가셋 포함 |", "|---|---|---|---|---|---|---|"]
    for src in sorted({r["source"] for r in notice_rows}) + ["합계"]:
        rs = notice_rows if src == "합계" else [r for r in notice_rows if r["source"] == src]
        u = [r for r in rs if r["status"] == "사용"]
        tot = sum(int(r["chars"]) for r in u)
        L.append(f"| {src} | {len(rs)} | {len(u)} | {len(rs) - len(u)} | {tot:,} | {tot // max(1, len(u)):,} | {sum(bool(r['eval_set']) for r in rs)} |")
    L.append("\n## 제외한 공고와 사유\n")
    for r in notice_rows:
        if r["status"] == "제외":
            L.append(f"- {r['notice_id']} {r['title'][:40]}: {r['reason']}")
    L.append("\n## 파일 단위 판정\n")
    L.append("| 판정 | 파일 수 |\n|---|---|")
    for k, v in Counter(r["status"] + (f" ({r['reason'].split('(')[0]})" if r["reason"] else "") for r in file_rows).most_common():
        L.append(f"| {k} | {v} |")
    L.append(f"\n기준: 스캔본 = PDF 쪽당 {SCAN_CHARS_PER_PAGE}자 미만 / 너무 짧음 = {MIN_CHARS}자 미만 / 깨짐 = 한글 비율 {MIN_HANGUL:.0%} 미만 / "
             "서식 = 첨부 중 이름에 서식·신청서·양식·서약서 등 / 중복 = 본문 앞 5,000자가 같음")
    out = "\n".join(L)
    (DATA / "data_summary.md").write_text(out, encoding="utf-8")
    print(out[:1500])
    print(f"\n저장: data/data_inventory.csv, data/data_inventory_files.csv, data/data_summary.md, data/corpus/ ({len(use)}개)")


if __name__ == "__main__":
    main()
