"""
3단계 파싱·정제·청킹: data/corpus/*.txt -> 정제 -> 청킹 2종 -> 통계표·수동 검사표

정제 규칙 (문서 유형별 적용, 규칙서: docs/cleaning_rules.md)
  R1 페이지 번호 줄 삭제        ('- 3 -', '3 / 12', '3' 만 있는 줄)
  R2 반복 머리말·꼬리말 삭제    (PDF: 3쪽 이상, 전체 쪽의 40% 이상에 똑같이 나오는 줄)
  R3 웹페이지 메뉴 문구 삭제    (K-Startup 상세페이지: 공유하기, URL복사, 인쇄, 페이스북 ...)
  R4 글자 사이 띄어쓰기 복구    ('경 상 남 도' -> '경상남도', 표에서 자주 생김)
  R5 특수·제어 문자 삭제        (HWP 사용자 정의 기호, 사용 영역 문자)
  R6 공백·빈 줄 정리

청킹 전략
  A1 고정 길이: 800자, 겹침 100자 (지금 판정 파이프라인 v0~v6이 쓰는 방식)
  A2 조항 단위: □·1.·제N조 같은 큰 제목과 ○·➊·가. 같은 항목 경계에서 자르고, 1,000자를 넘으면 그 안에서만 고정 길이로 나눔.
                 100자 미만 조각은 다음 조각에 붙임. 각 청크에 가장 가까운 큰 제목을 '섹션'으로 붙임

메타데이터 (청크마다): chunk_id, strategy, notice_id, title, source, org, deadline, file, page, section, chunk_index, chars, eval_set

사용법 (govchat 폴더에서):
    python build_chunks.py
결과
  data/chunks/A1.jsonl, data/chunks/A2.jsonl   청크 + 메타데이터
  data/chunks/chunk_stats.md                    청크 통계표 (개수, 평균, 최단, 최장, 중앙값 ...) + 정제 규칙별 삭제 줄 수
  data/chunks/chunk_review.csv                  수동 검사표: 무작위 30개(A1 15 + A2 15), '판정' 칸에 정상/잘림/버림 기입
"""
import csv, json, random, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data")
CORPUS, OUT = DATA / "corpus", DATA / "chunks"
A1_SIZE, A1_OVERLAP = 800, 100
A2_MAX, A2_MIN = 1000, 100
SEED, N_REVIEW = 42, 15

PAGE_NO = re.compile(r"^\s*(-\s*\d{1,3}\s*-|\d{1,3}\s*/\s*\d{1,3}|page\s*\d+)\s*$", re.I)   # 어디서든 페이지 번호로 보는 형식
BARE_NO = re.compile(r"^\s*\d{1,3}\s*$")     # 숫자만 있는 줄: 표 칸일 수 있어서 PDF 쪽의 첫 줄·마지막 줄일 때만 페이지 번호로 봄
WEB_MENU = {"공유하기", "URL복사", "인쇄", "페이스북", "트위터", "카카오톡", "RSS 피드", "스크랩", "닫기", "목록", "바로가기",
            "본문 바로가기", "모집마감", "모집중", "이전글", "다음글", "TOP"}
SPACED = re.compile(r"(?:[가-힣] ){3,}[가-힣]")
H1 = re.compile(r"^\s*(\d{1,2}\s*[.)]?\s+[가-힣A-Za-z]|[□■◆◇▶►]\s*|제\s*\d+\s*조|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\.?\s|\[?\s*붙임\s*\d*)")
H2 = re.compile(r"^\s*([ㅇ○◦●∘◯]\s|[➊-➓❶-❿①-⑳]|[가-하]\.\s|\d{1,2}\)\s|\(\d{1,2}\)\s|[-·•]\s)")
PAGE_MARK = re.compile(r"^\[p\.(\d+)\]$")


# ---------------- 정제 ----------------
def doc_type(text, name):
    if "[p.1]" in text:
        return "PDF"
    if name.startswith("KS-") and ("공유하기" in text or "URL복사" in text):
        return "상세페이지+첨부"
    return "HWP·HWPX·기타"


def clean_v2(text, counter, dtype=""):
    lines = text.split("\n")
    # 쪽 나누기 (PDF만 [p.N] 표시가 있음)
    pages, cur = [], []
    for l in lines:
        if PAGE_MARK.match(l.strip()):
            pages.append(cur); cur = [l]
        else:
            cur.append(l)
    pages.append(cur)
    pages = [p for p in pages if p]
    repeat = set()
    real_pages = [p for p in pages if p and PAGE_MARK.match(p[0].strip())]
    if len(real_pages) >= 3:
        cnt = Counter()
        for p in real_pages:
            cnt.update({l.strip() for l in p[1:] if 2 <= len(l.strip()) <= 60})
        repeat = {l for l, c in cnt.items() if c >= 3 and c >= 0.4 * len(real_pages)}
    edge = set()                                   # PDF 각 쪽의 첫·마지막 내용 줄 위치
    idx, page_lines = 0, []
    for i, l in enumerate(lines):
        if PAGE_MARK.match(l.strip()):
            if page_lines:
                edge.update({page_lines[0], page_lines[-1]})
            page_lines = []
        elif l.strip():
            page_lines.append(i)
    if page_lines and "[p." in text:
        edge.update({page_lines[0], page_lines[-1]})
    out = []
    for i, l in enumerate(lines):
        s = l.strip()
        if PAGE_MARK.match(s):
            out.append(s); continue
        if not s:
            out.append(""); continue
        if PAGE_NO.match(s) or (i in edge and BARE_NO.match(s)):
            counter["R1 페이지 번호"] += 1; continue
        if s in repeat:
            counter["R2 반복 머리말·꼬리말"] += 1; continue
        if dtype == "상세페이지+첨부" and s in WEB_MENU:     # 웹 메뉴 문구는 상세페이지에서만 지움 (PDF의 '목록' 같은 표 칸 보호)
            counter["R3 웹 메뉴 문구"] += 1; continue
        if SPACED.search(s):
            s2 = SPACED.sub(lambda m: m.group(0).replace(" ", ""), s)
            if s2 != s:
                counter["R4 글자 사이 띄어쓰기"] += 1; s = s2
        s2 = re.sub(r"[-\u0000-\u0008\u000b-\u001f]", "", s)
        if s2 != s:
            counter["R5 특수·제어 문자"] += 1; s = s2
        s = re.sub(r"[ \t　]+", " ", s)
        if s:
            out.append(s)
    t = "\n".join(out)
    t2 = re.sub(r"\n{3,}", "\n\n", t)
    if t2 != t:
        counter["R6 빈 줄 정리"] += 1
    return t2.strip()


# ---------------- 청킹 ----------------
def page_of(text, pos):
    ms = list(re.finditer(r"\[p\.(\d+)\]", text[:pos + 8]))
    return int(ms[-1].group(1)) if ms else None


def strip_marks(t):
    return re.sub(r"\[p\.\d+\]\n?", "", t).strip()


def chunk_a1(text):
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + A1_SIZE]
        out.append((i, piece, ""))
        i += A1_SIZE - A1_OVERLAP
    return out


def chunk_a2(text):
    """제목·항목 줄을 경계로 블록을 나누고, A2_MAX를 넘지 않게 이어 붙인다."""
    lines = text.split("\n")
    blocks, cur, start, pos, section = [], [], 0, 0, ""
    for l in lines:
        s = l.strip()
        is_h1, is_h2 = bool(H1.match(s)) and len(s) < 80, bool(H2.match(s))
        if (is_h1 or is_h2) and cur:
            blocks.append((start, "\n".join(cur), section))
            cur, start = [], pos
        if is_h1:
            section = strip_marks(s)[:40]
        cur.append(l)
        pos += len(l) + 1
    if cur:
        blocks.append((start, "\n".join(cur), section))
    out, buf, bstart, bsec = [], "", 0, ""
    for st, b, sec in blocks:
        if len(b) > A2_MAX:                         # 한 항목이 너무 길면 그 안에서만 고정 길이로
            if buf:
                out.append((bstart, buf, bsec)); buf = ""
            i = 0
            while i < len(b):
                out.append((st + i, b[i:i + A1_SIZE], sec)); i += A1_SIZE - A1_OVERLAP
            continue
        if buf and len(buf) + len(b) + 1 > A2_MAX:
            out.append((bstart, buf, bsec)); buf = ""
        if not buf:
            bstart, bsec = st, sec
        buf = (buf + "\n" + b) if buf else b
    if buf:
        out.append((bstart, buf, bsec))
    merged = []                                       # 너무 짧은 조각은 다음 조각에 붙임
    for c in out:
        if merged and len(strip_marks(merged[-1][1])) < A2_MIN:
            p = merged.pop(); merged.append((p[0], p[1] + "\n" + c[1], p[2] or c[2]))
        else:
            merged.append(c)
    if len(merged) > 1 and len(strip_marks(merged[-1][1])) < A2_MIN:   # 마지막 짧은 조각은 앞 조각에 붙임
        last = merged.pop(); p = merged.pop(); merged.append((p[0], p[1] + "\n" + last[1], p[2]))
    return merged


# ---------------- 실행 ----------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {r["notice_id"]: r for r in csv.DictReader(open(DATA / "notices_merged.csv", encoding="utf-8-sig"))} \
        if (DATA / "notices_merged.csv").exists() else {}
    inv = {r["notice_id"]: r for r in csv.DictReader(open(DATA / "data_inventory.csv", encoding="utf-8-sig"))}
    use = {k for k, r in inv.items() if r.get("status") == "사용"}   # 목록표에서 '사용'인 공고만 (지난 실행의 남은 파일 무시)
    files = sorted(f for f in CORPUS.glob("*.txt") if f.stem in use)
    rule_cnt = defaultdict(Counter)
    chunks = {"A1": [], "A2": []}
    for f in files:
        nid = f.stem
        raw = f.read_text(encoding="utf-8")
        dt = doc_type(raw, nid)
        text = clean_v2(raw, rule_cnt[dt], dt)
        m, iv = meta.get(nid, {}), inv.get(nid, {})
        period = m.get("period", "")
        dl = re.findall(r"\d{4}[-.]?\d{2}[-.]?\d{2}", period)
        base = {"notice_id": nid, "title": m.get("title") or iv.get("title", ""), "source": iv.get("source", ""),
                "org": m.get("org", ""), "deadline": dl[-1] if dl else "", "file": iv.get("body_files", ""),
                "eval_set": iv.get("eval_set", ""), "doc_type": dt}
        for name, fn in (("A1", chunk_a1), ("A2", chunk_a2)):
            for k, (st, piece, sec) in enumerate(fn(text)):
                body = strip_marks(piece)
                if not body:
                    continue
                chunks[name].append({**base, "chunk_id": f"{name}-{nid}-{k:03d}", "strategy": name,
                                     "page": page_of(text, st) or "", "section": sec, "chunk_index": k,
                                     "chars": len(body), "text": body})
    for name, cs in chunks.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 통계표
    L = ["# 청크 통계표\n", f"코퍼스: 공고 {len(files)}개 (data/corpus)\n",
         "| 전략 | 청크 수 | 공고당 평균 | 평균 길이 | 중앙값 | 최단 | 최장 | 10% | 90% | 100자 미만 | 섹션 있음 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, cs in chunks.items():
        ls = sorted(c["chars"] for c in cs)
        q = lambda p: ls[min(len(ls) - 1, int(len(ls) * p))]
        L.append(f"| {name} | {len(cs):,} | {len(cs) / max(1, len(files)):.1f} | {statistics.mean(ls):.0f} | {statistics.median(ls):.0f} | "
                 f"{ls[0]} | {ls[-1]} | {q(.1)} | {q(.9)} | {sum(x < 100 for x in ls)} | {sum(bool(c['section']) for c in cs):,} |")
    L += ["\n## 정제 규칙별 삭제·수정 줄 수 (문서 유형별)\n", "| 규칙 | " + " | ".join(rule_cnt) + " |", "|---|" + "---|" * len(rule_cnt)]
    for r in sorted({k for c in rule_cnt.values() for k in c}):
        L.append(f"| {r} | " + " | ".join(str(rule_cnt[d][r]) for d in rule_cnt) + " |")
    L.append("\n| 문서 유형 | 공고 수 |\n|---|---|")
    for d, n in Counter(doc_type(f.read_text(encoding="utf-8"), f.stem) for f in files).items():
        L.append(f"| {d} | {n} |")
    (OUT / "chunk_stats.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # 수동 검사표 (무작위)
    rng = random.Random(SEED)
    rows = []
    for name, cs in chunks.items():
        for c in rng.sample(cs, min(N_REVIEW, len(cs))):
            t = c["text"]
            hint = []
            if not (H1.match(t) or H2.match(t) or t[:1] in "□○ㅇ◦※[(" ):
                hint.append("문장 중간에서 시작?")
            if not re.search(r"[.다음함임요)\]」]\s*$", t):
                hint.append("문장 중간에서 끝남?")
            if len(re.sub(r"[^가-힣]", "", t)) < 0.3 * len(t):
                hint.append("한글 적음(표·숫자 위주)")
            rows.append({"chunk_id": c["chunk_id"], "strategy": name, "notice_id": c["notice_id"], "title": c["title"][:40],
                         "section": c["section"], "page": c["page"], "chars": c["chars"], "자동 의심": " / ".join(hint),
                         "판정(정상/잘림/버림)": "", "메모": "", "text": t})
    with open(OUT / "chunk_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n저장: {OUT}/A1.jsonl, A2.jsonl, chunk_stats.md, chunk_review.csv ({len(rows)}개)")


if __name__ == "__main__":
    main()
