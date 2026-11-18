"""
K-Startup 공고문 자동 내려받기 (상세페이지 스크래핑)

K-Startup API는 첨부파일 링크를 주지 않아서, 지금까지는 상세페이지에서 손으로 받았다.
이 스크립트는 data/notices_kstartup.csv 의 detail_url(상세페이지)을 열어
  1) 페이지 본문 텍스트를 data/files/KS-<번호>/본문_상세페이지.txt 로 저장하고
  2) 페이지의 첨부파일(공고문 hwp/hwpx/pdf/docx 등)을 같은 폴더에 내려받는다.
이미 받은 파일은 건너뛰므로 여러 번 실행해도 된다 (공고가 갱신되면 다시 실행).

사용법 (govchat 폴더에서):
    pip install requests beautifulsoup4
    python collect_kstartup_files.py --debug KS-179677   # 한 공고만: 페이지 HTML 저장 + 찾은 링크 출력 (처음 한 번)
    python collect_kstartup_files.py                      # notices_kstartup.csv 전체
    python collect_kstartup_files.py --only data/kstartup_manual.csv   # 기업마당에 없는 공고만

결과: data/files/KS-<번호>/..., data/download_log_kstartup.csv (공고별 성공/실패, 수집일)
"""
import argparse, csv, re, time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, unquote

DATA = Path("data")
UA = {"User-Agent": "Mozilla/5.0 (govchat research crawler; contact via GitHub)"}
DOC_EXT = (".hwp", ".hwpx", ".pdf", ".docx", ".doc", ".zip")
LINK_PAT = re.compile(r"filedown|fileDown|download|afile|atchFile|/file/", re.I)


def safe(s):
    return (re.sub(r'[\\/:*?"<>|]', "_", s).strip() or "file")[:150]


def filename_from(res, fallback):
    cd = res.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    if m:
        name = unquote(m.group(1))
        try:                                     # 서버가 보낸 한글 이름이 latin-1로 읽혀 깨진 경우 복구 (UTF-8 또는 CP949)
            raw = name.encode("latin-1")
        except UnicodeEncodeError:
            return name
        for enc in ("utf-8", "cp949"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return name
    return fallback


def page_links(html, base):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href, onclick = a.get("href") or "", a.get("onclick") or ""
        text = " ".join(a.get_text(" ").split()) or a.get("title") or ""
        target = None
        if LINK_PAT.search(href) or href.lower().endswith(DOC_EXT):
            target = urljoin(base, href)
        elif LINK_PAT.search(onclick):
            m = re.search(r"['\"]([^'\"]*(?:down|file)[^'\"]*)['\"]", onclick, re.I)
            if m:
                target = urljoin(base, m.group(1))
        if target and not target.startswith("javascript"):
            links.append((target, text))
    seen, out = set(), []
    for u, t in links:
        if u not in seen:
            seen.add(u); out.append((u, t))
    # 본문 텍스트: 스크립트·메뉴 제거 후 가장 긴 본문 영역
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    blocks = sorted((b.get_text("\n", strip=True) for b in soup.find_all(["div", "section", "article"])), key=len, reverse=True)
    text = blocks[0] if blocks else soup.get_text("\n", strip=True)
    return out, text


def collect(session, nid, url, debug=False):
    folder = DATA / "files" / safe(nid)
    res = session.get(url, headers=UA, timeout=30)
    res.raise_for_status()
    res.encoding = "utf-8"
    for _ in range(3):        # 마감된 공고는 'result_url' 안내 페이지가 와서 자바스크립트로 다른 주소(bizpbanc-deadline.do)로 넘어감
        m = re.search(r'id="result_url"[^>]*data-url="([^"]+)"', res.text)
        if not m:
            break
        url = urljoin(url, m.group(1).replace("&amp;", "&"))
        res = session.get(url, headers=UA, timeout=30)
        res.raise_for_status()
        res.encoding = "utf-8"
    links, text = page_links(res.text, url)
    if debug:
        Path(f"debug_{nid}.html").write_text(res.text, encoding="utf-8")
        print(f"페이지 HTML 저장: debug_{nid}.html (최종 주소: {url})")
        print(f"본문 텍스트 {len(text):,}자: {text[:120]!r}")
        for u, t in links:
            print(f"  링크: {t[:40]!r} -> {u}")
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    page = folder / "본문_상세페이지.txt"
    if not page.exists() and len(text) >= 200:      # 너무 짧으면 본문을 못 읽은 것이라 저장 안 함
        page.write_text(text, encoding="utf-8")
        rows.append(("본문_상세페이지.txt", f"성공 {len(text):,}자"))
    for i, (u, t) in enumerate(links, 1):
        try:
            r = session.get(u, headers={**UA, "Referer": url}, timeout=60)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200 or ("text/html" in ctype and len(r.content) < 20000):
                raise RuntimeError(f"HTTP {r.status_code}, {ctype}")
            name = safe(filename_from(r, t or f"첨부_{i}"))
            if not Path(name).suffix:
                name += ".bin"
            path = folder / name
            if path.exists() and path.stat().st_size > 0:
                rows.append((name, "이미 있음")); continue
            path.write_bytes(r.content)
            rows.append((name, f"성공 {len(r.content) // 1024}KB"))
        except Exception as e:
            rows.append((t or u, f"실패: {e}"))
        time.sleep(0.7)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=str(DATA / "notices_kstartup.csv"), help="notice_id, detail_url 열이 있는 CSV")
    ap.add_argument("--debug", help="한 공고(예: KS-179677)만 받으며 HTML과 링크를 보여 줌")
    a = ap.parse_args()
    import requests
    rows = list(csv.DictReader(open(a.only, encoding="utf-8-sig")))
    if a.debug:
        rows = [r for r in rows if r["notice_id"] == a.debug] or \
               [{"notice_id": a.debug, "detail_url": f"https://www.k-startup.go.kr/web/contents/bizpbanc-ongoing.do?schM=view&pbancSn={a.debug.split('-')[-1]}"}]
    s = requests.Session()
    log, today = [], datetime.now().strftime("%Y-%m-%d")
    for r in rows:
        try:
            res = collect(s, r["notice_id"], r["detail_url"], debug=bool(a.debug))
        except Exception as e:
            res = [("상세페이지", f"실패: {e}")]
        for name, result in res:
            log.append({"notice_id": r["notice_id"], "file": name, "result": result, "collected_at": today})
        print(f"{r['notice_id']}: " + ", ".join(f"{n[:25]}({x[:10]})" for n, x in res))
        time.sleep(1)
    path = DATA / "download_log_kstartup.csv"
    old = list(csv.DictReader(open(path, encoding="utf-8-sig"))) if path.exists() else []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["notice_id", "file", "result", "collected_at"]); w.writeheader(); w.writerows(old + log)
    ok = sum(l["result"].startswith(("성공", "이미")) for l in log)
    print(f"\n파일 {ok}/{len(log)}개 확보 -> {path}")


if __name__ == "__main__":
    main()
