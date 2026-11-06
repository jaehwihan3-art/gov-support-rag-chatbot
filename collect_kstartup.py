"""
K-Startup 지원사업 공고 수집 (창업진흥원_K-Startup 조회서비스 / getAnnouncementInformation01)

준비
  1) pip install requests
  2) 이 파일과 같은 폴더에 .env 파일을 만들고 한 줄 작성 (공공데이터포털 '일반 인증키'):
         KSTARTUP_KEY=여기에_인증키
     .env 파일은 다른 사람과 공유하거나 깃허브에 올리지 마세요.

실행
  python collect_kstartup.py --test          # 1페이지만 받아 응답 구조 확인 (처음 한 번)
  python collect_kstartup.py                 # 모집 중 공고 전체 수집

결과
  data/raw/kstartup/page_0001.json ...       API 원본 응답 (재현용)
  data/notices_kstartup_all.csv              받은 공고 전체 (정리본)
  data/notices_kstartup.csv                  수집 범위(대전 + 전국)만 남긴 목록

참고
  - 이 API는 첨부파일(공고문) 링크를 주지 않습니다. 공고문은 detail_url(상세페이지)에서 받습니다.
  - 요청변수는 공공데이터포털 미리보기 형식을 따름: cond[항목명::EQ] / cond[항목명::LIKE] / ::GTE / ::LTE
    예) cond[rcrt_prgs_yn::EQ]=Y (모집 중), cond[supt_regin::LIKE]=대전
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

URL = "https://apis.data.go.kr/B552735/kisedKstartupService01/getAnnouncementInformation01"
OUT = Path("data")
RAW = OUT / "raw" / "kstartup"

FIELDS = {  # 정리본 열 이름 : API 응답 항목명
    "pbanc_sn": "pbanc_sn",
    "title": "biz_pbanc_nm",
    "integrated_yn": "intg_pbanc_yn",
    "integrated_biz_nm": "intg_pbanc_biz_nm",
    "field": "supt_biz_clsfc",
    "region": "supt_regin",
    "start": "pbanc_rcpt_bgng_dt",
    "end": "pbanc_rcpt_end_dt",
    "org": "pbanc_ntrp_nm",
    "sprv_inst": "sprv_inst",
    "dept": "biz_prch_dprt_nm",
    "target": "aply_trgt",
    "biz_years": "biz_enyy",
    "target_age": "biz_trgt_age",
    "target_text": "aply_trgt_ctnt",
    "exclude_text": "aply_excl_trgt_ctnt",   # 문서 표기가 aply_exclt_trgt_ctnt 인 곳도 있어 아래에서 둘 다 확인
    "preference": "prfn_matr",
    "content": "pbanc_ctnt",
    "recruiting": "rcrt_prgs_yn",
    "detail_url": "detl_pg_url",
    "guide_url": "biz_gdnc_url",
    "apply_url": "biz_aply_url",
    "contact": "prch_cnpl_no",
}
ALT_NAMES = {"aply_excl_trgt_ctnt": ["aply_exclt_trgt_ctnt"], "rcrt_prgs_yn": ["Rcrt_prgs_yn"]}


def load_key():
    key = os.environ.get("KSTARTUP_KEY")
    env = Path(__file__).with_name(".env")
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("KSTARTUP_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("인증키가 없습니다. .env 파일에 KSTARTUP_KEY=인증키 를 적어주세요.")
    return key


def find_items(obj):
    """응답 JSON 안에서 공고 목록(dict 리스트)을 찾는다. 응답 구조가 문서와 달라도 동작하도록."""
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and any(k in obj[0] for k in ("biz_pbanc_nm", "pbanc_sn")):
            return obj
        for x in obj:
            r = find_items(x)
            if r is not None:
                return r
    elif isinstance(obj, dict):
        for k in ("data", "items", "item", "body", "response"):
            if k in obj:
                r = find_items(obj[k])
                if r is not None:
                    return r
        if any(k in obj for k in ("biz_pbanc_nm", "pbanc_sn")):
            return [obj]
        for v in obj.values():
            r = find_items(v)
            if r is not None:
                return r
    return None


def total_count(obj):
    for k in ("totalCount", "matchCount"):
        if isinstance(obj, dict) and k in obj:
            return int(obj[k])
    if isinstance(obj, dict):
        for v in obj.values():
            t = total_count(v)
            if t is not None:
                return t
    return None


def get(item, name):
    for n in [name] + ALT_NAMES.get(name, []):
        if item.get(n) not in (None, ""):
            return str(item[n]).strip()
    return ""


def normalize(item):
    row = {"source": "K-Startup", "notice_id": f"KS-{get(item, 'pbanc_sn')}"}
    for col, name in FIELDS.items():
        row[col] = get(item, name)
    if row["detail_url"] and not row["detail_url"].startswith("http"):
        row["detail_url"] = "https://" + row["detail_url"]
    return row


def in_scope(row):
    """수집 범위: 대전 지역 공고 + 전국 대상 공고 (지역이 비어 있으면 전국으로 보고 포함)."""
    r = row["region"]
    return (not r) or ("대전" in r) or ("전국" in r)


def fetch(key, page, per_page, extra):
    import requests
    params = {"serviceKey": key, "page": page, "perPage": per_page, "returnType": "json", **extra}
    for attempt in range(3):
        try:
            res = requests.get(URL, params=params, timeout=30)
            text = res.text
            if res.status_code != 200 or text.lstrip().startswith("<"):
                # 인증키 오류 등은 XML로 오는 경우가 많음
                raise RuntimeError(f"HTTP {res.status_code}: {text[:300]}")
            return json.loads(text)
        except Exception as e:
            print(f"  요청 실패({attempt + 1}/3): {e}")
            time.sleep(2 * (attempt + 1))
    sys.exit("API 호출에 계속 실패했습니다. 위 오류 메시지를 확인하세요 "
             "(SERVICE_KEY_IS_NOT_REGISTERED_ERROR 면 키 반영을 잠시 기다린 뒤 다시 실행).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="1페이지만 받아 응답 구조를 출력")
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--all", action="store_true", help="모집 종료 공고까지 받기 (기본: 모집 중만)")
    a = ap.parse_args()

    key = load_key()
    RAW.mkdir(parents=True, exist_ok=True)
    extra = {} if a.all else {"cond[rcrt_prgs_yn::EQ]": "Y"}

    if a.test:
        data = fetch(key, 1, 5, extra)
        (RAW / "test.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        items = find_items(data) or []
        print(f"전체 건수(응답 기준): {total_count(data)} / 이번에 받은 건수: {len(items)}")
        if items:
            print("첫 공고의 항목명:", sorted(items[0].keys()))
            print("첫 공고 정리본:", json.dumps(normalize(items[0]), ensure_ascii=False, indent=2))
        print(f"원본 응답 저장: {RAW / 'test.json'}  ← 이 파일을 보내주시면 구조를 확인해드릴 수 있어요")
        return

    rows, seen = [], set()
    for page in range(1, a.max_pages + 1):
        data = fetch(key, page, a.per_page, extra)
        (RAW / f"page_{page:04d}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        items = find_items(data) or []
        print(f"page {page}: {len(items)}건")
        new = 0
        for it in items:
            row = normalize(it)
            if row["notice_id"] in seen:
                continue
            seen.add(row["notice_id"]); rows.append(row); new += 1
        if len(items) < a.per_page or new == 0:
            break
        time.sleep(0.3)

    # 모집 중 필터가 서버에서 적용되지 않았을 때를 대비해 한 번 더 거른다
    if not a.all:
        rows = [r for r in rows if r["recruiting"].upper() in ("", "Y")]

    cols = ["source", "notice_id"] + list(FIELDS)
    OUT.mkdir(exist_ok=True)
    for path, rs in ((OUT / "notices_kstartup_all.csv", rows),
                     (OUT / "notices_kstartup.csv", [r for r in rows if in_scope(r)])):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rs)
        print(f"저장: {path} ({len(rs)}건)")


if __name__ == "__main__":
    main()
