"""
기업마당 지원사업 공고 수집 (기업마당 정책정보 개방 '지원사업정보 API')

준비
  1) pip install requests
  2) collect_kstartup.py 와 같은 폴더의 .env 파일에 한 줄 추가 (기업마당에서 받은 인증키):
         BIZINFO_KEY=여기에_기업마당_인증키
  3) 기업마당은 사용 신청 때 적은 IP에서만 호출될 수 있습니다. 다른 곳에서 실행해 막히면 IP 등록을 확인하세요.

실행
  python collect_bizinfo.py --test      # 5건만 받아 응답 구조 확인 (처음 한 번)
  python collect_bizinfo.py             # 현재 게시된 공고 전체 수집

결과
  data/raw/bizinfo/all.json             API 원본 응답
  data/notices_bizinfo_all.csv          받은 공고 전체 (정리본)
  data/notices_bizinfo.csv              수집 범위만 남긴 목록: (대전 공고 + 전국 대상 공고) 중 창업·소상공인 관련

지역 판단 (기업마당 응답에는 지역 칸이 없어서 추정함)
  - 순서: 공고명 앞 [지역] → 소관 기관이 지자체면 그 지역 → 해시태그(시·도 10개 이상이거나 없으면 전국)
  - 대전이면 포함, 다른 지역이면 제외, 전국이면 포함 (판단 근거는 region_basis 열)
  - 추정이므로 평가셋 공고는 원문으로 지역 조건을 다시 확인할 것
"""
import argparse, csv, html, json, os, re, sys, time
from pathlib import Path

URL = "https://www.bizinfo.go.kr/uss/rss/bizinfoApi.do"
OUT = Path("data")
RAW = OUT / "raw" / "bizinfo"

REGIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
           "충북", "충남", "전북", "전남", "경북", "경남", "제주",
           "충청북도", "충청남도", "전라북도", "전북특별자치도", "전라남도", "경상북도", "경상남도", "강원특별자치도"]
TOPIC_WORDS = ["창업", "스타트업", "소상공인", "자영업", "예비창업", "초기기업", "1인 창조기업", "1인창조기업"]

FIELDS = {  # 정리본 열 이름 : 응답 항목명
    "pblanc_id": "pblancId",
    "title": "pblancNm",
    "summary": "bsnsSumryCn",
    "period": "reqstBeginEndDe",
    "ministry": "jrsdInsttNm",      # 소관부처·지자체
    "org": "excInsttNm",            # 사업수행기관
    "target": "trgetNm",
    "field": "pldirSportRealmLclasCodeNm",
    "field_sub": "pldirSportRealmMlsfcCodeNm",
    "hashtags": "hashtags",
    "contact": "refrncNm",
    "detail_url": "pblancUrl",
    "created": "creatPnttm",
}


def load_key():
    key = os.environ.get("BIZINFO_KEY")
    env = Path(__file__).with_name(".env")
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("BIZINFO_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("인증키가 없습니다. .env 파일에 BIZINFO_KEY=인증키 를 한 줄 추가해주세요.")
    return key


def clean(v):
    s = html.unescape(str(v or ""))
    s = re.sub(r"<[^>]+>", " ", s)          # 요약에 섞인 HTML 태그 제거
    return re.sub(r"\s+", " ", s).strip()


def normalize(item):
    row = {"source": "기업마당", "notice_id": f"BI-{clean(item.get('pblancId'))}"}
    for col, name in FIELDS.items():
        row[col] = clean(item.get(name))
    if row["detail_url"] and not row["detail_url"].startswith("http"):
        row["detail_url"] = "https://www.bizinfo.go.kr" + ("" if row["detail_url"].startswith("/") else "/") + row["detail_url"]
    # 첨부파일 관련 항목이 있으면 그대로 보관 (API 문서에 따라 이름이 다를 수 있어 '파일' 관련 키를 모두 수집)
    files = {k: clean(v) for k, v in item.items() if re.search(r"(file|flpth|atch)", k, re.I) and v}
    row["file_fields"] = json.dumps(files, ensure_ascii=False) if files else ""
    row["region_basis"] = ""
    row["region_guess"] = guess_region(row)
    row["topic_match"] = "Y" if topic_match(row) else "N"
    return row


LOCAL_GOV = re.compile(r"(특별시|광역시|특별자치시|특별자치도|통합특별시|도)$")
R17 = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
       "충북", "충남", "전북", "전남", "경북", "경남", "제주"]


def guess_region(row):
    """지역 추정. 반환값 예: '전국', '대전', '서울특별시' (근거는 region_basis 열에 기록)
    1) 공고명 앞 [지역] 표시  2) 소관 기관이 지자체면 그 지역
    3) 해시태그의 시·도가 10개 이상(기업마당은 전국 공고에 17개 시·도를 모두 붙임)이거나 없으면 전국"""
    head = re.match(r"\s*\[([^\]]+)\]", row["title"])
    if head:
        row["region_basis"] = "공고명"
        return head.group(1).strip()
    ministry = row["ministry"].strip()
    if LOCAL_GOV.search(ministry):
        row["region_basis"] = "소관기관"
        return ministry
    tags = [t.strip() for t in row["hashtags"].split(",")]
    found = [r for r in R17 if r in tags or ("전남광주" in tags and r in ("전남", "광주"))]
    row["region_basis"] = "해시태그"
    if not found or len(found) >= 10:
        return "전국"
    return "|".join(found)


def in_region_scope(row):
    r = row["region_guess"]
    return r == "전국" or "대전" in r


def topic_match(row):
    text = " ".join([row["title"], row["summary"], row["target"], row["hashtags"], row["field"]])
    return any(w in text for w in TOPIC_WORDS)


def fetch(key, count):
    import requests
    params = {"crtfcKey": key, "dataType": "json", "searchCnt": count}
    for attempt in range(3):
        try:
            res = requests.get(URL, params=params, timeout=60)
            if res.status_code != 200:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text[:300]}")
            data = res.json()
            if isinstance(data, dict) and data.get("reqErr"):
                raise RuntimeError(f"기업마당 API 오류: {data['reqErr']}")
            return data
        except ValueError:
            raise SystemExit(f"JSON이 아닌 응답이 왔습니다(인증키·IP 확인): {res.text[:300]}")
        except Exception as e:
            print(f"  요청 실패({attempt + 1}/3): {e}")
            time.sleep(2 * (attempt + 1))
    sys.exit("API 호출에 계속 실패했습니다. 위 오류 메시지를 확인하세요.")


def items_of(data):
    if isinstance(data, dict):
        for k in ("jsonArray", "items", "item", "data"):
            if isinstance(data.get(k), list):
                return data[k]
    return data if isinstance(data, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--count", type=int, default=3000, help="한 번에 받을 최대 건수")
    a = ap.parse_args()

    key = load_key()
    RAW.mkdir(parents=True, exist_ok=True)

    if a.test:
        data = fetch(key, 5)
        (RAW / "test.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        items = items_of(data)
        print(f"받은 건수: {len(items)}")
        if items:
            print("첫 공고의 항목명:", sorted(items[0].keys()))
            print("첫 공고 정리본:", json.dumps(normalize(items[0]), ensure_ascii=False, indent=2))
        print(f"원본 응답 저장: {RAW / 'test.json'}")
        return

    data = fetch(key, a.count)
    (RAW / "all.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    rows, seen = [], set()
    for it in items_of(data):
        row = normalize(it)
        if row["notice_id"] not in seen:
            seen.add(row["notice_id"]); rows.append(row)
    if len(rows) >= a.count:
        print(f"주의: 받은 건수가 요청 한도({a.count})와 같습니다. --count 를 늘려 다시 실행하세요.")

    cols = ["source", "notice_id"] + list(FIELDS) + ["file_fields", "region_guess", "region_basis", "topic_match"]
    scope = [r for r in rows if in_region_scope(r) and r["topic_match"] == "Y"]
    OUT.mkdir(exist_ok=True)
    for path, rs in ((OUT / "notices_bizinfo_all.csv", rows), (OUT / "notices_bizinfo.csv", scope)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rs)
        print(f"저장: {path} ({len(rs)}건)")
    dj = sum("대전" in r["region_guess"] for r in scope)
    print(f"  범위 내 구성: 대전 {dj}건 / 전국(지역 표시 없음) {len(scope) - dj}건")


if __name__ == "__main__":
    main()
