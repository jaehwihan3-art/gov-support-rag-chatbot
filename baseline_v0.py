"""
기준선 v0 (고치기 전 점수) - 제안서 5-7 '반복 개선 루프' 1번 단계

구조 (제안서 4장 그대로, 가장 단순한 설정):
  1) 공고 본문(본문_ 파일) -> 텍스트 추출
  2) 고정 길이 청킹 (800자, 겹침 100자)  ※ 청킹 비교 실험의 A1
  3) text-embedding-3-small 임베딩 -> 조건별 고정 질의 8개로 Top-K=5 검색 (공고 하나 안에서만)
  4) GPT-4o-mini가 검색된 청크에서 조건을 JSON으로 추출 (공고당 1회)
  5) Python 코드가 프로필과 비교해 판정 (나이·지역·업종·업력·매출·기업규모는 코드,
     신청 제외 사유·기타 필수 자격은 '사용자 응답 시뮬레이터'가 프로필 문장을 보고 예/아니오/모름으로 답함)
     -> 실제 서비스에서 챗봇이 사용자에게 질문하고 답을 받는 단계를 대신함

사용법 (govchat 폴더에서):
    pip install openai pypdf
    python baseline_v0.py --split dev
    python evaluate.py data/pairs/pairs_dev.csv data/baseline/v0/predictions_dev.csv

- OpenAI 키는 govchat/.env 에  OPENAI_API_KEY=...  로 넣는다 (코드·채팅에 키를 쓰지 않음).
- 단계별 결과를 data/baseline/v0/ 에 캐시한다. 다시 실행하면 끝난 단계는 건너뛴다 (--fresh 로 처음부터).
- 홀드아웃은 개발이 끝난 뒤 한 번만: --split holdout
"""
import argparse, csv, glob, hashlib, json, math, os, re, struct, sys, time, zipfile, zlib, html
from concurrent.futures import ThreadPoolExecutor
from datetime import date

VERSION = "v0"
CFG = {
    "version": VERSION,
    "text_source": "본문_ 파일만 (붙임 제외)",
    "chunking": "고정 길이 800자, 겹침 100자",
    "embedding": "text-embedding-3-small",
    "retrieval": "공고 ID 필터 + 조건별 고정 질의 8개 x Top-K 5 (합집합, 원문 순서)",
    "llm": "gpt-4o-mini, temperature 0, JSON 모드",
    "judge": "Python 규칙 (법률 용어 -> 기업규모 표, 창업기업 84개월 미만)",
    "simulator": "gpt-4o-mini, temperature 0 (신청 제외 사유·기타 필수 자격에 프로필 문장으로 예/아니오/모름 응답)",
    "ref_date": "2026-11-11",
}
CHUNK, OVERLAP, TOPK = 800, 100, 5
QUERIES = [
    "신청 자격 지원 대상 모집 대상",
    "대표자 연령 나이 만 세 이상 이하 출생",
    "지역 소재지 사업장 주소 관할",
    "업종 지원 제외 업종 업종 코드 한국표준산업분류",
    "업력 창업 개업일 사업 개시 년 이내 미만",
    "매출액 연매출 원 이하 미만",
    "신청 제외 대상 지원 제외 참여 제한 체납 휴폐업",
    "필수 요건 자격 요건 증빙 제출 서류",
]
POS, NEG, COND = "신청 가능", "신청 불가", "조건부 가능"
OUT = os.path.join("data", "baseline", VERSION)


# ---------------- 0. 준비 ----------------
def load_env():
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없음: govchat/.env 파일에 OPENAI_API_KEY=... 한 줄을 넣어 주세요.")


_client = None
def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  재시도 {i + 1}: {type(e).__name__} {str(e)[:80]}")
            time.sleep(2 ** i * 2)


def chat_json(system, user):
    def call():
        r = client().chat.completions.create(
            model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return json.loads(r.choices[0].message.content), r.usage
    return retry(call)


def embed(texts):
    out = []
    for i in range(0, len(texts), 100):
        r = retry(lambda: client().embeddings.create(model="text-embedding-3-small", input=texts[i:i + 100]))
        out += [d.embedding for d in r.data]
    return out


# ---------------- 1. 텍스트 추출 ----------------
def hwp_text(path):
    """구형 .hwp (OLE) 본문 텍스트. 외부 라이브러리 없이 읽음."""
    data = open(path, "rb").read()
    u = lambda fmt, a, b: struct.unpack(fmt, data[a:b])[0]
    ss, mss = 1 << u("<H", 30, 32), 1 << u("<H", 32, 34)
    nfat, dirstart, cutoff, mfs = u("<I", 44, 48), u("<I", 48, 52), u("<I", 56, 60), u("<I", 60, 64)
    ds, nd = u("<I", 68, 72), u("<I", 72, 76)
    sec = lambda n: data[512 + n * ss:512 + (n + 1) * ss]
    difat = list(struct.unpack("<109I", data[76:512])); d = ds
    for _ in range(nd):
        v = struct.unpack("<%dI" % (ss // 4), sec(d)); difat += v[:-1]; d = v[-1]
    fat = []
    for n in difat[:nfat]:
        fat += struct.unpack("<%dI" % (ss // 4), sec(n))
    def chain(s):
        o = b""
        while s < 0xFFFFFFFA:
            o += sec(s); s = fat[s]
        return o
    dirs = chain(dirstart); ents = []
    for i in range(len(dirs) // 128):
        e = dirs[i * 128:(i + 1) * 128]; nl = struct.unpack("<H", e[64:66])[0]
        ents.append((e[:nl].decode("utf-16le").rstrip("\x00"), struct.unpack("<I", e[116:120])[0], struct.unpack("<I", e[120:124])[0]))
    mini = chain(ents[0][1]); mf = []
    if mfs < 0xFFFFFFFA:
        b = chain(mfs); mf = list(struct.unpack("<%dI" % (len(b) // 4), b))
    def read(name):
        for n, s, sz in ents:
            if n == name:
                if sz < cutoff:
                    o = b""
                    while s < 0xFFFFFFFA:
                        o += mini[s * mss:(s + 1) * mss]; s = mf[s]
                    return o[:sz]
                return chain(s)[:sz]
    comp = read("FileHeader")[36] & 1; out = []
    for s in sorted([n for n, _, _ in ents if n.startswith("Section")], key=lambda x: int(x[7:])):
        d = read(s)
        if comp:
            d = zlib.decompress(d, -15)
        i = 0
        while i + 4 <= len(d):
            h = struct.unpack("<I", d[i:i + 4])[0]; tag = h & 0x3FF; size = (h >> 20) & 0xFFF; i += 4
            if size == 0xFFF:
                size = struct.unpack("<I", d[i:i + 4])[0]; i += 4
            rec = d[i:i + size]; i += size
            if tag == 67:                       # 문단 텍스트: 제어문자(8글자 확장/인라인)는 건너뜀
                w = struct.unpack("<%dH" % (len(rec) // 2), rec[:len(rec) // 2 * 2])
                j, buf = 0, []
                while j < len(w):
                    c = w[j]
                    if c < 32:
                        if c == 9:
                            buf.append("\t")
                        j += 8 if c in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23) else 1
                        continue
                    buf.append(chr(c)); j += 1
                out.append("".join(buf))
    return "\n".join(out)


def xml_zip_text(path, pattern):
    z = zipfile.ZipFile(path)
    names = sorted(n for n in z.namelist() if re.match(pattern, n))
    parts = []
    for n in names:
        x = z.read(n).decode("utf-8", "ignore")
        x = re.sub(r"</(hp:p|w:p)>", "\n", x)
        x = re.sub(r"<[^>]+>", "", x)
        parts.append(html.unescape(x))
    return "\n".join(parts)


def file_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(f"[p.{i + 1}]\n" + (p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages))
    if ext == ".hwp":
        return hwp_text(path)
    if ext == ".hwpx":
        return xml_zip_text(path, r"Contents/section\d+\.xml")
    if ext == ".docx":
        return xml_zip_text(path, r"word/document\.xml")
    if ext in (".txt", ".md"):
        return open(path, encoding="utf-8", errors="ignore").read()
    raise ValueError(f"읽을 수 없는 형식: {ext}")


def clean(t):
    t = re.sub("[\ud800-\udfff]", "", t)                # PDF에서 깨진 글자(짝 없는 서로게이트) 제거
    t = re.sub(r"[ \t　]+", " ", t)
    t = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", t)       # 페이지 번호 '- 3 -'
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def notice_folders():
    """순번 -> data/files/<notice_id> (폴더 이름은 eval_candidates.csv의 notice_id)"""
    rows = csv.DictReader(open(os.path.join("data", "eval_candidates.csv"), encoding="utf-8-sig"))
    return {f"{int(r['순번']):02d}": os.path.join("data", "files", r["notice_id"]) for r in rows}


# ---------------- 2~3. 청킹·임베딩·검색 ----------------
def chunks_of(t):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i + CHUNK]); i += CHUNK - OVERLAP
    return out


def cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    return s / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


_qvec = None
def retrieve(num, text):
    global _qvec
    path = os.path.join(OUT, "chunks", f"{num}.json")
    if os.path.exists(path):
        cs = json.load(open(path, encoding="utf-8"))
    else:
        parts = chunks_of(text)
        cs = [{"i": i, "text": c, "vec": v} for i, (c, v) in enumerate(zip(parts, embed(parts)))]
        json.dump(cs, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    if _qvec is None:
        _qvec = embed(QUERIES)
    hit = set()
    for qv in _qvec:
        hit |= {c["i"] for c in sorted(cs, key=lambda c: -cos(qv, c["vec"]))[:TOPK]}
    return [cs[i] for i in sorted(hit)], len(cs)


# ---------------- 4. LLM 조건 추출 ----------------
EXTRACT_SYS = """너는 정부 지원사업 공고문에서 신청 자격 조건을 뽑아 JSON으로 만드는 도구다.
주어진 공고문 발췌만 근거로 한다. 발췌에 없는 조건은 만들지 말고 null 또는 빈 배열로 둔다.
규칙:
- 범위는 모두 '이상/이하'로 바꾼다. 예: '40세 미만' -> age.max 39, '7년 미만'(84개월 미만) -> biz_months.max 83, '1억 미만' -> revenue.max 99999999.
- 업력은 개월, 매출은 원 단위 정수(연매출).
- legal_terms: 신청 대상을 정하는 법률 용어를 아래 목록에서만 고른다: 예비창업자, 소상공인, 소기업, 중소기업, 창업기업, 중견기업. 대상이 여럿이면 모두 넣는다. 대상 제한이 없으면 빈 배열.
- region: 신청자 소재지 조건. '대전', '대전 중구'처럼 시도 약칭(+ 구/시). 전국이면 null.
- industry: 허용 업종(allowed_ksic)과 제외 업종(excluded_ksic)을 한국표준산업분류 코드 숫자(앞자리만이어도 됨)로. 코드를 모르면 빈 배열.
- biz_months.founded_after / founded_before: '2026.1.1 이후 창업' 같은 날짜 조건이면 YYYY-MM-DD.
- age.birth_year_min: '1988년 이후 출생'이면 1988.
- exclusions: 하나라도 해당하면 신청할 수 없는 사유 (체납, 휴폐업, 참여 제한 등). 업종·나이·지역처럼 위 칸에 넣은 조건은 다시 넣지 않는다.
- other_requirements: 위 칸으로 표현 못 하는 필수 자격을 예/아니오 질문으로 (예: '운영사로부터 투자를 받았나요?').
- evidence에는 근거 문장을 원문에서 그대로 짧게 옮긴다.
출력 형식:
{"legal_terms": [], "age": {"min": null, "max": null, "birth_year_min": null} 또는 null,
 "region": [] 또는 null, "industry": {"allowed_ksic": [], "excluded_ksic": []} 또는 null,
 "biz_months": {"min": null, "max": null, "founded_after": null, "founded_before": null} 또는 null,
 "revenue": {"min": null, "max": null} 또는 null,
 "exclusions": [{"text": "", "evidence": ""}], "other_requirements": [{"question": "", "evidence": ""}]}"""


def extract(num, folder):
    path = os.path.join(OUT, "extract", f"{num}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(folder, "본문_*")))
    if not files:
        raise FileNotFoundError(f"{folder}에 본문_ 파일이 없음")
    text = clean("\n\n".join(file_text(f) for f in files))
    open(os.path.join(OUT, "texts", f"{num}.txt"), "w", encoding="utf-8").write(text)
    hits, n_chunks = retrieve(num, text)
    ctx = "\n\n---\n\n".join(c["text"] for c in hits)
    res, usage = chat_json(EXTRACT_SYS, "공고문 발췌:\n\n" + ctx)
    res["_meta"] = {"files": [os.path.basename(f) for f in files], "chars": len(text), "chunks": n_chunks,
                    "retrieved": [c["i"] for c in hits], "tokens": usage.total_tokens}
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


# ---------------- 5. 사용자 응답 시뮬레이터 + 판정 ----------------
SIM_SYS = """너는 지원사업 신청자 역할이다. 아래 '내 정보'만 보고 질문에 답한다.
- 내 정보에 근거가 있으면 '예' 또는 '아니오'.
- '신청 제외 사유: 해당하는 것 없음'이라고 적혀 있으면, 제외 사유 질문은 내 정보의 업종·규모 등으로 판단할 수 있는 것 말고는 '아니오'.
- 내 정보로 알 수 없으면 '모름'. 추측하지 않는다.
출력은 JSON 객체 하나: {"answers": ["예" | "아니오" | "모름", ...]}  (질문 순서대로, 개수 맞춤)"""


def simulate(pid, profile_text, ex, n):
    path = os.path.join(OUT, "sim", f"{pid}.json")
    qs = [f"[신청 제외 사유] 다음에 해당하나요? {e.get('text', '')}" for e in ex.get("exclusions") or []] + \
         [f"[필수 자격] {q.get('question', '')}" for q in ex.get("other_requirements") or []]
    key = hashlib.md5(("\n".join(qs) + profile_text).encode()).hexdigest()
    if os.path.exists(path):
        c = json.load(open(path, encoding="utf-8"))
        if c.get("key") == key:
            return c["answers"], qs
    if not qs:
        ans = []
    else:
        user = "내 정보:\n" + profile_text + "\n\n질문:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(qs))
        res, _ = chat_json(SIM_SYS, user)
        ans = (res.get("answers") or []) + ["모름"] * len(qs)
        ans = ans[:len(qs)]
    json.dump({"key": key, "questions": qs, "answers": ans}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return ans, qs


TERM_SIZES = {"예비창업자": {"예비창업자"}, "소상공인": {"소상공인"}, "소기업": {"소상공인", "소기업(소상공인 아님)"},
              "중소기업": {"소상공인", "소기업(소상공인 아님)", "중기업"}, "창업기업": {"소상공인", "소기업(소상공인 아님)", "중기업"},
              "중견기업": {"중견기업"}}
SIDO_FULL = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
             "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원",
             "강원도": "강원", "충청북도": "충북", "충청남도": "충남", "전북특별자치도": "전북", "전라북도": "전북",
             "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"}


def num(v):
    try:
        return None if v in (None, "") else int(float(v))
    except (TypeError, ValueError):
        return None


def as_list(v):
    return [v] if isinstance(v, str) else (v or [])


def judge(ex, r, answers, ref):
    fails, unknown = [], []
    ex = {**ex, "legal_terms": as_list(ex.get("legal_terms")), "region": as_list(ex.get("region"))}
    size = r["size"]; pre = size == "예비창업자"

    terms = [t for t in ex.get("legal_terms") or [] if t in TERM_SIZES]
    if terms:
        ok = set().union(*(TERM_SIZES[t] for t in terms))
        if size not in ok:
            fails.append("기업규모")
        if "창업기업" in terms and not pre:          # 창업기업 = 사업 개시 후 7년(84개월) 미만
            m = num(r["biz_months"])
            if m is None:
                unknown.append("창업기업(업력)")
            elif m >= 84:
                fails.append("창업기업(업력)")

    a = ex.get("age") if isinstance(ex.get("age"), dict) else {}
    if any(a.get(k) is not None for k in ("min", "max", "birth_year_min")):
        if not r["birth"]:
            unknown.append("age")
        else:
            b = date.fromisoformat(r["birth"])
            age = ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))
            if (num(a.get("min")) is not None and age < num(a["min"])) or (num(a.get("max")) is not None and age > num(a["max"])):
                fails.append("age")
            if num(a.get("birth_year_min")) is not None and b.year < num(a["birth_year_min"]):
                fails.append("birth_year")

    reg = [x for x in ex.get("region") or [] if x and "전국" not in x]
    if reg:
        ok = False
        for x in reg:
            p = x.replace(",", " ").split()
            sd = SIDO_FULL.get(p[0], p[0][:2])
            if r["sido"] == sd and (len(p) == 1 or r["gu"] == p[1]):
                ok = True
        if not ok:
            fails.append("region")

    ind = ex.get("industry") if isinstance(ex.get("industry"), dict) else {}
    code = re.sub(r"\D", "", r["ksic"])
    al = [re.sub(r"\D", "", str(x)) for x in ind.get("allowed_ksic") or []]
    exl = [re.sub(r"\D", "", str(x)) for x in ind.get("excluded_ksic") or []]
    al, exl = [x for x in al if x], [x for x in exl if x]
    if al and not any(code.startswith(x) for x in al):
        fails.append("industry")
    if any(code.startswith(x) for x in exl):
        fails.append("industry")

    b = ex.get("biz_months") if isinstance(ex.get("biz_months"), dict) else {}
    if not pre and any(b.get(k) not in (None, "") for k in ("min", "max", "founded_after", "founded_before")):
        m = num(r["biz_months"])
        if m is None:
            unknown.append("biz_months")
        else:
            if num(b.get("min")) is not None and m < num(b["min"]):
                fails.append("biz_months")
            if num(b.get("max")) is not None and m > num(b["max"]):
                fails.append("biz_months")
            try:
                op = date.fromisoformat(r["opened"])
                if b.get("founded_after") and op < date.fromisoformat(b["founded_after"]):
                    fails.append("founded_after")
                if b.get("founded_before") and op > date.fromisoformat(b["founded_before"]):
                    fails.append("founded_before")
            except ValueError:
                pass

    rv = ex.get("revenue") if isinstance(ex.get("revenue"), dict) else {}
    if not pre and (num(rv.get("min")) is not None or num(rv.get("max")) is not None):
        v = num(r["revenue"])
        if v is None:
            unknown.append("revenue")
        elif (num(rv.get("min")) is not None and v < num(rv["min"])) or (num(rv.get("max")) is not None and v > num(rv["max"])):
            fails.append("revenue")

    n_ex = len(ex.get("exclusions") or [])
    for i, ans in enumerate(answers):
        kind = "제외" if i < n_ex else "기타"
        if (kind == "제외" and ans == "예") or (kind == "기타" and ans == "아니오"):
            fails.append(f"{kind}{i + 1}")
        elif ans not in ("예", "아니오"):
            unknown.append(f"{kind}{i + 1}")

    if fails:
        return NEG, fails
    if unknown:
        return COND, unknown
    return POS, []


# ---------------- 실행 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--fresh", action="store_true", help="캐시를 지우고 처음부터")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    load_env()
    if a.fresh and os.path.isdir(OUT):
        import shutil; shutil.rmtree(OUT)
    for d in ("texts", "chunks", "extract", "sim"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    json.dump(CFG, open(os.path.join(OUT, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ref = date.fromisoformat(CFG["ref_date"])

    pairs = list(csv.DictReader(open(os.path.join("data", "pairs", f"pairs_{a.split}.csv"), encoding="utf-8-sig")))
    folders = notice_folders()
    nums = sorted({f"{int(r['순번']):02d}" for r in pairs})

    print(f"[1/2] 조건 추출: 공고 {len(nums)}개 (캐시 있으면 건너뜀)")
    exs = {}
    for n in nums:
        try:
            exs[n] = extract(n, folders[n])
            m = exs[n]["_meta"]
            print(f"  {n}: {m['chars']:,}자, 청크 {m['chunks']}개 중 {len(m['retrieved'])}개 사용")
        except Exception as e:
            print(f"  {n}: 실패 - {e}  -> 이 공고의 쌍은 '조건부 가능'으로 처리")
            exs[n] = None

    print(f"[2/2] 판정: 쌍 {len(pairs)}개")
    def one(r):
        n = f"{int(r['순번']):02d}"
        ex = exs[n]
        if ex is None:
            return r, COND, ["추출 실패"], [], []
        try:
            ans, qs = simulate(r["pair_id"], r["profile_text"], ex, n)
        except Exception as e:                  # 한 쌍이 실패해도 전체는 계속
            print(f"  {r['pair_id']}: 응답 시뮬레이터 실패 - {str(e)[:80]}")
            return r, COND, ["시뮬레이터 실패"], [], []
        pred, why = judge(ex, r, ans, ref)
        return r, pred, why, qs, ans
    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(one, pairs))

    pred_path = os.path.join(OUT, f"predictions_{a.split}.csv")
    with open(pred_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["pair_id", "pred"])
        for r, pred, *_ in results:
            w.writerow([r["pair_id"], pred])
    with open(os.path.join(OUT, f"pred_detail_{a.split}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "gold", "pred", "맞음", "gold_reason", "pred_reason", "case_type", "target_condition", "질문과 응답"])
        for r, pred, why, qs, ans in results:
            w.writerow([r["pair_id"], r["gold"], pred, "O" if pred == r["gold"] else "X", r["gold_reason"], "|".join(why),
                        r["case_type"], r["target_condition"], " / ".join(f"{q} -> {x}" for q, x in zip(qs, ans))])
    ok = sum(p == r["gold"] for r, p, *_ in results)
    print(f"\n정답 {ok}/{len(results)} ({ok / len(results):.1%})")
    print(f"저장: {pred_path}, pred_detail_{a.split}.csv, config.json")
    print(f"다음: python evaluate.py data/pairs/pairs_{a.split}.csv {pred_path}")


if __name__ == "__main__":
    main()
