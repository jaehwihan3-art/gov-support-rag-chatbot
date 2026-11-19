"""
RAG 판정 파이프라인 (버전별) - 제안서 5-7 '반복 개선 루프'
  v0: 기준선 (고치기 전 점수)
  v1: v0에서 '조건 추출 프롬프트'만 바꿈 (신청자 본인 자격만 뽑기, 칸마다 근거 문장)
      나머지(텍스트·청킹·임베딩·검색·시뮬레이터·판정 코드)는 v0와 같음 -> 점수 차이 = 프롬프트 효과
  v2: v1에서 '업종 제외 표 코드 판정'만 추가
      공고가 '소상공인 정책자금 융자제외 업종', '창업 제외 업종(창업지원법 시행령 제4조)', '사치·향락·사행 업종'을
      가리키면 코드가 업종 코드 표로도 판정 (표 또는 시뮬레이터 중 하나라도 '해당'이면 불가).
      추출·시뮬레이터 결과는 v1 것을 그대로 씀 (추가 비용 없음)
  v3: v2에서 '업종을 코드 대신 이름으로 뽑은 경우' 처리만 추가
      LLM이 excluded_ksic / allowed_ksic에 코드 대신 업종 이름(예: '단순 카페 운영, 도·소매')을 넣으면
      v0~v2 판정 코드는 숫자만 읽어서 그 조건을 통째로 버렸음 -> 이름은 시뮬레이터 질문으로 돌림.
      추출 결과는 v1 것 재사용, 질문이 바뀐 쌍만 시뮬레이터 다시 호출
  v4: v3에서 '조건 추출 프롬프트 2차'만 바꿈 (v3 오답 분석 반영)
      정답에 없는 지역·나이·업력 조건 만들기, '창업기업 또는 일반 기업'을 창업기업으로 좁히기,
      부분 제외 업종('○○ 중 일부')을 코드 전체로 제외하기, 답할 수 없는 질문(기준 없는 질문·안내 문구·거꾸로 묻기) 막기
      -> 결과: 정답 수는 v3와 같은 수준, 지역 오추출은 안 고쳐짐 (채택 안 함)
  v5: v3에서 '숫자·지역 조건 근거 검증(코드)'만 추가 (추출은 v1 결과 그대로 재사용, 추가 LLM 비용 거의 없음)
      LLM이 뽑은 지역·나이·업력·매출 조건이 공고 원문에 실제로 '신청자 조건'으로 적혀 있는지 코드가 확인하고,
      확인이 안 되면 그 조건을 버림 (예: 지역 이름이 기관명·기술 출처·캠퍼스 주소에만 나오면 지역 조건 아님)
  v6: v5에서 '신청 제외 사유 2차 추출'만 추가
      검색 없이 공고 원문 전체를 주고 신청 제외 사유만 한 번 더 뽑아 1차 결과에 합침 (공고당 LLM 1회 추가)
      -> 검색에서 빠지거나 각주·참고에 있던 제외 사유(프랜차이즈, 참여 제한, 고용조정, 중견기업 제외 등)를 줄이려는 것

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
    python baseline.py --version v1 --split dev
    python evaluate.py data/pairs/pairs_dev.csv data/baseline/v1/predictions_dev.csv --out data/baseline/v1/eval_dev.txt

- OpenAI 키는 govchat/.env 에  OPENAI_API_KEY=...  로 넣는다 (코드·채팅에 키를 쓰지 않음).
- 단계별 결과를 data/baseline/v0/ 에 캐시한다. 다시 실행하면 끝난 단계는 건너뛴다 (--fresh 로 처음부터).
- 홀드아웃은 개발이 끝난 뒤 한 번만: --split holdout
"""
import argparse, csv, glob, hashlib, json, math, os, re, struct, sys, time, zipfile, zlib, html
from concurrent.futures import ThreadPoolExecutor
from datetime import date

VERSION = "v0"   # main에서 --version 값으로 바뀜
CFG = {
    "version": VERSION,
    "extract_prompt": "",
    "text_source": "본문_ 파일만 (붙임 제외)",
    "chunking": "고정 길이 800자, 겹침 100자",
    "embedding": "text-embedding-3-small",
    "retrieval": "공고 ID 필터 + 조건별 고정 질의 8개 x Top-K 5 (합집합, 원문 순서)",
    "llm": "gpt-4o-mini, temperature 0, JSON 모드",
    "judge": "Python 규칙 (법률 용어 -> 기업규모 표, 창업기업 84개월 미만)",
    "simulator": "gpt-4o-mini, temperature 0, 질문별 1회 호출 (신청 제외 사유·기타 필수 자격에 프로필 문장으로 예/아니오/모름 응답, 결과는 data/baseline/simcache에 버전 공유)",
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
PROMPT_NOTE = {"v7": "v6 + 조건 성격 검토 단계(원문 전체로 필수/가점/트랙 일부/의무/다른 대상 판정, 필수만 사용) + 업력 예외 질문 + 근로자 나이 검증 보강",
               "v0": "v0 기본 추출 프롬프트",
               "v6": "v5 + 신청 제외 사유만 원문 전체에서 2차 추출해 합침",
               "v5": "v3 + 지역·나이·업력·매출 조건을 원문 문장으로 코드 검증 (근거 없으면 버림)",
               "v4": "v3 + 추출 프롬프트 2차 (지역·나이·업력 오추출, 창업기업 범위, 부분 제외 업종, 답할 수 있는 질문)",
               "v3": "v2 + 업종 이름(코드 아님)으로 뽑힌 허용·제외 업종을 시뮬레이터 질문으로",
               "v2": "v1 프롬프트 그대로 + 업종 제외 표(융자제외·창업제외·사치향락사행) 코드 판정",
               "v1": "v1: 신청자 본인 자격만 (지원 한도·근로자 조건·채용 날짜·우대/가점·세부 트랙 조건 제외), 숫자 칸마다 근거 문장, OR 조건은 질문 하나로"}


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


def chat_json(system, user, model="gpt-4o-mini"):
    def call():
        r = client().chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
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
    shared = os.path.join("data", "baseline", "v0", "chunks", f"{num}.json")   # 청킹·임베딩이 같으면 v0 결과 재사용
    if not os.path.exists(path) and os.path.exists(shared):
        path = shared
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
EXTRACT_SYS_V0 = """너는 정부 지원사업 공고문에서 신청 자격 조건을 뽑아 JSON으로 만드는 도구다.
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


EXTRACT_SYS_V1 = """너는 정부 지원사업 공고문에서 '신청 자격'만 골라 JSON으로 만드는 도구다.
주어진 공고문 발췌만 근거로 한다. 발췌에 없는 조건은 만들지 말고 null 또는 빈 배열로 둔다.

[가장 중요] 신청자(대표자 본인 또는 신청하는 기업)가 '신청할 때' 갖춰야 하는 공통 자격만 뽑는다.
다음은 자격이 아니므로 어떤 칸에도 넣지 않는다:
- 지원금·대출·보증·바우처의 한도나 금액 (예: '최대 7천만원 융자', '업체당 2,500만원 지원', '한도 30백만원')
- 채용할 근로자·참여 인력·대체인력의 나이나 조건 (예: '만 15~34세 청년을 채용', '18세 이상 근로자')
- 채용일·근로계약일·사업 기간·신청 기간·협약 기간 같은 날짜
- 우대·가점·평가·심사 기준 (예: '뿌리산업 영위기업 우대', '후속투자 유치 시 가점')
- 선정 뒤에 해야 하는 의무, 제출 서류 목록
- 여러 세부 자금·트랙 중 일부에만 붙은 조건 (모든 신청자에게 공통인 조건만)
- 운영기관이 재량으로 판단하는 사유 (예: '기타 부적합하다고 인정하는 자', '평가 점수 미달')

칸별 규칙:
- 범위는 모두 '이상/이하'로 바꾼다. 예: '40세 미만' -> 39, '7년 미만' -> biz_months.max 83, '1억 미만' -> revenue.max 99999999.
- legal_terms: 신청 대상을 정하는 법률 용어를 다음 목록에서만 고른다: 예비창업자, 소상공인, 소기업, 중소기업, 창업기업, 중견기업. 목록에 없는 말(청년사업자, 벤처기업 등)은 넣지 않는다. 대상이 여럿이면 모두, 제한이 없으면 빈 배열.
- age: 대표자 본인의 나이 조건일 때만.
- region: 신청 기업의 소재지(본사·사업장) 또는 대표자 주소 조건일 때만. '대전', '대전 중구'처럼 시도 약칭(+구). 전국이면 null. 기술 출처나 앞으로 이전할 예정 같은 조건은 region이 아니라 other_requirements 질문으로.
- industry: 공고에 업종 코드나 업종명이 자격으로 적힌 경우만. 범위('C10~C34')는 두 자리 코드를 하나씩 모두 적는다(10, 11, ..., 34). 코드를 확실히 모르면 코드를 지어내지 말고 exclusions에 문장으로 넣는다.
- biz_months: 업력(창업·개업 후 기간) 조건일 때만, 개월. founded_after / founded_before는 창업일·개업일 조건일 때만 YYYY-MM-DD.
- revenue: 신청 기업의 연매출 조건일 때만, 원 단위 정수.
- 숫자를 넣은 칸(age, biz_months, revenue)마다 evidence에 근거 문장을 원문 그대로 짧게 적는다. 근거 문장을 댈 수 없으면 그 칸은 null.
- exclusions: 하나라도 해당하면 신청할 수 없는 사유 (체납, 휴폐업, 참여 제한 등). 위 칸에 넣은 조건은 다시 넣지 않는다.
- other_requirements: 위 칸으로 표현 못 하는 '필수' 자격을 예/아니오 질문 하나로. 'A 또는 B' 같은 선택 조건은 질문을 나누지 말고 한 질문으로 묻는다 (예: '저신용자이거나 저소득층인가요?').

출력 형식 (JSON 객체 하나):
{"legal_terms": [], "age": {"min": null, "max": null, "birth_year_min": null, "evidence": ""} 또는 null,
 "region": [] 또는 null, "industry": {"allowed_ksic": [], "excluded_ksic": []} 또는 null,
 "biz_months": {"min": null, "max": null, "founded_after": null, "founded_before": null, "evidence": ""} 또는 null,
 "revenue": {"min": null, "max": null, "evidence": ""} 또는 null,
 "exclusions": [{"text": "", "evidence": ""}], "other_requirements": [{"question": "", "evidence": ""}]}"""
EXTRACT_SYS_V4 = EXTRACT_SYS_V1.replace(
    "칸별 규칙:",
    """v1에서 자주 틀린 것 (꼭 지킬 것):
- region: 공고한 기관의 소재지(예: 대전창조경제혁신센터, 대전시가 공고), 입주 공간·센터·캠퍼스의 위치, 이전받을 기술의 출처,
  앞으로 이전할 예정지는 지역 조건이 아니다. 신청 대상 문장에 '○○ 소재 기업', '관내 사업장', '○○에 사업장(본사)을 둔' 처럼
  신청 기업의 현재 소재지를 요구할 때만 region을 채운다. 그 밖에는 null.
- age: '청년 채용', '만 15~34세 근로자', '18세 이상 근로자'처럼 채용·고용·참여 인력의 나이는 대표자 나이가 아니다. 
  '대표자', '사업주', '신청자(본인)'의 나이가 명시될 때만 age를 채운다. 출생연도 조건도 같다.
- biz_months: '창업 후 ○년 이내/미만', '업력 ○년 이상'처럼 신청 기업의 업력이 명시될 때만. 지원 기간·협약 기간·이전 예정 기한·
  '○년 ○월까지 창업 예정' 같은 날짜는 업력이 아니다.
- legal_terms: '창업기업'은 신청 대상이 창업기업으로만 한정될 때만 넣는다. '창업기업 또는 성장기업', '창업기업 및 중소기업'처럼
  일반 기업도 신청할 수 있으면 '창업기업' 대신 '중소기업'을 넣는다.
- industry: '○○업 중 △△'처럼 업종 코드의 일부만 제외하는 경우는 코드를 excluded_ksic에 넣지 말고 exclusions에 문장으로 넣는다.
  excluded_ksic에는 코드 전체가 제외되는 것만, 숫자 코드로만 넣는다 (업종 이름은 넣지 않는다).
- other_requirements 질문은 신청자가 자기 정보로 예/아니오를 답할 수 있어야 한다:
  기준이 되는 숫자(신용점수, 금액, 기간 등)를 질문에 그대로 넣는다 (예: '신용점수가 NICE 600점 이상인가요?').
  '○○를 따를 수 있나요?', '대표자가 참여해야 하나요?', '서류를 제출해야 하나요?' 같은 안내·의무 문구는 질문으로 만들지 않는다.
  우대·가점 항목, 여러 항목 중 '하나라도 해당하면 우선 선발'하는 조건은 필수 자격이 아니다.
- exclusions의 text는 '해당하면 신청 불가'인 상태를 그대로 쓴다 (예: '국세·지방세 체납 중'). '체납 사실이 없나요?'처럼 거꾸로 쓰지 않는다.

칸별 규칙:""")
EXTRACT_SYS = {"v0": EXTRACT_SYS_V0, "v1": EXTRACT_SYS_V1, "v2": EXTRACT_SYS_V1, "v3": EXTRACT_SYS_V1, "v4": EXTRACT_SYS_V4, "v5": EXTRACT_SYS_V1, "v6": EXTRACT_SYS_V1, "v7": EXTRACT_SYS_V1}
BASE = {"v2": "v1", "v3": "v1", "v5": "v1", "v6": "v1", "v7": "v1"}      # 이 버전은 추출·시뮬레이터 결과를 앞 버전 것으로 재사용 (바꾼 건 판정 코드뿐)


def cached(sub, name):
    """OUT/sub/name 경로. 없고 재사용할 앞 버전 결과가 있으면 복사해 온다."""
    p = os.path.join(OUT, sub, name)
    if not os.path.exists(p) and VERSION in BASE:
        b = os.path.join("data", "baseline", BASE[VERSION], sub, name)
        if os.path.exists(b):
            import shutil; shutil.copy(b, p)
    return p


EXCL_SYS = """너는 정부 지원사업 공고문 전체를 읽고 '신청 제외 사유'만 빠짐없이 뽑아 JSON으로 만드는 도구다.
신청 제외 사유 = 신청자(대표자 또는 신청 기업)에게 하나라도 해당하면 신청할 수 없는 상태.
'신청 제외', '지원 제외', '참여 제한', '지원 대상에서 제외', '신청 불가', '지원 불가' 같은 말이 붙은 항목을 모두 찾는다.
본문, 표, 각주(*, ※), 참고·붙임에 있는 것도 포함한다. 예: 체납, 휴·폐업, 참여 제한·제재, 중복 수혜, 프랜차이즈 가맹점,
무점포, 비영리 법인, 대기업·중견기업, 특정 업종(이름 그대로), 사업주 가족 채용, 고용조정 등.
넣지 않는 것: 선정 평가 점수 미달, 운영기관이 재량으로 부적합하다고 판단하는 경우, 허위 서류 제출, 선정 뒤 환수·취소 사유,
지원금 지급 제외 건(일부 건수 제외 등), 여러 세부 자금·트랙 중 일부에만 붙은 제외 사유.
text에는 '해당하면 신청 불가'인 상태를 짧게 쓴다 (예: '국세·지방세 체납 중'). 거꾸로 쓰지 않는다.
evidence에는 근거 문장을 원문에서 짧게 옮긴다.
출력 형식 (JSON 객체 하나): {"exclusions": [{"text": "", "evidence": ""}]}"""


def _norm(t):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", t or "")


def merge_exclusions(first, second):
    """2차로 뽑은 제외 사유 중 1차와 겹치지 않는 것만 붙인다 (글자 겹침으로 판단)."""
    out = list(first or [])
    seen = [_norm(e.get("text", "")) for e in out]
    added = []
    for e in second or []:
        t = _norm(e.get("text", ""))
        if not t:
            continue
        dup = False
        for s_ in seen:
            if not s_:
                continue
            short, long_ = sorted((t, s_), key=len)
            grams = {short[i:i + 2] for i in range(len(short) - 1)}
            if short in long_ or (grams and len([g for g in grams if g in long_]) / len(grams) >= 0.7):
                dup = True
                break
        if not dup:
            out.append({"text": e.get("text", ""), "evidence": e.get("evidence", ""), "source": "2차"})
            seen.append(t); added.append(e.get("text", ""))
    return out, added


def extract_exclusions(num):
    """v6: 원문 전체에서 신청 제외 사유만 다시 뽑는다 (검색 안 씀)."""
    path = os.path.join(OUT, "extract2", f"{num}.json")
    v6 = os.path.join("data", "baseline", "v6", "extract2", f"{num}.json")
    for p_ in (path, v6 if VERSION == "v7" else ""):      # v7은 v6의 2차 추출 결과를 그대로 씀
        if p_ and os.path.exists(p_):
            return json.load(open(p_, encoding="utf-8"))
    text = notice_text(num)
    if not text:
        return {"exclusions": []}
    res, usage = chat_json(EXCL_SYS, "공고문 전체:\n\n" + text[:60000])
    res = {"exclusions": [e for e in res.get("exclusions") or [] if isinstance(e, dict)], "_tokens": usage.total_tokens}
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


def extract(num, folder):
    path = cached("extract", f"{num}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(folder, "본문_*")))
    if not files:
        raise FileNotFoundError(f"{folder}에 본문_ 파일이 없음")
    text = clean("\n\n".join(file_text(f) for f in files))
    open(os.path.join(OUT, "texts", f"{num}.txt"), "w", encoding="utf-8").write(text)
    hits, n_chunks = retrieve(num, text)
    ctx = "\n\n---\n\n".join(c["text"] for c in hits)
    res, usage = chat_json(EXTRACT_SYS[VERSION], "공고문 발췌:\n\n" + ctx)
    res["_meta"] = {"files": [os.path.basename(f) for f in files], "chars": len(text), "chunks": n_chunks,
                    "retrieved": [c["i"] for c in hits], "tokens": usage.total_tokens}
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


# ---------------- 4-2. v7: 조건 성격 검토 ----------------
ROLE_MODEL = "gpt-4.1-mini"
ROLE_SYS = """너는 정부 지원사업 공고문을 검토하는 도구다. 다른 도구가 공고에서 뽑은 [추출된 조건] 목록이 있다.
공고문 전체를 읽고, 각 조건이 원문에서 어떤 성격인지 하나씩 판정한다.

role 값 (하나만):
- "필수": 모든 신청자(대표자 본인 또는 신청 기업)가 신청할 때 갖춰야 하는 조건. 제외 사유라면 해당하면 누구든 신청할 수 없는 사유.
- "일부트랙": 여러 트랙·분야·유형·세부 사업 중 일부에만 붙은 조건. 다른 트랙으로는 이 조건 없이 신청할 수 있다.
  ('창업기업 또는 성장기업 신청 가능'처럼 둘 중 아무거나 되면, 한쪽에만 붙은 업력 조건은 일부트랙이다.)
- "가점우대": 우대, 가점, 우선 선발, 배정 몫(예산을 나눠 주는 구분), 평가·심사 기준.
- "의무절차": 선정 후 지켜야 할 의무, 제출 서류, 신청·접수 방법, 운영 방식 안내.
- "다른대상": 채용할 근로자, 지원 대상 청년, 참여 인력, 대체인력 등 신청자가 아닌 사람의 속성(나이·학력·취업 여부 등)에 걸린 조건.
  단, 신청자가 해야 하는 행위나 신청자 사업장의 상태(예: '청년을 정규직으로 채용했나요?', '4대 보험에 가입했나요?', '근로자를 신규 채용한 사업주')는
  신청자 조건이므로 "필수"다.
- "분류정보": 사이트의 분류 칸(예: K-Startup '창업업력: 1년미만, 2년미만, ... 10년미만')처럼 해당 구간을 나열한 안내. 자격 문장이 아님.
- "근거없음": 원문에서 이 조건을 찾을 수 없음.

추가로 확인할 것:
- biz_exception: 업력 상한에 예외가 있으면(예: '7년 이내(신산업 분야는 최대 10년)', '○○ 수혜기업은 업력 조건 면제') 예외 상한(개월, 면제면 null)과
  신청자에게 물을 예/아니오 질문을 적는다. 예외가 없으면 null.
- pre_founder_allowed: 예비창업자(사업자등록 전)도 신청할 수 있다고 원문에 적혀 있으면 true, 아니면 false.

JSON으로만 답한다:
{"items": [{"key": "목록의 key 그대로", "role": "...", "evidence": "판단 근거 원문(짧게)"}],
 "biz_exception": {"max_months": 119 또는 null, "question": "신산업 분야 기업인가요?"} 또는 null,
 "pre_founder_allowed": true/false}"""
ROLE_DROP_FIELD = {"일부트랙", "가점우대", "의무절차", "다른대상", "분류정보"}


def role_items(ex):
    """검토할 조건 목록 (key, 설명)"""
    items = []
    a = ex.get("age") if isinstance(ex.get("age"), dict) else None
    if a and any(a.get(k) is not None for k in ("min", "max", "birth_year_min")):
        items.append(("age", f"대표자 나이 조건: 최소 {a.get('min')}세, 최대 {a.get('max')}세, 출생연도 {a.get('birth_year_min')} 이후 / 근거: {a.get('evidence', '')}"))
    if as_list(ex.get("region")):
        items.append(("region", f"신청 기업 소재지 조건: {as_list(ex.get('region'))}"))
    ind = ex.get("industry") if isinstance(ex.get("industry"), dict) else None
    if ind and (ind.get("allowed_ksic") or []):
        items.append(("industry", f"허용 업종(이 업종만 신청 가능): {ind.get('allowed_ksic')[:40]}"))
    b = ex.get("biz_months") if isinstance(ex.get("biz_months"), dict) else None
    if b and any(b.get(k) not in (None, "") for k in ("min", "max", "founded_after", "founded_before")):
        items.append(("biz_months", f"업력 조건: 최소 {b.get('min')}개월, 최대 {b.get('max')}개월 / 근거: {b.get('evidence', '')}"))
    rv = ex.get("revenue") if isinstance(ex.get("revenue"), dict) else None
    if rv and any(rv.get(k) is not None for k in ("min", "max")):
        items.append(("revenue", f"매출 조건: 최소 {rv.get('min')}원, 최대 {rv.get('max')}원 / 근거: {rv.get('evidence', '')}"))
    for i, e in enumerate(ex.get("exclusions") or []):
        items.append((f"excl:{i}", f"신청 제외 사유: {e.get('text', '')}"))
    for i, q in enumerate(ex.get("other_requirements") or []):
        items.append((f"other:{i}", f"필수 자격 질문: {q.get('question', '')}"))
    return items


def role_check(num, ex):
    path = os.path.join(OUT, "roles", f"{num}.json")
    ph = hashlib.md5((ROLE_SYS + ROLE_MODEL).encode()).hexdigest()[:8]      # 검토 프롬프트가 바뀌면 다시 검토
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old.get("_prompt") == ph:
            return old
    items = role_items(ex)
    text = notice_text(num) or ""
    if not items or not text:
        res = {"items": [], "biz_exception": None, "pre_founder_allowed": False}
    else:
        lst = "\n".join(f"- key={k}: {d}" for k, d in items)
        res, usage = chat_json(ROLE_SYS, f"[추출된 조건]\n{lst}\n\n[공고문 전체]\n{text[:45000]}", model=ROLE_MODEL)
        res["_tokens"] = usage.total_tokens
    res["_prompt"] = ph
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


def apply_roles(ex, roles):
    """필수가 아닌 조건을 버리고(로그 남김), 업력 예외·예비창업자 허용을 반영"""
    ex = json.loads(json.dumps(ex)); log = []
    role = {str(it.get("key")): it.get("role") for it in roles.get("items") or [] if isinstance(it, dict)}
    for f in ("age", "region", "biz_months", "revenue"):
        if role.get(f) in ROLE_DROP_FIELD:
            ex[f] = None; log.append(f"{f} 버림 ({role[f]})")
    if role.get("industry") in ROLE_DROP_FIELD and isinstance(ex.get("industry"), dict):
        ex["industry"]["allowed_ksic"] = []; log.append(f"허용 업종 버림 ({role['industry']})")
    keep_e = []
    for i, e in enumerate(ex.get("exclusions") or []):
        r_ = role.get(f"excl:{i}")
        if r_ in ROLE_DROP_FIELD:
            log.append(f"제외 사유 버림 ({r_}): {e.get('text', '')[:40]}")
        else:
            keep_e.append(e)
    ex["exclusions"] = keep_e
    keep_q = []
    for i, q in enumerate(ex.get("other_requirements") or []):
        r_ = role.get(f"other:{i}")
        if r_ in ROLE_DROP_FIELD:
            log.append(f"자격 질문 버림 ({r_}): {q.get('question', '')[:40]}")
        else:
            keep_q.append(q)
    ex["other_requirements"] = keep_q
    be = roles.get("biz_exception")
    if isinstance(be, dict) and be.get("question") and isinstance(ex.get("biz_months"), dict) and ex["biz_months"].get("max") is not None:
        ex["_biz_exception"] = {"max_months": num_(be.get("max_months")), "question": be["question"]}
        log.append(f"업력 예외: {be}")
    terms = as_list(ex.get("legal_terms"))
    known = [t for t in terms if t in TERM_SIZES]      # '여성기업'처럼 목록 밖 단어만 있으면 추가하지 않음 (예비창업자만 남는 버그 방지)
    if roles.get("pre_founder_allowed") is True and known and "예비창업자" not in terms:
        ex["legal_terms"] = terms + ["예비창업자"]; log.append("신청 대상에 예비창업자 추가")
    return ex, log


def biz_exception_for(ex, r):
    """업력이 상한을 넘지만 예외 상한 안이면, 상한을 예외값으로 바꾸고 예외 질문을 필수 자격으로 추가"""
    be = ex.get("_biz_exception")
    b = ex.get("biz_months") if isinstance(ex.get("biz_months"), dict) else None
    m = num(r.get("biz_months"))
    if not be or not b or m is None or num(b.get("max")) is None or m <= num(b["max"]):
        return ex
    if be.get("max_months") is not None and m > be["max_months"]:
        return ex
    ex = json.loads(json.dumps(ex))
    ex["biz_months"]["max"] = be.get("max_months")
    if "창업기업" in as_list(ex.get("legal_terms")):           # 창업기업(84개월 미만) 법정 기준도 예외로 풂
        ex["legal_terms"] = list(dict.fromkeys(t if t != "창업기업" else "중소기업" for t in as_list(ex.get("legal_terms"))))
    ex["other_requirements"] = list(ex.get("other_requirements") or []) + [{"question": be["question"], "evidence": "업력 예외"}]
    return ex


# ---------------- 5. 사용자 응답 시뮬레이터 + 판정 ----------------
SIM_SYS = """너는 지원사업 신청자 역할이다. 아래 '내 정보'만 보고 질문 하나에 답한다.
- 질문이 내 정보의 'Q.' 항목과 같은 뜻이면 그 항목의 답(예/아니오)을 그대로 따른다.
- 내 정보에서 바로 알 수 있는 사실로 판단해도 된다 (예: 개업일과 개인/법인사업자가 적혀 있으면 사업자등록이 되어 있음).
- 내 정보에 '공고의 다음 제외 사유에 해당함' 아래 적힌 사유와 질문이 같은 내용이면 '예'.
- 내 정보에 '신청 제외 사유: 해당하는 것 없음' 또는 '그 밖의 신청 제외 사유: 해당하는 것 없음'이 있으면, 신청 제외 사유 질문은
  (재무 상태·대출 이력·과거 수혜처럼 내 정보에 없는 내용이라도) '아니오'로 답한다. 단, 내 정보의 업종·나이·소재지 같은 사실과 직접 맞으면 '예'.
- 내 정보로 알 수 없으면 '모름'. 추측하지 않는다.
출력은 JSON 객체 하나: {"answer": "예" | "아니오" | "모름"}"""
SIM_MODE = "질문별 1회 호출"
SIM_MODEL = "gpt-4o-mini"          # --sim-model 로 바꿈 (평가 장치용 모델: 시스템 점수와 별개)          # 평가 장치: 질문을 하나씩 따로 물어 다른 질문에 끌려 답이 바뀌지 않게 함
SIM_CACHE = os.path.join("data", "baseline", "simcache")   # (질문, 프로필) -> 답. 버전끼리 공유


def ask_one(profile_text, q):
    key = hashlib.md5((SIM_MODE + SIM_MODEL + SIM_SYS + "\n" + q + "\n" + profile_text).encode()).hexdigest()
    path = os.path.join(SIM_CACHE, key[:2], key + ".json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))["answer"]
    res, _ = chat_json(SIM_SYS, "내 정보:\n" + profile_text + "\n\n질문: " + q, model=SIM_MODEL)
    ans = res.get("answer")
    ans = ans if ans in ("예", "아니오", "모름") else "모름"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"question": q, "answer": ans}, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return ans


def simulate(pid, profile_text, ex, n):
    qs = [f"[신청 제외 사유] 다음에 해당하나요? {e.get('text', '')}" for e in ex.get("exclusions") or []] + \
         [f"[필수 자격] {q.get('question', '')}" for q in ex.get("other_requirements") or []]
    ans = [ask_one(profile_text, q) for q in qs]
    json.dump({"mode": SIM_MODE, "questions": qs, "answers": ans},
              open(os.path.join(OUT, "sim", f"{pid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
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


# ---- v2: 업종 제외 표 (공고가 표를 가리키기만 하고 내용은 붙임에 있는 경우를 코드로 판정) ----
# 소상공인 정책자금 융자제외 업종 (별표1 중 코드 전체가 제외되는 것만. '○○ 중' 일부 제외는 판정에 안 씀)
LOAN_EXCL = ["56211", "56212", "46333", "63992", "64", "65", "66", "68", "711", "712", "731", "75993",
             "86", "91113", "91121", "91242", "91249", "91291", "96992"]
LOAN_EXCEPT = ["6821", "68221", "68224", "86902"]          # 표의 단서로 허용되는 업종
# 중소기업창업 지원법 시행령 제4조 창업 제외 업종 (금융·보험, 부동산, 유흥주점, 사행시설, 무도장)
STARTUP_EXCL = ["64", "65", "66", "68", "56211", "56212", "91249", "91291"]
TABLE_RULES = [
    (re.compile(r"창업\s*(에서\s*)?제외|시행령\s*제\s*4\s*조"), STARTUP_EXCL, []),
    (re.compile(r"융자\s*제외|정책자금.{0,10}제외\s*업종|사치|향락|사행|투기"), LOAN_EXCL, LOAN_EXCEPT),
]


def table_hit(text, ksic):
    """제외 사유 문장이 업종 표를 가리키면 (해당 여부 True/False), 아니면 None"""
    code = re.sub(r"\D", "", ksic)
    for rx, table, exc in TABLE_RULES:
        if rx.search(text or ""):
            return any(code.startswith(t) for t in table) and not any(code.startswith(e) for e in exc)
    return None


# ---- v5: 숫자·지역 조건 근거 검증 ----
LOC_KW = re.compile(r"소재|관내|사업장|본사|본점|주사무소|주소|거주|영위|운영\s*중|두고|둔\s|위치한\s*(기업|업체|소상공인)")
LOC_NOT = re.compile(r"대학|출연|연구기관|연구원|센터|캠퍼스|파크|이전\s*(예정|완료|계약)|이전받|이전할|공공기술|임대료|임차|월세|임직원|지점|주거")
WORKER = re.compile(r"채용|근로자|미취업|구직|참여자|인력|직원|청년을|피보험자")
OWNER = re.compile(r"대표|사업주|신청자|본인")
BIZ_KW = re.compile(r"창업|업력|개업|사업\s*(을\s*)?개시|설립|영업")


def windows(text):
    """원문을 줄 단위로 나누고, 표·줄바꿈으로 끊긴 문장을 위해 앞뒤 한 줄씩 붙인 창을 만든다."""
    ls = [l.strip() for l in text.split("\n") if l.strip()]
    return [" ".join(ls[i:i + 2]) for i in range(len(ls))]


ELIG_HEAD = re.compile(r"(신청|지원|모집|참여|입주|입점)\s*(자격|대상|요건)|자격\s*요건|신청\s*제외|지원\s*제외")


def elig_windows(text, span=30):
    """'신청자격·지원대상' 같은 제목 줄부터 아래 span줄까지만 (지역 조건은 이 부분에서만 찾음)"""
    ls = [l.strip() for l in text.split("\n") if l.strip()]
    keep = set()
    for i, l in enumerate(ls):
        if ELIG_HEAD.search(l):
            keep.update(range(i, min(len(ls), i + span)))
    idx = sorted(keep)
    return [" ".join(ls[i:i + 2]) for i in idx]


def notice_text(num):
    for v in (VERSION, BASE.get(VERSION, ""), "v1", "v0"):
        p = os.path.join("data", "baseline", v, "texts", f"{num}.txt") if v else ""
        if p and os.path.exists(p):
            return open(p, encoding="utf-8").read()
    return None


def verify_conditions(ex, num):
    """v5: 지역·나이·업력·매출 조건이 원문에 신청자 조건으로 적혀 있는지 확인. 확인 안 되면 버린다."""
    text = notice_text(num)
    if not text:
        return ex, ["원문 없음: 검증 생략"]
    ex = json.loads(json.dumps(ex)); W = windows(text); E = elig_windows(text) or W; log = []
    # 지역
    reg = as_list(ex.get("region"))
    keep = []
    for x in reg:
        p = str(x).replace(",", " ").split()
        if not p:
            continue
        toks = [SIDO_FULL.get(p[0], p[0][:2])] + p[1:]
        tok = toks[-1]                      # 가장 좁은 단위(구 또는 시도)로 찾음
        ok = any(tok in w and LOC_KW.search(w) and not LOC_NOT.search(w) for w in E)
        (keep.append(x) if ok else log.append(f"지역 '{x}' 버림 (신청자 소재지 문장 없음)"))
    ex["region"] = keep or None
    # 나이
    a = ex.get("age") if isinstance(ex.get("age"), dict) else None
    if a and any(a.get(k) is not None for k in ("min", "max", "birth_year_min")):
        nums = [f"{num_(a.get(k))}세" for k in ("min", "max") if num_(a.get(k)) is not None]
        if a.get("birth_year_min"):
            nums.append(str(num_(a["birth_year_min"])))
        hits = [w for w in W if any(n in w.replace(" ", "") for n in nums)]
        ok = any(OWNER.search(w) or not WORKER.search(w) for w in hits)
        if ok and VERSION == "v7":   # '18세 이상 근로자를 채용한 사업주'처럼 숫자 바로 뒤가 근로자면 사업주가 있어도 근로자 나이
            near = lambda w: any(re.search(re.escape(n) + r".{0,25}?(근로자|청년을|인력|직원|구직자|채용)", w.replace(" ", "")) for n in nums)
            ok = any((OWNER.search(w) or not WORKER.search(w)) and not near(w) for w in hits)
        if not ok:
            ex["age"] = None; log.append(f"나이 {nums} 버림 ({'근로자·채용 대상 나이' if hits else '원문에 없음'})")
    # 업력
    b = ex.get("biz_months") if isinstance(ex.get("biz_months"), dict) else None
    if b:
        for k in ("min", "max"):
            m = num_(b.get(k))
            if m is None:
                continue
            cands = [f"{m}개월"] + ([f"{round(m / 12)}년"] if m >= 12 and (m % 12 in (0, 11)) else []) + ([f"{(m + 1) // 12}년"] if (m + 1) % 12 == 0 else [])
            ok = any(any(c in w.replace(" ", "") for c in cands) and BIZ_KW.search(w) for w in W)
            if not ok:
                b[k] = None; log.append(f"업력 {k}={m} 버림 (업력 문장 없음)")
    # 매출
    rv = ex.get("revenue") if isinstance(ex.get("revenue"), dict) else None
    if rv and any(num_(rv.get(k)) is not None for k in ("min", "max")):
        if not any("매출" in w for w in W):
            ex["revenue"] = None; log.append("매출 조건 버림 (원문에 '매출' 없음)")
    return ex, log


def num_(v):
    try:
        return None if v in (None, "") else int(float(v))
    except (TypeError, ValueError):
        return None


def industry_names_to_questions(ex):
    """v3: 업종 칸에 코드가 아닌 이름이 들어 있으면 시뮬레이터가 답할 질문으로 옮긴다 (코드는 그대로 둠)."""
    ind = ex.get("industry") if isinstance(ex.get("industry"), dict) else None
    if not ind:
        return ex
    is_code = lambda x: bool(re.fullmatch(r"[A-Za-z]?\d{1,5}|[A-Za-z]", str(x).strip()))   # 'C' 같은 대분류 문자도 코드
    ex = json.loads(json.dumps(ex))
    ind = ex["industry"]
    exn = [str(x) for x in ind.get("excluded_ksic") or [] if not is_code(x)]
    aln = [str(x) for x in ind.get("allowed_ksic") or [] if not is_code(x)]
    ind["excluded_ksic"] = [x for x in ind.get("excluded_ksic") or [] if is_code(x)]
    ind["allowed_ksic"] = [x for x in ind.get("allowed_ksic") or [] if is_code(x)]
    # '융자제외 업종에 해당하지 않는 경우'처럼 표를 가리키는 이름은 질문으로 만들지 않고 업종 표로만 판정
    refs = [x for x in exn if any(rx.search(x) for rx, _, _ in TABLE_RULES)]
    exn = [x for x in exn if x not in refs]
    if refs:
        ex["exclusions"] = (ex.get("exclusions") or []) + [{"text": refs[0], "evidence": "업종 칸(표 참조)", "table_only": True}]
    if exn:
        ex["exclusions"] = (ex.get("exclusions") or []) + [{"text": "다음 업종을 영위하는 경우: " + ", ".join(exn), "evidence": "업종 칸(이름)"}]
    if aln:
        ex["other_requirements"] = (ex.get("other_requirements") or []) + [{"question": "다음 업종 중 하나를 영위하나요? " + ", ".join(aln), "evidence": "업종 칸(이름)"}]
    return ex


pairs_first = {}


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

    exl_items = ex.get("exclusions") or []
    n_ex = len(exl_items)
    for i, ans in enumerate(answers):
        kind = "제외" if i < n_ex else "기타"
        if kind == "제외" and VERSION not in ("v0", "v1"):
            hit = table_hit(exl_items[i].get("text", "") + " " + exl_items[i].get("evidence", ""), r["ksic"])
            if hit:                             # 업종 표에 걸리면 코드가 불가로 판정
                fails.append(f"업종표{i + 1}")
                continue
            if exl_items[i].get("table_only"):  # v3: 표 참조로만 만든 항목은 시뮬레이터 답을 안 씀
                continue
            # 표에 안 걸리면 시뮬레이터 답을 그대로 씀 (둘 중 하나라도 '해당'이면 불가: 틀린 '가능'을 줄이는 쪽)
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
    ap.add_argument("--version", default="v1", choices=sorted(EXTRACT_SYS))
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--fresh", action="store_true", help="캐시를 지우고 처음부터")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sim-model", default="gpt-4o-mini", help="사용자 응답 시뮬레이터 모델 (평가 장치)")
    a = ap.parse_args()
    global VERSION, OUT, SIM_MODEL
    VERSION, OUT, SIM_MODEL = a.version, os.path.join("data", "baseline", a.version), a.sim_model
    CFG.update(version=VERSION, extract_prompt=PROMPT_NOTE[VERSION], simulator_model=SIM_MODEL)
    load_env()
    if a.fresh and os.path.isdir(OUT):
        import shutil; shutil.rmtree(OUT)
    for d in ("texts", "chunks", "extract", "sim", "verify", "extract2", "roles"):
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
            if VERSION in ("v6", "v7"):         # 신청 제외 사유 2차 추출 후 합치기
                ex2 = extract_exclusions(n)
                merged, added = merge_exclusions(exs[n].get("exclusions"), ex2.get("exclusions"))
                exs[n] = {**exs[n], "exclusions": merged, "_added_exclusions": added}
                print(f"  {n}: 2차 추출로 제외 사유 {len(added)}개 추가")
            if VERSION == "v7":                 # 조건 성격 검토 -> 필수만 남김
                roles = role_check(n, exs[n])
                exs[n], rlog = apply_roles(exs[n], roles)
                json.dump(rlog, open(os.path.join(OUT, "roles", f"{n}_적용.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                print(f"  {n}: 성격 검토로 {len(rlog)}건 조정")
            m = exs[n]["_meta"]
            print(f"  {n}: {m['chars']:,}자, 청크 {m['chunks']}개 중 {len(m['retrieved'])}개 사용")
        except Exception as e:
            print(f"  {n}: 실패 - {e}  -> 이 공고의 쌍은 '조건부 가능'으로 처리")
            exs[n] = None

    print(f"[2/2] 판정: 쌍 {len(pairs)}개")
    global pairs_first
    pairs_first = {}
    for r in pairs:
        pairs_first.setdefault(f"{int(r['순번']):02d}", r)
    def one(r):
        n = f"{int(r['순번']):02d}"
        ex = exs[n]
        if ex is None:
            return r, COND, ["추출 실패"], [], []
        if VERSION not in ("v0", "v1", "v2"):
            ex = industry_names_to_questions(ex)
        if VERSION in ("v5", "v6", "v7"):
            ex, vlog = verify_conditions(ex, n)
            if r is pairs_first.get(n):
                json.dump(vlog, open(os.path.join(OUT, "verify", f"{n}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if VERSION == "v7":
            ex = biz_exception_for(ex, r)
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
    print(f"다음: python evaluate.py data/pairs/pairs_{a.split}.csv {pred_path} --out {os.path.join(OUT, f'eval_{a.split}.txt')}")


if __name__ == "__main__":
    main()
