"""7단계 서비스 로직 (Streamlit 화면과 분리해 테스트하기 쉽게 둠)

- 공고 질문: hybrid 검색(임베딩 + BM25) -> 답변 Q1(근거 인용 먼저, 전제 확인) -> 인용 검증 가드   (6단계 질의응답 최종 설정)
- 자격 확인: 판정 v7 (추출 -> 제외 사유 2차 추출 -> 조건 성격 검토 -> 원문 검증) 결과를 쓰고,
  시뮬레이터 대신 사용자가 제외 사유·자격 질문에 직접 예/아니오/모름으로 답한다.
- 로그: SQLite(data/app_logs.db)에 질문·답변·인용·응답 시간·피드백, 판정 입력·결과를 남긴다.
"""
import json, re, sqlite3, time
from datetime import date, datetime
from pathlib import Path

import baseline as B
import run_qa_eval as R

DB = Path("data/app_logs.db")
ASK_ELIG = re.compile(r"(받을\s*수\s*있|신청\s*(할\s*수\s*있|가능|돼|되나)|해당\s*(돼|되나|하나))")
ASSERT = re.compile(r"(신청할\s*수\s*있습니다|신청\s*가능합니다|지원받을\s*수\s*있습니다|지원\s*대상입니다|"
                    r"(지원|혜택|컨설팅|교육|융자|보증)을\s*받을\s*수\s*있습니다)")
QA_CFG = {"strategy": "A2", "k": 5, "gen": "Q1S", "guard": True, "retriever": "hybrid"}   # Q1S = 평가 최종 Q1 + 서비스 규칙(자격 단정 금지, 탐색형 목록)
EXPLORE = re.compile(r"(어떤|무슨|뭐가\s*있|뭐\s*있|추천|찾아\s*줘|목록|종류)|(지원|사업|프로그램)\s*(같은\s*거|이|가|은|는|좀|들)?\s*(알려\s*줘|있어|있나요|있을까|있습니까)\s*\??\s*$")   # 사업을 찾는 질문
SIZES = ["예비창업자", "소상공인", "소기업(소상공인 아님)", "중기업", "중견기업"]
SIDOS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]


# ---------------- DB 로그 ----------------
def db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS qa_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, question TEXT, answer TEXT, found INTEGER,
        guard_fired INTEGER, premise TEXT, citations TEXT, quotes TEXT, retrieved TEXT, latency_ms INTEGER, tokens INTEGER,
        config TEXT, feedback INTEGER, feedback_note TEXT, error TEXT)""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(qa_logs)")}
    for c, t in (("explore", "INTEGER"), ("quote_ok", "INTEGER"), ("quote_all", "INTEGER"), ("raw_output", "TEXT")):
        if c not in cols:      # 예전 DB에도 칸 추가
            con.execute(f"ALTER TABLE qa_logs ADD COLUMN {c} {t}")
    con.execute("""CREATE TABLE IF NOT EXISTS judge_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, notice TEXT, title TEXT, profile TEXT, questions TEXT,
        answers TEXT, pred TEXT, reasons TEXT, latency_ms INTEGER, feedback INTEGER, feedback_note TEXT)""")
    return con


def log_qa(session, question, res, latency_ms, error=""):
    with db() as con:
        cur = con.execute(
            "INSERT INTO qa_logs (ts, session, question, answer, found, guard_fired, premise, citations, quotes, retrieved, latency_ms, tokens, config, error, "
            "explore, quote_ok, quote_all, raw_output) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), session, question, res.get("answer", ""), int(bool(res.get("found"))),
             int(bool(res.get("guard_fired"))), res.get("premise", ""), json.dumps(res.get("citations", []), ensure_ascii=False),
             json.dumps(res.get("quotes", []), ensure_ascii=False), json.dumps([h["chunk_id"] for h in res.get("hits", [])]),
             latency_ms, res.get("tokens", 0), json.dumps(QA_CFG), error,
             int(bool(res.get("explore"))), res.get("quote_ok", 0), res.get("quote_all", 0), json.dumps(res.get("raw", {}), ensure_ascii=False)))
        return cur.lastrowid


def log_judge(session, n, title, profile, questions, answers, pred, reasons, latency_ms):
    with db() as con:
        cur = con.execute(
            "INSERT INTO judge_logs (ts, session, notice, title, profile, questions, answers, pred, reasons, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), session, n, title, json.dumps(profile, ensure_ascii=False),
             json.dumps(questions, ensure_ascii=False), json.dumps(answers, ensure_ascii=False), pred,
             json.dumps(reasons, ensure_ascii=False), latency_ms))
        return cur.lastrowid


def set_feedback(table, row_id, value, note=""):
    assert table in ("qa_logs", "judge_logs")
    with db() as con:
        con.execute(f"UPDATE {table} SET feedback=?, feedback_note=? WHERE id=?", (value, note, row_id))


def read_logs(table, limit=200):
    assert table in ("qa_logs", "judge_logs")
    con = db()
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))]
    con.close()
    return rows


# ---------------- 공고 질문 ----------------
_INDEX = {}


def qa_index():
    if "A2" not in _INDEX:
        B.load_env()
        R.CFG.update(gen=QA_CFG["gen"], guard=QA_CFG["guard"], retriever=QA_CFG["retriever"])
        _INDEX["A2"] = R.load_index(QA_CFG["strategy"])
    return _INDEX["A2"]


def answer_question(q):
    """질문 하나 -> {"answer", "found", "quotes", "citations", "premise", "hits", "guard_fired", "tokens"}"""
    chunks, vec = qa_index()
    qv = R.embed_questions([q])[0]
    explore = bool(EXPLORE.search(q))
    hits = R.search(qv, chunks, vec, QA_CFG["k"], q)
    if explore:      # 탐색형: 일반 검색 상위 5개는 그대로 두고, 아직 안 나온 공고의 조각을 3개까지 덧붙임
        seen = {c["notice_id"] for c, _ in hits}
        for c, sc in R.search(qv, chunks, vec, 40, q):
            if c["notice_id"] not in seen:
                hits.append((c, sc)); seen.add(c["notice_id"])
            if len(hits) >= QA_CFG["k"] + 3:
                break
    out, usage = R.answer(q, hits)
    raw = {"found": out.get("found"), "quotes": out.get("quotes"), "answer": out.get("answer")}   # 가드 전 원래 출력 (진단용)
    out, q_ok, q_all, fired = R.apply_guard(out, hits)
    ans = str(out.get("answer", ""))
    if out.get("found") and (explore or ASK_ELIG.search(q)):     # 프롬프트로 막아도 단정하는 경우가 있어 코드로 한 번 더
        ans = ASSERT.sub(lambda m: "신청을 검토해볼 수 있습니다" if "신청" in m.group(0) else "지원 대상에 해당할 수 있습니다", ans)
        if "자격 확인" not in ans:
            ans += "\n\n정확한 신청 가능 여부는 '자격 확인' 탭에서 내 정보를 넣어 확인하세요."
    return {"answer": ans, "found": bool(out.get("found")), "quotes": out.get("quotes") or [],
            "citations": out.get("citations") or [], "premise": str(out.get("premise", "")), "guard_fired": fired,
            "hits": [c for c, _ in hits], "tokens": usage.prompt_tokens + usage.completion_tokens, "explore": explore,
            "raw": raw, "quote_ok": q_ok, "quote_all": q_all}


# ---------------- 자격 확인 (판정 v7) ----------------
def _setup_judge():
    B.VERSION, B.OUT = "v7", str(Path("data/baseline/v7"))


def list_notices():
    """정답표가 있는 공고 30개 (판정 v7 결과가 캐시돼 있음): [(번호, 공고명, notice_id)]"""
    out = []
    for f in sorted(Path("data/answer_key").glob("answer_key_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        out.append((f.stem[-2:], d.get("title", ""), d.get("notice_id", "")))
    return out


_COND = {}


def prepare_conditions(n):
    """공고 n의 v7 조건 (추출 + 2차 제외 사유 + 성격 검토 + 원문 검증). 캐시된 결과만 쓰므로 API 호출 없음"""
    if n in _COND:
        return _COND[n]
    _setup_judge()
    path = Path(B.OUT) / "extract" / f"{n}.json"
    if not path.exists():
        raise FileNotFoundError(f"공고 {n}의 v7 추출 결과가 없습니다. baseline.py --version v7 을 먼저 실행하세요.")
    ex = json.load(open(path, encoding="utf-8"))
    ex2 = B.extract_exclusions(n)
    ex["exclusions"], _ = B.merge_exclusions(ex.get("exclusions"), ex2.get("exclusions"))
    roles = B.role_check(n, ex)
    ex, rlog = B.apply_roles(ex, roles)
    ex = B.industry_names_to_questions(ex)
    ex, vlog = B.verify_conditions(ex, n)
    ex["_log"] = rlog + vlog
    _COND[n] = ex
    return ex


def profile_row(p, today=None):
    """화면 입력 -> 판정 코드가 쓰는 프로필 형식"""
    today = today or date.today()
    opened = p.get("opened")
    months = ""
    if p["size"] != "예비창업자" and opened:
        months = (today.year - opened.year) * 12 + (today.month - opened.month) - (1 if today.day < opened.day else 0)
    return {"size": p["size"], "birth": p["birth"].isoformat() if p.get("birth") else "", "sido": p["sido"], "gu": p.get("gu", ""),
            "ksic": p.get("ksic", ""), "biz_months": str(months), "opened": opened.isoformat() if opened else "",
            "revenue": str(int(p["revenue_manwon"] * 10000)) if p.get("revenue_manwon") not in (None, "") else ""}


def _same(a, b):
    """두 질문이 사실상 같은지 (글자 2-gram 80% 이상 겹침)"""
    a, b = re.sub(r"[^0-9A-Za-z가-힣]", "", a), re.sub(r"[^0-9A-Za-z가-힣]", "", b)
    if not a or not b:
        return False
    short, long_ = sorted((a, b), key=len)
    g = {short[i:i + 2] for i in range(len(short) - 1)}
    return short in long_ or (bool(g) and sum(x in long_ for x in g) / len(g) >= 0.8)


def questions_for(ex, r):
    """사용자에게 물을 질문 [(index, 종류, 질문, 근거)]. 업종 표로 판정하는 항목(table_only)은 묻지 않고,
    거의 같은 질문은 한 번만 묻는다 (ex['_dups'] = {중복 index: 대표 index}, 판정 때 같은 답을 씀)"""
    ex = B.biz_exception_for(ex, r)
    ex = json.loads(json.dumps(ex)); ex["_dups"] = {}
    qs = []
    for i, e in enumerate(ex.get("exclusions") or []):
        if not e.get("table_only"):
            t, ev = e.get("text", ""), e.get("evidence", "")
            if len(t) < 15 and len(ev) > len(t):       # 2차 추출의 '재참여 불가' 같은 짧은 이름은 원문 문장으로 물음
                t = re.sub(r"^\s*[①-⑳\d]+[).]?\s*", "", ev)
            dup = next((q for q in qs if q[1] == "제외" and _same(q[2], t)), None)
            if dup:
                ex["_dups"][i] = dup[0]; continue
            qs.append((i, "제외", f"다음에 해당하나요? {t}", ev))
    n_ex = len(ex.get("exclusions") or [])
    for j, q in enumerate(ex.get("other_requirements") or []):
        qs.append((n_ex + j, "자격", q.get("question", ""), q.get("evidence", "")))
    return ex, qs


REASON_NAME = {"업종 코드 미입력": "업종 (업종 코드를 넣으면 확인할 수 있음)", "revenue_unknown": "매출", "기업규모": "신청 대상(기업 규모)", "창업기업(업력)": "창업기업 업력(7년 미만)", "age": "대표자 나이", "birth_year": "대표자 출생연도",
               "region": "소재지", "industry": "업종", "biz_months": "업력", "founded_after": "창업일", "founded_before": "창업일", "revenue": "매출"}


def explain(ex, reasons):
    """판정 코드(예: '제외3', 'age')를 사람이 읽는 문장 + 근거로"""
    out = []
    n_ex = len(ex.get("exclusions") or [])
    for code in reasons:
        m = re.match(r"(제외|기타|업종표)(\d+)", code)
        if m:
            i = int(m.group(2)) - 1
            if m.group(1) in ("제외", "업종표") and i < n_ex:
                e = ex["exclusions"][i]
                out.append((("업종 제외 표: " if m.group(1) == "업종표" else "제외 사유: ") + e.get("text", ""), e.get("evidence", "")))
            elif m.group(1) == "기타" and 0 <= i - n_ex < len(ex.get("other_requirements") or []):
                q = ex["other_requirements"][i - n_ex]
                out.append(("필수 자격: " + q.get("question", ""), q.get("evidence", "")))
            else:
                out.append((code, ""))
            continue
        ev = ""
        for f in ("age", "biz_months", "revenue"):
            if code.startswith(f.split("_")[0]) and isinstance(ex.get(f), dict):
                ev = ex[f].get("evidence", "")
        if code == "revenue":
            ev = (ex.get("revenue") or {}).get("evidence", "") if isinstance(ex.get("revenue"), dict) else ""
        if code == "region":
            ev = ", ".join(B.as_list(ex.get("region")))
        if code == "기업규모":
            ev = "신청 대상: " + ", ".join(B.as_list(ex.get("legal_terms")))
        out.append((REASON_NAME.get(code, code), ev))
    return out


def judge(ex_p, r, answers_by_index):
    """사용자 답 -> (판정, 이유 코드). 묻지 않은 항목(table_only)은 '모름'으로 채움 (판정 코드가 무시함)"""
    _setup_judge()
    n = len(ex_p.get("exclusions") or []) + len(ex_p.get("other_requirements") or [])
    dups = {int(k): v for k, v in (ex_p.get("_dups") or {}).items()}
    answers = [answers_by_index.get(i, answers_by_index.get(dups.get(i), "모름")) for i in range(n)]
    extra = []
    if not re.sub(r"\D", "", r.get("ksic", "")):     # 업종 코드 미입력: 업종 조건은 '확인 필요'로 (빈 코드를 '해당 없음'으로 보면 틀린 불가가 남)
        ind = ex_p.get("industry") if isinstance(ex_p.get("industry"), dict) else {}
        if ind.get("allowed_ksic") or ind.get("excluded_ksic") or any(e.get("table_only") for e in ex_p.get("exclusions") or []):
            ex_p = json.loads(json.dumps(ex_p)); ex_p["industry"] = None; extra.append("업종 코드 미입력")
    pred, why = B.judge(ex_p, r, answers, date.today())
    if extra and pred == B.POS:
        pred, why = B.COND, extra
    elif extra and pred == B.COND:
        why = why + extra
    return pred, why
