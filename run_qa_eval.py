"""4단계 Q&A 평가 자동 실행: 질문 30개 -> 검색 -> 답변 생성 -> 채점 -> 요약표

  python run_qa_eval.py --strategy A2 --k 5 --name A2_k5
  python run_qa_eval.py --strategy A1 --k 5 --name A1_k5

입력  data/eval/qa_eval.csv          확정 평가셋 (make_qa_draft.py로 만든 초안을 사람이 확인해 확정)
      data/chunks/{A1,A2}.jsonl      청크
출력  data/eval/runs/<name>/results.csv        질문별 검색 결과·답변·인용·채점
      data/eval/runs/<name>/summary.md         지표 요약
      data/eval/runs/<name>/human_grading.csv  사람 채점용 시트 (LLM 판정은 안 보이게 뺌 -> 편향 없이 채점)
      data/index/<strategy>.npz                청크 임베딩 캐시 (청크가 바뀌면 다시 만듦)

지표
  hit@k       정답 근거 구절이 검색된 상위 k개 청크 안에 들어 있는 비율 (근거 있는 25문항: 답 있음 20 + 함정 5)
              근거가 여러 개면 모두 들어 있어야 hit. MRR은 첫 근거 청크 순위의 역수 평균
  답변 정확도  LLM 채점(정답 1 / 부분정답 0.5 / 오답 0) 평균, 유형별로도 집계
  인용 정확도  답변이 인용한 청크에 정답 근거 구절이 들어 있는 비율 (근거 있는 문항 중 답한 것 기준)
  무응답 처리  문서에 없음 5문항 중 '확인 불가'로 답한 비율 / 답 있는 25문항 중 잘못 '확인 불가'라고 한 비율
"""
import argparse, csv, hashlib, json, os, re, statistics, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import baseline as B

GEN_MODEL, GRADE_MODEL = "gpt-4o-mini", "gpt-4.1-mini"   # 채점은 답변과 다른 모델로 (같은 모델이 자기 답을 후하게 보는 것 방지)

GEN_SYS = """너는 소상공인·예비창업자에게 정부 지원사업 공고 내용을 알려주는 상담 챗봇이다.
반드시 아래 [검색된 공고 조각]에 적힌 내용만으로 답한다. 조각에 없는 내용은 추측하지 않는다.

규칙
1. 조각에서 답을 찾으면 found=true, 핵심(금액·기간·조건·신청처 등)을 짧게 답하고 근거로 쓴 조각의 id를 citations에 넣는다.
2. 조각에 답이 없거나, 질문한 사업·지역·연도의 공고가 조각에 없으면 found=false, answer에 "제공된 공고에서는 확인할 수 없습니다"와
   함께 확인할 곳(조각에 문의처가 있으면 그 번호)을 안내한다. 비슷한 다른 사업 내용을 질문한 사업인 것처럼 답하지 않는다.
3. 질문에 틀린 전제(잘못된 금액·나이·조건 등)가 있으면 먼저 그 전제가 틀렸다고 바로잡고 공고의 실제 내용을 말한다.
4. 질문자가 지원 제외 대상에 해당하면 신청 방법보다 먼저 제외 대상이라는 것을 알린다.

JSON으로만 답한다: {"found": true/false, "answer": "...", "citations": ["조각 id", ...]}"""

GRADE_SYS = """너는 정부 지원사업 챗봇의 답변을 채점하는 평가자다. 질문, 질문 유형, 정답(사람이 공고에서 확인한 기준), 챗봇 답변을 보고 판정한다.

유형별 기준
- answerable: 정답의 핵심 사실(숫자·기간·가능/불가·신청처)이 맞으면 정답. 핵심은 맞지만 중요한 조건 일부가 빠졌으면 부분정답.
  핵심 사실이 틀리거나, '확인할 수 없다'고 답했으면 오답. 정답에 있는 부가 정보가 빠진 것만으로는 감점하지 않는다.
- not_in_doc: 공고에서 확인할 수 없다고 하고 문의처 등을 안내하면 정답. 확인 불가라고 하면서 추측을 덧붙였으면 부분정답.
  가능/불가나 금액을 단정하거나, 다른 사업·연도의 내용을 질문한 것인 양 답했으면 오답.
- trap: 질문의 틀린 전제를 명확히 바로잡고 실제 내용을 말하면 정답. 실제 내용은 말했지만 전제가 틀렸다고 분명히 하지 않았으면 부분정답.
  전제를 받아들였거나(예: 틀린 금액에 동의, 제외 대상인데 신청 방법만 안내) 실제 내용이 틀리면 오답.

JSON으로만 답한다: {"grade": "정답"|"부분정답"|"오답", "reason": "한 문장"}"""

GEN_SYS_Q1 = """너는 소상공인·예비창업자에게 정부 지원사업 공고 내용을 알려주는 상담 챗봇이다.
반드시 아래 [검색된 공고 조각]에 적힌 내용만으로 답한다. 아래 순서대로 JSON 칸을 채운다.

1. quotes: 질문에 답하는 데 필요한 문장을 조각에서 원문 그대로 옮긴다 (조각 id와 함께). 질문한 사업(지역·사업명)과 같은 공고의 조각에서만 고른다. 관련 문장이 없으면 빈 목록.
2. premise: 질문에 사용자가 사실이라고 가정한 내용(금액·나이·기간·자격 등)이 있으면 quotes와 비교해 "맞음", "틀림", "확인 불가" 중 하나와 이유를 쓴다. 가정이 없으면 "없음".
3. found: quotes가 하나라도 있으면 true, 없을 때만 false.
4. answer:
   - found=true: quotes 내용으로 핵심(금액·기간·가능/불가·신청처)을 먼저 답한다.
   - premise가 "틀림"이면 첫 문장을 "아닙니다"로 시작해 틀린 부분을 바로잡고 공고의 실제 내용을 말한다.
   - 질문자가 지원 제외 대상에 해당하면 신청 방법보다 먼저 제외 대상이라고 알린다.
   - quotes에 관련 사실만 있고 질문한 내용 자체는 없으면, 그 사실을 말하되 "질문하신 OO는 공고에서 확인되지 않습니다"라고 분명히 쓴다.
   - found=false: "제공된 공고에서는 확인할 수 없습니다"라고 하고, 조각에 문의처가 있으면 함께 안내한다. 비슷한 다른 사업 내용을 질문한 사업인 것처럼 답하지 않는다.
5. citations: quotes에 쓴 조각 id.

JSON으로만 답한다: {"quotes": [{"id": "조각 id", "text": "원문 문장"}], "premise": "...", "found": true/false, "answer": "...", "citations": ["조각 id"]}"""

GRADE_SYS_G1 = """너는 정부 지원사업 챗봇의 답변을 채점하는 평가자다. 질문, 질문 유형, 정답(사람이 공고에서 확인한 기준), 챗봇 답변을 보고 판정한다.

먼저 '핵심'을 정한다. 핵심은 질문이 직접 묻는 것이다 (금액을 물으면 금액, 기간을 물으면 기간, 가능 여부를 물으면 가능/불가).
정답 칸에서 질문이 묻지 않은 나머지 정보(다른 채널의 금액, 부가 조건, 문의처, 서류 등)는 보너스이며 빠져도 감점하지 않는다.

유형별 기준
- answerable: 핵심이 정답과 맞으면 정답. 핵심은 맞지만 결론을 바꾸는 단서(예: "3개월 이내에 신청해야 가능")가 빠지면 부분정답.
  핵심이 틀리거나, 정답이 있는데 '확인할 수 없다'고 답하면 오답.
- not_in_doc: 공고에서 확인할 수 없다고 하고 가능/불가·금액·시기를 단정하지 않으면 정답 (문의처 안내는 보너스).
  확인 불가라고 밝히지 않고 관련된 다른 사실로 답을 대신했거나, 단정했거나, 다른 사업·연도 내용을 질문한 것처럼 답했으면 오답.
- trap: 질문의 전제가 틀렸다는 것을 분명히 말하고(예: "아닙니다") 공고의 실제 내용을 말하면 정답.
  실제 내용은 맞게 말했지만 전제가 틀렸다고 명시하지 않았으면 부분정답.
  '확인할 수 없다'고 답했거나(정답이 공고에 있으므로), 전제를 받아들였거나, 제외 대상인데 신청 방법만 안내했으면 오답.

JSON으로만 답한다: {"grade": "정답"|"부분정답"|"오답", "reason": "한 문장"}"""

GEN_PROMPTS = {"Q0": GEN_SYS, "Q1": GEN_SYS_Q1}
GRADE_PROMPTS = {"G0": GRADE_SYS, "G1": GRADE_SYS_G1, "G2": GRADE_SYS_G1}   # G2 = G1 + 아래 코드 규칙
CFG = {"gen": "Q0", "grader": "G0", "guard": False, "retriever": "dense"}
REFUSE = re.compile(r"확인할 수 없|확인되지 않|확인이 어렵|알 수 없")
REFUSE_MSG = "제공된 공고에서는 확인할 수 없습니다."


def rule_grade(row, ans, g, reason):
    """G2: LLM 채점기가 프롬프트 규칙을 무시한 경우(함정 거절을 부분정답으로 봐줌 등)를 코드로 바로잡음"""
    if CFG["grader"] != "G2":
        return g, reason
    refused = bool(REFUSE.search(ans)) and not re.match(r"\s*아닙니다", ans)
    if row["type"] in ("answerable", "trap") and refused and len(ans) < 80:
        return "오답", "[규칙] 정답이 공고에 있는데 '확인할 수 없다'고 답함"
    return g, reason


def apply_guard(out, hits):
    """인용 검증 가드: 인용 문장이 검색된 청크에 실제로 있는지 확인. 인용이 하나도 확인되지 않으면 답하지 않고 거절로 바꿈.
    id를 잘못 적은 인용은 문장이 들어 있는 청크 id로 고침. 반환: (out, 확인된 인용 수, 전체 인용 수, 가드 작동 여부)"""
    quotes = out.get("quotes")
    if quotes is None:
        return out, 0, 0, False
    valid = []
    letters = lambda x: re.sub(r"[^0-9A-Za-z가-힣]", "", x)   # 따옴표·★·괄호 같은 기호 차이는 무시

    def contains(c, raw):
        ct, t = letters(c["text"]), letters(raw)
        if t and t in ct:
            return True
        # 떨어진 줄을 이어 붙여 인용한 경우: 조각(4자 이상)의 70% 이상이 같은 청크에 있으면 인정
        pieces = [letters(x) for x in re.split(r"[\n*※·,()/|:;]+", raw)]
        pieces = [x for x in pieces if len(x) >= 4]
        return bool(pieces) and sum(x in ct for x in pieces) >= 0.7 * len(pieces)
    for q in quotes if isinstance(quotes, list) else []:
        raw = str(q.get("text", "")) if isinstance(q, dict) else ""
        if len(letters(raw)) < 6:
            continue
        cited = [c for c, _ in hits if c["chunk_id"] == q.get("id")]
        hit = next((c for c in cited + [c for c, _ in hits] if contains(c, raw)), None)   # 적힌 id를 먼저 확인
        if hit:
            valid.append({"id": hit["chunk_id"], "text": raw})
    fired = False
    out = dict(out)
    if CFG["guard"]:
        out["quotes"] = valid
        out["citations"] = list(dict.fromkeys(v["id"] for v in valid))
        if out.get("found") and not valid:
            out["found"], out["answer"], fired = False, REFUSE_MSG, True
    return out, len(valid), len(quotes) if isinstance(quotes, list) else 0, fired

SCORE = {"정답": 1.0, "부분정답": 0.5, "오답": 0.0}


def norm(s):
    return re.sub(r"\s+", "", s)


def load_index(strategy):
    chunks = [json.loads(l) for l in open(f"data/chunks/{strategy}.jsonl", encoding="utf-8")]
    key = hashlib.md5("".join(c["chunk_id"] + c["text"] for c in chunks).encode()).hexdigest()
    path = Path(f"data/index/{strategy}.npz")
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return chunks, z["vec"]
    print(f"[{strategy}] 청크 {len(chunks)}개 임베딩 만드는 중 (처음 한 번만, text-embedding-3-small)...")
    # 공고명을 앞에 붙여 임베딩: 조각만으로는 어느 사업인지 모르는 경우가 많음
    vec = np.array(B.embed([f"{c['title']}\n{c['text']}"[:6000] for c in chunks]), dtype=np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=key, vec=vec)
    return chunks, vec


def grams(t):
    """한국어용 간단 토큰: 공백으로 나눈 단어 + 글자 2-gram (형태소 분석기 없이)"""
    t = re.sub(r"[^0-9A-Za-z가-힣]+", " ", t.lower())
    out = []
    for w in t.split():
        out.append(w)
        out += [w[i:i + 2] for i in range(len(w) - 1)]
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        from math import log
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(docs)
        df = Counter(w for d in self.tf for w in d)
        n = len(docs)
        self.idf = {w: log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q):
        qs = set(q)
        out = np.zeros(len(self.tf), dtype=np.float32)
        for i, (tf, L) in enumerate(zip(self.tf, self.len)):
            sc = 0.0
            for w in qs:
                f = tf.get(w)
                if f:
                    sc += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[i] = sc
        return out


_BM25 = {}


def bm25_for(chunks):
    key = id(chunks)
    if key not in _BM25:
        _BM25[key] = BM25([grams(c["title"] + " " + c["text"]) for c in chunks])
    return _BM25[key]


def search(qvec, chunks, vec, k, qtext="", mode=None):
    """dense: 임베딩 코사인 / bm25: 키워드 / hybrid: 두 순위를 RRF(1/(60+순위))로 합침"""
    mode = mode or CFG["retriever"]
    if mode in ("bm25", "hybrid"):
        bs = bm25_for(chunks).scores(grams(qtext))
        if mode == "bm25":
            top = np.argsort(-bs)[:k]
            return [(chunks[i], float(bs[i])) for i in top]
        ds = vec @ qvec
        rd, rb = np.argsort(-ds)[:100], np.argsort(-bs)[:100]
        rrf = defaultdict(float)
        for r, i in enumerate(rd):
            rrf[int(i)] += 1 / (60 + r + 1)
        for r, i in enumerate(rb):
            rrf[int(i)] += 1 / (60 + r + 1)
        top = sorted(rrf, key=lambda i: -rrf[i])[:k]
        return [(chunks[i], rrf[i]) for i in top]
    return _dense(qvec, chunks, vec, k)


def embed_questions(texts):
    """질문 임베딩 캐시 (data/index/questions.json) - 같은 질문은 다시 API를 부르지 않음"""
    path = Path("data/index/questions.json")
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    miss = [t for t in texts if t not in cache]
    if miss:
        for t, v in zip(miss, B.embed(miss)):
            cache[t] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    q = np.array([cache[t] for t in texts], dtype=np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _dense(qvec, chunks, vec, k):
    sims = vec @ qvec
    top = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in top]


def evidence_rank(evs, hits):
    """근거 구절마다 처음 들어 있는 검색 순위(1부터). 없으면 None"""
    ranks = []
    for e in evs:
        r = next((i + 1 for i, (c, _) in enumerate(hits) if norm(e) in norm(c["text"])), None)
        ranks.append(r)
    return ranks


def answer(q, hits):
    ctx = "\n\n".join(f"[id: {c['chunk_id']}] (공고: {c['title']})\n{c['text']}" for c, _ in hits)
    out, usage = B.chat_json(GEN_PROMPTS[CFG["gen"]], f"[검색된 공고 조각]\n{ctx}\n\n[질문]\n{q}", model=GEN_MODEL)
    return out, usage


def grade(row, ans):
    user = (f"[질문 유형] {row['type']}\n[질문] {row['question']}\n[정답] {row['gold_answer']}\n"
            f"[챗봇 답변] {ans}")
    out, _ = B.chat_json(GRADE_PROMPTS[CFG["grader"]], user, model=GRADE_MODEL)
    g = out.get("grade", "오답")
    return rule_grade(row, ans, g if g in SCORE else "오답", out.get("reason", ""))


def run_one(row, chunks, vec, k, qvec):
    hits = search(qvec, chunks, vec, k, row["question"])
    evs = [e for e in row["evidence"].split(" || ") if e]
    ranks = evidence_rank(evs, hits) if evs else []
    out, usage = answer(row["question"], hits)
    out, q_ok, q_all, fired = apply_guard(out, hits)
    found = bool(out.get("found"))
    ans = str(out.get("answer", ""))
    retrieved_ids = [c["chunk_id"] for c, _ in hits]
    cites = [str(x) for x in out.get("citations", []) or []]
    valid_cites = [x for x in cites if x in retrieved_ids]
    cited_text = norm(" ".join(c["text"] for c, _ in hits if c["chunk_id"] in valid_cites))
    cite_ok = (all(norm(e) in cited_text for e in evs) if (evs and found) else None)
    g, reason = grade(row, ans)
    return {
        "qid": row["qid"], "type": row["type"], "question": row["question"], "gold_answer": row["gold_answer"],
        "evidence": row["evidence"], "hit_rank": max(ranks) if ranks and all(ranks) else "",
        "first_rank": min(r for r in ranks if r) if any(ranks) else "",
        "retrieved": " | ".join(f"{c['chunk_id']}({s:.3f})" for c, s in hits),
        "found": found, "answer": ans, "citations": " | ".join(cites),
        "invalid_citations": " | ".join(x for x in cites if x not in retrieved_ids),
        "citation_ok": "" if cite_ok is None else cite_ok,
        "llm_grade": g, "llm_reason": reason,
        "quotes": json.dumps(out.get("quotes", []), ensure_ascii=False) if "quotes" in out else "",
        "premise": str(out.get("premise", "")),
        "quote_ok": q_ok, "quote_all": q_all, "guard_fired": fired,
        "tokens": usage.prompt_tokens + usage.completion_tokens,
    }


def pct(a, b):
    return f"{a}/{b} ({a / b * 100:.0f}%)" if b else "-"


def summarize(res, args):
    ev = [r for r in res if r["evidence"]]
    L = [f"# Q&A 평가 결과: {args.name}\n",
         f"- 청킹: {args.strategy} / 검색 top-{args.k} / 답변 {GEN_MODEL} 프롬프트 {CFG['gen']} / 채점 {GRADE_MODEL} 프롬프트 {CFG['grader']} / 문항 {len(res)}개" + (f" / 답변 재사용: {args.regrade_from}" if args.regrade_from else ""),
         f"- 실행: {time.strftime('%Y-%m-%d %H:%M')}\n",
         "## 검색 (근거 있는 문항: 답 있음 + 함정)\n", "| 지표 | 값 |", "|---|---|"]
    for kk in sorted({1, 3, args.k}):
        if kk <= args.k:
            n = sum(1 for r in ev if r["hit_rank"] and r["hit_rank"] <= kk)
            L.append(f"| hit@{kk} | {pct(n, len(ev))} |")
    mrr = statistics.mean(1 / r["first_rank"] if r["first_rank"] else 0 for r in ev) if ev else 0
    L.append(f"| MRR | {mrr:.3f} |")

    L += ["\n## 답변 (LLM 채점: 정답 1, 부분정답 0.5)\n", "| 유형 | 문항 | 정답 | 부분 | 오답 | 점수 |", "|---|---|---|---|---|---|"]
    for t in ["answerable", "not_in_doc", "trap", "전체"]:
        rs = res if t == "전체" else [r for r in res if r["type"] == t]
        c = Counter(r["llm_grade"] for r in rs)
        sc = sum(SCORE[r["llm_grade"]] for r in rs) / len(rs) if rs else 0
        L.append(f"| {t} | {len(rs)} | {c['정답']} | {c['부분정답']} | {c['오답']} | {sc * 100:.1f}% |")

    answered = [r for r in ev if r["found"]]
    L += ["\n## 인용·무응답\n", "| 지표 | 값 |", "|---|---|",
          f"| 인용 정확도 (근거 있는 문항 중 답한 것: 인용 청크에 정답 근거가 있음) | {pct(sum(1 for r in answered if r['citation_ok'] is True), len(answered))} |",
          f"| 검색 결과에 없는 id 인용 (지어낸 인용) | {sum(1 for r in res if r['invalid_citations'])}건 |",
          f"| 문서에 없음 → '확인 불가'로 답함 | {pct(sum(1 for r in res if r['type'] == 'not_in_doc' and not r['found']), sum(1 for r in res if r['type'] == 'not_in_doc'))} |",
          f"| 답 있는 문항인데 '확인 불가'로 답함 (과잉 거절) | {pct(sum(1 for r in ev if not r['found']), len(ev))} |",
          f"| 과잉 거절 중 검색 실패(근거가 top-{args.k}에 없음) | {sum(1 for r in ev if not r['found'] and not r['hit_rank'])}건 |",
          f"\n토큰 합계(답변): {sum(r['tokens'] for r in res):,}"]
    L += ["\n## 틀린 문항\n", "| qid | 유형 | 판정 | 근거 순위 | 이유 |", "|---|---|---|---|---|"]
    for r in res:
        if r["llm_grade"] != "정답":
            L.append(f"| {r['qid']} | {r['type']} | {r['llm_grade']} | {r['hit_rank'] or ('-' if not r['evidence'] else '없음')} | {r['llm_reason']} |")
    return "\n".join(L)


def metrics(res, k):
    ev = [r for r in res if r["evidence"]]
    sc = lambda rs: round(sum(SCORE[r["llm_grade"]] for r in rs) / len(rs) * 100, 1) if rs else ""
    answered = [r for r in ev if r["found"]]
    return {
        "hit@1": round(sum(1 for r in ev if r["hit_rank"] and r["hit_rank"] <= 1) / len(ev) * 100, 1),
        "hit@k": round(sum(1 for r in ev if r["hit_rank"]) / len(ev) * 100, 1),
        "MRR": round(statistics.mean(1 / r["first_rank"] if r["first_rank"] else 0 for r in ev), 3),
        "정확도": sc(res), "answerable": sc([r for r in res if r["type"] == "answerable"]),
        "not_in_doc": sc([r for r in res if r["type"] == "not_in_doc"]), "trap": sc([r for r in res if r["type"] == "trap"]),
        "인용정확도": round(sum(1 for r in answered if r["citation_ok"] is True) / len(answered) * 100, 1) if answered else "",
        "과잉거절": sum(1 for r in ev if not r["found"]),
        "모르는데답함": sum(1 for r in res if r["type"] == "not_in_doc" and r["found"]),
        "인용문장확인율": (round(sum(int(r.get("quote_ok") or 0) for r in res) / max(1, sum(int(r.get("quote_all") or 0) for r in res)) * 100, 1)
                     if any(r.get("quote_all") not in ("", None, 0, "0") for r in res) else ""),
        "가드작동": sum(1 for r in res if str(r.get("guard_fired")) == "True"),
        "토큰": sum(r["tokens"] for r in res),
    }


def log_experiment(res, args):
    """실행마다 data/eval/experiments.csv에 한 줄 추가 (6단계 실험 로그)"""
    path = Path("data/eval/experiments.csv")
    row = {"시각": time.strftime("%Y-%m-%d %H:%M"), "실행": args.name, "청킹": args.strategy, "검색": args.retriever, "top_k": args.k,
           "답변": args.gen + ("+guard" if args.guard else ""), "채점": args.grader, "답변재사용": args.regrade_from, "문항": len(res),
           **metrics(res, args.k), "메모": args.note}
    old = list(csv.DictReader(open(path, encoding="utf-8-sig"))) if path.exists() else []
    fields = list(old[0]) if old else []
    fields += [k for k in row if k not in fields]
    fields = [f for f in fields if f != "메모"] + ["메모"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # 새 지표 칸이 생겨도 표가 맞게 다시 씀
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old + [row])
    print(f"실험 로그에 추가: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="A2", choices=["A1", "A2"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--name", default=None)
    ap.add_argument("--only", default="", help="일부 문항만 (예: A01,T02)")
    ap.add_argument("--gen", default="Q0", choices=list(GEN_PROMPTS), help="답변 프롬프트 버전")
    ap.add_argument("--grader", default="G0", choices=list(GRADE_PROMPTS), help="채점 프롬프트 버전")
    ap.add_argument("--retriever", default="dense", choices=["dense", "bm25", "hybrid"], help="검색 방식")
    ap.add_argument("--guard", action="store_true", help="인용 검증 가드 켜기 (Q1 이상)")
    ap.add_argument("--regrade-from", default="", help="이 실행의 답변을 그대로 두고 채점만 다시 (예: A2_k5)")
    ap.add_argument("--note", default="", help="실험 로그에 남길 한 줄 설명")
    args = ap.parse_args()
    args.name = args.name or f"{args.strategy}_k{args.k}"
    CFG["gen"], CFG["grader"], CFG["guard"], CFG["retriever"] = args.gen, args.grader, args.guard, args.retriever
    B.load_env()

    rows = list(csv.DictReader(open("data/eval/qa_eval.csv", encoding="utf-8-sig")))
    if args.only:
        rows = [r for r in rows if r["qid"] in args.only.split(",")]
    chunks, vec = load_index(args.strategy)
    qvecs = embed_questions([r["question"] for r in rows])

    if args.regrade_from:
        src = Path("data/eval/runs") / args.regrade_from
        old = list(csv.DictReader(open(src / "results.csv", encoding="utf-8-sig")))
        if args.only:
            old = [r for r in old if r["qid"] in args.only.split(",")]
        gold = {r["qid"]: r for r in rows}

        by_id = {c["chunk_id"]: c for c in chunks}

        def regrade(r):
            r = dict(r)
            if CFG["guard"] and r.get("quotes"):
                hits = [(by_id[x.split("(")[0]], 0) for x in r["retrieved"].split(" | ") if x.split("(")[0] in by_id]
                o = {"found": r["found"] == "True", "answer": r["answer"], "quotes": json.loads(r["quotes"]),
                     "citations": [x for x in r["citations"].split(" | ") if x]}
                o, r["quote_ok"], r["quote_all"], r["guard_fired"] = apply_guard(o, hits)
                r["found"], r["answer"] = str(o["found"]), o["answer"]
                r["citations"] = " | ".join(o["citations"])
                evs = [e for e in r["evidence"].split(" || ") if e]
                ct = norm(" ".join(c["text"] for c, _ in hits if c["chunk_id"] in o["citations"]))
                r["citation_ok"] = str(all(norm(e) in ct for e in evs)) if (evs and o["found"]) else ""
            r["found"] = r["found"] == "True"
            r["citation_ok"] = {"True": True, "False": False}.get(r["citation_ok"], "")
            r["hit_rank"] = int(r["hit_rank"]) if r["hit_rank"] else ""
            r["first_rank"] = int(r["first_rank"]) if r["first_rank"] else ""
            r["tokens"] = int(r["tokens"] or 0)
            r["llm_grade"], r["llm_reason"] = grade(gold[r["qid"]], r["answer"])
            return r
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(regrade, old))
    else:
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda i: run_one(rows[i], chunks, vec, args.k, qvecs[i]), range(len(rows))))
    for r in res:
        print(f"{r['qid']} {r['type']:<10} 근거순위={r['hit_rank'] or '-':<3} found={str(r['found']):<5} {r['llm_grade']}")

    out = Path("data/eval/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    hg_path = out / "human_grading.csv"
    copied = False
    if args.regrade_from and (Path("data/eval/runs") / args.regrade_from / "human_grading.csv").exists():
        import shutil   # 답변이 같으니 사람 판정도 그대로 가져옴
        shutil.copy(Path("data/eval/runs") / args.regrade_from / "human_grading.csv", hg_path)
        copied = True
    elif hg_path.exists() and any(r.get("사람 판정(정답/부분정답/오답)", "").strip() for r in csv.DictReader(open(hg_path, encoding="utf-8-sig"))):
        hg_path = out / "human_grading_new.csv"   # 이미 채운 사람 판정은 덮어쓰지 않음
        print(f"[알림] 사람 판정이 채워진 human_grading.csv가 있어서 새 시트는 {hg_path.name}로 저장")
    if copied:
        pass
    else:
      with open(hg_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "type", "question", "gold_answer", "answer", "사람 판정(정답/부분정답/오답)", "메모"])
        w.writeheader()
        for r in res:
            w.writerow({"qid": r["qid"], "type": r["type"], "question": r["question"], "gold_answer": r["gold_answer"],
                        "answer": r["answer"], "사람 판정(정답/부분정답/오답)": "", "메모": ""})
    s = summarize(res, args)
    (out / "summary.md").write_text(s, encoding="utf-8")
    log_experiment(res, args)
    print("\n" + s + f"\n\n저장: {out}/results.csv, summary.md, human_grading.csv")


if __name__ == "__main__":
    main()
