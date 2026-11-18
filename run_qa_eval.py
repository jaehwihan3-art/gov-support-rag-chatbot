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


def search(qvec, chunks, vec, k):
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
    out, usage = B.chat_json(GEN_SYS, f"[검색된 공고 조각]\n{ctx}\n\n[질문]\n{q}", model=GEN_MODEL)
    return out, usage


def grade(row, ans):
    user = (f"[질문 유형] {row['type']}\n[질문] {row['question']}\n[정답] {row['gold_answer']}\n"
            f"[챗봇 답변] {ans}")
    out, _ = B.chat_json(GRADE_SYS, user, model=GRADE_MODEL)
    g = out.get("grade", "오답")
    return (g if g in SCORE else "오답"), out.get("reason", "")


def run_one(row, chunks, vec, k, qvec):
    hits = search(qvec, chunks, vec, k)
    evs = [e for e in row["evidence"].split(" || ") if e]
    ranks = evidence_rank(evs, hits) if evs else []
    out, usage = answer(row["question"], hits)
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
        "tokens": usage.prompt_tokens + usage.completion_tokens,
    }


def pct(a, b):
    return f"{a}/{b} ({a / b * 100:.0f}%)" if b else "-"


def summarize(res, args):
    ev = [r for r in res if r["evidence"]]
    L = [f"# Q&A 평가 결과: {args.name}\n",
         f"- 청킹: {args.strategy} / 검색 top-{args.k} / 답변 {GEN_MODEL} / 채점 {GRADE_MODEL} / 문항 {len(res)}개",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="A2", choices=["A1", "A2"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--name", default=None)
    ap.add_argument("--only", default="", help="일부 문항만 (예: A01,T02)")
    args = ap.parse_args()
    args.name = args.name or f"{args.strategy}_k{args.k}"
    B.load_env()

    rows = list(csv.DictReader(open("data/eval/qa_eval.csv", encoding="utf-8-sig")))
    if args.only:
        rows = [r for r in rows if r["qid"] in args.only.split(",")]
    chunks, vec = load_index(args.strategy)
    qvecs = np.array(B.embed([r["question"] for r in rows]), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda i: run_one(rows[i], chunks, vec, args.k, qvecs[i]), range(len(rows))))
    for r in res:
        print(f"{r['qid']} {r['type']:<10} 근거순위={r['hit_rank'] or '-':<3} found={str(r['found']):<5} {r['llm_grade']}")

    out = Path("data/eval/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    with open(out / "human_grading.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "type", "question", "gold_answer", "answer", "사람 판정(정답/부분정답/오답)", "메모"])
        w.writeheader()
        for r in res:
            w.writerow({"qid": r["qid"], "type": r["type"], "question": r["question"], "gold_answer": r["gold_answer"],
                        "answer": r["answer"], "사람 판정(정답/부분정답/오답)": "", "메모": ""})
    s = summarize(res, args)
    (out / "summary.md").write_text(s, encoding="utf-8")
    print("\n" + s + f"\n\n저장: {out}/results.csv, summary.md, human_grading.csv")


if __name__ == "__main__":
    main()
