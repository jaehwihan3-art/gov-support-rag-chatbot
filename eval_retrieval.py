"""6단계 검색 실험: 답변 생성 없이 검색 지표만 비교 (질문 임베딩 30개만 API 사용, 캐시됨)

  python eval_retrieval.py

청킹(A1, A2) x 검색(dense, bm25, hybrid) x k 조합의 hit@k, MRR을 계산해
data/eval/retrieval_compare.md 와 retrieval_compare.csv 로 저장한다.
hit 판정은 run_qa_eval.py와 같다: 정답 근거 구절이 상위 k개 청크 안에 모두 들어 있으면 hit.
"""
import csv, statistics
from pathlib import Path

import baseline as B
import run_qa_eval as R

KS = [1, 3, 5, 8, 10]


def main():
    B.load_env()
    rows = [r for r in csv.DictReader(open("data/eval/qa_eval.csv", encoding="utf-8-sig")) if r["evidence"]]
    qv = R.embed_questions([r["question"] for r in rows])
    table, miss = [], {}
    for strat in ("A1", "A2"):
        chunks, vec = R.load_index(strat)
        for mode in ("dense", "bm25", "hybrid"):
            ranks = []
            for i, r in enumerate(rows):
                hits = R.search(qv[i], chunks, vec, max(KS), r["question"], mode=mode)
                rk = R.evidence_rank([e for e in r["evidence"].split(" || ") if e], hits)
                ranks.append(max(rk) if all(rk) else None)
            line = {"청킹": strat, "검색": mode}
            for k in KS:
                line[f"hit@{k}"] = round(sum(1 for x in ranks if x and x <= k) / len(rows) * 100, 1)
            line["MRR"] = round(statistics.mean(1 / x if x else 0 for x in ranks), 3)
            table.append(line)
            miss[(strat, mode)] = [r["qid"] for r, x in zip(rows, ranks) if not x or x > 5]
            print(line, "top-5 밖:", miss[(strat, mode)])
    out = Path("data/eval")
    with open(out / "retrieval_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    L = ["# 검색 방식 비교 (근거 있는 25문항)\n", "| 청킹 | 검색 | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | top-5 밖 문항 |",
         "|---|---|" + "---|" * (len(KS) + 2)]
    for t in table:
        L.append(f"| {t['청킹']} | {t['검색']} | " + " | ".join(f"{t[f'hit@{k}']}%" for k in KS)
                 + f" | {t['MRR']} | {', '.join(miss[(t['청킹'], t['검색'])])} |")
    (out / "retrieval_compare.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L) + f"\n\n저장: {out}/retrieval_compare.md, retrieval_compare.csv")


if __name__ == "__main__":
    main()
