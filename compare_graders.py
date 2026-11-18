"""LLM 채점 vs 사람 채점 비교 (4단계)

  python compare_graders.py --name A2_k5

입력  data/eval/runs/<name>/results.csv        LLM 판정(llm_grade)
      data/eval/runs/<name>/human_grading.csv  사람 판정 (LLM 판정을 보지 않고 채운 것)
출력  data/eval/runs/<name>/grader_agreement.md  일치율, 카피 계수, 혼동표, 판정이 갈린 문항
"""
import argparse, csv
from collections import Counter
from pathlib import Path

LABELS = ["정답", "부분정답", "오답"]


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    d = Path("data/eval/runs") / ap.parse_args().name
    llm = {r["qid"]: r for r in csv.DictReader(open(d / "results.csv", encoding="utf-8-sig"))}
    try:   # 엑셀에서 그냥 'CSV'로 저장하면 cp949가 되므로 둘 다 받음
        hum = {r["qid"]: r for r in csv.DictReader(open(d / "human_grading.csv", encoding="utf-8-sig"))}
    except UnicodeDecodeError:
        hum = {r["qid"]: r for r in csv.DictReader(open(d / "human_grading.csv", encoding="cp949"))}
    col = "사람 판정(정답/부분정답/오답)"
    pairs, diff = [], []
    for q, h in hum.items():
        hg = h[col].strip().replace(" ", "")
        if hg not in LABELS:
            continue
        lg = llm[q]["llm_grade"]
        pairs.append((hg, lg))
        if hg != lg:
            diff.append((q, h["type"], hg, lg, llm[q]["llm_reason"], h.get("메모", "")))
    if not pairs:
        raise SystemExit(f"사람 판정이 비어 있음: {d / 'human_grading.csv'}의 '{col}' 칸을 정답/부분정답/오답으로 채워 주세요.")
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    pass_agree = sum((a == "정답") == (b == "정답") for a, b in pairs)
    hs = sum({"정답": 1, "부분정답": .5, "오답": 0}[a] for a, _ in pairs) / n
    ls = sum({"정답": 1, "부분정답": .5, "오답": 0}[b] for _, b in pairs) / n
    L = [f"# LLM 채점 vs 사람 채점: {d.name}\n", f"비교 문항 {n}개\n", "| 지표 | 값 |", "|---|---|",
         f"| 3단계 판정 일치율 | {agree}/{n} ({agree / n * 100:.0f}%) |",
         f"| 정답 여부(정답 vs 나머지) 일치율 | {pass_agree}/{n} ({pass_agree / n * 100:.0f}%) |",
         f"| 카파 계수 (우연 일치 보정, 0.6 이상이면 상당한 일치) | {kappa(pairs):.2f} |",
         f"| 답변 정확도: 사람 / LLM | {hs * 100:.1f}% / {ls * 100:.1f}% |",
         "\n## 혼동표 (행: 사람, 열: LLM)\n", "| 사람 \\ LLM | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    c = Counter(pairs)
    for a in LABELS:
        L.append(f"| {a} | " + " | ".join(str(c[(a, b)]) for b in LABELS) + " |")
    L += ["\n## 판정이 갈린 문항\n", "| qid | 유형 | 사람 | LLM | LLM 이유 | 사람 메모 |", "|---|---|---|---|---|---|"]
    L += [f"| {q} | {t} | {a} | {b} | {r} | {m} |" for q, t, a, b, r, m in diff]
    (d / "grader_agreement.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
