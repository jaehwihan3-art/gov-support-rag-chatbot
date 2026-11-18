"""
시스템 예측과 정답 쌍을 비교해, 결과를 4가지로 나눠 보고한다.

  정답               : 예측 == 정답
  조건부로 물러섬    : 정답은 '가능' 또는 '불가'인데 '조건부 가능'으로 답함 (안전하지만 쓸모가 줄어듦)
  틀린 신청 가능     : 정답은 '불가' 또는 '조건부'인데 '신청 가능'으로 답함 (가장 위험한 오류)
  틀린 신청 불가     : 정답은 '가능' 또는 '조건부'인데 '신청 불가'로 답함 (기회 손실)

※ 정답이 '조건부'(프로필 정보 부족)인 쌍에서 '조건부'라고 답하면 정답으로 센다.

predictions.csv 형식: pair_id,pred[,filter_pass]
  pred        : '신청 가능' / '신청 불가' / '조건부 가능' (2차 판정 결과)
  filter_pass : Y/N - 이 프로필에서 해당 공고가 1차 후보 필터를 통과했는지 (있으면 1차 필터 누락률도 계산)

사용법:
    python evaluate.py pairs_holdout.csv predictions.csv [--max-abstain 0.2] [--out eval.txt]
"""
import argparse, csv
from collections import Counter, defaultdict
from math import comb

POS, NEG, COND = "신청 가능", "신청 불가", "조건부 가능"
CATS = ["정답", "조건부로 물러섬", "틀린 신청 가능", "틀린 신청 불가"]


def category(gold, pred):
    if pred == gold:
        return "정답"
    if pred == COND:
        return "조건부로 물러섬"
    if pred == POS:
        return "틀린 신청 가능"
    if pred == NEG:
        return "틀린 신청 불가"
    return "예측 없음"


def upper95(k, n):
    """비율의 95% 신뢰상한 (Clopper-Pearson, 단측). k=0이면 약 3/n."""
    if n == 0:
        return float("nan")
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, i) * mid**i * (1 - mid)**(n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > 0.05 else (lo, mid)
    return hi


def table(title, groups):
    """groups: {행 이름: Counter(카테고리)} -> 마크다운 표 출력"""
    print(f"\n### {title}")
    print("| 구분 | 쌍 수 | " + " | ".join(CATS) + " |")
    print("|---|---|" + "---|" * len(CATS))
    for name, c in groups.items():
        n = sum(c.values())
        cells = [f"{c[k]} ({c[k] / n:.0%})" for k in CATS]
        print(f"| {name} | {n} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold"); ap.add_argument("pred")
    ap.add_argument("--max-abstain", type=float, default=0.2,
                    help="허용할 최대 '조건부로 물러섬' 비율 (기본 20%%)")
    ap.add_argument("--out", help="결과를 이 파일에도 UTF-8로 저장 (PowerShell의 > 대신 사용)")
    a = ap.parse_args()
    if a.out:
        import sys
        class Tee:
            def __init__(self, *fs): self.fs = fs
            def write(self, x):
                for f in self.fs: f.write(x)
            def flush(self):
                for f in self.fs: f.flush()
        sys.stdout = Tee(sys.stdout, open(a.out, "w", encoding="utf-8"))

    gold = {r["pair_id"]: r for r in csv.DictReader(open(a.gold, encoding="utf-8-sig"))}
    prows = list(csv.DictReader(open(a.pred, encoding="utf-8-sig")))
    pred = {r["pair_id"]: r["pred"].strip() for r in prows}
    fpass = {r["pair_id"]: r.get("filter_pass", "").strip().upper() for r in prows if r.get("filter_pass", "").strip()}
    cat = {k: category(g["gold"], pred.get(k)) for k, g in gold.items()}

    # 1) 전체 + 정답 판정별 4분류
    total = Counter(cat.values())
    by_gold = defaultdict(Counter)
    for k, g in gold.items():
        by_gold[f"정답={g['gold']}"][cat[k]] += 1
    table("판정 결과 4분류", {"전체": total, **dict(sorted(by_gold.items()))})

    # 2) 핵심 지표
    n = len(gold)
    neg = [k for k, g in gold.items() if g["gold"] == NEG]
    fp = [k for k in neg if pred.get(k) == POS]
    decided = [k for k, g in gold.items() if g["gold"] != COND]           # 정답이 확정(가능/불가)인 쌍
    abstain = [k for k in decided if cat[k] == "조건부로 물러섬"]
    answered = [k for k in decided if pred.get(k) in (POS, NEG)]
    sel_acc = sum(cat[k] == "정답" for k in answered) / len(answered) if answered else float("nan")

    print("\n### 핵심 지표")
    print(f"- 정답률: {total['정답'] / n:.1%}")
    print(f"- 조건부로 물러섬 비율: {len(abstain) / len(decided):.1%} (확정 정답 쌍 {len(decided)}개 기준, 상한 {a.max_abstain:.0%})")
    print(f"- 확정 판정 정확도: {sel_acc:.1%} ('가능/불가'로 답한 {len(answered)}건 중 맞힌 비율)")
    print(f"- FPR(틀린 신청 가능 / 실제 부적격): {len(fp)}/{len(neg)} = {len(fp) / len(neg):.1%}"
          f" (95% 상한 {upper95(len(fp), len(neg)):.1%})")
    ok = len(abstain) / len(decided) <= a.max_abstain
    print(f"- 판정: {'통과' if ok else '미달'} - 물러섬 비율이 상한을 {'넘지 않음' if ok else '넘음. FPR이 낮아도 쓸모가 부족한 상태'}")

    # 2-1) 1차 후보 필터 누락률: 정답이 '신청 가능'인데 1차 필터에서 빠진 쌍
    if fpass:
        pos = [k for k, g in gold.items() if g["gold"] == POS and k in fpass]
        miss = [k for k in pos if fpass[k] == "N"]
        neg_f = [k for k, g in gold.items() if g["gold"] == NEG and k in fpass]
        print(f"- 1차 필터 누락률: {len(miss)}/{len(pos)} = {len(miss) / len(pos):.1%}"
              f" (95% 상한 {upper95(len(miss), len(pos)):.1%}) - 정답이 '신청 가능'인데 1차 필터에서 빠진 쌍")
        if neg_f:
            passed = sum(fpass[k] == "Y" for k in neg_f)
            print(f"- 참고: 실제 부적격 쌍의 1차 필터 통과율 {passed}/{len(neg_f)} = {passed / len(neg_f):.0%}"
                  " (낮을수록 후보를 잘 좁힘)")
        if miss:
            by_m = defaultdict(int)
            for k in miss:
                by_m[f"{gold[k]['case_type']}/{gold[k]['target_condition']}"] += 1
            print("  누락된 쌍의 유형/대상 조건:", dict(by_m))

    # 3) 어디서 물러서고 어디서 틀리는가
    by_cond, by_type, by_notice = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for k, g in gold.items():
        by_cond[g["target_condition"]][cat[k]] += 1
        by_type[g["case_type"]][cat[k]] += 1
        by_notice[g["notice_id"]][cat[k]] += 1
    table("대상 조건별", dict(sorted(by_cond.items())))
    table("케이스 유형별 (경계값 포함)", dict(sorted(by_type.items())))
    table("공고별", dict(sorted(by_notice.items())))

    missing = [k for k in gold if k not in pred]
    if missing:
        print(f"\n경고: 예측이 없는 쌍 {len(missing)}개")


if __name__ == "__main__":
    main()
