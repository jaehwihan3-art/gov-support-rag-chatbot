"""
버전별 핵심 지표를 한 표로 모은다 (반복 개선 루프 기록표).

사용법 (govchat 폴더에서):
    python compare_versions.py                 # 개발셋, data/baseline/ 아래 모든 버전
    python compare_versions.py --split holdout
결과는 화면과 data/baseline/compare_<split>.md 에 저장.
"""
import argparse, csv, glob, os, re

POS, NEG, COND = "신청 가능", "신청 불가", "조건부 가능"


def cat(x):
    """판정 이유를 조건 종류로 묶음 (이유까지 맞았는지 비교용)"""
    x = re.sub(r"\d+$", "", x.strip())
    if x.startswith(("biz", "창업기업", "founded")):
        return "업력"
    if x.startswith(("제외", "업종표")):
        return "제외" if x.startswith("제외") else "industry"
    if x.startswith("기타"):
        return "기타"
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    gold = {r["pair_id"]: r for r in csv.DictReader(open(f"data/pairs/pairs_{a.split}.csv", encoding="utf-8-sig"))}
    ver = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"data/baseline/*/predictions_{a.split}.csv"))
    ver.sort(key=lambda v: int(re.sub(r"\D", "", v) or 0))
    L = [f"## 버전별 핵심 지표 ({a.split}, {len(gold)}쌍)\n",
         "| 버전 | 바꾼 것 | 정답률 | '가능' 맞힘 | FPR | 틀린 신청 불가 | 조건부로 물러섬 | 이유까지 맞은 불가 |",
         "|---|---|---|---|---|---|---|---|"]
    for v in ver:
        pred = {r["pair_id"]: r["pred"] for r in csv.DictReader(open(f"data/baseline/{v}/predictions_{a.split}.csv", encoding="utf-8-sig"))}
        det = {}
        dp = f"data/baseline/{v}/pred_detail_{a.split}.csv"
        if os.path.exists(dp):
            det = {r["pair_id"]: r for r in csv.DictReader(open(dp, encoding="utf-8-sig"))}
        note = ""
        cp = f"data/baseline/{v}/config.json"
        if os.path.exists(cp):
            import json
            note = json.load(open(cp, encoding="utf-8")).get("extract_prompt", "")[:40]
        ids = [k for k in gold if k in pred]
        n = len(ids)
        ok = sum(pred[k] == gold[k]["gold"] for k in ids)
        gp = [k for k in ids if gold[k]["gold"] == POS]
        gn = [k for k in ids if gold[k]["gold"] == NEG]
        sure = [k for k in ids if gold[k]["gold"] != COND]
        fp = sum(pred[k] == POS for k in gn)
        fneg = sum(pred[k] == NEG and gold[k]["gold"] != NEG for k in ids)
        abst = sum(pred[k] == COND for k in sure)
        why = "-"
        if det:
            w = sum(1 for k in gn if pred[k] == NEG and
                    cat(gold[k]["gold_reason"]) in {cat(x) for x in det[k]["pred_reason"].split("|") if x})
            why = f"{w}/{len(gn)}"
        L.append(f"| {v} | {note} | {ok / n:.1%} | {sum(pred[k] == POS for k in gp)}/{len(gp)} | "
                 f"{fp}/{len(gn)} = {fp / len(gn):.1%} | {fneg} ({fneg / n:.0%}) | {abst / len(sure):.1%} | {why} |")
    out = "\n".join(L)
    print(out)
    open(f"data/baseline/compare_{a.split}.md", "w", encoding="utf-8").write(out + "\n")
    print(f"\n저장: data/baseline/compare_{a.split}.md")


if __name__ == "__main__":
    main()
