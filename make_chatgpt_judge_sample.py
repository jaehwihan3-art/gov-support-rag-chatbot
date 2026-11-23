"""자격 판정 홀드아웃에서 ChatGPT(웹 검색) 비교용 10쌍을 무작위로 뽑고, 붙여넣을 질문 문장을 만든다.

  python make_chatgpt_judge_sample.py  -> data/eval/chatgpt_judge/sample.csv, prompts.md

뽑는 방법: 정답 비율대로 층을 나눠 무작위 (가능 4, 불가 4, 조건부 2, seed 2026). 우리 v7 결과는 보지 않고 뽑는다.
"""
import csv, json, random
from pathlib import Path

OUT = Path("data/eval/chatgpt_judge")
P = list(csv.DictReader(open("data/pairs/pairs_holdout.csv", encoding="utf-8-sig")))
titles = {}
for f in Path("data/answer_key").glob("answer_key_*.json"):
    d = json.load(open(f, encoding="utf-8"))
    titles[f.stem[-2:]] = d.get("title", "")
rng = random.Random(2026)
pick = []
for gold, n in (("신청 가능", 4), ("신청 불가", 4), ("조건부 가능", 2)):
    pick += rng.sample([r for r in P if r["gold"] == gold], n)
pick.sort(key=lambda r: r["pair_id"])

OUT.mkdir(parents=True, exist_ok=True)
L = ["# ChatGPT 자격 판정 비교용 질문 10개\n", "문항마다 **새 대화**를 열고, 아래 상자 안 내용을 그대로 붙여넣는다.\n"]
rows = []
for i, r in enumerate(pick, 1):
    t = titles[r["pair_id"][:2]]
    q = (f"아래는 내 정보야 (기준일 2026년 11월 11일).\n\n{r['profile_text']}\n\n"
         f"이 정보로 「{t}」에 신청할 수 있는지 판단해줘.\n"
         "'신청 가능', '신청 불가', '조건부 가능(정보가 더 있어야 판단 가능)' 중 하나로 먼저 답하고, 이유를 공고 조건과 함께 알려줘.")
    L += [f"## {i}. {r['pair_id']}\n", "```", q, "```\n"]
    rows.append({"no": i, "pair_id": r["pair_id"], "notice": t, "gold": r["gold"], "gold_reason": r["gold_reason"],
                 "ChatGPT 판정(가능/불가/조건부)": "", "ChatGPT 이유 요약": "", "맞음(O/X)": ""})
(OUT / "prompts.md").write_text("\n".join(L), encoding="utf-8")
with open(OUT / "sample.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\n".join(f"{x['no']}. {x['pair_id']} 정답={x['gold']} ({x['gold_reason']}) {x['notice'][:40]}" for x in rows))
print(f"저장: {OUT}/prompts.md, sample.csv")
