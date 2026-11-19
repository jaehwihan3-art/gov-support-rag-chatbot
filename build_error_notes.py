"""5단계 오답노트: 틀린 사례를 원인 유형별로 집계하고 대표 사례를 뽑는다.

  python build_error_notes.py

입력  data/baseline/v5/pred_detail_dev.csv      자격 판정 v5 (개발셋 240쌍) 질문별 결과
      data/baseline/v5/extract/<공고>.json      v5가 공고에서 뽑은 조건 (원인 확인용)
      data/eval/runs/A2_k5/results.csv          질의응답 A2 결과 (+ human_grading.csv 사람 판정)
      data/eval/runs/A1_k5/results.csv          질의응답 A1 결과 (LLM 판정)
출력  data/error_notes/judge_errors.csv         판정 오류 74건 + 원인 유형
      data/error_notes/qa_errors.csv            질의응답 오류 + 원인 유형
      data/error_notes/error_tally.md           유형별 집계표

판정 쪽 원인 분류는 아래 RULES를 위에서부터 적용한다. 규칙은 틀린 사례를 공고별로 직접 열어
(추출 JSON + 시뮬레이터 질문·응답) 확인한 뒤 만들었다. 새 버전 결과에 쓰면 '미분류'가 늘 수 있으니 그때 규칙을 추가한다.
"""
import csv, re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("data/error_notes")

# (원인 유형, 단계, 조건 함수) - 위에서부터 처음 맞는 것
RULES = [
    ("대상 혼동: 근로자·청년 조건을 신청자 조건으로 추출", "추출",
     lambda n, r: (n == "08" and r["pred_reason"].startswith("birth_year")) or (n == "22" and r["pred_reason"] == "제외1")),
    ("업종 범위 오추출: 일부 트랙·분야의 업종표를 전체 조건으로", "추출",
     lambda n, r: n in ("06", "14") and r["pred_reason"] == "industry"),
    ("선택지 목록 오추출: K-Startup 업력 구간 목록을 업력 상한으로", "추출",
     lambda n, r: n == "28" and r["pred_reason"] == "biz_months"),
    ("업력 예외 조항 누락 (신산업 분야 10년·수혜기업 면제)", "추출",
     lambda n, r: n in ("01", "18", "02") and "창업기업(업력)" in r["pred_reason"]),
    ("선택지 목록 오추출: K-Startup 업력 구간 목록을 업력 상한으로", "추출",
     lambda n, r: n == "28" and "창업기업(업력)" in r["pred_reason"]),
    ("가점·우대·배정 몫·트랙 조건을 필수 요건으로 추출", "추출",
     lambda n, r: (n == "25" and "창업기업(업력)" in r["pred_reason"]) or (n == "21" and r["pred_reason"] == "biz_months")),
    ("가점·우대·배정 몫·트랙 조건을 필수 요건으로 추출", "추출",
     lambda n, r: (n == "18" and "기타5" in r["pred_reason"]) or (n == "15" and "기타1" in r["pred_reason"])
     or (n == "12" and r["pred_reason"] == "age")),
    ("신청 대상에서 예비창업자 누락", "추출",
     lambda n, r: r["pred_reason"].startswith("기업규모") and r["case_type"] == "typical" and r["pred"] == "신청 불가"),
    ("자격이 아닌 문장(절차·의무·모호한 표현)을 자격 질문으로 만듦", "질문 생성",
     lambda n, r: (n, re.sub(r"\|.*", "", r["pred_reason"])) in {("03", "기타11"), ("05", "기타7"), ("25", "기타2"), ("09", "기타6")}),
    ("기준값 없는 질문(저신용·신용평점 일정기준)에 응답 오류", "질문 생성",
     lambda n, r: (n == "27" and "기타1" in r["pred_reason"]) or (n == "12" and "기타6" in r["pred_reason"])),
    ("제외 사유 질문에 '모름' 응답 → 조건부로 물러섬", "응답(시뮬레이터)",
     lambda n, r: r["pred"] == "조건부 가능" and "제외" in r["pred_reason"]),
    ("정보가 없는 항목에 응답이 '예'로 추측", "응답(시뮬레이터)",
     lambda n, r: n == "03" and r["pair_id"] == "03-P08"),
    ("자격 조건·제외 사유 추출 누락", "추출",
     lambda n, r: r["pred"] == "신청 가능"),
]


def outcome(r):
    if r["맞음"] == "O":
        return "정답"
    return {"조건부 가능": "조건부로 물러섬", "신청 가능": "틀린 신청 가능"}.get(r["pred"], "틀린 신청 불가")


def judge():
    rows = list(csv.DictReader(open("data/baseline/v5/pred_detail_dev.csv", encoding="utf-8-sig")))
    bad = []
    for r in rows:
        if r["맞음"] == "O":
            continue
        n = r["pair_id"][:2]
        cause, stage = next(((c, s) for c, s, f in RULES if f(n, r)), ("미분류", "-"))
        bad.append({"pair_id": r["pair_id"], "공고": n, "결과": outcome(r), "gold": r["gold"], "pred": r["pred"],
                    "gold_reason": r["gold_reason"], "pred_reason": r["pred_reason"], "case_type": r["case_type"],
                    "원인 유형": cause, "단계": stage})
    return rows, bad


def qa(name, human=False):
    d = Path("data/eval/runs") / name
    res = list(csv.DictReader(open(d / "results.csv", encoding="utf-8-sig")))
    hg = {}
    if human and (d / "human_grading.csv").exists():
        for r in csv.DictReader(open(d / "human_grading.csv", encoding="utf-8-sig")):
            hg[r["qid"]] = r["사람 판정(정답/부분정답/오답)"].strip()
    out = []
    for r in res:
        g = hg.get(r["qid"]) or r["llm_grade"]
        if g == "정답":
            continue
        found = r["found"] == "True"
        if r["type"] == "not_in_doc":
            cause = "모르는데 답함 (공고에 없는 내용을 다른 사실로 대체)" if found else "확인 불가는 했지만 안내 부족"
        elif not found:
            cause = "근거를 찾고도 거절 (과잉 거절)" if r["hit_rank"] else "검색 실패 → 거절"
            if r["type"] == "trap":
                cause += " · 함정 전제 미교정"
        else:
            cause = "정보 누락 (핵심 조건 일부 빠짐)" if g == "부분정답" else "사실 오류·질문 오해"
        out.append({"run": name, "qid": r["qid"], "type": r["type"], "판정": g, "found": found,
                    "근거 순위": r["hit_rank"] or ("-" if not r["evidence"] else "top-5 밖"),
                    "원인 유형": cause, "question": r["question"], "answer": r["answer"]})
    return res, out


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, x)) + " |" for x in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, jb = judge()
    with open(OUT / "judge_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(jb[0])); w.writeheader(); w.writerows(jb)

    L = ["# 오류 집계표 (자동 생성: build_error_notes.py)\n",
         f"## 1. 자격 판정 v5 (개발셋 {len(rows)}쌍 중 오류 {len(jb)}건)\n", "### 원인 유형별\n"]
    oc = ["조건부로 물러섬", "틀린 신청 가능", "틀린 신청 불가"]
    by = defaultdict(Counter)
    notices = defaultdict(set)
    for b in jb:
        by[(b["원인 유형"], b["단계"])][b["결과"]] += 1
        notices[(b["원인 유형"], b["단계"])].add(b["공고"])
    tr = sorted(by.items(), key=lambda x: -sum(x[1].values()))
    L += table(["원인 유형", "단계", "건수", "비율"] + oc + ["관련 공고"],
               [[c, s, sum(v.values()), f"{sum(v.values()) / len(jb) * 100:.0f}%"] + [v[o] for o in oc] + [", ".join(sorted(notices[(c, s)]))]
                for (c, s), v in tr])
    st = Counter(b["단계"] for b in jb)
    L += ["\n### 단계별\n"] + table(["단계", "건수", "비율"], [[s, n, f"{n / len(jb) * 100:.0f}%"] for s, n in st.most_common()])
    nc = Counter(b["공고"] for b in jb)
    top = nc.most_common(10)
    L += [f"\n### 공고별 집중도\n", f"오류 {len(jb)}건이 공고 {len(nc)}개에서 나왔고, 상위 10개 공고가 {sum(n for _, n in top)}건({sum(n for _, n in top) / len(jb) * 100:.0f}%)을 차지한다. "
          "공고 하나의 추출 실수가 그 공고의 프로필 여러 쌍을 함께 틀리게 만든다.\n"]
    L += table(["공고", "오류 수", "주된 원인"], [[n, k, Counter(b["원인 유형"] for b in jb if b["공고"] == n).most_common(1)[0][0]] for n, k in top])

    L += ["\n## 2. 질의응답 (30문항)\n"]
    allq = []
    for name, human in (("A2_k5", True), ("A1_k5", False)):
        res, qb = qa(name, human)
        allq += qb
        L += [f"\n### {name} ({'사람 판정' if human else 'LLM 판정'} 기준, 오류 {len(qb)}/{len(res)})\n"]
        c = Counter(q["원인 유형"] for q in qb)
        L += table(["원인 유형", "건수", "문항"], [[k, v, ", ".join(q["qid"] for q in qb if q["원인 유형"] == k)] for k, v in c.most_common()])
    with open(OUT / "qa_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(allq[0])); w.writeheader(); w.writerows(allq)
    (OUT / "error_tally.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {OUT}/judge_errors.csv, qa_errors.csv, error_tally.md")


if __name__ == "__main__":
    main()
