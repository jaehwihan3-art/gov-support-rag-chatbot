"""6단계 실험 그래프: data/eval/experiments.csv, retrieval_compare.csv -> docs/figures/*.png

  python plot_experiments.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):   # 한글 글꼴
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # 범주 색 1~3번 (순서 고정)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = Path("docs/figures")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def qa_chart():
    rows = [r for r in csv.DictReader(open("data/eval/experiments.csv", encoding="utf-8-sig")) if r["채점"] == "G2"]
    if not rows:
        print("G2로 채점한 실행이 없어 질의응답 그래프를 건너뜀"); return
    labels = [f"{r['실행']}\n{r.get('검색') or 'dense'} · {r['답변']}" for r in rows]
    acc = [float(r["정확도"]) for r in rows]
    ref = [float(r["과잉거절"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.7 * len(rows) + 1.6), sharey=True)
    for ax, vals, title, unit, color in ((axes[0], acc, "답변 정확도 (G2 채점, %)", "%", BLUE),
                                         (axes[1], ref, "과잉 거절 (답이 있는 25문항 중, 건)", "건", ORANGE)):
        y = range(len(rows))
        ax.barh(y, vals, color=color, height=0.55)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, f"{v:g}{unit}", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, max(vals) * 1.18 + 1)
        style(ax)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("질의응답 실험: 같은 채점기(G2)로 본 버전별 결과", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qa_experiments.png", dpi=160)
    print("저장:", OUT / "qa_experiments.png")


def retrieval_chart():
    p = Path("data/eval/retrieval_compare.csv")
    if not p.exists():
        print("retrieval_compare.csv 없음 (python eval_retrieval.py 먼저)"); return
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    ks = [int(c.split("@")[1]) for c in rows[0] if c.startswith("hit@")]
    colors = {"dense": BLUE, "bm25": ORANGE, "hybrid": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, strat in zip(axes, ("A1", "A2")):
        for r in [r for r in rows if r["청킹"] == strat]:
            ys = [float(r[f"hit@{k}"]) for k in ks]
            ax.plot(ks, ys, color=colors[r["검색"]], linewidth=2, marker="o", markersize=5, label=r["검색"])
        ax.set_title(f"{strat} {'고정 길이' if strat == 'A1' else '조항 단위'}", fontsize=10, color=INK, loc="left")
        ax.set_xticks(ks, [f"top-{k}" for k in ks])
        ax.set_xlim(0.5, ks[-1] + 0.5)
        ax.set_ylim(0, 105)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_ylabel("hit@k (%)", color=MUTED)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("검색 방식별 hit@k: 정답 근거가 상위 k개 청크 안에 든 비율 (25문항)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "retrieval_hitk.png", dpi=160)
    print("저장:", OUT / "retrieval_hitk.png")


def judge_chart():
    """자격 판정 버전별 정답률·FPR (data/baseline/compare_dev.md 표에서 읽음)"""
    import re
    p = Path("data/baseline/compare_dev.md")
    if not p.exists():
        print("compare_dev.md 없음 (python compare_versions.py 먼저)"); return
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(v\d+)\s*\|[^|]*\|\s*([\d.]+)%\s*\|[^|]*\|\s*\d+/\d+\s*=\s*([\d.]+)%", line)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    if not rows:
        print("compare_dev.md에서 버전 행을 못 읽음"); return
    vs = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, vals, title, color in ((axes[0], [r[1] for r in rows], "정답률 (%)", BLUE),
                                   (axes[1], [r[2] for r in rows], "틀린 신청 가능 FPR (%, 낮을수록 좋음)", ORANGE)):
        ax.bar(vs, vals, color=color, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:g}", ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, max(vals) * 1.15)
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.suptitle("자격 판정 버전별 결과 (개발셋 240쌍)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "judge_versions.png", dpi=160)
    print("저장:", OUT / "judge_versions.png")


if __name__ == "__main__":
    qa_chart()
    retrieval_chart()
    judge_chart()
