"""7단계 서비스 로그 분석: data/app_logs.db -> docs/service_log_report.md

  python analyze_logs.py

질문 수, 답한 비율, 탐색형 질문 비율, 인용 가드 작동, 응답 시간(중앙값·90%), 피드백, 오류,
판정 결과 분포, 답하지 못한 질문 목록을 정리한다. 질문 원문에 개인 정보가 있을 수 있어 보고서에는 앞 40자만 싣는다.
"""
import json, sqlite3, statistics
from collections import Counter
from pathlib import Path

DB, OUT = Path("data/app_logs.db"), Path("docs/service_log_report.md")


def q(con, sql):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(sql)]


def pct(a, b):
    return f"{a}/{b} ({a / b * 100:.0f}%)" if b else "-"


def main():
    if not DB.exists():
        raise SystemExit("data/app_logs.db 없음: streamlit run app.py 로 질문을 먼저 해보세요.")
    con = sqlite3.connect(DB)
    qa = q(con, "SELECT * FROM qa_logs ORDER BY id")
    jl = q(con, "SELECT * FROM judge_logs ORDER BY id")
    L = ["# 서비스 로그 분석\n", f"기간 {qa[0]['ts'][:16] if qa else '-'} ~ {qa[-1]['ts'][:16] if qa else '-'} · 세션 {len({r['session'] for r in qa + jl})}개\n"]
    if qa:
        ms = sorted(r["latency_ms"] for r in qa if not r["error"])
        fb = [r["feedback"] for r in qa if r["feedback"] is not None]
        ex = [r for r in qa if r.get("explore")]
        L += ["## 공고 질문\n", "| 지표 | 값 |", "|---|---|",
              f"| 질문 수 | {len(qa)} |",
              f"| 답한 비율 (근거 인용 확인됨) | {pct(sum(r['found'] for r in qa), len(qa))} |",
              f"| 탐색형 질문 (사업 찾기) | {pct(len(ex), len(qa))} |",
              f"| 인용 검증 가드가 거절로 바꿈 | {pct(sum(r['guard_fired'] for r in qa), len(qa))} |",
              f"| 응답 시간 중앙값 / 90% | {statistics.median(ms) / 1000:.1f}초 / {ms[int(len(ms) * 0.9) - 1 if len(ms) > 1 else 0] / 1000:.1f}초 |" if ms else "| 응답 시간 | - |",
              f"| 평균 토큰 | {statistics.mean(r['tokens'] or 0 for r in qa):,.0f} |",
              f"| 피드백 👍 / 👎 | {fb.count(1)} / {fb.count(-1)} (피드백 {len(fb)}건) |",
              f"| 오류 | {sum(1 for r in qa if r['error'])} |"]
        miss = [r for r in qa if not r["found"]]
        if miss:
            L += ["\n### 답하지 못한 질문\n", "| id | 질문(앞 40자) | 가드 | 인용 확인 | 원인 추정 |", "|---|---|---|---|---|"]
            for r in miss:
                raw = json.loads(r.get("raw_output") or "{}")
                why = ("가드: 인용 문장이 원문과 다름" if r["guard_fired"] else
                       "모델이 근거 없음으로 판단" if raw.get("found") is False or not raw else "-")
                L.append(f"| {r['id']} | {r['question'][:40]} | {'O' if r['guard_fired'] else ''} | {r.get('quote_ok') or 0}/{r.get('quote_all') or 0} | {why} |")
        bad = [r for r in qa if r["feedback"] == -1]
        if bad:
            L += ["\n### 👎 받은 답변\n", "| id | 질문(앞 40자) | 답변(앞 60자) |", "|---|---|---|"]
            L += [f"| {r['id']} | {r['question'][:40]} | {r['answer'][:60].replace(chr(10), ' ')} |" for r in bad]
    if jl:
        c = Counter(r["pred"] for r in jl)
        reasons = Counter(x for r in jl for x in json.loads(r["reasons"] or "[]"))
        fb = [r["feedback"] for r in jl if r["feedback"] is not None]
        L += ["\n## 자격 확인\n", "| 지표 | 값 |", "|---|---|", f"| 판정 수 | {len(jl)} |",
              f"| 결과 | " + " / ".join(f"{k} {v}" for k, v in c.most_common()) + " |",
              f"| 판정한 공고 | {len({r['notice'] for r in jl})}개 |",
              f"| 피드백 👍 / 👎 | {fb.count(1)} / {fb.count(-1)} |",
              f"| 자주 걸린 조건 | " + ", ".join(f"{k}({v})" for k, v in reasons.most_common(5)) + " |"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L) + f"\n\n저장: {OUT}")


if __name__ == "__main__":
    main()
