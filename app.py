"""7단계: 정부 지원사업 공고 챗봇 (Streamlit)

  streamlit run app.py

탭 1 공고 질문 : 공고 원문 근거로 답하고 인용 문장을 보여줌 (hybrid 검색 + 답변 Q1 + 인용 검증 가드)
탭 2 자격 확인 : 공고를 고르고 내 정보 + 질문에 답하면 판정 v7로 신청 가능 여부와 근거를 보여줌
탭 3 로그      : data/app_logs.db 에 쌓인 질문·답변·판정·피드백
"""
import time, uuid
from datetime import date

import pandas as pd
import streamlit as st

import service as S

st.set_page_config(page_title="정부 지원사업 공고 도우미", page_icon="📄", layout="wide")
if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:8]
    st.session_state.chat = []          # [{"q", "res", "log_id", "ms"}]

st.title("정부 지원사업 공고 도우미")
st.caption("소상공인·초기 창업자용 · 2026년 공고 원문(기업마당·K-Startup 169건)을 근거로 답합니다. "
           "참고용이며, 최종 자격은 공고 기관에 확인하세요.")

tab_qa, tab_judge, tab_log = st.tabs(["공고 질문", "자격 확인", "로그"])

# ---------------- 탭 1: 공고 질문 ----------------
with tab_qa:
    st.write("공고 내용을 물어보세요. 예: *청년일자리도약장려금은 기업에 얼마를 지원해?*, *대전 라이더 배달비는 건당 얼마야?*")
    for i, turn in enumerate(st.session_state.chat):
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            res = turn["res"]
            if res.get("error"):
                st.error(f"오류: {res['error']}")
                continue
            st.write(res["answer"])
            if res.get("explore"):
                st.caption("사업을 찾는 질문으로 보고 여러 공고에서 찾았습니다.")
            if not res["found"]:
                st.caption("검색된 공고에서 근거 문장을 찾지 못해 답하지 않았습니다." + (" (인용 검증에서 걸러짐)" if res["guard_fired"] else ""))
            if res["quotes"]:
                with st.expander(f"근거 원문 {len(res['quotes'])}개"):
                    by_id = {c["chunk_id"]: c for c in res["hits"]}
                    for qt in res["quotes"]:
                        c = by_id.get(qt["id"], {})
                        st.markdown(f"> {qt['text']}")
                        st.caption(f"{c.get('title', '')}" + (f" · {c['page']}쪽" if c.get("page") else "") + f" · {qt['id']}")
            with st.expander(f"검색된 공고 조각 {len(res['hits'])}개"):
                for c in res["hits"]:
                    st.markdown(f"**{c['title'][:60]}** · `{c['chunk_id']}`")
                    st.text(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
            cols = st.columns([1, 1, 8])
            fb = turn.get("fb")
            if cols[0].button("👍", key=f"up{i}", disabled=fb is not None):
                S.set_feedback("qa_logs", turn["log_id"], 1); turn["fb"] = 1; st.rerun()
            if cols[1].button("👎", key=f"down{i}", disabled=fb is not None):
                S.set_feedback("qa_logs", turn["log_id"], -1); turn["fb"] = -1; st.rerun()
            cols[2].caption(f"응답 {turn['ms'] / 1000:.1f}초" + (" · 피드백 고마워요" if fb is not None else ""))

    q = st.chat_input("공고에 대해 물어보세요")
    if q:
        t0 = time.time()
        try:
            with st.spinner("공고를 찾아 읽는 중…"):
                res = S.answer_question(q)
            err = ""
        except Exception as e:          # API 키·네트워크 오류도 로그에 남김
            res, err = {"answer": "", "found": False, "quotes": [], "hits": []}, f"{type(e).__name__}: {e}"
            res["error"] = err
        ms = int((time.time() - t0) * 1000)
        log_id = S.log_qa(st.session_state.sid, q, res, ms, err)
        st.session_state.chat.append({"q": q, "res": res, "log_id": log_id, "ms": ms})
        st.rerun()

# ---------------- 탭 2: 자격 확인 ----------------
with tab_judge:
    notices = S.list_notices()
    label = {n: f"[{n}] {t}" for n, t, _ in notices}
    n = st.selectbox("공고 선택", [x[0] for x in notices], format_func=lambda x: label[x])
    try:
        ex = S.prepare_conditions(n)
    except Exception as e:
        st.error(str(e)); st.stop()

    st.subheader("내 정보")
    c1, c2, c3 = st.columns(3)
    size = c1.selectbox("사업자 유형", S.SIZES, index=1)
    birth = c1.date_input("대표자 생년월일", value=date(1990, 1, 1), min_value=date(1940, 1, 1), max_value=date.today())
    sido = c2.selectbox("사업장 소재지(시도)", S.SIDOS, index=S.SIDOS.index("대전"))
    gu = c2.text_input("시군구 (예: 서구, 유성구)", value="")
    ksic = c3.text_input("업종 코드 (한국표준산업분류 5자리, 예: 56111 한식 음식점, 56221 커피 전문점)", value="",
                         help="사업자등록증명원의 업종 코드. 비워 두면 업종 조건은 '확인 필요'로 표시합니다.")
    pre = size == "예비창업자"
    opened = c3.date_input("개업일", value=date(2022, 1, 1), min_value=date(1970, 1, 1), max_value=date.today(), disabled=pre)
    rev_known = c3.checkbox("매출을 알고 있음", value=False, disabled=pre)
    revenue = c3.number_input("직전 연도 매출 (만원)", min_value=0, value=0, step=1000, disabled=pre or not rev_known)
    prof = {"size": size, "birth": birth, "sido": sido, "gu": gu.strip(), "ksic": ksic.strip(),
            "opened": None if pre else opened, "revenue_manwon": None if (pre or not rev_known) else revenue}
    r = S.profile_row(prof)
    ex_p, qs = S.questions_for(ex, r)

    with st.expander("이 공고에서 확인된 기본 조건 (코드가 자동 판정)"):
        rows = [("신청 대상", ", ".join(S.B.as_list(ex.get("legal_terms"))) or "제한 없음"),
                ("소재지", ", ".join(S.B.as_list(ex.get("region"))) or "제한 없음")]
        for f, nm in (("age", "대표자 나이"), ("biz_months", "업력"), ("revenue", "매출")):
            v = ex.get(f) if isinstance(ex.get(f), dict) else None
            if v and any(v.get(k) not in (None, "") for k in ("min", "max", "birth_year_min", "founded_after", "founded_before")):
                rows.append((nm, v.get("evidence", "") or str({k: v[k] for k in ("min", "max") if v.get(k) is not None})))
        ind = ex.get("industry") if isinstance(ex.get("industry"), dict) else {}
        if ind.get("allowed_ksic"):
            rows.append(("허용 업종 코드", ", ".join(map(str, ind["allowed_ksic"]))[:200]))
        st.table(pd.DataFrame(rows, columns=["조건", "공고 내용"]))

    st.subheader(f"확인 질문 {len(qs)}개")
    st.caption("모르면 '모름'으로 두세요. '모름'이 있으면 '조건부 가능'으로 안내합니다.")
    with st.form(f"judge_{n}"):
        answers = {}
        for idx, kind, text, ev in qs:
            answers[idx] = st.radio(f"[{kind}] {text}", ["모름", "예", "아니오"], horizontal=True, key=f"q{n}_{idx}",
                                    help=f"근거: {ev}" if ev else None)
        go = st.form_submit_button("판정하기", type="primary")
    if go:
        t0 = time.time()
        pred, why = S.judge(ex_p, r, answers)
        ms = int((time.time() - t0) * 1000)
        reasons = S.explain(ex_p, why)
        lid = S.log_judge(st.session_state.sid, n, label[n], r, [q[2] for q in qs], {str(k): v for k, v in answers.items()}, pred, why, ms)
        st.session_state[f"judge_res_{n}"] = (pred, reasons, lid)
    if f"judge_res_{n}" in st.session_state:
        pred, reasons, lid = st.session_state[f"judge_res_{n}"]
        {"신청 가능": st.success, "조건부 가능": st.warning, "신청 불가": st.error}[pred](f"판정: **{pred}**")
        if reasons:
            st.write("확인 필요한 항목" if pred == "조건부 가능" else "걸린 조건")
            for t, ev in reasons:
                st.markdown(f"- {t}" + (f"  \n  <small>근거: {ev}</small>" if ev else ""), unsafe_allow_html=True)
        st.caption("판정은 공고 원문에서 뽑은 조건과 입력한 정보로 계산한 참고 결과입니다. 공고 기관의 최종 판단과 다를 수 있습니다.")
        c = st.columns([1, 1, 8])
        if c[0].button("👍", key=f"jup{lid}"):
            S.set_feedback("judge_logs", lid, 1); st.toast("피드백 고마워요")
        if c[1].button("👎", key=f"jdown{lid}"):
            S.set_feedback("judge_logs", lid, -1); st.toast("피드백 고마워요")

# ---------------- 탭 3: 로그 ----------------
with tab_log:
    qa = pd.DataFrame(S.read_logs("qa_logs"))
    jl = pd.DataFrame(S.read_logs("judge_logs"))
    st.subheader("공고 질문")
    if qa.empty:
        st.info("아직 질문 로그가 없습니다.")
    else:
        m = st.columns(5)
        m[0].metric("질문 수", len(qa))
        m[1].metric("답한 비율", f"{qa['found'].mean() * 100:.0f}%")
        m[2].metric("평균 응답 시간", f"{qa['latency_ms'].mean() / 1000:.1f}초")
        fb = qa["feedback"].dropna()
        m[3].metric("👍 비율", f"{(fb > 0).mean() * 100:.0f}%" if len(fb) else "-", help=f"피드백 {len(fb)}건")
        m[4].metric("오류", int((qa["error"].fillna("") != "").sum()))
        st.dataframe(qa[["id", "ts", "question", "answer", "found", "guard_fired", "latency_ms", "tokens", "feedback", "error"]],
                     width="stretch", hide_index=True)
    st.subheader("자격 확인")
    if jl.empty:
        st.info("아직 판정 로그가 없습니다.")
    else:
        m = st.columns(3)
        m[0].metric("판정 수", len(jl))
        m[1].metric("결과 분포", " / ".join(f"{k} {v}" for k, v in jl["pred"].value_counts().items()))
        fb = jl["feedback"].dropna()
        m[2].metric("👍 비율", f"{(fb > 0).mean() * 100:.0f}%" if len(fb) else "-")
        st.dataframe(jl[["id", "ts", "title", "pred", "reasons", "latency_ms", "feedback"]], width="stretch", hide_index=True)
    st.caption(f"DB: {S.DB} · 세션 {st.session_state.sid}")
