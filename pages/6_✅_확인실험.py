"""
확인 실험 검증 페이지 - 예측값 vs 실측값 비교 및 모형 타당성 판정.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="확인 실험 검증", page_icon="✅", layout="wide")
st.title("✅ 확인 실험 검증")
st.caption("최적 조건으로 실시한 확인 실험 결과를 예측값과 비교하여 모형 타당성을 판정합니다.")

# ── 이전 분석 결과 자동 로드 ────────────────────────────
ar = st.session_state.get("analysis_result", {})
has_session = bool(ar)

# ─────────────────────────────────────────────────────────
# Step 1. 예측값 입력
# ─────────────────────────────────────────────────────────
st.markdown("### Step 1. 예측값 및 최적 조건")

if has_session:
    st.success("✅ Taguchi 분석 페이지의 결과를 자동으로 불러왔습니다.")
    y_pred_default  = float(ar["y_pred"])
    response_default = ar["response_col"]
    opt_df = ar["opt_df"]

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        response_name = st.text_input("응답변수", value=response_default)
        y_pred = st.number_input("예측 최적값 (ŷ)", value=y_pred_default, format="%.4f")
    with col_i2:
        st.markdown("**최적 인자 조건**")
        st.dataframe(opt_df, use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ Taguchi 분석 페이지에서 분석을 먼저 실행하거나, 아래에 직접 입력하세요.")
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        response_name = st.text_input("응답변수 이름", value="Response_Y")
        y_pred = st.number_input("예측 최적값 (ŷ)", value=0.0, format="%.4f")

# ─────────────────────────────────────────────────────────
# Step 2. 확인 실험 결과 입력
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 2. 확인 실험 결과 입력")
st.caption("최적 조건으로 실시한 확인 실험 값을 입력하세요. 3회 이상 권장.")

n_runs = st.number_input("실험 횟수 (n)", min_value=1, max_value=20, value=3, step=1)

results = []
n_cols_per_row = min(int(n_runs), 5)
rows_needed = (int(n_runs) + n_cols_per_row - 1) // n_cols_per_row

for row_i in range(rows_needed):
    cols_ui = st.columns(n_cols_per_row)
    for col_i in range(n_cols_per_row):
        run_idx = row_i * n_cols_per_row + col_i
        if run_idx < int(n_runs):
            with cols_ui[col_i]:
                val = st.number_input(
                    f"Run {run_idx+1}", value=0.0, format="%.4f",
                    key=f"run_{run_idx}"
                )
                results.append(val)

# ─────────────────────────────────────────────────────────
# Step 3. 검증 분석
# ─────────────────────────────────────────────────────────
if st.button("🔍 검증 분석 실행", type="primary", use_container_width=True):
    arr = np.array(results, dtype=float)
    n   = len(arr)
    mu  = float(np.mean(arr))
    s   = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se  = s / np.sqrt(n) if n > 1 else 0.0

    if n > 1:
        t_cv   = stats.t.ppf(0.975, df=n - 1)
        ci_lo  = mu - t_cv * se
        ci_hi  = mu + t_cv * se
    else:
        ci_lo = ci_hi = mu

    err_abs = abs(mu - y_pred)
    err_pct = err_abs / abs(y_pred) * 100 if y_pred != 0 else float("inf")

    if err_pct <= 15:
        judge     = "PASS"
        judge_fn  = st.success
        judge_msg = "✅ 오차율 ≤15% — 모형 신뢰 가능, 최적 조건 채택 권장"
    elif err_pct <= 30:
        judge     = "WARNING"
        judge_fn  = st.warning
        judge_msg = "⚠️ 오차율 15~30% — 교호작용 의심, 2차 실험 권장"
    else:
        judge     = "FAIL"
        judge_fn  = st.error
        judge_msg = "❌ 오차율 >30% — 모형 부적합, 실험 재설계 필요"

    st.session_state["confirm_result"] = dict(
        arr=arr, n=n, mu=mu, s=s, se=se,
        ci_lo=ci_lo, ci_hi=ci_hi,
        err_abs=err_abs, err_pct=err_pct,
        y_pred=y_pred, judge=judge,
        judge_msg=judge_msg, response_name=response_name,
    )

# ─────────────────────────────────────────────────────────
# Step 4. 결과 표시
# ─────────────────────────────────────────────────────────
if "confirm_result" in st.session_state:
    r = st.session_state["confirm_result"]
    st.markdown("---")
    st.markdown("### Step 3. 검증 결과")

    # 지표 4개
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("예측값 (ŷ)",      f"{r['y_pred']:.4f}")
    c2.metric("실측 평균 (ȳ)",    f"{r['mu']:.4f}")
    c3.metric("오차 (ȳ−ŷ)",      f"{r['mu']-r['y_pred']:+.4f}")
    c4.metric("오차율",           f"{r['err_pct']:.1f}%")

    # 판정
    {"PASS": st.success, "WARNING": st.warning, "FAIL": st.error}[r['judge']](r['judge_msg'])

    # 차트 + 통계 테이블
    cv1, cv2 = st.columns(2)

    with cv1:
        fig = go.Figure()
        # CI 배경
        fig.add_shape(type="rect",
            x0=-0.5, x1=0.5, y0=r['ci_lo'], y1=r['ci_hi'],
            fillcolor="rgba(59,130,246,0.12)", line=dict(width=0))
        # 실측 점
        jitter = np.random.uniform(-0.18, 0.18, r['n'])
        fig.add_trace(go.Scatter(
            x=jitter, y=r['arr'], mode="markers",
            marker=dict(size=13, color="#3B82F6", opacity=0.75),
            name="실측값"))
        # 실측 평균
        fig.add_trace(go.Scatter(
            x=[0], y=[r['mu']], mode="markers",
            marker=dict(size=18, color="#1D4ED8", symbol="diamond"),
            name=f"실측 평균 {r['mu']:.4f}"))
        # 예측값 라인
        fig.add_hline(y=r['y_pred'], line_dash="dash", line_color="red",
                      annotation_text=f"예측값 {r['y_pred']:.4f}",
                      annotation_position="top right")
        # CI 라인
        for y_ci, label in [(r['ci_lo'], "95% CI 하한"), (r['ci_hi'], "95% CI 상한")]:
            fig.add_hline(y=y_ci, line_dash="dot", line_color="#60A5FA",
                          annotation_text=label, annotation_position="bottom right")
        fig.update_layout(
            title=f"예측값 vs 실측값 — {r['response_name']}",
            xaxis=dict(showticklabels=False, range=[-0.55, 0.55]),
            yaxis_title=r['response_name'], height=380)
        st.plotly_chart(fig, use_container_width=True)

    with cv2:
        st.markdown("**통계 요약**")
        summary = pd.DataFrame([
            {"항목": "실험 횟수 (n)",        "값": str(r['n'])},
            {"항목": "실측 평균 (ȳ)",         "값": f"{r['mu']:.4f}"},
            {"항목": "표준편차 (s)",           "값": f"{r['s']:.4f}"},
            {"항목": "표준오차 (SE)",          "값": f"{r['se']:.4f}"},
            {"항목": "95% CI 하한",            "값": f"{r['ci_lo']:.4f}"},
            {"항목": "95% CI 상한",            "값": f"{r['ci_hi']:.4f}"},
            {"항목": "예측값 (ŷ)",            "값": f"{r['y_pred']:.4f}"},
            {"항목": "오차 |ȳ−ŷ|",           "값": f"{r['err_abs']:.4f}"},
            {"항목": "오차율 (%)",             "값": f"{r['err_pct']:.1f}%"},
            {"항목": "판정",                  "값": r['judge']},
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if r['ci_lo'] <= r['y_pred'] <= r['ci_hi']:
            st.success("✅ 예측값이 95% CI 내 포함 → 가법성 모형 타당")
        else:
            st.warning("⚠️ 예측값이 95% CI 밖 → 교호작용 또는 잡음 의심")

    # ── 다음 단계 가이드 ──────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 4. 다음 단계 가이드")

    if r['judge'] == "PASS":
        st.success("""
**✅ 확인 실험 통과 — 최적 조건 채택**

1. 해당 조건으로 양산 기준 확정
2. 관리도(Control Chart) 설정
3. 공정 능력 지수(Cp, Cpk) 산출
        """)
    elif r['judge'] == "WARNING":
        st.warning("""
**⚠️ 2차 분석 권장**

1. 👉 **교호작용 분석** 페이지 → 인자 간 상호작용 확인
2. 최적값이 경계 수준이면 👉 **2차 실험 추천** 페이지 → 범위 조정
3. 범위 조정 후 Taguchi 설계 페이지에서 재실험 설계
        """)
    else:
        st.error("""
**❌ 모형 재설계 필요**

1. 누락 인자 검토 (재료 로트, 온도, 작업자 등)
2. 인자 수준 범위 대폭 변경 후 재실험
3. 교호작용 분석 후 L36 이상 대형 OA 또는 RSM(CCD) 전환 고려
        """)
