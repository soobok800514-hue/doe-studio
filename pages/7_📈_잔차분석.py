"""
잔차 분석 페이지 - Taguchi 가법 모형의 잔차 진단 4종.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(page_title="잔차 분석", page_icon="📈", layout="wide")
st.title("📈 잔차 분석 (Residual Analysis)")
st.caption("Taguchi 가법 모형의 잔차를 진단하여 모형 적합성과 이상치를 확인합니다.")

# ── 세션 데이터 로드 ──────────────────────────────────────
ar = st.session_state.get("analysis_result", {})

if not ar:
    st.warning("⚠️ Taguchi 분석 페이지에서 먼저 분석을 실행하세요.")
    st.stop()

df_clean    = ar["df_clean"]
factor_cols = ar["factor_cols"]
response_col = ar["response_col"]

st.info(f"**분석 대상** — 응답: `{response_col}` | 인자: {', '.join(f'`{f}`' for f in factor_cols)} | {len(df_clean)}행")

# ── 가법 모형으로 적합값 / 잔차 계산 ─────────────────────
# ŷ_i = grand_mean + Σ_j (level_mean_j(i) − grand_mean)
grand_mean = df_clean[response_col].mean()

fitted = []
for _, row in df_clean.iterrows():
    y_hat = grand_mean
    for fc in factor_cols:
        lv_mean = df_clean[df_clean[fc] == row[fc]][response_col].mean()
        y_hat += (lv_mean - grand_mean)
    fitted.append(y_hat)

fitted    = np.array(fitted)
actual    = df_clean[response_col].values.astype(float)
residuals = actual - fitted
order     = np.arange(1, len(residuals) + 1)

# 표준화 잔차
std_res = residuals / (residuals.std(ddof=1) + 1e-12)

st.markdown("---")
st.markdown("### 잔차 진단 4종 플롯")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "① 정규확률도 (Normal Probability Plot)",
        "② 잔차 vs 적합값",
        "③ 잔차 vs 실험 순서",
        "④ 잔차 히스토그램",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

BLUE = "#3B82F6"
RED  = "#EF4444"

# ① 정규확률도 (Q-Q Plot)
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
fig.add_trace(go.Scatter(
    x=osm, y=osr, mode="markers",
    marker=dict(color=BLUE, size=9, opacity=0.8),
    name="잔차", showlegend=False,
), row=1, col=1)
x_line = np.array([osm.min(), osm.max()])
fig.add_trace(go.Scatter(
    x=x_line, y=slope * x_line + intercept,
    mode="lines", line=dict(color=RED, dash="dash", width=1.5),
    name="정규선", showlegend=False,
), row=1, col=1)
fig.update_xaxes(title_text="이론적 분위수", row=1, col=1)
fig.update_yaxes(title_text="샘플 분위수", row=1, col=1)

# ② 잔차 vs 적합값
fig.add_trace(go.Scatter(
    x=fitted, y=residuals, mode="markers",
    marker=dict(color=BLUE, size=9, opacity=0.8),
    showlegend=False,
), row=1, col=2)
fig.add_hline(y=0, line_dash="dash", line_color=RED, row=1, col=2)
# 2σ 밴드
s_res = residuals.std(ddof=1)
for sign in [1, -1]:
    fig.add_hline(y=sign * 2 * s_res, line_dash="dot",
                  line_color="gray", row=1, col=2)
fig.update_xaxes(title_text="적합값 (ŷ)", row=1, col=2)
fig.update_yaxes(title_text="잔차 (e)", row=1, col=2)

# ③ 잔차 vs 순서 (시간 흐름)
fig.add_trace(go.Scatter(
    x=order, y=residuals, mode="lines+markers",
    line=dict(color=BLUE, width=1.5),
    marker=dict(size=8), showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color=RED, row=2, col=1)
for sign in [1, -1]:
    fig.add_hline(y=sign * 2 * s_res, line_dash="dot",
                  line_color="gray", row=2, col=1)
fig.update_xaxes(title_text="실험 순서", row=2, col=1)
fig.update_yaxes(title_text="잔차 (e)", row=2, col=1)

# ④ 히스토그램
fig.add_trace(go.Histogram(
    x=residuals, nbinsx=max(5, len(residuals)//2),
    marker_color=BLUE, opacity=0.75, showlegend=False,
), row=2, col=2)
fig.update_xaxes(title_text="잔차 (e)", row=2, col=2)
fig.update_yaxes(title_text="빈도", row=2, col=2)

fig.update_layout(height=700, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ── 잔차 통계 요약 ────────────────────────────────────────
st.markdown("---")
st.markdown("### 잔차 통계 요약")

_, sw_p  = stats.shapiro(residuals) if len(residuals) >= 3 else (None, None)
_, dw_p  = None, None  # Durbin-Watson은 수동 계산
dw_stat  = float(np.sum(np.diff(residuals)**2) / np.sum(residuals**2)) if len(residuals) > 1 else None

col_s1, col_s2 = st.columns(2)
with col_s1:
    stat_df = pd.DataFrame([
        {"항목": "잔차 평균",        "값": f"{residuals.mean():.4f}"},
        {"항목": "잔차 표준편차",     "값": f"{s_res:.4f}"},
        {"항목": "최대 잔차",         "값": f"{residuals.max():.4f}"},
        {"항목": "최소 잔차",         "값": f"{residuals.min():.4f}"},
        {"항목": "Shapiro-Wilk p값", "값": f"{sw_p:.4f}" if sw_p else "-"},
        {"항목": "Durbin-Watson",    "값": f"{dw_stat:.4f}" if dw_stat else "-"},
    ])
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

with col_s2:
    st.markdown("**해석 기준**")
    # 정규성
    if sw_p and sw_p >= 0.05:
        st.success(f"✅ Shapiro-Wilk p={sw_p:.3f} ≥ 0.05 → 정규성 만족")
    elif sw_p:
        st.warning(f"⚠️ Shapiro-Wilk p={sw_p:.3f} < 0.05 → 정규성 의심")

    # Durbin-Watson (2 근처이면 자기상관 없음)
    if dw_stat:
        if 1.5 <= dw_stat <= 2.5:
            st.success(f"✅ Durbin-Watson={dw_stat:.3f} (1.5~2.5) → 자기상관 없음")
        else:
            st.warning(f"⚠️ Durbin-Watson={dw_stat:.3f} → 실험 순서에 계통 오차 의심")

    # 이상치
    outliers = np.where(np.abs(std_res) > 2)[0]
    if len(outliers) == 0:
        st.success("✅ |표준화 잔차| > 2 인 이상치 없음")
    else:
        st.warning(f"⚠️ 이상치 의심 Run: {[int(i)+1 for i in outliers]}")

# ── 잔차 데이터 테이블 ────────────────────────────────────
with st.expander("📋 잔차 상세 데이터"):
    res_df = df_clean[factor_cols + [response_col]].copy().reset_index(drop=True)
    res_df["적합값 (ŷ)"]  = fitted.round(4)
    res_df["잔차 (e)"]    = residuals.round(4)
    res_df["표준화 잔차"] = std_res.round(4)
    st.dataframe(res_df, use_container_width=True)
