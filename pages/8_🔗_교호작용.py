"""
교호작용 분석 페이지 - 인자 간 상호작용 플롯 및 해석.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="교호작용 분석", page_icon="🔗", layout="wide")
st.title("🔗 교호작용 분석 (Interaction Analysis)")
st.caption("인자 간 상호작용을 시각화합니다. 선이 평행하면 교호작용 없음, 교차하면 교호작용 존재.")

# ── 세션 로드 ──────────────────────────────────────────────
ar = st.session_state.get("analysis_result", {})
if not ar:
    st.warning("⚠️ Taguchi 분석 페이지에서 먼저 분석을 실행하세요.")
    st.stop()

df_clean     = ar["df_clean"]
factor_cols  = ar["factor_cols"]
response_col = ar["response_col"]
level_map    = ar["level_map"]

if len(factor_cols) < 2:
    st.error("교호작용 분석에는 인자가 2개 이상 필요합니다.")
    st.stop()

st.info(f"**응답**: `{response_col}` | **인자**: {', '.join(f'`{f}`' for f in factor_cols)}")

# ── 교호작용 플롯 생성 ────────────────────────────────────
pairs = list(combinations(factor_cols, 2))
st.markdown(f"### 교호작용 플롯 ({len(pairs)}쌍)")
st.caption("**해석**: 각 선이 평행 → 교호작용 없음 / 선이 교차 → 교호작용 존재 → 모형에 포함 필요")

COLOR_SET = [
    "#3B82F6", "#EF4444", "#10B981", "#F59E0B",
    "#8B5CF6", "#EC4899", "#14B8A6",
]

interaction_scores = {}

for pair_idx, (fa, fb) in enumerate(pairs):
    st.markdown(f"#### {fa} × {fb}")

    lvs_a = level_map[fa]
    lvs_b = level_map[fb]

    # 각 (A수준, B수준) 조합별 평균 계산
    means_matrix = {}
    for lv_b in lvs_b:
        means = []
        for lv_a in lvs_a:
            mask = (df_clean[fa] == lv_a) & (df_clean[fb] == lv_b)
            subset = df_clean.loc[mask, response_col]
            means.append(float(subset.mean()) if len(subset) > 0 else np.nan)
        means_matrix[lv_b] = means

    # 교호작용 강도 계산: 선의 기울기 차이 (범위)
    slopes = []
    for lv_b, means in means_matrix.items():
        if len(means) >= 2 and not any(np.isnan(m) for m in means):
            slopes.append(means[-1] - means[0])
    interaction_score = (max(slopes) - min(slopes)) if len(slopes) >= 2 else 0
    interaction_scores[(fa, fb)] = interaction_score

    # 플롯
    fig = go.Figure()
    for b_idx, lv_b in enumerate(lvs_b):
        means = means_matrix[lv_b]
        x_labels = [f"Lv{i+1}\n({v})" for i, v in enumerate(lvs_a)]
        color = COLOR_SET[b_idx % len(COLOR_SET)]
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=means,
            mode="lines+markers",
            name=f"{fb}=Lv{lvs_b.index(lv_b)+1} ({lv_b})",
            line=dict(color=color, width=2.5),
            marker=dict(size=12, color=color),
        ))

    fig.update_layout(
        xaxis_title=fa,
        yaxis_title=f"평균 {response_col}",
        height=320,
        legend=dict(title=fb, orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 판정
    grand_range = df_clean[response_col].max() - df_clean[response_col].min()
    ratio = interaction_score / grand_range * 100 if grand_range > 0 else 0
    if ratio < 10:
        st.success(f"✅ 교호작용 약함 (기울기 차이 {ratio:.1f}%) — 주효과 분석 신뢰")
    elif ratio < 25:
        st.warning(f"⚠️ 교호작용 보통 (기울기 차이 {ratio:.1f}%) — 해석 시 주의")
    else:
        st.error(f"❌ 교호작용 강함 (기울기 차이 {ratio:.1f}%) — 주효과만으로 최적화 불충분")

    st.markdown("---")

# ── 교호작용 강도 순위 요약 ───────────────────────────────
st.markdown("### 교호작용 강도 순위")

grand_range = df_clean[response_col].max() - df_clean[response_col].min()
rows_rank = []
for (fa, fb), score in sorted(interaction_scores.items(), key=lambda x: -x[1]):
    ratio = score / grand_range * 100 if grand_range > 0 else 0
    rows_rank.append({
        "인자 쌍": f"{fa} × {fb}",
        "기울기 차이": f"{score:.4f}",
        "전체 범위 대비 (%)": f"{ratio:.1f}%",
        "판정": "❌ 강한 교호작용" if ratio >= 25 else ("⚠️ 보통" if ratio >= 10 else "✅ 약함"),
    })

rank_df = pd.DataFrame(rows_rank)
st.dataframe(rank_df, use_container_width=True, hide_index=True)

# ── 조치 가이드 ───────────────────────────────────────────
st.markdown("### 교호작용이 강할 때 조치")
st.info("""
**① 현재 OA에 교호작용 열 지정 (L8, L16 적용 가능)**
- 직교배열 선형 그래프(Linear Graph)를 참조하여 교호작용 열 배정
- 해당 열을 인자가 아닌 교호작용 추정에 사용

**② 완전요인배치로 전환**
- 인자 수가 3개 이하면 완전요인배치(Full Factorial)로 교호작용 완전 추정

**③ 2차 실험 추천 페이지 활용**
- 교호작용이 큰 인자 쌍의 범위를 조정하여 재실험 설계
""")
