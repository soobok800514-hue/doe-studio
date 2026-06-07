"""
2차 실험 추천 페이지 - 1차 결과 기반 인자 범위 자동 조정 및 새 설계표 생성.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from core.taguchi import recommend_oa, build_design, OA_CATALOG

st.set_page_config(page_title="2차 실험 추천", page_icon="🔄", layout="wide")
st.title("🔄 2차 실험 범위 추천")
st.caption("1차 Taguchi 결과를 기반으로 인자 범위를 자동으로 조정하고 새 실험 설계표를 생성합니다.")

# ── 세션 로드 ──────────────────────────────────────────────
ar = st.session_state.get("analysis_result", {})
if not ar:
    st.warning("⚠️ Taguchi 분석 페이지에서 먼저 분석을 실행하세요.")
    st.stop()

df_clean     = ar["df_clean"]
factor_cols  = ar["factor_cols"]
response_col = ar["response_col"]
level_map    = ar["level_map"]
opt          = ar["opt"]          # {factor: optimal_level_index (1-based)}
direction    = ar["direction"]
mean_tbl     = ar["mean_tbl"]

st.info(f"**1차 분석** — 응답: `{response_col}` | 인자: {', '.join(f'`{f}`' for f in factor_cols)} | 최적 방향: `{direction}`")

# ── 1차 최적 수준 표시 ────────────────────────────────────
st.markdown("### Step 1. 1차 최적 수준 확인")

rows_1st = []
for fc in factor_cols:
    lvs    = level_map[fc]
    n_lvs  = len(lvs)
    opt_lv = opt[fc]          # 1-based
    opt_val = lvs[opt_lv - 1]
    # 경계 여부
    if opt_lv == 1:
        boundary = "하한 경계 ↓"
        action   = "범위 하향 확장"
    elif opt_lv == n_lvs:
        boundary = "상한 경계 ↑"
        action   = "범위 상향 확장"
    else:
        boundary = "중간 수준"
        action   = "범위 좁히기 (정밀화)"
    rows_1st.append({
        "인자":         fc,
        "수준 수":      n_lvs,
        "최적 수준":    f"Lv{opt_lv} ({opt_val})",
        "경계 여부":    boundary,
        "권장 조치":    action,
    })

df_1st = pd.DataFrame(rows_1st)
st.dataframe(df_1st, use_container_width=True, hide_index=True)

# ── 2차 실험 범위 자동 추천 ──────────────────────────────
st.markdown("---")
st.markdown("### Step 2. 2차 실험 범위 자동 추천")
st.caption("경계 수준은 범위를 확장, 중간 수준은 ±50% 범위로 좁혀 정밀 탐색.")

new_factors = {}
factor_configs = []

for fc in factor_cols:
    lvs    = level_map[fc]
    n_lvs  = len(lvs)
    opt_lv = opt[fc]
    opt_val = float(lvs[opt_lv - 1])

    if n_lvs >= 2:
        step = (float(lvs[-1]) - float(lvs[0])) / (n_lvs - 1)
    else:
        step = abs(float(lvs[0])) * 0.1 or 1.0

    if opt_lv == 1:
        # 하한 경계 → 아래로 확장
        new_lo = opt_val - step * 1.5
        new_hi = opt_val + step * 0.5
    elif opt_lv == n_lvs:
        # 상한 경계 → 위로 확장
        new_lo = opt_val - step * 0.5
        new_hi = opt_val + step * 1.5
    else:
        # 중간 → 좁히기
        new_lo = opt_val - step * 0.5
        new_hi = opt_val + step * 0.5

    factor_configs.append({
        "fc": fc, "n_lvs": n_lvs,
        "opt_val": opt_val, "new_lo": new_lo, "new_hi": new_hi, "step": step,
    })

# 사용자가 범위 조정 가능
st.markdown("**추천 범위 (조정 가능)**")

for cfg in factor_configs:
    fc     = cfg["fc"]
    n_lvs  = cfg["n_lvs"]
    with st.expander(f"⚙️ {fc} — 1차 최적값: {cfg['opt_val']}", expanded=True):
        col_lo, col_hi, col_nv = st.columns(3)
        with col_lo:
            lo = st.number_input(f"하한 (Lo)", value=float(cfg["new_lo"]),
                                  format="%.4f", key=f"lo_{fc}")
        with col_hi:
            hi = st.number_input(f"상한 (Hi)", value=float(cfg["new_hi"]),
                                  format="%.4f", key=f"hi_{fc}")
        with col_nv:
            nv = st.selectbox(f"수준 수", options=[2, 3],
                               index=0 if n_lvs == 2 else 1, key=f"nv_{fc}")

        # 수준값 자동 생성
        if nv == 2:
            levels = [lo, hi]
        else:
            mid = (lo + hi) / 2
            levels = [lo, mid, hi]

        unit = st.text_input("단위", value="-", key=f"unit_{fc}")

        st.markdown(f"→ 생성 수준: **{[round(v,4) for v in levels]}**")
        new_factors[fc] = {"levels": [round(v, 4) for v in levels], "unit": unit}

# ── 2차 설계표 생성 ──────────────────────────────────────
st.markdown("---")
st.markdown("### Step 3. 2차 실험 설계표 생성")

level_counts = [len(new_factors[fc]["levels"]) for fc in factor_cols]

try:
    recommended_oa = recommend_oa(len(factor_cols), level_counts)
    n_runs_rec = (
        int(np.prod(level_counts))
        if recommended_oa == "FULL"
        else OA_CATALOG[recommended_oa]["runs"]
    )
    level_str = " × ".join(str(l) for l in level_counts)
    if recommended_oa == "FULL":
        st.success(f"💡 추천: **완전요인배치** ({level_str} = {n_runs_rec}회)")
    else:
        st.success(f"💡 추천 OA: **{recommended_oa}** ({n_runs_rec}회 시험)")
except ValueError as e:
    st.error(f"OA 추천 실패: {e}")
    recommended_oa = None

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    randomize = st.checkbox("시험 순서 무작위화", value=True)
with col_opt2:
    seed = st.number_input("무작위 시드", value=42, step=1)

if recommended_oa and st.button("🎯 2차 설계표 생성", type="primary", use_container_width=True):
    try:
        design_df = build_design(
            recommended_oa, new_factors,
            randomize=randomize, seed=int(seed),
        )
        st.session_state["current_design"]  = design_df
        st.session_state["current_factors"] = new_factors
        st.session_state["current_oa"]      = recommended_oa

        st.success(f"✅ 2차 설계표 ({len(design_df)}회) 생성 완료 — Taguchi 설계 페이지에도 자동 저장됨")
        st.dataframe(design_df, use_container_width=True, height=400)

        # CSV 다운로드
        csv = design_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📥 2차 설계표 CSV 다운로드",
            data=csv,
            file_name=f"DOE_2차실험_{recommended_oa}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"설계표 생성 오류: {e}")

# ── 진행 흐름 요약 ────────────────────────────────────────
st.markdown("---")
st.markdown("### 2차 실험 이후 흐름")
st.info("""
1. 위 설계표를 CSV로 다운로드 → 실험 진행 → 결과 입력
2. **Taguchi 분석** 페이지 → CSV 업로드 → 2차 분석 실행
3. **확인 실험 검증** 페이지 → 2차 예측값 vs 실측값 비교
4. 오차율 ≤15% → 최적 조건 확정 / 오차율 >15% → 3차 반복 또는 RSM 전환
""")
