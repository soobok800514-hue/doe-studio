"""
Taguchi 설계 페이지 - 인자 입력 → OA 자동 추천 → 설계표 생성 → CSV 다운로드.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from core.taguchi import OA_CATALOG, recommend_oa, build_design

st.set_page_config(page_title="Taguchi 설계", page_icon="🔬", layout="wide")
st.title("🔬 Taguchi 설계표 생성")
st.caption("인자와 수준을 입력하면 직교배열표를 자동으로 추천하고 시험계획표를 생성합니다.")

# ============================================================
# Step 1: 인자 정보 입력
# ============================================================
st.markdown("### Step 1. 인자 정보 입력")

col_a, col_b = st.columns([1, 3])
with col_a:
    n_factors = st.number_input(
        "인자 수",
        min_value=2, max_value=13, value=3, step=1,
        help="2 ~ 13개 사이로 입력. 일반적으로 3~5개 권장."
    )

st.markdown("**인자별 상세 정보**")

# 동적 입력 폼
factors = {}
factor_levels_count = []

for i in range(int(n_factors)):
    cols = st.columns([2, 1, 1.2, 1.2, 1.2, 1])
    with cols[0]:
        fname = st.text_input(
            f"인자 {i+1} 이름",
            value=f"X{i+1}",
            key=f"fname_{i}",
        )
    with cols[1]:
        n_levels = st.selectbox(
            "수준 수",
            options=[2, 3],
            index=1,  # 기본 3수준
            key=f"nlv_{i}",
        )
    factor_levels_count.append(n_levels)

    # 수준값 입력
    levels = []
    with cols[2]:
        lv1 = st.number_input(f"Lv1", value=0.0 if i == 0 else float(i*10),
                               key=f"lv1_{i}", format="%.4f")
        levels.append(lv1)
    with cols[3]:
        lv2 = st.number_input(f"Lv2", value=1.0 if i == 0 else float(i*10 + 5),
                               key=f"lv2_{i}", format="%.4f")
        levels.append(lv2)
    if n_levels == 3:
        with cols[4]:
            lv3 = st.number_input(f"Lv3", value=2.0 if i == 0 else float(i*10 + 10),
                                   key=f"lv3_{i}", format="%.4f")
            levels.append(lv3)
    else:
        with cols[4]:
            st.write("")
    with cols[5]:
        unit = st.text_input("단위", value="-", key=f"unit_{i}")

    factors[fname] = {"levels": levels, "unit": unit}

# ============================================================
# Step 2: OA 자동 추천
# ============================================================
st.markdown("---")
st.markdown("### Step 2. 직교배열표 선택")

try:
    recommended = recommend_oa(int(n_factors), factor_levels_count)
    st.success(f"💡 **추천 OA**: `{recommended}` "
               f"({OA_CATALOG[recommended]['runs']}회 시험, "
               f"최대 {OA_CATALOG[recommended]['max_factors']}인자 수용)")
except ValueError as e:
    st.error(f"❌ 추천 실패: {e}")
    recommended = "L9"

# 사용자가 다른 OA 선택 가능
available_oas = list(OA_CATALOG.keys())
oa_choice = st.selectbox(
    "사용할 OA 선택",
    options=available_oas,
    index=available_oas.index(recommended),
    format_func=lambda x: f"{x} ({OA_CATALOG[x]['runs']} runs, {OA_CATALOG[x]['type']})",
)

# 호환성 검사
oa_info = OA_CATALOG[oa_choice]
warning = None
dummy_factors = []

if n_factors > oa_info["max_factors"]:
    warning = f"⚠️ {oa_choice}는 최대 {oa_info['max_factors']}인자만 지원하지만 {n_factors}개 입력됨."
else:
    # auto_reorder + 더미 수준 고려한 배치 가능 여부 검사
    from collections import defaultdict, deque as _dq
    _avail = defaultdict(_dq)
    for col_lv in oa_info["levels"]:
        _avail[col_lv].append(col_lv)

    for fname in list(factors.keys()):
        n_lv = len(factors[fname]["levels"])
        if _avail[n_lv]:
            _avail[n_lv].popleft()
        elif n_lv == 2 and _avail[3]:
            _avail[3].popleft()
            dummy_factors.append(fname)
        else:
            warning = (f"⚠️ 인자 '{fname}'({n_lv}수준)을 {oa_choice}에 배치할 수 없습니다. "
                       f"다른 OA를 선택하거나 인자 수준을 조정하세요.")
            break

if warning:
    st.warning(warning)
elif dummy_factors:
    st.info(
        f"ℹ️ **더미 수준(Dummy Level) 적용**: "
        f"**{', '.join(dummy_factors)}** 인자가 3수준 컬럼에 배치됩니다. "
        f"(OA 수준 3 → 실제 수준 1 반복 — Phadke 1989 표준 기법)"
    )

# OA 메타 표시
with st.expander("📋 선택한 OA 상세 정보"):
    st.markdown(f"""
    - **이름**: {oa_choice}
    - **시험 횟수 (Runs)**: {oa_info['runs']}
    - **최대 인자 수**: {oa_info['max_factors']}
    - **수준 구조**: {oa_info['levels']}
    - **유형**: {oa_info['type']}
    """)
    st.markdown("**원형 OA 매트릭스 (수준 인덱스 1, 2, 3):**")
    oa_df = pd.DataFrame(
        oa_info["array"],
        columns=[f"Col{i+1}" for i in range(oa_info["array"].shape[1])],
        index=[f"Run {i+1}" for i in range(oa_info["array"].shape[0])],
    )
    st.dataframe(oa_df, use_container_width=True)

# ============================================================
# Step 3: 옵션
# ============================================================
st.markdown("---")
st.markdown("### Step 3. 설계 옵션")

col1, col2 = st.columns(2)
with col1:
    randomize = st.checkbox(
        "시험 순서 무작위화 (권장)",
        value=True,
        help="잡음 편향을 방지하기 위해 일반적으로 ON. 단, 변경 비용이 큰 인자가 있다면 OFF 고려."
    )
with col2:
    seed = st.number_input("무작위 시드 (재현용)", value=42, step=1)

# ============================================================
# Step 4: 설계표 생성
# ============================================================
st.markdown("---")
st.markdown("### Step 4. 설계표 생성")

if st.button("🎯 설계표 생성", type="primary", use_container_width=True):
    try:
        design_df = build_design(
            oa_choice, factors,
            randomize=randomize, seed=int(seed),
        )

        st.session_state["current_design"] = design_df
        st.session_state["current_factors"] = factors
        st.session_state["current_oa"] = oa_choice

        st.success(f"✅ {oa_choice} 설계표 ({len(design_df)}회 시험) 생성 완료")

    except Exception as e:
        st.error(f"❌ 생성 실패: {e}")

# 결과 표시
if "current_design" in st.session_state:
    design_df = st.session_state["current_design"]

    st.markdown("#### 생성된 설계표")
    st.dataframe(design_df, use_container_width=True, height=400)

    # CSV 다운로드
    csv = design_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 CSV 다운로드 (Excel에서 한글 깨짐 없음)",
        data=csv,
        file_name=f"DOE_design_{st.session_state['current_oa']}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 직접 입력 옵션
    with st.expander("📝 웹에서 직접 결과 입력 (선택)"):
        st.info("아래 표에서 Response_Y 컬럼을 직접 클릭하여 결과를 입력할 수 있습니다.")
        edited_df = st.data_editor(
            design_df,
            num_rows="dynamic",
            use_container_width=True,
            key="design_editor",
        )
        if st.button("💾 결과 저장", key="save_results"):
            st.session_state["current_design"] = edited_df
            st.success("✅ 저장됨. 이제 '📊 Taguchi 분석' 페이지로 이동하세요.")

    st.markdown("---")
    st.markdown("### 다음 단계")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.info(
            """
            **시험 진행 후:**
            1. CSV에 결과 입력
            2. 📊 Taguchi 분석 페이지로 이동
            3. 데이터 업로드 후 분석
            """
        )
    with col_n2:
        st.info(
            """
            **다중 응답 최적화가 필요하면:**
            1. 분석 페이지에서 회귀모형 학습
            2. 🎯 반응최적화 페이지로 이동
            3. 응답별 사양 설정 후 최적 조건 탐색
            """
        )
else:
    st.info("👆 위 정보를 입력한 후 [설계표 생성] 버튼을 눌러주세요.")
