"""
예제 데이터 페이지 - 신입사원이 익숙한 도메인의 가상 데이터로 전체 워크플로우 체험.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from core.examples import EXAMPLES, build_example_dataframe

st.set_page_config(page_title="예제 데이터", page_icon="📁", layout="wide")
st.title("📁 예제 데이터셋")
st.caption("학습용 가상 예제로 분석 워크플로우를 한 번에 체험하세요.")

st.markdown(
    """
    아래 예제들은 자동차 부품 시험에서 자주 만나는 시나리오를 가상의 데이터로 구성한 것입니다.
    예제를 선택하면 설계표 + 응답값이 함께 로드되어,
    바로 **📊 Taguchi 분석** 또는 **🎯 반응최적화** 페이지에서 사용할 수 있습니다.
    """
)

# 예제 선택
ex_name = st.selectbox("예제 선택", list(EXAMPLES.keys()))
example = EXAMPLES[ex_name]

st.markdown(f"### {example['name']}")
st.markdown(f"**설명**: {example['description']}")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(f"**직교배열표**: `{example['oa']}`")
    st.markdown(f"**인자 수**: {len(example['factors'])}")
    st.markdown(f"**응답 수**: {len(example['responses'])}")

with col2:
    st.markdown("**인자 정보:**")
    for fname, finfo in example["factors"].items():
        st.markdown(f"- `{fname}`: {finfo['levels']} {finfo['unit']}")

st.markdown("**응답 사양:**")
for rname, rinfo in example["responses"].items():
    goal_label = {"max": "🔼 최대화", "min": "🔽 최소화", "target": "🎯 목표값"}[rinfo["goal"]]
    spec_str = ""
    if "L" in rinfo: spec_str += f" L={rinfo['L']}"
    if "T" in rinfo: spec_str += f" T={rinfo['T']}"
    if "U" in rinfo: spec_str += f" U={rinfo['U']}"
    st.markdown(f"- `{rname}` ({rinfo['unit']}): {goal_label}{spec_str}")
    if "note" in rinfo:
        st.caption(f"  💡 {rinfo['note']}")

# 데이터 로드
df = build_example_dataframe(example)

st.markdown("### 데이터 미리보기")
st.dataframe(df, use_container_width=True, height=400)

# 다운로드 & 세션 저장
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "📥 CSV 다운로드",
        data=csv,
        file_name=f"example_{example['oa']}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_btn2:
    if st.button("📊 이 예제로 분석 시작", type="primary", use_container_width=True):
        st.session_state["loaded_df"] = df
        st.session_state["loaded_example"] = example
        st.session_state["opt_loaded_df"] = df
        st.session_state["opt_loaded_example"] = example
        st.success(
            "✅ 예제가 로드되었습니다. 왼쪽 메뉴에서 "
            "**📊 Taguchi 분석** 또는 **🎯 반응최적화** 페이지로 이동하세요."
        )

st.markdown("---")
st.markdown("### 🎓 학습 미션 (신입사원용)")
st.markdown(
    f"""
    이 **{ex_name}** 예제로 다음을 단계적으로 수행해보세요:

    1. **📊 Taguchi 분석** 페이지에서:
        - 인자/응답 컬럼 선택
        - 평균 응답표에서 **가장 중요한 인자(Rank 1)** 가 무엇인지 확인
        - ANOVA에서 **p<0.05** 인 유의 인자 확인
        - 주효과도에서 기울기 가장 급한 인자 확인

    2. **🎯 반응최적화** 페이지에서 (응답이 2개 이상일 때):
        - 응답 사양 자동 채우기 활용
        - 회귀모형 R² 확인 (0.7 이상이면 OK)
        - 최적화 실행
        - 결과의 D 값과 인자 조건 확인

    3. **검증 미션**:
        - 같은 데이터를 Minitab에 입력해 결과 일치 여부 확인
        - 차이가 있다면 슬랙 `#doe-studio` 채널에 공유
    """
)
