"""
DOE Studio - 사내 시험계획법 자동화 도구
=====================================
실행: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="DOE Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바 상단 안내
with st.sidebar:
    st.markdown("### 🔬 DOE Studio")
    st.caption("v0.1.3 · 사내 시험계획법 도구")
    st.markdown("---")
    st.info(
        "**신입사원이라면?**\n\n"
        "👈 왼쪽 메뉴에서 **📘 사용법** 페이지를 먼저 읽어주세요."
    )

# 메인 페이지
st.title("🔬 DOE Studio")
st.subheader("Taguchi 직교배열표 · 반응최적화 사내 자동화 도구")

st.markdown(
    """
    이 도구는 기존에 Minitab으로 진행하던 **시험계획법(DOE)** 작업을
    웹 UI에서 동일하게 수행할 수 있도록 만들어졌습니다.
    신입사원도 페이지를 따라가며 익힐 수 있도록 설계되어 있습니다.
    """
)

st.markdown("### 📚 주요 기능")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        #### 1️⃣ 시험계획표 자동 생성
        - **Taguchi 직교배열표** : L4, L8, L9, L12, L16, L18, L27
        - 인자/수준 입력 → OA 자동 추천
        - 무작위 실행 순서 자동 생성
        - CSV 다운로드 → 시험 진행 후 결과 입력
        """
    )
    st.markdown(
        """
        #### 2️⃣ 시험결과 분석
        - **S/N 비** : Larger / Smaller / Nominal-the-best
        - **수준별 평균/S/N 응답표**
        - **주효과도** (Main Effect Plot)
        - **ANOVA** : 기여율, F-값, p-value, 풀링
        - **최적 수준 자동 식별**
        """
    )

with col2:
    st.markdown(
        """
        #### 3️⃣ 반응최적화
        - **Derringer-Suich Desirability Function**
        - 다중 응답 동시 최적화 (목표값/최대/최소)
        - 응답별 가중치 조정 가능
        - 전체 D 최대화 → 최적 인자 조건 자동 탐색
        - What-if 분석 (인자 변경 시 응답 실시간 예측)
        """
    )
    st.markdown(
        """
        #### 4️⃣ 학습용 예제
        - **시트벨트 PT/LL** (L9, 자동차 안전)
        - **도장 두께** (L8, 표면처리)
        - **사출 성형** (L18, 혼합 수준)
        - 예제 데이터로 전체 워크플로우를 한 번에 체험
        """
    )

st.markdown("---")
st.markdown("### 🚀 시작하기")
st.markdown(
    """
    1. 처음이라면 → **📘 사용법** (왼쪽 메뉴)
    2. 바로 만들기 → **🔬 Taguchi 설계**
    3. 시험 결과 있음 → **📊 Taguchi 분석**
    4. 다중 응답 최적화 → **🎯 반응최적화**
    5. 예제로 체험 → **📁 예제 데이터**
    6. 결과 신뢰성 확인 → **🛡️ 검증**
    """
)

st.markdown("---")
with st.expander("ℹ️ 이 도구는 어떻게 만들어졌나요?"):
    st.markdown(
        """
        - **Frontend**: Streamlit (Python)
        - **DOE**: 자체 구현 Taguchi OA + pyDOE3
        - **통계**: statsmodels, scipy
        - **시각화**: Plotly
        - **참고문헌**:
            - Phadke, M.S. (1989). *Quality Engineering Using Robust Design.*
            - Derringer, G. and Suich, R. (1980).
              *Simultaneous Optimization of Several Response Variables.*
              Journal of Quality Technology, 12(4), 214-219.
            - NIST/SEMATECH e-Handbook of Statistical Methods.

        문의: 사내 품질혁신팀
        """
    )
