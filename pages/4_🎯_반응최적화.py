"""
반응최적화 페이지 - Derringer-Suich Desirability 기반 다중 응답 동시 최적화.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf

from core.desirability import (
    ResponseSpec, optimize_desirability,
    d_larger, d_smaller, d_target, overall_desirability,
)

st.set_page_config(page_title="반응최적화", page_icon="🎯", layout="wide")
st.title("🎯 반응최적화 (Desirability Function)")
st.caption("다중 응답을 동시에 만족시키는 최적 인자 조건을 자동 탐색합니다.")

# ============================================================
# Step 1: 데이터 입력
# ============================================================
st.markdown("### Step 1. 데이터 준비")

source = st.radio(
    "데이터 출처",
    ["📁 CSV 파일 업로드", "💾 현재 세션의 설계표 사용", "📚 예제 데이터 사용"],
    horizontal=True,
)

df = None
example_meta = None  # 예제의 응답 사양 정보 (자동 채우기용)

if source == "📁 CSV 파일 업로드":
    file = st.file_uploader("CSV 파일", type=["csv"])
    if file:
        for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(file, encoding=enc)
                break
            except Exception:
                file.seek(0)
elif source == "💾 현재 세션의 설계표 사용":
    if "current_design" in st.session_state:
        df = st.session_state["current_design"].copy()
    else:
        st.warning("Taguchi 설계 페이지에서 먼저 생성하세요.")
elif source == "📚 예제 데이터 사용":
    from core.examples import EXAMPLES, build_example_dataframe
    ex_name = st.selectbox("예제 선택", list(EXAMPLES.keys()))
    if st.button("📚 예제 로드"):
        ex = EXAMPLES[ex_name]
        df = build_example_dataframe(ex)
        st.session_state["opt_loaded_example"] = ex
        st.session_state["opt_loaded_df"] = df
    elif "opt_loaded_df" in st.session_state:
        df = st.session_state["opt_loaded_df"]
        example_meta = st.session_state.get("opt_loaded_example")

if df is not None and len(df) > 0:
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# Step 2: 인자/응답 컬럼 지정
# ============================================================
if df is not None and len(df) > 0:
    st.markdown("---")
    st.markdown("### Step 2. 인자/응답 컬럼 지정")

    exclude = {"Run_Order", "Std_Order"}
    all_cols = [c for c in df.columns if c not in exclude]

    col1, col2 = st.columns(2)
    with col1:
        factor_candidates = [c for c in all_cols if df[c].nunique() <= 5
                             and pd.api.types.is_numeric_dtype(df[c])]
        factor_cols = st.multiselect("인자 컬럼 (숫자형)", options=all_cols,
                                     default=factor_candidates)
    with col2:
        response_candidates = [c for c in all_cols if c not in factor_cols
                               and pd.api.types.is_numeric_dtype(df[c])]
        response_cols = st.multiselect("응답 컬럼 (최소 1개)", options=response_candidates,
                                        default=response_candidates)

    if factor_cols and response_cols:
        df_clean = df.dropna(subset=response_cols).copy()
        st.caption(f"분석 대상: {len(df_clean)} 행 (NaN 행 제외)")

        # 인자 코드값 변환 (코드: -1 ~ +1)
        # 회귀모형 안정성을 위해 코드값으로 회귀, 예측 시 다시 실제값으로 매핑
        factor_ranges = {}
        for fc in factor_cols:
            lo = df_clean[fc].min()
            hi = df_clean[fc].max()
            factor_ranges[fc] = (lo, hi)

        # ============================================================
        # Step 3: 회귀모형 학습
        # ============================================================
        st.markdown("---")
        st.markdown("### Step 3. 회귀모형 학습")

        include_interactions = st.checkbox(
            "2-way 상호작용 포함",
            value=False,
            help="True면 X1:X2 같은 교호작용항 포함. 데이터가 충분히 많을 때만 권장."
        )

        models = {}
        model_summaries = {}
        for rc in response_cols:
            # 안전하게 컬럼명 처리 (한글, 공백 회피)
            df_model = df_clean[factor_cols + [rc]].copy()
            # 컬럼명 정리: 영문 별칭 사용
            col_alias = {c: f"X{i}" for i, c in enumerate(factor_cols)}
            df_model = df_model.rename(columns=col_alias)
            df_model = df_model.rename(columns={rc: "Y"})

            x_terms = [col_alias[c] for c in factor_cols]
            if include_interactions and len(x_terms) >= 2:
                # 2-way 교호작용 추가
                interactions = []
                for i in range(len(x_terms)):
                    for j in range(i+1, len(x_terms)):
                        interactions.append(f"{x_terms[i]}:{x_terms[j]}")
                formula = f"Y ~ " + " + ".join(x_terms + interactions)
            else:
                formula = f"Y ~ " + " + ".join(x_terms)

            try:
                model = smf.ols(formula, data=df_model).fit()
                models[rc] = (model, col_alias)
                model_summaries[rc] = {
                    "R²": model.rsquared,
                    "R²_adj": model.rsquared_adj,
                    "F": model.fvalue,
                    "p": model.f_pvalue,
                }
            except Exception as e:
                st.error(f"응답 '{rc}' 회귀모형 학습 실패: {e}")
                models[rc] = None

        # 회귀 진단표
        if model_summaries:
            st.markdown("**회귀 모형 적합도**")
            summary_df = pd.DataFrame(model_summaries).T
            for c in ["R²", "R²_adj", "p"]:
                if c in summary_df.columns:
                    summary_df[c] = summary_df[c].apply(lambda v: f"{v:.4f}" if pd.notnull(v) else "-")
            if "F" in summary_df.columns:
                summary_df["F"] = summary_df["F"].apply(lambda v: f"{v:.3f}" if pd.notnull(v) else "-")
            st.dataframe(summary_df, use_container_width=True)
            st.caption("💡 R²가 0.7 이상이면 모형이 데이터를 잘 설명. p<0.05면 통계적으로 유의.")

        # ============================================================
        # Step 4: 응답별 사양 설정
        # ============================================================
        st.markdown("---")
        st.markdown("### Step 4. 응답별 사양 (Desirability Spec) 설정")

        specs = []
        for rc in response_cols:
            with st.expander(f"📐 응답 '{rc}' 사양 설정", expanded=True):
                # 자동 추천값
                y_min = float(df_clean[rc].min())
                y_max = float(df_clean[rc].max())
                y_mean = float(df_clean[rc].mean())

                # 예제 메타가 있으면 자동 채우기
                preset_goal = None
                preset_L = None
                preset_T = None
                preset_U = None
                if example_meta and "responses" in example_meta:
                    if rc in example_meta["responses"]:
                        resp_meta = example_meta["responses"][rc]
                        preset_goal = resp_meta.get("goal")
                        preset_L = resp_meta.get("L")
                        preset_T = resp_meta.get("T")
                        preset_U = resp_meta.get("U")
                        st.caption(f"💡 예제 추천값: {resp_meta.get('note', '')}")

                cols_g = st.columns([1, 1, 1, 1, 1])
                with cols_g[0]:
                    goal_options = ["max", "min", "target"]
                    default_idx = goal_options.index(preset_goal) if preset_goal in goal_options else 0
                    goal = st.selectbox(
                        "목표", goal_options,
                        index=default_idx,
                        format_func=lambda x: {"max": "🔼 최대화", "min": "🔽 최소화", "target": "🎯 목표값"}[x],
                        key=f"goal_{rc}",
                    )
                with cols_g[1]:
                    L = st.number_input(
                        "L (하한)",
                        value=preset_L if preset_L is not None else y_min * 0.9,
                        format="%.4f",
                        key=f"L_{rc}",
                    )
                with cols_g[2]:
                    T = st.number_input(
                        "T (목표)",
                        value=preset_T if preset_T is not None else y_mean,
                        format="%.4f",
                        key=f"T_{rc}",
                    )
                with cols_g[3]:
                    U = st.number_input(
                        "U (상한)",
                        value=preset_U if preset_U is not None else y_max * 1.1,
                        format="%.4f",
                        key=f"U_{rc}",
                    )
                with cols_g[4]:
                    weight = st.number_input(
                        "가중치",
                        value=1.0, min_value=0.1, max_value=10.0, step=0.1,
                        key=f"w_{rc}",
                    )

                shape_s = st.slider(
                    "형상 모수 s (1=선형, >1=보수적, <1=관대)",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key=f"s_{rc}",
                )

                try:
                    spec_kwargs = {"name": rc, "goal": goal, "s": shape_s, "weight": weight}
                    if goal == "max":
                        spec_kwargs.update({"L": L, "T": T})
                    elif goal == "min":
                        spec_kwargs.update({"T": T, "U": U})
                    elif goal == "target":
                        spec_kwargs.update({"L": L, "T": T, "U": U})
                    specs.append(ResponseSpec(**spec_kwargs))

                    # Desirability 곡선 미리보기
                    y_range = np.linspace(y_min * 0.8, y_max * 1.2, 200)
                    spec_obj = ResponseSpec(**spec_kwargs)
                    d_vals = spec_obj.desirability(y_range)
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Scatter(
                        x=y_range, y=d_vals, mode="lines",
                        line=dict(color="#3B82F6", width=2.5),
                        name="Desirability",
                    ))
                    # 실제 데이터 위치 표시
                    fig_d.add_trace(go.Scatter(
                        x=df_clean[rc], y=[0.05]*len(df_clean),
                        mode="markers",
                        marker=dict(symbol="line-ns", size=15, color="gray"),
                        name="시험 데이터",
                        showlegend=True,
                    ))
                    fig_d.update_layout(
                        height=250,
                        xaxis_title=rc, yaxis_title="d",
                        yaxis_range=[-0.05, 1.05],
                        margin=dict(l=20, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                except Exception as e:
                    st.error(f"사양 설정 오류: {e}")

        # ============================================================
        # Step 5: 최적화 실행 (이산 수준 격자 탐색)
        # ============================================================
        st.markdown("---")
        st.markdown("### Step 5. 최적화 실행")
        st.caption("정의된 인자 수준 조합 전체를 탐색하여 Desirability가 가장 높은 수준 조합을 찾습니다.")

        if st.button("🚀 최적화 실행", type="primary", use_container_width=True):
            with st.spinner("수준 조합 탐색 중..."):
                import itertools

                # 예측 함수 생성
                predict_funcs = {}
                for rc in response_cols:
                    if models.get(rc) is None:
                        continue
                    model, alias_map = models[rc]
                    def make_pred(m, alias):
                        def f(x):
                            data = {alias[fc]: [x[i]] for i, fc in enumerate(factor_cols)}
                            df_pred = pd.DataFrame(data)
                            return float(m.predict(df_pred).values[0])
                        return f
                    predict_funcs[rc] = make_pred(model, alias_map)

                # 인자별 이산 수준 추출
                factor_levels = {fc: sorted(df_clean[fc].unique()) for fc in factor_cols}
                all_combos = list(itertools.product(*[factor_levels[fc] for fc in factor_cols]))
                weights_list = [s.weight for s in specs]

                best_D = -1.0
                best_combo = None
                best_pred = {}
                best_ind_d = {}

                for combo in all_combos:
                    x = np.array(combo, dtype=float)
                    try:
                        pred_vals = {}
                        d_vals = []
                        for spec in specs:
                            if models.get(spec.name) is None:
                                continue
                            y_hat = predict_funcs[spec.name](x)
                            pred_vals[spec.name] = y_hat
                            d_vals.append(float(spec.desirability(y_hat)))
                        if len(d_vals) < len(specs):
                            continue
                        D = overall_desirability(d_vals, weights_list)
                        if D > best_D:
                            best_D = D
                            best_combo = combo
                            best_pred = pred_vals
                            best_ind_d = {
                                s.name: float(s.desirability(pred_vals[s.name]))
                                for s in specs
                            }
                    except Exception:
                        continue

                if best_combo is not None:
                    optimal_levels = {
                        fc: factor_levels[fc].index(val) + 1
                        for fc, val in zip(factor_cols, best_combo)
                    }
                    st.session_state["opt_result"] = {
                        "optimal_factors": dict(zip(factor_cols, best_combo)),
                        "optimal_levels": optimal_levels,
                        "factor_levels": factor_levels,
                        "predicted_responses": best_pred,
                        "individual_desirabilities": best_ind_d,
                        "overall_desirability": best_D,
                        "n_combos": len(all_combos),
                    }
                else:
                    st.error("최적 조합을 찾지 못했습니다. 사양 범위를 확인하세요.")

        # 결과 표시
        if "opt_result" in st.session_state:
            result = st.session_state["opt_result"]

            st.markdown("#### 최적화 결과")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("전체 Desirability D", f"{result['overall_desirability']:.4f}",
                          help="1에 가까울수록 좋음. 0이면 한 응답 이상 사양 미달.")
                st.caption(f"탐색한 수준 조합 수: {result.get('n_combos', '-')}개")

            if result["optimal_factors"]:
                with col_r2:
                    st.markdown("**최적 인자 조건**")
                    rows = []
                    for fc in factor_cols:
                        lv_num = result["optimal_levels"][fc]
                        lv_val = result["optimal_factors"][fc]
                        all_lvs = result["factor_levels"][fc]
                        rows.append({
                            "인자": fc,
                            "최적 수준": f"수준 {lv_num}",
                            "값": f"{lv_val:.4f}",
                            "전체 수준": " / ".join(
                                [f"Lv{i+1}={v:.4f}" for i, v in enumerate(all_lvs)]
                            ),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("**예측 응답값과 개별 Desirability**")
                resp_rows = []
                for rc in response_cols:
                    if rc in result["predicted_responses"]:
                        ind_d = result["individual_desirabilities"][rc]
                        resp_rows.append({
                            "응답": rc,
                            "예측값": f"{result['predicted_responses'][rc]:.4f}",
                            "개별 d": f"{ind_d:.4f}",
                            "상태": "✅" if ind_d > 0.6 else ("⚠️" if ind_d > 0.3 else "❌"),
                        })
                st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)

                if result["overall_desirability"] > 0.7:
                    st.success(f"✅ D = {result['overall_desirability']:.3f} : 우수한 최적해")
                elif result["overall_desirability"] > 0.4:
                    st.warning(f"⚠️ D = {result['overall_desirability']:.3f} : 일부 응답 트레이드오프 발생. 가중치 조정 검토.")
                else:
                    st.error(f"❌ D = {result['overall_desirability']:.3f} : 사양이 너무 엄격하거나 모형이 부적합.")

                st.info(
                    "💡 **확인 실험 권장**: 위 최적 인자 조건으로 1~3회 실시험을 진행하여 "
                    "예측값과 실측값이 일치하는지 검증하세요."
                )
        else:
            st.info("👆 위 [최적화 실행] 버튼을 눌러주세요.")
