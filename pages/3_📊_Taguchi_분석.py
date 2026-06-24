"""
Taguchi 분석 페이지 - 응답표 / S/N / 주효과도 / ANOVA / 최적 수준.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.taguchi import sn_ratio
from core.analysis import (
    response_table, find_optimal_levels, predict_optimum,
    anova_taguchi, get_level_map,
)
from core.export import to_excel, to_pdf

st.set_page_config(page_title="Taguchi 분석", page_icon="📊", layout="wide")
st.title("📊 Taguchi 분석")
st.caption("시험 결과를 업로드하여 S/N 비, 주효과, ANOVA를 분석합니다.")

# ============================================================
# Step 1: 데이터 입력
# ============================================================
st.markdown("### Step 1. 데이터 준비")

source = st.radio(
    "데이터 출처",
    ["📁 CSV 파일 업로드", "💾 현재 세션의 설계표 사용 (Taguchi 설계 페이지에서 생성)", "📚 예제 데이터 사용"],
    horizontal=True,
)

df = None

if source == "📁 CSV 파일 업로드":
    file = st.file_uploader("CSV 파일 선택", type=["csv"])
    if file:
        # 한글 호환 인코딩 자동 감지
        for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(file, encoding=enc)
                break
            except Exception:
                file.seek(0)
                continue
        if df is not None:
            st.success(f"✅ {len(df)}행 로드 완료")

elif source == "💾 현재 세션의 설계표 사용 (Taguchi 설계 페이지에서 생성)":
    if "current_design" in st.session_state:
        df = st.session_state["current_design"].copy()
        st.success(f"✅ 세션에서 {len(df)}행 로드")
    else:
        st.warning("⚠️ 먼저 [🔬 Taguchi 설계] 페이지에서 설계표를 생성하세요.")

elif source == "📚 예제 데이터 사용":
    from core.examples import EXAMPLES, build_example_dataframe
    ex_name = st.selectbox("예제 선택", list(EXAMPLES.keys()))
    if st.button("📚 예제 로드"):
        ex = EXAMPLES[ex_name]
        df = build_example_dataframe(ex)
        st.session_state["loaded_example"] = ex
        st.session_state["loaded_df"] = df
    elif "loaded_df" in st.session_state:
        df = st.session_state["loaded_df"]

if df is not None and len(df) > 0:
    st.markdown("**데이터 미리보기**")
    st.dataframe(df.head(20), use_container_width=True)

# ============================================================
# Step 2: 컬럼 선택
# ============================================================
if df is not None and len(df) > 0:
    st.markdown("---")
    st.markdown("### Step 2. 인자/응답 컬럼 지정")

    # 인자 후보 (Run_Order, Std_Order 등 메타 컬럼만 제외)
    exclude = {"Run_Order", "Std_Order"}
    all_cols = [c for c in df.columns if c not in exclude]

    col1, col2 = st.columns(2)
    with col1:
        # 숫자가 아니거나 수준 수가 적은 컬럼 = 인자 후보
        factor_candidates = [
            c for c in all_cols if df[c].nunique() <= 5
        ]
        factor_cols = st.multiselect(
            "인자 컬럼 선택",
            options=all_cols,
            default=factor_candidates,
        )

    with col2:
        # 응답 후보 (인자 아닌 숫자형)
        response_candidates = [
            c for c in all_cols if c not in factor_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        response_col = st.selectbox("응답 컬럼 선택", options=response_candidates)

    if factor_cols and response_col:
        # 데이터 행 필터링
        df_clean = df.dropna(subset=[response_col]).copy()
        st.caption(f"분석 대상: {len(df_clean)} 행 (NaN 행 제외 후)")

        # ============================================================
        # Step 3: S/N 비 옵션
        # ============================================================
        st.markdown("---")
        st.markdown("### Step 3. 분석 옵션")

        # 적용된 옵션 초기화 (처음 방문 시)
        if "applied_opts" not in st.session_state:
            st.session_state["applied_opts"] = {
                "sn_type": "smaller",
                "direction": "min",
                "pool_threshold": 0.0,
            }
        # 위젯 키 초기화 (처음 방문 시)
        if "sn_type_widget" not in st.session_state:
            st.session_state["sn_type_widget"] = st.session_state["applied_opts"]["sn_type"]
        if "direction_widget" not in st.session_state:
            st.session_state["direction_widget"] = st.session_state["applied_opts"]["direction"]
        if "pool_threshold_widget" not in st.session_state:
            st.session_state["pool_threshold_widget"] = st.session_state["applied_opts"]["pool_threshold"]

        col_sn1, col_sn2 = st.columns(2)
        with col_sn1:
            st.selectbox(
                "S/N 비 유형",
                options=["larger", "smaller", "nominal"],
                format_func=lambda x: {
                    "larger": "🔼 Larger-the-better (망대: 클수록 좋음)",
                    "smaller": "🔽 Smaller-the-better (망소: 작을수록 좋음)",
                    "nominal": "🎯 Nominal-the-best (망목: 목표치에 근접)",
                }[x],
                key="sn_type_widget",
            )
        with col_sn2:
            st.selectbox(
                "최적 수준 선택 방향 (평균 기준)",
                options=["max", "min"],
                format_func=lambda x: {
                    "max": "🔼 응답값이 큰 수준 (강도, 효율 등)",
                    "min": "🔽 응답값이 작은 수준 (소음, 결함 등)",
                }[x],
                key="direction_widget",
            )

        st.slider(
            "ANOVA 풀링 임계값 (기여율 % 미만 인자를 오차에 합산)",
            min_value=0.0, max_value=20.0, step=0.5,
            key="pool_threshold_widget",
            help="0이면 풀링 안 함. 5~10이 일반적."
        )

        # 적용 버튼 및 현재 적용 상태 표시
        col_btn, col_status = st.columns([2, 5])
        with col_btn:
            apply_clicked = st.button("✅ 적용", type="primary", use_container_width=True)
        with col_status:
            _opts = st.session_state["applied_opts"]
            _sn_label = {"larger": "망대(Larger)", "smaller": "망소(Smaller)", "nominal": "망목(Nominal)"}[_opts["sn_type"]]
            _dir_label = {"max": "최대(응답값 큰 수준)", "min": "최소(응답값 작은 수준)"}[_opts["direction"]]
            st.caption(
                f"현재 적용된 옵션 — S/N: **{_sn_label}** | 방향: **{_dir_label}** | 풀링 임계: **{_opts['pool_threshold']}%**"
            )

        if apply_clicked:
            st.session_state["applied_opts"]["sn_type"] = st.session_state["sn_type_widget"]
            st.session_state["applied_opts"]["direction"] = st.session_state["direction_widget"]
            st.session_state["applied_opts"]["pool_threshold"] = st.session_state["pool_threshold_widget"]
            st.success("✅ 분석 옵션이 적용되었습니다. 아래 결과를 확인하세요.")

        # 분석에는 적용된 옵션만 사용
        sn_type = st.session_state["applied_opts"]["sn_type"]
        direction = st.session_state["applied_opts"]["direction"]
        pool_threshold = st.session_state["applied_opts"]["pool_threshold"]

        # ============================================================
        # Step 4: 분석 실행
        # ============================================================
        st.markdown("---")
        st.markdown("### Step 4. 분석 결과")

        # ----- 4-1. 응답표 (평균) -----
        st.markdown("#### 4-1. 평균 응답표")
        mean_tbl = response_table(df_clean, factor_cols, response_col, statistic="mean")
        st.dataframe(mean_tbl.round(4), use_container_width=True)

        with st.expander("💡 응답표 해석"):
            st.markdown(
                f"""
                - **Delta**: 수준별 평균의 최대-최소 차이
                - **Rank**: Delta가 큰 인자가 1순위 (응답에 더 큰 영향)
                - 가장 중요한 인자: **{mean_tbl.loc['Rank'].astype(int).idxmin()}**
                """
            )

        # ----- 4-2. S/N 응답표 -----
        st.markdown(f"#### 4-2. S/N 응답표 ({sn_type})")
        sn_tbl = None  # 내보내기를 위해 스코프 밖에서 초기화

        # 반복이 없으면 nominal 불가
        n_replicates = df_clean.groupby(factor_cols).size().max()
        if sn_type == "nominal" and n_replicates < 2:
            st.warning("⚠️ Nominal-the-best S/N은 반복 측정이 필요합니다. 평균만 표시.")
        else:
            try:
                sn_tbl = response_table(
                    df_clean, factor_cols, response_col,
                    statistic="sn", sn_type=sn_type,
                )
                st.dataframe(sn_tbl.round(4), use_container_width=True)
            except Exception as e:
                st.error(f"S/N 응답표 계산 오류: {e}")

        # ----- 4-3. 주효과도 -----
        st.markdown("#### 4-3. 주효과도 (Main Effect Plot)")

        n_factors = len(factor_cols)
        cols_per_row = min(3, n_factors)
        rows = (n_factors + cols_per_row - 1) // cols_per_row

        fig = make_subplots(rows=rows, cols=cols_per_row,
                            subplot_titles=factor_cols)

        grand_mean = df_clean[response_col].mean()
        level_map = get_level_map(df_clean, factor_cols)

        for idx, fcol in enumerate(factor_cols):
            row = idx // cols_per_row + 1
            col = idx % cols_per_row + 1

            levels = level_map[fcol]
            means = [df_clean[df_clean[fcol] == lv][response_col].mean() for lv in levels]

            fig.add_trace(
                go.Scatter(
                    x=[str(lv) for lv in levels],
                    y=means,
                    mode="lines+markers",
                    marker=dict(size=12, color="#3B82F6"),
                    line=dict(width=2.5),
                    showlegend=False,
                ),
                row=row, col=col,
            )
            # 전체 평균선
            fig.add_hline(
                y=grand_mean, line_dash="dash", line_color="gray",
                row=row, col=col,
                annotation_text=f"평균={grand_mean:.2f}",
                annotation_position="top right",
            )

        fig.update_layout(height=300*rows, showlegend=False,
                          title_text=f"주효과도 - 응답: {response_col}")
        st.plotly_chart(fig, use_container_width=True)

        # ----- 4-4. ANOVA -----
        st.markdown("#### 4-4. ANOVA (분산분석)")
        anova_df = anova_taguchi(
            df_clean, factor_cols, response_col,
            pool_threshold=pool_threshold if pool_threshold > 0 else None,
        )
        # 보기 좋게 포매팅
        anova_display = anova_df.copy()
        for c in ["SS", "MS"]:
            if c in anova_display.columns:
                anova_display[c] = anova_display[c].apply(
                    lambda v: f"{v:.4f}" if pd.notnull(v) else "-"
                )
        if "F" in anova_display.columns:
            anova_display["F"] = anova_display["F"].apply(
                lambda v: f"{v:.3f}" if pd.notnull(v) else "-"
            )
        if "P" in anova_display.columns:
            anova_display["P"] = anova_display["P"].apply(
                lambda v: f"{v:.4f}" if pd.notnull(v) else "-"
            )
        if "Contribution(%)" in anova_display.columns:
            anova_display["Contribution(%)"] = anova_display["Contribution(%)"].apply(
                lambda v: f"{v:.2f}" if pd.notnull(v) else "-"
            )

        st.dataframe(anova_display, use_container_width=True, hide_index=True)

        # 유의 인자 강조
        sig_factors = []
        for _, row in anova_df.iterrows():
            if row["Source"] in factor_cols and pd.notnull(row["P"]) and row["P"] < 0.05:
                sig_factors.append(row["Source"])

        if sig_factors:
            st.success(f"✅ 통계적으로 유의한 인자 (p<0.05): **{', '.join(sig_factors)}**")
        else:
            st.warning("⚠️ p<0.05를 만족하는 유의 인자 없음. 풀링을 시도하거나 시험 횟수를 늘려보세요.")

        # ----- 4-5. 최적 수준 -----
        st.markdown("#### 4-5. 최적 수준 추천")

        opt = find_optimal_levels(mean_tbl, factor_cols, direction=direction)
        y_pred = predict_optimum(df_clean, factor_cols, response_col, opt)

        # 표시
        rows_opt = []
        for fcol in factor_cols:
            levels = level_map[fcol]
            opt_idx = opt[fcol] - 1
            if 0 <= opt_idx < len(levels):
                actual_val = levels[opt_idx]
            else:
                actual_val = "?"
            rows_opt.append({
                "인자": fcol,
                "최적 수준 인덱스": opt[fcol],
                "최적 수준 값": actual_val,
                "유의 여부": "✅" if fcol in sig_factors else "-",
            })
        opt_df = pd.DataFrame(rows_opt)
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        st.metric(
            label=f"🎯 최적 조건에서 예측 응답값 ({direction})",
            value=f"{y_pred:.4f}",
            help="가법성 가정 기반: T_bar + Σ(M_optimum - T_bar)"
        )

        st.info(
            "💡 **확인 실험 권장**: 위 최적 조건으로 1~3회 추가 시험하여 "
            "예측치와 실측치가 일치하는지 검증하세요."
        )

        # 후속 분석 페이지를 위해 세션에 저장
        st.session_state["analysis_result"] = {
            "df_clean": df_clean,
            "factor_cols": factor_cols,
            "response_col": response_col,
            "mean_tbl": mean_tbl,
            "sn_tbl": sn_tbl,
            "anova_df": anova_df,
            "opt_df": opt_df,
            "y_pred": y_pred,
            "opt": opt,
            "level_map": level_map,
            "sn_type": sn_type,
            "direction": direction,
            "sig_factors": sig_factors,
        }

        # ============================================================
        # Step 5: 결과 내보내기
        # ============================================================
        st.markdown("---")
        st.markdown("### 📤 결과 내보내기")

        fname_base = f"DOE_분석_{response_col}"
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            try:
                excel_bytes = to_excel(
                    df_data=df_clean,
                    mean_tbl=mean_tbl,
                    sn_tbl=sn_tbl,
                    anova_df=anova_df,
                    opt_df=opt_df,
                    y_pred=y_pred,
                    response_col=response_col,
                    sn_type=sn_type,
                    factor_cols=factor_cols,
                )
                st.download_button(
                    label="📥 Excel 다운로드 (.xlsx)",
                    data=excel_bytes,
                    file_name=f"{fname_base}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Excel 생성 오류: {e}")

        with col_dl2:
            try:
                pdf_bytes = to_pdf(
                    df_data=df_clean,
                    mean_tbl=mean_tbl,
                    sn_tbl=sn_tbl,
                    anova_df=anova_df,
                    opt_df=opt_df,
                    y_pred=y_pred,
                    response_col=response_col,
                    sn_type=sn_type,
                    factor_cols=factor_cols,
                )
                st.download_button(
                    label="📄 PDF 다운로드 (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{fname_base}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF 생성 오류: {e}")

else:
    if df is not None:
        st.warning("데이터가 비어있습니다.")
    else:
        st.info("👆 위에서 데이터를 먼저 준비하세요.")
