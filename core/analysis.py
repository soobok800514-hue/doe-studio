"""
Taguchi 분석 모듈
================
- 수준별 평균 응답표 (Mean Response Table)
- 수준별 S/N 응답표 (S/N Response Table)
- 주효과 (Main Effect) 분석
- ANOVA (분산분석)
- 최적 수준 예측 및 예측치 (Predicted value at optimum)

핵심 공식
---------
주효과 (level i, factor X):
    M_i(X) = mean( y for runs where factor X is at level i )

랭킹 (Delta) = max(M_i) - min(M_i)
  -> Delta가 큰 인자가 중요한 인자

ANOVA SS for factor X:
    SS_X = n_per_level * sum( (M_i(X) - grand_mean)^2 for all levels )

예측치 (예: 인자 A,B,C가 모두 의미있고 최적 수준이 A2, B1, C3일 때):
    y_pred = T_bar + (M_A2 - T_bar) + (M_B1 - T_bar) + (M_C3 - T_bar)
    여기서 T_bar = 전체 평균
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

try:
    from .taguchi import sn_ratio
except ImportError:
    from taguchi import sn_ratio


# ============================================================
# 1) 수준별 응답표
# ============================================================

def response_table(
    df: pd.DataFrame,
    factor_cols: List[str],
    response_col: str,
    statistic: str = "mean",
    sn_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    수준별 응답표 생성 (Minitab의 Response Table for Means / for S/N과 동등).

    각 인자별로 수준을 정렬한 후 "Level 1", "Level 2", ... 인덱스로 통일.
    인자마다 수준값이 다를 수 있으므로 (예: A=[10,20,30], B=[100,200]),
    "Level N"의 의미는 "각 인자의 N번째 작은 수준값"으로 통일.

    Parameters
    ----------
    df : DataFrame
        설계표 + 응답값
    factor_cols : list of str
        인자 컬럼명들
    response_col : str
        응답 컬럼명
    statistic : str
        "mean" : 평균
        "sn"   : S/N 비
    sn_type : str
        "larger", "smaller", "nominal" (statistic="sn"일 때만)

    Returns
    -------
    DataFrame: 인덱스=Level 1/2/3..., Delta, Rank / 컬럼=인자명
    """
    # 인자별 수준 매핑: {인자: [sorted unique values]}
    level_maps = {f: sorted(df[f].dropna().unique()) for f in factor_cols}
    max_levels = max(len(v) for v in level_maps.values())

    # 데이터 행렬 (행=수준 인덱스, 열=인자)
    matrix = np.full((max_levels, len(factor_cols)), np.nan)
    deltas = {}

    for j, fcol in enumerate(factor_cols):
        levels = level_maps[fcol]
        level_stats = []
        for i, lv in enumerate(levels):
            subset = df[df[fcol] == lv][response_col].dropna().values
            if len(subset) == 0:
                level_stats.append(np.nan)
                continue
            if statistic == "mean":
                level_stats.append(np.mean(subset))
            elif statistic == "sn":
                if sn_type is None:
                    raise ValueError("statistic='sn'일 때 sn_type 필요")
                level_stats.append(sn_ratio(subset, sn_type))
            else:
                raise ValueError(f"statistic은 'mean' 또는 'sn'. 입력: {statistic}")
            matrix[i, j] = level_stats[-1]

        valid = [v for v in level_stats if not np.isnan(v)]
        deltas[fcol] = max(valid) - min(valid) if valid else np.nan

    # DataFrame 구성
    index_labels = [f"Level {i+1}" for i in range(max_levels)]
    table = pd.DataFrame(matrix, index=index_labels, columns=factor_cols)

    # Delta, Rank 행
    table.loc["Delta"] = [deltas[f] for f in factor_cols]
    ranks = pd.Series(deltas).rank(ascending=False, method="min").astype(int)
    table.loc["Rank"] = [ranks[f] for f in factor_cols]

    return table


def get_level_map(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, List]:
    """각 인자의 정렬된 실제 수준값 리스트를 반환."""
    return {f: sorted(df[f].dropna().unique()) for f in factor_cols}


# ============================================================
# 2) 최적 수준 찾기
# ============================================================

def find_optimal_levels(
    response_tbl: pd.DataFrame,
    factor_cols: List[str],
    direction: str = "max",
) -> Dict[str, float]:
    """
    응답표에서 각 인자의 최적 수준 (level) 자동 식별.

    Parameters
    ----------
    response_tbl : DataFrame
        response_table()의 출력
    factor_cols : list
        인자명 리스트
    direction : str
        "max" : 응답값이 큰 수준이 좋음 (예: 강도, 효율, S/N비)
        "min" : 응답값이 작은 수준이 좋음 (예: 결함률, 소음)

    Returns
    -------
    dict : {인자명: 최적 수준값}
    """
    optimum = {}
    level_rows = [idx for idx in response_tbl.index if idx.startswith("Level")]

    for fcol in factor_cols:
        col = response_tbl.loc[level_rows, fcol]
        if direction == "max":
            best_label = col.idxmax()
        elif direction == "min":
            best_label = col.idxmin()
        else:
            raise ValueError(f"direction은 'max' 또는 'min'. 입력: {direction}")
        # "Level 2" -> 2
        level_value = int(best_label.split()[-1])
        optimum[fcol] = level_value

    return optimum


def predict_optimum(
    df: pd.DataFrame,
    factor_cols: List[str],
    response_col: str,
    optimum_levels: Dict[str, int],
    significant_factors: Optional[List[str]] = None,
) -> float:
    """
    최적 조건에서의 예측치 계산 (Taguchi 가법성 가정 기반).

    공식:
        y_pred = T_bar + sum_{i in significant} (M_i_optimum - T_bar)

    Parameters
    ----------
    df : DataFrame
        설계 + 결과
    factor_cols : list
        모든 인자
    response_col : str
        응답 컬럼
    optimum_levels : dict
        find_optimal_levels()의 출력
    significant_factors : list, optional
        ANOVA에서 유의한 인자만 사용. None이면 전체 인자.

    Returns
    -------
    float : 최적 조건에서의 예측 응답값
    """
    if significant_factors is None:
        significant_factors = factor_cols

    T_bar = df[response_col].mean()
    y_pred = T_bar

    for fcol in significant_factors:
        if fcol not in optimum_levels:
            continue
        # 해당 인자의 컬럼 값들 중에서 수준 인덱스에 해당하는 실제값 찾기
        # df[fcol]은 실제 수준값을 가지고 있고, optimum_levels[fcol]은 수준 인덱스(1,2,3)
        unique_levels = sorted(df[fcol].dropna().unique())
        level_idx = optimum_levels[fcol] - 1
        if level_idx < 0 or level_idx >= len(unique_levels):
            continue
        target_value = unique_levels[level_idx]
        M_optimum = df[df[fcol] == target_value][response_col].mean()
        y_pred += (M_optimum - T_bar)

    return y_pred


# ============================================================
# 3) ANOVA (분산분석)
# ============================================================

def anova_taguchi(
    df: pd.DataFrame,
    factor_cols: List[str],
    response_col: str,
    pool_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Taguchi 스타일 ANOVA 테이블 생성.

    Parameters
    ----------
    df : DataFrame
        시험 결과
    factor_cols : list
        인자 컬럼들
    response_col : str
        응답 컬럼
    pool_threshold : float, optional
        기여율(%)이 이 값 미만이면 오차항으로 풀링.
        예: 5.0이면 기여율 5% 미만 인자는 error에 합산.
        None이면 풀링 안 함.

    Returns
    -------
    DataFrame : ANOVA 테이블
        컬럼: Source, DF, SS, MS, F, P, Contribution(%)
    """
    df_clean = df.dropna(subset=[response_col]).copy()
    n_total = len(df_clean)
    grand_mean = df_clean[response_col].mean()

    # 전체 변동 (Total Sum of Squares)
    ss_total = ((df_clean[response_col] - grand_mean) ** 2).sum()

    # 인자별 SS, DF
    rows = []
    for fcol in factor_cols:
        levels = sorted(df_clean[fcol].dropna().unique())
        ss_factor = 0.0
        for lv in levels:
            subset = df_clean[df_clean[fcol] == lv][response_col]
            n_lv = len(subset)
            mean_lv = subset.mean()
            ss_factor += n_lv * (mean_lv - grand_mean) ** 2
        df_factor = len(levels) - 1
        rows.append({
            "Source": fcol,
            "DF": df_factor,
            "SS": ss_factor,
        })

    # 오차 SS = Total - sum(factor SS)
    ss_factors_sum = sum(r["SS"] for r in rows)
    ss_error = ss_total - ss_factors_sum
    df_error = n_total - 1 - sum(r["DF"] for r in rows)

    # 기여율 계산 후 풀링 적용
    if pool_threshold is not None:
        pooled_indices = []
        for i, r in enumerate(rows):
            contrib = 100 * r["SS"] / ss_total if ss_total > 0 else 0
            if contrib < pool_threshold:
                pooled_indices.append(i)
        # 풀링: 해당 인자들의 SS, DF를 error에 합산
        for idx in sorted(pooled_indices, reverse=True):
            ss_error += rows[idx]["SS"]
            df_error += rows[idx]["DF"]
            rows.pop(idx)

    # MS, F, P 계산
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    final_rows = []
    for r in rows:
        ms = r["SS"] / r["DF"] if r["DF"] > 0 else np.nan
        if ms_error and ms_error > 0 and not np.isnan(ms):
            f_stat = ms / ms_error
            # p-value via F distribution
            p_val = 1 - stats.f.cdf(f_stat, r["DF"], df_error) if df_error > 0 else np.nan
        else:
            f_stat = np.nan
            p_val = np.nan
        contrib = 100 * r["SS"] / ss_total if ss_total > 0 else 0
        final_rows.append({
            "Source": r["Source"],
            "DF": r["DF"],
            "SS": r["SS"],
            "MS": ms,
            "F": f_stat,
            "P": p_val,
            "Contribution(%)": contrib,
        })

    # Error 행
    final_rows.append({
        "Source": "Error",
        "DF": df_error,
        "SS": ss_error,
        "MS": ms_error,
        "F": np.nan,
        "P": np.nan,
        "Contribution(%)": 100 * ss_error / ss_total if ss_total > 0 else 0,
    })

    # Total 행
    final_rows.append({
        "Source": "Total",
        "DF": n_total - 1,
        "SS": ss_total,
        "MS": np.nan,
        "F": np.nan,
        "P": np.nan,
        "Contribution(%)": 100.0,
    })

    anova_df = pd.DataFrame(final_rows)
    return anova_df


# ============================================================
# 4) 자체 테스트
# ============================================================

if __name__ == "__main__":
    # 예제: L9 설계로 가상 데이터 생성 후 분석
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from taguchi import build_design

    factors = {
        "A": {"levels": [10, 20, 30], "unit": "mm"},
        "B": {"levels": [1, 2, 3], "unit": "-"},
        "C": {"levels": [100, 200, 300], "unit": "MPa"},
    }
    df = build_design("L9", factors, randomize=False, seed=0)
    # 가상 응답: A는 큰 효과, B는 중간, C는 작음
    rng = np.random.default_rng(0)
    df["Y"] = (
        50
        + 2.0 * (df["A"] - 20) / 10        # A 효과
        + 0.5 * (df["B"] - 2)              # B 효과 (작음)
        + 0.1 * (df["C"] - 200) / 100      # C 효과 (매우 작음)
        + rng.normal(0, 0.5, len(df))      # 잡음
    )
    df["Response_Y"] = df["Y"]

    print("=" * 60)
    print("분석 모듈 자체 테스트")
    print("=" * 60)
    print("\n[입력 데이터]")
    print(df.to_string(index=False))

    print("\n[평균 응답표]")
    tbl = response_table(df, ["A", "B", "C"], "Y", statistic="mean")
    print(tbl.round(3))

    print("\n[최적 수준 (Larger-the-better 가정)]")
    opt = find_optimal_levels(tbl, ["A", "B", "C"], direction="max")
    print(opt)

    print("\n[최적 조건 예측치]")
    y_pred = predict_optimum(df, ["A", "B", "C"], "Y", opt)
    print(f"  예측 응답값: {y_pred:.4f}")

    print("\n[ANOVA]")
    anova_tbl = anova_taguchi(df, ["A", "B", "C"], "Y")
    print(anova_tbl.round(4).to_string(index=False))
