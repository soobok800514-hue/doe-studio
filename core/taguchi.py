"""
Taguchi 직교배열표 (Orthogonal Array, OA) 모듈
================================================
표준 OA 테이블 정의 및 자동 추천 기능.

지원 OA:
- L4(2^3)   : 2수준 3인자 (4회 시험)
- L8(2^7)   : 2수준 7인자 (8회 시험)
- L9(3^4)   : 3수준 4인자 (9회 시험)
- L12(2^11) : 2수준 11인자 (12회 시험, Plackett-Burman 계열)
- L16(2^15) : 2수준 15인자 (16회 시험)
- L18(2^1 x 3^7) : 혼합수준 (18회 시험)
- L27(3^13) : 3수준 13인자 (27회 시험)

S/N 비 계산식 (Taguchi):
- Larger-the-better : eta = -10*log10( mean(1/y^2) )
- Smaller-the-better: eta = -10*log10( mean(y^2) )
- Nominal-the-best  : eta = 10*log10( ybar^2 / s^2 )

참고: Phadke, M.S. (1989) "Quality Engineering Using Robust Design"
     Roy, R.K. (2001) "Design of Experiments Using the Taguchi Approach"
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1) 표준 직교배열표 (값은 1, 2, 3 의 수준 인덱스)
# ============================================================

# L4 (2^3) - 4 runs, up to 3 factors at 2 levels
L4 = np.array([
    [1, 1, 1],
    [1, 2, 2],
    [2, 1, 2],
    [2, 2, 1],
])

# L8 (2^7) - 8 runs, up to 7 factors at 2 levels
L8 = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 2, 2, 2],
    [1, 2, 2, 1, 1, 2, 2],
    [1, 2, 2, 2, 2, 1, 1],
    [2, 1, 2, 1, 2, 1, 2],
    [2, 1, 2, 2, 1, 2, 1],
    [2, 2, 1, 1, 2, 2, 1],
    [2, 2, 1, 2, 1, 1, 2],
])

# L9 (3^4) - 9 runs, up to 4 factors at 3 levels
L9 = np.array([
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
])

# L12 (2^11) - 12 runs, up to 11 factors at 2 levels
L12 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
    [1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2],
    [1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1],
    [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1],
    [2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1],
    [2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2],
    [2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1],
    [2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2],
    [2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1],
])

# L16 (2^15) - 16 runs, up to 15 factors at 2 levels
L16 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1],
    [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
    [1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
    [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    [2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1],
    [2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],
    [2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2],
    [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
    [2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2],
    [2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2],
    [2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1],
])

# L18 (2^1 x 3^7) - 18 runs
# Column 1: 2-level, Columns 2~8: 3-level
L18 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 2, 2, 2, 2, 2],
    [1, 1, 3, 3, 3, 3, 3, 3],
    [1, 2, 1, 1, 2, 2, 3, 3],
    [1, 2, 2, 2, 3, 3, 1, 1],
    [1, 2, 3, 3, 1, 1, 2, 2],
    [1, 3, 1, 2, 1, 3, 2, 3],
    [1, 3, 2, 3, 2, 1, 3, 1],
    [1, 3, 3, 1, 3, 2, 1, 2],
    [2, 1, 1, 3, 3, 2, 2, 1],
    [2, 1, 2, 1, 1, 3, 3, 2],
    [2, 1, 3, 2, 2, 1, 1, 3],
    [2, 2, 1, 2, 3, 1, 3, 2],
    [2, 2, 2, 3, 1, 2, 1, 3],
    [2, 2, 3, 1, 2, 3, 2, 1],
    [2, 3, 1, 3, 2, 3, 1, 2],
    [2, 3, 2, 1, 3, 1, 2, 3],
    [2, 3, 3, 2, 1, 2, 3, 1],
])

# L27 (3^13) - 27 runs, up to 13 factors at 3 levels
def _generate_L27():
    """L27 직교배열표 자동 생성 (GF(3) 기반)."""
    rows = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # 13개 컬럼 생성 (모듈로 3 연산)
                row = [
                    i + 1,                              # col 1
                    j + 1,                              # col 2
                    ((i + j) % 3) + 1,                  # col 3
                    ((i + 2*j) % 3) + 1,                # col 4
                    k + 1,                              # col 5
                    ((i + k) % 3) + 1,                  # col 6
                    ((i + 2*k) % 3) + 1,                # col 7
                    ((j + k) % 3) + 1,                  # col 8
                    ((i + j + k) % 3) + 1,              # col 9
                    ((i + 2*j + k) % 3) + 1,            # col 10
                    ((j + 2*k) % 3) + 1,                # col 11
                    ((i + j + 2*k) % 3) + 1,            # col 12
                    ((i + 2*j + 2*k) % 3) + 1,          # col 13
                ]
                rows.append(row)
    return np.array(rows)

L27 = _generate_L27()


# ============================================================
# 2) OA 카탈로그 (메타데이터)
# ============================================================

OA_CATALOG = {
    "L4":  {"array": L4,  "runs": 4,  "levels": [2, 2, 2],          "max_factors": 3,  "type": "2수준"},
    "L8":  {"array": L8,  "runs": 8,  "levels": [2]*7,              "max_factors": 7,  "type": "2수준"},
    "L9":  {"array": L9,  "runs": 9,  "levels": [3]*4,              "max_factors": 4,  "type": "3수준"},
    "L12": {"array": L12, "runs": 12, "levels": [2]*11,             "max_factors": 11, "type": "2수준 (PB계열)"},
    "L16": {"array": L16, "runs": 16, "levels": [2]*15,             "max_factors": 15, "type": "2수준"},
    "L18": {"array": L18, "runs": 18, "levels": [2] + [3]*7,        "max_factors": 8,  "type": "혼합수준 (2x3^7)"},
    "L27": {"array": L27, "runs": 27, "levels": [3]*13,             "max_factors": 13, "type": "3수준"},
}


# ============================================================
# 3) OA 자동 추천
# ============================================================

def recommend_oa(n_factors: int, levels_per_factor: List[int]) -> str:
    """
    인자 수와 수준 정보를 기반으로 가장 작은 적합한 OA를 추천.

    Parameters
    ----------
    n_factors : int
        인자의 수
    levels_per_factor : List[int]
        각 인자의 수준 수 (예: [2, 2, 3, 3])

    Returns
    -------
    str : 추천 OA 이름 (예: "L9")

    예시
    ----
    >>> recommend_oa(3, [2, 2, 2])
    'L4'
    >>> recommend_oa(4, [3, 3, 3, 3])
    'L9'
    >>> recommend_oa(2, [2, 3])
    'L18'
    """
    n_2 = sum(1 for l in levels_per_factor if l == 2)
    n_3 = sum(1 for l in levels_per_factor if l == 3)
    if n_2 + n_3 != n_factors:
        raise ValueError("지원되는 수준: 2 또는 3. 4수준 이상은 별도 구현 필요.")

    only_2 = (n_3 == 0)
    only_3 = (n_2 == 0)

    if only_2:
        if n_factors <= 3:
            return "L4"
        elif n_factors <= 7:
            return "L8"
        elif n_factors <= 11:
            return "L12"
        elif n_factors <= 15:
            return "L16"
        else:
            raise ValueError(f"2수준 {n_factors}인자는 표준 OA로 처리 불가. L32 이상 필요.")
    elif only_3:
        if n_factors <= 4:
            return "L9"
        elif n_factors <= 13:
            return "L27"
        else:
            raise ValueError(f"3수준 {n_factors}인자는 표준 OA로 처리 불가.")
    else:
        # 혼합 수준
        # L18 = 2수준 컬럼 1개 + 3수준 컬럼 7개
        # 2수준 인자는 더미 수준(Dummy Level)으로 3수준 컬럼에도 배치 가능
        # 조건: n_3 <= 7, n_2 + n_3 <= 8 (총 8컬럼)
        if n_3 <= 7 and (n_2 + n_3) <= 8:
            return "L18"
        else:
            # 표준 OA로 직접 처리 불가능한 혼합 케이스
            # 가장 작은 권장 대안 계산
            n_all_3 = n_factors
            n_all_2 = n_factors
            if n_all_3 <= 4:
                alt_3 = "L9"
            elif n_all_3 <= 13:
                alt_3 = "L27"
            else:
                alt_3 = "L36 이상"
            if n_all_2 <= 3:
                alt_2 = "L4"
            elif n_all_2 <= 7:
                alt_2 = "L8"
            elif n_all_2 <= 11:
                alt_2 = "L12"
            elif n_all_2 <= 15:
                alt_2 = "L16"
            else:
                alt_2 = "L32 이상"

            raise ValueError(
                f"이 조합 (2수준 {n_2}개 + 3수준 {n_3}개)은 표준 직교배열표로 "
                f"직접 처리할 수 없습니다.\n"
                f"\n"
                f"  💡 권장 해결책 (가장 작은 시험 횟수 기준):\n"
                f"  1) 2수준 인자에 중간값을 추가하여 모두 3수준으로 통일 → {alt_3} 사용\n"
                f"     (예: X=[0, 1] → X=[0, 0.5, 1])\n"
                f"  2) 3수준 인자에서 중간 수준을 제거하여 모두 2수준으로 통일 → {alt_2} 사용\n"
                f"     (예: X=[10, 15, 20] → X=[10, 20])\n"
                f"  3) 완전요인배치 (모든 조합 시험): "
                f"{' × '.join(str(l) for l in levels_per_factor)} = "
                f"{int(np.prod(levels_per_factor))}회"
            )


def build_design(
    oa_name: str,
    factors: Dict[str, Dict],
    randomize: bool = True,
    seed: Optional[int] = 42,
    auto_reorder: bool = True,
) -> pd.DataFrame:
    """
    선택한 OA로 시험계획 매트릭스 생성.

    Parameters
    ----------
    oa_name : str
        OA 이름 (예: "L9")
    factors : dict
        {"인자명": {"levels": [수준1, 수준2, ...], "unit": "단위"}, ...}
        levels의 길이는 해당 OA의 수준 수와 일치해야 함.
    randomize : bool
        시험 순서 무작위화 여부 (Minitab 기본값과 동일하게 True 권장)
    seed : int
        무작위 시드
    auto_reorder : bool
        True (기본): 인자를 OA 컬럼 구조에 맞게 자동 재정렬.
        예) L18 (2수준 1개 + 3수준 7개)에 사용자가 (3수준, 2수준, 3수준) 순으로
            인자를 입력해도 자동으로 2수준 인자를 첫 컬럼으로 재배치.
        False: 입력 순서를 그대로 OA 컬럼에 1:1 매핑 (구버전 동작).

    Returns
    -------
    pd.DataFrame
        Run, 각 인자별 실제 수준값, 응답값 컬럼이 포함된 데이터프레임.

    예시
    ----
    >>> factors = {
    ...     "PT_시점":   {"levels": [5, 10, 15], "unit": "ms"},
    ...     "LL_임계":   {"levels": [3.0, 4.0, 5.0], "unit": "kN"},
    ...     "Webbing":   {"levels": [800, 1000, 1200], "unit": "N/mm"},
    ... }
    >>> df = build_design("L9", factors)
    """
    if oa_name not in OA_CATALOG:
        raise ValueError(f"지원하지 않는 OA: {oa_name}. 사용 가능: {list(OA_CATALOG.keys())}")

    oa = OA_CATALOG[oa_name]["array"]
    expected_levels = OA_CATALOG[oa_name]["levels"]
    n_runs, n_cols = oa.shape
    n_factors = len(factors)
    factor_names = list(factors.keys())

    if n_factors > n_cols:
        raise ValueError(f"{oa_name}는 최대 {n_cols}개 인자만 지원. 입력: {n_factors}")

    # 인자→OA 컬럼 매핑 결정
    if auto_reorder:
        # 수준 수별로 인자를 그룹화한 후 OA 컬럼 구조에 맞게 배치
        from collections import defaultdict
        factors_by_lv = defaultdict(list)
        for fname in factor_names:
            n_lv = len(factors[fname]["levels"])
            factors_by_lv[n_lv].append(fname)

        # OA 컬럼별로 어떤 인자를 매핑할지 결정 (사용자 입력 순서 보존)
        factor_to_column: Dict[str, int] = {}
        # 가용 OA 컬럼을 수준 수별로 정리
        from collections import deque
        available_cols_by_lv = defaultdict(deque)
        for col_idx, col_lvl in enumerate(expected_levels):
            available_cols_by_lv[col_lvl].append(col_idx)

        for fname in factor_names:
            n_lv = len(factors[fname]["levels"])
            if available_cols_by_lv[n_lv]:
                factor_to_column[fname] = available_cols_by_lv[n_lv].popleft()
            elif n_lv == 2 and available_cols_by_lv[3]:
                # 더미 수준(Dummy Level): 2수준 인자를 3수준 컬럼에 배치
                # OA 수준 1→Lv1, 2→Lv2, 3→Lv1(반복) — modulo 처리
                factor_to_column[fname] = available_cols_by_lv[3].popleft()
            else:
                # 매핑 실패 → 친절한 오류 메시지
                user_summary = {}
                for f in factor_names:
                    lv = len(factors[f]["levels"])
                    user_summary[lv] = user_summary.get(lv, 0) + 1
                oa_summary = {}
                for lv in expected_levels:
                    oa_summary[lv] = oa_summary.get(lv, 0) + 1

                # 가장 작은 권장 대안 계산
                n_all_3 = n_factors
                n_all_2 = n_factors
                if n_all_3 <= 4:
                    alt_3 = "L9 (9회 시험)"
                elif n_all_3 <= 13:
                    alt_3 = "L27 (27회 시험)"
                else:
                    alt_3 = "L36 이상 필요"
                if n_all_2 <= 3:
                    alt_2 = "L4 (4회 시험)"
                elif n_all_2 <= 7:
                    alt_2 = "L8 (8회 시험)"
                elif n_all_2 <= 11:
                    alt_2 = "L12 (12회 시험)"
                elif n_all_2 <= 15:
                    alt_2 = "L16 (16회 시험)"
                else:
                    alt_2 = "L32 이상 필요"

                n_full = int(np.prod([len(factors[f]["levels"]) for f in factor_names]))

                msg = (
                    f"{oa_name}에 인자 '{fname}'({n_lv}수준)을 매핑할 컬럼이 없습니다.\n"
                    f"\n"
                    f"  입력 인자 구성: "
                    f"2수준 {user_summary.get(2, 0)}개 + 3수준 {user_summary.get(3, 0)}개 "
                    f"(총 {n_factors}개)\n"
                    f"  {oa_name} 컬럼 구성: "
                    f"2수준 {oa_summary.get(2, 0)}개 + 3수준 {oa_summary.get(3, 0)}개 "
                    f"(총 {n_cols}개)\n"
                    f"\n"
                    f"  💡 해결 방법 (세 가지 중 선택):\n"
                    f"  ① 2수준 인자에 중간값을 추가하여 모두 3수준으로 → {alt_3}\n"
                    f"     예: X=[0, 1] → X=[0, 0.5, 1]\n"
                    f"  ② 3수준 인자에서 중간 수준을 제거하여 모두 2수준으로 → {alt_2}\n"
                    f"     예: X=[10, 15, 20] → X=[10, 20]\n"
                    f"  ③ 완전요인배치 사용 (모든 조합): {n_full}회 시험 "
                    f"(인자 수가 적을 때만 권장)\n"
                )
                raise ValueError(msg)
    else:
        # 구버전 동작: 입력 순서대로 1:1 매핑
        factor_to_column = {fname: i for i, fname in enumerate(factor_names)}
        for i, fname in enumerate(factor_names):
            if i >= len(expected_levels):
                break
            actual = len(factors[fname]["levels"])
            expected = expected_levels[i]
            if actual != expected:
                raise ValueError(
                    f"인자 '{fname}'은 {expected}수준이어야 함. 입력: {actual}수준."
                )

    # 설계 매트릭스 구성 (사용자 입력 순서대로 컬럼 배치)
    design_data = {"Std_Order": list(range(1, n_runs + 1))}
    for fname in factor_names:
        col_idx = factor_to_column[fname]
        level_values = factors[fname]["levels"]
        n_lvl = len(level_values)
        # modulo 처리: 더미 수준 인자(2수준)가 3수준 컬럼에 있을 때 OA값 3 → index 0(Lv1 반복)
        design_data[fname] = [level_values[(oa[r, col_idx] - 1) % n_lvl] for r in range(n_runs)]

    df = pd.DataFrame(design_data)

    # 무작위화
    if randomize:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_runs)
        df = df.iloc[perm].reset_index(drop=True)
        df.insert(0, "Run_Order", list(range(1, n_runs + 1)))
    else:
        df.insert(0, "Run_Order", df["Std_Order"])

    # 응답값 컬럼 (사용자 입력 대기)
    df["Response_Y"] = np.nan

    return df


# ============================================================
# 4) S/N 비 계산
# ============================================================

def sn_ratio(y: np.ndarray, sn_type: str = "larger") -> float:
    """
    Taguchi S/N 비 계산.

    Parameters
    ----------
    y : array-like
        하나의 시험 조건에서 측정한 반복 측정값들.
        반복이 없으면 단일 값이어도 됨 (단, nominal-the-best는 반복 필수).
    sn_type : str
        "larger"  : Larger-the-better (망대특성, 강도/효율 등)
        "smaller" : Smaller-the-better (망소특성, 결함/소음 등)
        "nominal" : Nominal-the-best (망목특성, 목표치 정확도)

    Returns
    -------
    float : S/N 비 (dB)

    공식
    ----
    Larger : eta = -10 * log10( mean(1/y^2) )
    Smaller: eta = -10 * log10( mean(y^2) )
    Nominal: eta = 10 * log10( ybar^2 / s^2 )

    Phadke (1989), Roy (2001) 표준 정의.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.nan

    if sn_type == "larger":
        # 0 또는 음수가 있으면 계산 불가
        if np.any(y <= 0):
            return np.nan
        return -10.0 * np.log10(np.mean(1.0 / (y ** 2)))

    elif sn_type == "smaller":
        return -10.0 * np.log10(np.mean(y ** 2))

    elif sn_type == "nominal":
        if len(y) < 2:
            # 분산을 알 수 없음 -> NaN
            return np.nan
        ybar = np.mean(y)
        s2 = np.var(y, ddof=1)
        if s2 <= 0 or ybar == 0:
            return np.nan
        return 10.0 * np.log10(ybar ** 2 / s2)

    else:
        raise ValueError(f"sn_type은 'larger', 'smaller', 'nominal' 중 하나여야 함. 입력: {sn_type}")


# ============================================================
# 5) 자체 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Taguchi OA 모듈 자체 테스트")
    print("=" * 60)

    # OA 직교성 검증 (모든 OA에 대해)
    for name, info in OA_CATALOG.items():
        oa = info["array"]
        n_runs, n_cols = oa.shape
        print(f"\n[{name}] {n_runs} runs, {n_cols} columns, type={info['type']}")
        print(f"  Shape: {oa.shape}, OK" if oa.shape[0] == info["runs"] else "  X 크기 불일치!")

    # 추천 테스트
    print("\n[OA 추천 테스트]")
    cases = [
        (3, [2, 2, 2]),
        (4, [3, 3, 3, 3]),
        (5, [2, 2, 2, 2, 2]),
        (4, [2, 3, 3, 3]),
    ]
    for n, lv in cases:
        rec = recommend_oa(n, lv)
        print(f"  {n}인자, levels={lv} -> {rec}")

    # 설계 생성 예시
    print("\n[설계 생성 예시: L9]")
    factors = {
        "PT_시점":   {"levels": [5, 10, 15], "unit": "ms"},
        "LL_임계":   {"levels": [3.0, 4.0, 5.0], "unit": "kN"},
        "Webbing":   {"levels": [800, 1000, 1200], "unit": "N/mm"},
    }
    df = build_design("L9", factors, randomize=True, seed=42)
    print(df.to_string(index=False))

    # S/N 비 예시
    print("\n[S/N 비 계산 예시]")
    y_sample = [3.5, 3.7, 3.6]  # 3회 반복
    print(f"  데이터: {y_sample}")
    print(f"  Larger : {sn_ratio(y_sample, 'larger'):.4f} dB")
    print(f"  Smaller: {sn_ratio(y_sample, 'smaller'):.4f} dB")
    print(f"  Nominal: {sn_ratio(y_sample, 'nominal'):.4f} dB")
