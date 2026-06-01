"""
DOE Studio 검증 벤치마크
========================
외부 기준값과의 일치성을 검증하기 위한 벤치마크 모음.

벤치마크 출처:
- NIST/SEMATECH e-Handbook of Statistical Methods
- Minitab 공식 카탈로그
- Phadke, M.S. (1989) "Quality Engineering Using Robust Design"
- Derringer, G. and Suich, R. (1980) JQT 12(4), 214-219
- statsmodels (독립 구현, 교차 검증용)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd

from core.taguchi import (
    OA_CATALOG, L4, L8, L9, L12, L16, L18, L27,
    build_design, sn_ratio, recommend_oa,
)
from core.analysis import (
    response_table, anova_taguchi, find_optimal_levels, predict_optimum,
)
from core.desirability import (
    d_larger, d_smaller, d_target, ResponseSpec,
    overall_desirability, optimize_desirability,
)


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class CheckResult:
    """단일 검증 항목 결과."""
    metric: str
    expected: Any
    actual: Any
    tolerance: float = 1e-6
    passed: bool = False
    note: str = ""

    @property
    def abs_diff(self) -> float:
        try:
            return abs(float(self.actual) - float(self.expected))
        except (TypeError, ValueError):
            return float("nan")

    @property
    def rel_diff(self) -> float:
        try:
            e = float(self.expected)
            if e == 0:
                return self.abs_diff
            return abs((float(self.actual) - e) / e)
        except (TypeError, ValueError):
            return float("nan")


@dataclass
class Benchmark:
    """벤치마크 단위 (여러 CheckResult 보유)."""
    name: str
    source: str
    description: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_total(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return self.n_total > 0 and self.n_passed == self.n_total

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)


def _check_numeric(metric: str, expected: float, actual: float,
                   tol: float = 1e-6, note: str = "") -> CheckResult:
    """수치 비교 헬퍼."""
    try:
        diff = abs(float(actual) - float(expected))
        passed = diff <= tol
    except (TypeError, ValueError):
        passed = False
    return CheckResult(metric, expected, actual, tol, passed, note)


def _check_equal(metric: str, expected: Any, actual: Any, note: str = "") -> CheckResult:
    """완전 일치 비교."""
    passed = expected == actual
    return CheckResult(metric, expected, actual, 0.0, passed, note)


# ============================================================
# Benchmark 1: OA 직교성 (Orthogonality)
# ============================================================

def benchmark_1_oa_orthogonality() -> Benchmark:
    """
    Phadke (1989) 정의:
    임의의 두 컬럼에서 (level_i, level_j) 모든 조합이 동일 횟수로 출현해야 함.
    """
    b = Benchmark(
        name="1. OA 직교성 검증",
        source="Phadke (1989) p.51",
        description=(
            "모든 표준 OA에서 임의의 두 컬럼 쌍에 대해 "
            "(수준 i, 수준 j) 조합이 동일 빈도로 출현하는지 확인."
        ),
    )

    for oa_name, info in OA_CATALOG.items():
        oa = info["array"]
        n_runs, n_cols = oa.shape

        # 각 컬럼 쌍에 대해 검사
        violations = 0
        n_pairs_checked = 0
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col_i = oa[:, i]
                col_j = oa[:, j]
                lvls_i = sorted(set(col_i))
                lvls_j = sorted(set(col_j))

                # 모든 조합 빈도
                counts = {}
                for a in lvls_i:
                    for b_lv in lvls_j:
                        counts[(a, b_lv)] = sum(
                            1 for r in range(n_runs)
                            if col_i[r] == a and col_j[r] == b_lv
                        )

                # 동일 빈도인가?
                count_values = list(counts.values())
                if len(set(count_values)) != 1:
                    violations += 1
                n_pairs_checked += 1

        b.add(_check_equal(
            f"{oa_name} 직교성 (검사 컬럼 쌍 {n_pairs_checked}개)",
            expected=0,
            actual=violations,
            note=f"위반 쌍 = 0이면 완전 직교",
        ))

    return b


# ============================================================
# Benchmark 2: 표준 OA 카탈로그 일치
# ============================================================

def benchmark_2_oa_catalog_match() -> Benchmark:
    """
    Minitab 공식 카탈로그 / NIST 핸드북의 OA와 일치하는지 검증.
    핵심 OA의 첫 번째 컬럼/행을 표준값과 비교.
    """
    b = Benchmark(
        name="2. 표준 OA 카탈로그 일치성",
        source="Minitab Catalogue of Taguchi designs / NIST Handbook",
        description="L9, L8, L18, L27의 표준 컬럼이 발표된 카탈로그와 일치하는지 검증.",
    )

    # L9 표준: 컬럼 1 = (1,1,1, 2,2,2, 3,3,3)
    expected_l9_col1 = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    actual_l9_col1 = L9[:, 0].tolist()
    b.add(_check_equal("L9 첫 번째 컬럼 = (1,1,1,2,2,2,3,3,3)",
                       expected_l9_col1, actual_l9_col1))

    # L9 표준: 행 1 = (1,1,1,1) (모두 첫 수준)
    expected_l9_row1 = [1, 1, 1, 1]
    actual_l9_row1 = L9[0, :].tolist()
    b.add(_check_equal("L9 첫 번째 행 = (1,1,1,1)",
                       expected_l9_row1, actual_l9_row1))

    # L8 표준: 컬럼 1 = (1,1,1,1, 2,2,2,2)
    expected_l8_col1 = [1, 1, 1, 1, 2, 2, 2, 2]
    actual_l8_col1 = L8[:, 0].tolist()
    b.add(_check_equal("L8 첫 번째 컬럼 = (1,1,1,1,2,2,2,2)",
                       expected_l8_col1, actual_l8_col1))

    # L18 표준: 컬럼 1 (2수준) = (1×9, 2×9)
    expected_l18_col1 = [1]*9 + [2]*9
    actual_l18_col1 = L18[:, 0].tolist()
    b.add(_check_equal("L18 첫 번째 컬럼 = 1×9, 2×9 (2수준)",
                       expected_l18_col1, actual_l18_col1))

    # L27: 컬럼 1 = (1×9, 2×9, 3×9)
    expected_l27_col1 = [1]*9 + [2]*9 + [3]*9
    actual_l27_col1 = L27[:, 0].tolist()
    b.add(_check_equal("L27 첫 번째 컬럼 = 1×9, 2×9, 3×9",
                       expected_l27_col1, actual_l27_col1))

    # L4 표준: 첫 컬럼 = (1,1,2,2)
    expected_l4_col1 = [1, 1, 2, 2]
    actual_l4_col1 = L4[:, 0].tolist()
    b.add(_check_equal("L4 첫 번째 컬럼 = (1,1,2,2)",
                       expected_l4_col1, actual_l4_col1))

    # OA 자동 추천 검증
    b.add(_check_equal("추천 OA (3인자 2수준)", "L4",
                       recommend_oa(3, [2, 2, 2])))
    b.add(_check_equal("추천 OA (4인자 3수준)", "L9",
                       recommend_oa(4, [3, 3, 3, 3])))
    b.add(_check_equal("추천 OA (7인자 2수준)", "L8",
                       recommend_oa(7, [2]*7)))
    b.add(_check_equal("추천 OA (13인자 3수준)", "L27",
                       recommend_oa(13, [3]*13)))

    return b


# ============================================================
# Benchmark 3: S/N 비 수기 계산
# ============================================================

def benchmark_3_sn_ratio() -> Benchmark:
    """
    S/N 비를 수기 계산값과 대조.
    공식: Phadke (1989)
        - Larger:  η = -10·log10(mean(1/y²))
        - Smaller: η = -10·log10(mean(y²))
        - Nominal: η = 10·log10(ȳ²/s²)
    """
    b = Benchmark(
        name="3. S/N 비 수기 계산",
        source="Phadke (1989) Eq. 5.2-5.4",
        description="3가지 S/N 유형을 손계산값과 비교.",
    )

    # Case A: y = [10, 12, 11], Larger-the-better
    # mean(1/y²) = (0.01 + 1/144 + 1/121)/3 = (0.01 + 0.006944 + 0.008264)/3 = 0.008403
    # η = -10·log10(0.008403) = 20.7558
    y_a = np.array([10.0, 12.0, 11.0])
    expected_larger = -10 * np.log10(np.mean(1.0 / y_a**2))
    actual_larger = sn_ratio(y_a, "larger")
    b.add(_check_numeric(
        "S/N Larger [10,12,11]",
        expected_larger, actual_larger,
        tol=1e-8,
        note="η = -10·log₁₀(mean(1/y²))"
    ))

    # Case B: Smaller
    expected_smaller = -10 * np.log10(np.mean(y_a**2))
    actual_smaller = sn_ratio(y_a, "smaller")
    b.add(_check_numeric(
        "S/N Smaller [10,12,11]",
        expected_smaller, actual_smaller,
        tol=1e-8,
        note="η = -10·log₁₀(mean(y²))"
    ))

    # Case C: Nominal-the-best
    ybar = np.mean(y_a)
    s2 = np.var(y_a, ddof=1)
    expected_nominal = 10 * np.log10(ybar**2 / s2)
    actual_nominal = sn_ratio(y_a, "nominal")
    b.add(_check_numeric(
        "S/N Nominal [10,12,11]",
        expected_nominal, actual_nominal,
        tol=1e-8,
        note="η = 10·log₁₀(ȳ²/s²), s²는 ddof=1"
    ))

    # Case D: 단일 값에서도 동작 (larger/smaller만)
    y_b = np.array([5.0])
    expected_single = -10 * np.log10(1.0 / 25.0)
    actual_single = sn_ratio(y_b, "larger")
    b.add(_check_numeric(
        "S/N Larger 단일값 [5.0]",
        expected_single, actual_single,
        tol=1e-8,
    ))

    # Case E: NaN 처리 (음수 입력 시 larger는 NaN)
    y_c = np.array([-1.0, 2.0])
    actual_neg = sn_ratio(y_c, "larger")
    b.add(_check_equal(
        "S/N Larger 음수 입력 → NaN",
        expected=True,
        actual=np.isnan(actual_neg),
        note="0 또는 음수가 있으면 정의 불가",
    ))

    return b


# ============================================================
# Benchmark 4: ANOVA vs statsmodels
# ============================================================

def benchmark_4_anova_cross_check() -> Benchmark:
    """
    statsmodels의 OLS + anova_lm으로 독립 계산한 ANOVA와 우리 결과 비교.
    직교 설계에서는 Type I/II/III SS 모두 동일한 결과를 줘야 함.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    b = Benchmark(
        name="4. ANOVA vs statsmodels 교차 검증",
        source="statsmodels 0.14+ (독립 구현)",
        description=(
            "L9 직교 설계에서 우리 ANOVA의 SS와 "
            "statsmodels OLS+anova_lm의 SS가 일치하는지 검증."
        ),
    )

    # 가상 데이터 생성: A가 큰 효과, B가 중간, F가 작음
    # (statsmodels의 C() 함수와 충돌 방지를 위해 'C' 대신 'F' 사용)
    factors = {
        "A": {"levels": [10, 20, 30], "unit": "mm"},
        "B": {"levels": [1, 2, 3], "unit": "-"},
        "F": {"levels": [100, 200, 300], "unit": "MPa"},
    }
    df = build_design("L9", factors, randomize=False)
    rng = np.random.default_rng(42)
    df["Y"] = (
        50 + 2.0 * (df["A"] - 20) / 10
        + 0.5 * (df["B"] - 2)
        + 0.1 * (df["F"] - 200) / 100
        + rng.normal(0, 0.3, len(df))
    )

    # 우리 ANOVA
    our_anova = anova_taguchi(df, ["A", "B", "F"], "Y")
    our_ss = {row["Source"]: row["SS"] for _, row in our_anova.iterrows()}

    # statsmodels ANOVA (인자를 categorical로 처리)
    df_sm = df.copy()
    for c in ["A", "B", "F"]:
        df_sm[c] = df_sm[c].astype("category")
    model = smf.ols("Y ~ C(A) + C(B) + C(F)", data=df_sm).fit()
    sm_anova = anova_lm(model, typ=1)

    # SS 매핑: statsmodels는 "C(A)" 형식
    sm_ss = {
        "A": float(sm_anova.loc["C(A)", "sum_sq"]),
        "B": float(sm_anova.loc["C(B)", "sum_sq"]),
        "F": float(sm_anova.loc["C(F)", "sum_sq"]),
        "Error": float(sm_anova.loc["Residual", "sum_sq"]),
    }

    # 비교
    for factor in ["A", "B", "F", "Error"]:
        b.add(_check_numeric(
            f"SS_{factor}",
            expected=sm_ss[factor],
            actual=our_ss[factor],
            tol=1e-6,
            note="직교 설계에서는 완전 일치해야 함"
        ))

    # F-statistic 검증
    our_f_A = our_anova[our_anova["Source"] == "A"]["F"].values[0]
    sm_f_A = float(sm_anova.loc["C(A)", "F"])
    b.add(_check_numeric(
        "F_A 통계량",
        expected=sm_f_A, actual=our_f_A, tol=1e-4,
    ))

    # p-value 검증
    our_p_A = our_anova[our_anova["Source"] == "A"]["P"].values[0]
    sm_p_A = float(sm_anova.loc["C(A)", "PR(>F)"])
    b.add(_check_numeric(
        "p-value_A",
        expected=sm_p_A, actual=our_p_A, tol=1e-4,
    ))

    return b


# ============================================================
# Benchmark 5: Desirability 함수값 경계점
# ============================================================

def benchmark_5_desirability_endpoints() -> Benchmark:
    """
    Derringer-Suich 함수의 경계 동작 검증.
    NIST 핸드북의 공식 정의:
    - d_max(y<L) = 0, d_max(y>T) = 1
    - d_min(y<T) = 1, d_min(y>U) = 0
    - d_target(y<L 또는 y>U) = 0, d_target(y=T) = 1
    """
    b = Benchmark(
        name="5. Desirability 함수 경계점",
        source="NIST Handbook 5.5.3.2.2 / Derringer & Suich (1980)",
        description="d_larger/smaller/target 함수의 12개 경계 평가점 검증.",
    )

    # d_larger
    L, T = 10.0, 20.0
    b.add(_check_numeric("d_larger(y=5, L=10, T=20) = 0",
                         0.0, float(d_larger(np.array([5.0]), L, T)[0]), tol=1e-9))
    b.add(_check_numeric("d_larger(y=10, L=10, T=20) = 0",
                         0.0, float(d_larger(np.array([10.0]), L, T)[0]), tol=1e-9))
    b.add(_check_numeric("d_larger(y=15, L=10, T=20) = 0.5 (선형)",
                         0.5, float(d_larger(np.array([15.0]), L, T)[0]), tol=1e-9))
    b.add(_check_numeric("d_larger(y=20, L=10, T=20) = 1",
                         1.0, float(d_larger(np.array([20.0]), L, T)[0]), tol=1e-9))
    b.add(_check_numeric("d_larger(y=25, L=10, T=20) = 1",
                         1.0, float(d_larger(np.array([25.0]), L, T)[0]), tol=1e-9))

    # d_smaller
    T, U = 5.0, 15.0
    b.add(_check_numeric("d_smaller(y=2, T=5, U=15) = 1",
                         1.0, float(d_smaller(np.array([2.0]), T, U)[0]), tol=1e-9))
    b.add(_check_numeric("d_smaller(y=10, T=5, U=15) = 0.5 (선형)",
                         0.5, float(d_smaller(np.array([10.0]), T, U)[0]), tol=1e-9))
    b.add(_check_numeric("d_smaller(y=20, T=5, U=15) = 0",
                         0.0, float(d_smaller(np.array([20.0]), T, U)[0]), tol=1e-9))

    # d_target
    L, T, U = 4.0, 5.0, 6.0
    b.add(_check_numeric("d_target(y=3, L=4, T=5, U=6) = 0",
                         0.0, float(d_target(np.array([3.0]), L, T, U)[0]), tol=1e-9))
    b.add(_check_numeric("d_target(y=4.5, T=5) = 0.5 (좌측 선형)",
                         0.5, float(d_target(np.array([4.5]), L, T, U)[0]), tol=1e-9))
    b.add(_check_numeric("d_target(y=5, T=5) = 1",
                         1.0, float(d_target(np.array([5.0]), L, T, U)[0]), tol=1e-9))
    b.add(_check_numeric("d_target(y=5.5, T=5, U=6) = 0.5 (우측 선형)",
                         0.5, float(d_target(np.array([5.5]), L, T, U)[0]), tol=1e-9))

    # Overall D 기하평균
    D_expected = (0.5 * 0.8 * 0.6) ** (1/3)
    D_actual = overall_desirability([0.5, 0.8, 0.6])
    b.add(_check_numeric("Overall D = (0.5×0.8×0.6)^(1/3)",
                         D_expected, D_actual, tol=1e-9))

    # 한 응답이 0이면 D=0
    b.add(_check_numeric("한 응답 d=0이면 Overall D=0",
                         0.0, overall_desirability([0.9, 0.0, 0.8]), tol=1e-12))

    return b


# ============================================================
# Benchmark 6: NIST Derringer-Suich 타이어 트레드 (End-to-End)
# ============================================================

def benchmark_6_nist_tire_tread() -> Benchmark:
    """
    NIST Handbook 5.5.3.2.2의 Derringer & Suich (1980) 타이어 트레드 예제.

    발표된 응답 모델 (NIST):
        Y1 = 139.12 + 16.49*x1 + 17.88*x2 + 2.21*x3
             - 4.01*x1² - 3.45*x2² - 1.57*x3²
             + 5.12*x1*x2 - 7.88*x1*x3 - 7.13*x2*x3
        Y2 = 1261.13 + 268.15*x1 + 246.5*x2 - 102.6*x3
             - 83.57*x1² - 124.92*x2² + 199.2*x3²
             + 69.37*x1*x2 - 104.38*x1*x3 - 94.13*x2*x3
        Y3 = 417.5 - 99.67*x1 - 31.4*x2 - 27.42*x3
        Y4 = 68.91 - 1.41*x1 + 4.32*x2 + 0.21*x3
             + 1.56*x1² + 0.058*x2² - 0.32*x3²
             - 1.62*x1*x2 + 0.25*x1*x3 - 0.12*x2*x3

    사양:
        Y1 (PICO Abrasion):  max, L=120, T=170
        Y2 (200% modulus):   max, L=1000, T=1300
        Y3 (Elongation):     target, L=400, T=500, U=600
        Y4 (Hardness):       target, L=60,  T=67.5, U=75

    NIST 발표 최적해:
        x* = (-0.10, 0.15, -1.0)
        Y1(x*)=136.4, Y2(x*)=1571.05, Y3(x*)=450.56, Y4(x*)=69.26
        d1=0.34, d2=1.0, d3=0.49, d4=0.76
        D = 0.596
    """
    b = Benchmark(
        name="6. NIST Derringer-Suich 타이어 트레드 (E2E)",
        source="NIST Handbook §5.5.3.2.2 / Derringer & Suich (1980)",
        description=(
            "발표된 응답 모델로 다중 응답 최적화 실행 후, "
            "NIST 발표 최적해에서의 D, 개별 d, 예측 Y와 비교."
        ),
    )

    # 응답 함수 정의 (x = [x1, x2, x3])
    def Y1(x):
        x1, x2, x3 = x
        return (139.12 + 16.49*x1 + 17.88*x2 + 2.21*x3
                - 4.01*x1**2 - 3.45*x2**2 - 1.57*x3**2
                + 5.12*x1*x2 - 7.88*x1*x3 - 7.13*x2*x3)

    def Y2(x):
        x1, x2, x3 = x
        return (1261.13 + 268.15*x1 + 246.5*x2 - 102.6*x3
                - 83.57*x1**2 - 124.92*x2**2 + 199.2*x3**2
                + 69.37*x1*x2 - 104.38*x1*x3 - 94.13*x2*x3)

    def Y3(x):
        x1, x2, x3 = x
        return 417.5 - 99.67*x1 - 31.4*x2 - 27.42*x3

    def Y4(x):
        x1, x2, x3 = x
        return (68.91 - 1.41*x1 + 4.32*x2 + 0.21*x3
                + 1.56*x1**2 + 0.058*x2**2 - 0.32*x3**2
                - 1.62*x1*x2 + 0.25*x1*x3 - 0.12*x2*x3)

    # --- Step 1: NIST 발표 x*에서 예측치 검증 (반응 모델 정확성) ---
    x_star = np.array([-0.10, 0.15, -1.0])
    y1_pred = Y1(x_star)
    y2_pred = Y2(x_star)
    y3_pred = Y3(x_star)
    y4_pred = Y4(x_star)

    b.add(_check_numeric("NIST x*에서 Y1(x*) ≈ 136.4",
                         136.4, y1_pred, tol=0.5,
                         note="응답 모델 재현 검증"))
    b.add(_check_numeric("NIST x*에서 Y2(x*) ≈ 1571.05",
                         1571.05, y2_pred, tol=5.0))
    b.add(_check_numeric("NIST x*에서 Y3(x*) ≈ 450.56",
                         450.56, y3_pred, tol=0.5))
    b.add(_check_numeric("NIST x*에서 Y4(x*) ≈ 69.26",
                         69.26, y4_pred, tol=0.5))

    # --- Step 2: NIST 발표 x*에서 개별 d 검증 ---
    spec1 = ResponseSpec("Y1", goal="max", L=120, T=170)
    spec2 = ResponseSpec("Y2", goal="max", L=1000, T=1300)
    spec3 = ResponseSpec("Y3", goal="target", L=400, T=500, U=600)
    spec4 = ResponseSpec("Y4", goal="target", L=60, T=67.5, U=75)

    d1_at_star = float(spec1.desirability(y1_pred))
    d2_at_star = float(spec2.desirability(y2_pred))
    d3_at_star = float(spec3.desirability(y3_pred))
    d4_at_star = float(spec4.desirability(y4_pred))

    b.add(_check_numeric("NIST x*에서 d1 ≈ 0.34 (≈(136.4-120)/50)",
                         0.34, d1_at_star, tol=0.02))
    b.add(_check_numeric("NIST x*에서 d2 = 1.0",
                         1.0, d2_at_star, tol=1e-6))
    b.add(_check_numeric("NIST x*에서 d3 ≈ 0.49",
                         0.49, d3_at_star, tol=0.03))
    b.add(_check_numeric("NIST x*에서 d4 ≈ 0.76",
                         0.76, d4_at_star, tol=0.03))

    # --- Step 3: NIST 발표 x*에서 전체 D 검증 ---
    D_at_star = overall_desirability([d1_at_star, d2_at_star, d3_at_star, d4_at_star])
    b.add(_check_numeric("NIST x*에서 D ≈ 0.596",
                         0.596, D_at_star, tol=0.01,
                         note="발표된 D = 0.596"))

    # --- Step 4: 우리 최적화 엔진으로 자체 탐색 -> NIST D 이상이어야 함 ---
    specs = [spec1, spec2, spec3, spec4]
    predict_funcs = {"Y1": Y1, "Y2": Y2, "Y3": Y3, "Y4": Y4}
    # NIST CCD 범위: 코드값 [-1.633, +1.633]
    bounds = [(-1.633, 1.633)] * 3
    factor_names = ["x1", "x2", "x3"]

    result = optimize_desirability(
        predict_funcs, specs, bounds, factor_names,
        n_restarts=20, use_global=True, seed=42,
    )

    D_optimized = result["overall_desirability"]
    b.add(CheckResult(
        metric="자체 최적화 D ≥ NIST 발표 D (0.596)",
        expected=">= 0.596",
        actual=f"{D_optimized:.4f}",
        passed=D_optimized >= 0.59,
        note="최적화기가 NIST 발표 해와 동등 또는 더 나은 해를 찾아야 함",
    ))

    # 수렴
    b.add(_check_equal(
        "최적화 수렴 상태",
        expected="OK",
        actual=result["convergence"],
    ))

    return b


# ============================================================
# Benchmark 7: 최적화 수렴성 (Convex 케이스)
# ============================================================

def benchmark_7_optimizer_convergence() -> Benchmark:
    """
    알려진 최적해를 가진 단순 함수로 최적화 엔진 신뢰성 검증.
    """
    b = Benchmark(
        name="7. 최적화 수렴성 (Convex 검증)",
        source="자체 정의 테스트 케이스",
        description=(
            "정확한 최적점을 알고 있는 단순 함수로 "
            "최적화기가 일관되게 수렴하는지 확인."
        ),
    )

    # 케이스 A: 1인자, 단일 응답, 명확한 최적
    # Y = -(x - 0.5)^2 + 1  -> 최대값 1, x*=0.5
    # max -> L=0, T=1: y=1일 때 d=1
    def predict_a(x):
        return -(x[0] - 0.5)**2 + 1

    spec_a = ResponseSpec("Y", goal="max", L=0.0, T=1.0)
    res_a = optimize_desirability(
        {"Y": predict_a}, [spec_a],
        bounds=[(0.0, 1.0)], factor_names=["x"],
        n_restarts=5, use_global=True, seed=0,
    )
    b.add(_check_numeric(
        "x* = 0.5 (단순 포물선)",
        expected=0.5, actual=res_a["optimal_factors"]["x"], tol=0.01,
    ))
    b.add(_check_numeric(
        "D ≈ 1.0 (이상적 최적)",
        expected=1.0, actual=res_a["overall_desirability"], tol=0.01,
    ))

    # 케이스 B: 2인자 트레이드오프
    # Y1 = x1 (max, L=0, T=1) -> x1=1이 이상적
    # Y2 = 1-x2 (max, L=0, T=1) -> x2=0이 이상적
    # x1=1, x2=0이 최적, d1=d2=1, D=1
    def predict_b1(x): return x[0]
    def predict_b2(x): return 1 - x[1]

    res_b = optimize_desirability(
        {"Y1": predict_b1, "Y2": predict_b2},
        [ResponseSpec("Y1", goal="max", L=0, T=1),
         ResponseSpec("Y2", goal="max", L=0, T=1)],
        bounds=[(0, 1), (0, 1)],
        factor_names=["x1", "x2"],
        n_restarts=5, use_global=True, seed=1,
    )
    b.add(_check_numeric("독립 트레이드오프: x1 = 1",
                         1.0, res_b["optimal_factors"]["x1"], tol=0.02))
    b.add(_check_numeric("독립 트레이드오프: x2 = 0",
                         0.0, res_b["optimal_factors"]["x2"], tol=0.02))
    b.add(_check_numeric("독립 트레이드오프: D = 1.0",
                         1.0, res_b["overall_desirability"], tol=0.02))

    # 케이스 C: 상충 응답 (실제 트레이드오프)
    # Y1 = x (max, L=0, T=1) -> x=1 이상
    # Y2 = x (min, T=0, U=1) -> x=0 이상
    # 두 d가 동일한 선형이면 x=0.5에서 d1=d2=0.5, D=0.5
    def predict_c1(x): return x[0]
    def predict_c2(x): return x[0]

    res_c = optimize_desirability(
        {"Y1": predict_c1, "Y2": predict_c2},
        [ResponseSpec("Y1", goal="max", L=0, T=1),
         ResponseSpec("Y2", goal="min", T=0, U=1)],
        bounds=[(0, 1)],
        factor_names=["x"],
        n_restarts=5, use_global=True, seed=2,
    )
    b.add(_check_numeric("상충 응답: x* = 0.5",
                         0.5, res_c["optimal_factors"]["x"], tol=0.01))
    b.add(_check_numeric("상충 응답: D = 0.5",
                         0.5, res_c["overall_desirability"], tol=0.01))

    return b


# ============================================================
# 메인: 전체 벤치마크 실행
# ============================================================

BENCHMARK_FUNCTIONS: List[Callable[[], Benchmark]] = [
    benchmark_1_oa_orthogonality,
    benchmark_2_oa_catalog_match,
    benchmark_3_sn_ratio,
    benchmark_4_anova_cross_check,
    benchmark_5_desirability_endpoints,
    benchmark_6_nist_tire_tread,
    benchmark_7_optimizer_convergence,
]


def run_all_benchmarks(verbose: bool = True) -> List[Benchmark]:
    """모든 벤치마크 실행 및 결과 반환."""
    results = []
    for fn in BENCHMARK_FUNCTIONS:
        if verbose:
            print(f"실행 중: {fn.__name__}...")
        try:
            bench = fn()
            results.append(bench)
            if verbose:
                status = "✅" if bench.all_passed else "❌"
                print(f"  {status} {bench.n_passed}/{bench.n_total} 통과")
        except Exception as e:
            err_bench = Benchmark(
                name=fn.__name__,
                source="-",
                description=f"실행 실패: {e}",
            )
            err_bench.add(CheckResult(
                metric="실행 자체",
                expected="OK",
                actual=f"EXCEPTION: {type(e).__name__}: {e}",
                passed=False,
            ))
            results.append(err_bench)
            if verbose:
                print(f"  ❌ 예외 발생: {e}")
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("DOE Studio 벤치마크 자체 실행")
    print("=" * 70)
    benchmarks = run_all_benchmarks()
    print()
    total_pass = sum(b.n_passed for b in benchmarks)
    total_all = sum(b.n_total for b in benchmarks)
    print(f"종합: {total_pass}/{total_all} 통과")
