"""
DOE Studio 통합 자체 테스트
==========================
모든 핵심 모듈을 한 번에 검증합니다.
배포 전 또는 코드 수정 후 실행:

    python tests/run_self_test.py
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PASS = "✅"
FAIL = "❌"
fail_count = 0


def check(name, condition, msg=""):
    global fail_count
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}: {msg}")
        fail_count += 1


print("=" * 60)
print("DOE Studio 통합 자체 테스트")
print("=" * 60)

# ------------------------------------------------------------
# 1. Taguchi 모듈
# ------------------------------------------------------------
print("\n[1] Taguchi 모듈")
from core.taguchi import OA_CATALOG, recommend_oa, build_design, sn_ratio

check("L4 ~ L27 모든 OA 로드", len(OA_CATALOG) == 7)
check("L9 shape 정확", OA_CATALOG["L9"]["array"].shape == (9, 4))
check("L18 shape 정확", OA_CATALOG["L18"]["array"].shape == (18, 8))
check("L27 shape 정확", OA_CATALOG["L27"]["array"].shape == (27, 13))

# OA 추천
check("OA 추천 (3인자 2수준 → L4)", recommend_oa(3, [2, 2, 2]) == "L4")
check("OA 추천 (4인자 3수준 → L9)", recommend_oa(4, [3, 3, 3, 3]) == "L9")
check("OA 추천 (혼합 → L18)", recommend_oa(2, [2, 3]) == "L18")

# S/N 비
y = np.array([3.5, 3.7, 3.6])
sn_l = sn_ratio(y, "larger")
sn_s = sn_ratio(y, "smaller")
sn_n = sn_ratio(y, "nominal")
check("S/N larger 양수", sn_l > 0)
check("S/N smaller 음수", sn_s < 0)
check("S/N nominal 양수", sn_n > 0)

# 설계 생성
factors = {
    "A": {"levels": [10, 20, 30], "unit": "mm"},
    "B": {"levels": [1, 2, 3], "unit": "-"},
}
df = build_design("L9", factors, randomize=True, seed=42)
check("L9 설계 9행 생성", len(df) == 9)
check("A 컬럼이 3수준", df["A"].nunique() == 3)

# ------------------------------------------------------------
# 2. 분석 모듈
# ------------------------------------------------------------
print("\n[2] 분석 모듈")
from core.analysis import response_table, anova_taguchi, find_optimal_levels, predict_optimum

factors3 = {
    "A": {"levels": [10, 20, 30], "unit": "mm"},
    "B": {"levels": [1, 2, 3], "unit": "-"},
    "C": {"levels": [100, 200, 300], "unit": "MPa"},
}
df3 = build_design("L9", factors3, randomize=False)
rng = np.random.default_rng(0)
df3["Y"] = (50 + 2.0 * (df3["A"] - 20) / 10 + 0.5 * (df3["B"] - 2)
            + rng.normal(0, 0.5, len(df3)))

# 응답표
tbl = response_table(df3, ["A", "B", "C"], "Y", statistic="mean")
check("응답표에 Level/Delta/Rank 행", "Delta" in tbl.index and "Rank" in tbl.index)
check("A가 가장 큰 Rank", int(tbl.loc["Rank", "A"]) == 1)

# ANOVA
anova = anova_taguchi(df3, ["A", "B", "C"], "Y")
check("ANOVA 행 = 인자 3 + Error + Total", len(anova) == 5)
check("ANOVA에 p-value 존재", "P" in anova.columns)

# 최적 수준
opt = find_optimal_levels(tbl, ["A", "B", "C"], direction="max")
check("최적 수준 dict 생성", len(opt) == 3)

# 예측
y_pred = predict_optimum(df3, ["A", "B", "C"], "Y", opt)
check("예측치 NaN 아님", not np.isnan(y_pred))

# ------------------------------------------------------------
# 3. Desirability 모듈
# ------------------------------------------------------------
print("\n[3] Desirability 모듈")
from core.desirability import (
    d_larger, d_smaller, d_target, ResponseSpec,
    overall_desirability, optimize_desirability,
)

check("d_larger(y<L) = 0", float(d_larger(np.array([5.0]), L=10, T=20)[0]) == 0.0)
check("d_larger(y>T) = 1", float(d_larger(np.array([25.0]), L=10, T=20)[0]) == 1.0)
check("d_smaller(y<T) = 1", float(d_smaller(np.array([5.0]), T=10, U=20)[0]) == 1.0)
check("d_target 양 끝 = 0",
      float(d_target(np.array([5.0]), L=10, T=15, U=20)[0]) == 0.0)

# Overall D
D = overall_desirability([0.8, 0.6, 0.9])
check("Overall D 0~1 범위", 0 < D < 1)
check("한 응답=0이면 D=0", overall_desirability([0.8, 0.0, 0.9]) == 0.0)

# 최적화
specs = [
    ResponseSpec("Y1", goal="max", L=30, T=50),
    ResponseSpec("Y2", goal="min", T=4, U=6),
]
def f1(x): return 25 + 0.5 * (15 - x[0]) + 2.0 * (x[1] - 3)
def f2(x): return 6.0 - 0.15 * (15 - x[0]) + 0.3 * (x[1] - 3)

res = optimize_desirability(
    {"Y1": f1, "Y2": f2}, specs,
    bounds=[(5, 15), (3, 5)], factor_names=["x1", "x2"],
    n_restarts=5, use_global=True, seed=42,
)
check("최적화 OK 수렴", res["convergence"] == "OK")
check("최적 인자 dict 반환", res["optimal_factors"] is not None)
check("D > 0", res["overall_desirability"] > 0)

# ------------------------------------------------------------
# 4. 예제 데이터 모듈
# ------------------------------------------------------------
print("\n[4] 예제 데이터 모듈")
from core.examples import EXAMPLES, build_example_dataframe

check("예제 3개 등록", len(EXAMPLES) == 3)
for name in EXAMPLES:
    ex = EXAMPLES[name]
    df_ex = build_example_dataframe(ex)
    check(f"  {name}: 데이터 생성 OK", len(df_ex) > 0)
    # 응답값이 채워져 있는지
    for resp_name in ex["responses"]:
        check(f"  {name}: 응답 '{resp_name}' 비어있지 않음",
              df_ex[resp_name].notna().sum() > 0)

# ------------------------------------------------------------
# 5. End-to-End 시나리오
# ------------------------------------------------------------
print("\n[5] End-to-End 시나리오 (시트벨트 예제)")
ex = EXAMPLES["시트벨트 PT/LL (L9)"]
df_e2e = build_example_dataframe(ex)
factor_cols = list(ex["factors"].keys())

# Step 1: ANOVA
anova_e2e = anova_taguchi(df_e2e, factor_cols, "MaxLoad")
check("E2E ANOVA 정상", len(anova_e2e) > 0)

# Step 2: 최적 수준
tbl_e2e = response_table(df_e2e, factor_cols, "MaxLoad", statistic="mean")
opt_e2e = find_optimal_levels(tbl_e2e, factor_cols, direction="max")
check("E2E 최적 수준 도출", len(opt_e2e) == 3)

# Step 3: Desirability (모의 예측 함수 사용)
specs_e2e = [
    ResponseSpec("MaxLoad", goal="target",
                 L=ex["responses"]["MaxLoad"]["L"],
                 T=ex["responses"]["MaxLoad"]["T"],
                 U=ex["responses"]["MaxLoad"]["U"]),
    ResponseSpec("ChestDisp", goal="min",
                 T=ex["responses"]["ChestDisp"]["T"],
                 U=ex["responses"]["ChestDisp"]["U"]),
]
def pred1(x): return float(df_e2e["MaxLoad"].mean())
def pred2(x): return float(df_e2e["ChestDisp"].mean())
bounds_e2e = [(df_e2e[c].min(), df_e2e[c].max()) for c in factor_cols]

res_e2e = optimize_desirability(
    {"MaxLoad": pred1, "ChestDisp": pred2}, specs_e2e,
    bounds=bounds_e2e, factor_names=factor_cols, n_restarts=3,
)
check("E2E 최적화 수렴", res_e2e["convergence"] == "OK")
check("E2E D > 0.5", res_e2e["overall_desirability"] > 0.5)

# ------------------------------------------------------------
# 결과 요약
# ------------------------------------------------------------
print("\n" + "=" * 60)
if fail_count == 0:
    print("🎉 모든 테스트 통과")
    sys.exit(0)
else:
    print(f"⚠️  {fail_count}개 테스트 실패")
    sys.exit(1)
