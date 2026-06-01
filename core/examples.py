"""
사내 학습용 예제 데이터셋 모듈
==============================
자동차 시험 도메인의 가상 예제로 신입사원이 익숙한 맥락에서 학습할 수 있도록 함.
실제 사내 시험 데이터로 추후 교체.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


# ============================================================
# 예제 1: 시트벨트 PT/LL 시스템 최적화 (L9, 3수준 3인자)
# ============================================================

EXAMPLE_SEATBELT_L9 = {
    "name": "시트벨트 PT/LL 시스템 최적화 (L9)",
    "description": (
        "프리텐셔너 발화 시점, 로드리미터 임계값, 웨빙 강성 3가지를 변경하여 "
        "충돌 시 어깨벨트 최대 하중과 흉부 변위를 동시에 최적화하는 시나리오. "
        "실제 사내 DOE 절차를 모사한 학습용 가상 데이터입니다."
    ),
    "oa": "L9",
    "factors": {
        "PT_시점": {"levels": [5, 10, 15], "unit": "ms"},
        "LL_임계": {"levels": [3.0, 4.0, 5.0], "unit": "kN"},
        "Webbing_강성": {"levels": [800, 1000, 1200], "unit": "N/mm"},
    },
    "responses": {
        "MaxLoad": {"unit": "N", "goal": "target", "L": 3000, "T": 3500, "U": 4000,
                    "note": "흉부 손상 방지 위해 너무 크지도 작지도 않은 목표값"},
        "ChestDisp": {"unit": "mm", "goal": "min", "T": 25, "U": 40,
                       "note": "흉부 변위는 작을수록 좋음 (FMVSS 208 < 63mm)"},
    },
    # 가상 결과 데이터 (Run_Order 무관, Std_Order 기준)
    "results_by_std_order": {
        # PT, LL, Webbing이 모두 영향, PT가 가장 큰 효과
        # Std 순서: (1,1,1), (1,2,2), (1,3,3), (2,1,2), (2,2,3), (2,3,1), (3,1,3), (3,2,1), (3,3,2)
        1: {"MaxLoad": 3200, "ChestDisp": 32},
        2: {"MaxLoad": 3450, "ChestDisp": 30},
        3: {"MaxLoad": 3680, "ChestDisp": 28},
        4: {"MaxLoad": 3550, "ChestDisp": 30},
        5: {"MaxLoad": 3720, "ChestDisp": 27},
        6: {"MaxLoad": 3380, "ChestDisp": 31},
        7: {"MaxLoad": 3850, "ChestDisp": 26},
        8: {"MaxLoad": 3550, "ChestDisp": 28},
        9: {"MaxLoad": 3680, "ChestDisp": 27},
    },
}


# ============================================================
# 예제 2: 도장 공정 최적화 (L8, 2수준 4인자)
# ============================================================

EXAMPLE_PAINT_L8 = {
    "name": "도장 두께 균일성 (L8)",
    "description": (
        "분사 압력, 노즐 거리, 컨베이어 속도, 도료 온도 4인자를 2수준으로 변경하여 "
        "도막 두께의 평균을 목표치(80um)에 맞추는 시나리오."
    ),
    "oa": "L8",
    "factors": {
        "분사압력": {"levels": [2.0, 3.5], "unit": "bar"},
        "노즐거리": {"levels": [200, 300], "unit": "mm"},
        "컨베이어속도": {"levels": [0.5, 1.0], "unit": "m/s"},
        "도료온도": {"levels": [20, 35], "unit": "degC"},
    },
    "responses": {
        "두께평균": {"unit": "um", "goal": "target", "L": 70, "T": 80, "U": 90,
                     "note": "목표 80um, ±10um 허용"},
        "두께편차": {"unit": "um", "goal": "min", "T": 2.0, "U": 8.0,
                     "note": "편차는 작을수록 좋음"},
    },
    "results_by_std_order": {
        1: {"두께평균": 72, "두께편차": 6.5},
        2: {"두께평균": 78, "두께편차": 4.2},
        3: {"두께평균": 81, "두께편차": 3.5},
        4: {"두께평균": 86, "두께편차": 5.1},
        5: {"두께평균": 75, "두께편차": 5.8},
        6: {"두께평균": 84, "두께편차": 3.0},
        7: {"두께평균": 79, "두께편차": 3.8},
        8: {"두께평균": 88, "두께편차": 4.5},
    },
}


# ============================================================
# 예제 3: 사출 성형 (L18, 혼합 수준)
# ============================================================

EXAMPLE_INJECTION_L18 = {
    "name": "사출 성형 - 인장강도 (L18)",
    "description": (
        "재료(2수준) + 사출속도, 보압, 금형온도, 사출온도, 냉각시간, 보압시간, 게이트크기 "
        "(7개 3수준) 의 혼합 수준 설계로 인장강도 최대화."
    ),
    "oa": "L18",
    "factors": {
        "재료": {"levels": ["PP", "PP-GF"], "unit": "-"},
        "사출속도": {"levels": [30, 50, 70], "unit": "mm/s"},
        "보압": {"levels": [40, 60, 80], "unit": "MPa"},
        "금형온도": {"levels": [40, 60, 80], "unit": "degC"},
        "사출온도": {"levels": [200, 220, 240], "unit": "degC"},
        "냉각시간": {"levels": [10, 15, 20], "unit": "s"},
        "보압시간": {"levels": [3, 5, 7], "unit": "s"},
        "게이트크기": {"levels": [1.0, 1.5, 2.0], "unit": "mm"},
    },
    "responses": {
        "인장강도": {"unit": "MPa", "goal": "max", "L": 30, "T": 50,
                     "note": "클수록 좋음, 최소 30 이상"},
    },
    "results_by_std_order": {
        # 가상 데이터 (인자 효과 반영)
        1: {"인장강도": 32}, 2: {"인장강도": 35}, 3: {"인장강도": 36},
        4: {"인장강도": 33}, 5: {"인장강도": 37}, 6: {"인장강도": 38},
        7: {"인장강도": 34}, 8: {"인장강도": 38}, 9: {"인장강도": 39},
        10: {"인장강도": 41}, 11: {"인장강도": 45}, 12: {"인장강도": 47},
        13: {"인장강도": 42}, 14: {"인장강도": 46}, 15: {"인장강도": 48},
        16: {"인장강도": 43}, 17: {"인장강도": 47}, 18: {"인장강도": 49},
    },
}


# ============================================================
# 등록된 예제 목록
# ============================================================

EXAMPLES = {
    "시트벨트 PT/LL (L9)": EXAMPLE_SEATBELT_L9,
    "도장 두께 (L8)": EXAMPLE_PAINT_L8,
    "사출성형 (L18)": EXAMPLE_INJECTION_L18,
}


def load_example(name: str) -> Dict:
    """예제 이름으로 데이터 로드."""
    if name not in EXAMPLES:
        raise ValueError(f"예제 '{name}' 없음. 사용 가능: {list(EXAMPLES.keys())}")
    return EXAMPLES[name]


def build_example_dataframe(example: Dict) -> pd.DataFrame:
    """
    예제 dict로부터 결과가 채워진 DataFrame을 생성.
    Streamlit에서 바로 분석에 사용 가능.
    """
    try:
        from .taguchi import build_design
    except ImportError:
        from taguchi import build_design

    df = build_design(example["oa"], example["factors"],
                      randomize=False)  # 학습용은 표준 순서로
    # Response_Y 컬럼 제거 (예제는 다중 응답)
    if "Response_Y" in df.columns:
        df = df.drop(columns=["Response_Y"])

    # 응답 채우기
    for std_order, response_vals in example["results_by_std_order"].items():
        for resp_name, val in response_vals.items():
            if resp_name not in df.columns:
                df[resp_name] = np.nan
            df.loc[df["Std_Order"] == std_order, resp_name] = val

    return df


if __name__ == "__main__":
    for name in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        ex = EXAMPLES[name]
        print(f"  설명: {ex['description'][:50]}...")
        print(f"  OA: {ex['oa']}")
        print(f"  인자 수: {len(ex['factors'])}")
        print(f"  응답 수: {len(ex['responses'])}")
        df = build_example_dataframe(ex)
        print("\n  데이터 미리보기:")
        print(df.head(5).to_string(index=False))
