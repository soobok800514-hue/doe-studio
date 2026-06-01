"""
Derringer-Suich Desirability Function 모듈
==========================================
다중 반응 최적화 (Multi-Response Optimization).

참고문헌:
    Derringer, G. and Suich, R. (1980).
    "Simultaneous Optimization of Several Response Variables."
    Journal of Quality Technology, 12(4), 214-219.

    NIST/SEMATECH e-Handbook of Statistical Methods (Section 5.5.3.2.2).

개별 desirability d_i (y_i): 0 ~ 1 사이 값
- 1 : 완전히 바람직함
- 0 : 완전히 바람직하지 않음

전체 desirability D = (d_1 * d_2 * ... * d_k)^(1/k)   (기하평균)

3가지 목표 유형:
1) Larger-the-better (최대화): 클수록 좋음
   d = 0                          if y < L
   d = ((y - L) / (T - L))^s      if L <= y <= T
   d = 1                          if y > T

2) Smaller-the-better (최소화): 작을수록 좋음
   d = 1                          if y < T
   d = ((U - y) / (U - T))^s      if T <= y <= U
   d = 0                          if y > U

3) Target (목표값): 특정 값에 가까울수록 좋음
   d = 0                              if y < L or y > U
   d = ((y - L) / (T - L))^s          if L <= y <= T
   d = ((U - y) / (U - T))^t          if T <= y <= U

여기서:
   L = Lower bound (하한)
   T = Target (목표값)
   U = Upper bound (상한)
   s, t = shape parameter (보통 1; 1보다 크면 목표 근처에서만 d 상승)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple, Optional
from scipy.optimize import minimize, differential_evolution


# ============================================================
# 1) 개별 desirability 함수
# ============================================================

def d_larger(y: np.ndarray, L: float, T: float, s: float = 1.0) -> np.ndarray:
    """
    Larger-the-better desirability.

    Parameters
    ----------
    y : array-like or scalar
        예측 응답값
    L : float
        하한 (이보다 작으면 d=0)
    T : float
        목표값 또는 이상값 (이보다 크면 d=1)
    s : float
        형상 모수 (default 1.0 = 선형)
        s > 1 : 목표 근처에서만 d 상승 (보수적)
        s < 1 : 낮은 값에서도 d 상승 (관대)

    Returns
    -------
    d : array-like, 0 ~ 1
    """
    y = np.asarray(y, dtype=float)
    d = np.zeros_like(y)

    mask_mid = (y >= L) & (y <= T)
    mask_high = y > T

    if T > L:
        d[mask_mid] = ((y[mask_mid] - L) / (T - L)) ** s
    d[mask_high] = 1.0

    return d if d.ndim > 0 else float(d)


def d_smaller(y: np.ndarray, T: float, U: float, s: float = 1.0) -> np.ndarray:
    """
    Smaller-the-better desirability.

    Parameters
    ----------
    T : float
        목표값 또는 이상값 (이보다 작으면 d=1)
    U : float
        상한 (이보다 크면 d=0)
    """
    y = np.asarray(y, dtype=float)
    d = np.zeros_like(y)

    mask_low = y < T
    mask_mid = (y >= T) & (y <= U)

    d[mask_low] = 1.0
    if U > T:
        d[mask_mid] = ((U - y[mask_mid]) / (U - T)) ** s

    return d if d.ndim > 0 else float(d)


def d_target(y: np.ndarray, L: float, T: float, U: float,
             s: float = 1.0, t: float = 1.0) -> np.ndarray:
    """
    Target (nominal-the-best) desirability.

    Parameters
    ----------
    L : float
        하한
    T : float
        목표값
    U : float
        상한
    s : float
        L-T 구간의 형상 모수
    t : float
        T-U 구간의 형상 모수
    """
    y = np.asarray(y, dtype=float)
    d = np.zeros_like(y)

    mask_low = (y >= L) & (y < T)
    mask_high = (y >= T) & (y <= U)

    if T > L:
        d[mask_low] = ((y[mask_low] - L) / (T - L)) ** s
    if U > T:
        d[mask_high] = ((U - y[mask_high]) / (U - T)) ** t

    return d if d.ndim > 0 else float(d)


# ============================================================
# 2) 응답 사양 (Response Specification)
# ============================================================

class ResponseSpec:
    """
    응답변수의 사양 정의 클래스.

    예시
    ----
    >>> # 최대화: 강도는 클수록 좋고, 적어도 30 이상이어야 함, 50이면 이상적
    >>> spec_strength = ResponseSpec("Strength", goal="max", L=30, T=50)
    >>>
    >>> # 최소화: 소음은 작을수록 좋고, 30 이하면 이상적, 60 초과는 불량
    >>> spec_noise = ResponseSpec("Noise", goal="min", T=30, U=60)
    >>>
    >>> # 목표: 두께는 5.0이 목표, 4.5~5.5 범위
    >>> spec_thickness = ResponseSpec("Thickness", goal="target", L=4.5, T=5.0, U=5.5)
    """
    def __init__(self, name: str, goal: str, L: float = None,
                 T: float = None, U: float = None,
                 s: float = 1.0, t: float = 1.0, weight: float = 1.0):
        self.name = name
        self.goal = goal
        self.L = L
        self.T = T
        self.U = U
        self.s = s
        self.t = t
        self.weight = weight  # 응답별 가중치 (기하평균 시 사용)

        # 검증
        if goal == "max":
            if L is None or T is None:
                raise ValueError(f"goal='max'는 L과 T 필요")
        elif goal == "min":
            if T is None or U is None:
                raise ValueError(f"goal='min'은 T와 U 필요")
        elif goal == "target":
            if L is None or T is None or U is None:
                raise ValueError(f"goal='target'은 L, T, U 모두 필요")
            if not (L <= T <= U):
                raise ValueError(f"L <= T <= U 이어야 함")
        else:
            raise ValueError(f"goal은 'max', 'min', 'target' 중 하나. 입력: {goal}")

    def desirability(self, y) -> np.ndarray:
        """예측값 y에 대한 desirability 계산."""
        if self.goal == "max":
            return d_larger(y, self.L, self.T, self.s)
        elif self.goal == "min":
            return d_smaller(y, self.T, self.U, self.s)
        elif self.goal == "target":
            return d_target(y, self.L, self.T, self.U, self.s, self.t)


# ============================================================
# 3) 전체 Desirability (기하평균)
# ============================================================

def overall_desirability(d_values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    가중 기하평균으로 전체 desirability 계산.

    D = (d_1^w1 * d_2^w2 * ... * d_k^wk)^(1 / sum(wi))

    어떤 d_i가 0이면 D = 0 (한 응답이라도 완전 불량이면 전체 불량)

    Parameters
    ----------
    d_values : list of float
        각 응답의 개별 desirability
    weights : list of float, optional
        각 응답의 가중치 (default: 모두 1)
    """
    d_arr = np.asarray(d_values, dtype=float)
    if weights is None:
        weights = np.ones(len(d_arr))
    else:
        weights = np.asarray(weights, dtype=float)

    if np.any(d_arr <= 0):
        return 0.0

    log_d = np.sum(weights * np.log(d_arr))
    D = np.exp(log_d / np.sum(weights))
    return float(D)


# ============================================================
# 4) 최적화 엔진
# ============================================================

def optimize_desirability(
    predict_funcs: Dict[str, Callable],
    specs: List[ResponseSpec],
    bounds: List[Tuple[float, float]],
    factor_names: List[str],
    n_restarts: int = 10,
    use_global: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Desirability 최대화로 최적 인자 조건 탐색.

    Parameters
    ----------
    predict_funcs : dict
        {응답명: 예측함수}. 예측함수는 인자 값 배열을 받아 예측 응답값 반환.
        예: predict_funcs["Strength"](X) -> float
        X는 길이 n_factors의 1D 배열.
    specs : list of ResponseSpec
        각 응답의 사양
    bounds : list of (lo, hi) tuples
        각 인자의 (하한, 상한) - 코드값 또는 실제값
    factor_names : list of str
        인자명 (결과 dict 작성용)
    n_restarts : int
        다중 시작점 개수 (지역최적해 회피)
    use_global : bool
        True면 differential_evolution + local refinement,
        False면 multi-start local만 사용
    seed : int
        무작위 시드

    Returns
    -------
    dict :
        - "optimal_factors": {인자명: 값}
        - "predicted_responses": {응답명: 예측값}
        - "individual_desirabilities": {응답명: d_i}
        - "overall_desirability": D
        - "convergence": 수렴 정보
    """

    response_names = [s.name for s in specs]
    weights = [s.weight for s in specs]

    def neg_D(x):
        """최소화 대상 = -D"""
        try:
            d_vals = []
            for spec in specs:
                y_pred = predict_funcs[spec.name](x)
                d_vals.append(float(spec.desirability(y_pred)))
            D = overall_desirability(d_vals, weights)
            return -D
        except Exception:
            return 0.0  # 실패 시 D=0 (즉 -D=0)

    best_x = None
    best_neg_D = 0.0  # D >= 0 이므로 -D <= 0; 좋을수록 작아짐

    rng = np.random.default_rng(seed)

    if use_global:
        # 1) Differential Evolution으로 전역 탐색
        try:
            res = differential_evolution(
                neg_D, bounds, seed=seed,
                maxiter=200, tol=1e-7, popsize=15,
                workers=1,
            )
            if res.fun < best_neg_D:
                best_neg_D = res.fun
                best_x = res.x
        except Exception:
            pass

    # 2) Multi-start local refinement
    for i in range(n_restarts):
        # 임의 시작점
        x0 = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds])
        try:
            res = minimize(neg_D, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_neg_D:
                best_neg_D = res.fun
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        return {
            "optimal_factors": None,
            "predicted_responses": None,
            "individual_desirabilities": None,
            "overall_desirability": 0.0,
            "convergence": "FAILED",
        }

    # 최적해에서 예측값 계산
    predicted = {}
    individual_d = {}
    for spec in specs:
        y_pred = float(predict_funcs[spec.name](best_x))
        predicted[spec.name] = y_pred
        individual_d[spec.name] = float(spec.desirability(y_pred))

    D_best = overall_desirability(list(individual_d.values()), weights)

    return {
        "optimal_factors": dict(zip(factor_names, best_x.tolist())),
        "predicted_responses": predicted,
        "individual_desirabilities": individual_d,
        "overall_desirability": D_best,
        "convergence": "OK",
    }


# ============================================================
# 5) OLS 기반 자동 예측함수 생성기 (편의 함수)
# ============================================================

def build_predict_func_from_ols(model, factor_cols: List[str],
                                 include_interactions: bool = False):
    """
    statsmodels OLS 결과로부터 예측 함수를 생성.

    Parameters
    ----------
    model : statsmodels OLS results 객체
    factor_cols : list of str
        인자명 (순서 중요)
    include_interactions : bool
        2-way 상호작용 포함 여부

    Returns
    -------
    callable : x (numpy array) -> 예측값
    """
    def predict(x):
        x = np.asarray(x).ravel()
        # statsmodels는 DataFrame을 기대
        data = {fc: [x[i]] for i, fc in enumerate(factor_cols)}
        df_pred = pd.DataFrame(data)
        return model.predict(df_pred).values[0]
    return predict


# ============================================================
# 6) 자체 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Desirability Function 모듈 자체 테스트")
    print("=" * 60)

    # 시나리오: 시트벨트 LL 시스템 최적화
    # - 응답1: 최대 하중 (Larger) -> 30 이상, 40이면 이상적
    # - 응답2: 변위 (Smaller) -> 4.0 이하 이상적, 6.0 초과 불량
    # - 인자: PT_timing (5~15 ms), LL_threshold (3~5 kN)

    # 가상 예측 모델 (실제로는 OLS 결과 사용)
    def y1_pred(x):
        """최대 하중. PT 빠르고 LL 높을수록 증가."""
        pt, ll = x[0], x[1]
        return 25 + 0.5 * (15 - pt) + 2.0 * (ll - 3) + 0.1 * (15 - pt) * (ll - 3)

    def y2_pred(x):
        """변위. PT 빠를수록 감소, LL 낮을수록 감소."""
        pt, ll = x[0], x[1]
        return 6.0 - 0.15 * (15 - pt) + 0.3 * (ll - 3)

    specs = [
        ResponseSpec("MaxLoad", goal="max", L=30, T=40),
        ResponseSpec("Displacement", goal="min", T=4.0, U=6.0),
    ]

    bounds = [(5, 15), (3, 5)]
    factor_names = ["PT_timing_ms", "LL_threshold_kN"]
    predict_funcs = {"MaxLoad": y1_pred, "Displacement": y2_pred}

    result = optimize_desirability(predict_funcs, specs, bounds, factor_names)

    print("\n[최적 인자 조건]")
    for k, v in result["optimal_factors"].items():
        print(f"  {k}: {v:.4f}")

    print("\n[예측 응답값]")
    for k, v in result["predicted_responses"].items():
        print(f"  {k}: {v:.4f}")

    print("\n[개별 Desirability]")
    for k, v in result["individual_desirabilities"].items():
        print(f"  {k}: {v:.4f}")

    print(f"\n[전체 Desirability D]")
    print(f"  D = {result['overall_desirability']:.4f}")
    print(f"  Convergence: {result['convergence']}")
