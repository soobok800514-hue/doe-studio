"""
페이지/브라우저 세션이 달라져도 반응최적화 결과가 확인실험 페이지에서
항상 우선적으로 재사용되도록 디스크에 캐시한다.

Streamlit의 st.session_state는 브라우저 탭(세션) 단위로 분리되어 있어,
반응최적화 페이지에서 최적 조합을 찾은 뒤 다른 탭/세션에서 확인실험 페이지를
열면 opt_result가 비어 있어 Taguchi 분석 결과로 잘못 대체되는 문제가 있었다.
"""
import pickle
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_OPT_RESULT_PATH = _CACHE_DIR / "opt_result.pkl"


def save_opt_result(result: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OPT_RESULT_PATH, "wb") as f:
        pickle.dump(result, f)


def load_opt_result() -> dict:
    if not _OPT_RESULT_PATH.exists():
        return {}
    try:
        with open(_OPT_RESULT_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}
