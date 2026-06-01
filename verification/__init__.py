"""
DOE Studio 검증 모듈
====================
Minitab/NIST 등 외부 기준에 대한 일치성 자동 검증.
"""
from .benchmarks import (
    CheckResult, Benchmark,
    run_all_benchmarks, BENCHMARK_FUNCTIONS,
)

__all__ = [
    "CheckResult", "Benchmark",
    "run_all_benchmarks", "BENCHMARK_FUNCTIONS",
]
