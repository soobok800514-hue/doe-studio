# 🔬 DOE Studio 검증 모듈

> Minitab/NIST/Phadke 등 외부 권위 출처에 대한 자동 일치성 검증.
> v0.1 배포 전 필수 통과해야 하는 회귀 테스트 + 매 분기 정기 실행 권장.

---

## 1. 왜 검증이 필요한가?

사내 도구가 Minitab을 대체하려면 **결과가 동일하다는 것을 증명**해야 합니다.
다음 두 가지를 답할 수 있어야 합니다:

1. **"이 도구는 정확한가?"** → 수학적 정의·표준 카탈로그와 일치 여부
2. **"이 도구는 안정적인가?"** → 코드 변경 후에도 회귀하지 않음

본 검증 모듈은 두 질문 모두에 자동으로 답할 수 있도록 7가지 벤치마크와 60개 이상의 개별 검증 항목으로 구성되어 있습니다.

---

## 2. 빠른 사용법

```bash
# 1) 기본 실행 (콘솔에 결과 출력 + 마크다운 보고서 자동 저장)
python verification/run_verification.py

# 2) 보고서 위치 지정
python verification/run_verification.py --output report_2026Q2.md

# 3) CSV도 함께 출력 (감사 추적용)
python verification/run_verification.py --csv audit_2026Q2.csv

# 4) 조용한 모드 (CI/CD용)
python verification/run_verification.py --quiet
# 종료 코드: 0 = 모두 통과, 1 = 일부 실패
```

기본적으로 `verification/reports/verification_YYYY-MM-DD_HHMMSS.md` 에 자동 저장됩니다.

---

## 3. 벤치마크 구성

| # | 벤치마크 | 검증 출처 | 검증 항목 수 |
|---|---|---|---|
| 1 | OA 직교성 검증 | Phadke (1989) | 7 |
| 2 | 표준 OA 카탈로그 일치성 | Minitab / NIST | 10 |
| 3 | S/N 비 수기 계산 | Phadke (1989) Eq.5.2-5.4 | 5 |
| 4 | ANOVA vs statsmodels 교차 검증 | statsmodels v0.14+ | 6 |
| 5 | Desirability 함수 경계점 | NIST Handbook 5.5.3.2.2 | 14 |
| 6 | NIST Derringer-Suich 타이어 트레드 (E2E) | Derringer & Suich (1980) | 11 |
| 7 | 최적화 수렴성 (Convex 검증) | 자체 정의 | 7 |
| | **합계** | | **60** |

### 가장 중요한 벤치마크: #6 NIST 타이어 트레드

이 벤치마크는 **Derringer & Suich 원논문(1980)** 의 실제 데이터를 사용하여
End-to-End로 우리 도구가 발표값을 재현하는지 검증합니다.

검증 흐름:
1. NIST에 발표된 4개 응답 모델 (Y1, Y2, Y3, Y4)을 그대로 사용
2. NIST가 보고한 최적해 x* = (-0.10, 0.15, -1.0) 에서 예측값 검증
3. 그 점에서의 개별 d, 전체 D 검증 (NIST 발표 D = 0.596)
4. **우리 최적화기**가 NIST와 동등 이상의 해를 찾는지 확인

---

## 4. 결과 해석 가이드

### ✅ 모두 통과
DOE Studio가 표준 알고리즘과 완전히 일치함을 의미합니다. 사내 분석에 자신 있게 사용 가능.

### ❌ 일부 실패
다음 단계로 진행:

1. 실패한 항목의 **절대오차 vs 허용오차** 확인
2. `verification/benchmarks.py` 의 해당 벤치마크 코드 검토
3. 의심되는 핵심 모듈 (`core/taguchi.py`, `core/analysis.py`, `core/desirability.py`) 점검
4. 사내 슬랙 `#doe-studio` 채널에 보고

### ⚠️ Tolerance가 큰 항목
NIST 발표값과의 비교는 논문의 반올림 표기 때문에 0.02~0.5 정도의 허용오차를 사용합니다. 이는 정상입니다.

---

## 5. CI/CD 통합

GitLab CI 예시 (`.gitlab-ci.yml`):

```yaml
verify:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - python verification/run_verification.py --quiet
  artifacts:
    when: always
    paths:
      - verification/reports/
    expire_in: 1 year
```

GitHub Actions 예시 (`.github/workflows/verify.yml`):

```yaml
name: Verification
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python verification/run_verification.py --quiet
      - uses: actions/upload-artifact@v3
        if: always()
        with:
          name: verification-reports
          path: verification/reports/
```

---

## 6. 새 벤치마크 추가하는 방법

`benchmarks.py` 에 함수를 추가하고 `BENCHMARK_FUNCTIONS` 리스트에 등록하면 됩니다.

```python
def benchmark_X_my_new_check() -> Benchmark:
    b = Benchmark(
        name="X. 새로운 검증 이름",
        source="출처 명시 (필수)",
        description="무엇을 검증하는지 한 줄 설명",
    )
    # 검증 로직
    expected = 1.234
    actual = my_function()
    b.add(_check_numeric("검증 항목 이름", expected, actual, tol=1e-6))
    return b


BENCHMARK_FUNCTIONS = [
    # ... 기존 ...
    benchmark_X_my_new_check,
]
```

권장 추가 항목:
- 사내 기존 Minitab 보고서 3건의 데이터로 결과 비교 (실제 사례 검증)
- Box-Behnken / CCD 모듈 추가 시 해당 검증
- 회사 표준 시험 시나리오 (시트벨트, 도장, 사출 등) 기반 검증

---

## 7. 정기 검증 권장 주기

| 시점 | 권장 행동 |
|---|---|
| 매 코드 커밋 | CI/CD에서 자동 실행 (`--quiet` 옵션) |
| 매 PR 머지 전 | 보고서 확인 후 머지 |
| 분기 1회 (정기 감사) | 보고서 PDF로 저장 → 품질혁신팀 회람 |
| 라이브러리 메이저 업데이트 시 | 즉시 실행 (`statsmodels`, `scipy`, `numpy`) |
| 신입사원 첫 사용 전 | 한 번 실행하여 환경 정상 확인 |

---

## 8. 참고 문헌

- NIST/SEMATECH e-Handbook of Statistical Methods. §5.5.3.2.2. [URL](https://itl.nist.gov/div898/handbook/pri/section5/pri5322.htm)
- Derringer, G. and Suich, R. (1980). *Journal of Quality Technology*, 12(4), 214-219.
- Phadke, M.S. (1989). *Quality Engineering Using Robust Design.* Prentice Hall.
- Minitab Catalogue of Taguchi designs. [URL](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/doe/supporting-topics/taguchi-designs/catalogue-of-taguchi-designs/)
