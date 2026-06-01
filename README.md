# 🔬 DOE Studio

> 사내 DOE(시험계획법) 자동화 도구. Taguchi 직교배열표와 Derringer-Suich
> 반응최적화를 핵심으로 하며, 기존 Minitab 워크플로우를 Python 웹 UI로 전환.

---

## 1. 빠른 시작 (신입사원용)

### 1.1 사전 준비
- Python 3.10 이상
- Git
- 사내 슬랙 `#doe-studio` 채널 가입 (질문/버그 제보용)

### 1.2 설치 (5분)

```bash
# 1) 프로젝트 받기
git clone <internal-git-url>/doe_studio.git
cd doe_studio

# 2) 가상환경 (권장)
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3) 패키지 설치
pip install -r requirements.txt

# 4) 실행
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 `http://localhost:8501` 접속.

### 1.3 첫 사용 (10분)

1. 왼쪽 메뉴 → **📘 사용법** : 개념 설명을 빠르게 훑어보기
2. 왼쪽 메뉴 → **📁 예제 데이터** : "시트벨트 PT/LL (L9)" 선택 후 "이 예제로 분석 시작" 클릭
3. 왼쪽 메뉴 → **📊 Taguchi 분석** : 자동 로드된 데이터로 응답표, S/N, ANOVA 확인
4. 왼쪽 메뉴 → **🎯 반응최적화** : Desirability로 최적 조건 자동 탐색

---

## 2. 프로젝트 구조

```
doe_studio/
├── app.py                          # 메인 진입점 (시작 페이지)
├── requirements.txt
├── README.md
├── QUICKSTART.md                   # 신입사원용 1장 빠른 시작
├── core/                           # 비즈니스 로직 (UI와 독립)
│   ├── __init__.py
│   ├── taguchi.py                  # OA 테이블 (L4~L27), S/N 비, 설계 생성
│   ├── analysis.py                 # 응답표, ANOVA, 최적 수준 예측
│   ├── desirability.py             # Derringer-Suich + 최적화 엔진
│   └── examples.py                 # 학습용 예제 데이터셋
├── pages/                          # Streamlit multi-page
│   ├── 1_📘_사용법.py               # 신입사원 튜토리얼
│   ├── 2_🔬_Taguchi_설계.py
│   ├── 3_📊_Taguchi_분석.py
│   ├── 4_🎯_반응최적화.py
│   ├── 5_📁_예제_데이터.py
│   └── 6_🛡️_검증.py                 # 외부 출처 일치성 검증
├── verification/                   # 외부 출처 검증 모듈
│   ├── __init__.py
│   ├── benchmarks.py               # 7가지 벤치마크 (60+ 검증 항목)
│   ├── run_verification.py         # CLI 실행기 + 보고서 생성
│   ├── README.md                   # 검증 방법론 + 사용법
│   └── reports/                    # 검증 결과 보고서 저장
├── tests/                          # 단위 테스트
│   └── run_self_test.py
└── data/                           # (선택) 시험 이력 저장 위치
```

---

## 3. 주요 기능

### 3.1 Taguchi 직교배열표 (OA)
지원되는 표준 OA:

| OA | Runs | 수준 구조 | 최대 인자 | 용도 |
|---|---|---|---|---|
| L4 | 4 | 2³ | 3 | 빠른 스크리닝 |
| L8 | 8 | 2⁷ | 7 | 다인자 2수준 |
| L9 | 9 | 3⁴ | 4 | 3수준 표준 (가장 흔함) |
| L12 | 12 | 2¹¹ | 11 | PB 계열 스크리닝 |
| L16 | 16 | 2¹⁵ | 15 | 정밀 2수준 |
| L18 | 18 | 2¹×3⁷ | 8 | 혼합 수준 |
| L27 | 27 | 3¹³ | 13 | 3수준 상호작용 |

- 인자/수준 정보 → OA 자동 추천
- 무작위 실행 순서 생성 (시드 기반 재현)
- CSV 다운로드 또는 웹에서 직접 결과 입력

### 3.2 S/N 비 (Signal-to-Noise Ratio)
- Larger-the-better: `η = -10·log₁₀( mean(1/y²) )`
- Smaller-the-better: `η = -10·log₁₀( mean(y²) )`
- Nominal-the-best: `η = 10·log₁₀( ȳ²/s² )` (반복 필요)

### 3.3 분석
- 수준별 평균/S/N 응답표 (Minitab Response Table for Means/S/N과 동등)
- Delta 및 Rank 자동 계산
- 주효과도 (Plotly 인터랙티브)
- ANOVA: DF, SS, MS, F, p-value, Contribution(%)
- 풀링(Pooling) 옵션 (기여율 임계값 기반)
- 최적 수준 자동 식별 및 예측치 계산

### 3.4 반응최적화 (Derringer-Suich Desirability)
- 3가지 응답 유형: Maximize, Minimize, Target
- 가중 기하평균으로 전체 D 계산
- Differential Evolution + Multi-start L-BFGS-B 최적화
- 응답별 Desirability 곡선 시각화

---

## 4. 학습용 예제

| 예제 | OA | 인자 | 응답 | 도메인 |
|---|---|---|---|---|
| 시트벨트 PT/LL | L9 | 3 (3수준) | 2 | 자동차 안전 |
| 도장 두께 | L8 | 4 (2수준) | 2 | 표면처리 |
| 사출 성형 | L18 | 8 (혼합) | 1 | 플라스틱 가공 |

각 예제는 **📁 예제 데이터** 페이지에서 한 번에 로드 가능.

---

## 5. Minitab과의 차이점

| 항목 | Minitab | DOE Studio |
|---|---|---|
| 라이선스 | 유료 (연 약 200만원/인) | 무료 (오픈소스) |
| OA 카탈로그 | L4 ~ L36, L50, L54+ | L4 ~ L27 (v0.1) |
| S/N 비 정의 | Phadke 표준 | **동일** (Phadke 1989) |
| ANOVA 알고리즘 | Type I/II/III SS | Type I (Sequential) |
| 풀링 | 수동 | 임계값 기반 자동 |
| Desirability | Derringer-Suich | **동일** |
| 한글/팀 컨텍스트 | 영문 위주 | 한글 UI + 사내 예제 |
| 사용자 인증 | 별도 라이선스 | (선택) LDAP/SSO |
| **자동 일치성 검증** | ❌ | ✅ 60+ 벤치마크 (NIST/Minitab 출처) |

⚠️ **공식 보고서가 필요한 분석** (GMP, ISO 등) **은 Minitab 결과를 정본으로 유지**하세요.

## 6. 외부 출처와의 일치성 검증

DOE Studio는 자체적으로 7가지 벤치마크 / 60개 이상의 개별 검증 항목으로
Minitab/NIST 등 외부 권위 출처와의 일치성을 자동 검증할 수 있습니다.

```bash
# 검증 실행 (CLI)
python verification/run_verification.py

# 또는 웹 UI에서 → 사이드바 [🛡️ 검증] 페이지
```

핵심 벤치마크:
- **OA 직교성** (Phadke 1989)
- **표준 OA 카탈로그 일치성** (Minitab / NIST)
- **ANOVA 교차 검증** (statsmodels 독립 구현)
- **NIST Derringer-Suich 타이어 트레드 E2E 재현** (1980년 원논문 데이터)

자세한 사용법은 `verification/README.md` 참조.

---

## 7. 자체 테스트 (개발자용)

각 모듈은 단독 실행 가능합니다.

```bash
# Taguchi 모듈 테스트
python -m core.taguchi

# 분석 모듈 테스트
python core/analysis.py

# Desirability 모듈 테스트
python core/desirability.py

# 예제 데이터 모듈 테스트
python core/examples.py
```

---

## 8. 로드맵

### v0.1 (현재)
- ✅ Taguchi L4~L27 OA
- ✅ S/N, ANOVA, 주효과
- ✅ Derringer-Suich Desirability
- ✅ 학습용 예제 3종
- ✅ 외부 출처 일치성 자동 검증 (60+ 벤치마크)

### v0.2 (계획)
- [ ] Box-Behnken, CCD (RSM)
- [ ] 2-way 상호작용 분석 (Linear Graph)
- [ ] 잔차분석 4종 (정규확률, 잔차vs적합값 등)
- [ ] 등고선도 / 표면도 (3D)
- [ ] PDF/PPT 자동 리포트 생성

### v0.3 (계획)
- [ ] 시험 이력 DB (PostgreSQL)
- [ ] 사용자 인증 (LDAP/SSO)
- [ ] 사내 표준 템플릿 라이브러리
- [ ] **사내 기존 Minitab 보고서 3건** 자동 회귀 테스트 추가 (v0.1 검증 모듈 확장)

---

## 9. 문의 / 기여

- 사내 슬랙: `#doe-studio`
- 이슈 제보: 사내 Git 저장소 Issue 페이지
- 기능 제안: 매월 첫째 주 월요일 정기 회의

## 10. 참고문헌

- Phadke, M.S. (1989). *Quality Engineering Using Robust Design.* Prentice Hall.
- Derringer, G. and Suich, R. (1980). "Simultaneous Optimization of Several Response Variables." *Journal of Quality Technology*, 12(4), 214-219.
- Roy, R.K. (2001). *Design of Experiments Using the Taguchi Approach.* Wiley.
- NIST/SEMATECH e-Handbook of Statistical Methods. https://itl.nist.gov/div898/handbook/
- Valdano, M. et al. (2023). "Characterisation of the features of seat-belt systems based on the analysis of large crash databases." IRCOBI Conference.
