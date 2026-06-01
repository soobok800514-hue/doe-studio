"""
검증 실행기 및 보고서 생성
=========================
모든 벤치마크를 실행하고 결과를 마크다운/CSV 보고서로 출력.

사용법:
    python verification/run_verification.py
    python verification/run_verification.py --output reports/2026-XX-XX.md
    python verification/run_verification.py --csv reports/2026-XX-XX.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.benchmarks import run_all_benchmarks, Benchmark, CheckResult


def fmt_value(v) -> str:
    """값을 보기 좋게 포매팅."""
    if isinstance(v, (int, bool)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if abs(v) < 1e-4 or abs(v) > 1e6:
            return f"{v:.4e}"
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)):
        return str(v)
    return str(v)


def generate_markdown_report(benchmarks: list[Benchmark]) -> str:
    """검증 결과를 마크다운 보고서로 변환."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_pass = sum(b.n_passed for b in benchmarks)
    total_all = sum(b.n_total for b in benchmarks)
    pass_rate = 100 * total_pass / total_all if total_all > 0 else 0
    overall_status = "✅ PASS" if total_pass == total_all else "❌ FAIL"

    lines = []
    lines.append("# DOE Studio 검증 보고서")
    lines.append("")
    lines.append(f"- **생성일시**: {now}")
    lines.append(f"- **DOE Studio 버전**: v0.1.0")
    lines.append(f"- **종합 결과**: {overall_status}  ({total_pass} / {total_all}, {pass_rate:.1f}%)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 종합 요약 표
    lines.append("## 1. 종합 요약")
    lines.append("")
    lines.append("| # | 벤치마크 | 출처 | 통과/전체 | 상태 |")
    lines.append("|---|---|---|---|---|")
    for i, b in enumerate(benchmarks, 1):
        status = "✅" if b.all_passed else "❌"
        lines.append(f"| {i} | {b.name} | {b.source} | {b.n_passed} / {b.n_total} | {status} |")
    lines.append("")

    # 검증 방법론
    lines.append("## 2. 검증 방법론")
    lines.append("")
    lines.append("본 검증은 다음 4단계 원칙으로 수행됩니다.")
    lines.append("")
    lines.append("1. **수학적 정의 검증** : 수기 계산값과 코드 출력의 직접 비교 (S/N 비, Desirability 등)")
    lines.append("2. **카탈로그 일치성 검증** : Minitab, NIST에 발표된 표준 직교배열표(OA)와 매트릭스 비교")
    lines.append("3. **독립 구현과의 교차 검증** : statsmodels OLS의 ANOVA와 우리 구현의 SS, F, p-value 비교")
    lines.append("4. **End-to-End 재현 검증** : 발표 논문의 결과(예: NIST Derringer-Suich 타이어 트레드)를 동일 입력으로 재현")
    lines.append("")
    lines.append("**허용 오차 (Tolerance)**:")
    lines.append("- 수학적 정의 검증: `1e-8 ~ 1e-9` (부동소수점 한계)")
    lines.append("- 교차 검증: `1e-6` (라이브러리 간 미세 차이)")
    lines.append("- E2E 재현: `0.02 ~ 0.5` (발표 논문의 반올림 표기 감안)")
    lines.append("")

    # 벤치마크별 상세
    lines.append("## 3. 벤치마크 상세")
    lines.append("")

    for i, bench in enumerate(benchmarks, 1):
        status = "✅ PASS" if bench.all_passed else "❌ FAIL"
        lines.append(f"### 3.{i} {bench.name} — {status}")
        lines.append("")
        lines.append(f"- **출처**: {bench.source}")
        lines.append(f"- **설명**: {bench.description}")
        lines.append(f"- **결과**: {bench.n_passed} / {bench.n_total} 통과")
        lines.append("")

        if bench.checks:
            lines.append("| # | 검증 항목 | 기대값 | 실제값 | 절대오차 | 허용오차 | 상태 |")
            lines.append("|---|---|---|---|---|---|---|")
            for j, c in enumerate(bench.checks, 1):
                check_status = "✅" if c.passed else "❌"
                diff_str = f"{c.abs_diff:.2e}" if not (isinstance(c.abs_diff, float) and (c.abs_diff != c.abs_diff)) else "-"
                tol_str = f"{c.tolerance:.0e}" if c.tolerance > 0 else "exact"
                exp_str = fmt_value(c.expected)
                act_str = fmt_value(c.actual)
                # 너무 긴 표현은 단축
                if len(exp_str) > 40:
                    exp_str = exp_str[:37] + "..."
                if len(act_str) > 40:
                    act_str = act_str[:37] + "..."
                lines.append(
                    f"| {j} | {c.metric} | `{exp_str}` | `{act_str}` | {diff_str} | {tol_str} | {check_status} |"
                )
            lines.append("")

            # 비고 표시
            notes = [c for c in bench.checks if c.note]
            if notes:
                lines.append("**비고:**")
                for c in notes[:5]:  # 최대 5개
                    lines.append(f"- {c.metric}: {c.note}")
                lines.append("")

        lines.append("---")
        lines.append("")

    # 출처 목록
    lines.append("## 4. 참고 문헌")
    lines.append("")
    lines.append("- **NIST/SEMATECH e-Handbook of Statistical Methods**, §5.5.3.2.2 \"Multiple responses: The desirability approach\". https://itl.nist.gov/div898/handbook/")
    lines.append("- **Derringer, G. and Suich, R.** (1980). \"Simultaneous Optimization of Several Response Variables.\" *Journal of Quality Technology*, 12(4), 214–219.")
    lines.append("- **Phadke, M.S.** (1989). *Quality Engineering Using Robust Design.* Prentice Hall.")
    lines.append("- **Minitab Catalogue of Taguchi Designs**. https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/doe/supporting-topics/taguchi-designs/catalogue-of-taguchi-designs/")
    lines.append("- **statsmodels** v0.14+ — OLS, anova_lm (독립 구현으로 사용)")
    lines.append("")

    # 사용 안내
    lines.append("## 5. 결과 해석 가이드")
    lines.append("")
    lines.append("- ✅ **모든 벤치마크 통과**: DOE Studio의 핵심 계산이 Minitab/NIST 기준과 일치함을 의미. 사내 분석에 자신 있게 사용 가능.")
    lines.append("- ⚠️ **일부 벤치마크 실패**: 실패한 항목의 `절대오차` 와 `허용오차` 를 비교. 라이브러리 업데이트나 코드 수정으로 인한 회귀 가능성 검토.")
    lines.append("- 📋 **사용자 자체 검증 권장**: 본 도구의 결과가 미덥지 않을 경우, 동일 데이터를 Minitab에 입력하여 결과 일치성을 확인하세요.")
    lines.append("")

    return "\n".join(lines)


def generate_csv_report(benchmarks: list[Benchmark]) -> list[list]:
    """CSV 형식 (감사 추적용)."""
    rows = [["Benchmark", "Source", "Metric", "Expected", "Actual",
             "AbsDiff", "RelDiff", "Tolerance", "Passed", "Note"]]
    for b in benchmarks:
        for c in b.checks:
            rows.append([
                b.name, b.source, c.metric,
                fmt_value(c.expected), fmt_value(c.actual),
                c.abs_diff, c.rel_diff,
                c.tolerance, c.passed, c.note,
            ])
    return rows


def main():
    parser = argparse.ArgumentParser(description="DOE Studio 검증 실행")
    parser.add_argument("--output", "-o", type=str,
                        default=None, help="마크다운 보고서 저장 경로")
    parser.add_argument("--csv", type=str,
                        default=None, help="CSV 보고서 저장 경로")
    parser.add_argument("--quiet", "-q", action="store_true", help="콘솔 출력 최소화")
    args = parser.parse_args()

    if not args.quiet:
        print("=" * 70)
        print("DOE Studio 검증 실행")
        print("=" * 70)

    benchmarks = run_all_benchmarks(verbose=not args.quiet)

    total_pass = sum(b.n_passed for b in benchmarks)
    total_all = sum(b.n_total for b in benchmarks)

    if not args.quiet:
        print()
        print("=" * 70)
        print(f"종합: {total_pass} / {total_all} 통과")
        print("=" * 70)

    # 기본 출력 경로
    if args.output is None:
        report_dir = Path(__file__).resolve().parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.output = str(report_dir / f"verification_{date_str}.md")

    # 마크다운 보고서 저장
    md = generate_markdown_report(benchmarks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    if not args.quiet:
        print(f"\n📄 마크다운 보고서: {args.output}")

    # CSV 보고서 저장
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(generate_csv_report(benchmarks))
        if not args.quiet:
            print(f"📊 CSV 보고서: {args.csv}")

    # 종료 코드
    sys.exit(0 if total_pass == total_all else 1)


if __name__ == "__main__":
    main()
