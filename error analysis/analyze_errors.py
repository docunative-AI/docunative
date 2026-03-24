"""
analyze_errors.py
-----------------
Error analysis for DocuNative eval JSONL (Eval 1 cross-lingual vs Eval 2 monolingual).

Run from the docunative/ directory (use project env so ``eval`` imports resolve):

    uv run python "error analysis/analyze_errors.py" --eval both
    uv run python "error analysis/analyze_errors.py" --eval 1 --language zh --export

See project plan: contingency tables, domain/field slices (Eval 1), refusal/decomposition (Eval 2).

Eval JSONL may include ``retrieved_chunks`` (top-k strings) if you re-ran
``python -m eval.evaluate ... --save-retrieval failure`` (or ``all``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# docunative/ root (parent of this folder)
_DOCUNATIVE_ROOT = Path(__file__).resolve().parent.parent
if str(_DOCUNATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOCUNATIVE_ROOT))

from eval.metrics import _is_refusal  # noqa: E402 — matches generate_charts / evaluate usage

RESULTS_DIR = _DOCUNATIVE_ROOT / "eval" / "results"

EVAL1_CANDIDATES = (
    RESULTS_DIR / "eval_results_eval1-h2.jsonl",
    RESULTS_DIR / "eval_results_h2_global.jsonl",
    RESULTS_DIR / "eval_results.jsonl",
)
EVAL2_CANDIDATES = (
    RESULTS_DIR / "eval_results_eval2-llm.jsonl",
    RESULTS_DIR / "eval_results_eval2_global.jsonl",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_eval_path(candidates: tuple[Path, ...], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(
        f"Error: no {label} JSONL found. Tried: " + ", ".join(str(x) for x in candidates)
    )


def filter_language(rows: list[dict], lang: str) -> list[dict]:
    if lang == "all":
        return rows
    return [r for r in rows if r.get("language") == lang]


def f1_band(f1: float, low: float, high: float) -> str:
    if f1 < low:
        return "low"
    if f1 <= high:
        return "mid"
    return "high"


def recall_zero(r: dict) -> bool:
    return float(r.get("recall_3") or 0) == 0.0


def contingency_counts(rows: list[dict], f1_low: float, f1_high: float) -> dict[tuple[str, str], int]:
    """Keys: (recall_bucket, f1_band) -> count."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        rz = "recall_0" if recall_zero(r) else "recall_gt0"
        band = f1_band(float(r.get("f1_score") or 0), f1_low, f1_high)
        counts[(rz, band)] += 1
    return dict(counts)


def print_contingency(title: str, rows: list[dict], f1_low: float, f1_high: float) -> None:
    c = contingency_counts(rows, f1_low, f1_high)
    bands = ["low", "mid", "high"]
    print(f"\n=== {title} (n={len(rows)}) ===")
    print(f"F1 bands: low < {f1_low}, mid {f1_low}–{f1_high}, high > {f1_high}")
    header = f"{'':16}" + "".join(f"{b:>12}" for b in bands)
    print(header)
    for rz in ["recall_0", "recall_gt0"]:
        row_cells = [str(c.get((rz, b), 0)) for b in bands]
        print(f"{rz:16}" + "".join(f"{cell:>12}" for cell in row_cells))


def eval1_domain_field_retrieval(rows: list[dict]) -> None:
    miss = [r for r in rows if recall_zero(r)]
    if not miss:
        print("\n(Eval 1) No recall_3==0 rows in this slice.")
        return
    by_domain: dict[str, list[dict]] = defaultdict(list)
    by_field: dict[str, list[dict]] = defaultdict(list)
    for r in miss:
        by_domain[r.get("domain", "?")].append(r)
        by_field[r.get("field", "?")].append(r)

    print(f"\n--- Eval 1: recall_3==0 by domain (n_miss={len(miss)}) ---")
    for dom, items in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        print(f"  {dom}: {len(items)}")

    print(f"\n--- Eval 1: recall_3==0 by field (top 25) ---")
    for field, items in sorted(by_field.items(), key=lambda x: -len(x[1]))[:25]:
        print(f"  {field}: {len(items)}")


def eval2_decomposition(rows: list[dict], f1_low: float) -> None:
    refusal_n = sum(1 for r in rows if _is_refusal(str(r.get("prediction", ""))))
    r0 = sum(1 for r in rows if recall_zero(r))
    low_f1 = [r for r in rows if float(r.get("f1_score") or 0) < f1_low]
    low_non_ref = [
        r
        for r in low_f1
        if not _is_refusal(str(r.get("prediction", "")))
    ]
    print(f"\n--- Eval 2: decomposition (n={len(rows)}) ---")
    print(f"  refusal (phrase match):     {refusal_n} ({100 * refusal_n / max(len(rows), 1):.1f}%)")
    print(f"  recall_3 == 0:               {r0} ({100 * r0 / max(len(rows), 1):.1f}%)")
    print(f"  F1 < {f1_low} (total):          {len(low_f1)} ({100 * len(low_f1) / max(len(rows), 1):.1f}%)")
    print(f"  F1 < {f1_low} & not refusal:   {len(low_non_ref)}")

    by_dom: dict[str, list[dict]] = defaultdict(list)
    for r in low_f1:
        by_dom[r.get("domain", "?")].append(r)
    print(f"\n--- Eval 2: F1 < {f1_low} by domain ---")
    for dom, items in sorted(by_dom.items(), key=lambda x: -len(x[1])):
        print(f"  {dom}: {len(items)}")

    nli_c: dict[str, int] = defaultdict(int)
    for r in low_f1:
        nli_c[str(r.get("nli_label", "?"))] += 1
    print(f"\n--- Eval 2: nli_label among F1 < {f1_low} ---")
    for k, v in sorted(nli_c.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


def row_export_dict(r: dict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "doc_id": r.get("doc_id"),
        "language": r.get("language"),
        "domain": r.get("domain"),
        "field": r.get("field"),
        "f1_score": r.get("f1_score"),
        "em_score": r.get("em_score"),
        "recall_3": r.get("recall_3"),
        "nli_label": r.get("nli_label"),
        "question": r.get("question"),
        "ground_truth": r.get("ground_truth"),
        "prediction": r.get("prediction"),
        "source_quote": r.get("source_quote"),
    }
    if extra:
        base.update(extra)
    return base


def export_csv(path: Path, dict_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not dict_rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict_rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(dict_rows)


def run_eval1(rows: list[dict], f1_low: float, f1_high: float, export: bool, export_dir: Path) -> None:
    print_contingency("Eval 1 — cross-lingual (English Q, native doc)", rows, f1_low, f1_high)
    eval1_domain_field_retrieval(rows)
    if export:
        miss = [r for r in rows if recall_zero(r)]
        ok_low = [
            r
            for r in rows
            if not recall_zero(r) and float(r.get("f1_score") or 0) < f1_low
        ]
        export_csv(
            export_dir / "eval1_recall3_zero.csv",
            [row_export_dict(r, {"bucket": "recall_3_zero"}) for r in miss],
        )
        export_csv(
            export_dir / "eval1_low_f1_recall_ok.csv",
            [row_export_dict(r, {"bucket": "low_f1_recall_gt0"}) for r in ok_low],
        )
        print(f"\nWrote: {export_dir / 'eval1_recall3_zero.csv'} ({len(miss)} rows)")
        print(f"Wrote: {export_dir / 'eval1_low_f1_recall_ok.csv'} ({len(ok_low)} rows)")


def run_eval2(rows: list[dict], f1_low: float, f1_high: float, export: bool, export_dir: Path) -> None:
    print_contingency("Eval 2 — monolingual (native Q, native doc)", rows, f1_low, f1_high)
    eval2_decomposition(rows, f1_low)
    if export:
        out: list[dict[str, Any]] = []
        for r in rows:
            pred = str(r.get("prediction", ""))
            ref = _is_refusal(pred)
            rz = recall_zero(r)
            f1 = float(r.get("f1_score") or 0)
            if ref:
                bucket = "refusal"
            elif rz:
                bucket = "retrieval_miss"
            elif f1 < f1_low:
                bucket = "low_f1_non_refusal"
            else:
                bucket = "ok_or_mid"
            out.append(row_export_dict(r, {"bucket": bucket, "is_refusal": ref}))
        export_csv(export_dir / "eval2_all_tagged.csv", out)
        print(f"\nWrote: {export_dir / 'eval2_all_tagged.csv'} ({len(out)} rows)")


def per_language_sections(
    eval_name: str,
    all_rows: list[dict],
    langs: list[str],
    f1_low: float,
    f1_high: float,
    export: bool,
    export_dir: Path,
) -> None:
    for lang in langs:
        sub = filter_language(all_rows, lang)
        if not sub:
            continue
        print(f"\n{'#' * 60}\n# {eval_name} — language={lang} (n={len(sub)})\n{'#' * 60}")
        if eval_name == "eval1":
            run_eval1(sub, f1_low, f1_high, False, export_dir)
        else:
            run_eval2(sub, f1_low, f1_high, False, export_dir)

    if export:
        export_dir.mkdir(parents=True, exist_ok=True)
        for lang in langs:
            sub = filter_language(all_rows, lang)
            if not sub:
                continue
            subdir = export_dir / f"lang_{lang}"
            if eval_name == "eval1":
                run_eval1(sub, f1_low, f1_high, True, subdir)
            else:
                run_eval2(sub, f1_low, f1_high, True, subdir)


def main() -> None:
    p = argparse.ArgumentParser(description="DocuNative eval error analysis from JSONL results.")
    p.add_argument("--eval", choices=("1", "2", "both"), default="both", help="Which eval file to analyze")
    p.add_argument(
        "--language",
        choices=("zh", "hi", "pl", "all"),
        default="all",
        help="Filter rows by document language; 'all' includes aggregate and per-language sections",
    )
    p.add_argument("--f1-low", type=float, default=0.2, help="Upper bound of 'low' F1 band")
    p.add_argument("--f1-high", type=float, default=0.5, help="Upper bound of 'mid' F1 band")
    p.add_argument(
        "--export",
        action="store_true",
        help="Write CSVs under --export-dir (per-language subdirs when --language all)",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=_DOCUNATIVE_ROOT / "error analysis" / "exports",
        help="Output directory for CSV exports",
    )
    p.add_argument(
        "--per-language",
        action="store_true",
        help="When --language all, also print per-language contingency + slices",
    )
    args = p.parse_args()
    f1_low, f1_high = args.f1_low, args.f1_high
    if f1_low >= f1_high:
        raise SystemExit("--f1-low must be less than --f1-high")

    langs = ["zh", "hi", "pl"]

    if args.eval in ("1", "both"):
        path1 = resolve_eval_path(EVAL1_CANDIDATES, "Eval 1")
        e1 = load_jsonl(path1)
        print(f"Loaded Eval 1: {path1.name} ({len(e1)} rows)")
        if args.language == "all":
            run_eval1(e1, f1_low, f1_high, args.export and not args.per_language, args.export_dir)
            if args.per_language:
                per_language_sections("eval1", e1, langs, f1_low, f1_high, args.export, args.export_dir)
        else:
            e1 = filter_language(e1, args.language)
            run_eval1(e1, f1_low, f1_high, args.export, args.export_dir)

    if args.eval in ("2", "both"):
        path2 = resolve_eval_path(EVAL2_CANDIDATES, "Eval 2")
        e2 = load_jsonl(path2)
        print(f"Loaded Eval 2: {path2.name} ({len(e2)} rows)")
        if args.language == "all":
            run_eval2(e2, f1_low, f1_high, args.export and not args.per_language, args.export_dir)
            if args.per_language:
                per_language_sections("eval2", e2, langs, f1_low, f1_high, args.export, args.export_dir)
        else:
            e2 = filter_language(e2, args.language)
            run_eval2(e2, f1_low, f1_high, args.export, args.export_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
