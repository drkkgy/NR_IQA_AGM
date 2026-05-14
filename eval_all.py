"""
Reproduce multiple paper number in one command.

Iterates the ``CHECKPOINT_REGISTRY`` defined in ``eval_checkpoint.py``, runs
the unified evaluation pipeline on each registered run, and prints an
SRCC/PLCC summary table at the end.

Datasets that are not present under ``--data_dir`` are skipped by default
(with a warning); pass ``--strict`` to fail loudly on the first missing one.

Usage
-----
    # Run multiple registered checkpoint on its native test set
    python eval_all.py

    # Run only the gating-architecture checkpoints
    python eval_all.py --filter Gating

    # Run on a non-default dataset root, larger batch size
    python eval_all.py --data_dir /data/IQA --batch_size 8

    # Crash on the first missing dataset instead of skipping
    python eval_all.py --strict

Author: Ankit Yadav
"""

import argparse
import json
import os
import sys
import traceback
import warnings

from configs.default import _make_dataset_paths
from eval_checkpoint import CHECKPOINT_REGISTRY, run_eval

warnings.simplefilter(action="ignore", category=FutureWarning)


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Dataset availability check                                         │
# ╰──────────────────────────────────────────────────────────────────────╯

# Mirrors _load_eval_dataset's cross-dataset resolution in eval_checkpoint.py.
_CROSS_DATASET_TEST = {
    "KonIQ_10K_CLIVE": "CLIVE",
    "CLIVE_KonIQ_10K": "KonIQ_10K",
}


def _resolve_test_dataset(dataset_id: str) -> str:
    return _CROSS_DATASET_TEST.get(dataset_id, dataset_id)


def _dataset_available(dataset_id: str, dataset_paths: dict) -> bool:
    resolved = _resolve_test_dataset(dataset_id)
    path = dataset_paths.get(resolved)
    return path is not None and os.path.isdir(path)


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Result table printer                                               │
# ╰──────────────────────────────────────────────────────────────────────╯

def _print_summary(rows: list[dict]) -> None:
    """Print a Markdown-style summary table of all eval results."""
    if not rows:
        print("\nNo runs were evaluated.")
        return

    headers = ("Run", "Arch", "Train", "Test", "SRCC", "PLCC", "Status")

    def fmt_cell(row: dict, key: str) -> str:
        v = row.get(key, "")
        if key in ("SRCC", "PLCC") and isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v is not None else ""

    table = [tuple(fmt_cell(r, k) for k in headers) for r in rows]
    widths = [max(len(h), max(len(r[i]) for r in table)) for i, h in enumerate(headers)]

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"

    print("\n" + "=" * len(header_line))
    print("Evaluation Summary")
    print("=" * len(header_line))
    print(header_line)
    print(sep)
    for r in table:
        print("| " + " | ".join(f"{c:<{w}}" for c, w in zip(r, widths)) + " |")
    print()


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Main                                                                │
# ╰──────────────────────────────────────────────────────────────────────╯

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple registered checkpoint and print a summary table.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./Dataset",
                        help="Root directory containing dataset subfolders.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device string, e.g. cuda, cuda:0, cuda:1, cpu")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only evaluate runs whose name contains this substring\n"
                             "(case-insensitive). E.g. --filter Gating")
    parser.add_argument("--strict", action="store_true",
                        help="Fail loudly on the first missing dataset or failed run.\n"
                             "Default: skip-with-warning and continue.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save combined JSON results.\n"
                             "Default: results/eval_all.json")

    args = parser.parse_args()

    # ── Select runs ──────────────────────────────────────────────────────
    selected = list(CHECKPOINT_REGISTRY.items())
    if args.filter is not None:
        needle = args.filter.lower()
        selected = [(n, e) for n, e in selected if needle in n.lower()]
    if not selected:
        print(f"No registered runs match filter '{args.filter}'.", file=sys.stderr)
        sys.exit(1)

    dataset_paths = _make_dataset_paths(args.data_dir)

    print(f"Evaluating {len(selected)} run(s) "
          f"{'(filter: ' + args.filter + ')' if args.filter else ''}".rstrip())
    print(f"Data root: {args.data_dir}")
    print(f"Device:    {args.device}\n")

    # ── Evaluate each run ────────────────────────────────────────────────
    rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    for idx, (name, entry) in enumerate(selected, 1):
        print(f"\n{'━' * 72}")
        print(f"[{idx}/{len(selected)}] Run: {name}  "
              f"(arch={entry['arch']}, train={entry['train']}, test={entry['test']})")
        print("━" * 72)

        ckpt_dir   = entry["path"]
        dataset_id = entry["dataset_id"]

        # Pre-flight: checkpoint exists?
        if not os.path.isdir(ckpt_dir):
            msg = f"checkpoint dir missing: {ckpt_dir}"
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(2)
            print(f"SKIP: {msg}")
            rows.append({
                "Run": name, "Arch": entry["arch"],
                "Train": entry["train"], "Test": entry["test"],
                "SRCC": None, "PLCC": None, "Status": "skip (no ckpt)",
            })
            continue

        # Pre-flight: dataset available?
        if not _dataset_available(dataset_id, dataset_paths):
            resolved = _resolve_test_dataset(dataset_id)
            msg = (f"dataset '{dataset_id}' (resolved: {resolved}) "
                   f"not found under {args.data_dir}")
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(2)
            print(f"SKIP: {msg}")
            rows.append({
                "Run": name, "Arch": entry["arch"],
                "Train": entry["train"], "Test": entry["test"],
                "SRCC": None, "PLCC": None, "Status": "skip (no data)",
            })
            continue

        # Run evaluation
        try:
            result = run_eval(
                ckpt_dir=ckpt_dir,
                dataset_id=dataset_id,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                arch=entry["arch"],
                device_str=args.device,
                seed=entry["seed"],
                eval_split=entry["eval_split"],
            )
            result["run"] = name
            rows.append({
                "Run": name, "Arch": entry["arch"],
                "Train": entry["train"], "Test": entry["test"],
                "SRCC": result["SRCC"], "PLCC": result["PLCC"],
                "Status": "ok",
            })
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failed.append((name, err))
            if args.strict:
                print(f"ERROR: run '{name}' failed: {err}", file=sys.stderr)
                traceback.print_exc()
                sys.exit(3)
            print(f"FAIL: {err}")
            rows.append({
                "Run": name, "Arch": entry["arch"],
                "Train": entry["train"], "Test": entry["test"],
                "SRCC": None, "PLCC": None, "Status": "fail",
            })

    # ── Summary ──────────────────────────────────────────────────────────
    _print_summary(rows)

    if failed:
        print(f"{len(failed)} run(s) failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")

    # ── Save combined JSON ───────────────────────────────────────────────
    out_path = args.output
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        out_path = "results/eval_all.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=4)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
