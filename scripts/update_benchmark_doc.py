#!/usr/bin/env python3
"""Render benchmark.json into the architecture comparison document."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

MARKER = "<!-- BENCHMARK_TABLE -->"

COLUMNS = [
    ("arch", "architecture", "{}"),
    ("params", "params", "{:,}"),
    ("seconds", "wall time (s)", "{:.0f}"),
    ("val_rmse_m", "held-out well RMSE (m)", "{:.3f}"),
    ("val_r2", "held-out R²", "{:.3f}"),
    ("grid_head_L0_rmse", "head field RMSE (m)", "{:.3f}"),
    ("grid_head_L0_r2", "head field R²", "{:.3f}"),
    ("grid_log10T_L0_rmse", "log10 T RMSE", "{:.3f}"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark", default="runs/benchmark/benchmark.json", nargs="?")
    ap.add_argument("-d", "--doc", default="docs/architecture_comparison.md")
    args = ap.parse_args()

    rows = json.load(open(args.benchmark))
    if not rows:
        sys.exit("no benchmark rows")

    header = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, sep]

    best = min(rows, key=lambda r: r.get("val_rmse_m", float("inf")))
    for r in rows:
        cells = []
        for key, _, fmt in COLUMNS:
            v = r.get(key)
            cells.append("—" if v is None else fmt.format(v))
        if r is best:
            cells[0] = f"**{cells[0]}**"
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    n_iter = "the configured budget"
    note = (
        f"\n\nBest held-out RMSE: **{best['arch']}** "
        f"({best['val_rmse_m']:.3f} m). Same data, seed and iteration budget "
        f"({n_iter}) for every row; width and depth are held equal, so parameter "
        f"counts differ.\n"
    )

    doc = pathlib.Path(args.doc)
    text = doc.read_text()
    if MARKER in text:
        head, _, tail = text.partition(MARKER)
        # Replace everything between the marker and the next heading.
        rest = tail.split("\n## ", 1)
        remainder = ("\n## " + rest[1]) if len(rest) > 1 else ""
        text = head + MARKER + "\n\n" + table + note + remainder
    else:
        text += "\n\n" + table + note
    doc.write_text(text)
    print(f"updated {doc}")
    print(table)


if __name__ == "__main__":
    main()
