#!/usr/bin/env python3
"""Tabulate several runs side by side, focusing on the fault barriers.

Used for the explicit-fault-representation experiment: does mapping the
coordinates through ``tanh(signed distance / width)`` recover more of the head
step across a barrier than appending the same indicator as an extra input?

    python scripts/compare_fault_representation.py runs/fc_off runs/fc_on_015 ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _label(run: pathlib.Path) -> str:
    cfg_path = run / "config_used.yaml"
    if not cfg_path.exists():
        return run.name
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    m = cfg.get("model", {})
    if not m.get("fault_coords"):
        return f"{m.get('arch')}, appended indicator"
    return f"{m.get('arch')}, mapped (bw x{m.get('fault_sigma_scale')})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        run = pathlib.Path(r)
        rp = run / "report.json"
        if not rp.exists():
            print(f"  (skipping {run}: no report.json)")
            continue
        rep = json.load(open(rp))
        row = {
            "label": _label(run),
            "cal_rmse": rep["head"]["train"]["rmse"],
            "val_rmse": rep["head"]["validation"]["rmse"],
            "val_r2": rep["head"]["validation"]["r2"],
            "grid_head_rmse": rep.get("grid", {}).get("head_L0", {}).get("rmse"),
            "grid_head_r2": rep.get("grid", {}).get("head_L0", {}).get("r2"),
            "logT_rmse": rep.get("grid", {}).get("log10T_L0", {}).get("rmse"),
            "final_pde": (rep.get("config", {}) or {}).get("train_seconds"),
        }
        for f in rep.get("faults", {}).get("faults", []):
            for rec in f["by_offset"]:
                key = f"f{f['fault']}@{rec['offset_m']:.0f}m"
                row[key + "_pred"] = rec["predicted_mean_abs_step_m"]
                row[key + "_ref"] = rec.get("reference_mean_abs_step_m")
                row[key + "_frac"] = rec.get("recovered_fraction")
        rows.append(row)

    if not rows:
        sys.exit("no runs with reports")

    print("\n" + "=" * 104)
    print("HEAD FIELD")
    print("=" * 104)
    print(f"{'run':<34s}{'cal RMSE':>10s}{'val RMSE':>10s}{'val R2':>9s}"
          f"{'grid RMSE':>11s}{'grid R2':>9s}{'log10T':>9s}")
    print("-" * 104)
    for r in rows:
        print(
            f"{r['label']:<34s}{r['cal_rmse']:>10.3f}{r['val_rmse']:>10.3f}"
            f"{r['val_r2']:>9.3f}"
            f"{(r['grid_head_rmse'] if r['grid_head_rmse'] is not None else float('nan')):>11.3f}"
            f"{(r['grid_head_r2'] if r['grid_head_r2'] is not None else float('nan')):>9.3f}"
            f"{(r['logT_rmse'] if r['logT_rmse'] is not None else float('nan')):>9.3f}"
        )

    step_keys = sorted({k[:-5] for r in rows for k in r if k.endswith("_frac")})
    if step_keys:
        print("\n" + "=" * 104)
        print("HEAD STEP HELD UP ACROSS EACH FAULT  (predicted / reference, m)")
        print("=" * 104)
        header = f"{'run':<34s}" + "".join(f"{k:>22s}" for k in step_keys)
        print(header)
        print("-" * 104)
        for r in rows:
            line = f"{r['label']:<34s}"
            for k in step_keys:
                p, ref = r.get(k + "_pred"), r.get(k + "_ref")
                frac = r.get(k + "_frac")
                cell = (
                    f"{p:.2f}/{ref:.2f} ({100 * frac:.0f}%)"
                    if p is not None and ref is not None and frac is not None
                    else "-"
                )
                line += f"{cell:>22s}"
            print(line)
    print()


if __name__ == "__main__":
    main()
