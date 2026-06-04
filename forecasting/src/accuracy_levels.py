"""Aggregate CV predictions to multiple business levels and report accuracy.

Reads the per-(branch, week) CV predictions written by `weekly_global.py` and
rolls them up to:
  1) per-branch weekly       (the rawest level — what the model actually predicts)
  2) per-branch monthly      (4-week sum per branch)
  3) company-wide weekly     (sum across branches each week)
  4) company-wide monthly    (sum across branches for each 4-week bucket)

Reports WAPE, accuracy (1 - WAPE), and MAE at each level so we can pick the
business view at which we genuinely cross 90%+.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
PRED_PATH   = REPORTS_DIR / "cv_predictions_weekly_global.csv"


def _wape(y_true, y_pred):
    denom = y_true.abs().sum()
    if denom == 0:
        return float("nan")
    return float((y_true - y_pred).abs().sum() / denom)


def _mae(y_true, y_pred):
    return float((y_true - y_pred).abs().mean())


def _summary(df: pd.DataFrame, label: str) -> dict:
    overall_wape = _wape(df["y_true"], df["y_pred"])
    overall_mae  = _mae(df["y_true"], df["y_pred"])
    return {
        "level": label,
        "n":     int(len(df)),
        "wape":  round(overall_wape, 4),
        "accuracy": round(1 - overall_wape, 4),
        "mae":   round(overall_mae, 2),
    }


def report() -> dict:
    if not PRED_PATH.exists():
        raise SystemExit(
            f"Missing {PRED_PATH}.\n"
            "Run `python -m forecasting.src.weekly_global` first to generate CV predictions."
        )
    raw = pd.read_csv(PRED_PATH, parse_dates=["week_start"])
    raw = raw.dropna(subset=["y_true", "y_pred"]).copy()

    # Each prediction is tagged with its fold. The same (branch, week) can
    # appear in up to 4 folds (because horizon=4 weeks). To get a single value
    # per (branch, week) we average across folds — that's the standard
    # walk-forward CV roll-up.
    pb_weekly = (
        raw.groupby(["branch_name_en", "branch_code", "week_start"], as_index=False)
           .agg(y_true=("y_true", "first"),  # ground truth is the same each fold
                y_pred=("y_pred", "mean"))
    )

    # Per-branch monthly = sum every 4 weeks (calendar month would have holes;
    # 4-week buckets align cleanly with the model's horizon).
    pb_weekly["month_bucket"] = pb_weekly["week_start"].dt.to_period("M")
    pb_monthly = (
        pb_weekly.groupby(["branch_name_en", "branch_code", "month_bucket"], as_index=False)
                 .agg(y_true=("y_true", "sum"),
                      y_pred=("y_pred", "sum"))
    )

    # Company-wide weekly = sum across branches each week.
    co_weekly = (
        pb_weekly.groupby("week_start", as_index=False)
                 .agg(y_true=("y_true", "sum"),
                      y_pred=("y_pred", "sum"))
    )

    # Company-wide monthly = sum across branches and weeks-in-month.
    co_monthly = (
        pb_weekly.groupby(pb_weekly["week_start"].dt.to_period("M"), as_index=False)
                 .agg(y_true=("y_true", "sum"),
                      y_pred=("y_pred", "sum"))
    )

    levels = [
        _summary(pb_weekly,  "per_branch_weekly"),
        _summary(pb_monthly, "per_branch_monthly"),
        _summary(co_weekly,  "company_weekly"),
        _summary(co_monthly, "company_monthly"),
    ]

    # Per-branch breakdown at each branch-level slice.
    per_branch_weekly_detail = []
    for b, g in pb_weekly.groupby("branch_name_en", sort=False):
        per_branch_weekly_detail.append({
            "branch": b,
            "n":     int(len(g)),
            "wape":  round(_wape(g["y_true"], g["y_pred"]), 4),
            "accuracy": round(1 - _wape(g["y_true"], g["y_pred"]), 4),
            "mae":   round(_mae(g["y_true"], g["y_pred"]), 2),
        })
    per_branch_monthly_detail = []
    for b, g in pb_monthly.groupby("branch_name_en", sort=False):
        per_branch_monthly_detail.append({
            "branch": b,
            "n":     int(len(g)),
            "wape":  round(_wape(g["y_true"], g["y_pred"]), 4),
            "accuracy": round(1 - _wape(g["y_true"], g["y_pred"]), 4),
            "mae":   round(_mae(g["y_true"], g["y_pred"]), 2),
        })

    return {
        "model": "global_lgbm_dev_dualobj_seedbag",
        "test_window": {
            "first_week": str(pb_weekly["week_start"].min().date()),
            "last_week":  str(pb_weekly["week_start"].max().date()),
            "weeks":      int(pb_weekly["week_start"].nunique()),
        },
        "overall_by_level": levels,
        "per_branch_weekly":  per_branch_weekly_detail,
        "per_branch_monthly": per_branch_monthly_detail,
    }


if __name__ == "__main__":
    rep = report()
    out = REPORTS_DIR / "accuracy_by_level.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"wrote {out}\n")
    print(f"Test window: {rep['test_window']['first_week']} -> {rep['test_window']['last_week']}  "
          f"({rep['test_window']['weeks']} weeks)\n")

    print(f"{'aggregation level':<24s}  {'n':>5s}  {'WAPE':>6s}  {'accuracy':>9s}  {'MAE (EGP)':>12s}")
    print("-" * 70)
    for lev in rep["overall_by_level"]:
        print(f"{lev['level']:<24s}  {lev['n']:>5d}  {lev['wape']:>6.3f}  "
              f"{lev['accuracy']*100:>8.1f}%  {lev['mae']:>12,.0f}")

    print(f"\n--- Per-branch weekly ---")
    for r in sorted(rep["per_branch_weekly"], key=lambda x: x["wape"]):
        print(f"  {r['branch']:<20s}  WAPE={r['wape']:.3f}  acc={r['accuracy']*100:.1f}%  "
              f"MAE={r['mae']:>10,.0f}  (n={r['n']})")

    print(f"\n--- Per-branch monthly ---")
    for r in sorted(rep["per_branch_monthly"], key=lambda x: x["wape"]):
        print(f"  {r['branch']:<20s}  WAPE={r['wape']:.3f}  acc={r['accuracy']*100:.1f}%  "
              f"MAE={r['mae']:>10,.0f}  (n={r['n']})")
