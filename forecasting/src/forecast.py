"""Produce the next-7-day forecast per branch using the ensemble (Prophet + LGBM)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forecasting.src.data_loader import get_panel
from forecasting.src.models.lgbm_model import lgbm_forecast
from forecasting.src.models.prophet_model import prophet_forecast

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
HORIZON = 7


def next_7d_forecast() -> dict:
    panel = get_panel()
    last_day = panel["posting_date"].max()
    test_dates = pd.date_range(last_day + pd.Timedelta(days=1), periods=HORIZON, freq="D")

    p = prophet_forecast(panel, test_dates).rename(columns={"y_pred": "prophet"})
    l = lgbm_forecast(panel, test_dates).rename(columns={"y_pred": "lgbm"})
    merged = p.merge(l, on=["branch_code", "posting_date"])
    merged["ensemble"] = (merged["prophet"] + merged["lgbm"]) / 2

    branches = panel[["branch_code", "branch_name_en"]].drop_duplicates()
    merged = merged.merge(branches, on="branch_code", how="left")

    by_branch: dict[str, dict] = {}
    for branch, g in merged.groupby("branch_name_en", sort=False):
        g = g.sort_values("posting_date")
        by_branch[branch] = {
            "branch_code": int(g["branch_code"].iloc[0]),
            "dates": [d.strftime("%Y-%m-%d") for d in g["posting_date"]],
            "prophet": [round(float(v), 2) for v in g["prophet"]],
            "lgbm":    [round(float(v), 2) for v in g["lgbm"]],
            "ensemble":[round(float(v), 2) for v in g["ensemble"]],
            "total_ensemble_7d": round(float(g["ensemble"].sum()), 2),
        }

    return {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "history_last_date": last_day.strftime("%Y-%m-%d"),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in test_dates],
        "horizon_days": HORIZON,
        "target": "daily_revenue_egp",
        "per_branch": by_branch,
        "company_total_ensemble_7d": round(sum(v["total_ensemble_7d"] for v in by_branch.values()), 2),
    }


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    out = next_7d_forecast()
    path = REPORTS_DIR / "next_7d.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {path}")
    print(f"\nForecast horizon: {out['forecast_dates'][0]} to {out['forecast_dates'][-1]}")
    print(f"Company total (ensemble, 7d): {out['company_total_ensemble_7d']:,.0f} EGP\n")
    print(f"{'branch':<20s} {'7d ensemble (EGP)':>20s}")
    for b, v in sorted(out["per_branch"].items(), key=lambda kv: -kv[1]["total_ensemble_7d"]):
        print(f"{b:<20s} {v['total_ensemble_7d']:>20,.0f}")
