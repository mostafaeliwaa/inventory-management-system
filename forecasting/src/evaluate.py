"""Walk-forward CV runner. For now: baseline only (Prophet + LGBM plug in later)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forecasting.src.data_loader import get_panel
from forecasting.src.metrics import summarize
from forecasting.src.models.baseline import naive_forecast, seasonal_naive_forecast
from forecasting.src.splits import Fold, walk_forward_folds

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
TARGET = "revenue"


def _pick_best_per_branch(all_preds: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Per branch, return the model name with lowest WAPE across CV folds."""
    from forecasting.src.metrics import wape
    out: dict[str, str] = {}
    candidates = [n for n in all_preds if n not in {"naive", "ensemble"}]  # ensemble built after
    branches = set()
    for df in all_preds.values():
        branches.update(df["branch_name_en"].unique())
    for b in branches:
        best_name, best_w = None, float("inf")
        for name in candidates:
            df = all_preds[name]
            d = df[df["branch_name_en"] == b]
            if d.empty:
                continue
            w = wape(d["y_true"].to_numpy(float), d["y_pred"].to_numpy(float))
            if w < best_w:
                best_w, best_name = w, name
        if best_name:
            out[b] = best_name
    return out


def _smart_ensemble(all_preds: dict[str, pd.DataFrame], picks: dict[str, str]) -> pd.DataFrame | None:
    if not picks:
        return None
    pieces = []
    for branch, model_name in picks.items():
        df = all_preds[model_name]
        pieces.append(df[df["branch_name_en"] == branch].copy())
    return pd.concat(pieces, ignore_index=True) if pieces else None


def _write_plots(panel: pd.DataFrame, all_preds: dict[str, pd.DataFrame]) -> None:
    import plotly.graph_objects as go
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for branch, g in panel.groupby("branch_name_en", sort=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=g["posting_date"], y=g[TARGET], name="actual", mode="lines", line=dict(color="black", width=1.5),
        ))
        for name, df in all_preds.items():
            d = df[df["branch_name_en"] == branch].sort_values("posting_date") if "branch_name_en" in df.columns \
                else df.merge(panel[["branch_code", "branch_name_en"]].drop_duplicates(), on="branch_code") \
                       .query("branch_name_en == @branch").sort_values("posting_date")
            fig.add_trace(go.Scatter(x=d["posting_date"], y=d["y_pred"], name=name, mode="lines+markers", opacity=0.7))
        fig.update_layout(title=f"{branch} — daily revenue: actual vs forecasts (walk-forward CV)",
                          xaxis_title="date", yaxis_title="revenue (EGP)", height=420)
        fig.write_html(str(PLOTS_DIR / f"{branch}.html"))


def _available_models() -> dict:
    """Return {name: fn} including baselines plus whichever heavy models import cleanly."""
    models = {
        "seasonal_naive": seasonal_naive_forecast,
        "naive": naive_forecast,
    }
    try:
        from forecasting.src.models.prophet_model import prophet_forecast
        models["prophet"] = prophet_forecast
    except ImportError as e:
        print(f"[skip] prophet not available: {e}")
    try:
        from forecasting.src.models.lgbm_model import lgbm_forecast
        models["lgbm"] = lgbm_forecast
    except ImportError as e:
        print(f"[skip] lightgbm not available: {e}")
    try:
        from forecasting.src.models.lgbm_direct import lgbm_direct_forecast
        models["lgbm_direct"] = lgbm_direct_forecast
    except ImportError as e:
        print(f"[skip] lgbm_direct not available: {e}")
    return models


def _run_one(model_fn, panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    train = panel[panel["posting_date"] < fold.train_end]
    test_dates = pd.date_range(fold.test_start, fold.test_end - pd.Timedelta(days=1), freq="D")
    preds = model_fn(train, test_dates, target=TARGET)
    truth = panel[panel["posting_date"].isin(test_dates)][["branch_code", "branch_name_en", "posting_date", TARGET]]
    truth = truth.rename(columns={TARGET: "y_true"})
    merged = truth.merge(preds, on=["branch_code", "posting_date"], how="left")
    merged["fold_id"] = fold.fold_id
    return merged


def run_all() -> dict:
    panel = get_panel()
    # With ~2 years of data, hold out ~6 months as test (~26 folds × 7-day horizon)
    # and require ~18 months of training so lag-364 and yearly seasonality have
    # at least one full cycle to learn from.
    folds = walk_forward_folds(panel["posting_date"], horizon=7, first_train_days=548, step=7)

    models = _available_models()
    all_preds: dict[str, pd.DataFrame] = {}
    for name, fn in models.items():
        print(f"[{name}] running {len(folds)} folds...")
        rows = [_run_one(fn, panel, f) for f in folds]
        all_preds[name] = pd.concat(rows, ignore_index=True)

    # Simple ensemble = mean of Prophet + best-performing LGBM variant per fold.
    if "prophet" in all_preds and ("lgbm" in all_preds or "lgbm_direct" in all_preds):
        lgbm_key = "lgbm_direct" if "lgbm_direct" in all_preds else "lgbm"
        p, l = all_preds["prophet"], all_preds[lgbm_key]
        ens = p.merge(l[["branch_code", "posting_date", "y_pred"]],
                      on=["branch_code", "posting_date"], suffixes=("_p", "_l"))
        ens["y_pred"] = (ens["y_pred_p"] + ens["y_pred_l"]) / 2
        ens = ens.drop(columns=["y_pred_p", "y_pred_l"])
        all_preds["ensemble"] = ens

    # Smart ensemble: pick winner per branch from per-branch WAPE.
    per_branch_winner = _pick_best_per_branch(all_preds)
    smart = _smart_ensemble(all_preds, per_branch_winner)
    if smart is not None:
        all_preds["smart_ensemble"] = smart

    _write_plots(panel, all_preds)

    report: dict = {
        "config": {
            "horizon": 7,
            "first_train_days": 60,
            "step": 7,
            "folds": len(folds),
            "cv_first_test": str(folds[0].test_start.date()),
            "cv_last_test": str((folds[-1].test_end - pd.Timedelta(days=1)).date()),
            "target": TARGET,
        },
        "per_branch": {},
        "overall": {},
    }
    for name, df in all_preds.items():
        by_branch = summarize(df, group="branch_name_en")
        overall = summarize(df).iloc[0].to_dict()
        for _, row in by_branch.iterrows():
            b = row["branch_name_en"]
            report["per_branch"].setdefault(b, {})[name] = {
                "mae": round(row["mae"], 2),
                "mape": round(row["mape"], 4),
                "wape": round(row["wape"], 4),
                "smape": round(row["smape"], 4),
                "n": int(row["n"]),
            }
        report["overall"][name] = {
            "mae": round(overall["mae"], 2),
            "mape": round(overall["mape"], 4),
            "wape": round(overall["wape"], 4),
            "smape": round(overall["smape"], 4),
            "n": int(overall["n"]),
        }
    return report


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    report = run_all()
    out_path = REPORTS_DIR / "results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(f"\n=== Overall (across all branches, all folds) ===")
    for name, m in report["overall"].items():
        print(f"{name:>16s}  WAPE={m['wape']:.3f}  MAE={m['mae']:>10.0f}  sMAPE={m['smape']:.3f}  n={m['n']}")
    print(f"\n=== Per-branch WAPE (lower is better) ===")
    branches = list(report["per_branch"].keys())
    name_list = list(report["overall"].keys())
    print(f"{'branch':<20s} " + "  ".join(f"{n:>16s}" for n in name_list))
    for b in branches:
        cells = [f"{report['per_branch'][b][n]['wape']:.3f}" for n in name_list]
        print(f"{b:<20s} " + "  ".join(f"{c:>16s}" for c in cells))
