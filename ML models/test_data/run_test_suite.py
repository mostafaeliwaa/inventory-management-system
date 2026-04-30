"""Run the 15 hand-crafted products in test_products.json against the live API
and print per-row predictions plus aggregate accuracy.

Usage:
    python test_data/run_test_suite.py                       # localhost:8003
    python test_data/run_test_suite.py --base-url http://api.example.com
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8003")
    ap.add_argument("--products", default=str(HERE / "test_products.json"))
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    products = json.loads(Path(args.products).read_text(encoding="utf-8"))["products"]

    try:
        h = requests.get(f"{base}/health", timeout=5).json()
    except Exception as e:
        print(f"ERROR: cannot reach {base}/health -> {e}")
        return 2
    print(f"API: {base}  health={h}\n")

    fmt = "{:<40} {:>10} {:>10} {:>15} {:>10} {:>15} {:>10}"
    print(fmt.format("name", "exp_demand", "got", "demand_ok",
                     "exp_dead", "got(prob)", "dead_ok"))
    print("-" * 115)

    n = len(products)
    demand_ok = dead_ok = 0
    mismatches = []

    for p in products:
        feats = p["features"]
        d = requests.post(f"{base}/predict/demand", json=feats, timeout=10).json()
        s = requests.post(f"{base}/predict/dead-stock", json=feats, timeout=10).json()

        d_ok = d["label"] == p["expected_demand"]
        s_ok = bool(s["is_dead_stock"]) == bool(p["expected_dead"])
        demand_ok += d_ok
        dead_ok += s_ok
        if not d_ok or not s_ok:
            mismatches.append((p["name"], p["expected_demand"], d["label"],
                               p["expected_dead"], s["is_dead_stock"], s["probability"]))

        dead_str = f"{s['is_dead_stock']}({s['probability']})"
        print(fmt.format(p["name"][:40], p["expected_demand"], d["label"],
                         "OK" if d_ok else "MISS",
                         str(p["expected_dead"]), dead_str,
                         "OK" if s_ok else "MISS"))

    print()
    print(f"Demand classifier accuracy:    {demand_ok}/{n}  ({demand_ok/n:.0%})")
    print(f"Dead-stock predictor accuracy: {dead_ok}/{n}  ({dead_ok/n:.0%})")

    if mismatches:
        print("\nMismatched rows:")
        for name, exp_d, got_d, exp_s, got_s, prob in mismatches:
            print(f"  - {name}")
            if exp_d != got_d:
                print(f"      demand: expected={exp_d}  got={got_d}")
            if bool(exp_s) != bool(got_s):
                print(f"      dead:   expected={exp_s}  got={got_s} (prob={prob})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
