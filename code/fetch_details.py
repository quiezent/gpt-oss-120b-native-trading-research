"""IBKR industry classification for the universe -> lists/industries.json (for the ML arm's features)."""
import json
from pathlib import Path

import trader
from tws import Gateway

out = Path(__file__).resolve().parent / "lists" / "industries.json"
syms = sorted(set(trader.universe()))
with Gateway("trader-training", timeout=120) as gw:
    rows = gw.details(syms)
data = {r["symbol"]: {k: r.get(k) for k in ("industry", "category", "subcategory", "stock_type", "long_name")} for r in rows if not r.get("error")}
errs = [r.get("symbol") for r in rows if r.get("error")]
out.write_text(json.dumps(data, indent=1), encoding="utf-8")
print(len(data), "classified;", len(errs), "errors", errs[:10])
