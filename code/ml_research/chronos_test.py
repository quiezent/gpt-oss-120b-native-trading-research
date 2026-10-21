"""Test chronos_tool.live() with a fake gateway (one cached symbol, one not in the cache, one unknown) and
sim_provider(). Writes runs/ml/research/chronos_livetest.json.

    .venv\\Scripts\\python.exe ml_research\\chronos_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trader                              # noqa: E402
from ml_research import chronos_tool as C  # noqa: E402


class FakeGW:
    """bars() serves daily bars; 'ZZNEW' is a synthetic symbol not in the cache (a copy of MSFT's prices x 0.5)."""

    def __init__(self, d):
        self.d = d

    def bars(self, symbol, duration="1 Y", bar_size="1 day", **kw):
        if symbol != "ZZNEW":
            raise ValueError(f"200: No security definition has been found for {symbol}")
        j = self.d.k["MSFT"]
        return [{"date": self.d.dates[i], "open": 0.5 * self.d.o[j, i], "high": 0.5 * self.d.h[j, i], "low": 0.5 * self.d.l[j, i],
                 "close": 0.5 * self.d.c[j, i], "volume": float(self.d.v[j, i])} for i in range(len(self.d.dates))]

    def quotes(self, symbols, **kw):
        return [{"symbol": s, "error": "fake"} for s in symbols]


def main():
    d = trader.Data()
    out = C.live(FakeGW(d), symbols=["MSFT", "ZZNEW", "NOPE"], data=d)
    a, b = out["rows"]["MSFT"], out["rows"]["ZZNEW"]
    inside = lambda r: r["range1_p10_p90"][0] < r["range1_pct"] < r["range1_p10_p90"][1]
    checks = {"msft_blend_with_interval": a["basis"] == "blend" and inside(a),
              "uncached_served_with_interval": b["basis"].startswith("bolt_recal") and inside(b),
              "uncached_close_halved": abs(b["close"] - 0.5 * a["close"]) < 0.01,
              "uncached_same_scale_as_msft": abs(b["range1_pct"] - a["range1_pct"]) < 0.5 * a["range1_pct"],
              "unknown_flagged": "note" in out["rows"]["NOPE"],
              "vol20_not_served": "vol20_pct" not in a}
    sp = C.sim_provider(d)
    sim = None
    if sp is not None:
        t = next(i for i, s in enumerate(d.dates) if s >= "20260105")
        got = sp(t)
        sim = {"as_of": got[0], "n": len(got[1]), "example": next(iter(got[1].items()))} if got else None
        checks["sim_as_of_prev_close"] = bool(got) and got[0] == d.dates[t - 1]
    res = {"live": out, "sim": sim if sp is not None else "sim_provider returned None (tool dropped)", "checks": checks}
    (C.OUT / "chronos_livetest.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    assert all(checks.values()), checks


if __name__ == "__main__":
    main()
