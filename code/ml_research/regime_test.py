"""Checks for the regime tool: live() on a fake gateway built from cached bars (incl. a symbol absent from the cache
and a missing implied-vol close), sim_provider() at a few sessions, and live() on a cache truncated at 20260601 (its
point forecast and features use nothing later; its interval quantiles and shipping choice come from the full validation).

    .venv\\Scripts\\python.exe ml_research\\regime_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trader  # noqa: E402
from ml_research import regime as R  # noqa: E402


class FakeGW:
    def __init__(self, d, iv):
        self.d, self.iv = d, iv
        self.calls = []

    def quotes(self, symbols):
        self.calls.append(("quotes", tuple(symbols)))
        out = []
        for s in symbols:
            j = self.d.k.get(s)
            if j is None:
                continue
            c = self.d.c[j]
            out.append({"symbol": s, "last": float(c[-1]), "bid": float(c[-1]) - 0.01, "ask": float(c[-1]) + 0.01,
                        "open": float(self.d.o[j, -1]), "high": float(self.d.h[j, -1]), "low": float(self.d.l[j, -1]),
                        "close": float(c[-2]), "volume": float(self.d.v[j, -1])})
        return out

    def stats(self, symbols):
        self.calls.append(("stats", tuple(symbols)))
        return [{"symbol": s, "implied_vol": self.iv, "hist_vol": None} for s in symbols if s in self.d.k]

    def bars(self, symbol, duration="1 Y", bar_size="1 day"):
        self.calls.append(("bars", symbol))
        j = self.d.k.get(symbol)
        if j is None:
            return []
        return [{"date": x, "open": float(self.d.o[j, i]), "high": float(self.d.h[j, i]), "low": float(self.d.l[j, i]),
                 "close": float(self.d.c[j, i]), "volume": float(self.d.v[j, i])} for i, x in enumerate(self.d.dates)][-21:]


def main():
    d = trader.Data()
    iv = R.load_iv()
    gw = FakeGW(d, iv[d.dates[-1]])
    out = {}
    # 1. normal live call, with a symbol absent from the cache
    a = R.live(gw, symbols=["SPY", "NBIZ"], data=d)
    out["live"] = a
    # 2. IV close missing for the last cached date -> taken from gw.stats
    orig = R.load_iv
    R.load_iv = lambda path=R.IV_FILE: {k: v for k, v in orig(path).items() if k != d.dates[-1]}
    b = R.live(gw, data=d)
    R.load_iv = orig
    out["live_iv_from_gw"] = {"spy_vol20_pct": b["spy_vol20_pct"], "notes": b["notes"]}
    assert abs(b["spy_vol20_pct"] - a["spy_vol20_pct"]) < 1e-6, "same IV value must give the same forecast"
    # 3. point in time: live as of an earlier close (truncated cache) uses nothing after it
    d2 = trader.Data(last="20260601")
    c = R.live(None, data=d2)
    out["live_as_of_20260601"] = {k: c[k] for k in ("as_of", "regime", "spy_vol20_pct", "spy_vol20_10_90_pct")}
    # 4. simulator provider: decision sessions in 2025-26
    at = R.sim_provider(d)
    for x in ("20250103", "20250407", "20250410", "20260601", d.dates[-1]):
        out[f"sim_{x}"] = at(d.dates.index(x))
    out["sim_20241231 (before OOS)"] = at(d.dates.index("20241231"))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
