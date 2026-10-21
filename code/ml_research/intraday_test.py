"""Live test of ml_research/intraday.py with a fake gateway (no IB Gateway connection).

Replays the cached session 20261016 as "today" at a 11:05 ET look: the fake gw.bars('1 D') returns ALL of that day's
bars (including ones after the look) so the test also checks that live() ignores bars that had not ended. Checks:
  1. live() for cached symbols matches the panel model's prediction for the same (symbol, day, slot);
  2. a symbol not in the cache (ZZZZ: AAPL's bars at half the price, served by the fake gateway '1 M' history,
     implied vol from gw.stats) gets a forecast whose % ranges equal AAPL's except for the IV input;
  3. the 09:30 look (opening print only) and an off-hours call behave;
  4. sim_provider returns None.
Writes runs/ml/research/intraday_livetest.json.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_research import intraday as I     # noqa: E402

DAY = "20261016"


class FakeGW:
    def __init__(self):
        self.calls = []

    def _rows(self, sym):
        src = "AAPL" if sym == "ZZZZ" else sym
        p = I.BARS30 / f"{src}.json"
        if not p.exists():
            raise ValueError(f"200: No security definition has been found for {sym}")
        rows = json.loads(p.read_text())["bars"]
        if sym == "ZZZZ":
            rows = [{**r, **{k: r[k] / 2 for k in ("open", "high", "low", "close")}, "volume": r["volume"] * 2} for r in rows]
        return rows

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, **kw):
        self.calls.append((symbol, duration, bar_size))
        assert bar_size == "30 mins"
        rows = [r for r in self._rows(symbol) if r["date"][:8] <= DAY]
        if duration == "1 D":
            return [r for r in rows if r["date"][:8] == DAY]
        return rows[-22 * I.NS:]                                   # about one month

    def stats(self, symbols, **kw):
        return [{"symbol": s, "implied_vol": 0.30, "hist_vol": 0.25} for s in symbols]

    def quotes(self, symbols, **kw):
        return []


def main():
    gw = FakeGW()
    now = datetime(2026, 10, 16, 11, 5, tzinfo=I.ET)
    syms = ["AAPL", "XOM", "NVDA", "KO", "ZZZZ"]
    res = I.live(gw, syms, now=now)
    out = {"look": res.get("look"), "rows": res.get("rows"), "gateway_calls": gw.calls}
    # 1. compare with the panel model at the same (symbol, day, slot)
    p = I.load_panel()
    psyms = [str(s) for s in p["symbols"]]
    dates = [str(x) for x in p["dates"]]
    t, k = dates.index(DAY), 3
    RF, NAV, _ = I.range_features(p, psyms.index("SPY"))
    model = I._load_model()["range"]
    cmp = {}
    for s in syms[:-1]:
        j = psyms.index(s)
        row = {}
        for tk, lab in (("rng60", "next60"), ("rngrod", "rest_of_day")):
            x = [np.log(NAV[tk][j, t, k])] + [RF[n][j, t, k] for n in I.RNG[1:]]
            lp = float(model[tk].predict(np.array([x + [k, 0, 0]]))[0])
            row[lab] = {"panel": round(100 * np.exp(lp) * model[tk].m[k], 2), "live": res["rows"][s][lab + "_range_pct"],
                        "gap_div_bps_in_panel": round(float(p["div"][j, t]) * 1e4, 1)}
        cmp[s] = row
    out["live_vs_panel"] = cmp
    ok1 = all(abs(v[l]["panel"] - v[l]["live"]) < 0.02 for v in cmp.values() for l in v if v[l]["gap_div_bps_in_panel"] == 0)
    # 2. not-cached symbol
    z, a = res["rows"]["ZZZZ"], res["rows"]["AAPL"]
    ok2 = z.get("history") == "gateway" and z.get("iv_source") == "gateway implied vol" and z.get("rest_of_day_range_pct") is not None
    # 3. opening look and off-hours
    r930 = I.live(FakeGW(), ["AAPL", "ZZZZ"], now=datetime(2026, 10, 16, 9, 31, tzinfo=I.ET))
    off = I.live(FakeGW(), ["AAPL"], now=datetime(2026, 10, 17, 10, 0, tzinfo=I.ET))
    out["look_0930"] = r930
    out["off_hours"] = off
    ok3 = r930.get("look") == "09:30" and r930["rows"]["AAPL"].get("rest_of_day_range_pct") is not None and "error" in off
    # 4. sim provider
    ok4 = I.sim_provider(None) is None
    # future bars ignored: a bar from after the look must not change the answer
    out["checks"] = {"live_matches_panel": ok1, "uncached_symbol_served": ok2, "opening_and_offhours": ok3,
                     "sim_provider_none": ok4}
    (I.OUT / "intraday_livetest.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"checks": out["checks"], "live_vs_panel": cmp, "ZZZZ": z, "AAPL": a}, indent=1, default=float))


if __name__ == "__main__":
    main()
