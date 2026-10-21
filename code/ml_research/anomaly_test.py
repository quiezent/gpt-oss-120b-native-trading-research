"""Test anomaly.live() with a fake gateway built from cached daily bars (no IB Gateway connection).

ZZPUMP is a symbol absent from the cache: the fake gateway serves NBIZ's bars under that name, so live() must go
through gw.bars / gw.quotes / gw.stats. NOSUCH is unknown to the gateway too. The intraday case replays the look at
11:00 ET on 15 Oct 2026, when rl1 bought NBIZ and AEHL (quotes from runs/live/2026-10-15/rl1-1100.json).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bars as barsmod                     # noqa: E402
from ml_research import anomaly           # noqa: E402

ET = ZoneInfo("America/New_York")
ALIAS = {"ZZPUMP": "NBIZ"}
LAST_1100 = {"NBIZ": 35.30, "ZZPUMP": 35.30, "AEHL": 12.34}      # the 11:00 ET scanner snapshot rl1 saw


class FakeGW:
    def __init__(self, today, frac):
        self.today, self.frac, self.calls = today, frac, []

    def _bars(self, s):
        doc = barsmod.load(ALIAS.get(s, s))
        if not doc:
            raise ValueError(f"200: No security definition has been found for {s}")
        return doc["bars"]

    def bars(self, symbol, duration="1 Y", bar_size="1 day", **kw):
        self.calls.append(("bars", symbol))
        rows = [r for r in self._bars(symbol) if r["date"] <= self.today]
        return rows[-252:]                                 # includes today's partial bar, as IB does mid-session

    def quotes(self, symbols):
        self.calls.append(("quotes", tuple(symbols)))
        out = []
        for s in symbols:
            try:
                b = [r for r in self._bars(s) if r["date"] <= self.today]
            except ValueError as e:
                out.append({"symbol": s, "error": str(e)})
                continue
            t, y = b[-1], b[-2]
            last = LAST_1100.get(s, t["close"])
            out.append({"symbol": s, "last": last, "bid": last, "ask": last, "open": t["open"], "high": max(t["open"], last),
                        "low": min(t["open"], last), "close": y["close"], "volume": t["volume"] * self.frac})
        return out

    def stats(self, symbols):
        self.calls.append(("stats", tuple(symbols)))
        return [{"symbol": s, "avg_volume": sum(r["volume"] for r in self._bars(s)[-90:]) / 90} for s in symbols]


def main():
    syms = ["NBIZ", "AEHL", "ZZPUMP", "AAPL", "MRNA", "NOSUCH"]
    cases = {"after_close_20261014": (datetime(2026, 10, 14, 20, 0, tzinfo=ET), "20261014", 1.0, False),
             "intraday_20261015_1100": (datetime(2026, 10, 15, 11, 0, tzinfo=ET), "20261015", 0.35, True)}
    out = {}
    for name, (now, today, frac, intraday) in cases.items():
        gw = FakeGW(today, frac)
        res = anomaly.live(gw, syms, intraday=intraday, now=now)
        out[name] = {"result": res, "gateway_calls": [list(c) if isinstance(c, tuple) else c for c in gw.calls]}
        print(name)
        for s, r in res.items():
            print(" ", s, json.dumps(r))
        print("  calls:", gw.calls)
    # sanity: the out-of-cache alias must equal the cached original when both come from the same bars
    a, b = out["after_close_20261014"]["result"]["ZZPUMP"], out["after_close_20261014"]["result"]["NBIZ"]
    assert a["anomaly_pct"] == b["anomaly_pct"] and a["bars_from"] == "gateway" and b["bars_from"] == "cache", (a, b)
    assert "error" in out["after_close_20261014"]["result"]["NOSUCH"]
    assert out["intraday_20261015_1100"]["result"]["ZZPUMP"]["provisional_today"] is True
    # whole-universe mode from the cache (no gateway)
    allu = anomaly.live(None, None)
    top = list(allu.items())[:5]
    print("universe mode:", len(allu), "symbols; top 5:", [(s, r["anomaly_pct"], r["pump_risk_p"], r["reason"]) for s, r in top])
    out["universe_top5"] = dict(top)
    (anomaly.OUT / "anomaly_livetest.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
