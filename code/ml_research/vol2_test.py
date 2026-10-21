"""Checks for ml_research/vol2.py live() and sim_provider() with a fake gateway (no IB Gateway connection).

    .venv\\Scripts\\python.exe ml_research\\vol2_test.py     # writes runs/ml/research/vol2_livetest.json
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trader  # noqa: E402
from ml_research import vol2  # noqa: E402


class FakeGW:
    """stats(): decision-time IV (AAPL and ZZIV: AAPL's cached close IV +20%; NVR none; others their cached close IV).
    bars(): for symbols outside the cache, ZZZT = NVR's history (no IV), ZZIV = AAPL's history (daily, IV, 30-minute);
    NOPE has none. Cached symbols get their own IV / 30-minute history from the data files."""

    def __init__(self, d, iv, fail_stats=False):
        self.d, self.iv, self.fail = d, iv, fail_stats
        self.calls = []

    def stats(self, symbols, primary="", seconds=3.0):
        self.calls.append(("stats", list(symbols)))
        if self.fail:
            raise RuntimeError("fake stats failure")
        t = len(self.d.dates) - 1
        out = []
        for s in symbols:
            src = "AAPL" if s == "ZZIV" else s
            if src not in self.d.k or src == "NVR":
                out.append({"symbol": s, "implied_vol": None})
                continue
            v = self.iv[self.d.k[src], t]
            out.append({"symbol": s, "implied_vol": None if not np.isfinite(v) else float(v * (1.2 if src == "AAPL" else 1.0))})
        return out

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", **kw):
        self.calls.append(("bars", symbol, bar_size, what))
        src = {"ZZZT": "NVR", "ZZIV": "AAPL"}.get(symbol, symbol)
        if symbol == "NOPE":
            return []
        last = self.d.dates[-1]
        if what == "OPTION_IMPLIED_VOLATILITY":
            f = ROOT / "data" / "iv" / f"{src}.json"
            b = json.loads(f.read_text(encoding="utf-8"))["bars"] if f.exists() else []
            return [x for x in b if x["date"] <= last][-22:]
        if bar_size == "30 mins":
            f = ROOT / "data" / "bars30" / f"{src}.json"
            b = json.loads(f.read_text(encoding="utf-8"))["bars"] if f.exists() else []
            return [x for x in b if x["date"][:8] <= last][-13 * 22:]
        j = self.d.k[src]
        n = len(self.d.dates)
        return [{"date": x, "open": self.d.o[j, i], "high": self.d.h[j, i], "low": self.d.l[j, i], "close": self.d.c[j, i],
                 "volume": self.d.v[j, i]} for i, x in enumerate(self.d.dates[-252:], start=n - 252)]

    def quotes(self, symbols, **kw):
        return []

    def news(self, *a, **kw):
        return []


def close(a, b, tol=0.02):
    return a is not None and b is not None and abs(a - b) <= tol * abs(b) + 0.02


def main():
    d = trader.Data()
    iv = vol2.load_iv(d)
    vol2.LIVECACHE = Path(tempfile.mkdtemp()) / "vol2_livecache.json"     # keep fake symbols out of the real live cache
    res = {}
    syms = ["AAPL", "SPY", "NVR", "ZZZT", "ZZIV", "NOPE", "MSFT"]
    out = vol2.live(FakeGW(d, iv), symbols=syms, data=d, no_stale_check=True)
    res["with_gw"] = out
    R = out["rows"]
    assert R["NOPE"]["note"].startswith("not covered"), R["NOPE"]
    assert R["AAPL"]["iv_src"].startswith("gateway"), R["AAPL"]
    assert "HAR fallback" in R["NVR"]["note"] or all(v == "har" for v in out["ship"].values()), R["NVR"]
    for t in vol2.TARGETS:      # ZZZT = NVR's bars from the gateway: same HAR fallback numbers (gateway bars cover 1 year)
        assert R["ZZZT"]["model"][t] == R["NVR"]["model"][t], (t, R["ZZZT"], R["NVR"])
        assert R["ZZIV"]["model"][t] == R["AAPL"]["model"][t], (t, R["ZZIV"], R["AAPL"])
        for k in ("_pct", "_lo", "_hi"):
            assert close(R["ZZZT"][t + k], R["NVR"][t + k]), (t, k, R["ZZZT"], R["NVR"])
            assert close(R["ZZIV"][t + k], R["AAPL"][t + k]), (t, k, R["ZZIV"], R["AAPL"])
        for s in ("AAPL", "SPY", "MSFT"):
            r = R[s]
            assert r[t + "_lo"] < r[t + "_pct"] < r[t + "_hi"], (s, t, r)
    # decision-time IV moves the IV-model forecast (AAPL IV +20%)
    out0 = vol2.live(None, symbols=["AAPL"], data=d, no_stale_check=True)
    res["aapl_cache_iv"] = out0["rows"]["AAPL"]
    if any(v != "har" for v in out["ship"].values()):
        t = next(t for t, v in out["ship"].items() if v != "har")
        assert R["AAPL"][t + "_pct"] != out0["rows"]["AAPL"][t + "_pct"]
    # gateway stats failing: falls back to cached IV, says so
    outf = vol2.live(FakeGW(d, iv, fail_stats=True), symbols=["AAPL", "NVR"], data=d, no_stale_check=True)
    assert any("stats failed" in n for n in outf["notes"]) and outf["rows"]["AAPL"]["iv_src"].startswith("cache")
    res["stats_fail_notes"] = outf["notes"]
    # every cached symbol, no gateway
    outa = vol2.live(None, data=d, no_stale_check=True)
    n_iv = sum(1 for r in outa["rows"].values() if r.get("model") and "vol2" in r["model"].values())
    res["all_symbols"] = {"rows": len(outa["rows"]), "rows_using_iv_model": n_iv}
    # sim_provider: strictly the npz column d - 1
    at = vol2.sim_provider(d)
    z = np.load(vol2.OUT / "vol2_oos.npz", allow_pickle=True)
    for x in ("20250103", "20250415", "20260105", d.dates[-1]):
        i = d.dates.index(x)
        got = at(i)
        if got is None:
            res.setdefault("sim", {})[x] = None
            continue
        as_of, rows = got
        assert as_of == d.dates[i - 1]
        j = d.k["AAPL"]
        v = float(z["range1"][j, i - 1]) * 100
        assert abs(rows["AAPL"]["range1_pct"] - round(v, 2)) < 1e-9
        res.setdefault("sim", {})[x] = {"as_of": as_of, "n": len(rows), "AAPL": rows["AAPL"]}
    res["text"] = vol2.text(out)
    (vol2.OUT / "vol2_livetest.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(res["text"])
    print(json.dumps({k: res[k] for k in ("all_symbols", "sim")}, indent=1, default=str))
    print("ok")


if __name__ == "__main__":
    main()
