"""Offline tests of ML3 (forecast3.py): no gateway. A stub gw serves canned SPY bars and quotes; ML2's network calls
(EDGAR, the macro line, session ranges) are replaced by canned answers, and ML3's cache goes to a temporary folder.

    .venv\\Scripts\\python.exe ml3_test.py          # prints ALL OK and exits 0, else exits 1

Covers: gap1 rows and legend, the earnings and macro multipliers, the intraday market line (and size_scale_now when it
ships), the pre-open path and a failed quote, ML2's output unchanged inside ML3's (simulator and live paths), and that
compute() writes only into ML3's own folder.
"""

import copy
import json
import math
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import forecast             # noqa: E402
import forecast3 as f3      # noqa: E402
from ml_research import events, macro_feats    # noqa: E402

ET = ZoneInfo("America/New_York")
AS_OF = "20261106"
NIGHT = ["2026-11-09", "2026-11-10"]
MLINE = {"spy_vol20_pct": 17.6, "spy_vol20_10_90_pct": [13.1, 24.9], "spy_vol5_pct": 15.8, "spy_vol5_10_90_pct": [9.7, 26.6],
         "typical_spy_vol_pct": 12.0, "size_scale_vs_typical": 0.68, "macro": "macro: today none"}


def ml2_row(close, r1):
    return {"close": close, "range1_pct": r1, "range1_lo": r1 * 0.6, "range1_hi": r1 * 1.5, "absr1_pct": r1 / 2,
            "vol20_pct": 30.0, "note": "", "spread": 2.0, "spread_lo": 1.5, "spread_hi": 3.0, "adv_usd": 5e9, "sigma_d": 0.02}


ML2 = {"market": MLINE, "rows": {"AAPL": ml2_row(270.0, 2.0), "AMD": ml2_row(250.0, 5.0), "QCOM": ml2_row(160.0, 3.0),
                                 "SPY": ml2_row(670.0, 1.2), "BRK B": ml2_row(480.0, 1.5), "ZZZQ": ml2_row(10.0, 4.0)}}
FC3 = {"version": f3.VERSION, "as_of": AS_OF, "night": NIGHT, "macro": [], "market": {"lmkt5": 0.1, "lv5": 0.2, "long": 0},
       "rows": {"AAPL": [math.log(0.45), math.log(0.5)], "AMD": [math.log(1.4), math.log(1.3)], "QCOM": [math.log(0.9), math.log(0.8)],
                "SPY": [math.log(0.3), math.log(0.35)], "BRK B": [math.log(0.4), math.log(0.4)]},
       "spy": {"g": [0.4, 0.67, 0.8, 0.87, 0.93, 0.96, 0.99, 1.0, 1.06, 1.06, 1.08, 1.11, 1.11]}}
EV = {"AMD": {"calendar_day": "2026-11-09", "calendar_when": "after close", "calendar_sessions": 0, "calendar_reported": False},
      "QCOM": {"next_earn_day": "2026-11-10", "next_earn_when": "before open", "next_earn_sessions": 1, "next_earn_overdue": False},
      "AAPL": {"calendar_day": "2026-11-12", "calendar_when": "after close", "calendar_sessions": 3, "calendar_reported": False}}
SYMS = ["AAPL", "AMD", "QCOM", "SPY", "BRK B", "ZZZQ", "NOPE"]


def spy_bars(day, prev, upto, base=100.0):
    """The previous session's 13 bars closing at `base`, and today's bars through slot `upto` (the last one partial)."""
    rows = []
    for k, s in enumerate(f3.SLOTS):
        c = base * (1 + 0.0005 * (k - 12))
        rows.append({"date": f"{prev} {s}:00 US/Eastern", "open": c, "high": c * 1.001, "low": c * 0.999, "close": c})
    px = base * 0.99                                   # opens 1% down, then drifts 0.5% lower a bar
    for k, s in enumerate(f3.SLOTS[:upto + 1]):
        o, c = px, px * 0.995
        rows.append({"date": f"{day} {s}:00 US/Eastern", "open": o, "high": o * 1.002, "low": c * 0.998, "close": c})
        px = c
    return rows


class GW:
    """Live-style stub: SPY's 30-minute bars and a quote; no daily bars for outside names."""

    def __init__(self, bars=None, quote=None, quote_fails=False, daily=None):
        self._bars, self._quote, self._fail, self.calls, self._daily = bars or [], quote, quote_fails, [], daily or {}

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, **kw):
        self.calls.append((symbol, duration, bar_size))
        if (symbol, bar_size) == ("SPY", "30 mins"):
            return list(self._bars)
        return list(self._daily.get(symbol, [])) if (bar_size, what) == ("1 day", "ADJUSTED_LAST") else []

    def quotes(self, symbols, primary="", timeout=12.0):
        if self._fail:
            raise RuntimeError("no market data")
        return [dict(self._quote, symbol="SPY")]

    def details(self, s):
        return {}


class SimGW(GW):
    def forecast_provider(self):
        return AS_OF, copy.deepcopy(ML2)

    def session_ranges(self, syms):
        return "session ranges from 09:30 ET", {}


def strip(text):
    """ML3's text without what ML3 adds: its three kinds of line, the legend's gap1 column and each row's gap1 token."""
    out = []
    for ln in text.split("\n"):
        if ln.startswith(("gap1 ±(10-90):", "gap1 events tonight", "market now:")):
            continue
        if ln.startswith("sym close range1"):
            ln = ln.replace(" gap1±%(10-90)", "", 1)
        else:
            s = max((s for s in SYMS if ln.startswith(s + " ")), key=len, default=None)
            if s and not ln[len(s) + 1:].startswith("not covered"):
                p = ln[len(s) + 1:].split(" ")
                ln = " ".join([s] + p[:3] + p[4:])
        out.append(ln)
    return "\n".join(out)


def row(text, s):
    return next(ln for ln in text.split("\n") if ln.startswith(s + " ") and not ln.startswith("sym "))


def token(text, s):
    return row(text, s)[len(s) + 1:].split(" ")[3]


def expect(M, s, earn=None, macro=False):
    r, mk = FC3["rows"][s], FC3["market"]
    X = np.array([[r[0], r[1], mk["lmkt5"], mk["lv5"], mk["long"]]])
    q = f3.quantiles(M, X, f3.multiplier(M, earn, macro)[None])[0]
    return f"{f3._num(q[1])}({f3._num(q[0])}-{f3._num(q[2])})"


def live_patches(now):
    """ML2's live path without the network: its cache, EDGAR and the macro line are canned, the clock fixed."""
    forecast.live_provider = lambda: (AS_OF, copy.deepcopy(ML2))
    forecast._today_et = lambda: now.date()
    forecast._session = lambda gw, syms, iv_hint=None: ("session ranges from 09:30 ET", {})
    events.live = lambda gw, covered, base=None: {"rows": {s: dict(EV[s]) for s in covered if s in EV}}
    macro_feats.macro_today = lambda: "macro: today none"
    f3._now = lambda: now


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"ok  {name}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ml3_test_"))
    f3.OUT, f3.ONDEMAND = tmp / "forecast3", tmp / "forecast3" / "ondemand"
    ml2_before = sorted(p.name for p in forecast.OUT.rglob("*"))
    M = f3.load_model()
    args = {"symbols": SYMS, "notional_usd": 5000}

    # 1. simulator path: gap1 rows, legend, ML2 unchanged, no market-now line
    f3._MEM.clear()
    f3._MEM[AS_OF] = copy.deepcopy(FC3)
    gw = SimGW()
    out, ml2 = f3.run(gw, args), forecast.run(SimGW(), args)
    check("sim: ML2's text unchanged inside ML3's", strip(out) == ml2, f"\n{strip(out)}\n----\n{ml2}")
    cols = next(ln for ln in out.split("\n") if ln.startswith("sym close range1"))
    check("legend column after absr1", "absr1±% gap1±%(10-90) vol20%" in cols, cols)
    check("legend line", any(ln.startswith("gap1 ±(10-90):") for ln in out.split("\n")))
    for s in ("AAPL", "AMD", "QCOM", "SPY", "BRK B"):
        check(f"gap1 row {s}", token(out, s) == expect(M, s), f"{token(out, s)} vs {expect(M, s)}")
    check("gap1 na without bars", token(out, "ZZZQ") == "na", row(out, "ZZZQ"))
    check("not-covered row untouched", "NOPE not covered" in out)
    check("sim: no market-now line", "market now:" not in out)
    med, lo, hi = [float(x) for x in token(out, "AMD").replace("(", " ").replace("-", " ").rstrip(")").split()]
    check("gap1 is a size: positive, sorted, no sign", 0 < lo < med < hi and "+" not in token(out, "AMD"), token(out, "AMD"))

    # 2. events: the Alpha Vantage calendar (AMD tonight after the close), the E1 estimate (QCOM before the next open),
    #    another day (AAPL: none), and a CPI morning for every name
    now = datetime(2026, 11, 9, 12, 5, tzinfo=ET)
    live_patches(now)
    doc = copy.deepcopy(FC3)
    doc["macro"] = ["CPI 08:30 ET"]
    f3._MEM.clear()
    f3._MEM[AS_OF] = doc
    gw = GW(bars=spy_bars("20261109", "20261106", 5))
    out = f3.run(gw, args)
    ml2 = forecast.run(GW(bars=spy_bars("20261109", "20261106", 5)), args)
    check("live: ML2's text unchanged inside ML3's", strip(out) == ml2, f"\n{strip(out)}\n----\n{ml2}")
    check("AMD: calendar multiplier", token(out, "AMD") == expect(M, "AMD", "earn", True), token(out, "AMD"))
    check("QCOM: estimate multiplier", token(out, "QCOM") == expect(M, "QCOM", "earn_e1", True), token(out, "QCOM"))
    check("AAPL: earnings on another day, macro only", token(out, "AAPL") == expect(M, "AAPL", None, True), token(out, "AAPL"))
    base = float(expect(M, "AMD").split("(")[0])
    check("earnings raise the median", float(token(out, "AMD").split("(")[0]) > 3 * base)
    ev = next((ln for ln in out.split("\n") if ln.startswith("gap1 events tonight")), "")
    check("events line", "AMD earnings (Alpha Vantage calendar)" in ev and "QCOM earnings (our estimate)" in ev
          and "CPI 08:30 ET before the open (every name)" in ev and "AAPL" not in ev and NIGHT[0] in ev, ev)
    check("legend: outside names get no earnings check",
          any("Names outside the validated universe get no earnings check" in ln for ln in out.split("\n")))
    live_patches(datetime(2026, 11, 10, 10, 5, tzinfo=ET))     # Tuesday, but ML3's document is still Monday's night
    f3._MEM.clear()
    f3._MEM[AS_OF] = copy.deepcopy(doc)
    stale = f3.run(GW(bars=spy_bars("20261110", "20261109", 1)), args)
    check("a stale night: gap1 unavailable, no events line, no multipliers",
          f"gap1 unavailable (its forecasts are for the night after the {NIGHT[0]} close)" in stale
          and token(stale, "AMD") == "na" and not any(ln.startswith("gap1 events tonight") for ln in stale.split("\n")),
          "\n".join(stale.split("\n")[:4]))
    live_patches(now)
    f3._MEM.clear()
    f3._MEM[AS_OF] = doc
    check("_earn: wrong night", f3._earn("next earnings 2026-11-10 (after close; Alpha Vantage calendar)", NIGHT) is None)
    check("_earn: this morning is not tonight", f3._earn("next earnings est. 2026-11-09 (before open)", NIGHT) is None)
    check("multiplier: macro alone", np.allclose(f3.multiplier(M, None, True), M["mult"]["macro"]))

    # 3. intraday refresh at the 12:05 look (k = 5: bars 09:30..11:30 complete; the 12:00 bar is partial and ignored)
    mline = next(ln for ln in out.split("\n") if ln.startswith("market:"))
    nl = out.split("\n")[out.split("\n").index(mline) + 1]
    tb = f3.by_day(spy_bars("20261109", "20261106", 5))["20261109"]
    x = f3.so_far(tb, 5, math.log(0.99))
    px = tb["11:30"]["close"]
    want = f"market now: SPY {(px / 100 - 1) * 100:+.2f}% since the last close at the 12:00 look, range since the close {x * 100:.2f}%"
    check("market-now line after the market line", nl == want, f"{nl!r} vs {want!r}")
    check("no size_scale_now while it doesn't ship", "size_scale_now" not in out)
    check("SPY bars request differs from intraday.live's (IBKR refuses identical requests within 15 s)",
          ("SPY", "2 D", "30 mins") in gw.calls)
    f3.NOW_SHIPS = True
    try:
        out2 = f3.run(GW(bars=spy_bars("20261109", "20261106", 5)), args)
        s_now, z = f3.scale_now(0.68, 17.6 / 100 / math.sqrt(252), x, FC3["spy"]["g"][5], 5)
        check("size_scale_now when it ships", f"size_scale_now {s_now:.2f} (morning 0.68)" in out2 and z > 1 and s_now < 0.68,
              next(ln for ln in out2.split("\n") if ln.startswith("market now:")))
        check("09:30 look: the gap alone", abs(f3.so_far(tb, 0, math.log(0.99)) - abs(math.log(0.99))) < 1e-12)
        check("weights rise through the day", f3.weight(0) < 0.15 and f3.weight(12) > 0.85)
        check("a quiet open raises the scale at most 0.25^-w", f3.scale_now(1.0, 0.01, 1e-6, 1.0, 0)[0] <= 0.25 ** -f3.weight(0) + 1e-9)
    finally:
        f3.NOW_SHIPS = False

    # 4. before the open: the pre-market quote, and a failed quote
    live_patches(datetime(2026, 11, 9, 9, 5, tzinfo=ET))
    out = f3.run(GW(quote={"last": 101.0, "close": 100.0, "bid": 100.9, "ask": 101.1}), args)
    check("pre-open quote", "market now: SPY +1.00% since the last close (pre-market quote, delayed about 15 minutes)" in out,
          next(ln for ln in out.split("\n") if ln.startswith("market now:")))
    out = f3.run(GW(quote_fails=True), args)
    check("pre-open quote failed", "market now: SPY pre-market quote unavailable (RuntimeError)" in out)
    out = f3.run(GW(quote={"error": "no data"}), args)
    check("pre-open quote error row", "market now: SPY pre-market quote unavailable (ValueError)" in out)
    f3.NOW_SHIPS = True
    try:
        out = f3.run(GW(quote={"last": 101.0, "close": 100.0}), args)
        check("pre-open size_scale_now is the morning value", "size_scale_now 0.68 (the morning value until 09:30)" in out)
    finally:
        f3.NOW_SHIPS = False
    live_patches(datetime(2026, 11, 8, 10, 0, tzinfo=ET))
    check("no session on a Sunday", "market now: no session today" in f3.run(GW(), args))

    # 5. a name with no cached bars: a year of daily bars from the gateway, kept in ML3's ondemand folder
    rng = np.random.default_rng(1)
    days = [str(x).replace("-", "") for x in np.busday_offset("2026-11-06", -np.arange(280)[::-1], roll="backward")]
    c = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
    o = c * np.exp(rng.normal(0, 0.01, len(days)))
    daily = [{"date": d, "open": float(a), "high": float(max(a, b)), "low": float(min(a, b)), "close": float(b), "volume": 1e6}
             for d, a, b in zip(days, o, c)]
    live_patches(datetime(2026, 11, 9, 12, 5, tzinfo=ET))
    forecast._MEMO.clear()                             # the earlier calls' empty answers for ZZZQ are memoized 15 minutes
    out = f3.run(GW(bars=spy_bars("20261109", "20261106", 5), daily={"ZZZQ": daily}), args)
    l60, l250 = f3._row_from_bars(daily, AS_OF)
    FC3["rows"]["ZZZQ"] = [l60, l250]
    try:
        check("outside name scored from gateway bars", token(out, "ZZZQ") == expect(M, "ZZZQ", None, True), row(out, "ZZZQ"))
    finally:
        del FC3["rows"]["ZZZQ"]
    check("outside name kept per close", (f3.ONDEMAND / AS_OF / "ZZZQ.json").exists())

    # 6. ML2's errors come back as they are
    check("error passthrough", f3.run(SimGW(), {"symbols": []}) == "error: give at least one symbol")

    # 7. compute() on a synthetic close writes only into ML3's folder
    f3.BARS = tmp / "no_bars"
    rng = np.random.default_rng(0)
    n, k = 300, 130

    class D:
        pass
    d = D()
    d.sym = [f"S{i}" for i in range(k - 1)] + ["SPY"]
    d.k, d.etf = {s: j for j, s in enumerate(d.sym)}, {"SPY"}
    days = np.busday_offset("2025-09-01", np.arange(n), roll="forward")
    d.dates = [str(x).replace("-", "") for x in days]
    lc = np.cumsum(rng.normal(0, 0.01, (k, n)), axis=1)
    d.c = 100 * np.exp(lc)
    d.o = d.c * np.exp(rng.normal(0, 0.005, (k, n)))
    as_of, rows = f3.compute(d)
    check("compute: its own cache", (f3.OUT / f"fc_{as_of}.json").exists() and len(rows) == k, f"{as_of} {len(rows)}")
    doc = json.loads((f3.OUT / f"fc_{as_of}.json").read_text(encoding="utf-8"))
    check("compute: market features", all(np.isfinite(doc["market"][x]) for x in ("lmkt5", "lv5")), doc["market"])
    check("ML2's folders untouched", sorted(p.name for p in forecast.OUT.rglob("*")) == ml2_before)
    check("SPEC keeps ML2's name and parameters", f3.SPEC["name"] == "forecast"
          and f3.SPEC["parameters"] == forecast.SPEC["parameters"] and "gap1" in f3.SPEC["description"]
          and forecast.DESCRIPTION in f3.SPEC["description"] and "{track}" not in f3.SPEC["description"])
    check("version", f3.VERSION == "ML3.0" and forecast.VERSION == "ML2.2")


if __name__ == "__main__":
    try:
        main()
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
    print("ALL OK")
    sys.exit(0)
