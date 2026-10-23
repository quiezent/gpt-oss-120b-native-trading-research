"""Checks for forecast.run's look-time rows for names outside the cache, with a fake gateway (no IB Gateway, no EDGAR).

    .venv\\Scripts\\python.exe ml_research\\forecast_test.py

The fake is vol2_test.FakeGW (ZZIV = AAPL's data, ZZZT = NVR's, NOPE has none) plus details(), a short-history ZZSH and a
gateway that refuses BID_ASK (cost falls back to its cold model). Session ranges and the 8-K check are stubbed (they
depend on the clock and the network); the files go to a temp folder.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml_research"))
import forecast  # noqa: E402
import trader  # noqa: E402
from ml_research import events, vol2  # noqa: E402
from vol2_test import FakeGW  # noqa: E402


class FG(FakeGW):
    names = {"ZZIV": ("PROSHARES ULTRAPRO QQQ", "ETF"), "ZZZT": ("SOME BUILDER INC", "COMMON")}

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", **kw):
        self.calls.append(("bars", symbol, bar_size, what, duration, kw.get("timeout")))
        if what == "BID_ASK":
            raise RuntimeError("no bid/ask in the fake")
        if symbol == "ZZSH":
            return FakeGW.bars(self, "ZZZT", duration, bar_size, what)[-30:]
        if symbol not in self.d.k and symbol not in ("ZZIV", "ZZZT", "NOPE"):
            raise RuntimeError("162: HMDS query returned no data")
        return FakeGW.bars(self, symbol, duration, bar_size, what)

    def details(self, s):
        self.calls.append(("details", s))
        n = self.names.get(s, ("X CORP", "COMMON"))
        return {"symbol": s, "long_name": n[0], "stock_type": n[1]}


def main():
    tmp = Path(tempfile.mkdtemp())
    forecast.ONDEMAND = tmp / "ondemand"
    vol2.LIVECACHE = tmp / "lc.json"
    events.live = lambda gw, syms, base=None, **kw: {"rows": {}}
    sess_calls = []

    def fake_session(gw, syms, iv_hint=None):
        sess_calls.append((list(syms), dict(iv_hint or {})))
        return "session ranges at the 10:00 look", {s: (1.0, 0.5, 1.5, 2.0, 1.0, 3.0) for s in syms}
    forecast._session = fake_session
    as_of, fc = forecast.live_provider()
    fcr = fc["rows"]
    d = forecast._data(as_of)
    iv = vol2.load_iv(d)
    out = {}

    fg = FG(d, iv)
    txt = forecast.run(fg, {"symbols": ["AAPL", "ZZIV", "ZZZT", "ZZSH", "NOPE", "../X"]})
    lines = {ln.split(" ")[0]: ln for ln in txt.splitlines()}
    print(txt, "\n")
    assert "stale" not in txt.splitlines()[0], txt.splitlines()[0]
    assert txt.splitlines()[0].startswith(f"as of the close of {as_of[:4]}-")
    cols = [ln for ln in txt.splitlines() if ln.startswith("sym close")][0]
    assert cols == ("sym close range1%(10-90) absr1% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m "
                    "next60%(10-90) rest_of_day%(10-90) note"), cols
    a, z = fcr["AAPL"], fcr["NVR"]
    for sym, ref in (("ZZIV", a), ("ZZZT", z)):
        tok = lines[sym].split(" ")
        assert tok[2] == f"{ref['range1_pct']:g}({ref['range1_lo']:g}-{ref['range1_hi']:g})", (sym, tok[2], ref)
        assert (tok[3], tok[4]) == (f"{ref['absr1_pct']:g}", f"{ref['vol20_pct']:g}"), (sym, tok[3:5], ref)
    assert "outside the validated universe (an ETF" in lines["ZZIV"] and "leveraged" in lines["ZZIV"], lines["ZZIV"]
    assert "outside the validated universe;" in lines["ZZZT"] and "no 8-K earnings correction" in lines["ZZZT"], lines["ZZZT"]
    assert "HAR fallback" in lines["ZZZT"], lines["ZZZT"]
    assert "na na na" in lines["ZZSH"] and "fewer than 50 of the last 60 sessions" in lines["ZZSH"], lines["ZZSH"]
    assert lines["NOPE"].startswith("NOPE not covered"), lines["NOPE"]
    assert lines["../X"].startswith("../X not covered") or "not a symbol" in lines["../X"], lines["../X"]
    files = sorted(p.name for p in (forecast.ONDEMAND / as_of).glob("*"))
    assert files == ["ZZIV.json", "ZZSH.json", "ZZZT.json"], files
    assert not list(tmp.rglob("X.json")), "path traversal"
    daily = [c for c in fg.calls if len(c) == 6 and c[0] == "bars" and c[1] in ("ZZIV", "ZZZT") and c[2] == "1 day" and c[4] == "1 Y"]
    assert daily and all(c[3] == "ADJUSTED_LAST" for c in daily), daily
    assert all(c[5] is not None and c[5] <= forecast.PER_REQ_S for c in fg.calls if len(c) == 6), "uncapped bars"
    assert not any(c[0] == "stats" for c in fg.calls), "decision-time IV asked"
    assert sess_calls[0][0] == ["AAPL"] and sorted(sess_calls[1][0]) == ["ZZIV", "ZZZT"], sess_calls
    assert sess_calls[1][1].get("ZZIV") and "ZZZT" not in sess_calls[1][1], sess_calls[1]
    out["first"] = txt

    fg2 = FG(d, iv)                     # same close, same process: rows from the files, history bars from the memo
    txt2 = forecast.run(fg2, {"symbols": ["ZZIV", "ZZZT"], "notional_usd": 20000})
    l2 = {ln.split(" ")[0]: ln for ln in txt2.splitlines()}
    fetched = [c for c in fg2.calls if len(c) == 6 and (c[3] in ("ADJUSTED_LAST", "OPTION_IMPLIED_VOLATILITY") or
               (c[2] == "30 mins" and c[4] == "1 M")) or c[0] == "details"]
    assert not fetched, fetched
    assert l2["ZZIV"].split(" ")[6] != lines["ZZIV"].split(" ")[6], (l2["ZZIV"], lines["ZZIV"])   # cost moves with size
    out["second"] = txt2

    forecast._MEMO.clear()              # another process: no memo, but the files
    forecast.ONDEMAND_BUDGET_S = 0.0
    fg3 = FG(d, iv)
    txt3 = forecast.run(fg3, {"symbols": ["ZZIV", "MSFTX"]})
    l3 = {ln.split(" ")[0]: ln for ln in txt3.splitlines()}
    assert "outside the validated universe" in l3["ZZIV"], l3["ZZIV"]          # saved row still served
    assert "MSFTX not covered" in l3["MSFTX"] or "time cap" in l3["MSFTX"], l3["MSFTX"]
    assert not (forecast.ONDEMAND / as_of / "MSFTX.json").exists()
    forecast.ONDEMAND_BUDGET_S = 45.0

    class SimGW:                        # the simulator: unchanged ("not covered", no gateway calls)
        calls = []

        def forecast_provider(self):
            return as_of, fc
    txt4 = forecast.run(SimGW(), {"symbols": ["AAPL", "ZZIV"]})
    assert "ZZIV not covered" in txt4 and "outside the validated" not in txt4, txt4

    flag = lambda name, typ="ETF": forecast._flags({"row": {}, "long_name": name[:28], "stock_type": typ})
    for name in ("ISHARES ULTRA SHORT DURATION BOND ETF", "JPMORGAN ULTRA-SHORT INCOME ETF", "ISHR IBX USD HIYLD CB ETF-UI",
                 "ISHARES 20+ YEAR TREASURY BD", "ISHARES 0-3 MONTH TREASURY BOND", "INVESCO DB US DOLLAR INDEX BULLISH"):
        assert "bond" in flag(name) and "leveraged" not in flag(name), (name, flag(name))     # IBKR cuts names to 28
    for name in ("PROSHARES ULTRASHORT QQQ", "PROSHARES ULTRAPRO QQQ", "DIREXION DAILY SEMICONDUCTOR BULL 3X",
                 "PROSHARES SHORT S&P500"):
        assert "leveraged" in flag(name), (name, flag(name))
    for name in ("SPDR EURO STOXX 50 ETF", "VANECK SEMICONDUCTOR ETF", "SS SPDR S&P 500 ETF TRUST-US"):
        assert "bond" not in flag(name) and "leveraged" not in flag(name), (name, flag(name))
    assert "8-K" in flag("CLEVELAND-CLIFFS INC", "COMMON")
    assert forecast._final("ZZ: gateway IV history failed (162: Historical Market Data Service error message:HMDS query "
                           "returned no data: ZZ@SMART Option_Implied_Volatility)")
    assert not forecast._final("ZZ: gateway bars failed (162: Historical data request pacing violation)")
    assert not forecast._final("ZZ: gateway bars failed (No reply from IB Gateway within 7.16201s)")

    # The next day, vol2's live cache holds this name's IV through the previous close only: the close IV must be fetched,
    # not left missing (which fell back to HAR), and today's partial bars must not be cached.
    t = len(d.dates) - 1
    ivfile = ROOT / "data" / "iv" / "AAPL.json"
    import json as _json
    hist = {b["date"][:8]: b["close"] for b in _json.loads(ivfile.read_text())["bars"] if b["date"][:8] <= d.dates[t - 1]}
    vol2.LIVECACHE.write_text(_json.dumps({"iv": {"ZZIV": hist}, "rv": {}}))
    fg5 = FG(d, iv)
    v = vol2.live(fg5, symbols=["ZZIV"], data=d, no_stale_check=True, decision_iv=False)
    assert any(c[0] == "bars" and c[3] == "OPTION_IMPLIED_VOLATILITY" for c in fg5.calls), fg5.calls
    assert v["rows"]["ZZIV"].get("iv_pct") and "HAR fallback" not in v["rows"]["ZZIV"].get("note", ""), v["rows"]["ZZIV"]
    assert v["rows"]["ZZIV"]["vol20_pct"] == a["vol20_pct"], (v["rows"]["ZZIV"], a)
    lc = _json.loads(vol2.LIVECACHE.read_text())
    assert max(lc["iv"]["ZZIV"]) <= d.dates[t] and all(k <= d.dates[t] for k in lc["rv"].get("ZZIV", {})), "partial day cached"
    print("ok")


if __name__ == "__main__":
    main()
