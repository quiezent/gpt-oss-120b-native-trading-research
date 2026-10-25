"""Checks for ml_research/bondfx.py and its use in forecast.py: the group mapping, the base series, and that a bond ETF
row gets the same treatment from sim_provider() and from compute()-style input (and on demand), while stocks are left
alone. No IB Gateway, no EDGAR: the on-demand name is a fake gateway serving TLT's data under a bond fund's name.

    .venv\\Scripts\\python.exe ml_research\\bondfx_test.py
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml_research"))
import forecast  # noqa: E402
import ml  # noqa: E402
import trader  # noqa: E402
from ml_research import bondfx, events, risk, vol2  # noqa: E402

FUNDS = {"TLT": "Treasury", "IEF": "Treasury", "SHY": "Treasury", "TIP": "Treasury", "LQD": "credit", "HYG": "credit",
         "UUP": "FX", "FXE": "FX"}


def check_groups():
    assert bondfx.GROUPS == FUNDS
    for s, g in FUNDS.items():
        assert bondfx.group(s) == g and bondfx.group(s, "WHATEVER", "COMMON") == g, s
    etf = trader.etfs()
    assert set(FUNDS) <= etf and set(FUNDS) <= set(trader.universe())
    for s in ("SPY", "AAPL", "GLD", "XLF", "EEM", "SHYG"):           # SHYG is not listed: it needs a name and a type
        assert bondfx.group(s) is None, s
    for name in ("ISHARES ULTRA SHORT DURATION BOND ETF", "JPMORGAN ULTRA-SHORT INCOME ETF", "ISHR IBX USD HIYLD CB ETF-UI",
                 "ISHARES 20+ YEAR TREASURY BD", "ISHARES 0-3 MONTH TREASURY BOND", "INVESCO DB US DOLLAR INDEX BULLISH",
                 "VANGUARD TOTAL BOND MARKET"):
        assert bondfx.group("ZZBD", name[:28], "ETF") == bondfx.ONDEMAND, name           # IBKR cuts names to 28
        assert bondfx.group("ZZBD", name[:28], "COMMON") is None, name                   # a company, not a fund
    for name in ("SPDR EURO STOXX 50 ETF", "PROSHARES ULTRAPRO QQQ", "VANECK SEMICONDUCTOR ETF"):
        assert bondfx.group("ZZEQ", name[:28], "ETF") is None, name
    # amendment of 25 Oct: leveraged and inverse funds keep the leveraged flag, even when the name says Treasury or high
    # yield (TMF, TMV, TBT, UBT, SJB), and currency-hedged stock funds are not currency funds; hedged bond funds still are
    for name in ("DIREXION DAILY 20+ YR TREAS BULL 3X", "DIREX DAIL 20+ TREAS BU 3X ET-USD", "PROSH ULTRASHORT 20+Y TREAS",
                 "PROSHARES ULTRA 20+ YR TREAS", "PROSHARES SHORT HIGH YIELD", "DRX DLY 7-10 YR TREAS BEAR 3X"):
        assert bondfx.BOND_FX.search(name[:28]), name                    # the name has a bond term...
        assert bondfx.group("ZZLV", name[:28], "ETF") is None, name      # ...but the fund is leveraged or inverse
        f = forecast._flags({"row": {}, "symbol": "ZZLV", "long_name": name[:28], "stock_type": "ETF"})
        assert "leveraged" in f and "bond" not in f, (name, f)
    for name in ("ISHARES CURRENCY HEDGED MSCI EAFE ETF", "ISHARES CURRENCY HEDGED MSCI JAPAN ETF",
                 "ISHR CURRENCY HDG MSCI GERMANY"):
        assert bondfx.BOND_FX.search(name[:28]) and bondfx.group("ZZHE", name[:28], "ETF") is None, name
        f = forecast._flags({"row": {}, "symbol": "ZZHE", "long_name": name[:28], "stock_type": "ETF"})
        assert "bond" not in f and "leveraged" not in f, (name, f)
    for name in ("VANGUARD TOTAL INTL BD IDX ETF", "ISHARES CORE INTL AGGREGATE BOND", "ISHR INT RATE HDG HY BOND",
                 "WISDOMTREE EMERGING CURRENCY STRATEGY", "INVESCO DB US DOLLAR INDEX BEARISH"):
        assert bondfx.group("ZZBD", name[:28], "ETF") == bondfx.ONDEMAND, name
    assert forecast.BOND_FX is bondfx.BOND_FX and forecast.LEVERAGED is bondfx.LEVERAGED
    dec = bondfx.info()["decision"]
    assert set(dec) == {"Treasury", "credit", "FX", bondfx.ONDEMAND}, dec
    assert all(set(v) == set(bondfx.TARGETS) for v in dec.values()), dec


def check_base(d):
    """own_base at every close equals the research's naive series (risk.rmean of the range, ml.py's 20-session vol)."""
    for s in ("TLT", "SHY", "FXE"):
        j = d.k[s]
        with np.errstate(all="ignore"):
            rng = risk.rmean(((d.h - d.l) / d.c)[j:j + 1], 20)[0]
            vol = np.log(ml.roll(d.lr[j:j + 1], 20, lambda w, axis: np.nanstd(w, axis=axis, ddof=1))[0] * math.sqrt(252))
        for t in list(range(15, 30)) + list(range(len(d.dates) - 40, len(d.dates))):
            o = bondfx.own_base(d.h[j, :t + 1], d.l[j, :t + 1], d.c[j, :t + 1])
            for got, ref in ((o["range1"], rng[t]), (o["vol20"], vol[t])):
                assert (np.isnan(got) and not np.isfinite(ref)) or abs(got - ref) < 1e-12, (s, t, got, ref)


def check_sim_vs_compute(d):
    """The simulator's row and a compute()-style row built from the same vol2 values get the same numbers and note; the
    live fit gives nearly the same; stocks are untouched."""
    i = d.dates.index("20261015") + 1                     # decision session after the last vol2_oos close
    as_of = d.dates[i - 1]
    sim = forecast.sim_provider(d)(i)[1]["rows"]
    raw = vol2.sim_provider(d)(i)[1]
    dec = bondfx.info()["decision"]
    for s, g in FUNDS.items():
        r = raw[s]
        c = {"close": r["close"], "range1_pct": r["range1_pct"], "range1_lo": r["range1_lo"], "range1_hi": r["range1_hi"],
             "absr1_pct": r["absr1_pct"], "vol20_pct": r["vol20_pct"], "note": ""}          # as compute() builds a row
        k = d.k[s]
        own = bondfx.own_base(d.h[k, :i], d.l[k, :i], d.c[k, :i])
        live = dict(c)
        forecast._bondfx(c, g, own, as_of)
        keys = ("range1_pct", "range1_lo", "range1_hi", "absr1_pct", "vol20_pct")
        assert all(c[x] == sim[s][x] for x in keys), (s, c, sim[s])
        assert sim[s]["note"].startswith(c["note"]) and c["note"].startswith("bond, credit or currency fund: "), (s, sim[s])
        assert c["absr1_pct"] == r["absr1_pct"]                                          # absr1 is not touched
        if dec[g]["range1"] == "A":
            f = bondfx.fit_at("range1", "A", as_of)
            assert f["from"] <= as_of and abs(c["range1_pct"] - round(own["range1"] * f["k"] * 100, 2)) < 1e-9, (s, c, f)
            assert c["range1_pct"] != r["range1_pct"] and "range1 from its own last 20 sessions" in c["note"]
        if dec[g]["vol20"] == "B":
            f = bondfx.fit_at("vol20", "B", as_of)
            assert abs(c["vol20_pct"] - round(math.exp(math.log(r["vol20_pct"] / 100) + f["k"]) * 100, 1)) < 1e-9, (s, c)
        forecast._bondfx(live, g, own)                                                   # the live fit
        assert live["note"] == c["note"], (live["note"], c["note"])
        for x in ("range1_pct", "vol20_pct"):
            assert abs(live[x] / c[x] - 1) < 0.05, (s, x, live[x], c[x])
    for s in ("AAPL", "SPY", "GLD"):                                                    # no change outside the funds
        assert all(sim[s][x] == raw[s][x] for x in ("range1_pct", "range1_lo", "range1_hi", "vol20_pct")), s
        assert "bond" not in sim[s]["note"], sim[s]
    return {s: sim[s] for s in ("TLT", "LQD", "UUP", "AAPL")}


def check_notes(d):
    """Amendment of 25 Oct: every value apply() sets to None says why in the note, and the pooled models' fallback part
    drops the columns served from the fund's own history or withheld (B keeps it)."""
    dec = bondfx.info()["decision"]
    assert dec["Treasury"] == {"range1": "A", "vol20": "A"} and dec["credit"] == dec["FX"] == {"range1": "A", "vol20": "B"}
    assert dec[bondfx.ONDEMAND] == {"range1": "A", "vol20": "withheld"}, dec
    fb = bondfx.drop_fallback
    assert fb("no implied vol or intraday data: previous model for absr1, vol20", ["vol20"]) == \
        "no implied vol or intraday data: previous model for absr1"
    assert fb("no implied vol: HAR fallback for absr1, vol20; x", ["range1", "vol20"]) == "no implied vol: HAR fallback for absr1; x"
    assert fb("IV history too short: HAR fallback for vol20", ["vol20"]) == ""
    assert fb("intraday history missing: HAR fallback for absr1", ["vol20"]) == "intraday history missing: HAR fallback for absr1"
    assert fb("unusually low volatility (pending acquisition?)", ["vol20"]) == "unusually low volatility (pending acquisition?)"
    # the amendment's examples (simulator, close of 2 Jan 2025)
    at = forecast.sim_provider(d)
    rows = at(d.dates.index("20250102") + 1)[1]["rows"]
    r1 = "bond, credit or currency fund: range1 from its own last 20 sessions, recalibrated; "
    old = "no implied vol or intraday data: previous model for absr1"
    assert rows["TLT"]["note"] == r1 + "vol20 from its own last 20 sessions, recalibrated; " + old, rows["TLT"]
    assert rows["LQD"]["note"] == r1 + "vol20 withheld: no recalibration fitted yet; " + old, rows["LQD"]
    assert rows["LQD"]["vol20_pct"] is None and rows["TLT"]["vol20_pct"] is not None
    # every simulator row of the 8 funds: a None says why, and no fallback part names a column served from the fund's own
    # history or withheld (every session to 10 Mar 2025, then every 10th)
    i0, i1 = d.dates.index("20250102") + 1, d.dates.index("20250310") + 1
    nnone = 0
    for i in list(range(i0, i1)) + list(range(i1, len(d.dates) + 1, 10)):
        got = at(i)
        if not got:
            continue
        for s, g in FUNDS.items():
            r = got[1]["rows"][s]
            parts = r["note"].split("; ")
            for t in bondfx.TARGETS:
                if r[f"{t}_pct"] is None:
                    nnone += 1
                    assert any(p.startswith(f"{t} withheld: ") for p in parts), (got[0], s, r)
                if dec[g][t] == "A" or r[f"{t}_pct"] is None:
                    assert not any(bondfx.FALLBACK.fullmatch(p) and t in p.split(" for ")[-1].split(", ") for p in parts), \
                        (got[0], s, r)
    assert nnone >= 4 * 40, nnone            # credit and FX vol20 before the first B fit (closes 2 Jan-5 Mar 2025)
    # A without 20 sessions of the fund's own data, B with nothing to rescale, and a column withheld by the rule
    j = d.k["LQD"]
    short = bondfx.own_base(d.h[j, -15:], d.l[j, -15:], d.c[j, -15:])
    assert np.isnan(short["range1"]) and np.isnan(short["vol20"])
    row = {"range1_pct": 1.0, "range1_lo": 0.5, "range1_hi": 1.5, "vol20_pct": None, "vol20_lo": None, "vol20_hi": None,
           "note": "no implied vol: HAR fallback for absr1, vol20"}
    c = bondfx.apply(row, "credit", short)
    assert c == "bond, credit or currency fund: range1 withheld: no 20-session history; vol20 withheld: no model forecast " \
                "to rescale", c
    assert row["range1_pct"] is row["range1_lo"] is row["range1_hi"] is None, row
    assert row["note"] == "no implied vol: HAR fallback for absr1", row      # range1 was never in it; vol20 is now na
    full = bondfx.own_base(d.h[j], d.l[j], d.c[j])
    row = {"range1_pct": 1.0, "range1_lo": 0.5, "range1_hi": 1.5, "vol20_pct": 5.0, "vol20_lo": 4.0, "vol20_hi": 6.0,
           "note": "no implied vol: HAR fallback for absr1, vol20"}
    c = bondfx.apply(row, bondfx.ONDEMAND, full)
    assert c == bondfx.clause(bondfx.ONDEMAND) and "vol20 withheld: our models ran too high" in c, c
    assert row["vol20_pct"] is row["vol20_lo"] is row["vol20_hi"] is None and row["range1_pct"] != 1.0, row
    assert row["note"] == "no implied vol: HAR fallback for absr1", row
    row = {"range1_pct": 1.0, "range1_lo": 0.5, "range1_hi": 1.5, "vol20_pct": 5.0, "vol20_lo": 4.0, "vol20_hi": 6.0,
           "note": "no implied vol: HAR fallback for absr1, vol20"}
    bondfx.apply(row, "credit", full)                                        # B rescales the pooled value: it keeps it
    assert row["vol20_pct"] != 5.0 and row["note"] == "no implied vol: HAR fallback for absr1, vol20", row


def check_description():
    """F1: the DESCRIPTION's bond/FX bullet is the text pre-registered in the amendment of 25 Oct."""
    pre = (ROOT / "runs" / "ml" / "research" / "bondfx_prereg.md").read_text(encoding="utf-8")
    amend = pre[pre.index("## Amendment, 25 Oct"):]
    want = [ln[2:] for ln in amend.splitlines() if ln.startswith("> - **Bond, credit and currency ETFs**")]
    assert len(want) == 1 and want[0] in forecast.DESCRIPTION.splitlines(), want


def check_ondemand():
    """A fund outside the cache with a bond fund's IBKR name (TLT's data) is served by _ondemand() with the pooled
    decision (range1 from its own last 20 sessions with the live fit; vol20 as decided) and the on-demand note."""
    from forecast_test import FG
    tmp = Path(tempfile.mkdtemp())
    forecast.ONDEMAND = tmp / "ondemand"
    vol2.LIVECACHE = tmp / "lc.json"
    events.live = lambda gw, syms, base=None, **kw: {"rows": {}}
    forecast._session = lambda gw, syms, iv_hint=None: ("session ranges from 09:30 ET", {})

    class BG(FG):
        names = dict(FG.names, ZZBD=("ISHARES 20+ YEAR TREASURY BD", "ETF"), ZZTM=("DIREXION DAILY 20+ YR TREAS", "ETF"))

        def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", **kw):
            if symbol in ("ZZBD", "ZZTM"):
                if what == "BID_ASK":
                    raise RuntimeError("no bid/ask in the fake")
                self.calls.append(("bars", symbol, bar_size, what, duration, kw.get("timeout")))
                return FG.bars(self, "TLT", duration, bar_size, what)
            return FG.bars(self, symbol, duration, bar_size, what, **kw)
    as_of, fc = forecast.live_provider()
    d = forecast._data(as_of)
    txt = forecast.run(BG(d, vol2.load_iv(d)), {"symbols": ["ZZBD", "ZZTM", "AAPL"]})
    line = [ln for ln in txt.splitlines() if ln.startswith("ZZBD ")][0]
    lev = [ln for ln in txt.splitlines() if ln.startswith("ZZTM ")][0]       # a leveraged Treasury fund (amendment)
    assert "leveraged, inverse or volatility product" in lev and "bond, credit or currency fund" not in lev, lev
    assert not any(bondfx.FALLBACK.fullmatch(p) and "vol20" in p.split(" for ")[-1] for p in line.split("; ")), line
    tok = line.split(" ")
    dec = bondfx.info()["decision"][bondfx.ONDEMAND]
    j = d.k["TLT"]
    own = bondfx.own_base(d.h[j, -252:], d.l[j, -252:], d.c[j, -252:])
    if dec["range1"] == "A":
        f = bondfx.fit_at("range1", "A")
        p = [round(float(x) * 100, 2) for x in bondfx.serve(own["range1"], f, "range1")]
        h = lambda x: f"{round(x / 2, 2):g}"
        assert tok[2] == f"{h(p[0])}({h(p[1])}-{h(p[2])})", (tok[2], p)
    if dec["vol20"] == "withheld":
        assert tok[4] == "na", tok
    assert bondfx.clause(bondfx.ONDEMAND) in line and "(validated on 8 listed funds, not on this one)" in line, line
    assert "outside the validated universe (an ETF" in line and "leveraged" not in line, line
    aapl = [ln for ln in txt.splitlines() if ln.startswith("AAPL ")][0]
    assert "bond" not in aapl, aapl
    return line


def main():
    check_groups()
    check_description()
    d = trader.Data()
    check_base(d)
    rows = check_sim_vs_compute(d)
    check_notes(d)
    line = check_ondemand()
    for s, r in rows.items():
        print(s, {k: r.get(k) for k in ("range1_pct", "range1_lo", "range1_hi", "vol20_pct", "note")})
    print(line)
    print("ok")


if __name__ == "__main__":
    main()
