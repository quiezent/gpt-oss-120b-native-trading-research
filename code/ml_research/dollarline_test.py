"""F10 (runs/ml/research/dollarline_prereg.md, with Amendment 1): forecast.run's move$ and sleeve% columns on ten rows,
the sleeve_usd defaults and clamp, the wording rules, and the gpt-oss SPEC / MCP tool parity (F12). Offline: the forecast
cache, the 8-K check, the macro line, session ranges and the outside names' gateway work are stubbed, so it needs no
gateway, data files or clock.

    .venv\\Scripts\\python.exe ml_research\\dollarline_test.py

GOLDEN is forecast.run's output on the same stubs at 0973c1c, before the change: every existing line and column must come
out byte-identical, with only the move line and the two columns added.
"""

import asyncio
import copy
import math
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import forecast  # noqa: E402
from ml_research import cost, events, macro_feats  # noqa: E402

AS_OF = "20261023"
BANNED = re.compile(r"\b(upside|gain|profit)", re.I)       # gains, profitable; not "against"
NOT_A_BOUND = re.compile(r"\b(top|edge|bound)", re.I)       # the bracket is range1's 90% value (Amendment 1); not "stop"
SYMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
MARKET = {"spy_vol20_pct": 15.2, "spy_vol20_10_90_pct": [12.1, 19.4], "spy_vol5_pct": 14.0, "spy_vol5_10_90_pct": [10.2, 20.5],
          "typical_spy_vol_pct": 16.0, "size_scale_vs_typical": 1.05}
MACRO = "macro: no scheduled US releases today; next week: GDP Thu 08:30, PCE Fri 08:30"


def _row(close, r1, lo, hi, absr1, vol20, spread=None, adv=None, note=""):
    r = {"close": close, "range1_pct": r1, "range1_lo": lo, "range1_hi": hi, "absr1_pct": absr1, "vol20_pct": vol20, "note": note}
    if spread:
        r.update(spread=spread, spread_lo=round(spread * 0.75, 2), spread_hi=round(spread * 1.4, 2), adv_usd=adv, sigma_d=0.02)
    return r


# The cached rows (8) as compute() writes them: range1 is the full high-low span in %, the tool prints half of it.
ROWS = {
    "AAA": _row(100.0, 2.48, 1.7, 3.76, 1.6, 28, 2.0, 9e8),                                  # the prereg's example
    "BBB": _row(54.2, 8.0, 5.0, 12.0, 3.1, 45, 4.0, 2.5e8,
                "8-K 2026-10-22 16:05 ET items 2.02, 9.01: range1 corrected for it (uncorrected ±1.6)"),
    "CCC": _row(210.5, 2.0, 1.4, None, 1.2, 22, 1.5, 1.2e9),                                 # no 90% value
    "DDD": _row(35.0, None, None, None, None, None, 6.0, 4e7),                               # no vol
    "EEE": _row(7000.0, 3.0, 2.0, 4.4, 1.9, 27, 3.0, 3e8),                                   # one share above notional
    "FFF": _row(82.3, 0.2, 0.1, 0.32, 0.15, 3.1, 1.2, 2e9),                                  # bond-ETF-sized range
    "GGG": _row(18.7, 4.0, 2.6, 6.2, 2.5, 52),                                               # no cost row
    "JJJ": _row(31.4, 21.8, 14.0, 32.6, 9.5, 95, 8.0, 6e8),                                  # MRNA's 22 Oct size
}
HHH = _row(12.4, 5.6, 3.6, 8.4, 3.4, 61, note="outside the validated universe; no 8-K earnings correction")
SPAN = {s: (r["range1_pct"], r["range1_hi"]) for s, r in dict(ROWS, HHH=HHH).items()}
MOVE = {"AAA": "±62(94) ±0.21(0.31)", "BBB": "±200(300) ±0.67(1)", "CCC": "±50(na) ±0.17(na)", "DDD": "na na",
        "EEE": "±75(110) ±0.25(0.37)", "FFF": "±5(8) ±0.02(0.03)", "GGG": "±100(155) ±0.33(0.52)",
        "HHH": "±140(210) ±0.47(0.7)", "III": "na na", "JJJ": "±545(815) ±1.82(2.72)"}
LINE = ("move$ ±(90) and sleeve% ±(90): a move of one range1 ± value, up or down, on a ${n:,.0f} order, in $ and as % of a "
        "${s:,.0f} sleeve; in brackets, the same for range1's 90% value. Neither is a loss limit: held from the last close, a "
        "long or a short was at some point down by more than the bracketed $ in 23-25% of next sessions.")
COLS = "move$±(90) sleeve%±(90)"
NOTIONAL_DESC = "Order size, $, for the cost and move$ columns (default 3,000)."
SLEEVE_DESC = "Your sleeve's value, $, for the sleeve% column (default 30,000, a sleeve's starting value)."


class LiveGW:                       # the live path: no forecast_provider; every gateway use is stubbed below
    pass


class SimGW:                        # the simulator path
    def forecast_provider(self):
        return AS_OF, {"market": dict(MARKET, macro=MACRO), "rows": copy.deepcopy(ROWS)}


def _outside(gw, names, rows, notional, as_of, snote, sess, t0):
    """As forecast._outside returns: live cost rows for both outside names, a risk row scored on demand for HHH only."""
    assert names == ["HHH", "III"], names
    costs = {s: {"symbol": s, "price": p, **cost.order_cost(p, max(1, round(notional / p)), 9.0, 6.0, 14.0, adv, 0.03, 0.03)}
             for s, p, adv in (("HHH", 12.4, 3.2e7), ("III", 23.1, 1.1e8))}
    rows["HHH"] = copy.deepcopy(HHH)
    return costs, {}


def stub(session):
    forecast.live_provider = lambda: (AS_OF, {"market": dict(MARKET), "rows": copy.deepcopy(ROWS)})
    forecast._prev_session = lambda today: AS_OF
    forecast._outside = _outside
    events.live = lambda gw, syms, base=None, **kw: {"rows": {}}
    macro_feats.macro_today = lambda *a, **k: MACRO
    forecast._session = ((lambda gw, syms, iv_hint=None: ("session ranges at the 10:00 look",
                                                           {s: (1.0, 0.5, 1.5, 2.0, 1.0, 3.0) for s in syms}))
                         if session else (lambda gw, syms, iv_hint=None: ("session ranges from 09:30 ET", {})))


def outputs(sleeve=None):
    """forecast.run on the stubs: the ten rows live, live during the session, in the simulator, AAA alone, and AAA at the
    default notional."""
    a = {"symbols": SYMS, "notional_usd": 5000} | ({"sleeve_usd": sleeve} if sleeve is not None else {})
    out = {}
    stub(False)
    out["live"] = forecast.run(LiveGW(), a)
    out["aaa"] = forecast.run(LiveGW(), dict(a, symbols=["AAA"]))
    out["default"] = forecast.run(LiveGW(), {k: v for k, v in a.items() if k != "notional_usd"} | {"symbols": ["AAA"]})
    out["sim"] = forecast.run(SimGW(), a)
    stub(True)
    out["session"] = forecast.run(LiveGW(), a)
    return out


# forecast.run's output on these stubs at 0973c1c (before the change)
GOLDEN = {
    'live': (
        'as of the close of 2026-10-23; session ranges from 09:30 ET\n'
        'market: SPY vol20 15.2% (10-90 12.1-19.4) vol5 14.0% (10-90 10.2-20.5) typical 16.0% size_scale 1.05\n'
        'macro: no scheduled US releases today; next week: GDP Thu 08:30, PCE Fri 08:30\n'
        'cost_bps: round trip (spread, impact on entry and exit, both commissions) for a $5,000 order per symbol\n'
        'No column forecasts direction: the expected gain from direction is 0. The ± columns are moves up OR down: range1, next60 and rest_of_day are half the expected high-low span (a band around the price; no overnight gap), absr1 the expected size of the close-to-close move (gap included, so usually larger).\n'
        'sym close range1±%(10-90) absr1±% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m note\n'
        'AAA 100 1.24(0.85-1.88) 1.6 28 2(1.5-2.8) 6.5(5.7-7.7) A 900\n'
        'BBB 54.2 4(2.5-6) 3.1 45 4(3-5.6) 8.9(7.4-11.4) B 250 8-K 2026-10-22 16:05 ET items 2.02, 9.01: range1 corrected for it (uncorrected ±1.6)\n'
        'CCC 210.5 1(0.7-na) 1.2 22 1.5(1.1-2.1) 5.9(5.2-6.9) A 1200\n'
        'DDD 35 na(na-na) na na 6(4.5-8.4) 12.2(9.4-16.9) C 40\n'
        'EEE 7000 1.5(1-2.2) 1.9 27 3(2.2-4.2) 6.8(5.5-9) B 300\n'
        'FFF 82.3 0.1(0.05-0.16) 0.15 3.1 1.2(0.9-1.7) 5.5(5-6.3) A 2000\n'
        'GGG 18.7 2(1.3-3.1) 2.5 52 na na na na\n'
        'HHH 12.4 2.8(1.8-4.2) 3.4 61 9(6-14) 20.8(15.6-29.6) C 32 outside the validated universe; no 8-K earnings correction; daily volume under $50M (below 95% of the validated names)\n'
        'JJJ 31.4 10.9(7-16.3) 9.5 95 8(6-11.2) 12.6(10.2-16.4) B 600\n'
        'III 23.1 na na na 9(6-14) 15.3(11.1-22.4) B 110 not in the risk universe; cost from live data (rough)'
    ),
    'aaa': (
        'as of the close of 2026-10-23; session ranges from 09:30 ET\n'
        'market: SPY vol20 15.2% (10-90 12.1-19.4) vol5 14.0% (10-90 10.2-20.5) typical 16.0% size_scale 1.05\n'
        'macro: no scheduled US releases today; next week: GDP Thu 08:30, PCE Fri 08:30\n'
        'cost_bps: round trip (spread, impact on entry and exit, both commissions) for a $5,000 order per symbol\n'
        'No column forecasts direction: the expected gain from direction is 0. The ± columns are moves up OR down: range1, next60 and rest_of_day are half the expected high-low span (a band around the price; no overnight gap), absr1 the expected size of the close-to-close move (gap included, so usually larger).\n'
        'sym close range1±%(10-90) absr1±% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m note\n'
        'AAA 100 1.24(0.85-1.88) 1.6 28 2(1.5-2.8) 6.5(5.7-7.7) A 900'
    ),
    'default': (
        'as of the close of 2026-10-23; session ranges from 09:30 ET\n'
        'market: SPY vol20 15.2% (10-90 12.1-19.4) vol5 14.0% (10-90 10.2-20.5) typical 16.0% size_scale 1.05\n'
        'macro: no scheduled US releases today; next week: GDP Thu 08:30, PCE Fri 08:30\n'
        'cost_bps: round trip (spread, impact on entry and exit, both commissions) for a $3,000 order per symbol\n'
        'No column forecasts direction: the expected gain from direction is 0. The ± columns are moves up OR down: range1, next60 and rest_of_day are half the expected high-low span (a band around the price; no overnight gap), absr1 the expected size of the close-to-close move (gap included, so usually larger).\n'
        'sym close range1±%(10-90) absr1±% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m note\n'
        'AAA 100 1.24(0.85-1.88) 1.6 28 2(1.5-2.8) 9(8.3-10.2) A 900'
    ),
    'sim': (
        'as of the close of 2026-10-23; session ranges from 09:30 ET\n'
        'market: SPY vol20 15.2% (10-90 12.1-19.4) vol5 14.0% (10-90 10.2-20.5) typical 16.0% size_scale 1.05\n'
        'macro: no scheduled US releases today; next week: GDP Thu 08:30, PCE Fri 08:30\n'
        'cost_bps: round trip (spread, impact on entry and exit, both commissions) for a $5,000 order per symbol\n'
        'No column forecasts direction: the expected gain from direction is 0. The ± columns are moves up OR down: range1, next60 and rest_of_day are half the expected high-low span (a band around the price; no overnight gap), absr1 the expected size of the close-to-close move (gap included, so usually larger).\n'
        'sym close range1±%(10-90) absr1±% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m note\n'
        'AAA 100 1.24(0.85-1.88) 1.6 28 2(1.5-2.8) 6.5(5.7-7.7) A 900\n'
        'BBB 54.2 4(2.5-6) 3.1 45 4(3-5.6) 8.9(7.4-11.4) B 250 8-K 2026-10-22 16:05 ET items 2.02, 9.01: range1 corrected for it (uncorrected ±1.6)\n'
        'CCC 210.5 1(0.7-na) 1.2 22 1.5(1.1-2.1) 5.9(5.2-6.9) A 1200\n'
        'DDD 35 na(na-na) na na 6(4.5-8.4) 12.2(9.4-16.9) C 40\n'
        'EEE 7000 1.5(1-2.2) 1.9 27 3(2.2-4.2) 6.8(5.5-9) B 300\n'
        'FFF 82.3 0.1(0.05-0.16) 0.15 3.1 1.2(0.9-1.7) 5.5(5-6.3) A 2000\n'
        'GGG 18.7 2(1.3-3.1) 2.5 52 na na na na\n'
        'JJJ 31.4 10.9(7-16.3) 9.5 95 8(6-11.2) 12.6(10.2-16.4) B 600\n'
        'HHH not covered\n'
        'III not covered'
    ),
    'session': (
        'as of the close of 2026-10-23; session ranges at the 10:00 look\n'
        'market: SPY vol20 15.2% (10-90 12.1-19.4) vol5 14.0% (10-90 10.2-20.5) typical 16.0% size_scale 1.05\n'
        'macro: no scheduled US releases today; next week: GDP Thu 08:30, PCE Fri 08:30\n'
        'cost_bps: round trip (spread, impact on entry and exit, both commissions) for a $5,000 order per symbol\n'
        'No column forecasts direction: the expected gain from direction is 0. The ± columns are moves up OR down: range1, next60 and rest_of_day are half the expected high-low span (a band around the price; no overnight gap), absr1 the expected size of the close-to-close move (gap included, so usually larger).\n'
        'sym close range1±%(10-90) absr1±% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m next60±%(10-90) rest_of_day±%(10-90) note\n'
        'AAA 100 1.24(0.85-1.88) 1.6 28 2(1.5-2.8) 6.5(5.7-7.7) A 900 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'BBB 54.2 4(2.5-6) 3.1 45 4(3-5.6) 8.9(7.4-11.4) B 250 0.5(0.25-0.75) 1(0.5-1.5) 8-K 2026-10-22 16:05 ET items 2.02, 9.01: range1 corrected for it (uncorrected ±1.6)\n'
        'CCC 210.5 1(0.7-na) 1.2 22 1.5(1.1-2.1) 5.9(5.2-6.9) A 1200 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'DDD 35 na(na-na) na na 6(4.5-8.4) 12.2(9.4-16.9) C 40 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'EEE 7000 1.5(1-2.2) 1.9 27 3(2.2-4.2) 6.8(5.5-9) B 300 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'FFF 82.3 0.1(0.05-0.16) 0.15 3.1 1.2(0.9-1.7) 5.5(5-6.3) A 2000 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'GGG 18.7 2(1.3-3.1) 2.5 52 na na na na 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'HHH 12.4 2.8(1.8-4.2) 3.4 61 9(6-14) 20.8(15.6-29.6) C 32 na na outside the validated universe; no 8-K earnings correction; daily volume under $50M (below 95% of the validated names); no session range\n'
        'JJJ 31.4 10.9(7-16.3) 9.5 95 8(6-11.2) 12.6(10.2-16.4) B 600 0.5(0.25-0.75) 1(0.5-1.5)\n'
        'III 23.1 na na na 9(6-14) 15.3(11.1-22.4) B 110 na na not in the risk universe; cost from live data (rough)'
    ),
}


def expect(golden, notional, sleeve, sess, move):
    """The golden output with only the move line and the two columns added (the row's 9 leading tokens, 11 with the
    session columns, then the move tokens, then the note)."""
    k, out = 9 + 2 * sess, []
    for ln in golden.splitlines():
        sym = ln.split(" ")[0]
        if ln.startswith("sym close "):
            assert ln.endswith(" note"), ln
            ln = ln[:-len(" note")] + f" {COLS} note"
        elif sym in move and not ln.startswith(f"{sym} not covered"):
            t = ln.split(" ")
            ln = " ".join(t[:k] + [move[sym]] + t[k:])
        out.append(ln)
        if ln.startswith("cost_bps: round trip "):
            out.append(LINE.format(n=notional, s=sleeve))
    return "\n".join(out)


def main():
    assert GOLDEN, "GOLDEN missing"
    out = outputs()
    for case, sess in (("live", False), ("sim", False), ("session", True)):
        mv = MOVE if case != "sim" else {s: m for s, m in MOVE.items() if s not in ("HHH", "III")}
        exp = expect(GOLDEN[case], 5000, 30000, sess, mv)
        assert out[case] == exp, f"{case}:\n{out[case]}\n---- expected:\n{exp}"
    print(out["live"], "\n")
    assert "HHH not covered" in out["sim"] and "III not covered" in out["sim"], out["sim"]
    assert "outside the validated universe" in out["live"] and "not in the risk universe" in out["live"]

    # the arithmetic, independently of the expected tokens
    tok = re.compile(r"±(\d+(?:\.\d+)?)\((na|\d+(?:\.\d+)?)\) ±(\d+(?:\.\d+)?)\((na|\d+(?:\.\d+)?)\)")
    rows = {ln.split(" ")[0]: ln for ln in out["live"].splitlines()}
    checked = 0
    for s in SYMS:
        m = tok.search(rows[s])
        if s in ("DDD", "III"):                          # no vol: na, and no ± token at all
            assert m is None and f" {MOVE[s]} " in rows[s] + " ", rows[s]
            continue
        one, top = SPAN[s]
        for usd, pct, span in ((m.group(1), m.group(3), one), (m.group(2), m.group(4), top)):
            if span is None:
                assert usd == pct == "na", rows[s]
                continue
            assert abs(float(usd) - 5000 * span / 200) <= 0.5, (s, usd, span)
            assert abs(float(pct) - 5000 * span / (2 * 30000)) <= 0.005 + 1e-9, (s, pct, span)
            checked += 1
    assert checked == 15, checked                       # 8 rows x 2, less CCC's missing 90% value

    # sleeve_usd (default, clamp to 1,000-10,000,000, booleans rejected) and notional_usd
    cases = [(27500, 27500, "±62(94) ±0.23(0.34)"), (500, 1000, "±62(94) ±6.2(9.4)"), (2e7, 1e7, "±62(94) ±0(0)")]
    cases += [(v, 30000, MOVE["AAA"]) for v in ("abc", True, False, 0, -5, float("nan"), float("inf"))]
    for sleeve, want, shown in cases:
        o = outputs(sleeve)
        assert o["aaa"] == expect(GOLDEN["aaa"], 5000, want, False, {"AAA": shown}), (sleeve, o["aaa"])
        assert f"of a ${want:,.0f} sleeve;" in o["aaa"], (sleeve, o["aaa"])
    assert out["aaa"] == expect(GOLDEN["aaa"], 5000, 30000, False, {"AAA": MOVE["AAA"]}), out["aaa"]
    assert out["default"] == expect(GOLDEN["default"], 3000, 30000, False, {"AAA": "±37(56) ±0.12(0.19)"}), out["default"]

    # wording: the new text never says upside, gain or profit, and says up or down; every non-na token carries ±
    pre = (ROOT / "runs" / "ml" / "research" / "dollarline_prereg.md").read_text(encoding="utf-8")
    quotes, cur = [], []                                # the prereg's two quoted blocks, verbatim
    for ln in pre.splitlines() + [""]:
        if ln.startswith("> "):
            cur.append(ln[2:])
        elif cur:
            quotes, cur = quotes + ["\n".join(cur)], []
    assert len(quotes) == 2, quotes
    section, track = quotes
    D = forecast.DESCRIPTION
    assert section in D and track in D, "the description lacks the prereg's text"
    props = forecast.SPEC["parameters"]["properties"]
    assert props["notional_usd"]["description"] == NOTIONAL_DESC and props["sleeve_usd"]["description"] == SLEEVE_DESC, props
    assert props["sleeve_usd"]["type"] == "number" and forecast.SPEC["parameters"]["required"] == ["symbols"]
    new_text = [section, track, NOTIONAL_DESC, SLEEVE_DESC, LINE.format(n=5000, s=30000), COLS]
    new_text += [tok.search(rows[s]).group(0) for s in SYMS if s not in ("DDD", "III")]
    for t in new_text:
        assert not BANNED.search(t), t
        assert not NOT_A_BOUND.search(t), (NOT_A_BOUND.search(t), t)
    assert "up or down" in section and "up or down" in LINE and "either direction" in section
    assert "Neither is a loss limit" in LINE and "Neither is a loss limit" in section and "not loss limits" in track
    assert "range1's 90% value" in LINE and "range1's 90% value" in section and "range1's 90% value" in track
    assert any(ln.startswith("No column forecasts direction: the expected gain from direction is 0.")
               for ln in out["live"].splitlines())                # F9's line stays
    assert all(m == "na na" or m.count("±") == 2 for m in MOVE.values())      # and the outputs match MOVE exactly (above)

    # F12: the gpt-oss SPEC and the MCP tool carry the same text, and the MCP tool passes sleeve_usd through
    assert forecast.SPEC["description"] == D
    tmp = Path(tempfile.mkdtemp())
    os.environ.update(TRADER_SUBMIT_FILE=str(tmp / "submit.json"), TRADER_LOG_FILE=str(tmp / "log.jsonl"), TRADER_HAS_FORECAST="1")
    import mcp_server
    ft = {t.name: t for t in asyncio.run(mcp_server.server.list_tools())}["forecast"]
    assert ft.description == forecast.SPEC["description"]
    sch = ft.input_schema["properties"]
    for k in ("symbols", "notional_usd", "sleeve_usd"):         # every parameter too (forecast_fixes_prereg.md, item d)
        assert sch[k]["description"] == props[k]["description"], (k, sch[k])
    assert ft.input_schema.get("required") == ["symbols"], ft.input_schema
    got = []
    mcp_server.research = lambda name, args: got.append((name, args)) or "ok"
    mcp_server.forecast(["AAA"], notional_usd=5000, sleeve_usd=27500)
    assert got == [("forecast", {"symbols": ["AAA"], "notional_usd": 5000, "sleeve_usd": 27500})], got
    print("ok")


if __name__ == "__main__":
    main()
