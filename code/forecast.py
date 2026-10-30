"""The ML arm's extra research tool, `forecast`: risk and cost forecasts from our models (how much a stock is likely to
move and what a round trip costs, not which way). Only the -ml sleeves (base-ml, rl2-ml, opus-ml) get it, so comparing
them with their twins measures what the tool adds.

What ships (runs/ml/research/synthesis.md and synthesis_wave_a.md), all out-of-sample validated:
  - per symbol: next-session range, next-session |move|, 20-session vol (ml_research/risk.py); expected spread and the
    round-trip cost of an order of notional_usd (spread + square-root impact + commissions), liquidity tier, average
    dollar volume (ml_research/cost.py);
  - market: expected SPY volatility over 20 and 5 sessions from median-calibrated SPY implied vol (ml_research/regime.py;
    no regime model beat it), with 10-90% ranges and a size scale.
  - wave C (synthesis_wave_c.md): range1 corrected for an SEC 8-K since the previous close (mostly earnings;
    ml_research/events.py), with the latest 8-K and the next earnings date estimate in the row's note, and a macro
    calendar line in the header (ml_research/macro_feats.py).
  - F4 (bondfx.md): on bond, credit and currency ETFs, range1 and vol20 come from the fund's own last 20 sessions or the
    current model rescaled, or are withheld, as validated per group (ml_research/bondfx.py; _bondfx, live and simulator).
  - per row: move$ and sleeve%, range1 and its 90% value as dollars on the order and as a share of the sleeve (F10,
    runs/ml/research/dollarline_prereg.md and its Amendment 1): arithmetic on the row's own numbers, no model of their
    own, with how often a position held from the close went past them (runs/ml/research/dollarline_adverse.py).
No return or direction forecast ships: none we tested beat trading costs out of sample.

Live: compute() runs after the 08:15 refresh (live.py) and caches runs/ml/forecast/fc_<as_of>.json; run() reads the
latest file, prices orders at the requested size, and costs symbols outside the cached universe from the gateway
(cost.live), and adds the events correction (EDGAR fetched at look time) and the macro line. Training: sim_provider(data)
serves strictly out-of-sample predictions as of the close before the decision, with the same correction, note and line.
"""

import json
import math
import os
import re
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ml_research import bondfx

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "ml" / "forecast"
RES = ROOT / "runs" / "ml" / "research"
MAX_SYMBOLS = 20
DEFAULT_NOTIONAL = 3000.0
DEFAULT_SLEEVE = 30000.0        # the sleeves' start: the sleeve% base unless the model passes its sleeve's value

# Names outside the cache, scored at look time (live only; the simulator has no data for them).
ONDEMAND = OUT / "ondemand"     # <as_of>/<SYM>.json: one file per name and close, shared by the live loop and opus-ml
ONDEMAND_V = 3                  # rows saved by another version are ignored (3: rows carry the fund's own base, "own")
ONDEMAND_BUDGET_S = 40.0        # every gateway request for outside names (cost, on-demand rows, their session ranges) is
                                # refused this many seconds after the call starts: 5 s of F12's 45 s are left for scoring and
                                # the text (forecast_fixes_prereg.md); no on-demand name starts with under 5 s left (20 s cold)
CALL_CAP_S = 90.0               # the hard cap: that deadline is never later than this, whatever the budget
OUTSIDE_FLOOR_S = 45.0          # Amendments 1-2 (smoke of 24 Oct: item 2 alone scored 0 of 16 outside names in a fresh
                                # process): outside names keep at least 45 s from the start of _outside, within CALL_CAP_S
PER_REQ_S = 20.0                # longest wait for one bars request (the gateway waits 60 s by default)
SYM_OK = re.compile(r"[A-Z][A-Z0-9 .\-]{0,9}")      # what we fetch and use as a file name
BOND_FX = bondfx.BOND_FX        # bond, credit and currency funds by IBKR long name (one definition, in ml_research/bondfx.py)
LEVERAGED = bondfx.LEVERAGED    # leveraged, inverse and volatility products (one definition: bondfx.group excludes them)
_DATA = {}                      # as_of -> trader.Data through that close, once per process
_MEMO = {}                      # history bars requests -> (monotonic time, rows), reused for 15 minutes
_LOCK = threading.Lock()        # the MCP server may run tool calls in threads

# The tool description, verbatim from runs/ml/research/synthesis_wave_c.md, plus the dollar columns verbatim from
# dollarline_prereg.md (gpt-oss sees each line as a "// " comment; the MCP tool reads the same SPEC).
DESCRIPTION = """Risk and cost forecasts from our machine-learning models, plus today's company filings and macro calendar: how much a stock is likely to move and what a round trip costs, not which way it will go. Use them to size positions, to set stop and limit distances, and to skip trades whose cost eats much of the expected move. A move forecast is not a gain forecast: every ± column is as likely down as up.

**Ask for all your candidates in one call** (up to 20 symbols): each call uses one of your research calls, whatever its length.

**Header lines:**
- **market:** expected SPY volatility over the next 20 and 5 sessions, annualized %, with 10-90% ranges. Also size_scale = typical SPY vol / expected vol, which is about how much gross exposure keeps your risk at its usual level.
- **macro:** today's scheduled US releases (ET), with the first-release value once it is in our data and the Kalshi pre-release expectation, plus the next week's CPI, payrolls, PCE, GDP and FOMC. It is a calendar, not a forecast.

**Columns as of the last close:**
- range1 ±: the next session's expected high-low range, shown as ± half of it: a band around the price, % of price. The (10-90) range is the uncertainty of the forecast itself, not a 90% price bound. Out of sample, a stop at the band edge (one ± value away) was touched in about 40% of sessions, and one at twice the ± value in about 15%. It covers the session only: the overnight gap is not in it, so absr1 (close to close, gap included) is usually larger.
  - If the company filed an SEC 8-K since the previous close (or in the last 24 h) up to 09:00 ET, range1 is corrected for it, and the note gives the uncorrected value. Most of the correction is earnings.
- absr1 ±: expected size of the next close-to-close move, up or down, %.
- vol20: expected annualized volatility over the next 20 sessions, %. It is mostly the stock's own option-implied vol.
- spread: expected time-average quoted spread next session, bps, with its 10-90% range.
- cost: round-trip cost at notional_usd, bps, with its 10-90% range. It covers the spread, impact on entry and exit, and commissions ($0.005 a share, $1 minimum per order, so small orders pay more).
- tier: liquidity tier, A best to D thinnest.
- adv: average daily dollar volume, $ millions.
- note: the latest 8-K of the last 7 days, with its acceptance time (ET) and item codes (2.02 earnings, 1.01 material agreement, 2.01 acquisition done, 5.02 officer change, 3.02 unregistered equity sale, 7.01/8.01 press releases), and the next earnings date estimate when it falls within 10 sessions. The estimate is last year's same-quarter report plus 52 weeks.

**Session columns (09:30-16:00 ET), from today's 30-minute bars:**
- next60 ±: the expected high-low range over the next 60 minutes, shown as ± half of it, % of the current price, with its 10-90% range.
- rest_of_day ±: the same through the close. A stop closer than twice the rest_of_day ± value's 10% figure is likely to be touched by noise.

**Dollar columns (after the session columns, before the note), from range1:**
- move$ ±(90): what a move of one range1 ± value, up or down, is in dollars on an order of notional_usd (not rounded to whole shares); in brackets, the same for range1's 90% value (the 90 in range1's 10-90). For example, a one-± move on a $5,000 order is ±$62 (0.21% of a $30,000 sleeve); at range1's 90% value ±$94 (0.31%).
- sleeve% ±(90): the same two moves as % of sleeve_usd (default $30,000, a sleeve's starting value: pass your sleeve's current value).
- They measure the size of a move, not a result: either direction is as likely. A name without a range1 forecast shows na.
- Neither is a loss limit. range1 is half the session's high-low span and leaves out the overnight gap, so the dollars understate a move from the close: measured from the last close, the session's largest move, up or down, went past the bracketed figure in about 47% of sessions. absr1 (close to close, gap included) measures a move from the close.

**No return or direction forecast is offered.** Out of sample, none of the models we tested beat trading costs. That covers overnight gaps, intraday momentum, reversal, volume and time-of-day, news sentiment, industry, factor timing, momentum ranking and market timing. Our anomaly flag did not predict drops. Earnings drift was not tested.

**Track record** (out of sample; per-symbol figures cover today's S&P 500 stocks and ETFs, so they are survivor-biased; scored on the full high-low range, of which range1, next60 and rest_of_day show half):
- **range1:** R2 0.37 vs 0.29 for the 20-day average range (Jan 2025-Oct 2026). A plain recent-range average does nearly as well. Its 10-90% range held 81% of outcomes, but only 76% in the April 2025 shock half-year.
- **range1 after an 8-K:** uncorrected, R2 was 0.02 and the 10-90% range held 65%; corrected, R2 0.51 and 80% (9,603 stock-sessions, Mar 2025-Oct 2026, ahead in every half-year). Other sessions are unchanged.
  - The morning after an earnings release, the realized range averaged 2.1x the uncorrected forecast, which held only 27%; corrected, it held 80%. Other 8-K types change almost nothing.
  - On sessions that went on to gap more than 5%, the corrected range held 70% (47% uncorrected).
  - 8-Ks accepted 09:00-09:30 ET are not counted.
- **move$ and sleeve%:** arithmetic on range1 and its 90% value, with no model of their own; not loss limits. Held from the close (223,155 stock-sessions, Feb 2025-Oct 2026, overnight gap included), a long's next-session low went more than one ± below the close in 41% of sessions and more than range1's 90% value below it in 23%; a short's high went that far above it in 43% and 25%. One ± of the 20-day average range was passed in 42% (long) and 45% (short); a true 90% loss limit would be passed in 10%.
- **Earnings date estimate:** the next earnings filing came in the estimated session 47% of the time, within 1 session 60% and within 3 sessions 72% (3,445 reports, Jan 2025-Oct 2026). No probability is given: ours was not calibrated.
- **absr1:** R2 0.14 vs 0.13 for our previous model, Feb-Oct 2026 only. Daily move sizes are mostly unpredictable, so this is the typical size of a move, not tomorrow's.
- **vol20:** R2 0.62 vs 0.50 for our previous model and 0.32 for trailing 20-day vol (Mar 2025-Sep 2026), ahead in every half-year and unbiased on average. Nearly all of the gain is the implied vol itself.
- **Fallback:** when a stock has no recent implied vol or intraday data, the note says that absr1 and vol20 come from the previous model.
- **next60 and rest_of_day:** R2 0.58 and 0.56 vs 0.49 and 0.48 for the stock's 20-session average at the same time of day (Jan-Oct 2026), ahead at every look. The 10-90% ranges held 80% of outcomes, with these weak spots:
  - on big-gap opens (gap above about 5%, e.g. earnings), only about 55% at 09:30 (75-80% from 10:00), with 40-45% of outcomes above the top. An event correction for these columns failed our test, so they are not corrected.
  - on FOMC days, the ranges spanning the 14:00 statement ran too narrow: next60 held 44-65% at the 13:30-14:30 looks, and the realized range after 14:00 was 1.1-1.7x the forecast on all 6 FOMC days of 2026. This was seen in a breakdown, not tested in advance.
  - about 75% on the market's most volatile days;
  - ETFs and half-days were not validated.

  On the 29 CPI, payroll, PCE and GDP mornings of 2026, the session ranges already held 80-82%. Release flags did not improve them, so nothing is adjusted for releases.
- **SPY vol:** SPY's 30-day implied vol, de-biased by its median past error. 20-session R2 is 0.28 vs -0.20 for trailing vol, but about 0 outside the April 2025 shock. Its 10-90% ranges held 79-81% of outcomes, but only 48% in the shock half-year.
- **Spread:** log R2 0.970 vs 0.962 for yesterday's spread, and its 10-90% range held 80% of outcomes. Spreads are persistent, so this is careful bookkeeping more than prediction.
- **Impact:** a textbook square-root prior, not fitted.
- **Other US stocks and ETFs:** in live looks the same models score them on request from IBKR data, and the note says "outside the validated universe". In a check on 81 stocks outside the S&P list (after the fact, not pre-registered), range1's 10-90% range held 79% of outcomes (73% under $50M daily volume). The note also flags leveraged, inverse and volatility ETFs, bond, credit and currency funds (see below), implied vol above 99% and daily volume under $50M. These names get no 8-K earnings correction. Their cost comes from live data and is rough. A name the tool has no time for gets the cost columns only.
- **Bond, credit and currency ETFs** (TLT IEF SHY TIP, LQD HYG, UUP FXE, and funds outside the list whose name says bond, Treasury, credit or currency, except leveraged or inverse funds and currency-hedged stock funds): our pooled models ran too high on these (range1 about 1.7x; its 10-90% range held 38-40%). range1 is now the fund's own average range over its last 20 sessions, calibrated on these 8 funds through 15 Oct 2026. For Treasury funds, vol20 is the fund's own realized vol over its last 20 sessions, calibrated the same way; for credit and currency funds, it is our pooled vol20 rescaled on these funds; for funds outside the list, it is not shown. Out of sample, range1's 10-90% range held 80% on Treasury, 80% on credit and 80% on currency funds (Mar 2025-Oct 2026). vol20's held 82%, 81% and 86%, against 70%, 71% and 81% for the pooled model (29 Apr 2025-Sep 2026, after the April 2025 shock); from 6 Mar 2025, shock included, it held 79%, 76% and 83%. vol20 gets the level and the range right but not one fund's ups and downs: its log R2 is 0.86 on Treasury, 0.13 on credit and -0.23 on currency funds (worse than a flat forecast). absr1 and the session columns are not recalibrated for them."""

SPEC = {
    "name": "forecast",
    "description": DESCRIPTION,
    "parameters": {"type": "object", "properties": {
        "symbols": {"type": "array", "items": {"type": "string"},
                    "description": f"Up to {MAX_SYMBOLS} symbols. Put all your candidates in one call: each call uses one research call."},
        "notional_usd": {"type": "number",
                         "description": f"Order size, $, for the cost and move$ columns (default {DEFAULT_NOTIONAL:,.0f})."},
        "sleeve_usd": {"type": "number", "description": f"Your sleeve's value, $, for the sleeve% column (default "
                                                        f"{DEFAULT_SLEEVE:,.0f}, a sleeve's starting value)."}},
        "required": ["symbols"]},
}


def _r(x, k):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


# ---------------------------------------------------------------- live
def compute(data=None):
    """After the daily refresh: risk forecasts, spread forecasts and the market line from the latest cached close."""
    import trader
    from ml_research import cost, regime
    data = data or trader.Data()
    _DATA.clear()
    _DATA[data.dates[-1]] = data        # look-time scoring in this process (live.py) reuses it and vol2's features from it
    from ml_research import vol2
    v = vol2.live(gw=None, data=data, no_stale_check=True)      # all cached symbols; previous-model fallback in the note
    as_of, res = v["as_of"], v["rows"]
    d, cal, S, F, y, have = cost.build()
    t = len(d.dates) - 1
    M = cost._model()
    sm, cold = M["spread"], M["cold"]
    p = cost.predict(sm, cost.rows(F, sm["names"], [t]))
    lo, hi = cost.interval(sm["q"], p, F["ldv"][:, t], sm["edges"])
    pc = cost.predict(cold, cost.rows(F, cold["names"], [t]))
    lc, hc = cost.interval(cold["q"], pc, F["ldv"][:, t], cold["edges"])
    use = np.isfinite(p)
    p, lo, hi = np.where(use, p, pc), np.where(use, lo, lc), np.where(use, hi, hc)
    rows = {}
    for j, s in enumerate(d.sym):
        k = data.k.get(s)
        if k is None or np.isnan(data.c[k, -1]):
            continue
        r = res.get(s, {})
        rows[s] = {"close": _r(data.c[k, -1], 2), "range1_pct": r.get("range1_pct"), "range1_lo": r.get("range1_lo"),
                   "range1_hi": r.get("range1_hi"), "absr1_pct": r.get("absr1_pct"), "vol20_pct": r.get("vol20_pct"),
                   "note": r.get("note") or "", "spread": _r(math.exp(p[j]), 2) if np.isfinite(p[j]) else None,
                   "spread_lo": _r(math.exp(lo[j]), 2) if np.isfinite(lo[j]) else None,
                   "spread_hi": _r(math.exp(hi[j]), 2) if np.isfinite(hi[j]) else None,
                   "adv_usd": _r(math.exp(F["ldv"][j, t]), 0), "sigma_d": _r(math.exp(F["lvol"][j, t]), 5)}
        g = bondfx.group(s)
        if g:
            _bondfx(rows[s], g, bondfx.own_base(data.h[k], data.l[k], data.c[k]))
    m = regime.live(gw=None, data=data)
    market = {k: m.get(k) for k in ("spy_vol20_pct", "spy_vol20_10_90_pct", "spy_vol5_pct", "spy_vol5_10_90_pct",
                                    "typical_spy_vol_pct", "size_scale_vs_typical", "notes")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fc_{as_of}.json").write_text(json.dumps({"as_of": as_of, "market": market, "forecasts": rows}), encoding="utf-8")
    return as_of, rows


def live_provider():
    files = sorted(OUT.glob("fc_*.json"))
    if not files:
        return None
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    return d["as_of"], {"market": d["market"], "rows": d["forecasts"]}


def _today_et():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()


def _prev_session(today):
    try:
        from live import HOLIDAYS                     # inside: live imports forecast
    except Exception:                                 # noqa: BLE001
        HOLIDAYS = set()
    d = today - timedelta(days=1)
    while d.weekday() >= 5 or d.isoformat() in HOLIDAYS:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


# ---------------------------------------------------------------- training simulator
def sim_provider(data):
    """A function of decision session d (and the look time, "HH:MM" ET; the macro line's as-of) -> (as_of, {"market",
    "rows"}) from strictly out-of-sample predictions at d - 1."""
    from ml_research import events, macro_feats, regime, vol2
    risk_at = vol2.sim_provider(data)
    ev_at, macs = events.sim_provider(data), {}
    with np.load(RES / "cost_oos.npz", allow_pickle=True) as z:      # read each array once (an NpzFile re-reads on every [])
        c = {k: z[k] for k in z.files}
    reg = regime.sim_provider(data)
    ccol, crow = {str(x): i for i, x in enumerate(c["dates"])}, {str(s): j for j, s in enumerate(c["symbols"])}

    def at(d, look="09:00"):
        as_of = data.dates[d - 1]
        got = risk_at(d)
        if not got:
            return None
        ci = ccol.get(as_of)
        rows = {}
        for s, rr in got[1].items():
            row = {k: rr.get(k) for k in ("close", "range1_pct", "range1_lo", "range1_hi", "absr1_pct", "vol20_pct")}
            old = [t for t, m in rr.get("model", {}).items() if m == "har" and t != "range1"]   # range1 is HAR by design
            row["note"] = f"{vol2.FALLBACK_NOTE} {', '.join(old)}" if old else ""      # live's wording too
            cj = crow.get(s)
            if ci is not None and cj is not None and np.isfinite(c["spread"][cj, ci]):
                row.update({"spread": _r(c["spread"][cj, ci], 2), "spread_lo": _r(c["spread_lo"][cj, ci], 2),
                            "spread_hi": _r(c["spread_hi"][cj, ci], 2), "adv_usd": _r(c["adv_usd"][cj, ci], 0),
                            "sigma_d": _r(c["sigma_d"][cj, ci], 5)})
            g = bondfx.group(s)
            if g:
                k = data.k[s]
                _bondfx(row, g, bondfx.own_base(data.h[k, :d], data.l[k, :d], data.c[k, :d]), as_of)
            rows[s] = row
        ev = (ev_at(d) or (None, {}))[1]
        for s, row in rows.items():
            if s in ev:
                _apply_event(row, ev[s])
        m = reg(d) or {}
        market = {k: m.get(k) for k in ("spy_vol20_pct", "spy_vol20_10_90_pct", "spy_vol5_pct", "spy_vol5_10_90_pct",
                                        "typical_spy_vol_pct", "size_scale_vs_typical")}
        if look not in macs:
            macs[look] = macro_feats.sim_provider(data, look=look)
        market["macro"] = macs[look](d)[1]["macro_today"]
        return (as_of, {"market": market, "rows": rows}) if rows else None
    return at


def _bondfx(row, grp, own, as_of=None):
    """Bond, credit and currency funds (ml_research/bondfx.py, F4): range1 and vol20 from the fund's own last 20 sessions
    or the current model rescaled, or withheld, as validated per group, with the note clause first. compute() and
    sim_provider() call it and _ondemand() uses the same bondfx.apply and clause, so live and simulator rows agree;
    as_of (the simulator's close) picks the walk-forward fit, None the live fit."""
    c = bondfx.apply(row, grp, own, as_of)
    row["note"] = "; ".join(x for x in (c, row.get("note")) if x)


def _apply_event(row, ev):
    """The events correction and note on one forecast row (live and simulator): when range1_event applies, the range1
    triplet becomes the corrected one and the note gives the uncorrected value."""
    from ml_research import events
    if ev.get("range1_event_pct") is not None:
        row.update(range1_pct=ev["range1_event_pct"], range1_lo=ev["range1_event_lo"], range1_hi=ev["range1_event_hi"])
    row["note"] = "; ".join(x for x in (row.get("note"), events.note(ev)) if x)


# ---------------------------------------------------------------- the tool
def _move(r, notional, sleeve):
    """The move$ and sleeve% tokens of one row (dollarline_prereg.md): the dollar move of a `notional` order at one range1 ±
    (half the high-low span) and at range1's 90% value (half its full-span 90% value), and those as % of the sleeve. Week 2's
    models read range1 as an expected gain, so every value carries ± and none has a sign; na without range1. Neither is a
    loss limit (Amendment 1), so the bracket is never called a top or a bound."""
    ok = lambda x: x is not None and math.isfinite(x)
    one, p90 = r.get("range1_pct"), r.get("range1_hi")
    if not ok(one):
        return "na na"
    usd = lambda x: f"{notional * x / 200:.0f}" if ok(x) else "na"
    pct = lambda x: f"{round(notional * x / 2 / sleeve, 2):g}" if ok(x) else "na"
    return f"±{usd(one)}({usd(p90)}) ±{pct(one)}({pct(p90)})"


def _cost_row(r, notional):
    from ml_research import cost
    price = r.get("close")
    if not price or r.get("spread") is None or not r.get("adv_usd"):
        return None
    qty = max(1, round(notional / price))
    em = (r["absr1_pct"] / 100) if r.get("absr1_pct") else math.sqrt(2 / math.pi) * (r.get("sigma_d") or 0)
    return cost.order_cost(price, qty, r["spread"], r["spread_lo"], r["spread_hi"], r["adv_usd"], r.get("sigma_d") or 0, em)


def _data(as_of):
    """trader.Data through the forecast file's close, once per process (compute() leaves its own here)."""
    if as_of not in _DATA:
        import trader
        _DATA.clear()
        _DATA[as_of] = trader.Data(last=as_of)
    return _DATA[as_of]


class _Capped:
    """The gateway for the outside names of one forecast call. A bars request waits at most PER_REQ_S, nothing waits past
    `deadline`, and after it every request is refused at once (TimeoutError; vol2, intraday and cost catch it per symbol).
    History bars (anything but today's "1 D") are reused for 15 minutes: vol2 and intraday ask for the same month of
    30-minute bars, later calls ask again, and IBKR refuses identical requests within 15 s."""

    def __init__(self, gw, deadline):
        self.gw, self.deadline = gw, deadline

    def _left(self):
        left = self.deadline - time.monotonic()
        if left < 1:
            raise TimeoutError("forecast time cap reached")
        return left

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, **kw):
        key = (symbol, duration, bar_size, what, rth, kw.get("end", ""), kw.get("primary", ""))
        now = time.monotonic()
        with _LOCK_MEMO:
            for k in [k for k, v in _MEMO.items() if now - v[0] > 900]:
                del _MEMO[k]
            if key in _MEMO and duration != "1 D":
                return _MEMO[key][1]
        kw["timeout"] = min(kw.get("timeout") or PER_REQ_S, PER_REQ_S, self._left())
        rows = self.gw.bars(symbol, duration=duration, bar_size=bar_size, what=what, rth=rth, **kw)
        with _LOCK_MEMO:
            _MEMO[key] = (now, rows)
        return rows

    def stats(self, symbols, primary="", seconds=3.0):
        return self.gw.stats(symbols, primary=primary, seconds=min(seconds, self._left()))

    def quotes(self, symbols, primary="", timeout=12.0):
        return self.gw.quotes(symbols, primary=primary, timeout=min(timeout, self._left()))

    def __getattr__(self, name):            # details, ...: refused after the deadline (each may wait 30 s)
        f = getattr(self.gw, name)
        return (lambda *a, **k: (self._left(), f(*a, **k))[1]) if callable(f) else f


_LOCK_MEMO = threading.Lock()


def _save(p, doc):
    """Atomic write; two processes may score the same name at once (same content), so errors are ignored."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def _is_etf(doc):
    return (doc.get("stock_type") or "").upper() in ("ETF", "ETN")


def _flags(doc, bond=""):
    """The note's warnings for a name scored outside the validated universe; `bond` is bondfx.apply()'s clause for a
    bond, credit or currency fund (bondfx.group never gives a leveraged or inverse fund one)."""
    r, name, etf = doc["row"], doc.get("long_name") or "", _is_etf(doc)
    out = ["outside the validated universe" + (" (an ETF: session ranges were validated on stocks)" if etf else "")]
    if bondfx.group(doc.get("symbol", ""), name, doc.get("stock_type")):
        out.append(bond or bondfx.clause(bondfx.ONDEMAND))
    elif etf and LEVERAGED.search(name):
        out.append("leveraged, inverse or volatility product (daily reset): not validated")
    elif not etf:
        out.append("no 8-K earnings correction")
    if (r.get("iv_pct") or 0) > 99:
        out.append(f"implied vol {r['iv_pct']:g}% is above the training range (99%)")
    return "; ".join(out)


def _final(note):
    """A gateway failure that will not change on retry (no data for this contract, e.g. no listed options). Not every
    162: IBKR also uses it for pacing violations and cancelled queries, which a later call should retry."""
    n = note.lower()
    return "returned no data" in n or "no market data permissions" in n


def _ondemand(gw, syms, as_of, deadline):
    """Risk rows for names outside the cache from the cached models on IBKR data (vol2.live with the close IV, as
    compute() serves), kept per close in ONDEMAND/<as_of>/<SYM>.json so each name is fetched once whichever process asks.
    A name is not started unless 5 s are left (20 s while this process has not loaded the cache yet); a name whose IBKR
    data failed (other than "no data") is served but not kept, so a later call retries it.
    Returns ({symbol: row}, {symbol: why it has no row})."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from ml_research import vol2
    rows, why = {}, {}
    for s in syms:
        if not SYM_OK.fullmatch(s):
            why[s] = "not a symbol we look up"
            continue
        p = ONDEMAND / as_of / f"{s.replace(' ', '_')}.json"
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            doc = doc if doc.get("v") == ONDEMAND_V else None
        except (OSError, ValueError, AttributeError):
            doc = None
        failed = False
        if doc is None:
            cold = as_of not in _DATA or not vol2._PANEL
            if time.monotonic() > deadline - (20 if cold else 5):
                why[s] = "risk columns skipped (time cap)"
                continue
            og = _Capped(gw, deadline)
            try:
                d = _data(as_of)                     # first use in a process: about 12 s (the cache and vol2's features)
                try:
                    det = og.details(s) or {}
                except Exception:                    # noqa: BLE001 - only the note uses it
                    det = {}
                v = vol2.live(og, symbols=[s], data=d, no_stale_check=True, decision_iv=False)
            except Exception as e:                   # noqa: BLE001 - the time cap's refusal lands here too
                why[s] = "risk columns skipped (time cap)" if isinstance(e, TimeoutError) else \
                    f"risk columns unavailable ({type(e).__name__})"
                continue
            doc = {"v": ONDEMAND_V, "symbol": s, "as_of": as_of,
                   "scored_et": f"{datetime.now(ZoneInfo('America/New_York')):%Y-%m-%d %H:%M}",
                   "long_name": det.get("long_name"), "stock_type": det.get("stock_type"), "row": v["rows"].get(s) or {}}
            bad = [n for n in v["notes"] if n.startswith(f"{s}: gateway") and not _final(n)]
            if any("time cap" in n for n in bad):
                why[s] = "risk columns skipped (time cap)"
                continue
            failed = bool(bad) or not det
            if not failed:
                _save(p, doc)
        r = dict(doc["row"])
        if "close" not in r:
            why[s] = r.get("note") or "not covered by the risk models"
            continue
        g = bondfx.group(s, doc.get("long_name"), doc.get("stock_type"))
        bond = bondfx.apply(r, g, r.get("own")) if g else ""     # as compute() and sim_provider(); _flags adds the clause
        keep = [n for n in (r.get("note") or "").split("; ") if n and not (_is_etf(doc) and n.startswith("unusually low"))]
        r["note"] = "; ".join([_flags(doc, bond)] + keep + (["some IBKR data failed: retried next call"] if failed else []))
        rows[s] = r
    return rows, why


def _outside(gw, names, rows, notional, as_of, snote, sess, t0):
    """Names outside the cache: live cost rows (as before), then risk rows scored on demand (added to `rows`) and, during
    the session, their session ranges (added to `sess`). Runs after the cached names' session ranges and 8-K check, so it
    never delays or blanks them; every gateway request it makes is refused ONDEMAND_BUDGET_S after the call's start t0
    or OUTSIDE_FLOOR_S after _outside's start, whichever is later (CALL_CAP_S at the latest; Amendment 1's remedy). Returns ({symbol: cost row}, {symbol: why it has no risk row})."""
    from ml_research import cost
    end = min(t0 + CALL_CAP_S, max(t0 + ONDEMAND_BUDGET_S, time.monotonic() + OUTSIDE_FLOOR_S))
    cg = _Capped(gw, end)
    try:
        costs = {o["symbol"]: o for o in cost.live(cg, symbols=names, notional=notional)["orders"]}
    except Exception as e:                              # noqa: BLE001
        costs = {s: {"symbol": s, "error": f"{type(e).__name__}: {e}"[:120]} for s in names}
    ok = [s for s in names if s in costs and not costs[s].get("error")]
    with _LOCK:
        scored, why = _ondemand(cg, ok, as_of, end)
    rows.update(scored)
    if scored and snote.startswith("session ranges at"):
        if end - time.monotonic() < 1:
            sess.update({s: "session ranges skipped (time cap)" for s in scored})
        else:
            hint = {s: r["iv_pct"] / 100 for s, r in scored.items() if r.get("iv_pct")}
            why_s, more = _session(cg, list(scored), hint)
            sess.update(more or {s: why_s for s in scored})       # a failure as a whole: its reason in each row's note
    return costs, why


def _session(gw, syms, iv_hint=None):
    """In-session ranges (next 60 minutes, rest of day) from ml_research/intraday.py, 09:30-16:00 ET on full days only.
    Returns (header note, {symbol: (next60, lo, hi, rest, lo, hi) or error text})."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    mins = now.hour * 60 + now.minute
    if now.weekday() >= 5 or not (9 * 60 + 30 <= mins < 16 * 60):
        return "session ranges from 09:30 ET", {}
    try:
        import live
        if now.strftime("%Y-%m-%d") in live.HALF_DAYS:
            return "no session ranges on half-days (not validated)", {}
    except Exception:                                   # noqa: BLE001
        pass
    try:
        from ml_research import intraday
        out = intraday.live(gw, symbols=syms, now=now, iv_hint=iv_hint)
    except Exception as e:                              # noqa: BLE001
        return f"session ranges unavailable ({type(e).__name__}: {str(e)[:80]})", {}
    if out.get("error"):
        return f"session ranges unavailable ({out['error']})", {}
    res = {}
    for s, r in (out.get("rows") or {}).items():
        if r.get("error"):
            res[s] = r["error"]
            continue
        q = lambda lab: (r.get(f"{lab}_range_pct_10_50_90") or [None, None, None])
        res[s] = (r.get("next60_range_pct"), q("next60")[0], q("next60")[-1],
                  r.get("rest_of_day_range_pct"), q("rest_of_day")[0], q("rest_of_day")[-1])
    return f"session ranges at the {now:%H:%M} look", res


def run(gw, args):
    t0 = time.monotonic()                           # the outside names' budget counts from here
    syms = args.get("symbols") or []
    if isinstance(syms, str):
        syms = [syms]
    syms = list(dict.fromkeys(str(s).upper().strip() for s in syms if str(s).strip()))[:MAX_SYMBOLS]
    if not syms:
        return "error: give at least one symbol"
    try:
        notional = float(args.get("notional_usd") or DEFAULT_NOTIONAL)
    except (TypeError, ValueError):
        notional = DEFAULT_NOTIONAL
    notional = min(max(notional, 50.0), 1e6)
    sleeve = args.get("sleeve_usd")
    try:
        sleeve = DEFAULT_SLEEVE if isinstance(sleeve, bool) else float(sleeve or DEFAULT_SLEEVE)
    except (TypeError, ValueError):
        sleeve = DEFAULT_SLEEVE
    if not (math.isfinite(sleeve) and sleeve > 0):          # a sleeve at or below zero has no share to show
        sleeve = DEFAULT_SLEEVE
    sleeve = min(max(sleeve, 1e3), 1e7)                     # as notional is clamped: keeps sleeve% readable
    sim = hasattr(gw, "forecast_provider")
    got = gw.forecast_provider() if sim else live_provider()
    if not got:
        return "no forecasts are available for this date"
    as_of, fc = got
    head = f"as of the close of {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
    if not sim and as_of < _prev_session(_today_et()):      # the machine's own date is UTC+8: after 12:00 ET it is tomorrow
        head += " (stale: not from the last session)"
    rows = fc.get("rows") or {}
    covered = [s for s in syms if s in rows]
    if sim:                                         # rl4's intraday simulator serves its look's OOS session ranges
        snote, sess = gw.session_ranges(covered) if hasattr(gw, "session_ranges") else ("session ranges from 09:30 ET", {})
    else:
        snote, sess = _session(gw, covered)
    head += f"; {snote}"
    m = fc.get("market") or {}
    if not sim:
        from ml_research import events, macro_feats
        try:
            ev = events.live(gw, covered, base={s: rows[s] for s in covered}) if covered else {"rows": {}}
            if ev.get("error"):
                raise RuntimeError(ev["error"])
            for s, e in (ev.get("rows") or {}).items():
                if s in rows:
                    _apply_event(rows[s], e)
            if ev.get("notes"):
                head += "; events: " + "; ".join(ev["notes"])[:300]
        except Exception as e:                          # noqa: BLE001 - range1 stays uncorrected
            head += f"; events unavailable ({type(e).__name__}: {str(e)[:80]})"
        try:
            m = dict(m, macro=macro_feats.macro_today())
        except Exception:                               # noqa: BLE001
            m = dict(m, macro=None)
    if not m.get("macro"):
        head += "; macro calendar unavailable"
    costs, why = {}, {}
    outside = [s for s in syms if s not in rows]
    if outside and not sim:
        costs, why = _outside(gw, outside, rows, notional, as_of, snote, sess, t0)
    if m.get("spy_vol20_pct"):
        rng = lambda k: "-".join(f"{x:g}" for x in (m.get(k) or []) if x is not None) or "na"
        mline = (f"market: SPY vol20 {m['spy_vol20_pct']:g}% (10-90 {rng('spy_vol20_10_90_pct')}) vol5 "
                 f"{m.get('spy_vol5_pct') or 'na'}% (10-90 {rng('spy_vol5_10_90_pct')}) typical {m.get('typical_spy_vol_pct') or 'na'}% "
                 f"size_scale {m.get('size_scale_vs_typical') or 'na'}")
    else:
        mline = "market: no SPY implied vol today"
    cols = "sym close range1±%(10-90) absr1±% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m"
    if sess:
        cols += " next60±%(10-90) rest_of_day±%(10-90)"
    cols += " move$±(90) sleeve%±(90)"
    lines = [head, mline] + ([m["macro"]] if m.get("macro") else []) + [
        f"cost_bps: round trip (spread, impact on entry and exit, both commissions) for a ${notional:,.0f} order per symbol",
        f"move$ ±(90) and sleeve% ±(90): a move of one range1 ± value, up or down, on a ${notional:,.0f} order, in $ and as % "
        f"of a ${sleeve:,.0f} sleeve; in brackets, the same for range1's 90% value. Neither is a loss limit: held from the "
        f"last close, a long or a short was at some point down by more than the bracketed $ in 23-25% of next sessions.",
        "No column forecasts direction: the expected gain from direction is 0. The ± columns are moves up OR down: "
        "range1, next60 and rest_of_day are half the expected high-low span (a band around the price; no overnight gap), "
        "absr1 the expected size of the close-to-close move (gap included, so usually larger).", cols + " note"]
    f = lambda x: "na" if x is None else f"{x:g}"
    h = lambda x: "na" if x is None else f"{round(x / 2, 2):g}"          # a high-low span shown as a ± half-span
    outside = []
    for s in syms:
        r = rows.get(s)
        if not r:
            outside.append(s)
            continue
        c = costs[s] if s in costs else _cost_row(r, notional)       # scored outside names: live cost at this notional
        cc = (f"{f(c['spread_bps'])}({f(c['spread_bps_10'])}-{f(c['spread_bps_90'])}) {f(c['cost_bps'])}({f(c['cost_bps_10'])}-"
              f"{f(c['cost_bps_90'])}) {c['tier']} {f(c['adv_usd_m'])}") if c else "na na na na"
        line = (f"{s} {f(r.get('close'))} {h(r.get('range1_pct'))}({h(r.get('range1_lo'))}-{h(r.get('range1_hi'))}) "
                f"{f(r.get('absr1_pct'))} {f(r.get('vol20_pct'))} {cc}")
        note = r.get("note") or ""
        if s in costs and c.get("adv_usd_m") is not None and c["adv_usd_m"] < 50:
            note = "; ".join(x for x in (note, "daily volume under $50M (below 95% of the validated names)") if x)
        if sess:
            x = sess.get(s)
            if isinstance(x, tuple):
                line += f" {h(x[0])}({h(x[1])}-{h(x[2])}) {h(x[3])}({h(x[4])}-{h(x[5])})"
            else:
                line += " na na"
                note = "; ".join(t for t in (note, x or "no session range") if t)
        line += f" {_move(r, notional, sleeve)}"
        lines.append(f"{line} {note}".rstrip())
    if outside:
        if sim:
            lines += [f"{s} not covered" for s in outside]
        else:
            for o in (costs.get(s) or {"symbol": s, "error": "no cost row"} for s in outside):
                if o.get("error"):                  # a gateway failure names itself: cost.live's note has its cause
                    n = o.get("note") or ""
                    lines.append(f"{o['symbol']} not covered ("
                                 f"{'cost skipped (time cap): ask again' if 'time cap' in n else n or o['error']})")
                    continue
                lines.append(f"{o['symbol']} {f(o.get('price'))} na na na {f(o['spread_bps'])}({f(o['spread_bps_10'])}-"
                             f"{f(o['spread_bps_90'])}) {f(o['cost_bps'])}({f(o['cost_bps_10'])}-{f(o['cost_bps_90'])}) "
                             f"{o['tier']} {f(o['adv_usd_m'])}{' na na' if sess else ''} na na "
                             f"{why.get(o['symbol'], 'not in the risk universe')}; cost from live data (rough)")
    return "\n".join(lines)
