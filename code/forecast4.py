"""ML4.0 of the forecast tool: ML3.1's numbers with a short description, the full track record on request, no advice
wording, and one data fix (absr1 at or below zero). Built 15 Nov 2026 from Trader main 0576413 (forecast3.py ML3.1,
forecast.py ML2.3); review and design: runs/ml/research/week6_prep/ml4/ml3_review.md (section 5, upgrade 5 with fixes 1-4).

What is the same as ML3.1 (forecast3.py):
  - every number, for the same inputs, except absr1 on rows where ML2's model gives zero or less (below). ML2's text comes
    from forecast.run, called unchanged; gap1 comes from ML3's functions (forecast3.quantiles, multiplier, _earn) on a
    byte copy of ML3's fit (runs/ml/research/ml4_gap1_model.json = ml3_model_amend1.json at 0576413); the market-now line
    is forecast3._market_now, unchanged (dated by its print, ML3.1 fix 3);
  - the legend prints once a look (forecast.legend_once, ML2.3 fix 4); later calls print ML2's one-line reference;
  - sleeve% uses the sleeve's own value when sleeve_usd is left out (ML2.3 fix 1).
What changes (text only):
  - DESCRIPTION: one line per column with one line of measured accuracy; no instruction and no list of what failed.
  - forecast(track_record=true) returns the full out-of-sample record (TRACK_RECORD). It is one research call.
  - legend: ML2's "the expected gain from direction is 0" line and the 23-25% sentence (now in the description) are
    replaced by shorter factual lines; gap1's legend line is shorter.
  - absr1 (the data fix): ML2's absr1 model (vol2's blend_ols_intra, an OLS on levels with no floor; its 10-90 band is
    the point times quantile ratios, so it flips too) extrapolates below zero on very quiet names: bond and cash funds
    and stocks pending acquisition. A size cannot be below zero. Where the served absr1 is <= 0, ML4 shows vol20's
    one-day equivalent, sqrt(2/pi) * vol20 / sqrt(252), and says so in the note; with no vol20 shown, absr1 is na.
Not in ML4.0: the review's upgrades 6-9, and no direction column (direction_column() is the hook; it returns None).

ML4 writes only runs/ml/forecast4/ (fc_<as_of>.json from compute(), ondemand/<as_of>/<SYM>.json for gap1 rows of names
outside the daily-bar cache). It reads ML2's cache and on-demand rows through forecast.run, as ML3 does, and ML3's on-demand
gap1 rows (runs/ml/forecast3/ondemand/, filled by warm_outside.py) before computing its own. forecast.run is
ML2's code: on a live call it still writes ML2's own on-demand rows (runs/ml/forecast/ondemand/) and vol2's IV cache, as
it does for ML2's and ML3's sleeves; ML4 cannot stop that without changing ML2. ML2's compute() must have run for the
same close (live.compute_versions computes the versions in use; see HANDOVER.md).
"""

import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import forecast
import forecast3 as f3

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "ml" / "forecast4"          # ML4's own cache
ONDEMAND = OUT / "ondemand"                       # <as_of>/<SYM>.json: gap1 rows for names outside the daily-bar cache
MODEL = ROOT / "runs" / "ml" / "research" / "ml4_gap1_model.json"    # byte copy of ML3's fit (ml3_model_amend1.json)
VERSION = "ML4.2"                                 # recorded on looks, not shown to models
_MEM = {}                                         # as_of -> fc4 document, once per process
K_ABS = math.sqrt(2 / math.pi)                    # E|x| / sd for a normal move

DESCRIPTION = """Size and cost forecasts from our models: how far a stock is likely to move and what a round trip costs. No column forecasts direction; every ± value is a size, up or down. Up to 20 symbols a call; each call is one research call. Columns are as of the last close unless marked.

**Header lines:**
- market: expected SPY vol over the next 20 and 5 sessions, annualized %, with 10-90% ranges; size_scale = typical SPY vol / expected SPY vol.
- market now: SPY's move since the last close, and from 09:30 ET its range, with the time (ET) of the delayed print used. A fact, not a forecast.
- macro: today's scheduled US releases (ET), with first-release values once in our data and Kalshi pre-release expectations, and next week's CPI, payrolls, PCE, GDP and FOMC. A calendar.
- gap1 events tonight: the earnings releases and the CPI, payrolls, PCE or GDP release before 09:30 that raise tonight's gap1.

**Columns:**
- range1 ±(10-90): half the next session's expected high-low span, % of the last close, without the overnight gap. When the company filed an SEC 8-K since the previous close (or in the last 24 h) and before 09:00 ET, range1 is corrected for it and the note gives the uncorrected value; most such filings are earnings. Out of sample its 10-90% range held 81% of sessions (76% in the April 2025 shock half-year).
- absr1 ±: the typical size of the next close-to-close move, gap included, %. Out of sample R2 0.14 (previous model 0.13). Where the model's value is zero or below, absr1 is vol20's one-day equivalent, √(2/π) × vol20 / √252 (na without vol20), and the note says so.
- gap1 ±(10-90): the median and 10-90% range of the size of tonight's gap, today's close to the next open, %; it does not change during the day. It is raised for an earnings release tonight (about ×6.4 at the median on the Alpha Vantage calendar, ×2.7 on our estimate) and about ×1.14 for every name when CPI, payrolls, PCE or GDP comes out before the next open. In a replay of nights our estimate named, before the tool had the Alpha Vantage calendar, 87% of gaps fell below the raised median when no report came that night (a check declared in advance); when one came, 24% did and 16% went above the raised 90% value (computed after the fact). Through 20 Nov 2026 there is no live record of gap1 for a name the events line lists as "earnings (our estimate)", or of how often a report comes then; see the track record. Out of sample its 10-90% range held 78.7% of name-nights.
- vol20: expected annualized vol over the next 20 sessions, %, mostly the stock's own option-implied vol. Out of sample R2 0.62 (trailing 20-day vol 0.32).
- spread_bps(10-90): expected time-average quoted spread next session. cost_bps(10-90): round trip at notional_usd: spread, impact on entry and exit, commissions ($0.005 a share, $1 minimum an order). tier: liquidity, A best to D thinnest. adv_$m: average daily dollar volume, $ millions.
- next60 and rest_of_day ±(10-90), 09:30-16:00 ET: half the expected high-low span over the next 60 minutes and to the close, % of the current price, from today's 30-minute bars. Out of sample their 10-90% ranges held 80%.
- move$ ±(90) and sleeve% ±(90): one range1 ± value, and in brackets range1's 90% value, in $ on notional_usd and as % of your sleeve's value (sleeve_usd, else the value in your state). They are sizes, not gains and not loss limits: held from the last close, a long or a short was at some point down by more than the bracketed $ in 23-25% of next sessions.
- note: the latest 8-K of the last 7 days with its time (ET) and items (2.02 earnings, 1.01 material agreement, 2.01 acquisition done, 5.02 officer change, 3.02 unregistered equity sale, 7.01/8.01 press release); the next earnings date within 10 sessions, named when it comes from the Alpha Vantage calendar, else our estimate (last year's report plus 52 weeks; 47% of reports came in the estimated session); and flags such as outside the validated universe, a bond, credit or currency fund (range1 and vol20 from its own last 20 sessions, or vol20 withheld), or the previous model's fallback.

forecast(track_record=true) returns each column's full out-of-sample record, with its sources and weak spots; it is one research call like any other."""

# The record ML3.1's description carried (forecast.DESCRIPTION's track record section and forecast3.TRACK, at 0576413),
# moved here. Edits, all listed in HANDOVER.md: the paragraph of models that did not beat costs and the size_scale_now
# sentence are gone; commentary without a number is gone ("A plain recent-range average does nearly as well", "Daily move
# sizes are mostly unpredictable ...", "Spreads are persistent, so this is careful bookkeeping more than prediction");
# range1's stop sentence is put as the level a price reached; the absr1 fix's own figures are added.
TRACK_RECORD = """Forecast tool track record (out of sample). Per-symbol figures cover today's S&P 500 stocks and ETFs, so they are survivor-biased. range1, next60 and rest_of_day are scored on the full high-low range, of which the columns show half.
- range1: R2 0.37 vs 0.29 for the 20-day average range (Jan 2025-Oct 2026). Its 10-90% range held 81% of outcomes, 76% in the April 2025 shock half-year. A level one ± value from the last close on a given side was reached in about 40% of sessions, and one at twice the ± value in about 15%. range1 is a mean, not a median: 63% of half-spans fell below it, and half fell below 0.88 of it (95,624 name-sessions, Nov 2025-Aug 2026).
- range1 after an 8-K: uncorrected, R2 was 0.02 and the 10-90% range held 65%; corrected, R2 0.51 and 80% (9,603 stock-sessions, Mar 2025-Oct 2026, ahead in every half-year). Other sessions are unchanged. The morning after an earnings release, the realized range averaged 2.1x the uncorrected forecast, which held only 27%; corrected, it held 80%. Other 8-K types change almost nothing. On sessions that went on to gap more than 5%, the corrected range held 70% (47% uncorrected). 8-Ks accepted 09:00-09:30 ET are not counted.
- absr1: R2 0.14 vs 0.13 for our previous model, Feb-Oct 2026 only. It is the typical size of a move. In ML2's stored nightly forecasts of 15 Oct-12 Nov 2026, 35 of 10,972 rows had a model value of zero or below (SHY, TIP, HYG, IEF, LQD, FXE, UUP, and AES, PTC, TECH), as did 151 of 4,279 rows scored on request (mostly bond and cash funds); on the 35, the next close-to-close move over vol20's one-day equivalent had a median of 1.01 (51% above).
- gap1 (2 Jan 2025-6 Nov 2026, 523 names, 241,993 name-nights): the 10-90% range held 78.7% of gaps, and 47.8% fell below the median. The pinball loss was 7% lower than for the stock's own 60-night median gap with a calibrated range (which held 78.7%). On earnings nights the range held 81.6%, against about 21% without the earnings correction. Through 16 Oct 2026 the test used the realized report times as the calendar, which is optimistic; on the 349 Alpha Vantage calendar nights of 19 Oct-6 Nov 2026 it held 84%, with the median too high (65% of gaps fell below it). On nights our estimate named, it held 80% (43% without the correction); on CPI, payroll, PCE and GDP nights, 79% (73% without). Where our estimate named the night and no report came that night, 87% of gaps fell below the median, 20% below the 10% value and 0.2% above the 90% value (426 name-nights, Nov 2025-Aug 2026, a check declared in advance). Where our estimate named the night and a report did come that night, 24% of gaps fell below the median, 3% below the 10% value and 16% above the 90% value (780 name-nights, Nov 2025-Oct 2026, computed after the fact, with multipliers fit on earlier nights only; with today's fit, which used about 640 of these nights, 23%, 3% and 17%). Both sets are from a replay of nights before the tool had the Alpha Vantage calendar; the two counts cover different windows and give no share of named nights on which a report came. Since 2 Nov 2026 a name is listed as "earnings (our estimate)" only when the latest Alpha Vantage calendar snapshot has no entry for it that is dated today or later and gives a time of day; in the looks of 9-20 Nov 2026 no forecast table listed one (0 of 1,151 tables with gap1), so there is no live record of gap1 on such a night, and how often a report comes on one is not measured. On half of all nights it held 77-85% of names. It runs narrow on shock nights: 14% of names at the 8 Apr 2025 open, 17% at 3 Mar 2026's, 72% at 5 Nov 2026's and 57% at 6 Nov 2026's (67 names), when chips gapped 6-13% against ranges topping out at 4.9-7.7%. In the test the models were refit monthly on every past night; today's fit uses every night through 6 Nov 2026. This method was revised after a first version, fit on 2024 only, missed its coverage bar (76.5%), and it was checked on the same period. Outside the validated universe (a first check on 737 NASDAQ-listed stocks, after the fact): the 10-90% range held about 73% of gaps, against 78.5% on our stocks, and gaps passed its top more often (about 13-14% of nights against a 10% target); thinly traded names often open unchanged, and on these names the stock's own 60-night median gap did as well.
- vol20: R2 0.62 vs 0.50 for our previous model and 0.32 for trailing 20-day vol (Mar 2025-Sep 2026), ahead in every half-year and unbiased on average. Nearly all of the gain is the implied vol itself.
- Fallback: when a stock has no recent implied vol or intraday data, the note says that absr1 and vol20 come from the previous model.
- next60 and rest_of_day: R2 0.58 and 0.56 vs 0.49 and 0.48 for the stock's 20-session average at the same time of day (Jan-Oct 2026), ahead at every look. The 10-90% ranges held 80% of outcomes, with these weak spots: on big-gap opens (gap above about 5%, e.g. earnings), only about 55% at 09:30 (75-80% from 10:00), with 40-45% of outcomes above the top; an event correction for these columns failed our test, so they are not corrected. On FOMC days, the ranges spanning the 14:00 statement ran too narrow: next60 held 44-65% at the 13:30-14:30 looks, and the realized range after 14:00 was 1.1-1.7x the forecast on all 6 FOMC days of 2026 (seen in a breakdown, not tested in advance). About 75% on the market's most volatile days. ETFs and half-days were not validated. On the 29 CPI, payroll, PCE and GDP mornings of 2026, the session ranges already held 80-82%; release flags did not improve them, so nothing is adjusted for releases.
- SPY vol: SPY's 30-day implied vol, de-biased by its median past error. 20-session R2 is 0.28 vs -0.20 for trailing vol, but about 0 outside the April 2025 shock. Its 10-90% ranges held 79-81% of outcomes, but only 48% in the shock half-year.
- Spread: log R2 0.970 vs 0.962 for yesterday's spread, and its 10-90% range held 80% of outcomes.
- Impact: a textbook square-root prior, not fitted.
- move$ and sleeve%: arithmetic on range1 and its 90% value, with no model of their own; not loss limits. Held from the close (223,155 stock-sessions, Feb 2025-Oct 2026, overnight gap included), a long's next-session low went more than one ± below the close in 41% of sessions and more than range1's 90% value below it in 23%; a short's high went that far above it in 43% and 25%. One ± of the 20-day average range was passed in 42% (long) and 45% (short); a true 90% loss limit would be passed in 10%. Measured from the last close, the session's largest move, up or down, went past the bracketed figure in about 47% of sessions.
- Earnings date estimate: the next earnings filing came in the estimated session 47% of the time, within 1 session 60% and within 3 sessions 72% (3,445 reports, Jan 2025-Oct 2026). No probability is given: ours was not calibrated. The Alpha Vantage calendar's accuracy was not measured.
- Other US stocks and ETFs: in live looks the same models score them on request from IBKR data, and the note says "outside the validated universe". In a check on 81 stocks outside the S&P list (after the fact, not pre-registered), range1's 10-90% range held 79% of outcomes (73% under $50M daily volume). The note also flags leveraged, inverse and volatility ETFs, bond, credit and currency funds, implied vol above 99% and daily volume under $50M. These names get no 8-K earnings correction and no gap1 earnings adjustment. Their cost comes from live data and is rough. A name the tool has no time for gets the cost columns only.
- Bond, credit and currency ETFs (TLT IEF SHY TIP, LQD HYG, UUP FXE, and funds outside the list whose name says bond, Treasury, credit or currency, except leveraged or inverse funds and currency-hedged stock funds): our pooled models ran too high on these (range1 about 1.7x; its 10-90% range held 38-40%). range1 is the fund's own average range over its last 20 sessions, calibrated on these 8 funds through 15 Oct 2026. For Treasury funds, vol20 is the fund's own realized vol over its last 20 sessions, calibrated the same way; for credit and currency funds, it is our pooled vol20 rescaled on these funds; for funds outside the list, it is not shown. Out of sample, range1's 10-90% range held 80% on Treasury, 80% on credit and 80% on currency funds (Mar 2025-Oct 2026). vol20's held 82%, 81% and 86%, against 70%, 71% and 81% for the pooled model (29 Apr 2025-Sep 2026, after the April 2025 shock); from 6 Mar 2025, shock included, it held 79%, 76% and 83%. vol20's log R2 is 0.86 on Treasury, 0.13 on credit and -0.23 on currency funds. absr1 and the session columns are not recalibrated for them."""

_P = forecast.SPEC["parameters"]["properties"]
SPEC = {"name": "forecast", "description": DESCRIPTION, "parameters": {"type": "object", "properties": {
    "symbols": {"type": "array", "items": {"type": "string"},
                "description": f"Up to {forecast.MAX_SYMBOLS} symbols. Each call is one research call, whatever its length."},
    "notional_usd": dict(_P["notional_usd"]),
    "sleeve_usd": {"type": "number", "description": "Your sleeve's value, $, for the sleeve% column. Without it the tool uses "
                                                    "the value in your state."},
    "track_record": {"type": "boolean", "description": "true: each column's out-of-sample record is added after the rows "
                                                       "(symbols may then be empty)."}},
    "required": ["symbols"]}}

# ---------------------------------------------------------------- the legend lines ML4 prints in place of ML2's and ML3's
DIR_LINE = ("No column forecasts direction. Every ± value is a size, up or down: range1, next60 and rest_of_day are half the "
            "expected high-low span (no overnight gap), absr1 the close-to-close move (gap included), gap1 the overnight gap "
            "alone.")
GAP1_LINE = ("gap1 ±(10-90): median and 10-90% range of the size of the gap from today's close to the next open, %; it does "
             "not change during the day and includes the events line below. Names outside the validated universe get no "
             "earnings adjustment.")
LOSS_CUT = " Neither is a loss limit"             # ML2's move$ legend: the rest of that line is now in the description
ABS_NOTE = "absr1 from vol20 (the model's value was not above zero)"
ABS_NA = "absr1 na (the model's value was not above zero; no vol20)"


def direction_column(as_of, symbols):
    """The hook for a direction column: {symbol: token} to add a column after gap1, or None for no column (ML4.0: always
    None; no direction model has cleared chance). A future version that fills it must also add its column's line to
    DESCRIPTION and be a new version (ml_tools.VERSIONS)."""
    return None


def _truthy(x):
    return x is True or (isinstance(x, (int, float)) and not isinstance(x, bool) and x == 1) or \
        (isinstance(x, str) and x.strip().lower() in ("true", "yes", "1"))


def _now():
    return f3._now()


def load_model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def fix_absr1(tok, vol20_tok):
    """(absr1 token, note clause or None). A served absr1 at or below zero (ML2's level OLS extrapolating on very quiet
    names) becomes vol20's one-day equivalent, sqrt(2/pi) * vol20 / sqrt(252), or na without vol20. Others unchanged."""
    try:
        v = float(tok)
    except (TypeError, ValueError):
        return tok, None                              # na stays na
    if v > 0:
        return tok, None
    try:
        vol = float(vol20_tok)
    except (TypeError, ValueError):
        vol = None
    if vol is None or not math.isfinite(vol) or vol <= 0:
        return "na", ABS_NA
    x = K_ABS * vol / math.sqrt(252)
    s = f"{round(x, 2):g}"
    return (s if s != "0" else f"{x:.2g}"), ABS_NOTE


# ---------------------------------------------------------------- cache (ML3's document, built by ML3's own functions)
def compute(data=None):
    """After the daily refresh and ML2's compute(): ML3's gap1 inputs (forecast3.build, unchanged) for the latest cached
    close, in runs/ml/forecast4/fc_<as_of>.json, with names in data/bars outside the universe added when their last bar
    is this close (as forecast3.compute). Returns (as_of, rows)."""
    import trader
    data = data or trader.Data()
    doc = f3.build(data)
    doc["version"] = VERSION
    as_of = doc["as_of"]
    for p in f3.BARS.glob("*.json"):
        s = p.stem.replace("_", " ")
        if s not in doc["rows"]:
            r = f3._row_from_bars(f3._load_bars(s) or [], as_of)
            if r:
                doc["rows"][s] = list(r)
    if not (forecast.OUT / f"fc_{as_of}.json").exists():
        doc["ml2_missing"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    forecast._save(OUT / f"fc_{as_of}.json", doc)
    _MEM.clear()
    _MEM[as_of] = doc
    return as_of, doc["rows"]


def _fc4(as_of, sim):
    if as_of not in _MEM:
        if sim:
            import trader
            doc = f3.build(trader.Data(last=as_of))
        else:
            try:
                doc = json.loads((OUT / f"fc_{as_of}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        _MEM.clear()
        _MEM[as_of] = doc
    return _MEM[as_of]


def _outside_rows(gw, syms, as_of, budget):
    """forecast3._outside_rows with ML4's own folder: gap1 inputs for names with no cached daily bars, kept per close in
    ONDEMAND/<as_of>/<SYM>.json. ML3's file for the name and close is read first (read only): warm_outside.py fills ML3's
    folder before the first look, and the row is the same function of the same bars, so ML4 serves what ML3.1 would."""
    import bars as _bars
    out, deadline = {}, time.monotonic() + budget
    for s in syms:
        if not forecast.SYM_OK.fullmatch(s):
            continue
        p = ONDEMAND / as_of / f"{s.replace(' ', '_')}.json"
        hit = False
        for q in (f3.ONDEMAND / as_of / p.name, p):
            try:
                out[s] = json.loads(q.read_text(encoding="utf-8"))["row"]
                hit = True
                break
            except (OSError, ValueError, KeyError):
                pass
        if hit:
            continue
        rows = f3._load_bars(s)
        r = f3._row_from_bars(rows, as_of) if rows else None
        if r is None:
            try:
                g = forecast._Capped(gw, deadline).bars(s, duration="1 Y", bar_size="1 day", what="ADJUSTED_LAST")
                r = f3._row_from_bars(_bars.completed([dict(b, date=str(b["date"])[:8]) for b in g or []]), as_of)
            except Exception:                           # noqa: BLE001 - the time cap's refusal too: the row shows na
                r = None
        if r:
            out[s] = list(r)
            forecast._save(p, {"symbol": s, "as_of": as_of, "row": out[s]})
    return out


# ---------------------------------------------------------------- the tool
def run(gw, args, sleeve_value=None, look=None):
    """The forecast tool, ML4.0. args as ML3.1's, plus track_record (true: the record after the rows, so a cut reply keeps them; with no symbols, the record
    alone). One research call either way (the caller counts it)."""
    args = dict(args or {})
    tr = _truthy(args.pop("track_record", None))
    syms = args.get("symbols")
    if tr and (not syms or (isinstance(syms, (list, tuple)) and not any(str(s).strip() for s in syms))):
        return _record(look)
    text = _slot_label(_plain(_run(gw, args, sleeve_value=sleeve_value, look=look)))
    return f"{text}\n\n{_record(look)}" if tr else text


# ML4.1 (Mon 16 Nov 2026, the first live day): 7 looks asked for the record twice (about 2,400 tokens each time; one of
# them, base-ml 14:00, ran out of context and submitted nothing). The record does not change within a look, so it prints
# once a look, as the legend does (forecast._LEGEND_SEEN); a later call that asks for it gets TRACK_REF. look None: the
# full record every call, as before.
TRACK_REF = ("Track record: printed in your earlier forecast call this look (it does not change within a look); "
             "it is not repeated here.")
_TR_SEEN = {}               # look key -> True once that look printed the record (insertion order, capped)


def _record(look):
    if look is None:
        return TRACK_RECORD
    with forecast._LEGEND_LOCK:
        if look in _TR_SEEN:
            return TRACK_REF
        _TR_SEEN[look] = True
        while len(_TR_SEEN) > 512:
            del _TR_SEEN[next(iter(_TR_SEEN))]
    return TRACK_RECORD


# ML4.1: at the 09:30 look the delayed 30-minute bars do not hold the open yet (they run about 15 minutes behind), and two lines
# named a Python error. The same facts in plain words; nothing else on those lines changes. (Trader Training's review, 16 Nov:
# the Researcher's "and do not hold today's open yet" became "so they lack today's open so far", same fact, because ml4_test's
# banned-phrase check, which keeps model-facing tool text free of plain imperatives, flags "do not" in any output.)
_PLAIN = (
    ("session ranges unavailable (SPY's 30-minute bars are unavailable)",
     "session ranges not yet available (our delayed bars run about 15 minutes behind, so they lack today's open so far)"),
    ("market now: SPY's 30-minute bars unavailable (ValueError: no opening print or previous close yet)",
     "market now: not yet available (our delayed bars run about 15 minutes behind, so they lack today's open so far)"),
)


# ML4.2 (from Mon 23 Nov 2026; the Trading Model Researcher's decision of 20 Nov, a fix): the session ranges are a
# half-hour slot's figures (ml_research.intraday.live: inputs through the slot's last bar, priced at its close), but
# forecast._session labels them with the minute of the call. At a quarter-hour look (a 15-minute sleeve's 09:45, 10:15 ...)
# that is a false statement: the 09:45 call reads 09:30's figures. A call in the second half of a half-hour slot (minute
# 15-29 or 45-59 past it) has its label set to the slot the figures are from; a call in the first half (every on-time
# half-hour look) reads exactly as before. forecast.py, in ML2's frozen manifest, is unchanged.
_SLOT_LABEL = re.compile(r"session ranges at the (\d\d):(\d\d) look")


def _slot_label(text):
    def fix(m):
        mins = int(m.group(1)) * 60 + int(m.group(2))
        if mins < 9 * 60 + 30 or (mins - (9 * 60 + 30)) % 30 < 15:
            return m.group(0)
        k = min((mins - (9 * 60 + 30)) // 30, len(_SLOTS) - 1)
        return f"session ranges at the {_SLOTS[k]} look"
    return _SLOT_LABEL.sub(fix, text, count=1)


_SLOTS = [f"{9 + (30 + 30 * k) // 60:02d}:{(30 + 30 * k) % 60:02d}" for k in range(13)]   # 09:30 .. 15:30, as intraday.SLOTS


def _plain(text):
    """Only when the open-specific error is in the text (the market-now line's "no opening print or previous close yet"):
    the session-ranges line alone also comes from other causes later in the day (a bars timeout, a Gateway error, too few
    past sessions; e.g. opus-ml 11:30 on 10 Nov), and then its plain cause would be false (the review of 16 Nov evening)."""
    if _PLAIN[1][0] not in text:
        return text
    for old, new in _PLAIN:
        text = text.replace(old, new)
    return text


def _run(gw, args, sleeve_value=None, look=None):
    """forecast3.run's steps (ML3.1, 0576413) with ML4's paths, legend lines and absr1 fix. Numbers are ML3.1's."""
    t0 = time.monotonic()
    text = forecast.run(gw, args, sleeve_value=sleeve_value, look=look)
    lines = text.split("\n")
    if not lines[0].startswith("as of the close of "):
        return text
    sim = hasattr(gw, "forecast_provider")
    as_of = lines[0][19:29].replace("-", "")
    syms = args.get("symbols") or []
    syms = list(dict.fromkeys(str(s).upper().strip() for s in ([syms] if isinstance(syms, str) else syms)
                              if str(s).strip()))[:forecast.MAX_SYMBOLS]
    try:
        M, fc = load_model(), _fc4(as_of, sim)
        if fc is None:
            raise FileNotFoundError("no ML4 forecasts for this close")
    except Exception as e:                              # noqa: BLE001 - ML2's text is still served, gap1 na
        M, fc = None, None
        lines.insert(1, f"gap1 unavailable ({type(e).__name__}: {str(e)[:60]})")
    night = (fc or {}).get("night") or [None, None]
    if fc and not sim and night[0] and night[0] != _now().strftime("%Y-%m-%d"):   # a failed refresh: yesterday's night
        lines.insert(1, f"gap1 unavailable (its forecasts are for the night after the {night[0]} close)")
        M, fc, night = None, None, [None, None]
    ci = next((i for i, x in enumerate(lines) if x.startswith("sym close range1")), None)
    if ci is None:
        return text
    for i in range(ci):                                 # ML2's legend lines in ML4's words (numbers unchanged)
        if lines[i].startswith("No column forecasts direction"):
            lines[i] = DIR_LINE
        elif lines[i].startswith("move$ ±(90) and sleeve% ±(90):") and LOSS_CUT in lines[i]:
            lines[i] = lines[i].split(LOSS_CUT, 1)[0].rstrip()
    rows = {}
    for i in range(ci + 1, len(lines)):
        s = max((s for s in syms if lines[i] == s or lines[i].startswith(s + " ")), key=len, default=None)
        if s and not lines[i][len(s) + 1:].startswith("not covered"):
            rows[i] = s
    have = dict((fc or {}).get("rows") or {})
    need = [s for s in rows.values() if s not in have]
    if need and fc and not sim:
        have.update(_outside_rows(gw, need, as_of, min(f3.ONDEMAND_S, max(0.0, t0 + f3.CALL_S - time.monotonic()))))
    mk = (fc or {}).get("market") or {}
    mac = (fc or {}).get("macro") or []
    dirs = direction_column(as_of, list(rows.values()))     # ML4.0: None, so no column and no text
    events = []
    # value tokens a row has after its symbol once gap1 (and a direction column) is in: ML2's header less sym and note, + 1
    n_vals = len(lines[ci].split(" ")) - 2 + 1 + (1 if dirs else 0)
    for i, s in rows.items():
        tok = "na"
        r = have.get(s)
        X = np.array([[r[0], r[1], mk.get("lmkt5"), mk.get("lv5"), mk.get("long")]], float) if r else None
        if M and X is not None and np.all(np.isfinite(X)):
            e = f3._earn(lines[i], night)
            q = f3.quantiles(M, X, f3.multiplier(M, e, bool(mac))[None])[0]
            tok = f"{f3._num(q[1])}({f3._num(q[0])}-{f3._num(q[2])})"
            if e:
                events.append(f"{s} earnings ({'Alpha Vantage calendar' if e == 'earn' else 'our estimate'})")
        p = lines[i][len(s) + 1:].split(" ", 3)
        if len(p) >= 3:
            rest = p[3:]
            a_tok, clause = fix_absr1(p[2], rest[0].split(" ", 1)[0] if rest else None)
            extra = [str(dirs.get(s, "na"))] if dirs else []
            line = " ".join([s, p[0], p[1], a_tok, tok] + extra + rest)
            if clause:
                has_note = len(line[len(s) + 1:].split(" ")) > n_vals
                line = f"{line}; {clause}" if has_note else f"{line} {clause}"
            lines[i] = line
    lines[ci] = lines[ci].replace("absr1±% ", "absr1±% gap1±%(10-90) " + ("dir " if dirs else ""), 1)
    short = any(ln.endswith(forecast.LEGEND_REF) for ln in lines[:ci])
    add = [] if short else [GAP1_LINE]
    if night[0] and (events or mac):
        add.append(f"gap1 events tonight (after the {night[0]} close, before the {night[1]} open): "
                   + "; ".join(events + ([f"{', '.join(mac)} before the open (every name)"] if mac else [])))
    lines[ci:ci] = add
    mi = next((i for i, x in enumerate(lines) if x.startswith("market:")), None)
    now_line = f3._market_now(gw, lines[mi], fc, sim, _now()) if mi is not None else None
    if now_line:
        lines.insert(mi + 1, now_line)
    return "\n".join(lines)
