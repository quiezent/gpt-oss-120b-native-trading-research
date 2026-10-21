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
No return or direction forecast ships: none we tested beat trading costs out of sample.

Live: compute() runs after the 08:15 refresh (live.py) and caches runs/ml/forecast/fc_<as_of>.json; run() reads the
latest file, prices orders at the requested size, and costs symbols outside the cached universe from the gateway
(cost.live), and adds the events correction (EDGAR fetched at look time) and the macro line. Training: sim_provider(data)
serves strictly out-of-sample predictions as of the close before the decision, with the same correction, note and line.
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "ml" / "forecast"
RES = ROOT / "runs" / "ml" / "research"
MAX_SYMBOLS = 20
DEFAULT_NOTIONAL = 3000.0

# The tool description, verbatim from runs/ml/research/synthesis_wave_c.md (gpt-oss sees each line as a "// " comment).
DESCRIPTION = """Risk and cost forecasts from our machine-learning models, plus today's company filings and macro calendar: how much a stock is likely to move and what a round trip costs, not which way it will go. Use them to size positions, to set stop and limit distances, and to skip trades whose cost eats much of the expected move.

**Header lines:**
- **market:** expected SPY volatility over the next 20 and 5 sessions, annualized %, with 10-90% ranges. Also size_scale = typical SPY vol / expected vol, which is about how much gross exposure keeps your risk at its usual level.
- **macro:** today's scheduled US releases (ET), with the first-release value once it is in our data and the Kalshi pre-release expectation, plus the next week's CPI, payrolls, PCE, GDP and FOMC. It is a calendar, not a forecast.

**Columns as of the last close:**
- range1: expected high-low range of the next session, % of price, with its 10-90% range. A stop closer than about one range1 is likely to be hit by normal noise.
  - If the company filed an SEC 8-K since the previous close (or in the last 24 h) up to 09:00 ET, range1 is corrected for it, and the note gives the uncorrected value. Most of the correction is earnings.
- absr1: expected size of the next close-to-close move, %.
- vol20: expected annualized volatility over the next 20 sessions, %. It is mostly the stock's own option-implied vol.
- spread: expected time-average quoted spread next session, bps, with its 10-90% range.
- cost: round-trip cost at notional_usd, bps, with its 10-90% range. It covers the spread, impact on entry and exit, and commissions ($0.005 a share, $1 minimum per order, so small orders pay more).
- tier: liquidity tier, A best to D thinnest.
- adv: average daily dollar volume, $ millions.
- note: the latest 8-K of the last 7 days, with its acceptance time (ET) and item codes (2.02 earnings, 1.01 material agreement, 2.01 acquisition done, 5.02 officer change, 3.02 unregistered equity sale, 7.01/8.01 press releases), and the next earnings date estimate when it falls within 10 sessions. The estimate is last year's same-quarter report plus 52 weeks.

**Session columns (09:30-16:00 ET), from today's 30-minute bars:**
- next60: expected high-low range over the next 60 minutes, % of the current price, with its 10-90% range.
- rest_of_day: the same through the close. A stop inside rest_of_day's 10% value is likely to be touched by noise.

**No return or direction forecast is offered.** Out of sample, none of the models we tested beat trading costs. That covers overnight gaps, intraday momentum, reversal, volume and time-of-day, news sentiment, industry, factor timing, momentum ranking and market timing. Our anomaly flag did not predict drops. Earnings drift was not tested.

**Track record** (out of sample; per-symbol figures cover today's S&P 500 stocks and ETFs, so they are survivor-biased):
- **range1:** R2 0.37 vs 0.29 for the 20-day average range (Jan 2025-Oct 2026). A plain recent-range average does nearly as well. Its 10-90% range held 81% of outcomes, but only 76% in the April 2025 shock half-year.
- **range1 after an 8-K:** uncorrected, R2 was 0.02 and the 10-90% range held 65%; corrected, R2 0.51 and 80% (9,603 stock-sessions, Mar 2025-Oct 2026, ahead in every half-year). Other sessions are unchanged.
  - The morning after an earnings release, the realized range averaged 2.1x the uncorrected forecast, which held only 27%; corrected, it held 80%. Other 8-K types change almost nothing.
  - On sessions that went on to gap more than 5%, the corrected range held 70% (47% uncorrected).
  - 8-Ks accepted 09:00-09:30 ET are not counted.
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
- **Names outside the S&P list:** no risk columns. Their cost comes from live data and is rough."""

SPEC = {
    "name": "forecast",
    "description": DESCRIPTION,
    "parameters": {"type": "object", "properties": {
        "symbols": {"type": "array", "items": {"type": "string"}, "description": f"Up to {MAX_SYMBOLS} symbols."},
        "notional_usd": {"type": "number", "description": f"Order size to cost, $ (default {DEFAULT_NOTIONAL:,.0f})."}},
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
    from ml_research import vol2
    v = vol2.live(gw=None, data=data, no_stale_check=True)      # all cached symbols; HAR fallback flagged per row
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


def _prev_session(today):
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


# ---------------------------------------------------------------- training simulator
def sim_provider(data):
    """A function of decision session d -> (as_of, {"market", "rows"}) from strictly out-of-sample predictions at d - 1."""
    from ml_research import events, macro_feats, regime, vol2
    risk_at = vol2.sim_provider(data)
    ev_at, mac_at = events.sim_provider(data), macro_feats.sim_provider(data)
    with np.load(RES / "cost_oos.npz", allow_pickle=True) as z:      # read each array once (an NpzFile re-reads on every [])
        c = {k: z[k] for k in z.files}
    reg = regime.sim_provider(data)
    ccol, crow = {str(x): i for i, x in enumerate(c["dates"])}, {str(s): j for j, s in enumerate(c["symbols"])}

    def at(d):
        as_of = data.dates[d - 1]
        got = risk_at(d)
        if not got:
            return None
        ci = ccol.get(as_of)
        rows = {}
        for s, rr in got[1].items():
            row = {k: rr.get(k) for k in ("close", "range1_pct", "range1_lo", "range1_hi", "absr1_pct", "vol20_pct")}
            old = [t for t, m in rr.get("model", {}).items() if m == "har" and t != "range1"]   # range1 is HAR by design
            row["note"] = f"no implied vol or intraday data: previous model for {', '.join(old)}" if old else ""
            cj = crow.get(s)
            if ci is not None and cj is not None and np.isfinite(c["spread"][cj, ci]):
                row.update({"spread": _r(c["spread"][cj, ci], 2), "spread_lo": _r(c["spread_lo"][cj, ci], 2),
                            "spread_hi": _r(c["spread_hi"][cj, ci], 2), "adv_usd": _r(c["adv_usd"][cj, ci], 0),
                            "sigma_d": _r(c["sigma_d"][cj, ci], 5)})
            rows[s] = row
        ev = (ev_at(d) or (None, {}))[1]
        for s, row in rows.items():
            if s in ev:
                _apply_event(row, ev[s])
        m = reg(d) or {}
        market = {k: m.get(k) for k in ("spy_vol20_pct", "spy_vol20_10_90_pct", "spy_vol5_pct", "spy_vol5_10_90_pct",
                                        "typical_spy_vol_pct", "size_scale_vs_typical")}
        market["macro"] = mac_at(d)[1]["macro_today"]
        return (as_of, {"market": market, "rows": rows}) if rows else None
    return at


def _apply_event(row, ev):
    """The events correction and note on one forecast row (live and simulator): when range1_event applies, the range1
    triplet becomes the corrected one and the note gives the uncorrected value."""
    from ml_research import events
    if ev.get("range1_event_pct") is not None:
        row.update(range1_pct=ev["range1_event_pct"], range1_lo=ev["range1_event_lo"], range1_hi=ev["range1_event_hi"])
    row["note"] = "; ".join(x for x in (row.get("note"), events.note(ev)) if x)


# ---------------------------------------------------------------- the tool
def _cost_row(r, notional):
    from ml_research import cost
    price = r.get("close")
    if not price or r.get("spread") is None or not r.get("adv_usd"):
        return None
    qty = max(1, round(notional / price))
    em = (r["absr1_pct"] / 100) if r.get("absr1_pct") else math.sqrt(2 / math.pi) * (r.get("sigma_d") or 0)
    return cost.order_cost(price, qty, r["spread"], r["spread_lo"], r["spread_hi"], r["adv_usd"], r.get("sigma_d") or 0, em)


def _session(gw, syms):
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
        out = intraday.live(gw, symbols=syms, now=now)
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
    sim = hasattr(gw, "forecast_provider")
    got = gw.forecast_provider() if sim else live_provider()
    if not got:
        return "no forecasts are available for this date"
    as_of, fc = got
    head = f"as of the close of {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
    if not sim and as_of < _prev_session(date.today()):
        head += " (stale: not from the last session)"
    rows = fc.get("rows") or {}
    covered = [s for s in syms if s in rows]
    snote, sess = ("session ranges from 09:30 ET", {}) if sim else _session(gw, covered)
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
    if m.get("spy_vol20_pct"):
        rng = lambda k: "-".join(f"{x:g}" for x in (m.get(k) or []) if x is not None) or "na"
        mline = (f"market: SPY vol20 {m['spy_vol20_pct']:g}% (10-90 {rng('spy_vol20_10_90_pct')}) vol5 "
                 f"{m.get('spy_vol5_pct') or 'na'}% (10-90 {rng('spy_vol5_10_90_pct')}) typical {m.get('typical_spy_vol_pct') or 'na'}% "
                 f"size_scale {m.get('size_scale_vs_typical') or 'na'}")
    else:
        mline = "market: no SPY implied vol today"
    cols = "sym close range1%(10-90) absr1% vol20% spread_bps(10-90) cost_bps(10-90) tier adv_$m"
    if sess:
        cols += " next60%(10-90) rest_of_day%(10-90)"
    lines = [head, mline] + ([m["macro"]] if m.get("macro") else []) + [f"costs for a ${notional:,.0f} order per symbol",
                                                                        cols + " note"]
    f = lambda x: "na" if x is None else f"{x:g}"
    outside = []
    for s in syms:
        r = rows.get(s)
        if not r:
            outside.append(s)
            continue
        c = _cost_row(r, notional)
        cc = (f"{f(c['spread_bps'])}({f(c['spread_bps_10'])}-{f(c['spread_bps_90'])}) {f(c['cost_bps'])}({f(c['cost_bps_10'])}-"
              f"{f(c['cost_bps_90'])}) {c['tier']} {f(c['adv_usd_m'])}") if c else "na na na na"
        line = (f"{s} {f(r.get('close'))} {f(r.get('range1_pct'))}({f(r.get('range1_lo'))}-{f(r.get('range1_hi'))}) "
                f"{f(r.get('absr1_pct'))} {f(r.get('vol20_pct'))} {cc}")
        note = r.get("note") or ""
        if sess:
            x = sess.get(s)
            if isinstance(x, tuple):
                line += f" {f(x[0])}({f(x[1])}-{f(x[2])}) {f(x[3])}({f(x[4])}-{f(x[5])})"
            else:
                line += " na na"
                note = "; ".join(t for t in (note, x or "no session range") if t)
        lines.append(f"{line} {note}".rstrip())
    if outside:
        if sim:
            lines += [f"{s} not covered" for s in outside]
        else:
            from ml_research import cost
            try:
                res = cost.live(gw, symbols=outside, notional=notional)["orders"]
            except Exception as e:                      # noqa: BLE001
                res = [{"symbol": s, "error": f"{type(e).__name__}: {e}"[:120]} for s in outside]
            for o in res:
                if o.get("error"):
                    lines.append(f"{o['symbol']} not covered ({o['error']})")
                    continue
                lines.append(f"{o['symbol']} {f(o.get('price'))} na na na {f(o['spread_bps'])}({f(o['spread_bps_10'])}-"
                             f"{f(o['spread_bps_90'])}) {f(o['cost_bps'])}({f(o['cost_bps_10'])}-{f(o['cost_bps_90'])}) "
                             f"{o['tier']} {f(o['adv_usd_m'])}{' na na' if sess else ''} not in the risk universe; cost from "
                             "live data (rough)")
    return "\n".join(lines)
