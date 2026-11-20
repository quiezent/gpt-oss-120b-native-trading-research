"""ML3 of the forecast tool: ML2's output (forecast.py, called unchanged) plus the two things ML2 lacks, the overnight gap
and a market line refreshed during the session. Pre-registration and results: runs/ml/research/ml3.md.

  - gap1 per name: the expected size of the gap from today's close to the next open, |open / close - 1| %, with a 10-90%
    band. Linear quantile regressions on the name's own past gaps (60- and 250-night medians), the market's recent gaps,
    SPY's 5-session realized vol and a weekend flag (ml3_validate.py), times an event multiplier for an earnings release
    in the night (read from ML2's own note) and a tier-1 US release before 09:30. A size, never a direction.
  - market now: SPY's move since the last close (a pre-market quote before 09:30, today's 30-minute bars after). The
    size_scale_now built on it failed its test, so it is computed (NOW_SHIPS) but not shown.

ML3 never writes into ML2's folders: its cache is runs/ml/forecast3/ (fc_<as_of>.json and ondemand/), its model
runs/ml/research/ml3_model_amend1.json (written by ml3_validate.py --amend1). It reads ML2's cache through forecast.run,
so ML2's compute() must have run for the same close (live.compute_versions computes the default version first).
"""

import bisect
import json
import math
import re
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import forecast

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "ml" / "forecast3"        # ML3's own cache: ML2 takes the newest fc_*.json in runs/ml/forecast
ONDEMAND = OUT / "ondemand"                     # <as_of>/<SYM>.json: gap1 rows for names outside the daily-bar cache
MODEL = ROOT / "runs" / "ml" / "research" / "ml3_model_amend1.json"   # Amendment 1's fit (ml3.md): the pre-registered
                                                # 2024-only fit, ml3_model.json, missed its coverage bar
BARS = ROOT / "data" / "bars"
BARS30 = ROOT / "data" / "bars30"
MACRO = ROOT / "data" / "macro" / "calendar.json"
VERSION = "ML3.2"                               # recorded on looks, not shown to models (ML3.1: ML3 review fixes 1-4, 15 Nov;
                                                # ML3.2: ML2.4's two text cuts, which reach it through forecast.py; no number changes)
DELAY_MIN = 15                                  # IBKR's delayed data: the loop's header says prices are about 15 minutes old
TAUS = (0.1, 0.5, 0.9)
FEATS = ["lm60", "lm250", "lmkt5", "lv5", "long"]
FLOOR = 0.001                                   # %, the log floor for a gap of zero
TIER1 = {"cpi", "nfp", "pce", "gdp"}            # the macro multiplier's releases (before 09:30 ET on the next session)
SLOTS = [f"{9 + (30 + 30 * k) // 60:02d}:{(30 + 30 * k) % 60:02d}" for k in range(13)]   # 09:30 .. 15:30
G_DAYS, G_MIN = 60, 20                          # size_scale_now's normalizer: median of the previous 60 sessions (20 at least)
ONDEMAND_S = 15.0                               # gateway time for outside names' daily bars after ML2's call, within
CALL_S = 45.0                                   # F12's 45 s for the whole call (forecast.ONDEMAND_BUDGET_S leaves ML2 5 s)
_MEM = {}                                       # as_of -> fc3 document, read (live) or built (simulator) once per process


# ---------------------------------------------------------------- features (shared by compute(), run() and ml3_validate.py)
def gaps(o, c):
    """|ON| in %, per name and session: |log(open_t / close_{t-1})|, NaN where a bar is missing (column 0 is NaN)."""
    with np.errstate(all="ignore"):
        a = np.abs(np.log(o[..., 1:] / c[..., :-1])) * 100
    return np.concatenate([np.full(a.shape[:-1] + (1,), np.nan), a], axis=-1)


def tmed(x, n, need, last_only=False):
    """Trailing median over the last n columns through each column, NaN with fewer than `need` values."""
    out = np.full(x.shape, np.nan)
    for t in ([x.shape[-1] - 1] if last_only else range(x.shape[-1])):
        w = x[..., max(0, t - n + 1):t + 1]
        ok = np.sum(np.isfinite(w), axis=-1) >= need
        if np.any(ok):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN rows: masked just below
                out[..., t] = np.where(ok, np.nanmedian(w, axis=-1), np.nan)
    return out


def name_feats(A, last_only=False):
    """log m60 and log m250 of each name's past gaps (A from gaps()), through each column."""
    f = lambda n, need: np.log(np.maximum(tmed(A, n, need, last_only), FLOOR))
    return f(60, 40), f(250, 60)


def market_feats(A, stocks, spy_c):
    """Per column: log mkt5 (the mean of the last 5 nights' cross-sectional median gap of stocks, over that series'
    250-night median) and log v5 (SPY's 5-session realized vol over its 250-session median)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cnt = np.sum(np.isfinite(A[stocks]), axis=0)
        mk = np.where(cnt >= 100, np.nanmedian(A[stocks], axis=0), np.nan)
    n = len(mk)
    # at least 3 of the 5 nights: a close the morning refresh has only partly reached leaves the last one out
    mk5 = np.array([np.nanmean(mk[t - 4:t + 1]) if t >= 4 and np.sum(np.isfinite(mk[t - 4:t + 1])) >= 3 else np.nan
                    for t in range(n)])
    with np.errstate(all="ignore"):
        lr = np.r_[np.nan, np.log(spy_c[1:] / spy_c[:-1])]
    rv5 = np.array([math.sqrt(np.mean(lr[t - 4:t + 1] ** 2)) if t >= 5 else np.nan for t in range(n)])
    with np.errstate(all="ignore"):
        return np.log(mk5 / tmed(mk, 250, 60)), np.log(rv5 / tmed(rv5, 250, 60))


def load_model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _lin(m, X):
    return m["b"][0] + ((X - np.array(m["mu"])) / np.array(m["sd"])) @ np.array(m["b"][1:])


def quantiles(M, X, mult=None):
    """gap1's 10, 50 and 90% values (%), sorted, for rows X (columns FEATS), times mult (rows x 3) when given."""
    q = np.column_stack([np.exp(_lin(M["base"][str(t)], X)) for t in TAUS])
    if mult is not None:
        q = q * mult
    return np.sort(q, axis=1)


def multiplier(M, earn=None, macro=False):
    """The event multipliers (one per τ) for a night: earn is "earn" (a calendar names the night), "earn_e1" (ML2's E1
    estimate does) or None. Multipliers that did not ship are 1."""
    m = np.ones(3)
    for e in ([earn] if earn else []) + (["macro"] if macro else []):
        if M["ship"].get(e):
            m = m * np.array(M["mult"][e])
    return m


# ---------------------------------------------------------------- calendar
def _holidays():
    try:
        from live import HOLIDAYS, HALF_DAYS            # inside: live imports the tools
        return set(HOLIDAYS), set(HALF_DAYS)
    except Exception:                                    # noqa: BLE001
        return set(), set()


def next_session(day):
    """The session after `day` ("YYYY-MM-DD"): ML2's trading calendar (past sessions from SPY's bars, NYSE holidays
    through 2027), else weekdays less live.HOLIDAYS."""
    try:
        from ml_research import events
        cal = events.calendar()[0]
        i = bisect.bisect_right(cal, day)
        if i < len(cal):
            return cal[i]
    except Exception:                                    # noqa: BLE001
        pass
    hol = _holidays()[0]
    d = date.fromisoformat(day) + timedelta(days=1)
    while d.weekday() >= 5 or d.isoformat() in hol:
        d += timedelta(days=1)
    return d.isoformat()


def macro_before_open(day, rows=None):
    """The tier-1 releases (CPI, payrolls, PCE, GDP) on session `day` before 09:30 ET, as "name HH:MM ET" texts."""
    if rows is None:
        try:
            rows = json.loads(MACRO.read_text(encoding="utf-8"))["events"]
        except (OSError, ValueError, KeyError):
            return None
    return [f"{r.get('name') or r['key']} {r['time_et']} ET" for r in rows if r.get("date") == day and r.get("key") in TIER1
            and r.get("kind") != "revision_only" and (r.get("time_et") or "99:99") < "09:30"]


def _iso(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


# ---------------------------------------------------------------- size_scale_now
def so_far(bars, k, gap):
    """SPY's range since the last close at look slot k (log): today's 30-minute bars {HH:MM: bar}, completed bars only,
    and the opening gap (log) to place the previous close. k = 0 (the 09:30 look) is the opening gap alone."""
    o = bars["09:30"]["open"]
    c0 = o * math.exp(-gap)
    done = [bars[s] for s in SLOTS[:k] if s in bars]
    hi = max([c0, o] + [b["high"] for b in done])
    lo = min([c0, o] + [b["low"] for b in done])
    return math.log(hi / lo)


def weight(k):
    """The realized move's weight at slot k: the morning forecast counts as two bars, the opening gap as a quarter bar."""
    return (k + 0.25) / (k + 2.25)


def scale_now(size_scale, sigma_m, x, g, k):
    """size_scale × z^(-w): z is the move so far over what the morning daily vol sigma_m implies by slot k (g, the
    trailing median of x / sigma_m), floored at 0.25 as pre-registered (a quiet morning raises the scale at most
    0.25^-w: 1.2x at 09:30, 3.3x at 15:30)."""
    z = max(x / (sigma_m * g), 0.25)
    return size_scale * z ** -weight(k), z


def by_day(rows):
    """30-minute bars -> {YYYYMMDD: {HH:MM: bar}}."""
    out = {}
    for b in rows:
        p = str(b["date"]).split()
        if len(p) >= 2:
            out.setdefault(p[0].replace("-", ""), {})[p[1][:5]] = b
    return out


def g_norm(days30, gap, sigma, upto):
    """The normalizer G_k for each slot: the median of so_far / sigma_m over the previous G_DAYS full sessions before
    `upto` (YYYYMMDD) that have the opening gap and the morning vol. None for a slot with fewer than G_MIN."""
    past = [d for d in sorted(days30) if d < upto and len(days30[d]) == 13 and d in gap and sigma.get(d)][-G_DAYS:]
    out = []
    for k in range(13):
        v = [so_far(days30[d], k, gap[d]) / sigma[d] for d in past]
        out.append(float(np.median(v)) if len(v) >= G_MIN else None)
    return out


def morning_sigma(iv, dates, q1):
    """{day: the morning daily vol}: SPY implied vol at the previous close times exp(q1), as regime.live serves vol20."""
    return {d: iv[p] * math.exp(q1) / math.sqrt(252) for p, d in zip(dates[:-1], dates[1:]) if iv.get(p)}


# ---------------------------------------------------------------- live
def _load_bars(sym):
    p = BARS / f"{sym.replace(' ', '_')}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))["bars"]
    except (OSError, ValueError, KeyError):
        return None


def _row_from_bars(rows, as_of):
    """log m60 and log m250 from one name's daily bars through as_of, or None."""
    rows = [b for b in rows if str(b["date"])[:8] <= as_of and b.get("open") and b.get("close")]
    if len(rows) < 61 or str(rows[-1]["date"])[:8] != as_of:
        return None
    o = np.array([b["open"] for b in rows], float)
    c = np.array([b["close"] for b in rows], float)
    l60, l250 = name_feats(gaps(o, c)[None, :], last_only=True)
    return (float(l60[0, -1]), float(l250[0, -1])) if np.isfinite(l60[0, -1]) and np.isfinite(l250[0, -1]) else None


def _spy_market(data, t):
    """SPY's size_scale_now inputs at close t of `data`: G_k for the next session and the last close."""
    from ml_research import regime
    q1 = json.loads((forecast.RES / "regime.json").read_text(encoding="utf-8"))["intervals"]["vol20"]["live_q"][1]
    j = data.k["SPY"]
    dates = data.dates[:t + 1]
    gap = {dates[i]: math.log(data.o[j, i] / data.c[j, i - 1]) for i in range(1, t + 1)
           if np.isfinite(data.o[j, i]) and np.isfinite(data.c[j, i - 1])}
    sigma = morning_sigma(regime.load_iv(), dates, q1)
    try:
        days30 = by_day(json.loads((BARS30 / "SPY.json").read_text(encoding="utf-8"))["bars"])
    except (OSError, ValueError, KeyError):
        days30 = {}
    nxt = next_session(_iso(dates[-1])).replace("-", "")     # G_k for the next session: sessions through close t
    return {"g": g_norm(days30, gap, sigma, nxt), "spy_close": float(data.c[j, t]), "q1": q1}


def build(data, t=None):
    """The fc3 document for close t of `data` (default: its last): each name's gap1 inputs (log m60, log m250), the
    market features, the night, its weekend and macro flags, and SPY's size_scale_now normalizer."""
    t = len(data.dates) - 1 if t is None else t
    as_of = data.dates[t]
    o, c = data.o[:, :t + 1], data.c[:, :t + 1]
    A = gaps(o, c)
    stocks = np.array([s not in data.etf for s in data.sym])
    lmkt5, lv5 = market_feats(A, stocks, c[data.k["SPY"]])
    l60, l250 = name_feats(A, last_only=True)
    day = next_session(_iso(as_of))                 # tonight: after this session's close, before the next open
    nxt = next_session(day)
    long = int((date.fromisoformat(nxt) - date.fromisoformat(day)).days > 1)
    mac = macro_before_open(nxt) or []
    mk = {"lmkt5": float(lmkt5[-1]), "lv5": float(lv5[-1]), "long": long}
    rows = {}
    for s, j in data.k.items():
        if np.isfinite(l60[j, -1]) and np.isfinite(l250[j, -1]) and np.isfinite(c[j, -1]):
            rows[s] = [float(l60[j, -1]), float(l250[j, -1])]
    doc = {"version": VERSION, "as_of": as_of, "night": [day, nxt], "macro": mac, "market": mk, "rows": rows}
    try:
        doc["spy"] = _spy_market(data, t)
    except Exception as e:                          # noqa: BLE001 - size_scale_now says it is unavailable
        doc["spy"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return doc


def compute(data=None):
    """After the daily refresh and ML2's compute(): ML3's gap1 inputs and size_scale_now's normalizer from the latest
    cached close, in runs/ml/forecast3/fc_<as_of>.json. Names in data/bars outside the universe are added when their
    last bar is this close. Returns (as_of, rows)."""
    import trader
    data = data or trader.Data()
    doc = build(data)
    as_of = doc["as_of"]
    for p in BARS.glob("*.json"):
        s = p.stem.replace("_", " ")
        if s not in doc["rows"]:
            r = _row_from_bars(_load_bars(s) or [], as_of)
            if r:
                doc["rows"][s] = list(r)
    if not (forecast.OUT / f"fc_{as_of}.json").exists():
        doc["ml2_missing"] = True                   # run() still serves ML2's latest file, which says it is stale
    OUT.mkdir(parents=True, exist_ok=True)
    forecast._save(OUT / f"fc_{as_of}.json", doc)
    _MEM.clear()                                    # this process's looks read the new document
    _MEM[as_of] = doc
    return as_of, doc["rows"]


def _fc3(as_of, sim):
    """The fc3 document for ML2's as_of: the cached file (live) or one built in memory (simulator), once per process."""
    if as_of not in _MEM:
        if sim:
            import trader
            doc = build(trader.Data(last=as_of))
        else:
            try:
                doc = json.loads((OUT / f"fc_{as_of}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        _MEM.clear()
        _MEM[as_of] = doc
    return _MEM[as_of]


# ---------------------------------------------------------------- the tool
NOW_SHIPS = False       # size_scale_now failed its pre-registered test (ml3.md): the line gives SPY's move, a fact, only
CAL = re.compile(r"next earnings (\d{4}-\d{2}-\d{2}) \((after close|before open); Alpha Vantage calendar\)")
EST = re.compile(r"next earnings est\. (\d{4}-\d{2}-\d{2}) \((after close|before open)\)")

DESCRIPTION = forecast.DESCRIPTION + """

**ML3 adds an overnight gap column and a market-now line:**
- gap1 ±(10-90), after absr1: the expected size of the gap from today's close to the next open, up or down, % of the close: the median, with the 10-90% range of the gap's size (not a price bound). range1 leaves the gap out and absr1 includes it; gap1 is the gap alone. It comes from the name's own past gaps, the market's recent gaps and SPY's recent volatility, as of the last close, and it does not change during the day.
  - It is raised when the note names an earnings release after today's close or before the next open: about ×6.4 at the median when the date comes from the Alpha Vantage calendar, ×2.7 when it is our estimate (right about half the time, so its range is wide). A line above the columns names them.
  - It is raised about ×1.14 for every name when CPI, payrolls, PCE or GDP comes out before the next open.
- market now: SPY's move since the last close: before 09:30 from a pre-market quote, from 09:30 to 16:00 from today's 30-minute bars, with its range since the close and the time (ET) of the delayed print it uses. It is a fact, not a forecast. A size scale refreshed from it did not beat the morning size_scale out of sample, so size_scale stays the morning value.

**gap1 track record** (out of sample, 2 Jan 2025-6 Nov 2026, 523 names, 241,993 name-nights; survivor-biased): {track}"""
TRACK = ("the 10-90% range held 78.7% of gaps, and 47.8% fell below the median. The pinball loss was 7% lower than for the "
         "stock's own 60-night median gap with a calibrated range (which held 78.7%). On earnings nights the range held 81.6%, "
         "against about 21% without the earnings correction. Through 16 Oct 2026 the test used the realized report times as "
         "the calendar, which is optimistic; on the 349 Alpha Vantage calendar nights of 19 Oct-6 Nov 2026 it held 84%, with "
         "the median too high (65% of gaps fell below it). On nights our estimate named, it held 80% (43% without the "
         "correction); on CPI, payroll, PCE and GDP nights, 79% (73% without). On half of all nights it held 77-85% of "
         "names. It runs narrow on shock nights: 14% of names at the 8 Apr 2025 open, 17% at 3 Mar 2026's, 72% at 5 Nov "
         "2026's and 57% at 6 Nov 2026's (67 names), when chips gapped 6-13% against ranges topping out at 4.9-7.7%. In the "
         "test the models were refit monthly on every past night; today's fit uses every night through 6 Nov 2026. This "
         "method was revised after a first version, fit on 2024 only, missed its coverage bar (76.5%), and it was checked "
         "on the same period. Outside the validated universe (a first check on 737 NASDAQ-listed stocks, after the fact): "
         "the 10-90% range held about 73% of gaps, against 78.5% on our stocks, and gaps passed its top more often (about "
         "13-14% of nights against a 10% target); thinly traded names often open unchanged, and on these names the stock's "
         "own 60-night median gap did as well.")
SPEC = {"name": "forecast", "description": DESCRIPTION.replace("{track}", TRACK), "parameters": forecast.SPEC["parameters"]}


def _now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _num(x):
    return "na" if x is None or not math.isfinite(x) else f"{round(x, 2):g}"


def _earn(line, night):
    """The night's earnings source from ML2's note on the row: "earn" (Alpha Vantage calendar) or "earn_e1" (our
    estimate) when it names an after-close release on night[0] or a before-open one on night[1]; else None."""
    for kind, rx in (("earn", CAL), ("earn_e1", EST)):
        for day, when in rx.findall(line):
            if (day, when) in ((night[0], "after close"), (night[1], "before open")):
                return kind
    return None


def _outside_rows(gw, syms, as_of, budget):
    """gap1 inputs for names with no cached daily bars: a year of IBKR daily bars (the request ML2's vol2 makes for
    an outside name, so forecast's 15-minute memo serves it), kept per close in ONDEMAND/<as_of>/<SYM>.json."""
    import bars as _bars
    out, deadline = {}, time.monotonic() + budget
    for s in syms:
        if not forecast.SYM_OK.fullmatch(s):
            continue
        p = ONDEMAND / as_of / f"{s.replace(' ', '_')}.json"
        try:
            out[s] = json.loads(p.read_text(encoding="utf-8"))["row"]
            continue
        except (OSError, ValueError, KeyError):
            pass
        rows = _load_bars(s)
        r = _row_from_bars(rows, as_of) if rows else None
        if r is None:                                   # the memo answers at once; a new request only while time is left
            try:
                g = forecast._Capped(gw, deadline).bars(s, duration="1 Y", bar_size="1 day", what="ADJUSTED_LAST")
                r = _row_from_bars(_bars.completed([dict(b, date=str(b["date"])[:8]) for b in g or []]), as_of)
            except Exception:                           # noqa: BLE001 - the time cap's refusal too: the row shows na
                r = None
        if r:
            out[s] = list(r)
            forecast._save(p, {"symbol": s, "as_of": as_of, "row": out[s]})
    return out


def _market_now(gw, mline, fc3, sim, now):
    """The market-now line: SPY's move since the last close (pre-market quote before 09:30, today's 30-minute bars in
    the session) and, if NOW_SHIPS, size_scale_now next to the morning size_scale."""
    if sim:
        return None                                     # the simulator has no clock or bars for it
    m = re.search(r"size_scale ([\d.]+)", mline)
    v = re.search(r"SPY vol20 ([\d.]+)%", mline)
    ss = float(m.group(1)) if m else None
    mins = now.hour * 60 + now.minute
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 or next_session((now - timedelta(days=1)).strftime("%Y-%m-%d")) != today:
        return "market now: no session today"
    if mins < 570:
        try:
            q = (gw.quotes(["SPY"], timeout=8.0) or [{}])[0]
            px = q.get("last") or ((q["bid"] + q["ask"]) / 2 if q.get("bid") and q.get("ask") else None)
            if q.get("error") or not px or not q.get("close"):
                raise ValueError(q.get("error") or "no pre-market price")
            # ML3.1 (review fix 3): the line names its time, so a stale move is not read as a current one: the quote's
            # last trade time when the price is that trade, else the time it was asked for
            at = f"quote asked for at {now:%H:%M} ET"
            if q.get("last") and q.get("last_time"):
                try:
                    lt = datetime.fromisoformat(str(q["last_time"]).replace("Z", "+00:00")).astimezone(now.tzinfo)
                    at = f"last trade {lt:%H:%M} ET"
                except (ValueError, TypeError):
                    pass
            mv = (f"SPY {(px / q['close'] - 1) * 100:+.2f}% since the last close (pre-market quote, {at}, delayed about "
                  f"{DELAY_MIN} minutes)")
        except Exception as e:                          # noqa: BLE001
            mv = f"SPY pre-market quote unavailable ({type(e).__name__})"
        return f"market now: {mv}" + (f"; size_scale_now {ss:g} (the morning value until 09:30)" if NOW_SHIPS and ss else "")
    if mins >= 16 * 60:
        return "market now: the session has closed"
    if today in _holidays()[1]:
        return "market now: no intraday line on half-days"
    try:
        days = by_day(gw.bars("SPY", duration="2 D", bar_size="30 mins", what="TRADES", rth=True, timeout=10.0) or [])
        tb = days.get(now.strftime("%Y%m%d"), {})
        prev = [d for d in sorted(days) if d < now.strftime("%Y%m%d") and days[d]]
        if "09:30" not in tb or not prev:
            raise ValueError("no opening print or previous close yet")
        c0 = days[prev[-1]][max(days[prev[-1]])]["close"]
        k = min((mins - 570) // 30, 12)
        gap = math.log(tb["09:30"]["open"] / c0)
        x = so_far(tb, k, gap)
        done = [s for s in SLOTS[:k] if s in tb]
        px = tb[done[-1]]["close"] if done else tb["09:30"]["open"]
    except Exception as e:                              # noqa: BLE001
        return f"market now: SPY's 30-minute bars unavailable ({type(e).__name__}: {str(e)[:60]})"
    # ML3.1 (review fix 3, 15 Nov): the line said "at the 10:30 look", but in 60 lines of 9-13 Nov it matched the close of
    # the last (partial, delayed) bar within a median 0.05 pp, a price 15-30 minutes older. It now names that print: the
    # bar's end, or the look time less the delay if the bar is still open; the 09:30 open when no bar is used yet.
    if done:
        b0 = datetime.strptime(f"{now:%Y-%m-%d} {done[-1]}", "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
        at = max(b0, min(b0 + timedelta(minutes=30), now - timedelta(minutes=DELAY_MIN)))
        when = f"as of about {at:%H:%M} ET (the latest print of the {done[-1]} bar, delayed about {DELAY_MIN} minutes)"
    else:
        when = f"at the 09:30 ET open (delayed about {DELAY_MIN} minutes)"
    line = f"market now: SPY {(px / c0 - 1) * 100:+.2f}% since the last close {when}, range since the close {x * 100:.2f}%"
    g = ((fc3 or {}).get("spy") or {}).get("g") or [None] * 13
    if NOW_SHIPS and ss and v and g[k]:
        now_ss, z = scale_now(ss, float(v.group(1)) / 100 / math.sqrt(252), x, g[k], k)
        line += f"; size_scale_now {now_ss:.2f} (morning {ss:g}): the move so far is {z:.1f}x what the morning forecast implies by now"
    return line


def run(gw, args, sleeve_value=None, look=None):
    """ML2's text (forecast.run, unchanged) with gap1 after absr1 on each row, its legend, the night's events and
    the market-now line. ML2's errors and its "no forecasts" come back as they are. sleeve_value and look go to
    forecast.run as they are (ML3.1, fixes 1 and 4); gap1's legend line follows ML2's: on a look's later calls it is left
    out with ML2's, under ML2's one line naming the first call."""
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
        M, fc3 = load_model(), _fc3(as_of, sim)
        if fc3 is None:
            raise FileNotFoundError("no ML3 forecasts for this close")
    except Exception as e:                              # noqa: BLE001 - ML2's text is still served, gap1 na
        M, fc3 = None, None
        lines.insert(1, f"gap1 unavailable ({type(e).__name__}: {str(e)[:60]})")
    night = (fc3 or {}).get("night") or [None, None]
    # ML3.1: this check came after the row indices below, so its inserted line shifted every row by one and each row's
    # numbers were rewritten under the symbol of the row after it (found 15 Nov in forecast_fixes_test.py, on a stale night)
    if fc3 and not sim and night[0] and night[0] != _now().strftime("%Y-%m-%d"):   # a failed refresh: yesterday's night
        lines.insert(1, f"gap1 unavailable (its forecasts are for the night after the {night[0]} close)")
        M, fc3, night = None, None, [None, None]
    ci = next((i for i, x in enumerate(lines) if x.startswith("sym close range1")), None)
    if ci is None:
        return text
    rows = {}
    for i in range(ci + 1, len(lines)):
        s = max((s for s in syms if lines[i] == s or lines[i].startswith(s + " ")), key=len, default=None)
        if s and not lines[i][len(s) + 1:].startswith("not covered"):
            rows[i] = s
    have = dict((fc3 or {}).get("rows") or {})
    need = [s for s in rows.values() if s not in have]
    if need and fc3 and not sim:
        have.update(_outside_rows(gw, need, as_of, min(ONDEMAND_S, max(0.0, t0 + CALL_S - time.monotonic()))))
    mk = (fc3 or {}).get("market") or {}
    mac = (fc3 or {}).get("macro") or []
    events = []
    for i, s in rows.items():
        tok = "na"
        r = have.get(s)
        X = np.array([[r[0], r[1], mk.get("lmkt5"), mk.get("lv5"), mk.get("long")]], float) if r else None
        if M and X is not None and np.all(np.isfinite(X)):
            e = _earn(lines[i], night)
            q = quantiles(M, X, multiplier(M, e, bool(mac))[None])[0]
            tok = f"{_num(q[1])}({_num(q[0])}-{_num(q[2])})"
            if e:
                events.append(f"{s} earnings ({'Alpha Vantage calendar' if e == 'earn' else 'our estimate'})")
        p = lines[i][len(s) + 1:].split(" ", 3)
        if len(p) >= 3:
            lines[i] = " ".join([s] + p[:3] + [tok] + p[3:])
    lines[ci] = lines[ci].replace("absr1±% ", "absr1±% gap1±%(10-90) ", 1)
    short = any(ln.endswith(forecast.LEGEND_REF) for ln in lines[:ci])     # ML2 printed its one-line reference instead
    add = [] if short else [("gap1 ±(10-90): the expected size of the gap from today's close to the next open, up or down, % (median, "
            "with the 10-90% range of its size; it does not change during the day). It includes the night's events "
            "below. range1 leaves the gap out; absr1 includes it. Names outside the validated universe get no earnings "
            "check, so their gap1 is not raised for an earnings release.")]
    if night[0] and (events or mac):
        add.append(f"gap1 events tonight (after the {night[0]} close, before the {night[1]} open): "
                   + "; ".join(events + ([f"{', '.join(mac)} before the open (every name)"] if mac else [])))
    lines[ci:ci] = add
    mi = next((i for i, x in enumerate(lines) if x.startswith("market:")), None)
    now_line = _market_now(gw, lines[mi], fc3, sim, _now()) if mi is not None else None
    if now_line:
        lines.insert(mi + 1, now_line)
    return "\n".join(lines)
