"""Tool "intraday": models on 30-minute bars for decisions at the sleeves' looks (09:30, 10:00, ..., 15:30 ET).
Pre-registration: runs/ml/research/intraday_prereg.md (5 variants: mim, xs_lin, xs_gbm, xs_hks, range).

    .venv\\Scripts\\python.exe ml_research\\intraday.py            # validate(): walk-forward OOS, writes runs/ml/research/intraday*
    .venv\\Scripts\\python.exe ml_research\\intraday.py --final    # fit the shipped model(s) on every realized target

Steps: day t, slot k = 0..12. Decision price P[t,k] = open of the 09:30 bar (k = 0) or the close of bar k-1 (10:00..15:30).
Features use only bars complete at the decision time. data/bars30 is not dividend-adjusted; it is split-adjusted for
sessions through 16 Oct 2026 and as traded from 17 Oct 2026 (the desk appends each day as traded), so after a split a
name's sessions sit on two scales. The readers (_cached_history, load_panel) bring a name to one share scale with
_one_scale, from the adjusted daily cache (data/bars), only where price and share volume both show a standard split ratio
and the stored overnight step shows it at exactly one session; live, today's opening price counts as a step too.
Live, a name whose cache lacks the latest session SPY's cache has takes the gateway's month of bars when that has it.
The overnight gap is corrected for ex-dividend drops found from the adjusted daily cache.
"""

import json
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml_eval     # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "runs" / "ml" / "research"
BARS30 = ROOT / "data" / "bars30"
IVDIR = ROOT / "data" / "iv"
DAILY = ROOT / "data" / "bars"            # adjusted daily cache (IBKR ADJUSTED_LAST): the current share scale
CACHE = OUT / "intraday_panel.npz"
MODEL = OUT / "intraday_model.pkl"
ET = ZoneInfo("America/New_York")
NS = 13                                   # 30-minute bars per regular session
SLOTS = [f"{9 + (30 + 30 * k) // 60:02d}:{(30 + 30 * k) % 60:02d}" for k in range(NS)]   # 09:30 .. 15:30
TEST_FROM = "20260102"
RETRAIN_DAYS, MIN_DAYS, WIN = 21, 25, 20
COST_BPS = 10.0
MIM_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "SMH"]
TRACK_RECORD = (
    "Intraday range, out of sample 2 Jan - 16 Oct 2026 (199 sessions x 13 looks, 464 of today's S&P 500 stocks, so "
    "survivor-biased; one year of 30-minute bars): next-60-minute range R2 0.58 vs 0.49 for the stock's own 20-session "
    "average at the same time of day; rest-of-day range R2 0.56 vs 0.48. Within a single look the R2 (of log range) is "
    "0.40-0.46 vs 0.22-0.35 for that average, ahead at every look. The 10-90% intervals held 79.8% and 80.1% of outcomes (79-81% at "
    "every look, in both half-years and every volatility quintile), and the mean forecast matched the mean outcome in each "
    "forecast decile. It says how far, not which way: the intraday direction models (momentum, reversal, volume, gap, the "
    "market's first-half-hour move, same-time-of-day periodicity) did not beat 10 bps round-trip costs. The best of them, "
    "same-time-of-day periodicity, had IC t 4.5 but a 1.6 bps gross 30-minute decile spread against 20 bps of costs. "
    "Validated on stocks only. Weak spots: on big-gap opens (gap above about 5%, e.g. earnings) the 10-90% range held only "
    "about 55% at 09:30, with 40-45% of outcomes above the top; about 75% on the market's most volatile days; ETFs and "
    "half-days were not validated.")

SPEC = {
    "name": "intraday_range",
    "description": ("Expected high-low range for the next 60 minutes and for the rest of today, as % of the current price, "
                    "with a 10/50/90% band, from today's 30-minute bars, the stock's usual range at this time of day, "
                    "implied vol and the market's range so far. Use it for stop and target distances during the session "
                    "(09:30-16:00 ET): a stop inside the 10% value of the rest-of-day range is likely to be touched by "
                    "normal noise. It says how far a stock may move, not which way. Track record: " + TRACK_RECORD),
    "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"},
                                                                "description": "Up to 20 symbols."}},
                   "required": ["symbols"]},
}


# ---------------------------------------------------------------- data
def _bars_by_day(rows):
    day = {}
    for b in rows:
        dt, tm = str(b["date"]).split()[:2]
        day.setdefault(dt, {})[tm[:5]] = b
    return day


SPLIT_RATIOS = sorted({r for x in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 100, 3 / 2, 5 / 4, 4 / 3, 5 / 2)
                       for r in (x, 1 / x)})
_SCALE_SEEN = {}                          # last scale detection per symbol, for tests and logs only (no model text)


def _pos(x):
    x = float(x)                          # None or text raises; callers skip that bar
    return x if np.isfinite(x) and x > 0 else None


def _split_test(n, m):
    """(n_snap, None) if price ratio n is a standard split ratio and the share-count ratio m moved with it, else (None, why)."""
    if abs(np.log(n)) <= 0.2:
        return None, None
    r = min(SPLIT_RATIOS, key=lambda x: abs(np.log(n / x)))
    if abs(n / r - 1) > 0.05:
        return None, "unexplained"
    if not abs(np.log(m / r)) < 0.5 * abs(np.log(r)):
        return None, "price changed without the share count"
    return r, None


TODAY = "TODAY"                          # today's opening step in _one_scale; as text it sorts after every YYYYMMDD date


def _given_price(x):
    """x as a float if it is a positive finite number (not text, not a bool), else None."""
    if x is None or isinstance(x, (bool, np.bool_, str, bytes)):
        return None
    try:
        return _pos(x)
    except Exception:
        return None


def _step_at(days, order, cands, r, open_today=None):
    """The one session in cands whose as-traded overnight step matches a split of ratio r, else None.
    step(s) = ln(first open of s / last close of the stored session before s); step(TODAY) = ln(today's open / last
    close of the newest stored session). A match is |step + ln r| < 0.5 |ln r|. No match, two matches, a step that
    cannot be computed (a bad bar), another candidate with a step above 0.2 in size (a real move that may hide the
    split's own step), or a matching step whose nearest standard ratio is not r, all give None: no boundary."""
    hit, big = [], 0
    for s in cands:
        try:
            st = _step(days, order, s, open_today)
            if st is None:
                return None
            if abs(st + np.log(r)) < 0.5 * abs(np.log(r)):
                hit.append((s, st))
            elif abs(st) > 0.2:
                big += 1
        except Exception:
            return None
    if len(hit) != 1 or big:
        return None
    s, st = hit[0]
    return s if min(SPLIT_RATIOS, key=lambda x: abs(st + np.log(x))) == r else None


def _step(days, order, s, open_today=None):
    """ln(first open of s / last close of the stored session before s); for TODAY, today's open over the newest stored
    session's last close. None if a price is missing or not positive."""
    if s != TODAY and order.index(s) == 0:
        return None
    p = order[-1] if s == TODAY else order[order.index(s) - 1]
    o = open_today if s == TODAY else _pos(days[s][min(days[s])]["open"])
    pc = _pos(days[p][max(days[p])]["close"])
    return float(np.log(o / pc)) if o and pc else None


def _one_scale(sym, days, before=None, open_today=None):
    """{date: {HH:MM: bar}} brought to one share scale with the adjusted daily cache (DAILY).
    f = daily close / last stored close and v = daily volume / stored volume, per stored session with a daily bar.
    Pairs: for consecutive such sessions a < c, a step in f is a split of ratio r only if it snaps to SPLIT_RATIOS and v
    moves with it (_split_test), and the stored as-traded overnight step matches r at exactly one stored session s with
    a < s <= c (_step_at); the boundary is placed at s.
    Level (live only): when `before` (the look's day) is given and the daily cache's newest bar is earlier than it, the
    newest session R with f and v is tested against the cache's scale (ratio from 1 / f_R and v_R); the boundary is
    placed at the one matching step among the stored sessions after R and today's open (`open_today`, when it is a
    positive number). Without either there is no level step. In a replay (cache newer than `before`) there is none.
    A boundary at s with ratio r puts every stored session before s on a scale r times coarser: its prices are divided
    by the product of the ratios of the boundaries after it and its volume multiplied by it (a None volume stays None).
    Any rejection (a ratio that is unexplained or not carried by the share count, or one that cannot be placed), or a
    boundary while today's open jumps by more than 0.2 in log with no boundary at TODAY, returns the history as read.
    Returns `days` itself when every factor is 1 or anything fails; never raises and never mutates its input."""
    try:
        doc = json.loads((DAILY / f"{sym.replace(' ', '_')}.json").read_text())
        daily, newest = {}, max(str(b["date"]).replace("-", "")[:8] for b in doc["bars"])
        for b in doc["bars"]:
            try:
                daily[str(b["date"]).replace("-", "")[:8]] = (_pos(b["close"]), _pos(b["volume"]))
            except Exception:
                pass
        order, fv = sorted(days), {}
        for d in order:
            try:
                dc, dv = daily.get(d, (None, None))
                c, v = _pos(days[d][max(days[d])]["close"]), _pos(sum(float(b["volume"]) for b in days[d].values()))
                if dc and dv and c and v:
                    fv[d] = (dc / c, dv / v)
            except Exception:
                pass
        fd = sorted(fv)
        seen, bound = {"boundaries": [], "rejected": [], "level": None}, []     # bound: (session where it changes, r)
        for a, c in zip(fd, fd[1:]):
            n = fv[c][0] / fv[a][0]
            r, why = _split_test(n, fv[a][1] / fv[c][1])
            s = _step_at(days, order, [x for x in order if a < x <= c], r) if r else None
            if s:
                bound.append((s, r))
                seen["boundaries"].append([a, c, r, s])
            elif r or why:
                seen["rejected"].append([a, c, round(n, 4), why or "no as-traded jump"])
        if before is not None and fd and newest < str(before):
            R, op = fd[-1], _given_price(open_today)
            n = 1 / fv[R][0]
            r, why = _split_test(n, fv[R][1])
            cands = [x for x in order if x > R] + ([TODAY] if op else [])
            s = _step_at(days, order, cands, r, op) if r and cands else None
            if s:
                bound.append((s, r))
                seen["level"] = [R, r, s]
            elif r or why:
                seen["rejected"].append([R, "daily", round(n, 4), why or "no as-traded jump"])
        if bound and _given_price(open_today) and all(s != TODAY for s, _ in bound):
            st = _step(days, order, TODAY, _given_price(open_today))
            if st is not None and abs(st) > 0.2:           # today's open jumps and no boundary explains it: a split ex
                seen["rejected"].append([TODAY, "open", round(st, 4), "today's open jumps with no boundary"])
        if seen["rejected"]:                               # any failure: the history as read (rule item 5)
            _SCALE_SEEN[sym] = {**seen, "factors": {}}
            return days
        fac = {}
        for d in order:
            k = 1.0
            for s, r in bound:
                if s > d:
                    k *= r
            fac[d] = k
        seen["factors"] = {d: k for d, k in fac.items() if k != 1}
        _SCALE_SEEN[sym] = seen
        if not seen["factors"]:
            return days
        sc = lambda b, k: {**b, "open": b["open"] / k, "high": b["high"] / k, "low": b["low"] / k, "close": b["close"] / k,
                           "volume": b["volume"] * k if b["volume"] is not None else None}
        return {d: days[d] if fac[d] == 1 else {sl: sc(b, fac[d]) for sl, b in days[d].items()} for d in order}
    except Exception as e:
        _SCALE_SEEN[sym] = {"error": f"{type(e).__name__}: {e}"[:200]}
        return days


def load_panel(refresh=False):
    """Raw 30-minute OHLCV (symbols, days, 13) on SPY's full-day calendar, the dividend jump (symbols, days) and the
    implied vol at the previous close (symbols, days). Cached in runs/ml/research/intraday_panel.npz."""
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import trader
    d = trader.Data()
    syms = list(d.sym)
    spy = _bars_by_day(json.loads((BARS30 / "SPY.json").read_text())["bars"])
    dates = sorted(x for x, v in spy.items() if len(v) == NS and all(s in v for s in SLOTS))
    di = {x: i for i, x in enumerate(dates)}
    S, D = len(syms), len(dates)
    A = {k: np.full((S, D, NS), np.nan) for k in ("o", "h", "l", "c", "v")}
    for j, s in enumerate(syms):
        p = BARS30 / f"{s.replace(' ', '_')}.json"
        if not p.exists():
            continue
        for dt, bb in _one_scale(s, _bars_by_day(json.loads(p.read_text())["bars"])).items():
            i = di.get(dt)
            if i is None or len(bb) != NS:
                continue
            for k, sl in enumerate(SLOTS):
                b = bb.get(sl)
                if b:
                    for key, nm in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
                        A[key][j, i, k] = b[nm]
    # dividend jumps: f = log(adjusted daily close / raw last 30-minute close); an ex-date = a day-over-day jump > 15 bps
    # whose 3-day-median level shift confirms it (the closes of the next 2 days classify a scheduled, pre-announced event)
    f = np.full((S, D), np.nan)
    for i, dt in enumerate(dates):
        t = d.ix.get(dt)
        if t is not None:
            f[:, i] = np.log(d.c[:, t] / A["c"][:, i, NS - 1])
    jump = np.diff(f, axis=1, prepend=np.nan)
    med = lambda a, b: np.stack([np.nanmedian(f[:, max(i + a, 0):max(i + b, 1)], axis=1) for i in range(D)], axis=1)
    level = med(0, 3) - med(-3, 0)        # shift of the factor's 3-day median: an ex-date's step persists, noise doesn't
    div = np.nan_to_num(np.where((jump > 15e-4) & (level > 0.6 * jump), jump, 0.0))
    # implied vol as of the previous close (forward-filled up to 5 sessions)
    iv = np.full((S, D), np.nan)
    for j, s in enumerate(syms):
        p = IVDIR / f"{s.replace(' ', '_')}.json"
        if not p.exists():
            continue
        m = {b["date"][:8]: b["close"] for b in json.loads(p.read_text())["bars"] if b.get("close") and b["close"] > 0}
        allm = sorted(m)
        for i, dt in enumerate(dates):
            # latest IV date strictly before dt, at most ~7 calendar days old
            lo, hi = 0, len(allm)
            while lo < hi:
                mid = (lo + hi) // 2
                if allm[mid] < dt:
                    lo = mid + 1
                else:
                    hi = mid
            if lo and (datetime.strptime(dt, "%Y%m%d") - datetime.strptime(allm[lo - 1], "%Y%m%d")).days <= 7:
                iv[j, i] = m[allm[lo - 1]]
    etf = np.array([s in d.etf for s in syms])
    out = {**A, "div": div, "iv": iv, "symbols": np.array(syms), "dates": np.array(dates), "etf": etf}
    np.savez_compressed(CACHE, **out)
    return out


# ---------------------------------------------------------------- helpers
def past_mean(x, n=WIN, minv=10):
    """Mean over days t-n..t-1 (excluding t) along axis 1 of (S, D, ...) arrays; NaN if fewer than minv valid."""
    ok = ~np.isnan(x)
    z = np.zeros_like(x[:, :1])
    cs = np.concatenate([z, np.cumsum(np.where(ok, x, 0.0), axis=1)], axis=1)
    cn = np.concatenate([z, np.cumsum(ok, axis=1)], axis=1)
    D = x.shape[1]
    idx = np.arange(D)
    lo = np.maximum(idx - n, 0)
    s, c = cs[:, idx] - cs[:, lo], cn[:, idx] - cn[:, lo]
    return np.where(c >= minv, s / np.maximum(c, 1), np.nan)


def prices(p):
    """Decision prices P (S, D, 14): P[...,0] = open of the first bar, P[...,k] = close of bar k-1, P[...,13] = close."""
    return np.concatenate([p["o"][:, :, :1], p["c"]], axis=2)


def gaps(p):
    """Dividend-corrected log gap (S, D): log(open_t / close_{t-1}) - dividend_t."""
    prevc = np.concatenate([np.full((p["c"].shape[0], 1), np.nan), p["c"][:, :-1, NS - 1]], axis=1)
    return np.log(p["o"][:, :, 0] / prevc) - p["div"]


def cummax_prev(x, fn):
    """out[..., k] = fn over bars 0..k-1 (NaN at k = 0)."""
    acc = fn.accumulate(np.where(np.isnan(x), -np.inf if fn is np.maximum else np.inf, x), axis=2)
    out = np.concatenate([np.full(x.shape[:2] + (1,), np.nan), acc[:, :, :-1]], axis=2)
    return np.where(np.isfinite(out), out, np.nan)


def targets(p, stocks):
    P = prices(p)
    k = np.arange(NS)
    y30 = P[:, :, k + 1] / P[:, :, k] - 1
    y60 = P[:, :, np.minimum(k + 2, NS)] / P[:, :, k] - 1
    yrod = P[:, :, NS:NS + 1] / P[:, :, k] - 1
    rel = lambda y: np.where(stocks[:, None, None], y - np.nanmean(np.where(stocks[:, None, None], y, np.nan), axis=0), np.nan)
    return {"r30": rel(y30), "r60": rel(y60), "rrod": rel(yrod)}


def rank3(x, mask):
    """Cross-sectional percentile rank in [-0.5, 0.5] per (day, slot) among rows in mask."""
    S, D, K = x.shape
    flat = np.ascontiguousarray(np.where(mask[:, None, None], x, np.nan)).reshape(S, -1)
    of = np.full((S, D * K), np.nan)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        n = ok.sum()
        if n > 10:
            of[ok, c] = col[ok].argsort().argsort() / (n - 1) - 0.5
    return of.reshape(S, D, K)


def bc(row, shape):
    return np.broadcast_to(row, shape).copy()


# ---------------------------------------------------------------- features
def xs_features(p, stocks, spy):
    """Return-type features (S, D, 13) for variants 2-4."""
    P = prices(p)[:, :, :NS]
    S, D, _ = P.shape
    sh = (S, D, NS)
    with np.errstate(all="ignore"):
        ret_open = np.log(P / P[:, :, :1])
        lastr = np.log(p["c"] / p["o"])
        ret_last = np.concatenate([np.full((S, D, 1), np.nan), lastr[:, :, :-1]], axis=2)
        vm = past_mean(p["v"])
        vrel = np.log(p["v"] / vm)
        vol_tod = np.concatenate([np.full((S, D, 1), np.nan), vrel[:, :, :-1]], axis=2)
        g = gaps(p)
        gap = np.repeat(g[:, :, None], NS, axis=2)
        cc = p["c"][:, :, NS - 1]
        prevc = np.concatenate([np.full((S, 1), np.nan), cc[:, :-1]], axis=1)
        dret = np.log(cc / prevc) - p["div"]                                       # close-to-close, dividend-corrected
        prev_ret = np.repeat(np.concatenate([np.full((S, 1), np.nan), dret[:, :-1]], axis=1)[:, :, None], NS, axis=2)
        r30 = np.log(P[:, :, 1:] / P[:, :, :-1])
        r30 = np.concatenate([r30, np.log(p["c"][:, :, -1:] / P[:, :, -1:])], axis=2)   # slot-k 30-minute log return
        mkt = np.nanmean(np.where(stocks[:, None, None], r30, np.nan), axis=0)
        rel30 = r30 - mkt
        tail = np.cumsum(rel30[:, :, ::-1], axis=2)[:, :, ::-1]                      # sum over slots k..12
        two = rel30 + np.concatenate([rel30[:, :, 1:], np.zeros((S, D, 1))], axis=2)
        hks = {"30": past_mean(rel30), "60": past_mean(two), "rod": past_mean(tail)}
        iv20 = np.repeat(np.sqrt(past_mean(np.nanmean(r30 ** 2, axis=2)[:, :, None]))[:, :, 0][:, :, None], NS, axis=2)
    R = lambda x: rank3(x, stocks)
    F = {"ret_open": R(ret_open), "ret_last": R(ret_last), "vol_tod": R(vol_tod), "gap": R(gap)}
    has = ~np.isnan(F["gap"][:, :, 0])
    for n in ("ret_open", "ret_last", "vol_tod"):
        F[n][:, :, 0] = np.where(has, 0.0, np.nan)      # at the 09:30 look only the gap exists
    spy_open = bc(ret_open[spy], sh)
    F["mom_x_mkt"] = F["ret_open"] * spy_open
    F.update({"spy_open": spy_open, "spy_last": bc(ret_last[spy], sh), "spy_gap": bc(gap[spy], sh),
              "prev_ret": R(prev_ret), "iv20": R(iv20), "slot": bc(np.arange(NS, dtype=float), sh)})
    for h in hks:
        F["hks" + h] = R(hks[h])
    return F


LIN = ["ret_open", "ret_last", "vol_tod", "gap", "mom_x_mkt"]
GBM = LIN + ["slot", "spy_open", "spy_last", "spy_gap", "prev_ret", "iv20"]          # + the horizon's hks


def range_features(p, spy):
    """Range-model inputs (S, D, 13): logs; 0 where a feature does not exist yet (k = 0). Also the naive baselines."""
    P = prices(p)[:, :, :NS]
    S, D, _ = P.shape
    T = targets_range(p)
    with np.errstate(all="ignore"):
        nav = {k: past_mean(v) for k, v in T.items()}
        hi, lo = cummax_prev(p["h"], np.maximum), cummax_prev(p["l"], np.minimum)
        sofar = (hi - lo) / P
        sofar_r = np.log(sofar / past_mean(sofar))
        lb = (p["h"] - p["l"]) / p["c"]
        lb_prev = np.concatenate([np.full((S, D, 1), np.nan), lb[:, :, :-1]], axis=2)
        lb_r = np.log(lb_prev / past_mean(lb_prev))
        cv = np.concatenate([np.full((S, D, 1), np.nan), np.cumsum(p["v"], axis=2)[:, :, :-1]], axis=2)
        vol_r = np.log(cv / past_mean(cv))
        g = np.abs(gaps(p))
        drange = (np.nanmax(p["h"], axis=2) - np.nanmin(p["l"], axis=2)) / p["c"][:, :, NS - 1]
        prevr = np.concatenate([np.full((S, 1), np.nan), drange[:, :-1]], axis=1)
        r5 = past_mean(drange[:, :, None], n=5, minv=3)[:, :, 0]
    rep = lambda a: np.repeat(a[:, :, None], NS, axis=2)
    z = lambda a: np.where(np.isfinite(a), a, 0.0)
    F = {"sofar_r": z(sofar_r), "lb_r": z(lb_r), "vol_r": z(vol_r), "lgap": rep(np.log(g + 1e-4)),
         "liv": rep(np.log(p["iv"])), "lprev": rep(np.log(prevr)), "l5": rep(np.log(r5)),
         "spy_sofar_r": bc(z(sofar_r[spy]), (S, D, NS))}
    return F, nav, T


def targets_range(p):
    P = prices(p)[:, :, :NS]
    H, L = p["h"], p["l"]
    h60 = np.maximum(H, np.concatenate([H[:, :, 1:], H[:, :, -1:]], axis=2))
    l60 = np.minimum(L, np.concatenate([L[:, :, 1:], L[:, :, -1:]], axis=2))
    hrod = np.maximum.accumulate(H[:, :, ::-1], axis=2)[:, :, ::-1]
    lrod = np.minimum.accumulate(L[:, :, ::-1], axis=2)[:, :, ::-1]
    return {"rng60": (h60 - l60) / P, "rngrod": (hrod - lrod) / P}


RNG = ["lnaive", "sofar_r", "lb_r", "vol_r", "lgap", "liv", "lprev", "l5", "spy_sofar_r"]


# ---------------------------------------------------------------- per-slot OLS with residual quantiles
class SlotOLS:
    """Separate OLS per slot (last-but-two column of X is the slot). Stores training residual quantiles per slot."""

    def __init__(self, qs=(0.1, 0.5, 0.9)):
        self.qs = qs

    def fit(self, X, y):
        slot = X[:, -3].astype(int)
        Z = X[:, :-3]
        self.b, self.q, self.m = {}, {}, {}
        for k in range(NS):
            r = (slot == k) & np.isfinite(Z).all(1) & np.isfinite(y)
            A = np.column_stack([np.ones(r.sum()), Z[r]])
            b = np.linalg.lstsq(A, y[r], rcond=None)[0]
            res = y[r] - A @ b
            self.b[k], self.q[k] = b, np.quantile(res, self.qs)
            self.m[k] = float(np.mean(np.exp(res)))          # smearing factor for log targets
        return self

    def predict(self, X):
        slot = X[:, -3].astype(int)
        Z = X[:, :-3]
        out = np.full(len(X), np.nan)
        for k in range(NS):
            r = slot == k
            out[r] = self.b[k][0] + Z[r] @ self.b[k][1:]
        return np.where(np.isfinite(Z).all(1), out, np.nan)


def design(F, names, slot_arr, nsym):
    """features(idx) for ml_eval.walk_forward: rows step-major, symbol-minor; trailing columns slot, step, symbol."""
    S = nsym
    flat = {n: F[n].reshape(S, -1) for n in names}
    sl = slot_arr.reshape(S, -1)

    def features(idx):
        cols = [flat[n][:, idx].T.reshape(-1) for n in names]
        cols += [sl[:, idx].T.reshape(-1), np.repeat(idx, S).astype(float), np.tile(np.arange(S), len(idx)).astype(float)]
        return np.column_stack(cols)
    return features


def run_wf(F, names, y, fitter, predictor, horizon, test_from, nsym):
    """Walk-forward via ml_eval. Also returns the 10-90% interval from each fit's in-sample residual quantiles per slot
    (written by the predictor through the step and symbol columns)."""
    S, D, _ = y.shape
    slot = np.broadcast_to(np.arange(NS, dtype=float), (S, D, NS))
    feats = design(F, names, slot, nsym)
    lo, hi = np.full((S, D * NS), np.nan), np.full((S, D * NS), np.nan)

    def fit(X, yy):
        m = fitter(X, yy)
        r = yy - predictor(m, X)
        sl = X[:, -3].astype(int)
        m._iq = {k: np.nanquantile(r[sl == k], [0.1, 0.9]) for k in range(NS)}
        return m

    def predict(m, X):
        pr = predictor(m, X)
        sl, st, sy = X[:, -3].astype(int), X[:, -2].astype(int), X[:, -1].astype(int)
        for k in range(NS):
            r = sl == k
            lo[sy[r], st[r]] = pr[r] + m._iq[k][0]
            hi[sy[r], st[r]] = pr[r] + m._iq[k][1]
        return pr

    pred = ml_eval.walk_forward(feats, y.reshape(S, -1), steps_from=0, test_from=test_from, horizon=horizon,
                                retrain_every=RETRAIN_DAYS * NS, fit=fit, predict=predict, min_train=MIN_DAYS * NS)
    return pred, lo, hi


# ---------------------------------------------------------------- scoring helpers
def day_clustered(pred, target, steps, labels):
    ics = {}
    for t in steps:
        v = ml_eval.spearman(pred[:, t], target[:, t])
        if np.isfinite(v):
            ics.setdefault(str(labels[t])[:8], []).append(v)
    x = np.array([np.mean(v) for v in ics.values()])
    return {"days": len(x), "ic_day_mean": float(x.mean()), "ic_day_t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))}


def by_slot_ic(pred, target, steps):
    out = {}
    for k in range(NS):
        st = [t for t in steps if t % NS == k]
        v = [ml_eval.spearman(pred[:, t], target[:, t]) for t in st]
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v) > 2:
            out[SLOTS[k]] = {"ic": round(float(v.mean()), 4), "t": round(float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))), 2)}
    return out


def calib_return(pred, target, steps, lo=None, hi=None):
    """Realized vs predicted mean by predicted decile (pooled), and 80% interval coverage when given."""
    p, y = pred[:, steps].ravel(), target[:, steps].ravel()
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    qs = np.quantile(p, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, 9)
    out = {"by_decile_bps": [{"pred": round(float(p[dec == i].mean() * 1e4), 2), "real": round(float(y[dec == i].mean() * 1e4), 2)}
                             for i in range(10)]}
    if lo is not None:
        lo_, hi_ = lo[:, steps].ravel()[ok], hi[:, steps].ravel()[ok]
        out["cover80"] = float(np.mean((y >= lo_) & (y <= hi_)))
    return out


# ---------------------------------------------------------------- variant 1: market intraday momentum
def mim(p, syms, labels_d, t0):
    idx = [list(syms).index(s) for s in MIM_ETFS if s in list(syms)]
    names = [syms[i] for i in idx]
    with np.errstate(all="ignore"):
        g = gaps(p)
        r1 = g + np.log(p["c"][:, :, 0] / p["o"][:, :, 0])                     # previous close -> 10:00
        r12 = np.log(p["c"][:, :, 11] / p["c"][:, :, 10])                      # 15:00 -> 15:30
        y = p["c"][:, :, 12] / p["c"][:, :, 11] - 1                            # 15:30 -> 16:00
    X1, X12, Y = r1[idx], r12[idx], y[idx]
    n, D = Y.shape
    models = []

    def features(ix):
        return np.column_stack([X1[:, ix].T.reshape(-1), X12[:, ix].T.reshape(-1)])

    def fit(X, yy):
        A = np.column_stack([np.ones(len(X)), X])
        ok = np.isfinite(A).all(1) & np.isfinite(yy)
        b = np.linalg.lstsq(A[ok], yy[ok], rcond=None)[0]
        res = yy[ok] - A[ok] @ b
        m = {"b": b, "res": res}
        models.append(m)
        return m

    def predict(m, X):
        return m["b"][0] + X @ m["b"][1:]

    pred = ml_eval.walk_forward(features, Y, 0, t0, 1, RETRAIN_DAYS, fit, predict, min_train=MIN_DAYS)
    # intervals and P(up) from the block's training residuals (walk_forward fits blocks in order)
    lo, hi, pup = (np.full_like(pred, np.nan) for _ in range(3))
    blocks = [T for T in range(t0, D, RETRAIN_DAYS) if len(range(0, T - 1)) >= MIN_DAYS]
    for m, T in zip(models, blocks):
        te = np.arange(T, min(T + RETRAIN_DAYS, D))
        q10, q90 = np.quantile(m["res"], [0.1, 0.9])
        lo[:, te], hi[:, te] = pred[:, te] + q10, pred[:, te] + q90
        pup[:, te] = np.mean(m["res"][None, None, :] > -pred[:, te][:, :, None], axis=2)
    te = np.arange(t0, D)
    ok = np.isfinite(pred[:, te]) & np.isfinite(Y[:, te])
    yy, pp = Y[:, te][ok], pred[:, te][ok]
    r2 = float(1 - np.sum((yy - pp) ** 2) / np.sum(yy ** 2))                    # vs a zero forecast
    cost = COST_BPS / 1e4
    strat = np.where(np.isfinite(pred), np.sign(pred) * Y - cost, np.nan)[:, te]
    sign_rule = np.where(np.isfinite(X1), np.sign(X1) * Y - cost, np.nan)[:, te]
    gross = np.where(np.isfinite(pred), np.sign(pred) * Y, np.nan)[:, te]

    def daily(x):
        v = np.nanmean(x, axis=0)
        v = v[np.isfinite(v)]
        return {"days": len(v), "mean_bps": float(v.mean() * 1e4), "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))}
    halves = {}
    for i, t in enumerate(te):
        lab = str(labels_d[t])
        halves.setdefault(lab[:4] + ("H1" if lab[4:6] <= "06" else "H2"), []).append(np.nanmean(strat[:, i]))
    halves = {k: float(np.nanmean(v) * 1e4) for k, v in halves.items()}
    per_etf = {names[i]: {"net_bps": float(np.nanmean(strat[i]) * 1e4), "r2": None} for i in range(n)}
    for i in range(n):
        o = np.isfinite(pred[i, te]) & np.isfinite(Y[i, te])
        per_etf[names[i]]["r2"] = float(1 - np.sum((Y[i, te][o] - pred[i, te][o]) ** 2) / np.sum(Y[i, te][o] ** 2))
    # calibration: 80% interval coverage, P(up) reliability and Brier vs the training base rate
    up = (Y[:, te] > 0).astype(float)
    pu = pup[:, te]
    o = np.isfinite(pu) & np.isfinite(Y[:, te])
    brier = float(np.mean((pu[o] - up[o]) ** 2))
    base = float(np.mean(up[o]))
    rel = []
    for a, b in ((0, .45), (.45, .5), (.5, .55), (.55, 1.01)):
        r = o & (pu >= a) & (pu < b)
        if r.sum():
            rel.append({"p_bin": [a, min(b, 1.0)], "n": int(r.sum()), "p_mean": float(pu[r].mean()), "up_rate": float(up[r].mean())})
    cover = float(np.mean((Y[:, te][o] >= lo[:, te][o]) & (Y[:, te][o] <= hi[:, te][o])))
    # coefficient path (last model) for the report
    b = models[-1]["b"] if models else [np.nan] * 3
    full = np.isfinite(X1) & np.isfinite(Y)
    res = {"etfs": names, "test_days": int(len(te)), "r2_vs_zero": r2, "timing_net": daily(strat), "timing_gross": daily(gross),
           "net_by_half_bps": halves, "sign_rule_net": daily(sign_rule), "per_etf": per_etf, "cover80": cover,
           "brier": brier, "brier_climatology": float(base * (1 - base)), "reliability": rel,
           "last_coef": {"intercept_bps": float(b[0] * 1e4), "b_first_halfhour": float(b[1]), "b_1500_1530": float(b[2])},
           "full_sample_corr_r1_y": float(np.corrcoef(X1[full], Y[full])[0, 1])}
    res["pass"] = bool(r2 > 0 and res["timing_net"]["mean_bps"] > 0 and res["timing_net"]["t"] >= 2 and all(v > 0 for v in halves.values()))
    return res


def time_of_day(p, syms, stocks):
    spy = list(syms).index("SPY")
    P = prices(p)
    with np.errstate(all="ignore"):
        r = np.log(P[:, :, 1:] / P[:, :, :-1])
        vs = p["v"] / np.nansum(p["v"], axis=2, keepdims=True)
    out = {}
    for k in range(NS):
        st = r[stocks, :, k]
        out[SLOTS[k] + "-" + (SLOTS[k + 1] if k + 1 < NS else "16:00")] = {
            "spy_mean_bps": round(float(np.nanmean(r[spy, :, k]) * 1e4), 2),
            "spy_t": round(float(np.nanmean(r[spy, :, k]) / np.nanstd(r[spy, :, k], ddof=1) * np.sqrt(np.sum(np.isfinite(r[spy, :, k])))), 2),
            "spy_abs_bps": round(float(np.nanmean(np.abs(r[spy, :, k])) * 1e4), 1),
            "stock_abs_bps": round(float(np.nanmean(np.abs(st)) * 1e4), 1),
            "stock_volume_share_pct": round(float(np.nanmean(vs[stocks, :, k]) * 100), 1)}
    return out


def range_eval(p, syms, dates, stocks, spy, t0d, log=print):
    """Variant 5: walk-forward per-slot OLS of the log range; R^2 vs the naive, interval calibration."""
    S, D = len(syms), len(dates)
    t0 = t0d * NS
    test = list(range(t0, D * NS))
    oos = {}
    RF, NAV, RT = range_features(p, spy)
    rres = {}
    for tk in ("rng60", "rngrod"):
        F5 = dict(RF)
        F5["lnaive"] = np.log(NAV[tk])
        y = np.log(np.where(stocks[:, None, None] & (RT[tk] > 0), RT[tk], np.nan))
        y = np.where(np.isfinite(F5["lnaive"]), y, np.nan)
        models = []

        def rfit(X, yy):
            m = SlotOLS().fit(X, yy)
            models.append(m)
            return m
        h = NS
        lpred, _, _ = run_wf(F5, RNG, y, rfit, lambda m, X: m.predict(X), h, t0, S)
        lp3 = lpred.reshape(S, D, NS)
        lo3, hi3, lv3 = (np.full((S, D, NS), np.nan) for _ in range(3))
        blocks = [T for T in range(t0, D * NS, RETRAIN_DAYS * NS) if T - h >= MIN_DAYS * NS]
        for m, T in zip(models, blocks):
            te = np.arange(T, min(T + RETRAIN_DAYS * NS, D * NS))
            for k in range(NS):
                st = te[te % NS == k]
                dd = st // NS
                lo3[:, dd, k] = lp3[:, dd, k] + m.q[k][0]
                hi3[:, dd, k] = lp3[:, dd, k] + m.q[k][2]
                lv3[:, dd, k] = np.exp(lp3[:, dd, k]) * m.m[k]
        Yl = y.reshape(S, -1)
        nl = F5["lnaive"].reshape(S, -1)
        lvl = RT[tk].reshape(S, -1)
        lvl = np.where(np.isfinite(Yl), lvl, np.nan)
        r_log = ml_eval.r2_oos(lpred, Yl, nl, test)
        lnaive2 = past_mean(np.where(np.isfinite(y), y, np.nan)).reshape(S, -1)       # harder: 20-day mean of the log
        r_log2 = ml_eval.r2_oos(lpred, Yl, lnaive2, test)
        r_lvl = ml_eval.r2_oos(lv3.reshape(S, -1), lvl, NAV[tk].reshape(S, -1), test)
        # calibration
        ok = np.isfinite(lp3) & np.isfinite(y)
        ok[:, :t0d] = False
        inside = (y >= lo3) & (y <= hi3)
        below = y < lo3
        cov = {"overall": float(inside[ok].mean()), "below_q10": float(below[ok].mean()), "above_q90": float((y > hi3)[ok].mean())}
        cov["by_half"] = {}
        hv = np.array([dates[i][:4] + ("H1" if dates[i][4:6] <= "06" else "H2") for i in range(D)])
        for hk in sorted(set(hv[t0d:])):
            m_ = ok & (hv == hk)[None, :, None]
            cov["by_half"][hk] = float(inside[m_].mean())
        cov["by_slot"] = {SLOTS[k]: round(float(inside[:, :, k][ok[:, :, k]].mean()), 3) for k in range(NS)}
        q = np.nanquantile(np.where(ok, NAV[tk], np.nan), [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(q, NAV[tk])
        cov["by_naive_quintile"] = [round(float(inside[ok & (qi == i)].mean()), 3) for i in range(5)]
        # R2 per half and per slot (log) for model and naive
        r_half = {hk: ml_eval.r2_oos(lpred, Yl, nl, [t for t in test if hv[t // NS] == hk]) for hk in sorted(set(hv[t0d:]))}
        r_slot = {SLOTS[k]: ml_eval.r2_oos(lpred, Yl, nl, [t for t in test if t % NS == k]) for k in range(NS)}
        med_ratio = float(np.nanmedian((lvl / (lv3.reshape(S, -1)))[:, t0:]))
        fl, rl = lv3.reshape(S, -1)[:, test].ravel(), lvl[:, test].ravel()
        o_ = np.isfinite(fl) & np.isfinite(rl)
        fl, rl = fl[o_], rl[o_]
        qd = np.quantile(fl, np.linspace(0, 1, 11))
        dd_ = np.clip(np.searchsorted(qd, fl, side="right") - 1, 0, 9)
        reliab = [{"forecast_pct": round(float(fl[dd_ == i].mean() * 100), 3), "realized_pct": round(float(rl[dd_ == i].mean() * 100), 3)}
                  for i in range(10)]
        pss = bool(r_log["pass"] and r_lvl["pass"] and 0.75 <= cov["overall"] <= 0.85
                   and all(0.75 <= v <= 0.85 for v in cov["by_half"].values()))
        rres[tk] = {"r2_log": r_log, "r2_log_vs_mean_log_naive": r_log2, "r2_level": r_lvl, "coverage80": cov, "r2_log_by_half": r_half, "r2_log_by_slot": r_slot,
                    "median_realized_over_forecast": med_ratio, "mean_by_forecast_decile": reliab, "pass": pss,
                    "last_model": {SLOTS[k]: {"coef": dict(zip(["const"] + RNG, np.round(models[-1].b[k], 3).tolist())),
                                              "q10_q50_q90": np.round(models[-1].q[k], 3).tolist()} for k in range(NS)}}
        oos[f"range_{tk}_log"], oos[f"range_{tk}_lo"], oos[f"range_{tk}_hi"] = lpred, lo3.reshape(S, -1), hi3.reshape(S, -1)
        log("range", tk, json.dumps({"log": r_log, "lvl": r_lvl, "cov": cov["overall"], "half": cov["by_half"], "pass": pss}))
    return rres, oos


# ---------------------------------------------------------------- validate
def validate():
    import lightgbm as lgb
    p = load_panel()
    syms, dates, etf = [str(s) for s in p["symbols"]], [str(x) for x in p["dates"]], p["etf"].astype(bool)
    S, D = len(syms), len(dates)
    stocks = ~etf
    spy = syms.index("SPY")
    t0d = next(i for i, x in enumerate(dates) if x >= TEST_FROM)
    t0 = t0d * NS
    labels = np.repeat(np.array(dates), NS)
    test = list(range(t0, D * NS))
    log = lambda *a: print(*a, flush=True)
    res = {"test_window": [dates[t0d], dates[-1]], "test_days": D - t0d, "train_from": dates[0], "symbols": int(stocks.sum()),
           "cost_bps_round_trip": COST_BPS, "variants": {}}
    log("days", D, "test from", dates[t0d], "stocks", int(stocks.sum()))

    # 1. mim + time of day
    res["variants"]["mim"] = mim(p, syms, dates, t0d)
    res["time_of_day"] = time_of_day(p, syms, stocks)
    log("mim", json.dumps({k: res["variants"]["mim"][k] for k in ("r2_vs_zero", "timing_net", "net_by_half_bps", "pass")}))

    # 2-4. cross-sectional
    Y = targets(p, stocks)
    F = xs_features(p, stocks, spy)
    H = {"30": ("r30", 1), "60": ("r60", 2), "rod": ("rrod", NS)}
    oos = {}

    def score_all(name, hz, pred, lo=None, hi=None):
        tkey, h = H[hz]
        tgt = Y[tkey].reshape(S, -1)
        if hz != "rod":
            sc = ml_eval.score(pred, tgt, test, h, labels, cost_bps=COST_BPS)
            sc.update(day_clustered(pred, tgt, test, labels))
            sc["by_slot"] = by_slot_ic(pred, tgt, test)
            sc["pass"] = bool(sc["pass"] and sc["ic_day_t"] >= 2)
        else:
            sc = {"by_look": {}}
            P3, T3 = pred.reshape(S, D, NS), Y[tkey]
            for k in range(NS):
                s_ = ml_eval.score(P3[:, :, k], T3[:, :, k], list(range(t0d, D)), 1, dates, cost_bps=COST_BPS)
                s_["pass"] = bool(s_["pass"] and s_["ic_t"] >= 3.0)
                sc["by_look"][SLOTS[k]] = {x: s_[x] for x in ("ic", "ic_t", "spread_gross_pct", "spread_net_pct",
                                                               "top_decile_net_pct", "ic_by_half", "pass")}
            sc.update(day_clustered(pred, tgt, test, labels))
            sc["pass"] = any(v["pass"] for v in sc["by_look"].values())
            sc["spread_gross_pct"] = float(np.mean([v["spread_gross_pct"] for v in sc["by_look"].values()]))
        g = sc.get("spread_gross_pct", np.nan)
        sc["cost_share_of_gross"] = float(2 * COST_BPS / 100 / g) if g and g > 0 else None
        sc["calibration"] = calib_return(pred, tgt, test, lo, hi)
        log(name, hz, json.dumps({k: sc.get(k) for k in ("ic", "ic_t", "ic_day_t", "spread_gross_pct", "spread_net_pct", "pass")}))
        return sc

    def ols_fit(X, y):
        m = SlotOLS()
        lo, hi = np.nanpercentile(y, [1, 99])
        return m.fit(X, np.clip(y, lo, hi))

    part = OUT / "intraday_partial.npz"
    cache = dict(np.load(part)) if part.exists() else {}

    def cached(key, fn):
        if key not in cache:
            cache[key + "|p"], cache[key + "|lo"], cache[key + "|hi"] = fn()
            cache[key] = np.zeros(1)
            np.savez_compressed(part, **cache)
        return cache[key + "|p"], cache[key + "|lo"], cache[key + "|hi"]

    for hz, (tkey, h) in H.items():
        pred, lo, hi = cached(f"xs_lin_{hz}", lambda: run_wf(F, LIN, Y[tkey], ols_fit, lambda m, X: m.predict(X), h, t0, S))
        oos[f"xs_lin_{hz}"] = pred
        res["variants"].setdefault("xs_lin", {})[hz] = score_all("xs_lin", hz, pred, lo, hi)

    def gbm_fit(X, y):
        rng = np.random.default_rng(0)
        if len(y) > 600_000:
            ii = rng.choice(len(y), 600_000, replace=False)
            X, y = X[ii], y[ii]
        lo, hi = np.nanpercentile(y, [1, 99])
        ds = lgb.Dataset(X[:, :-2], np.clip(y, lo, hi))
        return lgb.train({"objective": "l2", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 1000,
                          "feature_fraction": 0.8, "seed": 0, "verbose": -1, "num_threads": 8}, ds, 200)

    for hz, (tkey, h) in H.items():
        names = GBM + ["hks" + hz]
        names = [n for n in names if n != "slot"] + ["slot"]          # slot stays a feature (third-last column is slot)
        pred, lo, hi = cached(f"xs_gbm_{hz}", lambda: run_wf(F, names[:-1], Y[tkey], gbm_fit, lambda m, X: m.predict(X[:, :-2]), h, t0, S))
        oos[f"xs_gbm_{hz}"] = pred
        res["variants"].setdefault("xs_gbm", {})[hz] = score_all("xs_gbm", hz, pred, lo, hi)

    for hz in H:
        pred = F["hks" + hz].reshape(S, -1).copy()
        pred[:, :t0] = np.nan
        oos[f"xs_hks_{hz}"] = pred
        res["variants"].setdefault("xs_hks", {})[hz] = score_all("xs_hks", hz, pred)

    for v in ("xs_lin", "xs_gbm", "xs_hks"):
        res["variants"][v]["pass"] = any(res["variants"][v][hz]["pass"] for hz in H)
    del F

    # 5. range
    rres, o5 = range_eval(p, syms, dates, stocks, spy, t0d, log)
    oos.update(o5)
    res["variants"]["range"] = {**rres, "pass": all(v["pass"] for v in rres.values())}
    np.savez_compressed(OUT / "intraday_oos.npz", symbols=np.array(syms), dates=np.array(dates), slots=np.array(SLOTS),
                        **{k: v.astype(np.float32) for k, v in oos.items()})
    (OUT / "intraday.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


# ---------------------------------------------------------------- final fit (shipped parts)
def fit_final():
    """Fit the shipped range models on every realized target (all days) and save runs/ml/research/intraday_model.pkl."""
    p = load_panel()
    syms, dates, etf = [str(x) for x in p["symbols"]], [str(x) for x in p["dates"]], p["etf"].astype(bool)
    S, D = len(syms), len(dates)
    spy = syms.index("SPY")
    RF, NAV, RT = range_features(p, spy)
    models = {}
    for tk in ("rng60", "rngrod"):
        F5 = dict(RF)
        F5["lnaive"] = np.log(NAV[tk])
        y = np.log(np.where(~etf[:, None, None] & (RT[tk] > 0), RT[tk], np.nan))
        y = np.where(np.isfinite(F5["lnaive"]), y, np.nan)
        slot = np.broadcast_to(np.arange(NS, dtype=float), (S, D, NS))
        X = design(F5, RNG, slot, S)(np.arange(D * NS))
        yy = y.reshape(S, -1).T.reshape(-1)
        ok = np.isfinite(yy)
        models[tk] = SlotOLS().fit(X[ok], yy[ok])
    out = {"range": {tk: {"b": m.b, "q": m.q, "m": m.m} for tk, m in models.items()}, "fit_through": dates[-1], "features": RNG}
    MODEL.write_bytes(pickle.dumps(out))          # plain dicts and arrays, so the pickle doesn't depend on __main__
    return out


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        raw = pickle.loads(MODEL.read_bytes())
        for tk, d in raw["range"].items():
            m = SlotOLS()
            m.b, m.q, m.m = d["b"], d["q"], d["m"]
            raw["range"][tk] = m
        _MODEL = raw
    return _MODEL


# ---------------------------------------------------------------- live
def _parse(rows, day=None):
    """gw.bars / cache rows -> {date: {HH:MM: bar}}."""
    out = {}
    for b in rows:
        parts = str(b["date"]).split()
        if len(parts) < 2:
            continue
        dt, tm = parts[0], parts[1][:5]
        if day is None or dt == day:
            out.setdefault(dt, {})[tm] = b
    return out


def _cached_history(sym, before, ndays=25, open_today=None):
    """The last `ndays` full sessions (13 bars) before `before` from data/bars30, brought to one share scale by
    _one_scale with `before` and today's opening price (the level step; see _one_scale). {} if the name has no file."""
    p = BARS30 / f"{sym.replace(' ', '_')}.json"          # I2: BRK B -> BRK_B
    if not p.exists():
        return {}
    days = _parse(json.loads(p.read_text())["bars"])
    full = sorted(d for d, v in days.items() if d < before and len(v) == NS)
    return _one_scale(sym, {d: days[d] for d in full[-ndays:]}, before=before, open_today=open_today)


GW_FILL = True                            # the missing-session fallback in _history; False = the cache, as before it existed
GW_FILL_TIMEOUT_S = 10.0                  # longest wait for its bars request (the gateway's default is 60 s). Trader Training,
#                                           21 Nov: 3 asks x 10 s keeps one call under ML4's 45 s outside-name budget
GW_FILL_MAX = 3                           # most fallback asks in one live() call; names past it keep the cache at that call.
#                                           Names never asked today take the asks first, retries what is left (live())
GW_FILL_RETRY_MIN = 30                    # a request that raised is not repeated for this many minutes of look time
_GW_HISTORY = {}                          # (sym, today) -> a gateway history that filled a missing session; asked once a day
_GW_FAILED = {}                           # (sym, today) -> look time of a request that raised (gw.bars itself), or None for
#                                           an answer that arrived but could not be read or used (not asked again that day)


def _may_ask(key, t):
    """True if the fallback may ask the gateway for key = (sym, today) at look time t: never asked, or its last ask
    raised at least GW_FILL_RETRY_MIN minutes of look time ago (or at a later look time: a replay going back)."""
    if key not in _GW_FAILED:
        return True
    f = _GW_FAILED[key]
    return f is not None and not timedelta(0) <= t - f < timedelta(minutes=GW_FILL_RETRY_MIN)


def _history(gw, sym, today, now, ref=None, open_today=None, asks=None, defer=None):
    """Last ~25 full sessions of 30-minute bars before today and their source. The cache (_cached_history) if it is
    recent, else the gateway ('1 M'). The missing-session fallback (GW_FILL; off = the cache, as before it existed):
    ref = the newest full session before today in SPY's cache. When the name's recent cache ends before ref (the desk's
    record missed a session), the gateway's month is used instead if it has at least 10 sessions, a newer one than the
    cache, and sits on the cache's scale (_same_scale); otherwise, or if the gateway fails, the cache.
    Its cost is bounded: the request waits at most GW_FILL_TIMEOUT_S; `asks` (a one-item list live() passes, starting
    at GW_FILL_MAX) counts down the asks of one live() call, and a name past it keeps the cache at that call; a month
    that was used is remembered per (sym, today), so it is asked for once a day; an answer that arrived but could not
    be read or used (rows that are not bar dicts with a date and numeric, not infinite, open, high, low, close and
    volume, or anything in reading them that raises; a session counts only with its 13 bars at the 13 SLOTS) is not
    asked for again that day; only a request that raised (gw.bars itself) is
    repeated, after GW_FILL_RETRY_MIN minutes of look time.
    `defer` (a list live() passes): a name whose earlier request raised (a retry) does not ask here; it is appended to
    defer and keeps the cache, so live() can give the asks to names never asked today first and retries what is left.
    The memo is check-then-act, not locked: one ask a look holds because the desk serialises research calls, and each
    process keeps its own memo. The gateway's bars are IBKR-adjusted, so they are not passed through _one_scale."""
    h = _cached_history(sym, today, open_today=open_today)
    src = "cache"
    if not h or (now.replace(tzinfo=None) - datetime.strptime(max(h), "%Y%m%d")).days > 10:
        rows = gw.bars(sym, duration="1 M", bar_size="30 mins", what="TRADES", rth=True)
        days = _parse(rows)
        full = sorted(d for d, v in days.items() if d < today and len(v) == NS)
        h, src = {d: days[d] for d in full[-25:]}, "gateway"
    elif GW_FILL and ref is not None and sym != "SPY" and max(h) < ref:
        key, t = (sym, today), now.replace(tzinfo=None)
        if key not in _GW_HISTORY and _may_ask(key, t) and (asks is None or asks[0] > 0):
            if defer is not None and key in _GW_FAILED:
                defer.append(sym)                   # a retry: asked after the names never asked today, if asks remain
                return h, src
            if asks is not None:
                asks[0] -= 1
            try:
                rows = gw.bars(sym, duration="1 M", bar_size="30 mins", what="TRADES", rth=True, timeout=GW_FILL_TIMEOUT_S)
            except Exception:
                _GW_FAILED[key] = t                 # the request raised: asked again after GW_FILL_RETRY_MIN
            else:
                try:
                    days = _parse(rows or [])
                    full = sorted(d for d, v in days.items() if d < today and len(v) == NS and set(v) == set(SLOTS))
                    g = {d: days[d] for d in full[-25:]}
                    for bb in g.values():           # every bar must carry the numbers live() puts in its panel
                        for b in bb.values():
                            for nm in ("open", "high", "low", "close", "volume"):
                                if np.isinf(float(b[nm])):
                                    raise ValueError("an infinite value in the gateway month")
                    ok = len(g) >= 10 and max(g) > max(h) and _same_scale(g, h, open_today)
                except Exception:
                    ok = False
                if ok:
                    _GW_HISTORY[key] = g
                else:
                    _GW_FAILED[key] = None          # answered, not readable or not usable: not asked again today
        if key in _GW_HISTORY:
            h, src = _GW_HISTORY[key], "gateway"
    return h, src


def _same_scale(g, h, open_today=None):
    """True if the gateway's month g sits on the cache's (one-scaled) scale h and on today's: the median of g's last
    close over h's on the sessions both hold is within 2%, and today's open (when given) is not a split-sized jump
    (above 0.4 in log and within 0.05 of ln r for a standard ratio r) from g's newest close."""
    common = [d for d in g if d in h]
    q = [_pos(g[d][max(g[d])]["close"]) / _pos(h[d][max(h[d])]["close"]) for d in common]
    if not q or not 0.98 <= float(np.median(q)) <= 1.02:
        return False
    op = _given_price(open_today)
    if op:
        x = abs(np.log(op / _pos(g[max(g)][max(g[max(g)])]["close"])))
        if x > 0.4 and any(abs(x - abs(np.log(r))) < 0.05 for r in SPLIT_RATIOS):
            return False
    return True


def _iv_prev(sym, today, gw):
    p = IVDIR / f"{sym.replace(' ', '_')}.json"
    if p.exists():
        m = [(b["date"][:8], b["close"]) for b in json.loads(p.read_text())["bars"] if b.get("close") and b["close"] > 0]
        m = [x for x in m if x[0] < today]
        if m and (datetime.strptime(today, "%Y%m%d") - datetime.strptime(m[-1][0], "%Y%m%d")).days <= 7:
            return m[-1][1], "iv cache"
    try:
        st = gw.stats([sym])[0]
        for key, lab in (("implied_vol", "gateway implied vol"), ("hist_vol", "historical vol (no implied vol; proxy)")):
            v = st.get(key)                                       # I3: only a finite, positive fraction counts
            if isinstance(v, (int, float)) and np.isfinite(v) and 0.02 <= v <= 5:
                return float(v), lab
    except Exception:
        pass
    return None, "none"


def _panel_from(hist, todays):
    """A small panel {o,h,l,c,v: (n, days, 13), div, iv} from {sym: history days} and {sym: today's complete bars}."""
    syms = list(hist)
    days = sorted(set().union(*[set(h) for h in hist.values()]))
    D = len(days) + 1
    di = {d: i for i, d in enumerate(days)}
    A = {k: np.full((len(syms), D, NS), np.nan) for k in ("o", "h", "l", "c", "v")}
    for j, s in enumerate(syms):
        for d, bb in list(hist[s].items()) + [("__today__", todays[s])]:
            i = di.get(d, D - 1)
            for k, sl in enumerate(SLOTS):
                b = bb.get(sl)
                if b:
                    for key, nm in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
                        A[key][j, i, k] = b[nm]
    return {**A, "div": np.zeros((len(syms), D)), "iv": np.full((len(syms), D), np.nan)}, syms


def _now_et(now):
    if now is None:
        return datetime.now(ET)
    return now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)


def live(gw, symbols=None, now=None, **kw):
    """Expected high-low range over the next 60 minutes and over the rest of today, % of the current price, with 10-90%
    intervals, at the latest look (09:30 .. 15:30 ET). Uses gw.bars(symbol, '1 D', '30 mins') for today's bars (only
    completed bars, plus the opening print at 09:30), the 30-minute cache (or gw.bars '1 M' when a symbol is not cached,
    the cache is old, or it lacks the latest session SPY's cache has) for the 20-session same-time averages, and data/iv
    (or gw.stats) for implied vol.
    Returns {"as_of", "look", "rows": {symbol: {...}}} or {"error": ...}. No return or direction forecast is served."""
    now = _now_et(now)
    today = now.strftime("%Y%m%d")
    mins = now.hour * 60 + now.minute
    if now.weekday() >= 5 or mins < 9 * 60 + 30 or mins >= 16 * 60:
        return {"error": "intraday range forecasts are available from 09:30 to 16:00 ET on trading days"}
    k = min((mins - (9 * 60 + 30)) // 30, NS - 1)                 # bars completed at this look = look slot k
    model = _load_model()["range"]
    if symbols is None:
        import trader
        etf = trader.etfs()
        symbols = [s for s in trader.universe() if s not in etf][:20]
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip()))[:20]
    need = list(dict.fromkeys(symbols + ["SPY"]))
    try:                                                          # the latest completed session, from SPY's cache
        ref = max(_cached_history("SPY", today), default=None)
    except Exception:
        ref = None
    asks = [GW_FILL_MAX]                                          # the missing-session fallback's asks left in this call
    defer, later = [], {}                                         # its retries, asked after the names never asked today
    hist, todays, info = {}, {}, {}
    for s in need:
        try:
            rows = gw.bars(s, duration="1 D", bar_size="30 mins", what="TRADES", rth=True)
            tb = _parse(rows, today).get(today, {})
            if "09:30" not in tb:
                info[s] = {"error": "no opening print yet for today"}
                continue
            done = {sl: b for sl, b in tb.items() if sl in SLOTS[:k]}            # bars that ended by this look
            if k == 0:
                done = {"09:30": {"open": tb["09:30"]["open"], "high": np.nan, "low": np.nan, "close": np.nan, "volume": np.nan}}
            nd = len(defer)
            h, src = _history(gw, s, today, now, ref=ref, open_today=tb["09:30"].get("open"), asks=asks, defer=defer)
            if len(defer) > nd:
                later[s] = (tb["09:30"].get("open"), done)
            if len(h) < 10:
                info[s] = {"error": f"only {len(h)} past sessions of 30-minute bars; need 10"}
                continue
            hist[s], todays[s], info[s] = h, done, {"history": src, "sessions": len(h)}
        except Exception as e:
            info[s] = {"error": f"no 30-minute bars: {e}"}
    for s in defer:                                               # retries, in the call's order, while asks remain
        if asks[0] <= 0:
            break
        try:
            h, src = _history(gw, s, today, now, ref=ref, open_today=later[s][0], asks=asks)
        except Exception:
            continue                                              # keep what the first pass gave this name
        if src == "gateway":
            hist[s], todays[s], info[s] = h, later[s][1], {"history": src, "sessions": len(h)}
    if "SPY" not in hist:
        return {"error": "SPY's 30-minute bars are unavailable", "detail": info.get("SPY")}
    out = {}
    for s in symbols:
        if "error" in info.get(s, {}):
            out[s] = info[s]
            continue
        pair = {s: hist[s], "SPY": hist["SPY"]} if s != "SPY" else {"SPY": hist["SPY"]}
        p, ps = _panel_from(pair, {x: todays[x] for x in pair})
        hint = (kw.get("iv_hint") or {}).get(s)     # forecast's look-time rows for names outside the cache: their close IV
        iv, ivsrc = (hint, "forecast row (IV at the last close)") if hint else _iv_prev(s, today, gw)
        if iv is None:
            out[s] = {"error": "no implied or historical vol"}
            continue
        p["iv"][0, -1] = iv
        RF, NAV, _ = range_features(p, ps.index("SPY"))
        P = prices(p)[0, -1, k]
        row = {"price": round(float(P), 2), "history": info[s]["history"], "iv_source": ivsrc}
        for tk, lab in (("rng60", "next60"), ("rngrod", "rest_of_day")):
            x = [np.log(NAV[tk][0, -1, k])] + [RF[n][0, -1, k] for n in RNG[1:]]
            m = model[tk]
            lp = float(m.predict(np.array([x + [k, 0, 0]], float))[0])
            if not np.isfinite(lp):
                row[lab + "_range_pct"] = None
                continue
            row[lab + "_range_pct"] = round(100 * float(np.exp(lp)) * m.m[k], 2)
            row[lab + "_range_pct_10_50_90"] = [round(100 * float(np.exp(lp + v)), 2) for v in m.q[k]]
            row[lab + "_naive_pct"] = round(100 * float(NAV[tk][0, -1, k]), 2)
        if s in ETFS_NOTE:
            row["note"] = "ETF: the model was validated on stocks only"
        out[s] = row
    return {"as_of": now.strftime("%Y-%m-%d %H:%M ET"), "look": SLOTS[k],
            "horizons": "next60 = the next two 30-minute bars (at 15:30, just the last one); rest_of_day = through the 16:00 close",
            "notes": "ranges are high minus low as % of price; the live gap is not dividend-corrected; half-days are not modelled",
            "track_record": TRACK_RECORD, "rows": out}


ETFS_NOTE = set(MIM_ETFS)


def sim_provider(data):
    """None: the tool needs today's 30-minute bars; the daily training simulator decides before the open from daily bars."""
    return None


# ---------------------------------------------------------------- report
def _f(x, k=3):
    return "na" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{k}f}"


def write_md(res, live_test=None):
    V = res["variants"]
    L = [f"# Tool \"intraday\": 30-minute models for the sleeves' looks (run {datetime.now(ET).strftime('%Y-%m-%d')})", "",
         f"Pre-registration: `runs/ml/research/intraday_prereg.md`. Data: `data/bars30`, {res['symbols']} stocks "
         f"(+ ETFs for part a), {res['train_from']}..{res['test_window'][1]} (one year; full days only). "
         f"**Out-of-sample test window: {res['test_window'][0]}..{res['test_window'][1]} ({res['test_days']} sessions)**, "
         "walk-forward via `ml_eval.walk_forward` (steps = 30-minute slots, monthly expanding refits, embargo = horizon). "
         f"Costs {res['cost_bps_round_trip']:g} bps per round trip per leg. Survivor-biased universe (today's S&P 500): every "
         "figure is optimistic.", "", "## Verdict", ""]
    ship = [k for k in ("mim", "xs_lin", "xs_gbm", "xs_hks", "range") if V[k]["pass"]]
    L += [f"- **Ships:** {', '.join(ship) if ship else 'nothing'}." + (" The range model (part c) beats the naive same-time "
          "average and its 10-90% intervals are calibrated out of sample." if "range" in ship else ""),
          "- **Does not ship:** " + ", ".join(k for k in ("mim", "xs_lin", "xs_gbm", "xs_hks", "range") if k not in ship) +
          ". No intraday return or direction forecast beat costs.", ""]
    m = V["mim"]
    L += ["## (a) Market intraday momentum (variant mim)", "",
          f"Pooled OLS over {len(m['etfs'])} ETFs ({', '.join(m['etfs'])}): 15:30-16:00 return on the previous-close-to-10:00 "
          f"return and the 15:00-15:30 return. {m['test_days']} test sessions.", "",
          "| measure | value |", "|---|---|",
          f"| OOS R^2 vs a zero forecast | {_f(m['r2_vs_zero'], 4)} |",
          f"| timing rule (long/short 15:30-close by forecast sign), gross per day | {_f(m['timing_gross']['mean_bps'], 2)} bps (t {_f(m['timing_gross']['t'], 2)}) |",
          f"| same, net of 10 bps | {_f(m['timing_net']['mean_bps'], 2)} bps (t {_f(m['timing_net']['t'], 2)}) |",
          f"| net by half-year | {', '.join(f'{k} {_f(v, 1)}' for k, v in m['net_by_half_bps'].items())} bps |",
          f"| Gao-Han-Li-Zhou sign rule (no fit), net | {_f(m['sign_rule_net']['mean_bps'], 2)} bps (t {_f(m['sign_rule_net']['t'], 2)}) |",
          f"| 80% interval coverage | {_f(m['cover80'], 3)} |",
          f"| P(up) Brier vs climatology | {_f(m['brier'], 4)} vs {_f(m['brier_climatology'], 4)} |",
          f"| last fit: slope on first half-hour, on 15:00-15:30 | {_f(m['last_coef']['b_first_halfhour'], 4)}, {_f(m['last_coef']['b_1500_1530'], 4)} |",
          f"| full-sample corr(first half-hour, last half-hour) | {_f(m['full_sample_corr_r1_y'], 4)} |", "",
          "Per ETF (net bps per day, OOS R^2): " + "; ".join(f"{k} {_f(v['net_bps'], 1)}, {_f(v['r2'], 3)}" for k, v in m["per_etf"].items()) + ".", "",
          f"**{'PASS' if m['pass'] else 'FAIL'}.** The forecast's sign earned nothing gross (about 0 bps a day), so the net result "
          "is just the 10 bps round trip. The first-half-hour slope is near zero and the full-sample correlation is 0.05: the "
          "Gao-Han-Li-Zhou effect does not show in this one-year sample. A perfect sign would earn the mean |15:30-16:00| move "
          "(about 12 bps for SPY), so costs alone would not rule the idea out; the lack of signal does.", "",
          "**Time-of-day profile** (descriptive, whole sample): mean SPY return and t, mean |return| for SPY and the average "
          "stock, and the stock volume share by half-hour.", "",
          "| slot | SPY mean bps | t | SPY mean abs bps | stock mean abs bps | stock volume % |", "|---|---|---|---|---|---|"]
    for k, v in res["time_of_day"].items():
        L.append(f"| {k} | {v['spy_mean_bps']} | {v['spy_t']} | {v['spy_abs_bps']} | {v['stock_abs_bps']} | {v['stock_volume_share_pct']} |")
    L += ["", "## (b) Cross-sectional relative returns (variants xs_lin, xs_gbm, xs_hks)", "",
          "30- and 60-minute horizons are scored with `ml_eval.score` over every test slot (pooled across looks); the rest-of-day "
          "horizon per look (13 day-level scorecards; a look would need IC t >= 3). Decile spread = top minus bottom decile of "
          "relative returns per period, gross; net subtracts 20 bps (both legs, 10 bps round trip each).", "",
          "| variant | horizon | IC | IC t | day-clustered IC t | gross spread bps | net spread bps | costs / gross | IC by half | pass |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v in ("xs_lin", "xs_gbm", "xs_hks"):
        for hz in ("30", "60"):
            s = V[v][hz]
            cs = s.get("cost_share_of_gross")
            L.append(f"| {v} | {hz} min | {_f(s['ic'], 4)} | {_f(s['ic_t'], 2)} | {_f(s['ic_day_t'], 2)} | {_f(s['spread_gross_pct'] * 100, 2)} | "
                     f"{_f(s['spread_net_pct'] * 100, 2)} | {'>100% (gross <= 0)' if cs is None else f'{cs * 100:.0f}%'} | "
                     f"{', '.join(f'{k} {_f(x, 4)}' for k, x in s['ic_by_half'].items())} | {'PASS' if s['pass'] else 'fail'} |")
        s = V[v]["rod"]
        best = max(s["by_look"].items(), key=lambda kv: kv[1]["ic_t"] if kv[1]["ic_t"] == kv[1]["ic_t"] else -9)
        cs = s.get("cost_share_of_gross")
        L.append(f"| {v} | rest of day | (per look) | best look {best[0]}: {_f(best[1]['ic_t'], 2)} | {_f(s['ic_day_t'], 2)} | "
                 f"{_f(s['spread_gross_pct'] * 100, 2)} (mean of looks) | best look {_f(best[1]['spread_net_pct'] * 100, 2)} | "
                 f"{'>100% (gross <= 0)' if cs is None else f'{cs * 100:.0f}%'} | | {'PASS' if s['pass'] else 'fail'} |")
    L += ["", "Rest-of-day by look (IC t / gross spread bps) for each variant:", ""]
    for v in ("xs_lin", "xs_gbm", "xs_hks"):
        L.append(f"- {v}: " + "; ".join(f"{k} {_f(x['ic_t'], 1)}/{_f(x['spread_gross_pct'] * 100, 1)}" for k, x in V[v]["rod"]["by_look"].items()))
    L += ["", "Calibration of the return forecasts (realized vs predicted mean, bps, by predicted decile; and coverage of the "
          "in-sample-residual 80% interval):", ""]
    for v in ("xs_lin", "xs_gbm", "xs_hks"):
        for hz in ("30", "60", "rod"):
            c = V[v][hz]["calibration"]
            cov = f", 80% coverage {_f(c['cover80'], 3)}" if "cover80" in c else ""
            if v == "xs_hks":       # a rank signal with no return scale: realized only
                L.append(f"- {v} {hz} (rank signal; realized bps by decile, low to high): " + " ".join(f"{d['real']}" for d in c["by_decile_bps"]))
            else:
                L.append(f"- {v} {hz}: " + " ".join(f"{d['pred']}/{d['real']}" for d in c["by_decile_bps"]) + cov)
    L += ["", "**FAIL for all three.** The fitted models (xs_lin, xs_gbm) have no reliable skill: IC t at most about 2 (pooled 1.9 for the best) and forecasts "
          "far more spread out than the outcomes (the top decile's predicted +3 to +6 bps realized about 0). Same-time-of-day "
          "periodicity (xs_hks, Heston-Korajczyk-Sadka) is a real effect, IC 0.012-0.014 with t 4.5 at 30 and 60 minutes and "
          "positive in both half-years, strongest at the 15:30 look (IC 0.039, t 4.8). But its gross decile spread is 1.6 bps "
          "(30 min), 3.0 bps (60 min) and 3.5-10 bps (rest of day) against 20 bps of long-short costs, so costs take 4 to 12 "
          "times the gross edge. Even long-only, the top decile beats the average stock by 1.0-1.8 bps against a 10 bps round "
          "trip. This agrees with the overnight family: the opening-gap reversal is gone by 10:00, and the gap adds nothing "
          "here.", ""]
    L += ["## (c) Intraday range (variant range): ships", "",
          "Targets: high - low over the next two 30-minute bars (\"next 60 minutes\"; at 15:30 just the last bar) and over the rest "
          "of the day, as % of the decision price. Per-slot OLS on the log range; point forecast = mean (log forecast + smearing); "
          "intervals = per-slot training residual quantiles 10/50/90. Naive = the symbol's 20-session mean of the same target at "
          "the same time of day.", "",
          "| target | OOS R^2 log (model / naive) | vs mean-of-log naive | R^2 in % of price (model / naive) | 80% coverage | below q10 / above q90 | by half-year | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for tk, lab in (("rng60", "next 60 min"), ("rngrod", "rest of day")):
        r = V["range"][tk]
        c = r["coverage80"]
        L.append(f"| {lab} | {_f(r['r2_log']['r2_model'])} / {_f(r['r2_log']['r2_naive'])} | {_f(r['r2_log_vs_mean_log_naive']['r2_naive'])} | "
                 f"{_f(r['r2_level']['r2_model'])} / {_f(r['r2_level']['r2_naive'])} | {_f(c['overall'])} | {_f(c['below_q10'])} / {_f(c['above_q90'])} | "
                 f"{', '.join(f'{k} {_f(x)}' for k, x in c['by_half'].items())} | {'PASS' if r['pass'] else 'fail'} |")
    for tk, lab in (("rng60", "next 60 min"), ("rngrod", "rest of day")):
        r = V["range"][tk]
        L += ["", f"**{lab}**: R^2 (log) by look, model/naive: " +
              "; ".join(f"{k} {_f(x['r2_model'], 2)}/{_f(x['r2_naive'], 2)}" for k, x in r["r2_log_by_slot"].items()) + ".",
              f"Coverage by look: {', '.join(f'{k} {x}' for k, x in r['coverage80']['by_slot'].items())}. "
              f"By naive-range quintile (low to high): {r['coverage80']['by_naive_quintile']}.",
              "Mean realized vs mean forecast (% of price) by forecast decile: " +
              " ".join(f"{d['forecast_pct']}/{d['realized_pct']}" for d in r["mean_by_forecast_decile"]) +
              f". Median realized / forecast mean: {_f(r['median_realized_over_forecast'], 2)} (ranges are right-skewed, so the "
              "median sits below the mean; the 50% quantile is served too)."]
    L += ["", "How to use it: a stop closer than the q10 of the rest-of-day range is likely to be touched by ordinary noise; "
          "the q90 is a stretch target. It says how far, not which way.", ""]
    if live_test:
        L += ["## Live test (fake gateway)", "",
              "`ml_research/intraday_test.py` replays 16 Oct 2026 at an 11:05 ET look with a fake gw (today's bars from "
              "gw.bars('1 D', '30 mins') including bars after the look, which live() must ignore). Checks: " +
              ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in live_test["checks"].items()) +
              ". ZZZZ (not cached) got its history from gw.bars('1 M') and implied vol from gw.stats.", ""]
    L += ["## Caveats", "",
          "- One year of 30-minute bars; the test window is about nine months, every look of every session.",
          "- Live, the overnight gap is not dividend-corrected (ex-dates unknown), so ex-dividend days slightly overstate |gap|. "
          "Half-days (13:00 close) are not modelled.",
          "- The live history comes from the `data/bars30` cache while it is at most 10 days old, else from gw.bars('1 M'). The "
          "cache ends 16 Oct 2026, so from about 27 Oct each symbol costs one extra gateway call unless the cache is refreshed.",
          "- sim_provider returns None: the daily training simulator decides before the open and has no 30-minute bars.", "",
          "## Track record line", "", TRACK_RECORD, ""]
    (OUT / "intraday.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if "--final" in sys.argv:
        fit_final()
    elif "--md" in sys.argv:
        lt = OUT / "intraday_livetest.json"
        write_md(json.loads((OUT / "intraday.json").read_text(encoding="utf-8")),
                 json.loads(lt.read_text(encoding="utf-8")) if lt.exists() else None)
    else:
        validate()
