"""Tool "events": SEC 8-K events, earnings dates and an event-conditional range check
(pre-registration: runs/ml/research/events_prereg.md; results: runs/ml/research/events.md and events.json).

What it does:
  - event features per symbol and session from data/edgar (8-K item groups, earnings timing, hours since the filing) and
    from the IBKR headline archive (data/news), typed by a regex;
  - an event-conditional correction of the shipped range models (vol2 range1, intraday next60 / rest_of_day), fit on
    their out-of-sample residuals, walk-forward monthly, against a pre-registered bar;
  - an earnings-date predictor from each firm's own item-2.02 history: P(an earnings 8-K before the decision k sessions
    from now), calibrated walk-forward.

    .venv\\Scripts\\python.exe ml_research\\events.py            # validate(): events.json, events.md, events_oos.npz
    .venv\\Scripts\\python.exe ml_research\\events.py --live AAPL MSFT   # live() (EDGAR submissions for those symbols)

Point-in-time rule: a filing is known from its accepted_utc (data/edgar/filings.csv; never filingDate, never the raw
acceptanceDateTime, whose time zone convention varies by company). Decision times: 09:00 ET for the daily model and
the simulator, 09:30 / 10:00 ET for the intraday looks.
"""

import json
import math
import pickle
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "ml" / "research"
EDGAR = ROOT / "data" / "edgar"
NEWS = ROOT / "data" / "news"
LIVE_DIR = EDGAR / "live"
MODEL = OUT / "events_model.pkl"
OOS = OUT / "events_oos.npz"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
MAX_SYMBOLS = 20
LIVE_MIN_AGE_S = 25 * 60          # a symbol's submissions JSON is refetched at most once per look (looks are 30 min apart)
KS = (1, 5, 10)
RANGE_TEST_FROM, INTRA_TEST_FROM, INTRA_OOS_FROM = "20250303", "20260202", "20260102"
NBOOT = 2000

GROUPS = {"mna": {"1.01", "1.02", "2.01"}, "mgmt": {"5.02"}, "cap": {"2.03", "3.02", "3.03"},
          "bad": {"2.05", "2.06", "4.01", "4.02"}, "vote": {"5.07"}}
KNOWN = set().union(*GROUPS.values()) | {"2.02", "7.01", "8.01", "9.01"}
ITEM_LABEL = {"1.01": "material agreement", "1.02": "agreement terminated", "1.03": "bankruptcy", "2.01": "acquisition/disposal done",
              "2.02": "earnings", "2.03": "new debt", "2.04": "debt acceleration", "2.05": "exit costs", "2.06": "impairment",
              "3.01": "listing notice", "3.02": "unregistered equity sale", "3.03": "holder rights changed", "4.01": "auditor change",
              "4.02": "non-reliance on past financials", "5.01": "change of control", "5.02": "officer/director change",
              "5.03": "charter/bylaws", "5.07": "shareholder vote", "7.01": "Reg FD release", "8.01": "other event",
              "9.01": "exhibits"}
EVF = ["earn_new", "earn_old"] + list(GROUPS) + ["fd", "other", "lh", "earn_ehist"]
HLF = ["up", "down", "init", "anl_other", "h_earn", "h_other"]

# ---------------------------------------------------------------- headline regex (pre-registered classes)
RX = {
    "up": re.compile(r"\bupgraded\b", re.I),
    "down": re.compile(r"\bdowngraded\b", re.I),
    "init": re.compile(r"\binitiated\b", re.I),
    "anl_other": re.compile(r"\b(reiterated|resumed|assumed|reinstated)\b|\btarget\b", re.I),
    "earnings": re.compile(r"\b(earnings|EPS|results?|Q[1-4]|[1-4]Q|quarter(ly)?|beats?|beat-and-raise|miss(es|ed)?|"
                           r"revenue|sales|comps?|same[- ]store|top[- ]and[- ]bottom[- ]line|FY ?\d{2,4}|profit|loss)\b", re.I),
    "guidance": re.compile(r"\b(guidance|guides?|guided|outlook|forecasts?|raises? .{0,20}(view|target|fy|full[- ]year)|"
                           r"lowers? .{0,20}(view|fy|full[- ]year)|cuts? .{0,20}(view|fy|full[- ]year))\b", re.I),
    "offering": re.compile(r"\b(offering|secondary|convertible|dilut\w*|share sale|stock sale|prices? .{0,30}shares)\b", re.I),
    "mna": re.compile(r"\b(acquir\w*|acquisition|merger|merge|buyout|takeover|tender offer|spin[- ]?off|divest\w*|"
                      r"to buy|deal to|bid for|activist)\b", re.I),
    "fda": re.compile(r"\b(FDA|approval|approved|clinical|trials?|phase [123I]+|PDUFA|CRL)\b", re.I),
    "lawsuit": re.compile(r"\b(lawsuit|sues?|sued|settle\w*|probe|investigation|antitrust|DOJ|FTC|verdict|court|subpoena)\b", re.I),
    "mgmt": re.compile(r"\b(CEO|CFO|chief executive|steps? down|resign\w*|appoint\w*|succession|successor|retire\w*)\b", re.I),
}


def headline_classes(provider, text):
    """Set of regex classes of one IBKR headline. Analyst-action wire (BRFUPDN): up/down/init/anl_other only."""
    if provider == "BRFUPDN":
        for k in ("up", "down", "init"):
            if RX[k].search(text):
                return {k}
        return {"anl_other"}
    return {k for k in ("earnings", "guidance", "offering", "mna", "fda", "lawsuit", "mgmt") if RX[k].search(text)}


# ---------------------------------------------------------------- calendar and time
def _iso(d):
    d = str(d)
    return d if "-" in d else f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def et_epoch(day, hh, mm):
    y, m, dd = map(int, _iso(day).split("-"))
    return int(datetime(y, m, dd, hh, mm, tzinfo=ET).timestamp())


_CAL = None


def calendar():
    """Trading days 'YYYY-MM-DD' from Sep 2023 through 2027 (daily bars plus extension), and early-close days."""
    global _CAL
    if _CAL is None:
        from ml_research import edgar_fetch as ef
        _CAL = (ef.trading_calendar(), set(ef.EARLY))
    return _CAL


def close_epoch(day):
    early = calendar()[1]
    return et_epoch(day, 13, 0) if _iso(day) in early else et_epoch(day, 16, 0)


def windows(days, hh, mm):
    """For each session day: decision time tau, window start ws = min(tau - 24h, previous session's close), the previous
    close pc (all UTC epoch seconds)."""
    cal = calendar()[0]
    pos = {d: i for i, d in enumerate(cal)}
    tau, ws, pc = [], [], []
    for d in days:
        d = _iso(d)
        i = pos[d]
        t = et_epoch(d, hh, mm)
        c = close_epoch(cal[i - 1]) if i > 0 else t - 86400
        tau.append(t), pc.append(c), ws.append(min(t - 86400, c))
    return np.array(tau, np.int64), np.array(ws, np.int64), np.array(pc, np.int64)


def _epoch_utc(s):
    return int(datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp())


# ---------------------------------------------------------------- loading
_FIL = None


def filings():
    """Per symbol: dict of sorted epoch arrays: 'all', 'earn' (earnings_dates.json), each group, 'fd', 'other';
    plus 'rows' [(t, items, accession)] for display."""
    global _FIL
    if _FIL is not None:
        return _FIL
    import csv
    out = {}
    with open(EDGAR / "filings.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = _epoch_utc(r["accepted_utc"])
            items = {x.strip() for x in (r["items"] or "").split(",") if x.strip()}
            for s in r["symbols"].split("|"):
                out.setdefault(s, []).append((t, items, r["accessionNumber"]))
    ed = json.loads((EDGAR / "earnings_dates.json").read_text())["symbols"]
    F = {}
    for s in set(out) | set(ed):
        rows = sorted(out.get(s, []))
        d = {"rows": rows, "all": np.array([t for t, _, _ in rows], np.int64)}
        for g, its in GROUPS.items():
            d[g] = np.array([t for t, i, _ in rows if i & its], np.int64)
        d["fd"] = np.array([t for t, i, _ in rows if i & {"7.01", "8.01"} and "2.02" not in i], np.int64)
        d["other"] = np.array([t for t, i, _ in rows if i - KNOWN], np.int64)
        e = sorted({(_epoch_utc(x["accepted_utc"]), x["first_session"]) for x in ed.get(s, [])})
        d["earn"] = np.array([t for t, _ in e], np.int64)
        d["earn_first"] = [fs for _, fs in e]
        F[s] = d
    _FIL = F
    return F


_HL = None


def headlines():
    """Per symbol: list of (epoch, provider, headline, classes), sorted by time."""
    global _HL
    if _HL is not None:
        return _HL
    H = {}
    for p in NEWS.glob("*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = []
        for h in j.get("headlines", []):
            try:
                t = _epoch_utc(h["time"])
            except (KeyError, ValueError):
                continue
            rows.append((t, h.get("provider", ""), h.get("headline", ""), headline_classes(h.get("provider", ""), h.get("headline", ""))))
        H[j.get("symbol", p.stem)] = sorted(rows, key=lambda x: x[0])
    _HL = H
    return H


def _count(arr, lo, hi):
    return np.searchsorted(arr, hi, "left") - np.searchsorted(arr, lo, "left")


# ---------------------------------------------------------------- earnings reaction history (ehist)
def reaction_table(data):
    """(first_session 'YYYY-MM-DD', symbol, ratio): range on the first reacting session / mean range of the 20 before."""
    F = filings()
    r = np.full(data.c.shape, np.nan)
    r[:, 1:] = (data.h[:, 1:] - data.l[:, 1:]) / data.c[:, :-1]
    pos = {_iso(x): i for i, x in enumerate(data.dates)}
    rows = []
    for s, d in F.items():
        j = data.k.get(s)
        if j is None:
            continue
        for fs in sorted(set(d["earn_first"])):
            i = pos.get(fs)
            if i is None or i < 21:
                continue
            base = np.nanmean(r[j, i - 20:i])
            if np.isfinite(r[j, i]) and np.isfinite(base) and base > 0:
                rows.append((fs, s, float(r[j, i] / base)))
    return sorted(rows)


def ehist_fn(table):
    """f(symbol, decision day 'YYYY-MM-DD') -> log median reaction over the firm's last <= 8 earnings whose reacting
    session is before the day (pooled median of all such if fewer than 2)."""
    days = [x[0] for x in table]
    vals = np.array([x[2] for x in table])
    by = {}
    for fs, s, v in table:
        by.setdefault(s, []).append((fs, v))
    cache = {}

    def f(sym, day):
        day = _iso(day)
        own = [v for fs, v in by.get(sym, []) if fs < day][-8:]
        if len(own) >= 2:
            return math.log(float(np.median(own)))
        if day not in cache:
            n = int(np.searchsorted(np.array(days), day, "left")) if days else 0
            cache[day] = math.log(float(np.median(vals[:n]))) if n >= 20 else math.log(2.5)
        return cache[day]
    return f


# ---------------------------------------------------------------- feature panel
def event_panel(syms, days, hh, mm, ehist=None):
    """Dict of (S, D) arrays: 8-K features (EVF), headline features (HLF), ev_any, hl_any, sess_since_earn.
    days: session days; decision at hh:mm ET on each."""
    F, H = filings(), headlines()
    tau, ws, pc = windows(days, hh, mm)
    S, D = len(syms), len(days)
    cal = calendar()[0]
    cpos = {d: i for i, d in enumerate(cal)}
    dpos = np.array([cpos[_iso(d)] for d in days])
    tau_cal = np.array([et_epoch(d, 9, 0) for d in cal], np.int64)
    P = {k: np.zeros((S, D)) for k in EVF + HLF + ["ev_any", "hl_any", "n8k"]}
    P["sess_since_earn"] = np.full((S, D), np.nan)
    for j, s in enumerate(syms):
        d = F.get(s)
        if d is not None and len(d["all"]):
            n = _count(d["all"], ws, tau)
            P["n8k"][j] = n
            P["ev_any"][j] = n > 0
            P["earn_new"][j] = _count(d["earn"], pc, tau) > 0
            P["earn_old"][j] = _count(d["earn"], ws, pc) > 0
            for g in list(GROUPS) + ["fd", "other"]:
                P[g][j] = _count(d[g], ws, tau) > 0
            hi = np.searchsorted(d["all"], tau, "left")
            last = np.where(n > 0, d["all"][np.maximum(hi - 1, 0)], tau)
            P["lh"][j] = np.where(n > 0, np.log1p((tau - last) / 3600.0), 0.0)
            if ehist is not None:
                P["earn_ehist"][j] = P["earn_new"][j] * np.array([ehist(s, x) for x in days])
            if len(d["earn"]):
                ie = np.searchsorted(d["earn"], tau, "left")           # earnings known before tau
                lastj = np.searchsorted(tau_cal, d["earn"], "left")   # window index of each report
                ok = ie > 0
                P["sess_since_earn"][j, ok] = dpos[ok] - lastj[ie[ok] - 1]
        h = H.get(s)
        if h:
            ht = np.array([x[0] for x in h], np.int64)
            for k in HLF:
                if k in ("h_earn", "h_other"):
                    sel = [x[1] != "BRFUPDN" and (("earnings" in x[3]) == (k == "h_earn")) for x in h]
                else:
                    sel = [k in x[3] for x in h]
                tk = ht[np.array(sel, bool)]
                if len(tk):
                    P[k][j] = _count(tk, ws, tau) > 0
            P["hl_any"][j] = _count(ht, ws, tau) > 0
    return P


# ---------------------------------------------------------------- models: OLS ratio, smoothed quantile regression, LightGBM
def _std(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def ols_fit(X, y):
    mu, sd = _std(X)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    return {"kind": "ols", "mu": mu, "sd": sd, "b": b}


def qr_fit(X, y, q, eps=0.01):
    """Linear quantile regression by L-BFGS on a softplus-smoothed pinball loss (sklearn's exact solver is too slow
    at this size); eps is in the units of y (log ratio)."""
    from scipy.optimize import minimize
    mu, sd = _std(X)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    b0 = np.linalg.lstsq(Z, y, rcond=None)[0]
    b0[0] += np.quantile(y - Z @ b0, q)
    n = len(y)

    def fg(b):
        u = y - Z @ b
        a = -u / eps
        loss = q * u + eps * np.logaddexp(0, a)
        g = -(Z.T @ (q - 1 / (1 + np.exp(-a)))) / n
        return loss.mean(), g
    r = minimize(fg, b0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return {"kind": "ols", "mu": mu, "sd": sd, "b": r.x}


def lin_pred(m, X):
    return m["b"][0] + ((X - m["mu"]) / m["sd"]) @ m["b"][1:]


def gbm_fit(X, y, obj, alpha=None):
    import lightgbm as lgb
    p = {"objective": obj, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.9,
         "seed": 0, "verbose": -1, "num_threads": 8}
    if alpha is not None:
        p["alpha"] = alpha
    return {"kind": "gbm", "m": lgb.train(p, lgb.Dataset(X, y), 200)}


def predict(m, X):
    return m["m"].predict(X) if m["kind"] == "gbm" else lin_pred(m, X)


def fit_residual(X, ratio, lr, kind):
    """ratio = y / p (level; None for intraday), lr = log(y / p) (log residual). Returns dict of models: point and q10/q90."""
    out = {}
    if ratio is not None:
        lo_, hi_ = np.quantile(ratio, [0.01, 0.995])
        tgt = np.clip(ratio, lo_, hi_)
    else:
        lo_, hi_ = np.quantile(lr, [0.005, 0.995])
        tgt = np.clip(lr, lo_, hi_)
    if kind == "lin":
        out["point"] = ols_fit(X, tgt)
        out["q10"], out["q90"] = qr_fit(X, lr, 0.1), qr_fit(X, lr, 0.9)
    else:
        out["point"] = gbm_fit(X, tgt, "regression")
        out["q10"], out["q90"] = gbm_fit(X, lr, "quantile", 0.1), gbm_fit(X, lr, "quantile", 0.9)
    return out


def apply_residual(M, X, level=True):
    """-> (point multiplier or log shift, q10, q90) on the log scale for the quantiles."""
    pt = predict(M["point"], X)
    a, b = predict(M["q10"], X), predict(M["q90"], X)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    if level:
        pt = np.clip(pt, 0.5, 6.0)
    return pt, lo, hi


# ---------------------------------------------------------------- scoring helpers
def r2(y, f):
    return float(1 - np.mean((y - f) ** 2) / np.var(y)) if len(y) > 2 else float("nan")


def boot_r2_diff(y, f_new, f_old, sess, n=NBOOT, seed=0):
    """Day-clustered bootstrap of R2(new) - R2(old): resample sessions with replacement. Returns (5%, 50%, 95%)."""
    u, inv = np.unique(sess, return_inverse=True)
    k = len(u)
    agg = lambda w: np.bincount(inv, w, k)
    N, Sy, Syy = agg(np.ones_like(y)), agg(y), agg(y * y)
    En, Eo = agg((y - f_new) ** 2), agg((y - f_old) ** 2)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = np.bincount(rng.integers(0, k, k), minlength=k)
        nn, sy, syy = c @ N, c @ Sy, c @ Syy
        sst = syy - sy * sy / nn
        out.append((c @ Eo - c @ En) / sst)
    return [round(float(x), 4) for x in np.quantile(out, [0.05, 0.5, 0.95])]


def cover(y, lo, hi):
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "coverage": round(float(((y >= lo) & (y <= hi)).mean()), 4),
            "below": round(float((y < lo).mean()), 4), "above": round(float((y > hi).mean()), 4)}


def half(day):
    day = str(day).replace("-", "")
    return day[:4] + ("H1" if day[4:6] <= "06" else "H2")


def month_starts(days, first):
    """Indices of the first session of each month from `first` on."""
    out, seen = [], set()
    for i, d in enumerate(days):
        d = str(d).replace("-", "")
        if d >= first and d[:6] not in seen:
            seen.add(d[:6])
            out.append(i)
    return out


# ---------------------------------------------------------------- headline check (regex vs 8-K labels)
def headline_check():
    F, H = filings(), headlines()
    cal = calendar()[0]
    cpos = {d: i for i, d in enumerate(cal)}
    close_cal = np.array([close_epoch(d) for d in cal], np.int64)
    lab_items = {"mna": GROUPS["mna"], "mgmt": GROUPS["mgmt"], "offering": {"3.02", "2.03", "3.03"}}
    res = {}
    stats = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in ["earnings", "mna", "mgmt", "offering"]}
    n_brfg = 0
    for s, rows in H.items():
        d = F.get(s)
        if d is None:
            continue
        ef = {cpos[x] for x in d["earn_first"] if x in cpos}
        # first reacting session of each 8-K by group
        fs_all = np.searchsorted(close_cal, d["all"], "left")      # first session whose close is after the filing
        grp_sess = {k: {int(fs_all[i]) for i, (_, its, _) in enumerate(d["rows"]) if its & v} for k, v in lab_items.items()}
        for t, prov, text, cls in rows:
            if prov == "BRFUPDN":
                continue
            n_brfg += 1
            si = int(np.searchsorted(close_cal, t, "left"))       # the headline's session
            near = lambda S_: any(si + o in S_ for o in (-1, 0, 1))
            for k in stats:
                lab = near(ef) if k == "earnings" else near(grp_sess[k])
                pr = k in cls
                stats[k]["tp" if lab and pr else "fp" if pr else "fn" if lab else "tn"] += 1
    for k, v in stats.items():
        p = v["tp"] / max(v["tp"] + v["fp"], 1)
        r = v["tp"] / max(v["tp"] + v["fn"], 1)
        res[k] = {**v, "precision": round(p, 3), "recall": round(r, 3)}
    res["n_brfg"] = n_brfg
    res["n_brfupdn"] = sum(1 for rows in H.values() for x in rows if x[1] == "BRFUPDN")
    e = res["earnings"]
    res["encoder_needed"] = not (e["precision"] >= 0.80 and e["recall"] >= 0.70)
    return res


# ---------------------------------------------------------------- earnings-date predictor
DAY = 86400
OFF_LO, OFF_HI, SMOOTH = -200, 320, 0.005


def firms():
    """One symbol per company (CIK), in universe order: (symbol, [symbols sharing the CIK])."""
    m = json.loads((EDGAR / "cik_map.json").read_text())
    by = {}
    for s, v in m.items():
        by.setdefault(v["cik"], []).append(s)
    return [(v[0], v) for v in by.values()]


def next_pred(T, variant):
    """Predicted epoch of the next earnings 8-K from the known report epochs T (sorted, >= 1)."""
    L = T[-1]
    if variant == "E1":
        if L - T[0] < 330 * DAY:                   # under a year of history: last + 91 (pre-registered fallback)
            return int(L + 91 * DAY)
        c = T + 364 * DAY
        c = c[c > L + 30 * DAY]
        return int(c.min()) if len(c) else int(L + 91 * DAY)
    g = np.diff(T) / DAY
    g = g[(g >= 60) & (g <= 120)]
    return int(L + (np.median(g) if len(g) >= 2 else 91) * DAY)


_TC = None


def tau_cal():
    global _TC
    if _TC is None:
        _TC = np.array([et_epoch(d, 9, 0) for d in calendar()[0]], np.int64)
    return _TC


def cycle_errors(variant):
    """All firms' (realized epoch, error in sessions) of the next-report prediction, sorted by realized time."""
    tc = tau_cal()
    F = filings()
    out = []
    for s, _ in firms():
        T = F.get(s, {}).get("earn", np.array([], np.int64))
        J = np.searchsorted(tc, T, "left")
        for n in range(1, len(T)):
            p = int(np.searchsorted(tc, next_pred(T[:n], variant), "left"))
            out.append((int(T[n]), int(J[n] - p)))
    out.sort()
    return np.array([x[0] for x in out], np.int64), np.array([x[1] for x in out], np.int64)


def hist_cum(err):
    """Cumulative mass over offsets OFF_LO..OFF_HI (clipped) with SMOOTH uniform mass."""
    n = OFF_HI - OFF_LO + 1
    h = np.bincount(np.clip(err, OFF_LO, OFF_HI) - OFF_LO, minlength=n).astype(float)
    h = (1 - SMOOTH) * h / max(h.sum(), 1) + SMOOTH / n
    return np.concatenate([[0.0], np.cumsum(h)])            # C[i] = mass of offsets < OFF_LO + i


def raw_prob(C, pidx, d, k):
    """P(report window index in [d+1, d+k] | index >= d+1) under the error distribution C around pidx."""
    m = lambda o: C[np.clip(o - OFF_LO, 0, len(C) - 1)]   # mass of offsets < o
    lo = d + 1 - pidx
    den = 1.0 - m(lo)
    num = m(lo + k) - m(lo)
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def cond_dist(C, pidx, d):
    """Most likely report window index >= d+1, and the 10/90% points of the conditional distribution."""
    h = np.diff(C)
    lo = max(d + 1 - pidx - OFF_LO, 0)
    w = h[lo:]
    if w.sum() <= 0:
        return None, None, None
    cw = np.cumsum(w) / w.sum()
    base = pidx + OFF_LO + lo
    return int(base + np.argmax(w)), int(base + np.searchsorted(cw, 0.1)), int(base + np.searchsorted(cw, 0.9))


def predictor_rows(variant, first="2024-03-01", last="2026-10-16"):
    """Rows (firm, d, pidx, outcome_k) for every firm and trading day in [first, last] with at least one known report."""
    tc = tau_cal()
    cal = calendar()[0]
    F = filings()
    d0, d1 = cal.index(first) if first in cal else next(i for i, x in enumerate(cal) if x >= first), cal.index(last)
    R = {"firm": [], "d": [], "pidx": [], **{f"y{k}": [] for k in KS}}
    for fi, (s, _) in enumerate(firms()):
        T = F.get(s, {}).get("earn", np.array([], np.int64))
        if not len(T):
            continue
        J = np.searchsorted(tc, T, "left")
        P = [None] + [int(np.searchsorted(tc, next_pred(T[:n], variant), "left")) for n in range(1, len(T) + 1)]
        for d in range(d0, d1 + 1):
            n = int(np.searchsorted(T, tc[d], "left"))
            if n == 0:
                continue
            R["firm"].append(fi), R["d"].append(d), R["pidx"].append(P[n])
            for k in KS:
                R[f"y{k}"].append(int(((J >= d + 1) & (J <= d + k)).any()))
    return {k: np.array(v) for k, v in R.items()}


def predictor_wf(variant, R, eval_from="2025-01-02", last="2026-10-16"):
    """Walk-forward monthly: error distribution from cycles realized before the month; isotonic recalibration on rows
    whose window closed before the month. Returns raw and calibrated probabilities per k, plus the base rate."""
    from sklearn.isotonic import IsotonicRegression
    tc = tau_cal()
    cal = calendar()[0]
    tr, er = cycle_errors(variant)
    dlast = cal.index(last)
    months = sorted({cal[d][:7] for d in R["d"]})
    raw = {k: np.full(len(R["d"]), np.nan) for k in KS}
    cal_p = {k: np.full(len(R["d"]), np.nan) for k in KS}
    base = {k: np.full(len(R["d"]), np.nan) for k in KS}
    nxt = np.full((len(R["d"]), 3), -1)
    mon = np.array([cal[d][:7] for d in R["d"]])
    for mth in months:
        rows = np.where(mon == mth)[0]
        m0 = int(R["d"][rows].min())
        C = hist_cum(er[tr < tc[m0]])
        for k in KS:
            raw[k][rows] = raw_prob(C, R["pidx"][rows], R["d"][rows], k)
            done = np.where((R["d"] + k <= m0) & np.isfinite(raw[k]))[0]
            done = done[R["d"][done] + k <= dlast]
            base[k][rows] = R[f"y{k}"][done].mean() if len(done) else np.nan
            if len(done) >= 2000:
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw[k][done], R[f"y{k}"][done])
                cal_p[k][rows] = iso.predict(raw[k][rows])
        for i in rows:
            nxt[i] = cond_dist(C, int(R["pidx"][i]), int(R["d"][i]))
    return raw, cal_p, base, nxt


def brier_card(R, p, base, k, eval_from, last):
    cal = calendar()[0]
    e0, dl = cal.index(eval_from), cal.index(last)
    m = (R["d"] >= e0) & (R["d"] + k <= dl) & np.isfinite(p) & np.isfinite(base)
    y, pp, bb, sess = R[f"y{k}"][m], p[m], base[m], R["d"][m]
    bs, bb_ = np.mean((pp - y) ** 2), np.mean((bb - y) ** 2)
    u, inv = np.unique(sess, return_inverse=True)
    A, B = np.bincount(inv, (pp - y) ** 2), np.bincount(inv, (bb - y) ** 2)
    rng = np.random.default_rng(1)
    bss = []
    for _ in range(NBOOT):
        c = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        bss.append(1 - (c @ A) / (c @ B))
    edges = [0, .02, .05, .1, .2, .3, .5, .7, .9, 1.0001]
    rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (pp >= a) & (pp < b)
        if s.sum():
            rel.append({"bin": f"{a:.2f}-{min(b, 1):.2f}", "n": int(s.sum()), "pred": round(float(pp[s].mean()), 4),
                        "obs": round(float(y[s].mean()), 4), "gap_pp": round(float(100 * (y[s].mean() - pp[s].mean())), 2)})
    ok_rel = all(abs(r["gap_pp"]) <= 5 for r in rel if r["n"] >= 100)
    q = np.quantile(bss, [0.05, 0.5, 0.95])
    return {"n": int(m.sum()), "sessions": int(len(u)), "base_rate": round(float(y.mean()), 4), "brier": round(float(bs), 5),
            "brier_base": round(float(bb_), 5), "bss": round(float(1 - bs / bb_), 4), "bss_boot_5_50_95": [round(float(x), 4) for x in q],
            "reliability": rel, "reliability_ok": ok_rel, "pass": bool(q[0] > 0 and ok_rel)}


def predictor_eval(log=print):
    """Both pre-registered variants, walk-forward; picks the better by Brier at k = 5. Also returns the per-row arrays
    of the chosen variant for the simulator."""
    res, keep = {}, {}
    cal = calendar()[0]
    tc = tau_cal()
    F = filings()
    fl = firms()
    for v in ("E1", "E2"):
        R = predictor_rows(v)
        raw, calp, base, nxt = predictor_wf(v, R)
        res[v] = {}
        for k in KS:
            res[v][f"k{k}"] = {"calibrated": brier_card(R, calp[k], base[k], k, "2025-01-02", "2026-10-16"),
                               "raw": brier_card(R, raw[k], base[k], k, "2025-01-02", "2026-10-16")}
            c = res[v][f"k{k}"]["calibrated"]
            log("predictor", v, k, json.dumps({x: c[x] for x in ("n", "base_rate", "bss", "bss_boot_5_50_95", "reliability_ok", "pass")}))
        # date accuracy: |most likely session - actual| on eval rows where a report followed within 10 sessions
        errs = []
        e0 = cal.index("2025-01-02")
        for i in np.where((R["d"] >= e0) & (R["y10"] == 1) & (nxt[:, 0] >= 0))[0]:
            J = np.searchsorted(tc, F[fl[R["firm"][i]][0]]["earn"], "left")
            j = J[J >= R["d"][i] + 1].min()
            errs.append(abs(int(nxt[i, 0]) - int(j)))
        errs = np.array(errs)
        res[v]["date_error_sessions_when_report_within_10"] = {
            "n": int(len(errs)), "exact": round(float((errs == 0).mean()), 3), "within1": round(float((errs <= 1).mean()), 3),
            "within3": round(float((errs <= 3).mean()), 3), "median": float(np.median(errs))}
        keep[v] = (R, raw, calp, nxt)
    win = min(("E1", "E2"), key=lambda v: res[v]["k5"]["calibrated"]["brier"])
    res["chosen"] = win
    res["pass"] = all(res[win][f"k{k}"]["calibrated"]["pass"] for k in KS)
    return res, keep[win]


def predictor_arrays(data, keep):
    """(S, D) arrays on data's grid: p1, p5, p10 (calibrated where available, else raw), next / lo / hi calendar index."""
    R, raw, calp, nxt = keep
    cal = calendar()[0]
    cpos = {d: i for i, d in enumerate(cal)}
    col = {cpos[_iso(x)]: t for t, x in enumerate(data.dates) if _iso(x) in cpos}
    S, D = len(data.sym), len(data.dates)
    A = {f"p{k}": np.full((S, D), np.nan) for k in KS}
    for k in ("next", "next_lo", "next_hi"):
        A[k] = np.full((S, D), -1, np.int32)
    fl = firms()
    for i in range(len(R["d"])):
        t = col.get(int(R["d"][i]))
        if t is None:
            continue
        for s in fl[R["firm"][i]][1]:
            j = data.k.get(s)
            if j is None:
                continue
            for k in KS:
                A[f"p{k}"][j, t] = calp[k][i] if np.isfinite(calp[k][i]) else raw[k][i]
            A["next"][j, t], A["next_lo"][j, t], A["next_hi"][j, t] = nxt[i]
    return A


# ---------------------------------------------------------------- event-conditional range: shared walk-forward
def _design(P, names, extra=None):
    return np.stack([P[n] for n in names] + (extra or []), axis=-1)


def _wf(X, ratio, lr, rows_model, train_ok, month_of, months, kind, level=True):
    """Monthly walk-forward on flattened rows. rows_model: rows the variant applies to; train_ok(m): rows usable for the
    fit of month m. Returns point (multiplier, or log shift), q10, q90 (NaN where not applied), number of fits."""
    n = len(lr)
    pt, q1, q9 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    fits = 0
    for m in months:
        tr = train_ok(m) & rows_model
        te = (month_of == m) & rows_model
        if tr.sum() < 200 or not te.any():
            continue
        M = fit_residual(X[tr], None if ratio is None else ratio[tr], lr[tr], kind)
        a, b, c = apply_residual(M, X[te], level)
        pt[te], q1[te], q9[te] = a, b, c
        fits += 1
    return pt, q1, q9, fits


def _block(yy, f_new, f_old, lo_new, hi_new, lo_old, hi_old, m, sess, level=True):
    out = {"n": int(m.sum()), "sessions": int(len(np.unique(sess[m])))}
    if not m.any():
        return out
    out.update({"r2_shipped": round(r2(yy[m], f_old[m]), 4), "r2_new": round(r2(yy[m], f_new[m]), 4)})
    out["r2_diff"] = round(out["r2_new"] - out["r2_shipped"], 4)
    out["cov_shipped"], out["cov_new"] = cover(yy[m], lo_old[m], hi_old[m]), cover(yy[m], lo_new[m], hi_new[m])
    if level:
        out["mean_y_over_shipped"] = round(float(yy[m].mean() / f_old[m].mean()), 3)
        out["mean_y_over_new"] = round(float(yy[m].mean() / f_new[m].mean()), 3)
    else:
        out["mean_log_err_shipped"] = round(float((yy[m] - f_old[m]).mean()), 3)
        out["mean_log_err_new"] = round(float((yy[m] - f_new[m]).mean()), 3)
    return out


# ---------------------------------------------------------------- range1 (daily, decision 09:00 ET)
def range1_eval(data, ehist, pA, log=print):
    z = np.load(OUT / "vol2_oos.npz", allow_pickle=True)
    assert [str(s) for s in z["symbols"]] == data.sym and [str(x) for x in z["dates"]] == data.dates
    S, D = len(data.sym), len(data.dates)
    p, lo, hi = z["range1"], z["range1_lo"], z["range1_hi"]
    y = np.full((S, D), np.nan)
    y[:, :-1] = (data.h[:, 1:] - data.l[:, 1:]) / data.c[:, :-1]
    gap = np.full((S, D), np.nan)
    gap[:, :-1] = data.o[:, 1:] / data.c[:, :-1] - 1
    days = data.dates[1:]                                # the decision session of column t is t + 1
    P = event_panel(data.sym, days, 9, 0, ehist)
    pad = lambda a: np.concatenate([a, np.full((S, 1), np.nan)], axis=1)
    P = {k: pad(v) for k, v in P.items()}
    p1 = np.full((S, D), np.nan)
    p1[:, :-1] = pA["p1"][:, 1:]
    eh = np.full((S, D), np.nan)
    F = filings()
    for j, s in enumerate(data.sym):
        if s in F:
            eh[j, :-1] = [ehist(s, x) for x in days]
    with np.errstate(all="ignore"):
        ok = np.isfinite(p) & np.isfinite(y) & (y > 0) & np.isfinite(lo) & np.isfinite(hi) & (p > 0)
    tcol = np.broadcast_to(np.arange(D), (S, D))
    dec = np.array(list(days) + ["99999999"])
    flat = lambda a: a[ok]
    yy, pp, ll, hh, tt = flat(y), flat(p), flat(lo), flat(hi), flat(tcol)
    fP = {k: flat(v) for k, v in P.items()}
    ev, hl = fP["ev_any"] > 0, fP["hl_any"] > 0
    month_of = dec[tt].astype("U6")
    months = sorted({dec[t][:6] for t in range(D - 1) if dec[t] >= RANGE_TEST_FROM})
    first_t = {m: min(t for t in range(D - 1) if dec[t][:6] == m) for m in months}
    train_ok = lambda m: tt <= first_t[m] - 2
    ratio, lr = yy / pp, np.log(yy / pp)
    XA = _design(fP, EVF)
    XB = _design(fP, EVF + HLF)
    ssn = np.minimum(np.nan_to_num(fP["sess_since_earn"], nan=200), 200)
    XC = np.column_stack([XB, np.log(pp), flat(eh), ssn, flat(p1)])
    variants = {"A": (XA, ev, "lin"), "B": (XB, ev | hl, "lin"), "C": (XC, np.ones(len(yy), bool), "gbm")}
    test = dec[tt] >= RANGE_TEST_FROM
    big = np.abs(flat(gap)) > 0.05
    earn = fP["earn_new"] > 0
    hv = np.array([half(x) for x in dec[tt]])
    out, served = {"test_rows": int(test.sum()), "event_share": round(float(ev[test].mean()), 4)}, {}
    for v, (X, rows, kind) in variants.items():
        mult, q1, q9, fits = _wf(X, ratio, lr, rows, train_ok, month_of, months, kind)
        app = np.isfinite(mult)
        f = np.where(app, pp * np.nan_to_num(mult, nan=1.0), pp)
        nlo = np.where(app, pp * np.exp(np.nan_to_num(q1)), ll)
        nhi = np.where(app, pp * np.exp(np.nan_to_num(q9)), hh)
        card = {"fits": fits, "rows_modelled_test": int((app & test).sum())}
        for name, m in (("event", test & ev), ("non_event", test & ~ev), ("earn_new", test & earn), ("event_not_earn", test & ev & ~earn),
                        ("gap5", test & big), ("gap5_event", test & big & ev), ("all", test)):
            card[name] = _block(yy, f, pp, nlo, nhi, ll, hh, m, tt)
        me, mn = test & ev, test & ~ev
        card["event"]["boot_r2_diff_5_50_95"] = boot_r2_diff(yy[me], f[me], pp[me], tt[me])
        card["non_event"]["boot_r2_diff_5_50_95"] = boot_r2_diff(yy[mn], f[mn], pp[mn], tt[mn]) if (f[mn] != pp[mn]).any() else [0.0, 0.0, 0.0]
        card["event_by_half"] = {}
        for h_ in sorted(set(hv[me])):
            mh = me & (hv == h_)
            card["event_by_half"][h_] = {"n": int(mh.sum()), "shipped": round(r2(yy[mh], pp[mh]), 4), "new": round(r2(yy[mh], f[mh]), 4),
                                         "cov_shipped": cover(yy[mh], ll[mh], hh[mh])["coverage"], "cov_new": cover(yy[mh], nlo[mh], nhi[mh])["coverage"]}
        e, ne = card["event"], card["non_event"]
        card["checks"] = {
            "1_r2_gain_ge_0.03_and_boot5_gt_0": bool(e["r2_diff"] >= 0.03 and e["boot_r2_diff_5_50_95"][0] > 0),
            "2_ahead_3_of_4_halves": bool(sum(x["new"] > x["shipped"] for x in card["event_by_half"].values()) >= 3),
            "3_event_coverage_0.72_0.88": bool(0.72 <= e["cov_new"]["coverage"] <= 0.88),
            "4_non_event_no_worse": bool(ne["r2_diff"] >= -0.005 and abs(ne["cov_new"]["coverage"] - ne["cov_shipped"]["coverage"]) <= 0.02)}
        card["pass"] = all(card["checks"].values())
        out[v] = card
        served[v] = (f, nlo, nhi)
        log("range1", v, json.dumps({"event": {k: e[k] for k in ("n", "r2_shipped", "r2_new", "r2_diff", "boot_r2_diff_5_50_95")},
                                     "cov_event": [e["cov_shipped"]["coverage"], e["cov_new"]["coverage"]],
                                     "gap5": [card["gap5"]["cov_shipped"]["coverage"], card["gap5"]["cov_new"]["coverage"]],
                                     "checks": card["checks"]}), flush=True)
    passing = [v for v in variants if out[v]["pass"]]
    win = max(passing, key=lambda v: out[v]["event"]["r2_new"]) if passing else None
    out["ship"] = win
    final = None
    if win:                                     # refit the shipped variant on every realized row, for live use
        X, rows, kind = variants[win]
        final = {"variant": win, "names": EVF + (HLF if win != "A" else []), "kind": kind,
                 "rows_rule": "ev_any or hl_any" if win == "B" else "ev_any" if win == "A" else "all",
                 "model": fit_residual(X[rows], ratio[rows], lr[rows], kind), "n_train": int(rows.sum()),
                 "last_target_day": str(dec[int(tt.max())])}
        if win == "C":
            final = None                        # C's live inputs (p1, ehist, log p) are not wired; never reached unless C alone passes
    grids = {}
    for v, (f, nlo, nhi) in served.items():
        for k, a in (("range1", f), ("range1_lo", nlo), ("range1_hi", nhi)):
            g = np.full((S, D), np.nan)
            g[ok] = np.where(test, a, np.nan)
            grids[f"{v}_{k}"] = g
    return out, grids, final


# ---------------------------------------------------------------- intraday (decisions 09:30, 10:00 ET)
def intraday_eval(data, ehist, pA, log=print):
    from ml_research import intraday as I
    pz = np.load(OUT / "intraday_panel.npz", allow_pickle=True)
    p = {k: pz[k] for k in pz.files}
    z = np.load(OUT / "intraday_oos.npz", allow_pickle=True)
    syms, dates = [str(s) for s in z["symbols"]], [str(x) for x in z["dates"]]
    assert syms == [str(s) for s in p["symbols"]] and dates == [str(x) for x in p["dates"]]
    S, D, NS = len(syms), len(dates), I.NS
    stocks = ~p["etf"].astype(bool)
    with np.errstate(all="ignore"):
        T = I.targets_range(p)
        g = I.gaps(p)
    dpos = {x: i for i, x in enumerate(data.dates)}
    p1 = np.full((S, D), np.nan)
    for i, x in enumerate(dates):
        t = dpos.get(x)
        if t is not None:
            for j, s in enumerate(syms):
                kk = data.k.get(s)
                if kk is not None:
                    p1[j, i] = pA["p1"][kk, t]
    eh = np.full((S, D), np.nan)
    F = filings()
    for j, s in enumerate(syms):
        if s in F:
            eh[j] = [ehist(s, x) for x in dates]
    out = {}
    months = sorted({x[:6] for x in dates if x >= INTRA_TEST_FROM})
    first_i = {m: min(i for i, x in enumerate(dates) if x[:6] == m) for m in months}
    darr = np.array(dates)
    with np.errstate(all="ignore"):
        lgap_all = np.log(np.abs(g) + 1e-4)
    for slot, (hh_, mm_) in ((0, (9, 30)), (1, (10, 0))):
        P = event_panel(syms, dates, hh_, mm_, ehist)
        for tk, key in (("rng60", "next60"), ("rngrod", "rest_of_day")):
            lp = z[f"range_{tk}_log"].reshape(S, D, NS)[:, :, slot].astype(float)
            lo = z[f"range_{tk}_lo"].reshape(S, D, NS)[:, :, slot].astype(float)
            hi = z[f"range_{tk}_hi"].reshape(S, D, NS)[:, :, slot].astype(float)
            with np.errstate(all="ignore"):
                yl = np.log(T[tk][:, :, slot])
            ok = stocks[:, None] & np.isfinite(lp) & np.isfinite(yl) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(g)
            di = np.broadcast_to(np.arange(D), (S, D))[ok]
            yy, pp, ll, hh = yl[ok], lp[ok], lo[ok], hi[ok]
            fP = {k: v[ok] for k, v in P.items()}
            ev, hl = fP["ev_any"] > 0, fP["hl_any"] > 0
            lg = lgap_all[ok]
            bg = np.abs(g[ok]) > 0.05
            base = [lg, bg.astype(float), fP["earn_new"] * lg]
            XA = _design(fP, EVF, base)
            XB = _design(fP, EVF + HLF, base)
            ssn = np.minimum(np.nan_to_num(fP["sess_since_earn"], nan=200), 200)
            XC = np.column_stack([XB, pp, eh[ok], ssn, p1[ok]])
            month_of = darr[di].astype("U6")
            train_ok = lambda m: (di <= first_i[m] - 2) & (darr[di] >= INTRA_OOS_FROM)
            lr = yy - pp
            test = darr[di] >= INTRA_TEST_FROM
            cell = f"{key}@{hh_:02d}:{mm_:02d}"
            out[cell] = {}
            allr = np.ones(len(yy), bool)
            # A_gated_posthoc was added after the pre-registered results (A-C) were seen: A fit and applied only on rows
            # with an 8-K in the window or a > 5% gap, other rows keep the shipped values. Reported, never shipped this wave.
            for v, (X, kind, rows) in {"A": (XA, "lin", allr), "B": (XB, "lin", allr), "C": (XC, "gbm", allr),
                                       "A_gated_posthoc": (XA, "lin", ev | bg)}.items():
                sh, q1, q9, fits = _wf(X, None, lr, rows, train_ok, month_of, months, kind, level=False)
                app = np.isfinite(sh)
                f = np.where(app, pp + np.nan_to_num(sh), pp)
                nlo = np.where(app, pp + np.nan_to_num(q1), ll)
                nhi = np.where(app, pp + np.nan_to_num(q9), hh)
                card = {"fits": fits}
                for name, m in (("gap5", test & bg), ("gap5_event", test & bg & ev), ("gap5_earn", test & bg & (fP["earn_new"] > 0)),
                                ("event", test & ev), ("non_event", test & ~ev), ("non_event_gap5", test & ~ev & bg), ("all", test)):
                    card[name] = _block(yy, f, pp, nlo, nhi, ll, hh, m, di, level=False)
                me = test & ev
                card["event"]["boot_r2_diff_5_50_95"] = boot_r2_diff(yy[me], f[me], pp[me], di[me])
                mn = test & ~ev
                card["non_event"]["boot_r2_diff_5_50_95"] = boot_r2_diff(yy[mn], f[mn], pp[mn], di[mn]) if (f[mn] != pp[mn]).any() else [0.0, 0.0, 0.0]
                c5, c5e = card["gap5"]["cov_new"]["coverage"], card["gap5_event"]["cov_new"]["coverage"]
                ne = card["non_event"]
                card["checks"] = {
                    "1_gap5_cov_0.70_0.90_all_and_event": bool(0.70 <= c5 <= 0.90 and 0.70 <= c5e <= 0.90),
                    "2_non_event_no_worse": bool(ne["r2_diff"] >= -0.005 and abs(ne["cov_new"]["coverage"] - ne["cov_shipped"]["coverage"]) <= 0.02),
                    "3_event_r2_not_worse": bool(card["event"]["boot_r2_diff_5_50_95"][0] > -0.005)}
                card["pass"] = all(card["checks"].values())
                out[cell][v] = card
                log("intraday", cell, v, json.dumps({
                    "gap5": [card["gap5"]["n"], card["gap5"]["cov_shipped"]["coverage"], c5],
                    "gap5_event": [card["gap5_event"]["n"], card["gap5_event"]["cov_shipped"]["coverage"], c5e],
                    "event_r2": [card["event"]["r2_shipped"], card["event"]["r2_new"], card["event"]["boot_r2_diff_5_50_95"]],
                    "non_event": [ne["r2_diff"], ne["cov_shipped"]["coverage"], ne["cov_new"]["coverage"]],
                    "checks": card["checks"]}), flush=True)
    cells = list(out)
    passing = [v for v in ("A", "B", "C") if all(out[c][v]["pass"] for c in cells)]
    win = max(passing, key=lambda v: np.mean([out[c][v]["event"]["r2_new"] for c in cells])) if passing else None
    out["ship"] = win
    return out


# ---------------------------------------------------------------- the tool: description, rows, text
TRACK_RECORD = (
    "Out of sample, walk-forward with monthly refits (today's S&P 500 members, so survivor-biased). range1_event: on "
    "sessions with an 8-K accepted since the previous close (or in the last 24 h), the shipped next-session range forecast "
    "explained R2 0.02 of the realized range and its 10-90% range held 65% of outcomes; with the event correction R2 0.51 and "
    "80% (Mar 2025-Oct 2026, 9,603 stock-sessions, ahead in all four half-years). Nearly all of the gain is earnings: on "
    "the morning after an earnings release the realized range averaged 2.1x the shipped forecast and its 10-90% range "
    "held only 27%; corrected, 80%. Other 8-K types change nothing (R2 0.33 either way). Earnings date estimate = last "
    "year's same-quarter report + 52 weeks (last + 91 days with under a year of history): the next item-2.02 filing came "
    "in the estimated session 47% of the time (3,445 reports, Jan 2025-Oct 2026), within 1 session 60%, within 3 sessions "
    "72%; misses include extra 2.02 filings such as preannouncements. No probability of reporting is given: ours was not calibrated within 5 "
    "points. Earnings drift (PEAD) was not tested: no point-in-time consensus EPS history.")

SPEC = {
    "name": "events",
    "description": (
        "Company events from SEC 8-K filings (acceptance time is when the market could know it): what each company filed "
        "in the last 7 days, latest 3 (item codes: 2.02 earnings, 1.01 material agreement, 2.01 acquisition done, 5.02 officer "
        "change, 3.02 unregistered equity sale, 7.01/8.01 press releases, ...), its last earnings report and an estimate of "
        "the next, and, when a filing or headline arrived in the last 24 h (or since the previous close, if longer), range1_event: the expected "
        "high-low range of today's session (% of the last close, with 10-90% range) corrected for that event. Use it with "
        "forecast's range1, which does not know about the event: after an earnings release prefer range1_event for stops "
        "and size. It says how far, not which way. " + TRACK_RECORD),
    "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"},
                                                                "description": "Up to 20 symbols."}},
                   "required": ["symbols"]},
}


def _norm_syms(symbols):
    if isinstance(symbols, str):
        symbols = [symbols]
    if not isinstance(symbols, (list, tuple)):
        return []
    out = []
    for s in symbols or []:
        s = str(s).upper().strip().replace("_", " ").replace("-", " ").replace(".", " ")
        if s and s not in out:
            out.append(s)
    return out[:MAX_SYMBOLS]


def _fmt_et(t):
    return datetime.fromtimestamp(int(t), ET).strftime("%Y-%m-%d %H:%M")


def _sess_label(t):
    et = datetime.fromtimestamp(int(t), ET)
    day = et.strftime("%Y-%m-%d")
    cal, early = calendar()
    hm = et.hour * 60 + et.minute
    if day not in set(cal):
        return "closed day"
    if hm < 570:
        return "before open"
    return "during session" if hm < (780 if day in early else 960) else "after close"


def _items_text(items):
    its = sorted(i for i in items if i != "9.01")
    return ", ".join(f"{i} {ITEM_LABEL.get(i, '')}".strip() for i in its) or "9.01 exhibits"


def _feature_row(rows8k, earn_t, hl, tau, ws, pc, ehist_val):
    """One symbol's model features at decision tau (window [ws, tau), previous close pc). rows8k: [(t, items)]."""
    f = {k: 0.0 for k in EVF + HLF}
    inw = [(t, i) for t, i in rows8k if ws <= t < tau]
    if inw:
        f["earn_new"] = float(any(pc <= t < tau for t in earn_t))
        f["earn_old"] = float(any(ws <= t < pc for t in earn_t))
        for g, its in GROUPS.items():
            f[g] = float(any(i & its for _, i in inw))
        f["fd"] = float(any(i & {"7.01", "8.01"} and "2.02" not in i for _, i in inw))
        f["other"] = float(any(i - KNOWN for _, i in inw))
        f["lh"] = math.log1p((tau - max(t for t, _ in inw)) / 3600.0)
        f["earn_ehist"] = f["earn_new"] * ehist_val
    hin = [cls for t, prov, cls in hl if ws <= t < tau]
    for t, prov, cls in hl:
        if not ws <= t < tau:
            continue
        for k in ("up", "down", "init", "anl_other"):
            if k in cls:
                f[k] = 1.0
        if prov != "BRFUPDN":
            f["h_earn" if "earnings" in cls else "h_other"] = 1.0
    return f, bool(inw), bool(hin)


def _range_event(final, feats, any8k, anyhl, base_pct, base_lo, base_hi):
    """The shipped variant's correction of range1 (all in %); None when it does not apply."""
    if final is None or base_pct is None:
        return None
    applies = any8k or (anyhl and final["rows_rule"] == "ev_any or hl_any")
    if not applies:
        return None
    X = np.array([[feats[n] for n in final["names"]]])
    m, q1, q9 = apply_residual(final["model"], X, level=True)
    return {"range1_event_pct": round(base_pct * float(m[0]), 2), "range1_event_lo": round(base_pct * math.exp(float(q1[0])), 2),
            "range1_event_hi": round(base_pct * math.exp(float(q9[0])), 2), "multiplier": round(float(m[0]), 2)}


def _earn_info(earn_t, earn_sess, tau):
    """Last earnings before tau and the next-date estimate (E1: last year's same-quarter report + 52 weeks), with the
    number of sessions from the decision at tau to the estimate's first reacting session."""
    T = np.array([t for t in earn_t if t < tau], np.int64)
    if not len(T):
        return {"last_earn": None, "next_earn_est": None}
    last = int(T[-1])
    est = next_pred(T, "E1")
    tc = tau_cal()
    cal = calendar()[0]
    j = int(np.searchsorted(tc, est, "left"))       # first session whose 09:00 decision follows the estimate
    est_day = datetime.fromtimestamp(est, ET).strftime("%Y-%m-%d")
    return {"last_earn": f"{_fmt_et(last)} ET ({earn_sess.get(last, _sess_label(last))})",
            "next_earn_est": f"{est_day} ({_sess_label(est)}; first reacting session {cal[j] if j < len(cal) else 'na'})",
            "next_earn_overdue": bool(est < tau), "next_earn_day": est_day, "next_earn_when": _sess_label(est),
            "next_earn_sessions": j - int(np.searchsorted(tc, tau, "left"))}


def note(row):
    """forecast's per-symbol event note, the same in live and in the simulator: the range1 correction, the latest 8-K
    of the last 7 days, and the next earnings estimate when it falls within 10 sessions (or is overdue)."""
    parts = []
    if row.get("range1_event_pct") is not None:
        parts.append(f"range1 corrected for {row.get('range1_event_reason') or 'an 8-K'} (uncorrected "     # ± half-span,
                     f"±{round(row.get('range1_shipped_pct') / 2, 2):g})")                                   # as the column
    rec = row.get("recent_8k") or []
    if rec:
        e = rec[0]
        parts.append(f"8-K {e['accepted_et'][5:]} ET ({e['when']}): {e['items']}"
                     + ("; EARNINGS since the previous close" if row.get("earnings_since_prev_close") else ""))
    n = row.get("next_earn_sessions")
    if row.get("next_earn_overdue"):
        parts.append(f"next earnings est. {row['next_earn_day']} (overdue)")
    elif n is not None and 0 <= n < 10:
        parts.append(f"next earnings est. {row['next_earn_day']} ({row['next_earn_when']})")
    return "; ".join(parts)


def _cause(earn_in_window, any8k):
    return "an earnings 8-K" if earn_in_window else "an 8-K" if any8k else "a headline"


def _load_final():
    if MODEL.exists():
        with open(MODEL, "rb") as f:
            return pickle.load(f)
    return {}


def sim_provider(data):
    """A function of decision session d -> (as_of, {symbol: row}) strictly point-in-time: filings and headlines
    accepted before 09:00 ET of session d, earnings history before then, and range1_event from the walk-forward OOS
    grid (column d - 1; from 3 Mar 2025 on, else not shown). No confirmed earnings dates (no point-in-time calendar)."""
    F, H = filings(), headlines()
    g = {}
    if OOS.exists():
        z = np.load(OOS, allow_pickle=True)
        if [str(s) for s in z["symbols"]] == data.sym and [str(x) for x in z["dates"]] == data.dates and "ship_range1" in z.files:
            g = {k: z[k] for k in ("ship_range1", "ship_range1_lo", "ship_range1_hi")}
            v2 = np.load(OUT / "vol2_oos.npz", allow_pickle=True)
            g["shipped"] = v2["range1"]
            g["rule"] = _load_final().get("range1", {}) or {}

    def at(d):
        if d < 1 or d >= len(data.dates):
            return None
        day = data.dates[d]
        tau, ws, pc = (int(x[0]) for x in windows([day], 9, 0))
        lo5 = tau - 7 * DAY
        out = {}
        for s in data.sym:
            f = F.get(s)
            if f is None:
                continue
            row = {"symbol": s}
            rec = [(t, i) for t, i, _ in f["rows"] if lo5 <= t < tau]
            row["recent_8k"] = [{"accepted_et": _fmt_et(t), "when": _sess_label(t), "items": _items_text(i),
                                 "age_h": round((tau - t) / 3600, 1)} for t, i in rec[-3:]][::-1]
            row["new_since_prev_close"] = any(pc <= t < tau for t, _ in rec)
            row["earnings_since_prev_close"] = bool(_count(f["earn"], np.array([pc]), np.array([tau]))[0])
            sess = {int(t): _sess_label(t) for t in f["earn"]}
            row.update(_earn_info(f["earn"], sess, tau))
            hl = [x for x in H.get(s, []) if ws <= x[0] < tau]
            row["analyst"] = {k: sum(k in x[3] for x in hl) for k in ("up", "down", "init")}
            j = data.k[s]
            inw = any(ws <= t < tau for t, _ in rec)
            applies = inw or (bool(hl) and g.get("rule", {}).get("rows_rule") == "ev_any or hl_any") if g else False
            if applies and np.isfinite(g["ship_range1"][j, d - 1]):
                row["range1_event_reason"] = _cause(bool(_count(f["earn"], np.array([ws]), np.array([tau]))[0]), inw)
                row["range1_shipped_pct"] = round(100 * float(g["shipped"][j, d - 1]), 2)
                row["range1_event_pct"] = round(100 * float(g["ship_range1"][j, d - 1]), 2)
                row["range1_event_lo"] = round(100 * float(g["ship_range1_lo"][j, d - 1]), 2)
                row["range1_event_hi"] = round(100 * float(g["ship_range1_hi"][j, d - 1]), 2)
            out[s] = row
        return (data.dates[d - 1], out)
    return at


# ---------------------------------------------------------------- live
def _session_now(now):
    """(session day 'YYYY-MM-DD' the next or current regular session, decision epoch for range1, prev close epoch),
    or {"error": ...} past the end of the trading calendar."""
    cal = calendar()[0]
    nt = int(now.timestamp())
    today = now.astimezone(ET).strftime("%Y-%m-%d")
    day = today if today in set(cal) and nt < close_epoch(today) else next((d for d in cal if d > today), None)
    if day is None:
        return {"error": f"no trading calendar after {cal[-1]}: extend edgar_fetch.trading_calendar()"}
    i = cal.index(day)
    pc = close_epoch(cal[i - 1])
    tau = min(nt, et_epoch(day, 9, 0)) if day == today else nt
    return day, tau, pc


def _submissions_live(cik, now, notes, sym):
    """The company's submissions JSON: at most one request per call; cached per day in data/edgar/live/<YYYYMMDD>/ and
    reused if younger than LIVE_MIN_AGE_S (so at most one request per symbol per look)."""
    from ml_research import edgar_fetch as ef
    day = now.astimezone(ET).strftime("%Y%m%d")
    p = LIVE_DIR / day / f"CIK{int(cik):010d}.json"
    if p.exists() and time.time() - p.stat().st_mtime < LIVE_MIN_AGE_S:
        return json.loads(p.read_text(encoding="utf-8")), "cache"
    t = ef.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    if t is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t, encoding="utf-8")
        return json.loads(t), "fetched"
    for q in sorted(LIVE_DIR.glob(f"*/CIK{int(cik):010d}.json"), reverse=True):
        notes.append(f"{sym}: EDGAR unreachable, using {q.parent.name}'s copy")
        return json.loads(q.read_text(encoding="utf-8")), "stale"
    notes.append(f"{sym}: EDGAR unreachable; only the archive through 2026-10-16")
    return None, "none"


def _live_headlines(gw, sym, archive):
    if gw is not None and hasattr(gw, "news"):
        try:
            rows = gw.news(sym, limit=10) or []
            out = []
            for r in rows:
                t = _epoch_utc(str(r["time"]).replace(" ", "T"))
                out.append((t, r.get("provider", ""), headline_classes(r.get("provider", ""), r.get("headline", ""))))
            return out, "gateway"
        except Exception:                                     # noqa: BLE001 - a news failure must not fail the tool
            pass
    return [(t, prov, cls) for t, prov, _, cls in archive], "archive"


def _confirmed_dates():
    """Alpha Vantage calendar snapshot (data/av/calendar/<date>.csv): symbol -> (reportDate, timeOfTheDay)."""
    import csv
    files = sorted((ROOT / "data" / "av" / "calendar").glob("*.csv"))
    if not files:
        return {}, None
    out = {}
    with open(files[-1], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["symbol"].replace("-", " ").replace(".", " "), (r["reportDate"], r.get("timeOfTheDay", "")))
    return out, files[-1].stem


_FINAL = None


def live(gw=None, symbols=None, now=None, base=None, **kw):
    """events for up to 20 symbols now: EDGAR submissions fetched live (one request per symbol per look at most),
    merged with the archive; range1_event from the shipped correction applied to forecast's range1.
    base: {symbol: {"range1_pct", "range1_lo", "range1_hi"}}, the forecast rows the caller holds (forecast.run). Without
    it (CLI use) the newest runs/ml/forecast/fc_*.json is read, and range1_event is left out unless that file is as of the
    previous session."""
    global _FINAL
    from ml_research import edgar_fetch as ef
    now = now or datetime.now(UTC)
    syms = _norm_syms(symbols)
    if not syms:
        return {"error": "give 1-20 symbols"}
    sn = _session_now(now)
    if isinstance(sn, dict):
        return sn
    day, tau_r, pc = sn
    comp = json.loads((EDGAR / "companies.json").read_text())
    F, H = filings(), headlines()
    if _FINAL is None:
        _FINAL = _load_final()
    final = _FINAL.get("range1")
    eh_tab = _FINAL.get("ehist_table", [])
    ehist = ehist_fn(eh_tab) if eh_tab else (lambda s, d: math.log(2.5))
    notes, rows = [], {}
    if base is not None:
        fc = {"as_of": "the caller's rows", "forecasts": base}
    else:
        fc = {}
        fc_files = sorted((ROOT / "runs" / "ml" / "forecast").glob("fc_*.json"))
        if fc_files:
            fc = json.loads(fc_files[-1].read_text(encoding="utf-8"))
        cal = calendar()[0]
        if _iso(fc.get("as_of") or "19000101") != cal[cal.index(day) - 1]:
            notes.append("forecast file stale")
            fc = {"as_of": fc.get("as_of"), "forecasts": {}}
    conf, conf_day = _confirmed_dates()
    nt = int(now.timestamp())
    ws_r = min(tau_r - 86400, pc)
    for s in syms:
        c = comp.get(s)
        if c is None:
            rows[s] = {"symbol": s, "note": "not covered (no SEC filer in our universe: ETFs and names outside the S&P list)"}
            continue
        arch = F.get(s, {"rows": [], "earn": np.array([], np.int64)})
        rows8k = [(t, i) for t, i, _ in arch["rows"]]
        seen = {a for _, _, a in arch["rows"]}
        earn_t = [int(t) for t in arch["earn"]]
        j, src = _submissions_live(c["cik"], now, notes, s)
        conv = c.get("raw_convention")
        if j is not None:
            rc = j.get("filings", {}).get("recent", {})
            for k in range(len(rc.get("accessionNumber", []))):
                if rc["form"][k] not in ("8-K", "8-K/A") or rc["accessionNumber"][k] in seen:
                    continue
                if rc["filingDate"][k] < "2026-09-01":
                    continue
                t = int(ef.raw_to_utc(rc["acceptanceDateTime"][k], conv or "shifted").timestamp())
                its = {x.strip() for x in (rc.get("items", [""] * (k + 1))[k] or "").split(",") if x.strip()}
                rows8k.append((t, its))
                if "2.02" in its:
                    earn_t.append(t)
            if conv is None:
                notes.append(f"{s}: acceptance-time convention unknown; new filing times may be off by the ET offset")
        rows8k.sort()
        earn_t = sorted(set(earn_t))
        hl, hsrc = _live_headlines(gw, s, H.get(s, []))
        feats, any8k, anyhl = _feature_row(rows8k, earn_t, hl, tau_r, ws_r, pc, ehist(s, day))
        row = {"symbol": s, "edgar": src, "headlines": hsrc}
        rec = [(t, i) for t, i in rows8k if nt - 7 * DAY <= t <= nt]
        row["recent_8k"] = [{"accepted_et": _fmt_et(t), "when": _sess_label(t), "items": _items_text(i),
                             "age_h": round((nt - t) / 3600, 1)} for t, i in rec[-3:]][::-1]
        row["new_since_prev_close"] = any(pc <= t <= nt for t, _ in rows8k)
        row["earnings_since_prev_close"] = any(pc <= t <= nt for t in earn_t)
        row.update(_earn_info(np.array(earn_t, np.int64), {}, nt))
        if s in conf and conf[s][0] >= now.astimezone(ET).strftime("%Y-%m-%d"):
            row["calendar"] = f"{conf[s][0]} {conf[s][1]}".strip() + f" (Alpha Vantage calendar of {conf_day}; may itself be an estimate)"
        row["analyst"] = {k: sum(k in x[2] for x in hl if ws_r <= x[0] < tau_r) for k in ("up", "down", "init")}
        fr = (fc.get("forecasts") or {}).get(s, {})
        ev = _range_event(final, feats, any8k, anyhl, fr.get("range1_pct"), fr.get("range1_lo"), fr.get("range1_hi"))
        if ev:
            row.update(ev)
            row["range1_event_reason"] = _cause(bool(feats["earn_new"] or feats["earn_old"]), any8k)
            row["range1_shipped_pct"] = fr.get("range1_pct")
        rows[s] = row
    return {"as_of": now.astimezone(ET).strftime("%Y-%m-%d %H:%M ET"), "session": day,
            "range1_event_window": f"filings and headlines from {_fmt_et(ws_r)} to {_fmt_et(tau_r)} ET",
            "forecast_as_of": fc.get("as_of"), "rows": rows, "notes": notes}


def text(out):
    """The tool's text for the model."""
    if "error" in out:
        return "events: " + out["error"]
    L = [f"events as of {out['as_of']} for session {out['session']}; range1_event uses {out['range1_event_window']} "
         f"and forecast's range1 as of the close of {out.get('forecast_as_of')}"]
    for s, r in out["rows"].items():
        if "note" in r and len(r) == 2:
            L.append(f"{s}: {r['note']}")
            continue
        parts = [s]
        if r.get("range1_event_pct") is not None:
            parts.append(f"range1_event {r['range1_event_pct']}% ({r['range1_event_lo']}-{r['range1_event_hi']}) vs forecast "
                         f"range1 {r.get('range1_shipped_pct')}% [{r.get('range1_event_reason', '')}]")
        if r.get("earnings_since_prev_close"):
            parts.append("EARNINGS released since the previous close")
        parts.append(f"last earnings {r.get('last_earn') or 'na'}")
        parts.append(f"next est. {r.get('next_earn_est') or 'na'}" + (" (overdue)" if r.get("next_earn_overdue") else ""))
        if r.get("calendar"):
            parts.append(f"calendar {r['calendar']}")
        a = r.get("analyst") or {}
        if any(a.values()):
            parts.append("analysts since prev close: " + ", ".join(f"{k} {v}" for k, v in a.items() if v))
        L.append("; ".join(parts))
        for e in r.get("recent_8k", []):
            L.append(f"    8-K {e['accepted_et']} ET ({e['when']}, {e['age_h']}h ago): {e['items']}")
    if out.get("notes"):
        L.append("notes: " + "; ".join(out["notes"]))
    return "\n".join(L)


# ---------------------------------------------------------------- validate
def validate(log=print):
    import trader
    t0 = time.time()
    data = trader.Data()
    res = {"as_of": data.dates[-1], "prereg": "runs/ml/research/events_prereg.md"}
    res["headlines"] = headline_check()
    log("headlines", json.dumps({k: res["headlines"][k] for k in ("earnings", "encoder_needed")}), flush=True)
    res["predictor"], keep = predictor_eval(log)
    for v in ("E1", "E2"):
        tr, er = cycle_errors(v)
        m = tr >= et_epoch("2025-01-02", 9, 0)
        a = np.abs(er[m])
        res["predictor"][v]["echo_date_error_sessions"] = {"cycles": int(m.sum()), "exact": round(float((a == 0).mean()), 3),
                                                          "within1": round(float((a <= 1).mean()), 3),
                                                          "within3": round(float((a <= 3).mean()), 3),
                                                          "p10_p50_p90": [float(x) for x in np.percentile(er[m], [10, 50, 90])]}
    pA = predictor_arrays(data, keep)
    table = reaction_table(data)
    eh = ehist_fn(table)
    res["range1"], grids, final = range1_eval(data, eh, pA, log)
    res["intraday"] = intraday_eval(data, eh, pA, log)
    res["pead"] = {"run": False, "reason": "no point-in-time consensus EPS history: data/av/earnings holds 2 symbols (pre-registered skip)"}
    res["ship"] = {"range1_event": res["range1"]["ship"], "intraday_event": res["intraday"]["ship"],
                   "earnings_probabilities": res["predictor"]["pass"], "earnings_date_estimate": "E1 echo date (lookup), accuracy reported",
                   "encoder": not res["headlines"]["encoder_needed"] and "not needed" or "needed"}
    npz = {"symbols": np.array(data.sym), "dates": np.array(data.dates), **{f"pred_{k}": v for k, v in pA.items()}}
    for k, v in grids.items():
        npz[k] = v.astype(np.float32)
    if res["range1"]["ship"]:
        w = res["range1"]["ship"]
        for k in ("range1", "range1_lo", "range1_hi"):
            npz[f"ship_{k}"] = grids[f"{w}_{k}"]
    np.savez_compressed(OOS, **npz)
    with open(MODEL, "wb") as f:
        pickle.dump({"range1": final, "ehist_table": table, "predictor_variant": "E1", "fit_as_of": data.dates[-1]}, f)
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "events.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    if "--live" in sys.argv:
        syms = [a for a in sys.argv[1:] if not a.startswith("--")]
        print(text(live(None, syms or ["AAPL"])))
    else:
        r = validate()
        print(json.dumps(r["ship"]))
