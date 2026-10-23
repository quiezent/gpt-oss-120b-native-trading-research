"""Tool "vol2": the shipped risk forecasts upgraded with option-implied volatility, with 10-90% intervals
(pre-registration: runs/ml/research/vol2_prereg.md; results: runs/ml/research/vol2.md and vol2.json).

Targets as in ml_research/risk.py: next-session high-low range (fraction of close), next-session |log return|, log
annualized realized vol over the next 20 sessions. New inputs: per-symbol 30-day implied vol (data/iv, IBKR daily
bars) and, for one variant, intraday realized vol from data/bars30.

    .venv\\Scripts\\python.exe ml_research\\vol2.py            # validate(): walk-forward OOS, vol2.json, vol2.md, vol2_oos.npz
    .venv\\Scripts\\python.exe ml_research\\vol2.py --live     # live() on the cache, no gateway

Features at t use data through the close of t only; targets start at t + 1.
"""

import json
import math
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml          # noqa: E402
import ml_eval     # noqa: E402
import trader      # noqa: E402
from ml_research import risk  # noqa: E402

OUT = ROOT / "runs" / "ml" / "research"
IV_DIR, B30_DIR = ROOT / "data" / "iv", ROOT / "data" / "bars30"
TEST_FROM, TEST_FROM_INTRA, RETRAIN, MIN_TRAIN, MIN_POOL = "20250303", "20260202", 21, 60, 20
TARGETS = ("range1", "absr1", "vol20")
H = {"range1": 1, "absr1": 1, "vol20": 20}
LEVEL = {"range1": True, "absr1": True, "vol20": False}
VARIANTS = ("iv", "har", "blend_ols", "blend_gbm", "blend_ols_intra")
IV_LVL = ["siv", "dsiv1", "dsiv5"]
IV_LOG = ["liv", "div1", "div5"]
INTRA_LVL = ["sintra1", "sintra5", "sintra22"]
INTRA_LOG = ["lintra1", "lintra5", "lintra22"]
QS = (0.1, 0.9)
MAX_SYMBOLS = 20


def feats(name, tgt):
    """Feature list of a variant for a target (None for the shipped HAR, which is not refit here)."""
    lvl = LEVEL[tgt]
    ivf = IV_LVL if lvl else IV_LOG
    if name == "iv":
        return ivf
    if name == "blend_ols":
        return (risk.HAR_LVL if lvl else risk.HAR_LOG) + ivf + (["sivrv", "spy_siv"] if lvl else ["ivrv", "spy_liv"])
    if name == "blend_gbm":
        return (risk.GBM[tgt] if lvl else ml.VOL_FEATS) + IV_LOG + ["ivrv", "spy_liv"]
    if name == "blend_ols_intra":
        return feats("blend_ols", tgt) + (INTRA_LVL if lvl else INTRA_LOG)
    return None


TRACK_RECORD = (
    "Out of sample (walk-forward, monthly refits, embargo = horizon; today's S&P 500 members, so survivor-biased). "
    "vol20 now comes from the stock's own 30-day option-implied vol (level, 1- and 5-day change): R2 0.62 vs 0.50 for the "
    "previous model and 0.32 for trailing 20-day vol, 3 Mar 2025-16 Sep 2026, ahead in all four half-years (0.37 vs 0.11 in "
    "the April 2025 shock half). absr1 adds implied vol and intraday realized vol from 30-minute bars: R2 0.143 vs 0.128 "
    "for the previous model, Feb-Oct 2026 only (8 months); daily move sizes remain mostly unpredictable. range1 is "
    "unchanged: no implied-vol model beat it (best 0.379 vs 0.369, not significant). Point forecasts are unbiased to within "
    "about 1% on average. The lo-hi columns are 10-90% ranges from past out-of-sample errors: range1 held 81% of outcomes "
    "(76% in the April 2025 half), absr1 80%, vol20 86% (too wide in calm periods, 87-91%; 67% in the April 2025 shock, "
    "when realized vol overshot). Rows without implied-vol or intraday history fall back to the previous models and say so.")


# ---------------------------------------------------------------- data
def _iv_file(sym):
    try:
        return {b["date"]: float(b["close"]) for b in json.loads((IV_DIR / f"{sym.replace(' ', '_')}.json").read_text(encoding="utf-8"))["bars"]
                if b.get("close") and b["close"] > 0}
    except (OSError, ValueError, KeyError):
        return {}


def load_iv(d):
    """(symbols, sessions) end-of-day implied vol (annualized fraction) aligned to d.dates; NaN where missing."""
    out = np.full(d.c.shape, np.nan)
    for j, s in enumerate(d.sym):
        iv = _iv_file(s)
        for i, x in enumerate(d.dates):
            if x in iv:
                out[j, i] = iv[x]
    return out


def load_intra(d):
    """(symbols, sessions) daily intraday realized variance from 30-minute bars (no overnight return); NaN if absent."""
    out = np.full(d.c.shape, np.nan)
    ix = {x: i for i, x in enumerate(d.dates)}
    for j, s in enumerate(d.sym):
        try:
            bars = json.loads((B30_DIR / f"{s.replace(' ', '_')}.json").read_text(encoding="utf-8"))["bars"]
        except (OSError, ValueError, KeyError):
            continue
        for day, v in rv_from_bars(bars).items():
            if day in ix:
                out[j, ix[day]] = v
    return out


def rv_from_bars(bars):
    """{YYYYMMDD: intraday realized variance} from 30-minute bars (first bar open->close, then close->close).
    Sessions with fewer than 6 usable returns are left out."""
    out = {}
    day, prev, acc, n = None, None, 0.0, 0
    for b in list(bars) + [{"date": "99999999", "open": 1, "close": 1}]:
        dt = str(b["date"]).replace("-", "")[:8]
        if dt != day:
            if day is not None and n >= 6:
                out[day] = acc
            day, acc, n, prev = dt, 0.0, 0, b.get("open")
        o, c = b.get("open"), b.get("close")
        if o and c and prev and o > 0 and c > 0 and prev > 0:
            acc += math.log(c / prev) ** 2
            n += 1
        prev = c
    return out


def iv_feats(iv, rv22_log, spy):
    """IV-derived features from an IV panel (log and level forms)."""
    with np.errstate(all="ignore"):
        liv = np.log(iv)
        siv = iv / np.sqrt(252)
        G = {"liv": liv, "div1": liv - risk.lag(liv, 1), "div5": liv - risk.lag(liv, 5), "ivrv": liv - rv22_log,
             "siv": siv, "dsiv1": siv - risk.lag(siv, 1), "dsiv5": siv - risk.lag(siv, 5),
             "sivrv": siv - np.exp(rv22_log) / np.sqrt(252)}
        G["spy_liv"] = np.broadcast_to(liv[spy], iv.shape).copy()
        G["spy_siv"] = np.broadcast_to(siv[spy], iv.shape).copy()
    return G


def intra_feats(rv):
    G = {}
    for n in (1, 5, 22):
        var = risk.rmean(rv, n)
        G[f"lintra{n}"] = risk.lvol(var)
        G[f"sintra{n}"] = np.sqrt(np.maximum(var, 0))
    return G


def panel(d, iv=None, intra=True):
    """All features M (risk.py's F, S, ml.py's V, IV and intraday), targets Y, naive baselines N."""
    F, S, Y, N, V, _ = risk.build(d)
    M = {**F, **S, **V}
    iv = load_iv(d) if iv is None else iv
    M.update(iv_feats(iv, F["lcc22"], d.k["SPY"]))
    if intra:
        M.update(intra_feats(load_intra(d)))
    return M, Y, N, iv


# ---------------------------------------------------------------- models
def _fit(kind, tgt):
    if kind == "ols":
        return (lambda Xa, ya: risk.ols_fit(Xa, risk.winsor(ya))) if LEVEL[tgt] else risk.ols_fit
    return (lambda Xa, ya: risk.gbm_fit(Xa, risk.winsor(ya))) if LEVEL[tgt] else risk.gbm_fit


def _pred(kind):
    return risk.ols_pred if kind == "ols" else risk.gbm_pred


def kind_of(name):
    return "gbm" if name == "blend_gbm" else "ols"


def first_ok(M, names, frac=0.5):
    """First session where at least `frac` of symbols have every feature in `names`."""
    ok = np.all([np.isfinite(M[k]) for k in names], axis=0).mean(0)
    return int(np.argmax(ok >= frac))


def wf(M, name, tgt, Y, t0):
    names = feats(name, tgt)
    start = first_ok(M, names)
    p = ml_eval.walk_forward(lambda idx: ml.rows(M, names, idx), Y[tgt], steps_from=start, test_from=t0, horizon=H[tgt],
                             retrain_every=RETRAIN, fit=_fit(kind_of(name), tgt), predict=_pred(kind_of(name)),
                             min_train=MIN_TRAIN)
    p[~np.isfinite(M["liv"])] = np.nan             # IV missing -> no IV forecast (the tool falls back to HAR)
    if name == "blend_ols_intra":
        p[~np.isfinite(M["lintra22"])] = np.nan
    return p, start


# ---------------------------------------------------------------- scoring
def half_of(x):
    return x[:4] + ("H1" if x[4:6] <= "06" else "H2")


def r2(y, f):
    return float(1 - np.mean((y - f) ** 2) / np.var(y))


def nw_t(x, lag):
    """Newey-West t-stat of the mean of a series (Bartlett kernel)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n, m = len(x), x.mean()
    e = x - m
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(m / math.sqrt(v / n)) if v > 0 else float("nan")


def compare(P, y, naive, steps, dates, h, bar="har"):
    """R^2 of each forecast in P (dict of (sym, days) arrays) on identical cells, NW t vs `bar`, by half-year."""
    mask = np.isfinite(y) & np.isfinite(naive)
    for p in P.values():
        mask &= np.isfinite(p)
    keep = np.zeros(mask.shape[1], bool)
    keep[steps] = True
    mask[:, ~keep] = False
    yy = y[mask]
    out = {"cells": int(mask.sum()), "sessions": int(mask.any(0).sum()),
           "first": dates[int(np.argmax(mask.any(0)))] if mask.any() else None, "r2_naive": r2(yy, naive[mask])}
    for k, p in P.items():
        out[f"r2_{k}"] = r2(yy, p[mask])
    halves = sorted({half_of(dates[t]) for t in steps if mask[:, t].any()})
    out["by_half"] = {}
    for hk in halves:
        mh = mask.copy()
        mh[:, [t for t in range(mask.shape[1]) if half_of(dates[t]) != hk]] = False
        out["by_half"][hk] = {"cells": int(mh.sum()), **{k: round(r2(y[mh], p[mh]), 4) for k, p in P.items()},
                              "naive": round(r2(y[mh], naive[mh]), 4)}
    se = {k: (p - y) ** 2 for k, p in P.items()}
    out["nw_t_vs_" + bar] = {}
    for k in P:
        if k == bar:
            continue
        dd = [np.mean(se[bar][mask[:, t], t] - se[k][mask[:, t], t]) for t in range(mask.shape[1]) if mask[:, t].sum() >= 20]
        out["nw_t_vs_" + bar][k] = round(nw_t(dd, h), 2)
        out.setdefault("halves_ahead", {})[k] = int(sum(v[k] > v[bar] for v in out["by_half"].values()))
    return out


def point_calibration(p, y, steps):
    m = np.zeros(p.shape, bool)
    m[:, steps] = True
    m &= np.isfinite(p) & np.isfinite(y)
    P, Y = p[m], y[m]
    slope = float(np.polyfit(P, Y, 1)[0])
    edges = np.quantile(P, np.linspace(0, 1, 11))
    rel = []
    for i in range(10):
        b = (P >= edges[i]) & ((P < edges[i + 1]) if i < 9 else (P <= edges[i + 1]))
        rel.append({"decile": i + 1, "pred_mean": round(float(P[b].mean()), 5), "realized_mean": round(float(Y[b].mean()), 5)})
    return {"cells": int(m.sum()), "mean_realized_over_pred": float(Y.mean() / P.mean()), "slope_y_on_pred": slope, "deciles": rel}


# ---------------------------------------------------------------- intervals
def resid(p, y, level):
    with np.errstate(all="ignore"):
        return y / p if level else y - p


def apply_q(p, q, level):
    return p * q if level else p + q


def quantiles(pp, rr, cond):
    """Interval quantiles from a residual pool: {'edges': tercile edges or None, 'q': [[q10, q90] per bucket]}."""
    if not cond:
        return {"edges": None, "q": [list(np.quantile(rr, QS))]}
    e = list(np.quantile(pp, [1 / 3, 2 / 3]))
    b = np.searchsorted(e, pp)
    return {"edges": e, "q": [list(np.quantile(rr[b == i], QS)) for i in range(3)]}


def bucket(p, edges):
    return np.zeros(np.shape(p), int) if edges is None else np.searchsorted(edges, p)


def interval_wf(p, y, h, level, cond, test_steps):
    """lo, hi (sym, days) from expanding pools of OOS residuals whose targets were realized before each session."""
    nd = p.shape[1]
    lo, hi = np.full(p.shape, np.nan), np.full(p.shape, np.nan)
    have = [t for t in range(nd) if np.isfinite(p[:, t]).any()]
    if not have:
        return lo, hi
    first = have[0]
    R = resid(p, y, level)
    cols_p, cols_r = {}, {}
    for t in range(first, nd):
        m = np.isfinite(R[:, t]) & np.isfinite(p[:, t])
        cols_p[t], cols_r[t] = p[m, t], R[m, t]
    for T in test_steps:
        last = T - 1 - h
        ts = [t for t in range(first, last + 1) if len(cols_r[t])]
        if len(ts) < MIN_POOL:
            continue
        pp, rr = np.concatenate([cols_p[t] for t in ts]), np.concatenate([cols_r[t] for t in ts])
        Q = quantiles(pp, rr, cond)
        b = bucket(p[:, T], Q["edges"])
        q = np.array(Q["q"])
        ok = np.isfinite(p[:, T])
        lo[ok, T] = apply_q(p[ok, T], q[b[ok], 0], level)
        hi[ok, T] = apply_q(p[ok, T], q[b[ok], 1], level)
    return lo, hi


def coverage(p, lo, hi, y, steps, dates):
    m = np.zeros(p.shape, bool)
    m[:, steps] = True
    m &= np.isfinite(lo) & np.isfinite(y)
    inside = (y >= lo) & (y <= hi)
    below, above = y < lo, y > hi

    def block(mm):
        n = int(mm.sum())
        return {"n": n, "coverage": round(float(inside[mm].mean()), 4), "below": round(float(below[mm].mean()), 4),
                "above": round(float(above[mm].mean()), 4),
                "median_width_rel": round(float(np.median((hi - lo)[mm] / np.abs(p[mm]))), 3)} if n else {"n": 0}
    out = {"all": block(m), "by_half": {}, "by_tercile": {}}
    for hk in sorted({half_of(dates[t]) for t in steps}):
        mh = m.copy()
        mh[:, [t for t in range(m.shape[1]) if half_of(dates[t]) != hk]] = False
        if mh.any():
            out["by_half"][hk] = block(mh)
    # terciles of the prediction within each session (cross-sectional), so they mean "low/mid/high risk names today"
    rk = np.full(p.shape, np.nan)
    for t in steps:
        ok = np.isfinite(p[:, t])
        if ok.sum() > 10:
            rk[ok, t] = p[ok, t].argsort().argsort() / (ok.sum() - 1)
    for i, name in enumerate(("low", "mid", "high")):
        out["by_tercile"][name] = block(m & (rk >= i / 3) & (rk <= (i + 1) / 3 if i == 2 else rk < (i + 1) / 3))
    errs = [abs(v["coverage"] - 0.8) for v in out["by_half"].values() if v["n"]] + \
           [abs(v["coverage"] - 0.8) for v in out["by_tercile"].values() if v["n"]]
    out["mean_abs_cov_err"] = round(float(np.mean(errs)), 4)
    return out


# ---------------------------------------------------------------- validate
def validate(reuse=False):
    """Walk-forward OOS evaluation of the 5 pre-registered variants, intervals, and the ship decision.
    reuse=True takes the variants' OOS predictions from vol2_allpreds.npz (written by an earlier full run) instead of
    refitting them; everything downstream is recomputed."""
    d = trader.Data()
    if reuse:
        z = np.load(OUT / "vol2_allpreds.npz", allow_pickle=True)
        assert [str(x) for x in z["dates"]] == d.dates
        reuse = {k: z[k] for k in z.files}
    M, Y, N, iv = panel(d)
    nd = len(d.dates)
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    t5 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM_INTRA)
    z = np.load(OUT / "risk_oos.npz", allow_pickle=True)
    assert [str(s) for s in z["symbols"]] == d.sym and [str(x) for x in z["dates"]] == d.dates
    shipped = {"range1": z["range1"], "absr1": z["absr1"], "vol20": np.log(z["vol20"])}
    res = {"as_of": d.dates[-1], "test_from": d.dates[t0], "test_from_intra": d.dates[t5], "variants": {}, "ship": {},
           "iv_coverage": {}, "train_start": {}}
    preds, npz = {}, {}
    for tgt in TARGETS:
        h, lvl = H[tgt], LEVEL[tgt]
        steps = list(range(t0, nd - h))
        P = {"har": shipped[tgt]}
        for v in ("iv", "blend_ols", "blend_gbm"):
            if reuse:
                P[v], st = reuse[f"{tgt}_{v}"], first_ok(M, feats(v, tgt))
            else:
                P[v], st = wf(M, v, tgt, Y, t0)
            res["train_start"][f"{tgt}_{v}"] = d.dates[st]
            print(tgt, v, "done", flush=True)
        card = compare(P, Y[tgt], N[tgt], steps, d.dates, h)
        # variant 5: own window, vs shipped and blend_ols on identical cells
        p5, st = (reuse[f"{tgt}_blend_ols_intra"], first_ok(M, feats("blend_ols_intra", tgt))) if reuse else             wf(M, "blend_ols_intra", tgt, Y, t5)
        res["train_start"][f"{tgt}_blend_ols_intra"] = d.dates[st]
        steps5 = list(range(t5, nd - h))
        card5 = compare({"har": P["har"], "blend_ols": P["blend_ols"], "blend_ols_intra": p5}, Y[tgt], N[tgt], steps5, d.dates, h)
        # pass rule
        passing = [v for v in ("iv", "blend_ols", "blend_gbm")
                   if card[f"r2_{v}"] > card["r2_har"] and card["nw_t_vs_har"][v] >= 2 and card["halves_ahead"][v] >= 3]
        win = max(passing, key=lambda v: card[f"r2_{v}"]) if passing else "har"
        # the winner of 1-4 may be the shipped HAR itself; variant 5 must beat whichever it is
        c5 = compare({"win": P[win], "intra": p5}, Y[tgt], N[tgt], steps5, d.dates, h, bar="win")
        card5["vs_winner"] = {"winner": win, **c5}
        if (card5["r2_blend_ols_intra"] > card5["r2_har"] and card5["nw_t_vs_har"]["blend_ols_intra"] >= 2
                and c5["r2_intra"] > c5["r2_win"] and c5["nw_t_vs_win"]["intra"] >= 2
                and c5["halves_ahead"]["intra"] >= len(c5["by_half"])):
            win = "blend_ols_intra"
        card["passing"], card["intra"] = passing, card5
        res["ship"][tgt] = win
        # composite as served: winner where it has a forecast, HAR otherwise
        pw = p5 if win == "blend_ols_intra" else P[win]
        comp = np.where(np.isfinite(pw), pw, P["har"])
        src = np.where(np.isfinite(pw) & (win != "har"), 1, np.where(np.isfinite(P["har"]), 0, -1)).astype(np.int8)
        steps_all = list(range(t0, nd - h))
        card["as_served"] = compare({"har": P["har"], "served": comp}, Y[tgt], N[tgt], steps_all, d.dates, h)
        okh = np.isfinite(P["har"][:, steps_all])
        res["iv_coverage"][tgt] = round(float((src[:, steps_all][okh] == 1).mean()), 4)
        # point calibration (OOS) of the winner and of HAR
        cal = (lambda p: point_calibration(p, Y[tgt], steps)) if lvl else             (lambda p: point_calibration(np.exp(p), np.exp(Y[tgt]), steps))       # vol20 in annualized vol, not logs
        card["calibration"] = {"har": cal(P["har"])}
        if win != "har":
            card["calibration"][win] = cal(pw)
        # intervals: I1 / I2 for the winner and for HAR (the fallback), evaluated on the test window
        all_steps = list(range(nd))
        iv_res = {}
        for who, p in (("har", P["har"]), (win, pw)) if win != "har" else (("har", P["har"]),):
            iv_res[who] = {}
            for meth, cond in (("I1", False), ("I2", True)):
                lo, hi = interval_wf(p, Y[tgt], h, lvl, cond, all_steps)
                iv_res[who][meth] = {"coverage": coverage(p, lo, hi, Y[tgt], steps, d.dates)}
                iv_res[who][meth]["_lo"], iv_res[who][meth]["_hi"] = lo, hi
            both = {m: iv_res[who][m]["coverage"]["mean_abs_cov_err"] for m in ("I1", "I2")}
            iv_res[who]["use"] = "I2" if both["I2"] < both["I1"] else "I1"
            # quantiles for live use: every OOS residual realized so far
            last = nd - 1 - h
            m = np.isfinite(p[:, :last + 1]) & np.isfinite(Y[tgt][:, :last + 1])
            iv_res[who]["live_q"] = quantiles(p[:, :last + 1][m], resid(p, Y[tgt], lvl)[:, :last + 1][m], iv_res[who]["use"] == "I2")
        card["intervals"] = {who: {k: ({"coverage": v["coverage"]} if isinstance(v, dict) and "coverage" in v else v)
                                   for k, v in r.items()} for who, r in iv_res.items()}
        # served intervals: the winner's interval where the winner served, HAR's otherwise
        uw, uh = iv_res[win]["use"], iv_res["har"]["use"]
        lo = np.where(src == 1, iv_res[win][uw]["_lo"], iv_res["har"][uh]["_lo"])
        hi = np.where(src == 1, iv_res[win][uw]["_hi"], iv_res["har"][uh]["_hi"])
        card["intervals"]["served"] = coverage(comp, lo, hi, Y[tgt], steps, d.dates)
        npz[tgt], npz[f"{tgt}_lo"], npz[f"{tgt}_hi"], npz[f"{tgt}_src"] = comp, lo, hi, src
        for v, p in P.items():
            preds[f"{tgt}_{v}"] = p
        preds[f"{tgt}_blend_ols_intra"] = p5
        res["variants"][tgt] = card
        print(tgt, json.dumps({k: v for k, v in card.items() if k.startswith("r2_") or k in ("nw_t_vs_har", "halves_ahead", "passing")}), "->", win, flush=True)
    np.savez_compressed(OUT / "vol2_oos.npz", symbols=np.array(d.sym), dates=np.array(d.dates), **npz)
    np.savez_compressed(OUT / "vol2_allpreds.npz", symbols=np.array(d.sym), dates=np.array(d.dates), **preds)
    (OUT / "vol2.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


# ---------------------------------------------------------------- training simulator
def sim_provider(data):
    """A function of decision session d -> (as_of, {symbol: row}) from the strictly walk-forward OOS forecasts and
    intervals in runs/ml/research/vol2_oos.npz at column d - 1 (the close before the decision)."""
    z = np.load(OUT / "vol2_oos.npz", allow_pickle=True)
    syms, col = [str(s) for s in z["symbols"]], {str(x): i for i, x in enumerate(z["dates"])}
    A = {k: z[k] for k in z.files}

    def at(d):
        as_of = data.dates[d - 1]
        i = col.get(as_of)
        if i is None:
            return None
        out = {}
        for j, s in enumerate(syms):
            k = data.k.get(s)
            if k is None or np.isnan(data.c[k, d - 1]) or not np.isfinite(A["range1"][j, i]):
                continue
            vals = {}
            for tgt in TARGETS:
                vals[tgt] = (A[tgt][j, i], A[f"{tgt}_lo"][j, i], A[f"{tgt}_hi"][j, i])
            out[s] = _row(data.c[k, d - 1], vals, {t: int(A[f"{t}_src"][j, i]) for t in TARGETS})
        return (as_of, out) if out else None
    return at


def _row(close, vals, src, iv=None, iv_src=None):
    """vals: {target: (point, lo, hi)} on model scale -> display row in % (vol20 exp'd and annualized)."""
    r = lambda x, k: None if x is None or not np.isfinite(x) else round(float(x), k)
    row = {"close": r(close, 2)}
    for tgt, (p, lo, hi) in vals.items():
        f = (lambda x: math.exp(x) * 100) if tgt == "vol20" else (lambda x: x * 100)
        g = lambda x: f(x) if x is not None and np.isfinite(x) else None
        k = 1 if tgt == "vol20" else 2
        row[f"{tgt}_pct"], row[f"{tgt}_lo"], row[f"{tgt}_hi"] = r(g(p), k), r(g(lo), k), r(g(hi), k)
    row["model"] = {t: ("vol2" if s == 1 else "har" if s == 0 else "none") for t, s in src.items()}
    if iv is not None:
        row["iv_pct"], row["iv_src"] = r(iv * 100, 1), iv_src
    return row


# ---------------------------------------------------------------- live
_FIT = {}
_PANEL = {}


def _info():
    return json.loads((OUT / "vol2.json").read_text(encoding="utf-8"))


def _fit_all(d, M, Y, ship):
    """Fit each shipped model (IV winner and HAR fallback) on every realized target. Cached per as-of date."""
    as_of = d.dates[-1]
    if as_of in _FIT:
        return _FIT[as_of]
    path = OUT / f"vol2_live_models_{as_of}.pkl"
    if path.exists():
        try:
            got = pickle.loads(path.read_bytes())
            if all((tgt, ship[tgt]) in got for tgt in TARGETS):      # same ship decision as when it was saved
                _FIT[as_of] = got
                return got
        except Exception:          # noqa: BLE001
            pass
    nd = len(d.dates)
    models = {}
    for tgt in TARGETS:
        h = H[tgt]
        # HAR fallback = the shipped models, refit exactly as risk.today() does
        tr = np.arange(risk.START, nd - h)
        if tgt == "vol20":
            Xa, ya = ml.rows(M, ml.VOL_FEATS, tr), Y["vol20"][:, tr].T.reshape(-1)
            ok = np.isfinite(Xa).all(1) & ~np.isnan(ya)
            models[(tgt, "har")] = ("mlpy", ml.VOL_FEATS, ml.fit(Xa[ok], ya[ok], "l2"))
        else:
            Xa, ya = ml.rows(M, risk.HAR_LVL, tr), Y[tgt][:, tr].T.reshape(-1)
            models[(tgt, "har")] = ("ols", risk.HAR_LVL, risk.ols_fit(Xa, risk.winsor(ya)))
        v = ship[tgt]
        if v != "har":
            names = feats(v, tgt)
            tr = np.arange(first_ok(M, names), nd - h)
            Xa, ya = ml.rows(M, names, tr), Y[tgt][:, tr].T.reshape(-1)
            ok = ~np.isnan(ya) & ~np.all(np.isnan(Xa), axis=1)
            models[(tgt, v)] = (kind_of(v), names, _fit(kind_of(v), tgt)(Xa[ok], ya[ok]))
    try:
        path.write_bytes(pickle.dumps(models))
    except OSError:
        pass
    _FIT[as_of] = models
    return models


def _predict(model, X):
    kind, names, m = model
    if kind == "mlpy":
        return m.predict(np.nan_to_num(X, nan=0.0))
    return _pred(kind)(m, X)


def _mini(d, sym, bars):
    """A Data-like object with SPY and one uncached symbol (daily bars from the gateway), on the cache's calendar."""
    ix = {x: i for i, x in enumerate(d.dates)}
    row = {k: np.full(len(d.dates), np.nan) for k in ("o", "h", "l", "c", "v")}
    for b in bars:
        x = str(b.get("date", "")).replace("-", "")[:8]
        if x in ix:
            for k, f in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
                try:                                      # V1: a malformed bar is skipped, not fatal
                    row[k][ix[x]] = float(b.get(f) or np.nan)
                except (TypeError, ValueError):
                    pass
    j = d.k["SPY"]
    st = lambda k: np.vstack([getattr(d, k)[j], row[k]])
    c = st("c")
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(c), axis=1, prepend=np.nan)
    return SimpleNamespace(o=st("o"), h=st("h"), l=st("l"), c=c, v=st("v"), lr=lr, dates=d.dates, k={"SPY": 0, sym: 1},
                           sym=["SPY", sym], etf=d.etf)


def _norm_iv(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not 0.02 <= x <= 5:   # V2: a fraction only (the gateway gives 0.24)
        return None
    return x


def _iv_row(ivhist_row, t, now, spy_hist, spy_now, rv22_log):
    """IV features at t for one symbol, with the decision-time IV (if any) standing in for IV_t."""
    hist = ivhist_row.copy()
    if now is not None:
        hist[t] = now
    spy = spy_hist.copy()
    if spy_now is not None:
        spy[t] = spy_now
    G = iv_feats(np.vstack([hist, spy])[:, :t + 1], np.vstack([rv22_log[:t + 1], np.zeros(t + 1)]), 1)
    return {k: float(v[0, t]) for k, v in G.items()}


LIVECACHE = OUT / "vol2_livecache.json"


def _load_lc():
    try:
        return json.loads(LIVECACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"iv": {}, "rv": {}}


def _save_lc(lc):
    try:
        LIVECACHE.write_text(json.dumps(lc), encoding="utf-8")
    except OSError:
        pass


def _align(row, hist, dates):
    """Fill NaNs of an aligned row from a {date: value} dict."""
    row = row.copy()
    for i, x in enumerate(dates):
        if not np.isfinite(row[i]) and x in hist:
            row[i] = hist[x]
    return row


def _need(row, t, idx):
    return any(not np.isfinite(row[t - k]) for k in idx)


def _iv_hist(s, row, t, d, gw, lc, notes, need=(1, 5)):
    """The symbol's end-of-day IV history aligned to d.dates: cache, then this tool's live cache, then (if a session in
    `need` is still missing: t-1 and t-5, plus t itself when no decision-time IV stands in for it) the gateway's daily
    OPTION_IMPLIED_VOLATILITY bars for the last month. Only completed sessions are kept (today's bar is partial)."""
    row = _align(row, lc["iv"].get(s, {}), d.dates)
    if gw is not None and _need(row, t, need):
        try:
            bars = gw.bars(s, duration="1 M", bar_size="1 day", what="OPTION_IMPLIED_VOLATILITY") or []
            got = {str(b["date"]).replace("-", "")[:8]: float(b["close"]) for b in bars if b.get("close") and b["close"] > 0}
            got = {k: v for k, v in got.items() if k <= d.dates[t]}
            lc["iv"].setdefault(s, {}).update(got)
            row = _align(row, got, d.dates)
            lc["dirty"] = True
        except Exception as e:     # noqa: BLE001
            notes.append(f"{s}: gateway IV history failed ({e})")
    return row


def _rv_hist(s, t, d, gw, lc, notes):
    """The symbol's daily intraday realized variance aligned to d.dates: data/bars30, this tool's live cache, then (if
    session t is missing) the gateway's 30-minute bars for the last month."""
    row = np.full(len(d.dates), np.nan)
    try:
        got = rv_from_bars(json.loads((B30_DIR / f"{s.replace(' ', '_')}.json").read_text(encoding="utf-8"))["bars"])
        row = _align(row, got, d.dates)
    except (OSError, ValueError, KeyError):
        pass
    row = _align(row, lc["rv"].get(s, {}), d.dates)
    if gw is not None and _need(row, t, (0,)):
        try:
            got = rv_from_bars(gw.bars(s, duration="1 M", bar_size="30 mins", what="TRADES") or [])
            got = {k: v for k, v in got.items() if k <= d.dates[t]}     # completed sessions only (today's is partial)
            lc["rv"].setdefault(s, {}).update(got)
            row = _align(row, got, d.dates)
            lc["dirty"] = True
        except Exception as e:     # noqa: BLE001
            notes.append(f"{s}: gateway 30-minute bars failed ({e})")
    return row


def _prev_session_et():
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:          # noqa: BLE001
        now = datetime.utcnow() - timedelta(hours=4)
    x = now.date() - timedelta(days=1)
    while x.weekday() >= 5:
        x -= timedelta(days=1)
    return x.strftime("%Y%m%d")


def live(gw=None, symbols=None, data=None, **kw):
    """Forecasts with 10-90% intervals for the session after the last cached close.
    symbols: up to 20 (None = every cached symbol, cached IV only, no gateway calls).
    gw (optional): gw.stats(symbols)[i]['implied_vol'] gives decision-time 30-day IV for requested symbols; gw.bars gives
    daily bars for a symbol outside the cache, and IV / 30-minute history the caches lack (data/iv and data/bars30 are
    not refreshed by live.py; fetched history is kept in runs/ml/research/vol2_livecache.json). Without IV (missing, or no IV history) a row falls back to the shipped HAR
    model and says so in `model` and `note`."""
    info = _info()
    ship = info["ship"]
    d = data or trader.Data()
    key = (id(d), d.dates[-1])
    if key not in _PANEL:          # features are fixed for a given cache; build once per process
        _PANEL.clear()
        _PANEL[key] = panel(d, intra=any(v == "blend_ols_intra" for v in ship.values()))
    M, Y, N, ivh = _PANEL[key]
    models = _fit_all(d, M, Y, ship)
    t = len(d.dates) - 1
    as_of = d.dates[t]
    notes = []
    if as_of < _prev_session_et() and not kw.get("no_stale_check"):
        notes.append(f"stale: the cache ends {as_of}, before the last completed session")
    if isinstance(symbols, str):                   # V2: input hardening
        symbols = [symbols]
    req = list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip()))[:MAX_SYMBOLS] if symbols else list(d.sym)
    live_iv = {}
    if gw is not None and symbols and kw.get("decision_iv", True):   # forecast's look-time rows use the close IV, as compute()
        try:
            st = gw.stats(req + (["SPY"] if "SPY" not in req else []))
            for r in st or []:
                v = _norm_iv(r.get("implied_vol")) if isinstance(r, dict) else None
                if v is not None:
                    live_iv[str(r.get("symbol", "")).upper()] = v
        except Exception as e:     # noqa: BLE001
            notes.append(f"gateway stats failed ({e}); cached end-of-day IV used where present")
    jspy = d.k["SPY"]
    spy_now = live_iv.get("SPY")
    use_intra = any(v == "blend_ols_intra" for v in ship.values())
    lc = _load_lc()
    spy_hist = _iv_hist("SPY", ivh[jspy], t, d, gw if symbols else None, lc, notes)
    rows = {}
    for s in req:
        if s in d.k:
            j = d.k[s]
            close = d.c[j, t]
            if not np.isfinite(close):
                rows[s] = {"note": "not covered (no close in the cache)"}
                continue
            base = {k: float(M[k][j, t]) for k in M}
            ivrow, rv22 = ivh[j], M["lcc22"][j]
        else:
            got, bsrc = None, None         # dividend-adjusted, as the cache: data/bars (refreshed each morning for held
            try:                           # names) when it reaches the last cached close, else the gateway
                doc = trader.barsmod.load(s)
            except (OSError, ValueError):
                doc = None
            if doc and doc.get("what") == "ADJUSTED_LAST" and any(b.get("date") == as_of for b in doc.get("bars", [])[-3:]):
                got, bsrc = doc["bars"], "data/bars"
            elif gw is not None:
                try:
                    got, bsrc = gw.bars(s, duration="1 Y", bar_size="1 day", what="ADJUSTED_LAST"), "gateway daily bars"
                except Exception as e:     # noqa: BLE001
                    notes.append(f"{s}: gateway bars failed ({e})")
            if not got:
                rows[s] = {"note": "not covered (not in the cache and no gateway bars)"}
                continue
            m = _mini(d, s, got)
            if not np.isfinite(m.c[1, t]):
                rows[s] = {"note": "not covered (gateway bars do not reach the last cached close)"}
                continue
            if np.isfinite(m.c[1, max(0, t - 59):t + 1]).sum() < 50:    # the HAR features reach back 60 sessions; with
                rows[s] = {"note": "not covered (fewer than 50 of the last 60 sessions traded)"}   # fewer, vol20's
                continue                                                 # fallback would score zero-filled features
            F, S, _, _, V, _ = risk.build(m)
            base = {k: float(v[1, t]) for k, v in {**F, **S, **V}.items()}
            for k in M:             # cross-sectional market features come from the full cache
                if k.startswith("xs_"):
                    base[k] = float(M[k][0, t])
            close = m.c[1, t]
            ivrow, rv22 = np.full(len(d.dates), np.nan), F["lcc22"][1]
            notes.append(f"{s}: outside the cache; HAR features from {bsrc} (ADJUSTED_LAST, as the cache)")
        if symbols:                 # requested symbols: complete the IV (and intraday) history the models need
            ivrow = _iv_hist(s, ivrow, t, d, gw, lc, notes, need=(1, 5) if s in live_iv else (0, 1, 5))
            if use_intra:
                g_in = intra_feats(_rv_hist(s, t, d, gw, lc, notes)[None, :])
                base.update({k: float(v[0, t]) for k, v in g_in.items()})
        now = live_iv.get(s)
        g = _iv_row(ivrow, t, now, spy_hist, spy_now, rv22)
        iv_src = "gateway (decision time)" if now is not None else ("cache (close of " + as_of + ")" if np.isfinite(ivrow[t]) else None)
        feat = {**base, **g}
        vals, src, why = {}, {}, []
        for tgt in TARGETS:
            v = ship[tgt]
            p, used = np.nan, 0
            if v != "har":
                mdl = models[(tgt, v)]
                x = np.array([[feat.get(k, np.nan) for k in mdl[1]]])
                if np.isfinite(x).all() or (mdl[0] == "gbm" and np.isfinite(feat["liv"])):
                    p, used = float(_predict(mdl, x)[0]), 1
            if not np.isfinite(p):
                mdl = models[(tgt, "har")]
                x = np.array([[feat.get(k, np.nan) for k in mdl[1]]])
                p = float(_predict(mdl, x)[0]) if (mdl[0] == "mlpy" or np.isfinite(x).all()) else np.nan
                used = 0 if np.isfinite(p) else -1
                if v != "har" and used == 0:
                    why.append(tgt)
            who = v if used == 1 else "har"
            I = info["variants"][tgt]["intervals"][who]["live_q"]
            if np.isfinite(p):
                b = int(bucket(np.array([p]), I["edges"])[0])
                q = I["q"][b]
                vals[tgt] = (p, apply_q(p, q[0], LEVEL[tgt]), apply_q(p, q[1], LEVEL[tgt]))
            else:
                vals[tgt] = (np.nan, np.nan, np.nan)
            src[tgt] = used
        row = _row(close, vals, src, now if now is not None else (ivrow[t] if np.isfinite(ivrow[t]) else np.nan), iv_src)
        note = []
        if why:
            if not np.isfinite(feat["liv"]):
                reason = "no implied vol"
            elif not all(np.isfinite(feat[k]) for k in IV_LOG):
                reason = "IV history too short"
            else:
                reason = "intraday history missing"
            note.append(f"{reason}: HAR fallback for {', '.join(why)}")
        if row.get("vol20_pct") is not None and row["vol20_pct"] < 5 and s not in trader.etfs():   # a stock's signature; bond ETFs sit there
            note.append("unusually low volatility (pending acquisition?)")
        row["note"] = "; ".join(note)
        rows[s] = row
    if lc.pop("dirty", False):
        _save_lc(lc)
    if not symbols:
        n_fb = sum(1 for r in rows.values() if "HAR fallback" in r.get("note", ""))
        if n_fb:
            notes.append(f"{n_fb} symbols fell back to HAR (IV or intraday history missing in the cache; request them "
                         "by name with a gateway to fill it)")
    return {"as_of": as_of, "ship": ship, "rows": rows, "notes": notes, "track_record": TRACK_RECORD,
            "units": "range1 = next-session high-low range, % of last close; absr1 = next-session |close-to-close move|, %; "
                     "vol20 = annualized realized vol over the next 20 sessions, %; lo/hi = 10% and 90% quantiles"}


def text(out):
    """The live() result as the compact table an LLM tool returns."""
    L = [f"as of the close of {out['as_of'][:4]}-{out['as_of'][4:6]}-{out['as_of'][6:]}; lo-hi = 10-90% range"] + out["notes"]
    L.append("sym close range1% [lo-hi] absr1% [lo-hi] vol20% [lo-hi] iv% model note")
    f = lambda x: "na" if x is None else f"{x:g}"
    for s, r in out["rows"].items():
        if "close" not in r:
            L.append(f"{s} {r['note']}")
            continue
        m = "+".join(sorted(set(r["model"].values())))
        L.append(f"{s} {f(r['close'])} " + " ".join(f"{f(r[t + '_pct'])} [{f(r[t + '_lo'])}-{f(r[t + '_hi'])}]" for t in TARGETS)
                 + f" {f(r.get('iv_pct'))} {m} {r['note']}".rstrip())
    return "\n".join(L)


if __name__ == "__main__":
    if "--live" in sys.argv:
        o = live(symbols=["SPY", "AAPL", "NVDA", "AES"], no_stale_check=True)
        print(text(o))
    else:
        validate(reuse="--reuse" in sys.argv)
