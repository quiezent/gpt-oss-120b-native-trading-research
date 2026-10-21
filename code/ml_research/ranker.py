"""Family "ranker": cross-sectional relative-return ranking over 5, 10 and 20 sessions (momentum and price-based quality).

Pre-registration (written before any run): runs/ml/research/ranker_prereg.md. Five variants, each at h = 5, 10, 20:
  V1 mom12_1 rule, V2 residual momentum rule, V3 momentum/quality rank composite (no fitting),
  V4 LightGBM lambdarank (one query group per date), V5 LightGBM l2 on within-date return ranks (objective control).

Timing: every feature uses bars through the close of t; the target is log(close[t+h] / close[t]) demeaned over the
stocks of that date. Walk-forward via ml_eval (expanding, monthly retrain, embargo = h). ETFs are not ranked.

    .venv\\Scripts\\python.exe ml_research\\ranker.py            # validate(): ranker.json, ranker.md, ranker_oos.npz
    .venv\\Scripts\\python.exe ml_research\\ranker.py --live     # live() on a fake gateway built from cached bars

Interface for the tool layer:
  validate()                     run everything, write runs/ml/research/ranker.{json,md} and ranker_oos.npz
  live(gw, symbols=None, **kw)   per-symbol rank percentile, expected relative return with a 10-90% interval and
                                 P(beat the average stock), or a top/bottom list when symbols is None
  sim_provider(data)             function of decision session d -> the same outputs as of the close of d-1, from
                                 strictly out-of-sample predictions and point-in-time decile tables
  TRACK_RECORD                   one honest paragraph for the tool description
"""

import json
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml          # noqa: E402
import ml_eval     # noqa: E402
import trader      # noqa: E402
from ml_research import industry  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "runs" / "ml" / "research"
OOS = OUT / "ranker_oos.npz"
TEST_FROM = "20250102"
HORIZONS = (5, 10, 20)
STEPS_FROM, RETRAIN, MIN_TRAIN = 130, 21, 120
CALIB_SKIP = 60                       # calibration is scored after the first 60 test dates (tables need history)
FEATS = ["mom12_1", "mom6_1", "mom3_1", "resid_mom", "ind_rel_mom", "ind_mom", "hi52", "consistency", "maxdd",
         "resid_vol", "size", "rev5", "rev21", "beta"]
COMPOSITE = ["mom12_1", "resid_mom", "ind_rel_mom", "hi52", "consistency"]
VARIANTS = ["V1_mom12_1", "V2_resid_mom", "V3_composite", "V4_lgbm_rank", "V5_lgbm_reg"]
PARAMS = dict(num_leaves=7, min_data_in_leaf=200, learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)
RANK_PARAMS = dict(objective="lambdarank", label_gain=[0, 1, 2, 3, 4], lambdarank_truncation_level=500)
ROUNDS = 200

# Filled in from the validate() results (see runs/ml/research/ranker.md); the tool description quotes it verbatim.
TRACK_RECORD = (
    "Out of sample 2 Jan 2025 to Oct 2026 on today's S&P 500 stocks (survivor-biased): no ranking passed our bar (rank IC "
    "t >= 2 and positive net decile spread). The best, plain 12-1 momentum, had rank IC +0.024 (t 1.0) over 5 sessions and "
    "+0.014 (t 0.2) over 20; its top-minus-bottom decile earned +0.37% gross per 5 sessions, -0.12% net when entered at the "
    "next open. Residual momentum and a momentum/quality composite did no better, and LightGBM rankers were near zero or "
    "negative. Most of momentum's spread came from the top decile in 2025H2-2026H1, a pattern survivorship inflates. The "
    "10-90% ranges cover 79% of outcomes overall but are too wide for calm stocks and too narrow for volatile ones "
    "(86% / 70%), and P(beat) has no skill over a coin flip. Read the rank as a description of trend, not a forecast.")
NOTES = """
## Verdict
**Nothing ships as a forecast.** None of the 15 pre-registered scorecards (5 variants x 3 horizons) passes the harness bar;
the highest IC t is +1.04 (plain 12-1 momentum, 5 sessions), far below 2 (and below the 2.9 a 15-test Bonferroni reading
needs). The fitted LightGBM rankers (lambdarank and the l2 control) are the worst: IC t between -0.69 and +0.25, with a
strongly negative 2025H1 (the April 2025 shock) that later half-years do not make up.

**What the numbers do say.** Plain 12-1 momentum has positive IC in every half-year after 2025H1 at all three horizons
(+0.013 to +0.068), and its top decile beat the average stock by about +1.0% per 20 sessions gross (+0.89% net of 10 bps).
But the gain sits almost entirely in the top decile (realized relative return: decile 10 +1.56% per 20 sessions, deciles
2-8 all negative, decile 1 about zero), so the rank correlation is weak (IC t 0.2 on 22 independent periods). It is
also the pattern survivorship bias produces: stocks that ran up and were later added to the S&P 500 sit in this
universe for the whole test. With 22 non-overlapping 20-session periods and spread t 0.7, it cannot be told from noise.
Entering at the next open instead of the close removes most of the 5- and 10-session spreads and about half of the
20-session one.

**Uncertainty outputs.** The decile intervals (10-90% of realized relative returns of the same score decile, from
outcomes realized before the decision) cover 79-80% overall, which is right on target, but they are not conditional on
risk: 86-87% coverage for the calmest third of stocks and 70% for the most volatile third. The point-in-time decile means
are noisy and not monotone, and P(beat the average stock) has a Brier score slightly worse than the base rate
(skill -0.001 to -0.006). So the expected-return number and P(beat) carry no information beyond "about zero, about 50%".

**How an LLM trader should use it (if offered at all).** As a description, not a signal: "this stock is in the top decile
of 12-month trend" tells the model it is buying strength, and that such names had wider-than-average outcomes. Size and
stop distances should come from the risk forecasts (`forecast` tool), not from this interval, which ignores the stock's
own volatility. Don't use top/bottom lists as buy/sell lists: their tested edge is zero after costs at 5 sessions and not
significant at 20.
"""


# ---------------------------------------------------------------- features
def rsum(x, n, lagk=0, partial=False):
    """Sum of x over sessions t-lagk-n+1 .. t-lagk (NaN skipped) and the count of valid values, per (row, t).
    partial=False: NaN where the window starts before the first session; True: the window is truncated there."""
    nr, nd = x.shape
    ok = ~np.isnan(x)
    cs = np.concatenate([np.zeros((nr, 1)), np.cumsum(np.where(ok, x, 0.0), 1)], 1)
    cn = np.concatenate([np.zeros((nr, 1)), np.cumsum(ok, 1)], 1)
    e = np.arange(nd) - lagk
    b = e - n + 1
    valid = (e >= 0) & ((b >= 0) | partial)
    b = np.maximum(b, 0)
    s, k = np.full((nr, nd), np.nan), np.zeros((nr, nd))
    s[:, valid] = cs[:, e[valid] + 1] - cs[:, b[valid]]
    k[:, valid] = cn[:, e[valid] + 1] - cn[:, b[valid]]
    return s, k


def raw_panel(c, h, v, spy_lr, g):
    """Raw features (rows, days) from adjusted daily close/high/volume, SPY's daily log returns and group ids
    (-1 = no group). Everything at column t uses bars through the close of t only."""
    nr, nd = c.shape
    with np.errstate(all="ignore"):
        lc = np.log(c)
        lr = np.diff(lc, axis=1, prepend=np.nan)
        x = np.broadcast_to(spy_lr, lr.shape)
        pair = ~np.isnan(lr) & ~np.isnan(x)
        y_, x_ = np.where(pair, lr, np.nan), np.where(pair, x, np.nan)
        # beta over the trailing 252 sessions (at least 126 valid pairs)
        Sx, n = rsum(x_, 252, partial=True)
        Sy, _ = rsum(y_, 252, partial=True)
        Sxy, _ = rsum(x_ * y_, 252, partial=True)
        Sxx, _ = rsum(x_ * x_, 252, partial=True)
        beta = (Sxy - Sx * Sy / n) / (Sxx - Sx ** 2 / n)
        beta = np.where(n >= 126, beta, np.nan)

        def resid_stats(win, lagk, min_n):
            """Sum and std of residual returns (r - beta_t x r_SPY) over the window, with beta fixed at t."""
            sx, k = rsum(x_, win, lagk)
            sy, _ = rsum(y_, win, lagk)
            sxy, _ = rsum(x_ * y_, win, lagk)
            sxx, _ = rsum(x_ * x_, win, lagk)
            syy, _ = rsum(y_ * y_, win, lagk)
            se = sy - beta * sx
            se2 = syy - 2 * beta * sxy + beta ** 2 * sxx
            sd = np.sqrt(np.maximum(se2 - se ** 2 / k, 0) / (k - 1))
            bad = k < min_n
            return np.where(bad, np.nan, se), np.where(bad, np.nan, sd)

        se, sd = resid_stats(231, 21, 200)                    # t-251 .. t-21
        _, sd60 = resid_stats(60, 0, 40)
        F = {"mom12_1": industry.lagret(lc, 252, 21), "mom6_1": industry.lagret(lc, 126, 21),
             "mom3_1": industry.lagret(lc, 63, 21), "resid_mom": se / sd, "resid_vol": sd60 * np.sqrt(252),
             "rev5": industry.lagret(lc, 5), "rev21": industry.lagret(lc, 21), "beta": beta}
        up, k = rsum(np.where(np.isnan(lr), np.nan, (lr > 0).astype(float)), 231, 21)
        F["consistency"] = np.where(k >= 200, up / k, np.nan)
        hi = np.full((nr, nd), np.nan)
        dd = np.full((nr, nd), np.nan)
        for t in range(251, nd):
            w = c[:, t - 251:t + 1]
            hi[:, t] = np.nanmax(h[:, t - 251:t + 1], 1)
            dd[:, t] = np.nanmin(w / np.fmax.accumulate(w, axis=1) - 1, 1)
        F["hi52"], F["maxdd"] = c / hi, dd
        dv, kv = rsum(c * v, 20)
        F["size"] = np.where(kv >= 10, np.log(dv / kv), np.nan)
        stock = g >= 0
        F["ind_mom"] = industry.loo_mean(np.where(stock[:, None], F["mom12_1"], np.nan), g)
        F["ind_rel_mom"] = F["mom12_1"] - F["ind_mom"]
    return F, lc


def xrank(x, ref, extra=None):
    """Percentile rank in [-0.5, 0.5] per column among `ref` rows; `extra` rows are placed in the ref distribution
    without changing the ref ranks. Other rows NaN."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        col = x[:, t]
        m = ref & ~np.isnan(col)
        k = int(m.sum())
        if k < 20:
            continue
        out[m, t] = col[m].argsort().argsort() / (k - 1) - 0.5
        if extra is not None:
            e = extra & ~np.isnan(col)
            if e.any():
                srt = np.sort(col[m])
                pos = (np.searchsorted(srt, col[e], "left") + np.searchsorted(srt, col[e], "right")) / 2
                out[e, t] = np.clip(pos - 0.5, 0, k - 1) / (k - 1) - 0.5
    return out


def build(d, extra_bars=None):
    """Rank features R, raw features F and per-variant rule scores for the cached universe (plus optional extra
    symbols given as {symbol: {date: bar}} aligned to d.dates). Returns a dict."""
    g, _, _ = industry.groups(d)
    c, h, v = d.c, d.h, d.v
    syms = list(d.sym)
    ref = g >= 0                                              # ranked universe: the stocks
    extras = []
    if extra_bars:
        rows = {"close": [], "high": [], "volume": []}
        for s, bars in extra_bars.items():
            extras.append(s)
            for key in rows:
                rows[key].append([bars.get(dt, {}).get(key, np.nan) for dt in d.dates])
        c = np.vstack([c, np.array(rows["close"], float)])
        h = np.vstack([h, np.array(rows["high"], float)])
        v = np.vstack([v, np.array(rows["volume"], float)])
        g = np.concatenate([g, -np.ones(len(extras), int)])
        ref = np.concatenate([ref, np.zeros(len(extras), bool)])
        syms += extras
    ext = np.array([s in extras for s in syms])
    F, lc = raw_panel(c, h, v, d.lr[d.k["SPY"]], g)
    R = {k: xrank(F[k], ref, ext) for k in FEATS}
    comp = np.mean([R[k] for k in COMPOSITE], 0)               # NaN if any component is NaN
    rules = {"V1_mom12_1": np.where((ref | ext)[:, None], F["mom12_1"], np.nan),
             "V2_resid_mom": np.where((ref | ext)[:, None], F["resid_mom"], np.nan), "V3_composite": comp}
    return {"F": F, "R": R, "rules": rules, "lc": lc, "ref": ref, "ext": ext, "syms": syms, "g": g}


def targets(lc, o, h, ref):
    """h-session relative log return from the close of t, and from the open of t+1 (diagnostic), demeaned over stocks."""
    fwd = np.full_like(lc, np.nan)
    fwd[:, :-h] = lc[:, h:] - lc[:, :-h]
    fwd = np.where(ref[:, None], fwd, np.nan)
    with np.errstate(all="ignore"):
        lo = np.log(o)
    opn = np.full_like(lc, np.nan)
    opn[:, :-h] = lc[:, h:] - lo[:, 1:lc.shape[1] - h + 1]
    opn = np.where(ref[:, None], opn, np.nan)
    return fwd - np.nanmean(fwd, 0, keepdims=True), opn - np.nanmean(opn, 0, keepdims=True)


# ---------------------------------------------------------------- LightGBM rankers
def rows_with_step(R, idx):
    X = ml.rows(R, FEATS, idx)
    nsym = R[FEATS[0]].shape[0]
    return np.column_stack([X, np.repeat(np.asarray(idx, float), nsym)])


def lgb_fit(kind):
    """fit(X, y) for ml_eval.walk_forward: the last column of X is the step (date) index, used as the query group."""
    def fit(X, y):
        st, Xf = X[:, -1], X[:, :-1]
        ok = ~np.all(np.isnan(Xf), 1) & ~np.isnan(y)
        st, Xf, y = st[ok], Xf[ok], y[ok]
        _, starts, counts = np.unique(st, return_index=True, return_counts=True)
        keep = np.zeros(len(y), bool)
        lab = np.zeros(len(y))
        for s0, n in zip(starts, counts):
            if n < 20:
                continue
            r = y[s0:s0 + n].argsort().argsort() / (n - 1)
            lab[s0:s0 + n] = np.minimum((r * 5).astype(int), 4) if kind == "rank" else r - 0.5
            keep[s0:s0 + n] = True
        grp = counts[counts >= 20]
        if kind == "rank":
            ds = lgb.Dataset(Xf[keep], lab[keep].astype(int), group=grp)
            return lgb.train({**PARAMS, **RANK_PARAMS}, ds, num_boost_round=ROUNDS)
        return lgb.train({**PARAMS, "objective": "l2"}, lgb.Dataset(Xf[keep], lab[keep]), num_boost_round=ROUNDS)
    return fit


def lgb_predict(m, X):
    return m.predict(X[:, :-1])


def lgb_walk(R, y, t0, h, ref, kind):
    pred = ml_eval.walk_forward(lambda idx: rows_with_step(R, idx), y, steps_from=STEPS_FROM, test_from=t0, horizon=h,
                                retrain_every=RETRAIN, fit=lgb_fit(kind), predict=lgb_predict, min_train=MIN_TRAIN)
    return np.where(ref[:, None], pred, np.nan)


# ---------------------------------------------------------------- point-in-time decile tables
TAB = ["mean", "q10", "q90", "p_beat", "n"]


def deciles(p, ref):
    """Decile 0..9 of each ref row's score in one column (higher score = higher decile); -1 if not ranked."""
    return _pct_and_dec(p, ref)[1]


def pit_tables(pred, y, t0, h, ref):
    """tab[i, decile, :] = (mean, q10, q90, P(y > 0), n) of the realized relative returns of OOS cells (t >= t0) whose
    targets were realized by the close of i (t + h <= i). NaN until a decile has 50 cells."""
    nd = pred.shape[1]
    tab = np.full((nd, 10, len(TAB)), np.nan)
    pool = [[] for _ in range(10)]
    dec = np.full(pred.shape, -1)
    for t in range(t0, nd):
        dec[:, t] = deciles(pred[:, t], ref)
    for i in range(t0, nd):
        t = i - h
        if t >= t0:
            for j in np.where((dec[:, t] >= 0) & ~np.isnan(y[:, t]))[0]:
                pool[dec[j, t]].append(y[j, t])
        for k in range(10):
            a = np.asarray(pool[k])
            if len(a) >= 50:
                tab[i, k] = [a.mean(), np.quantile(a, 0.1), np.quantile(a, 0.9), np.mean(a > 0), len(a)]
    return tab, dec


def calibration(tab, dec, y, t0, ref, rvol=None):
    """OOS calibration of the decile tables: each cell (t >= t0 + CALIB_SKIP) is judged by the table known at the close
    of t. Coverage of the 10-90% interval, reliability per decile, Brier of P(beat) vs 0.5 and vs the pooled base rate."""
    nd = y.shape[1]
    P = {k: [] for k in ("dec", "mean", "q10", "q90", "pb", "base", "y", "rv")}
    for t in range(t0 + CALIB_SKIP, nd):
        m = (dec[:, t] >= 0) & ~np.isnan(y[:, t])
        if not m.any() or np.isnan(tab[t, 0, 0]):
            continue
        dk = dec[m, t]
        P["dec"].append(dk)
        P["y"].append(y[m, t])
        for name, col in (("mean", 0), ("q10", 1), ("q90", 2), ("pb", 3)):
            P[name].append(tab[t, dk, col])
        P["base"].append(np.full(m.sum(), np.nansum(tab[t, :, 3] * tab[t, :, 4]) / np.nansum(tab[t, :, 4])))
        P["rv"].append(rvol[m, t] if rvol is not None else np.full(m.sum(), np.nan))
    P = {k: np.concatenate(v) for k, v in P.items()}
    if not len(P["y"]):
        return None
    inside = (P["y"] >= P["q10"]) & (P["y"] <= P["q90"])
    beat = (P["y"] > 0).astype(float)
    rel = []
    for k in range(10):
        m = P["dec"] == k
        rel.append({"decile": k + 1, "pred_mean_pct": float(P["mean"][m].mean() * 100), "real_mean_pct": float(P["y"][m].mean() * 100),
                    "coverage": float(inside[m].mean()), "below_q10": float((P["y"][m] < P["q10"][m]).mean()),
                    "above_q90": float((P["y"][m] > P["q90"][m]).mean()), "pred_p_beat": float(P["pb"][m].mean()),
                    "real_p_beat": float(beat[m].mean()), "cells": int(m.sum())})
    by_vol = {}
    if rvol is not None:
        ok = ~np.isnan(P["rv"])
        qs = np.quantile(P["rv"][ok], [1 / 3, 2 / 3])
        for name, m in (("low", ok & (P["rv"] <= qs[0])), ("mid", ok & (P["rv"] > qs[0]) & (P["rv"] <= qs[1])),
                        ("high", ok & (P["rv"] > qs[1]))):
            by_vol[name] = float(inside[m].mean())
    brier = lambda p: float(np.mean((p - beat) ** 2))
    return {"cells": int(len(beat)), "coverage_10_90": float(inside.mean()), "below_q10": float((P["y"] < P["q10"]).mean()),
            "above_q90": float((P["y"] > P["q90"]).mean()), "coverage_by_resid_vol_tercile": by_vol,
            "brier": brier(P["pb"]), "brier_half": brier(np.full(len(beat), 0.5)), "brier_base_rate": brier(P["base"]),
            "brier_skill_vs_base": float(1 - brier(P["pb"]) / brier(P["base"])), "reliability": rel}


# ---------------------------------------------------------------- validate
def _fmt_score(s):
    hv = " / ".join(f"{v:+.3f}" for v in s["ic_by_half"].values())
    return (f"{s['ic']:+.4f} | {s['ic_t']:+.2f} | {s['spread_gross_pct']:+.2f} | {s['spread_t']:+.2f} | {s['spread_net_pct']:+.2f} | "
            f"{s['top_decile_net_pct']:+.2f} | {hv} | {s['periods']}")


def validate():
    t_start = time.time()
    d = trader.Data()
    B = build(d)
    ref, lc = B["ref"], B["lc"]
    nd = len(d.dates)
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    res = {"test_from": d.dates[t0], "last_date": d.dates[-1], "stocks": int(ref.sum()), "variants": {}, "best": {}}
    preds, tabs, decs = {}, {}, {}
    for h in HORIZONS:
        y, yo = targets(lc, d.o, h, ref)
        steps = list(range(t0, nd - h))
        P = {k: np.where(np.arange(nd)[None, :] >= t0, v, np.nan) for k, v in B["rules"].items()}
        P["V4_lgbm_rank"] = lgb_walk(B["R"], y, t0, h, ref, "rank")
        P["V5_lgbm_reg"] = lgb_walk(B["R"], y, t0, h, ref, "reg")
        for name in VARIANTS:
            p = np.where(ref[:, None], P[name], np.nan)
            sc = ml_eval.score(p, y, steps, h, d.dates, cost_bps=10.0)
            sc30 = ml_eval.score(p, y, steps, h, d.dates, cost_bps=30.0)
            sco = ml_eval.score(p, yo, [t for t in steps if t + 1 < nd], h, d.dates, cost_bps=10.0)
            first = next((d.dates[t] for t in steps if np.sum(~np.isnan(p[:, t])) >= 20), None)
            res["variants"][f"{name}_h{h}"] = {"variant": name, "h": h, "first_scored": first, "score": sc,
                                               "net_spread_30bps_pct": sc30["spread_net_pct"], "open_entry": sco,
                                               "ship": bool(sc["pass"] and sco["spread_net_pct"] > 0)}
            preds[(name, h)] = p
            print(f"h{h} {name}: IC {sc['ic']:+.4f} t {sc['ic_t']:+.2f} net {sc['spread_net_pct']:+.2f}% pass {sc['pass']} "
                  f"| open-entry net {sco['spread_net_pct']:+.2f}% ({time.time() - t_start:.0f}s)", flush=True)
        best = max(VARIANTS, key=lambda n: (res["variants"][f"{n}_h{h}"]["score"]["ic_t"],
                                            res["variants"][f"{n}_h{h}"]["score"]["spread_net_pct"]))
        tab, dec = pit_tables(preds[(best, h)], y, t0, h, ref)
        cal = calibration(tab, dec, y, t0, ref, rvol=B["F"]["resid_vol"])
        res["best"][str(h)] = {"variant": best, "calibration": cal}
        tabs[h], decs[h] = tab, dec
        # also the calibration of the plain 12-1 rule at this horizon (the textbook reference)
        if best != "V1_mom12_1":
            tb, dc = pit_tables(preds[("V1_mom12_1", h)], y, t0, h, ref)
            res["best"][str(h)]["calibration_V1"] = calibration(tb, dc, y, t0, ref, rvol=B["F"]["resid_vol"])
    shipped = [k for k, v in res["variants"].items() if v["ship"]]
    res["shipped"] = shipped
    # what live() and sim_provider() serve: the shipped (variant, h) with the highest IC t, else the best-IC-t one
    cand = shipped or [f"{res['best'][str(h)]['variant']}_h{h}" for h in HORIZONS]
    serve = max(cand, key=lambda k: res["variants"][k]["score"]["ic_t"])
    res["serve"] = {"key": serve, "variant": res["variants"][serve]["variant"], "h": res["variants"][serve]["h"],
                    "shipped": bool(shipped)}
    res["passed_bar"] = [k for k, v in res["variants"].items() if v["score"]["pass"]]
    res["seconds"] = round(time.time() - t_start, 1)
    np.savez_compressed(OOS, symbols=np.array(d.sym), dates=np.array(d.dates), tab_fields=np.array(TAB),
                        serve=np.array(serve), **{f"pred_h{h}": preds[(res["best"][str(h)]["variant"], h)] for h in HORIZONS},
                        **{f"tab_h{h}": tabs[h] for h in HORIZONS},
                        **{f"variant_h{h}": np.array(res["best"][str(h)]["variant"]) for h in HORIZONS})
    (OUT / "ranker.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "ranker.md").write_text(report(res), encoding="utf-8")
    print("serve", res["serve"], "shipped", shipped, f"{res['seconds']}s")
    return res


def report(res):
    L = ["# Ranker family: cross-sectional relative-return ranking (momentum, price-based quality)", "",
         f"*Written by `ml_research/ranker.py` validate(). Pre-registration (written before any run): "
         f"`runs/ml/research/ranker_prereg.md`. Scorecards: `ranker.json`. OOS predictions and point-in-time decile tables: "
         f"`ranker_oos.npz`. Test {res['test_from']} to the last realized target (bars end {res['last_date']}); "
         f"{res['stocks']} stocks.*", ""]
    if NOTES:
        L += [NOTES.strip(), ""]
    L += ["## Method",
          "- Universe: the stocks of `lists/universe523.txt` (ETFs not ranked), `trader.Data()` adjusted daily bars. "
          "Decision at the close of t; every feature uses bars through the close of t.",
          "- Target: log(close[t+h]/close[t]) demeaned across stocks (relative return), h = 5, 10, 20. Diagnostic: entry at "
          "the open of t+1 (log(close[t+h]/open[t+1]), demeaned), which is what a sleeve acting next morning gets.",
          "- Features: 12-1, 6-1, 3-1 momentum; residual momentum (sum/std of residual returns vs SPY over t-251..t-21, beta "
          "over 252 sessions); industry-relative and industry 12-1 momentum (leave-one-out IBKR groups); 52-week-high "
          "proximity; consistency (share of up days); 12-month max drawdown; 60-session residual vol; size (log dollar "
          "volume); 5- and 21-session reversal; beta. 12-month features exist from 20241014 (the cache starts 20231016).",
          "- V1-V3 are fixed rules. V4/V5 use `ml_eval.walk_forward` from session 130 (expanding, monthly retrain, embargo "
          "= h, min 120 training dates); their first fits have few dates with the 12-month features.",
          "- Scoring: `ml_eval.score`, 10 bps per round trip, its PASS bar (IC t >= 2 on non-overlapping periods, net decile "
          "spread > 0, positive IC in >= 2 of the 3 latest half-years). Ship rule (pre-registered): PASS and next-open-entry "
          "net spread > 0.", "",
          "## Results (out of sample)",
          "IC = mean daily rank IC; IC t on non-overlapping h-session periods; spreads are top-minus-bottom decile in % per "
          "period (net = both legs fully turned over each period); half-years 2025H1, 2025H2, 2026H1, 2026H2 (to mid-Oct).", "",
          "| Variant | h | IC | IC t | spread gross | spread t | spread net | top-decile net | IC by half-year | periods | "
          "net @30bps | open-entry IC t | open-entry net | pass | ship |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["variants"].items():
        s, so = v["score"], v["open_entry"]
        L.append(f"| {v['variant']} | {v['h']} | {_fmt_score(s)} | {v['net_spread_30bps_pct']:+.2f} | {so['ic_t']:+.2f} | "
                 f"{so['spread_net_pct']:+.2f} | {'yes' if s['pass'] else 'no'} | {'yes' if v['ship'] else 'no'} |")
    L += ["", f"Passed the harness bar: {', '.join(res['passed_bar']) or 'none'}. Shipped (bar + open entry): "
          f"{', '.join(res['shipped']) or 'none'}. 15 scorecards were run; a Bonferroni reading needs IC t >= 2.9.", "",
          "## Calibration of the uncertainty outputs (best variant per horizon)",
          "Each cell is judged by the decile table known at the close of t (realized OOS outcomes only), after the first "
          f"{CALIB_SKIP} test dates. Interval = 10-90% of realized relative returns in the score decile (target coverage 80%). "
          "P(beat) = share of the decile's cells with relative return > 0; Brier vs 0.5 and vs the pooled base rate.", "",
          "| h | variant | cells | coverage 10-90 | below q10 | above q90 | coverage by resid-vol tercile (low/mid/high) | "
          "Brier | Brier 0.5 | Brier base rate | skill vs base |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        b = res["best"][str(h)]
        for lab, cal in ((b["variant"], b["calibration"]), ("V1_mom12_1 (reference)", b.get("calibration_V1"))):
            if not cal:
                continue
            cv = cal["coverage_by_resid_vol_tercile"]
            L.append(f"| {h} | {lab} | {cal['cells']} | {cal['coverage_10_90']:.3f} | {cal['below_q10']:.3f} | {cal['above_q90']:.3f} | "
                     f"{cv.get('low', np.nan):.2f} / {cv.get('mid', np.nan):.2f} / {cv.get('high', np.nan):.2f} | {cal['brier']:.4f} | "
                     f"{cal['brier_half']:.4f} | {cal['brier_base_rate']:.4f} | {cal['brier_skill_vs_base']:+.4f} |")
    L += ["", "Reliability (best variant; % relative log return over h sessions):", ""]
    for h in HORIZONS:
        b = res["best"][str(h)]
        cal = b["calibration"]
        if not cal:
            continue
        L += [f"h = {h} ({b['variant']}):", "", "| decile | predicted mean | realized mean | coverage | predicted P(beat) | realized P(beat) | cells |",
              "|---|---|---|---|---|---|---|"]
        for r in cal["reliability"]:
            L.append(f"| {r['decile']} | {r['pred_mean_pct']:+.2f} | {r['real_mean_pct']:+.2f} | {r['coverage']:.2f} | "
                     f"{r['pred_p_beat']:.3f} | {r['real_p_beat']:.3f} | {r['cells']} |")
        L.append("")
    L += ["## Track record line for the tool description", "", TRACK_RECORD, "",
          "## Caveats",
          "- Survivor-biased universe (today's S&P 500 members). For momentum this bias is upward: stocks that ran up and "
          "were then added to the index are in the sample for the whole period, so past winners look better than they were "
          "in real time.",
          "- Short history: the cache starts 20231016, so 12-month features exist only from 20241014 and the test has ~22 "
          "independent 20-session periods; a real but small effect could hide in this noise.",
          "- Scanner names outside the S&P 500 (microcaps such as NBIZ, AEHL) are ranked against the S&P 500 cross-section "
          "from gateway bars; the out-of-sample record does not cover them.",
          "- Costs: the harness assumes full turnover of both legs each period (conservative for slow momentum signals) at "
          "10 bps; the sleeves' realistic cost is nearer 30 bps per round trip (column 'net @30bps').", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- serving helpers
def _row(score_pct, dec, tab, h):
    r = lambda x, k=2: None if x is None or not np.isfinite(x) else round(float(x), k)
    out = {"rank_pct": r(score_pct, 0), "decile": int(dec) + 1 if dec >= 0 else None, "horizon": h}
    if dec >= 0 and np.isfinite(tab[dec, 0]):
        out.update({"exp_rel_ret_pct": r(tab[dec, 0] * 100), "q10_pct": r(tab[dec, 1] * 100), "q90_pct": r(tab[dec, 2] * 100),
                    "p_beat": r(tab[dec, 3], 3)})
    return out


def _dec_of(pct):
    """Decile 0..9 from a percentile 0..100 (NaN -> -1); the same rule for the tables and the served rows."""
    out = np.full(len(pct), -1)
    ok = np.isfinite(pct)
    out[ok] = np.minimum(np.floor(pct[ok] / 10), 9).astype(int)
    return out


def _pct_and_dec(p, ref, ext=None):
    """Percentile 0..100 and decile of each row's score among the ref rows; extra rows placed in that distribution."""
    pct = (xrank(p[:, None], ref, ext)[:, 0] + 0.5) * 100
    return pct, _dec_of(pct)


def sim_provider(data):
    """Function of the decision session d -> (as_of, {symbol: row}) as of the close of d-1, from the strictly OOS
    predictions and point-in-time decile tables in ranker_oos.npz (served variant and horizon)."""
    z = np.load(OOS, allow_pickle=True)
    res = json.loads((OUT / "ranker.json").read_text(encoding="utf-8"))
    h = res["serve"]["h"]
    syms, dates = [str(s) for s in z["symbols"]], [str(x) for x in z["dates"]]
    col = {x: i for i, x in enumerate(dates)}
    pred, tab = z[f"pred_h{h}"], z[f"tab_h{h}"]
    ref = ~np.all(np.isnan(pred), 1)

    def at(d):
        as_of = data.dates[d - 1]
        i = col.get(as_of)
        if i is None or np.sum(~np.isnan(pred[:, i])) < 20:
            return None
        pct, dec = _pct_and_dec(pred[:, i], ref)
        out = {}
        for j, s in enumerate(syms):
            k = data.k.get(s)
            if k is None or np.isnan(pct[j]) or np.isnan(data.c[k, d - 1]):
                continue
            out[s] = _row(pct[j], dec[j], tab[i], h)
        return (as_of, out) if out else None
    return at


def _prev_session(today):
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def _day(x):
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s[:8]


def live(gw, symbols=None, n=10, data=None, **kw):
    """Ranks from the cached daily bars through the last close. symbols: list -> per-symbol rows (symbols outside the
    cache are fetched with gw.bars and placed in the cached stock cross-section); None -> top/bottom n lists.
    Returns {"as_of", "horizon", "variant", "shipped", "stale", "track_record", "rows" | "top"/"bottom"}."""
    res = json.loads((OUT / "ranker.json").read_text(encoding="utf-8"))
    sv = res["serve"]
    h, variant = sv["h"], sv["variant"]
    tab = np.load(OOS, allow_pickle=True)[f"tab_h{h}"]
    d = data or trader.Data()
    t = len(d.dates) - 1
    as_of = d.dates[t]
    syms = [str(s).upper().strip() for s in (symbols or [])]
    missing = [s for s in syms if s not in d.k]
    extra, notes = {}, {}
    for s in missing:
        try:
            bars = gw.bars(s, duration="2 Y", bar_size="1 day")
        except Exception as e:                                       # noqa: BLE001
            notes[s] = f"no bars: {str(e)[:80]}"
            continue
        bb = {_day(b["date"]): b for b in bars if _day(b["date"]) <= as_of}
        if len(bb) < 253:
            notes[s] = f"only {len(bb)} daily bars; 12-month momentum needs 253"
        extra[s] = bb
    B = build(d, extra or None)
    ref, ext = B["ref"], B["ext"]
    if variant in B["rules"]:
        p = B["rules"][variant][:, t]
    else:
        y, _ = targets(B["lc"][:len(d.sym)], d.o, h, ref[:len(d.sym)])
        tr = np.arange(STEPS_FROM, t - h + 1)                         # every realized target (t' + h <= t)
        Rsub = {k: v[:len(d.sym)] for k, v in B["R"].items()}
        X = rows_with_step(Rsub, tr)
        m = lgb_fit("rank" if variant == "V4_lgbm_rank" else "reg")(X, y[:, tr].T.reshape(-1))
        Xt = np.column_stack([np.column_stack([B["R"][k][:, t] for k in FEATS]), np.full(len(B["syms"]), float(t))])
        p = np.where(ref | ext, m.predict(Xt[:, :-1]), np.nan)
    pct, dec = _pct_and_dec(p, ref, ext)
    tb = tab[-1] if tab.shape[0] else np.full((10, len(TAB)), np.nan)
    out = {"as_of": as_of, "horizon": h, "variant": variant, "shipped": sv["shipped"],
           "stale": as_of < _prev_session(date.today()), "track_record": TRACK_RECORD}
    idx = {s: j for j, s in enumerate(B["syms"])}
    if symbols is None:
        order = [j for j in np.argsort(-np.nan_to_num(pct, nan=-1)) if ref[j] and np.isfinite(pct[j])]
        out["top"] = [{"symbol": B["syms"][j], **_row(pct[j], dec[j], tb, h)} for j in order[:n]]
        out["bottom"] = [{"symbol": B["syms"][j], **_row(pct[j], dec[j], tb, h)} for j in order[::-1][:n]]
        return out
    rows = {}
    for s in syms:
        j = idx.get(s)
        if j is None:
            rows[s] = {"note": notes.get(s, "not covered")}
            continue
        if s in d.etf:
            rows[s] = {"note": "ETF: not ranked (stocks only)"}
            continue
        r = _row(pct[j], dec[j], tb, h) if np.isfinite(pct[j]) else {"note": "insufficient history"}
        if ext[j]:
            r["note"] = (notes.get(s, "") + "; " if s in notes else "") + "outside the S&P 500 universe: the track record does not cover it"
        rows[s] = r
    out["rows"] = rows
    return out


# ---------------------------------------------------------------- fake gateway test
class FakeGW:
    """Serves cached daily bars; `alias` maps a fake symbol absent from the cache to a cached one's bars."""

    def __init__(self, d, alias=None):
        self.d, self.alias = d, alias or {}

    def bars(self, symbol, duration="1 Y", bar_size="1 day", **kw):
        s = self.alias.get(symbol, symbol)
        j = self.d.k.get(s)
        if j is None:
            raise ValueError(f"200: No security definition has been found for {symbol}")
        D = self.d
        return [{"date": D.dates[i], "open": float(D.o[j, i]), "high": float(D.h[j, i]), "low": float(D.l[j, i]),
                 "close": float(D.c[j, i]), "volume": float(D.v[j, i])} for i in range(len(D.dates)) if not np.isnan(D.c[j, i])]

    def quotes(self, symbols):
        return []

    def stats(self, symbols):
        return []


def test_live():
    d = trader.Data()
    gw = FakeGW(d, alias={"ZZFAKE": "NVDA"})
    a = live(gw, ["NVDA", "AAPL", "SPY", "ZZFAKE", "NOSUCH"], data=d)
    b = live(gw, None, n=5, data=d)
    print(json.dumps(a, indent=1)[:3000])
    print(json.dumps({k: b[k] for k in ("as_of", "horizon", "variant", "top", "bottom")}, indent=1)[:2500])
    return a, b


def write_report():
    """Rewrite ranker.md from the saved ranker.json (after editing NOTES or TRACK_RECORD; no refit)."""
    res = json.loads((OUT / "ranker.json").read_text(encoding="utf-8"))
    (OUT / "ranker.md").write_text(report(res), encoding="utf-8")


if __name__ == "__main__":
    if "--live" in sys.argv:
        test_live()
    elif "--report" in sys.argv:
        write_report()
    else:
        validate()
