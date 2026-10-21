"""Family "risk": forecasts for sizing and stops, not direction (pre-registration: runs/ml/research/risk_prereg.md).

  (a) next-20-session realized vol (log, annualized)      vol20_har (OLS), vol20_gbm (LightGBM)
  (b) next-day |r| and next-day high-low range / close_t  day_har (OLS), day_gbm (LightGBM)
  (c) P(5-session drawdown, close_t -> lows, < -2 sigma20) tail_gbm (LightGBM binary)

    .venv\\Scripts\\python.exe ml_research\\risk.py            # walk-forward OOS, scorecards, risk_oos.npz, risk.json
    .venv\\Scripts\\python.exe ml_research\\risk.py --today    # fit on all realized targets, forecast from the last close

Features at t use bars through the close of t only. Walk-forward via ml_eval (embargo = horizon, monthly retrain).
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml          # noqa: E402
import ml_eval     # noqa: E402
import trader      # noqa: E402

OUT = ROOT / "runs" / "ml" / "research"
TEST_FROM, START, RETRAIN, MIN_TRAIN = "20250102", 61, 21, 120
LN2 = np.log(2.0)
WINS = (1, 5, 22, 60)
BLOCKS = ("cc", "pk", "gk", "rs", "on")


def lag(x, n):
    """x shifted so out[:, t] = x[:, t - n] (n > 0)."""
    out = np.full_like(x, np.nan)
    out[:, n:] = x[:, :-n]
    return out


def rmean(x, n):
    """Trailing nanmean over n sessions ending at t (inclusive); NaN until n sessions exist or if < n/2 valid."""
    if n == 1:
        return x.copy()
    ok = ~np.isnan(x)
    cs = np.cumsum(np.where(ok, x, 0.0), axis=1)
    cn = np.cumsum(ok, axis=1)
    s = cs - lag(cs, n) if n < x.shape[1] else cs
    k = cn - lag(cn, n)
    s[:, n - 1], k[:, n - 1] = cs[:, n - 1], cn[:, n - 1]
    out = np.where(k >= max(1, n // 2), s / np.maximum(k, 1), np.nan)
    out[:, :n - 1] = np.nan
    return out


def rmax(x, a, b):
    """max over sessions t-b .. t-a (a <= b, both >= 0) of x, per (symbol, t)."""
    st = np.stack([lag(x, k) if k else x for k in range(a, b + 1)])
    with np.errstate(all="ignore"):
        out = np.nanmax(st, axis=0)
    return out


def lvol(var):
    """log annualized vol from a daily variance, floored at 1% a year."""
    return np.log(np.maximum(np.sqrt(np.maximum(var, 0) * 252), 0.01))


def build(d):
    """Feature dict F (symbols, days), targets Y, naive baselines N, and ml.py's vol features (for its baseline)."""
    o, h, l, c, lr = d.o, d.h, d.l, d.c, d.lr
    X, y, V, yv, f = ml.panel(d)
    nsym, nd = c.shape
    with np.errstate(all="ignore"):
        hl, co = np.log(h / l), np.log(c / o)
        daily = {"cc": lr ** 2, "pk": hl ** 2 / (4 * LN2), "gk": 0.5 * hl ** 2 - (2 * LN2 - 1) * co ** 2,
                 "rs": np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o), "on": np.log(o / lag(c, 1)) ** 2}
        F, S = {}, {}                                    # S: daily sigma levels (for the linear level model)
        for b in BLOCKS:
            for n in WINS:
                var = rmean(daily[b], n)
                F[f"l{b}{n}"] = lvol(var)
                S[f"s{b}{n}"] = np.sqrt(np.maximum(var, 0))
        spy = d.k["SPY"]
        bc = lambda a: np.broadcast_to(a, c.shape).copy()
        for n in (5, 22):
            F[f"spy_lcc{n}"], F[f"spy_lgk{n}"] = bc(F[f"lcc{n}"][spy]), bc(F[f"lgk{n}"][spy])
            S[f"spy_scc{n}"], S[f"spy_sgk{n}"] = bc(S[f"scc{n}"][spy]), bc(S[f"sgk{n}"][spy])
        stock = np.array([s not in d.etf for s in d.sym])
        F["xs_lgk5"] = bc(np.nanmedian(F["lgk5"][stock], axis=0))
        S["xs_sgk5"] = bc(np.nanmedian(S["sgk5"][stock], axis=0))
        sig20 = f["vol20"] / np.sqrt(252)                                                   # exact: ml.py's vol20 (ddof=1)
        absr = np.abs(lr)
        F["echo63_20"] = rmax(absr, 41, 64) / sig20      # target window t+1..t+20, one quarter earlier, +-2 sessions
        F["echo126_20"] = rmax(absr, 104, 127) / sig20
        F["echo63_1"] = rmax(absr, 60, 64) / sig20       # target t+1, one quarter earlier, +-2
        F["echo126_1"] = rmax(absr, 123, 127) / sig20
        F["echo63_5"] = rmax(absr, 56, 64) / sig20       # target t+1..t+5
        F["echo126_5"] = rmax(absr, 119, 127) / sig20
        F["beta"], F["ldvol"] = f["beta"], f["dvol"]
        F["etf"] = np.array([1.0 if s in d.etf else 0.0 for s in d.sym])[:, None] * np.ones_like(c)
        F["vr5_60"] = F["lcc5"] - F["lcc60"]
        F["ret5"], F["ret20"] = f["r5"], f["r20"]
        F["off_hi60"] = f["off_hi60"]
        # targets
        Y = {"vol20": yv,
             "absr1": np.concatenate([absr[:, 1:], np.full((nsym, 1), np.nan)], axis=1),
             "range1": np.concatenate([((h - l)[:, 1:] / c[:, :-1]), np.full((nsym, 1), np.nan)], axis=1)}
        lowmin = np.full_like(c, np.nan)
        for t in range(nd - 5):
            w = l[:, t + 1:t + 6]
            lowmin[:, t] = np.where(np.isnan(w).any(1), np.nan, w.min(1))
        dd = lowmin / c - 1
        Y["tail5"] = np.where(np.isnan(dd) | np.isnan(sig20), np.nan, (dd < -2 * sig20).astype(float))
        N = {"vol20": V["lvol20"], "absr1": np.sqrt(2 / np.pi) * sig20, "range1": rmean((h - l) / c, 20)}
    return F, S, Y, N, V, sig20


HAR_LOG = [f"l{b}{n}" for b in BLOCKS for n in WINS] + ["spy_lcc5", "spy_lcc22", "spy_lgk5", "spy_lgk22", "xs_lgk5"]
HAR_LVL = [f"s{b}{n}" for b in BLOCKS for n in WINS] + ["spy_scc5", "spy_scc22", "spy_sgk5", "spy_sgk22", "xs_sgk5"]
GBM_COMMON = HAR_LOG + ["beta", "ldvol", "etf", "vr5_60", "ret5", "ret20", "off_hi60"]
GBM = {"vol20": GBM_COMMON + ["echo63_20", "echo126_20"], "absr1": GBM_COMMON + ["echo63_1", "echo126_1"],
       "range1": GBM_COMMON + ["echo63_1", "echo126_1"], "tail5": GBM_COMMON + ["echo63_5", "echo126_5"]}
HORIZON = {"vol20": 20, "absr1": 1, "range1": 1, "tail5": 5}


def ols_fit(Xa, ya):
    ok = np.isfinite(Xa).all(1) & np.isfinite(ya)
    A = np.column_stack([np.ones(ok.sum()), Xa[ok]])
    return np.linalg.lstsq(A, ya[ok], rcond=None)[0]


def ols_pred(b, Xa):
    p = b[0] + np.nan_to_num(Xa) @ b[1:]
    return np.where(np.isfinite(Xa).all(1), p, np.nan)


def winsor(ya):
    lo, hi = np.nanpercentile(ya, [0.1, 99.9])
    return np.clip(ya, lo, hi)


def gbm_fit(Xa, ya, objective="l2"):
    ok = np.isfinite(ya)
    return ml.fit(np.where(np.isinf(Xa), np.nan, Xa)[ok], ya[ok], objective)


def gbm_pred(m, Xa):
    return m.predict(np.where(np.isinf(Xa), np.nan, Xa))


def wf(M, names, target, horizon, t0, fit, predict):
    return ml_eval.walk_forward(lambda idx: ml.rows(M, names, idx), target, steps_from=START, test_from=t0,
                                horizon=horizon, retrain_every=RETRAIN, fit=fit, predict=predict, min_train=MIN_TRAIN)


def auc(p, y):
    ok = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[ok], y[ok]
    npos, nneg = y.sum(), (1 - y).sum()
    if npos < 3 or nneg < 3:
        return np.nan
    r = p.argsort().argsort() + 1.0
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def xs_corr(p, y, steps):
    """mean daily cross-sectional Spearman of forecast vs realized (how well it ranks risk across names)."""
    return float(np.nanmean([ml_eval.spearman(p[:, t], y[:, t]) for t in steps]))


def tail_card(p, base, y, heur, steps, horizon):
    P, B, Yt, H = (a[:, steps].ravel() for a in (p, base, y, heur))
    ok = ~np.isnan(P) & ~np.isnan(Yt) & ~np.isnan(B)
    P, B, Yt, H = P[ok], B[ok], Yt[ok], H[ok]
    brier, brier0 = float(np.mean((P - Yt) ** 2)), float(np.mean((B - Yt) ** 2))
    daily = np.array([auc(p[:, t], y[:, t]) for t in steps])
    per = daily[::horizon]
    per = per[~np.isnan(per)]
    edges = np.quantile(P, np.linspace(0, 1, 11))
    rel = []
    for i in range(10):
        m = (P >= edges[i]) & ((P < edges[i + 1]) if i < 9 else (P <= edges[i + 1]))
        rel.append({"bin": i + 1, "pred_mean": round(float(P[m].mean()), 4), "observed": round(float(Yt[m].mean()), 4), "n": int(m.sum())})
    return {"cells": int(ok.sum()), "base_rate_test": float(Yt.mean()), "auc_pooled": auc(P, Yt),
            "auc_daily_mean": float(np.nanmean(daily)), "auc_daily_t": float((per.mean() - 0.5) / (per.std(ddof=1) / np.sqrt(len(per)))),
            "auc_daily_periods": int(len(per)), "brier": brier, "brier_base_rate": brier0, "brier_skill": 1 - brier / brier0,
            "auc_heuristic_vr5_60_pooled": auc(H, Yt), "reliability": rel,
            "pass": bool(auc(P, Yt) > 0.5 and (per.mean() - 0.5) / (per.std(ddof=1) / np.sqrt(len(per))) >= 2 and brier < brier0)}


def run():
    d = trader.Data()
    F, S, Y, N, V, sig20 = build(d)
    nd = len(d.dates)
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    steps = {k: list(range(t0, nd - HORIZON[k])) for k in HORIZON}
    preds, cards = {}, {}

    # (a) vol20: ml.py's model re-run in this harness (baseline), then variants 1 and 2
    okfit = lambda Xa, ya: ml.fit(Xa[np.isfinite(Xa).all(1)], ya[np.isfinite(Xa).all(1)], "l2")
    preds["vol20_mlpy"] = wf(V, ml.VOL_FEATS, Y["vol20"], 20, t0, okfit, lambda m, Xa: m.predict(np.nan_to_num(Xa, nan=0.0)))
    preds["vol20_har"] = wf(F, HAR_LOG, Y["vol20"], 20, t0, ols_fit, ols_pred)
    preds["vol20_gbm"] = wf(F, GBM["vol20"], Y["vol20"], 20, t0, gbm_fit, gbm_pred)
    print("vol20 done", flush=True)
    base = preds["vol20_mlpy"]
    for k in ("vol20_mlpy", "vol20_har", "vol20_gbm"):
        c = ml_eval.r2_oos(preds[k], Y["vol20"], N["vol20"], steps["vol20"])
        c["r2_mlpy_same_cells"] = ml_eval.r2_oos(base, Y["vol20"], N["vol20"], steps["vol20"])["r2_model"]
        both = ml_eval.r2_oos(preds[k], Y["vol20"], base, steps["vol20"])
        c["beats_mlpy"] = both["pass"]
        c["xs_spearman"] = xs_corr(preds[k], Y["vol20"], steps["vol20"][::20])
        c["xs_spearman_naive"] = xs_corr(N["vol20"], Y["vol20"], steps["vol20"][::20])
        c["by_half"] = by_half(preds[k], Y["vol20"], N["vol20"], steps["vol20"], d.dates)
        cards[k] = c

    # (b) next-day |r| and range: variants 3 and 4 (level targets, winsorized in training)
    for tgt in ("absr1", "range1"):
        preds[f"{tgt}_har"] = wf(S, HAR_LVL, Y[tgt], 1, t0, lambda Xa, ya: ols_fit(Xa, winsor(ya)), ols_pred)
        preds[f"{tgt}_gbm"] = wf(F, GBM[tgt], Y[tgt], 1, t0, lambda Xa, ya: gbm_fit(Xa, winsor(ya)), gbm_pred)
        for k in (f"{tgt}_har", f"{tgt}_gbm"):
            c = ml_eval.r2_oos(preds[k], Y[tgt], N[tgt], steps[tgt])
            c["xs_spearman"] = xs_corr(preds[k], Y[tgt], steps[tgt])
            c["xs_spearman_naive"] = xs_corr(N[tgt], Y[tgt], steps[tgt])
            c["by_half"] = by_half(preds[k], Y[tgt], N[tgt], steps[tgt], d.dates)
            cards[k] = c
        print(tgt, "done", flush=True)

    # (c) tail: variant 5, plus the training base rate as the Brier reference
    preds["tail5_gbm"] = wf(F, GBM["tail5"], Y["tail5"], 5, t0, lambda Xa, ya: gbm_fit(Xa, ya, "binary"), gbm_pred)
    rate = wf(F, ["etf"], Y["tail5"], 5, t0, lambda Xa, ya: float(np.mean(ya)), lambda m, Xa: np.full(len(Xa), m))
    cards["tail5_gbm"] = tail_card(preds["tail5_gbm"], rate, Y["tail5"], F["vr5_60"], steps["tail5"], 5)
    print("tail done", flush=True)

    # post-hoc baselines (added after the first run; NOT variants, not used to pass anything): the naive forecast
    # recalibrated by a walk-forward OLS (y ~ a + b * naive), so a scale-biased naive is not a straw man
    post = {}
    for tgt in ("vol20", "absr1", "range1"):
        preds[f"{tgt}_naivecal"] = wf(N, [tgt], Y[tgt], HORIZON[tgt], t0, ols_fit, ols_pred)
        post[f"{tgt}_naive_calibrated"] = ml_eval.r2_oos(preds[f"{tgt}_naivecal"], Y[tgt], N[tgt], steps[tgt])
    post["tail5_base_rate_train"] = float(np.nanmean(rate[:, steps["tail5"]]))
    np.savez_compressed(OUT / "risk_allpreds.npz", symbols=np.array(d.sym), dates=np.array(d.dates), **preds)

    res = {"test": f"{d.dates[t0]}..{d.dates[nd - 2]} (per-target last realized date)", "train_from": d.dates[START],
           "retrain_every": RETRAIN, "min_train": MIN_TRAIN, "variants": cards, "post_hoc_baselines": post,
           "units": {"vol20": "log annualized vol (ml.py's target)", "absr1": "fraction (|log return|)",
                     "range1": "fraction of close_t", "tail5": "probability"}}
    (OUT / "risk.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    sym = np.array(d.sym)
    dates = np.array(d.dates)
    # pred = best pre-registered variant (range1_har: next-day high-low range as a fraction of close_t). Extra arrays,
    # all strictly walk-forward OOS: vol20 = ml.py's vol model re-run here (annualized; still the best vol20 forecast),
    # absr1 = absr1_har, tail5 = tail5_gbm probability (ranks well within a day, poorly calibrated; use as a rank).
    np.savez_compressed(OUT / "risk_oos.npz", pred=preds["range1_har"], symbols=sym, dates=dates, range1=preds["range1_har"],
                        absr1=preds["absr1_har"], vol20=np.exp(preds["vol20_mlpy"]), tail5=preds["tail5_gbm"])
    print(json.dumps({k: {kk: v for kk, v in c.items() if kk not in ("reliability", "by_half")} for k, c in cards.items()}, indent=1))
    return res


def by_half(p, y, n, steps, dates):
    out = {}
    for key in sorted({dates[t][:4] + ("H1" if dates[t][4:6] <= "06" else "H2") for t in steps}):
        s = [t for t in steps if dates[t][:4] + ("H1" if dates[t][4:6] <= "06" else "H2") == key]
        r = ml_eval.r2_oos(p, y, n, s)
        out[key] = {"r2_model": round(r["r2_model"], 4), "r2_naive": round(r["r2_naive"], 4)}
    return out


def today(d=None):
    """Fit the shipped models on every realized target and forecast from the latest cached close.
    vol20: ml.py's vol model (the best vol20 forecast); absr1, range1: the HAR OLS variants; tail5: tail5_gbm, reported
    as a within-day percentile rank (its probabilities are miscalibrated out of sample).
    Returns (as_of, {symbol: {vol20_pct, vol20_naive_pct, absr1_pct, range1_pct, tail5_rank_pct}})."""
    d = d or trader.Data()
    F, S, Y, N, V, sig20 = build(d)
    nd, t = len(d.dates), len(d.dates) - 1
    tr = lambda h: np.arange(START, nd - h)
    out = {}
    Xa, ya = ml.rows(V, ml.VOL_FEATS, tr(20)), Y["vol20"][:, tr(20)].T.reshape(-1)
    ok = np.isfinite(Xa).all(1) & ~np.isnan(ya)
    out["vol20"] = np.exp(ml.fit(Xa[ok], ya[ok], "l2").predict(np.nan_to_num(ml.rows(V, ml.VOL_FEATS, [t]), nan=0.0)))
    for tgt in ("absr1", "range1"):
        Xa, ya = ml.rows(S, HAR_LVL, tr(1)), Y[tgt][:, tr(1)].T.reshape(-1)
        out[tgt] = ols_pred(ols_fit(Xa, winsor(ya)), ml.rows(S, HAR_LVL, [t]))
    Xa, ya = ml.rows(F, GBM["tail5"], tr(5)), Y["tail5"][:, tr(5)].T.reshape(-1)
    ok = ~np.isnan(ya)
    pt = gbm_pred(gbm_fit(Xa[ok], ya[ok], "binary"), ml.rows(F, GBM["tail5"], [t]))
    live = ~np.isnan(d.c[:, t]) & ~np.isnan(pt)
    rank = np.full(len(pt), np.nan)
    rank[live] = 100 * pt[live].argsort().argsort() / (live.sum() - 1)
    res = {}
    for j, s in enumerate(d.sym):
        if np.isnan(d.c[j, t]):
            continue
        r = lambda x, k=2: None if np.isnan(x) else round(float(x), k)
        res[s] = {"vol20_pct": r(out["vol20"][j] * 100, 1), "vol20_naive_pct": r(np.exp(N["vol20"][j, t]) * 100, 1),
                  "absr1_pct": r(out["absr1"][j] * 100), "range1_pct": r(out["range1"][j] * 100),
                  "tail5_rank_pct": r(rank[j], 0)}
    return d.dates[t], res


if __name__ == "__main__":
    if "--today" in sys.argv:
        as_of, res = today()
        (OUT / f"risk_today_{as_of}.json").write_text(json.dumps({"as_of": as_of, "forecasts": res}, indent=1), encoding="utf-8")
        print(as_of, len(res), "symbols;", {k: res[k] for k in ("SPY", "AAPL", "NVDA") if k in res})
    else:
        run()
