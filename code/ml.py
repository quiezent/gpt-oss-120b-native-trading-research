"""Classical ML forecasts for the ML-arm sleeves (base-ml, rl1-ml, opus-ml): the same LLMs as the clean
sleeves, plus one extra research tool (`forecast`), so the comparison measures what the tool adds.

Two LightGBM models on daily bars of the universe (lists/universe523.txt):
  ret: next 5-session log return minus SPY's (cross-sectionally demeaned), from features at the close.
  vol: next 20-session realized volatility (log, annualized).

    python ml.py validate          # walk-forward, out of sample from 2025-01: rank IC, decile spreads, vs baselines
    python ml.py train             # fit on everything, save runs/ml/{ret,vol}.txt and meta.json
    python ml.py score             # score the latest close -> runs/ml/scores/<date>.json (live.py runs this after refresh)

No lookahead: features at day t use bars through t; the ret target uses t+1..t+5; a model tested from day T
is trained only on days t <= T - 6, so no training target overlaps the test period. The universe is today's
index members (survivor-biased), so backtest figures are optimistic.
"""

import argparse
import json
import sys
from pathlib import Path

import warnings

import lightgbm as lgb
import numpy as np

import trader

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "ml"
H_RET, H_VOL = 5, 20
TEST_FROM = "20250102"
RETRAIN_EVERY = 21
STOCK_FEATS = ["r1", "r5", "r20", "r60", "r120", "vol5", "vol20", "vol60", "off_hi60", "off_lo60", "ma20", "ma50",
               "range5", "gap1", "dvol", "vratio", "beta", "resid"]
MARKET_FEATS = ["spy_r5", "spy_r20", "spy_vol20"]
FEATS = [f + "_rk" for f in STOCK_FEATS] + ["vol20", "beta", "etf"] + MARKET_FEATS
PARAMS = dict(objective="huber", alpha=1.0, num_leaves=15, learning_rate=0.03, min_data_in_leaf=300, bagging_fraction=0.8,
              bagging_freq=1, feature_fraction=0.8, lambda_l2=5.0, verbose=-1, seed=7)
ROUNDS = 300
VOL_FEATS = ["lvol5", "lvol20", "lvol60", "lrange5", "abs_r1", "beta", "ldvol", "spy_lvol20", "etf"]


def fit(X, y, objective):
    return lgb.train({**PARAMS, "objective": objective}, lgb.Dataset(X, y), num_boost_round=ROUNDS)


def roll(x, n, fn):
    """fn over a trailing window of n along axis 1 (window ending at t, inclusive); NaN until full."""
    out = np.full_like(x, np.nan)
    for t in range(n - 1, x.shape[1]):
        out[:, t] = fn(x[:, t - n + 1:t + 1], axis=1)
    return out


def rank_pct(x):
    """Cross-sectional percentile rank per column (day), in [-0.5, 0.5]; NaN stays NaN."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        col = x[:, t]
        ok = ~np.isnan(col)
        if ok.sum() > 10:
            r = col[ok].argsort().argsort()
            out[ok, t] = r / (ok.sum() - 1) - 0.5
    return out


def panel(d):
    """Features and targets for every (symbol, day). Arrays are (symbols, days)."""
    c, h, l, o, v, lr = d.c, d.h, d.l, d.o, d.v, d.lr
    with np.errstate(all="ignore"):
        lc = np.log(c)
        f = {}
        for n in (1, 5, 20, 60, 120):
            f[f"r{n}"] = np.concatenate([np.full((c.shape[0], n), np.nan), lc[:, n:] - lc[:, :-n]], axis=1)
        for n in (5, 20, 60):
            f[f"vol{n}"] = roll(lr, n, lambda w, axis: np.nanstd(w, axis=axis, ddof=1)) * np.sqrt(252)
        f["off_hi60"] = c / roll(h, 60, np.nanmax) - 1
        f["off_lo60"] = c / roll(l, 60, np.nanmin) - 1
        f["ma20"] = c / roll(c, 20, np.nanmean) - 1
        f["ma50"] = c / roll(c, 50, np.nanmean) - 1
        f["range5"] = roll((h - l) / c, 5, np.nanmean)
        f["gap1"] = np.concatenate([np.full((c.shape[0], 1), np.nan), o[:, 1:] / c[:, :-1] - 1], axis=1)
        f["dvol"] = np.log(roll(c * v, 20, np.nanmean))
        f["vratio"] = roll(v, 5, np.nanmean) / roll(v, 60, np.nanmean)
        spy = d.k["SPY"]
        x = lr[spy]
        beta = np.full_like(c, np.nan)
        resid = np.full_like(c, np.nan)
        for t in range(59, c.shape[1]):
            xs, ys = x[t - 59:t + 1], lr[:, t - 59:t + 1]
            ok = ~np.isnan(ys) & ~np.isnan(xs)
            xm = np.where(ok, xs, np.nan)
            cov = np.nanmean((ys - np.nanmean(ys, 1, keepdims=True)) * (xm - np.nanmean(xm, 1, keepdims=True)), 1)
            var = np.nanvar(xm, 1)
            b = cov / var
            beta[:, t] = b
            resid[:, t] = np.nanstd(ys - b[:, None] * xm, 1) * np.sqrt(252)
        f["beta"], f["resid"] = beta, resid
        # targets
        fwd = np.concatenate([lc[:, H_RET:] - lc[:, :-H_RET], np.full((c.shape[0], H_RET), np.nan)], axis=1)
        y = fwd - fwd[spy]
        y = y - np.nanmean(y, axis=0, keepdims=True)
        fvol = np.full_like(c, np.nan)
        for t in range(c.shape[1] - H_VOL):
            fvol[:, t] = np.nanstd(lr[:, t + 1:t + 1 + H_VOL], axis=1, ddof=1) * np.sqrt(252)
    etf = np.array([1.0 if s in d.etf else 0.0 for s in d.sym])[:, None] * np.ones_like(c)
    X = {k + "_rk": rank_pct(f[k]) for k in STOCK_FEATS}
    X.update({"vol20": f["vol20"], "beta": f["beta"], "etf": etf,
              "spy_r5": np.broadcast_to(f["r5"][spy], c.shape), "spy_r20": np.broadcast_to(f["r20"][spy], c.shape),
              "spy_vol20": np.broadcast_to(f["vol20"][spy], c.shape)})
    V = {"lvol5": np.log(f["vol5"]), "lvol20": np.log(f["vol20"]), "lvol60": np.log(f["vol60"]),
         "lrange5": np.log(f["range5"]), "abs_r1": np.abs(f["r1"]), "beta": f["beta"], "ldvol": f["dvol"],
         "spy_lvol20": np.broadcast_to(np.log(f["vol20"][spy]), c.shape), "etf": etf}
    return X, y, V, np.log(fvol), f


def rows(M, names, days):
    """Stack (symbols, days) arrays into (n, features) for the given day indices."""
    return np.stack([M[k][:, days].T.reshape(-1) for k in names], axis=1)


def spearman(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 20:
        return np.nan
    ra, rb = a[ok].argsort().argsort(), b[ok].argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]


def validate(d):
    X, y, V, yv, f = panel(d)
    nsym, ndays = y.shape
    first = 130                                           # r120 and 60-session windows need history
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    last = ndays - H_RET - 1                              # last day with a realized 5-session target
    pred = np.full_like(y, np.nan)
    pvol = np.full_like(y, np.nan)
    for T in range(t0, last + 1, RETRAIN_EVERY):
        tr = np.arange(first, T - H_RET)                  # targets end at t+5 <= T-1
        Xtr, ytr = rows(X, FEATS, tr), y[:, tr].T.reshape(-1)
        ok = ~np.isnan(ytr) & ~np.isnan(Xtr).all(1)
        lo, hi = np.nanpercentile(ytr[ok], [1, 99])
        m = fit(Xtr[ok], np.clip(ytr[ok], lo, hi), "huber")
        trv = np.arange(first, T - H_VOL)
        Vtr, yvtr = rows(V, VOL_FEATS, trv), yv[:, trv].T.reshape(-1)
        okv = ~np.isnan(yvtr) & np.isfinite(Vtr).all(1)
        mv = fit(Vtr[okv], yvtr[okv], "l2")
        te = np.arange(T, min(T + RETRAIN_EVERY, last + 1))
        pred[:, te] = m.predict(rows(X, FEATS, te)).reshape(len(te), nsym).T
        pvol[:, te] = mv.predict(np.nan_to_num(rows(V, VOL_FEATS, te), nan=0.0)).reshape(len(te), nsym).T
        print(f"  trained through {d.dates[T - H_RET - 1]}, tested {d.dates[te[0]]}..{d.dates[te[-1]]}", flush=True)
    days = np.arange(t0, last + 1)
    res = {}
    for name, sig in [("model", pred), ("momentum r20", f["r20"]), ("reversal -r5", -f["r5"]), ("low vol -vol20", -f["vol20"])]:
        ic = np.array([spearman(sig[:, t], y[:, t]) for t in days])
        ic5 = ic[::H_RET]                                 # non-overlapping 5-session periods for the t-stat
        spread = []
        for t in days[::H_RET]:
            s, r = sig[:, t], y[:, t]
            ok = ~np.isnan(s) & ~np.isnan(r)
            q = np.quantile(s[ok], [0.1, 0.9])
            spread.append(np.mean(r[ok][s[ok] >= q[1]]) - np.mean(r[ok][s[ok] <= q[0]]))
        spread = np.array(spread)
        res[name] = {"ic": float(np.nanmean(ic)), "ic_t": float(np.nanmean(ic5) / (np.nanstd(ic5, ddof=1) / np.sqrt(np.sum(~np.isnan(ic5))))),
                     "ic_pos": float(np.nanmean(ic > 0)), "decile_spread_5d_pct": float(np.mean(spread) * 100),
                     "spread_t": float(np.mean(spread) / (np.std(spread, ddof=1) / np.sqrt(len(spread)))),
                     "periods": int(len(ic5))}
    # vol: out-of-sample fit of log realized vol, model vs naive (vol20 carries forward)
    ok = ~np.isnan(pvol[:, days]) & ~np.isnan(yv[:, days]) & ~np.isnan(V["lvol20"][:, days])
    tv, pm, pn = yv[:, days][ok], pvol[:, days][ok], V["lvol20"][:, days][ok]
    r2 = lambda p: float(1 - np.mean((tv - p) ** 2) / np.var(tv))
    res["vol"] = {"r2_model": r2(pm), "r2_naive_vol20": r2(pn), "corr_model": float(np.corrcoef(tv, pm)[0, 1])}
    res["test"] = f"{d.dates[t0]}..{d.dates[last]}"
    return res


def fit_final(d):
    X, y, V, yv, f = panel(d)
    first, ndays = 130, y.shape[1]
    tr = np.arange(first, ndays - H_RET)
    Xtr, ytr = rows(X, FEATS, tr), y[:, tr].T.reshape(-1)
    ok = ~np.isnan(ytr) & ~np.isnan(Xtr).all(1)
    lo, hi = np.nanpercentile(ytr[ok], [1, 99])
    m = fit(Xtr[ok], np.clip(ytr[ok], lo, hi), "huber")
    trv = np.arange(first, ndays - H_VOL)
    Vtr, yvtr = rows(V, VOL_FEATS, trv), yv[:, trv].T.reshape(-1)
    okv = ~np.isnan(yvtr) & np.isfinite(Vtr).all(1)
    mv = fit(Vtr[okv], yvtr[okv], "l2")
    OUT.mkdir(parents=True, exist_ok=True)
    m.save_model(str(OUT / "ret.txt"))
    mv.save_model(str(OUT / "vol.txt"))
    return d.dates[tr[-1]]


def score(d):
    """Predictions from the latest close for every universe symbol."""
    X, y, V, yv, f = panel(d)
    t = y.shape[1] - 1
    m, mv = lgb.Booster(model_file=str(OUT / "ret.txt")), lgb.Booster(model_file=str(OUT / "vol.txt"))
    pr = m.predict(rows(X, FEATS, [t]))
    pv = np.exp(mv.predict(np.nan_to_num(rows(V, VOL_FEATS, [t]), nan=0.0)))
    ok = ~np.isnan(d.c[:, t])
    order = pr[ok].argsort().argsort()
    pct = np.full(len(pr), np.nan)
    pct[ok] = 100 * order / (ok.sum() - 1)
    out = {s: {"ret5_pct": round(float(pr[j] * 100), 2), "rank_pct": round(float(pct[j]), 1), "vol_pct": round(float(pv[j] * 100), 1)}
           for j, s in enumerate(d.sym) if ok[j]}
    as_of = d.dates[t]
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    (OUT / "scores" / f"{as_of}.json").write_text(json.dumps({"as_of": as_of, "scores": out}), encoding="utf-8")
    return as_of, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "train", "score"])
    ap.add_argument("--h", type=int, help="return horizon in sessions (default 5)")
    ap.add_argument("--last", help="last bar date (YYYYMMDD); default all cached bars")
    a = ap.parse_args()
    if a.h:
        global H_RET
        H_RET = a.h
    d = trader.Data(last=a.last)
    if a.cmd == "validate":
        res = validate(d)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"validation_h{H_RET}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
    elif a.cmd == "train":
        print("trained through", fit_final(d))
    else:
        as_of, out = score(d)
        print(as_of, len(out), "symbols")


if __name__ == "__main__":
    main()
