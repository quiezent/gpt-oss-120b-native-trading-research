"""Factors family: factor timing and market regime (pre-registration: runs/ml/research/factors_prereg.md).

Six cross-sectional stock factors (mom20, rev5, lowvol, size, hi52, lowbeta), their daily long-short returns, and
whether factor momentum or market state predicts which factor pays over the next 5 sessions. The timing weights turn
into a per-stock score: score(i,t) = sum_k w_k(t) * z_k(i,t). Also plain SPY timing from the same state.

    .venv\\Scripts\\python.exe ml_research\\factors.py run      # all variants, walk-forward OOS, writes runs/ml/research/factors*
    .venv\\Scripts\\python.exe ml_research\\factors.py today    # today's per-stock scores from the latest cached bars

Decision at the close of t: every feature uses bars through the close of t; targets run close(t) -> close(t+5).
"""

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml_eval     # noqa: E402
import trader      # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "runs" / "ml" / "research"
FAMILY = "factors"
H = 5
TEST_FROM = "20250102"
RETRAIN = 21
MIN_TRAIN = 150
COST_BPS = 10.0
FACTORS = ["mom20", "rev5", "lowvol", "size", "hi52", "lowbeta"]
STATE = ["spy_r20", "spy_trend", "spy_vol", "breadth", "disp"]
TIMING_FEATS = ["fmom20", "fmom60"] + STATE
MKT_FEATS = ["spy_r5"] + STATE
BEST = "V3_ridge_timing"              # best of V1-V4 by the pre-registered rule (IC t); none passes the bar


def rank_pct(x, mask):
    """Cross-sectional percentile rank in [-0.5, 0.5] among rows where mask is True; NaN elsewhere."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        ok = mask & ~np.isnan(x[:, t])
        if ok.sum() > 20:
            out[ok, t] = x[ok, t].argsort().argsort() / (ok.sum() - 1) - 0.5
    return out


def roll(x, n, fn, min_n=None):
    out = np.full_like(x, np.nan)
    for t in range((min_n or n) - 1, x.shape[1]):
        out[:, t] = fn(x[:, max(0, t - n + 1):t + 1], axis=1)
    return out


def build(d):
    """Everything point-in-time at the close of each session. Returns a dict of arrays."""
    nsym, n = d.c.shape
    stock = np.array([s not in d.etf for s in d.sym])
    spy = d.k["SPY"]
    c, lr = d.c, d.lr
    lc = np.log(c)
    shift = lambda x, k: np.concatenate([np.full((x.shape[0], k), np.nan), x[:, :-k]], axis=1)
    lead = lambda x, k: np.concatenate([x[:, k:], np.full((x.shape[0], k), np.nan)], axis=1)
    # raw factor characteristics at the close of t
    r20, r5 = lc - shift(lc, 20), lc - shift(lc, 5)
    vol60 = roll(lr, 60, lambda w, axis: np.nanstd(w, axis=axis, ddof=1))
    dvol = np.log(roll(c * d.v, 20, np.nanmean))
    hi = roll(d.h, 252, np.nanmax, min_n=60)
    hi52 = c / hi
    beta = np.full_like(c, np.nan)
    x = lr[spy]
    for t in range(59, n):
        xs, ys = x[t - 59:t + 1], lr[:, t - 59:t + 1]
        ok = ~np.isnan(ys) & ~np.isnan(xs)
        xm = np.where(ok, xs, np.nan)
        ym = np.where(ok, ys, np.nan)
        cov = np.nanmean((ym - np.nanmean(ym, 1, keepdims=True)) * (xm - np.nanmean(xm, 1, keepdims=True)), 1)
        b = cov / np.nanvar(xm, 1)
        b[ok.sum(1) < 40] = np.nan
        beta[:, t] = b
    raw = {"mom20": r20, "rev5": -r5, "lowvol": -vol60, "size": -dvol, "hi52": hi52, "lowbeta": -beta}
    Z = np.stack([rank_pct(raw[k], stock) for k in FACTORS])          # (K, nsym, n)
    # factor returns: quintile long-short formed at close t
    r1 = c / shift(c, 1) - 1                                            # simple return of session t
    fwd5 = lead(c, H) / c - 1                                           # close t -> close t+5
    K = len(FACTORS)
    F = np.full((K, n), np.nan)      # daily return realized on session t, portfolio formed at t-1
    G = np.full((K, n), np.nan)      # forward 5-session return of the portfolio formed at t
    for k in range(K):
        for t in range(n):
            z = Z[k, :, t]
            ok = ~np.isnan(z)
            if ok.sum() < 50:
                continue
            lo, hi_ = np.quantile(z[ok], [0.2, 0.8])
            L, S = ok & (z >= hi_), ok & (z <= lo)
            if t + 1 < n:
                F[k, t + 1] = np.nanmean(r1[L, t + 1]) - np.nanmean(r1[S, t + 1])
            if t + H < n:
                G[k, t] = np.nanmean(fwd5[L, t]) - np.nanmean(fwd5[S, t])
    cum = lambda m: np.array([[np.nansum(F[k, t - m + 1:t + 1]) if t >= m and np.isfinite(F[k, t - m + 1:t + 1]).all()
                               else np.nan for t in range(n)] for k in range(K)])
    fmom20, fmom60 = cum(20), cum(60)
    # market state at the close of t
    ma50 = roll(c, 50, np.nanmean)
    above = np.where(np.isnan(ma50) | np.isnan(c), np.nan, (c > ma50).astype(float))
    breadth = np.nanmean(above[stock], axis=0)
    breadth[np.sum(~np.isnan(above[stock]), 0) < 100] = np.nan
    cs = np.nanstd(lr[stock], axis=0)
    disp = roll(cs[None], 20, np.nanmean)[0]
    st = {"spy_r5": r5[spy], "spy_r20": r20[spy], "spy_trend": c[spy] / ma50[spy] - 1,
          "spy_vol": roll(lr[spy][None], 20, lambda w, axis: np.nanstd(w, axis=axis, ddof=1))[0] * np.sqrt(252),
          "breadth": breadth, "disp": disp}
    # stock target (harness): 5-session log return minus SPY, demeaned across stocks; and the next-open variant
    y = lead(lc, H) - lc
    y = y - y[spy]
    y[~stock] = np.nan
    y = y - np.nanmean(y, 0, keepdims=True)
    lo1 = np.log(lead(d.o, 1))
    yo = lead(lc, H) - lo1
    yo = yo - yo[spy]
    yo[~stock] = np.nan
    yo = yo - np.nanmean(yo, 0, keepdims=True)
    spy_fwd = (lead(lc, H) - lc)[spy]
    return {"Z": Z, "F": F, "G": G, "fmom20": fmom20, "fmom60": fmom60, "state": st, "y": y, "y_open": yo,
            "spy_fwd": spy_fwd, "stock": stock, "n": n, "nsym": nsym}


def stock_score(W, Z):
    """W (K, n) factor weights, Z (K, nsym, n) exposures -> (nsym, n)."""
    s = np.einsum("kt,kit->it", np.nan_to_num(W), np.nan_to_num(Z))
    s[np.isnan(Z).any(0) | np.isnan(W).any(0)[None, :]] = np.nan
    return s


def timing_features(B):
    """features(idx) for walk_forward over the K factors as 'symbols': rows step-major, factor-minor."""
    K = len(FACTORS)

    def f(idx):
        rows = []
        for t in idx:
            for k in range(K):
                rows.append([k, B["fmom20"][k, t], B["fmom60"][k, t]] + [B["state"][s][t] for s in STATE])
        return np.array(rows, float)
    return f


def ridge_fit(X, y, lam=0.5):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    lo, hi = np.percentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    ym = y.mean()
    b = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(X.shape[1]), Xs.T @ (y - ym))
    return mu, sd, ym, b


def ridge_predict(m, X):
    mu, sd, ym, b = m
    return ym + ((X - mu) / sd) @ b


def per_factor_ridge_fit(X, y):
    return {k: ridge_fit(X[X[:, 0] == k, 1:], y[X[:, 0] == k]) for k in range(len(FACTORS))}


def per_factor_ridge_predict(m, X):
    out = np.full(len(X), np.nan)
    for k, mk in m.items():
        r = X[:, 0] == k
        out[r] = ridge_predict(mk, X[r, 1:])
    return out


LGB_PARAMS = dict(objective="l2", num_leaves=4, min_data_in_leaf=40, learning_rate=0.05, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)


def lgb_fit(X, y):
    ok = np.isfinite(y)
    lo, hi = np.percentile(y[ok], [1, 99])
    ds = lgb.Dataset(X[ok], np.clip(y[ok], lo, hi), categorical_feature=[0], free_raw_data=False)
    return lgb.train(LGB_PARAMS, ds, num_boost_round=150)


def factor_level(W, G, steps):
    """Non-overlapping 5-session periods: timed sign(w)-weighted factor mix vs static equal mix."""
    per = [t for t in steps[::H] if np.isfinite(G[:, t]).all() and np.isfinite(W[:, t]).all()]
    timed = np.array([np.mean(np.sign(W[:, t]) * G[:, t]) for t in per])
    static = np.array([np.mean(G[:, t]) for t in per])
    diff = timed - static
    t_ = lambda x: float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))
    hit = float(np.mean([np.mean(np.sign(W[:, t]) == np.sign(G[:, t])) for t in per]))
    return {"periods": len(per), "timed_mean_pct": float(timed.mean() * 100), "timed_t": t_(timed),
            "static_mean_pct": float(static.mean() * 100), "static_t": t_(static),
            "timed_minus_static_pct": float(diff.mean() * 100), "timed_minus_static_t": t_(diff), "sign_hit_rate": hit}


def run():
    d = trader.Data()
    B = build(d)
    n, K = B["n"], len(FACTORS)
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    steps = list(range(t0, n - H))
    test = np.zeros(n, bool)
    test[t0:] = True
    first = int(np.argmax(np.isfinite(B["fmom60"]).all(0) & np.isfinite(np.stack([B["state"][s] for s in STATE])).all(0)))
    print("timing features start", d.dates[first], "test from", d.dates[t0], flush=True)
    res = {"family": FAMILY, "test": f"{d.dates[t0]}..{d.dates[steps[-1]]}", "stocks": int(B["stock"].sum()),
           "variants": {}, "references": {}, "factors": {}}
    # factor return summary over the test period (daily, annualized), and over the pre-test period
    for k, name in enumerate(FACTORS):
        for lab, rng in (("pretest", slice(first, t0)), ("test", slice(t0, n))):
            f = B["F"][k, rng]
            f = f[np.isfinite(f)]
            res["factors"].setdefault(name, {})[lab] = {"ann_ret_pct": float(f.mean() * 252 * 100),
                                                        "sharpe": float(f.mean() / f.std(ddof=1) * np.sqrt(252))}
    # references: single factors and static composite
    for k, name in enumerate(FACTORS):
        p = np.where(test, B["Z"][k], np.nan)
        res["references"][name] = ml_eval.score(p, B["y"], steps, H, d.dates, COST_BPS)
    Wst = np.ones((K, n))
    res["references"]["static_composite"] = ml_eval.score(np.where(test, stock_score(Wst, B["Z"]), np.nan), B["y"], steps, H, d.dates, COST_BPS)
    W = {"V1_fmom20": np.sign(B["fmom20"]), "V2_fmom60": np.sign(B["fmom60"])}
    feats = timing_features(B)
    print("V3 ridge", flush=True)
    W["V3_ridge_timing"] = ml_eval.walk_forward(feats, B["G"], first, t0, H, RETRAIN, per_factor_ridge_fit,
                                                per_factor_ridge_predict, min_train=MIN_TRAIN)
    print("V4 lgbm", flush=True)
    W["V4_lgbm_timing"] = ml_eval.walk_forward(feats, B["G"], first, t0, H, RETRAIN, lgb_fit,
                                               lambda m, X: m.predict(X), min_train=MIN_TRAIN)
    preds = {}
    for name, w in W.items():
        w = np.where(test[None, :], w, np.nan)
        p = stock_score(w, B["Z"])
        preds[name] = (w, p)
        sc = ml_eval.score(p, B["y"], steps, H, d.dates, COST_BPS)
        sc["factor_level"] = factor_level(w, B["G"], steps)
        sc["mean_weight"] = {f: float(np.nanmean(w[k, t0:])) for k, f in enumerate(FACTORS)}
        res["variants"][name] = sc
        print(name, {k: sc[k] for k in ("ic", "ic_t", "spread_net_pct", "pass")}, flush=True)
    # V5 market timing
    print("V5 market", flush=True)
    M = np.stack([B["state"][s] for s in MKT_FEATS])                   # (F, n)
    tgt = B["spy_fwd"][None, :]
    p5 = ml_eval.walk_forward(lambda idx: M[:, idx].T, tgt, first, t0, H, RETRAIN, ridge_fit, ridge_predict, min_train=MIN_TRAIN)
    naive = np.full((1, n), np.nan)
    for t in range(n):
        past = B["spy_fwd"][:max(0, t - H)]                             # targets realized by the close of t
        past = past[np.isfinite(past)]
        if len(past) >= 20:
            naive[0, t] = past.mean()
    r2 = ml_eval.r2_oos(p5, tgt, naive, steps)
    r2_zero = ml_eval.r2_oos(p5, tgt, np.zeros((1, n)), steps)
    per = steps[::H]
    pp, yy = p5[0, per], B["spy_fwd"][per]
    ok = np.isfinite(pp) & np.isfinite(yy)
    pp, yy = pp[ok], yy[ok]
    rho = float(np.corrcoef(pp, yy)[0, 1])
    pos = (pp > 0).astype(float)
    switches = np.abs(np.diff(np.concatenate([[1.0], pos])))
    timed = pos * (np.exp(yy) - 1) - switches * COST_BPS / 1e4         # 10 bps per in-or-out switch
    bh = np.exp(yy) - 1
    t_ = lambda x: float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))
    v5 = {"r2_oos": r2, "r2_vs_zero": r2_zero, "periods": int(ok.sum()), "corr": rho,
          "corr_t": float(rho * np.sqrt((len(pp) - 2) / (1 - rho ** 2))), "hit_rate": float(np.mean(np.sign(pp) == np.sign(yy))),
          "share_long": float(pos.mean()), "timed_total_pct": float((np.prod(1 + timed) - 1) * 100),
          "buyhold_total_pct": float((np.prod(1 + bh) - 1) * 100), "timed_minus_bh_per_period_pct": float(np.mean(timed - bh) * 100),
          "timed_minus_bh_t": t_(timed - bh), "switches": int(switches.sum())}
    v5["pass"] = bool(r2["r2_model"] > r2["r2_naive"] and v5["corr_t"] >= 2.0)
    res["variants"]["V5_market_timing"] = v5
    print("V5", json.dumps(v5), flush=True)
    # best per-stock variant by the pre-registered rule
    cand = [k for k in W]
    best = max(cand, key=lambda k: (res["variants"][k]["ic_t"], res["variants"][k]["spread_net_pct"]))
    res["best"] = best
    w, p = preds[best]
    res["best_open_target"] = ml_eval.score(p, B["y_open"], steps, H, d.dates, COST_BPS)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{FAMILY}_oos.npz", pred=p, symbols=np.array(d.sym), dates=np.array(d.dates),
                        weights=w, factors=np.array(FACTORS), variant=np.array(best),
                        spy_pred=p5[0], note=np.array("pred: per-stock timed-factor score at the close of each date "
                                                      "(NaN outside the test period and for ETFs); weights: factor "
                                                      "weights (factors x dates); spy_pred: V5 SPY 5-session forecast"))
    (OUT / f"{FAMILY}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("best", best)
    return res


def today(d=None, variant=None):
    """Per-stock scores at the latest cached close for the chosen variant: {as_of, weights, scores{sym: {...}}}."""
    d = d or trader.Data()
    variant = variant or BEST
    B = build(d)
    n, t = B["n"], B["n"] - 1
    if variant == "V1_fmom20":
        w = np.sign(B["fmom20"][:, t])
    elif variant == "V2_fmom60":
        w = np.sign(B["fmom60"][:, t])
    else:
        first = int(np.argmax(np.isfinite(B["fmom60"]).all(0)))
        tr = np.arange(first, n - H - 1)
        feats = timing_features(B)
        X, y = feats(tr), B["G"][:, tr].T.reshape(-1)
        ok = np.isfinite(y)
        if variant == "V3_ridge_timing":
            m = per_factor_ridge_fit(X[ok], y[ok])
            w = per_factor_ridge_predict(m, feats([t]))
        else:
            w = lgb_fit(X[ok], y[ok]).predict(feats([t]))
    z = B["Z"][:, :, t]
    s = (w[:, None] * z).sum(0)
    s[np.isnan(z).any(0)] = np.nan
    ok = np.isfinite(s)
    pct = np.full(len(s), np.nan)
    pct[ok] = 100 * s[ok].argsort().argsort() / (ok.sum() - 1)
    scores = {sym: {"score": round(float(s[j]), 3), "rank_pct": round(float(pct[j]), 1),
                    "exposures": {f: round(float(z[k, j]), 2) for k, f in enumerate(FACTORS)}}
              for j, sym in enumerate(d.sym) if ok[j]}
    return {"as_of": d.dates[t], "variant": variant, "weights": {f: float(w[k]) for k, f in enumerate(FACTORS)},
            "factor_mom20_pct": {f: round(float(B["fmom20"][k, t] * 100), 2) for k, f in enumerate(FACTORS)},
            "state": {s_: round(float(B["state"][s_][t]), 4) for s_ in MKT_FEATS}, "scores": scores}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "today":
        r = today()
        top = sorted(r["scores"].items(), key=lambda kv: -kv[1]["score"])[:10]
        print(json.dumps({k: v for k, v in r.items() if k != "scores"}, indent=1))
        print("top 10:", [k for k, _ in top])
    else:
        run()
