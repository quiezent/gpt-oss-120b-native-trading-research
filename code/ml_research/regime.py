"""Tool "regime": market regime for position sizing (pre-registration: runs/ml/research/regime_prereg.md).

A Gaussian HMM (numpy EM, causal filtered probabilities only) and plain regressions on SPY returns, SPY realized vol
(5/20), SPY implied vol, breadth and cross-sectional dispersion. Judged on what sizing needs: next 5/20-session SPY
vol, P(SPY drawdown worse than -3%/-5% within 5/20 sessions), next-period dispersion, each vs a naive baseline; SPY
return sign is reported too (expected: no skill).

Result (runs/ml/research/regime.md): no regime model beat SPY implied vol, so live()/sim_provider() serve median-calibrated
SPY IV for 5/20-session vol (with 10-90% intervals), base rates for the drawdown probabilities, and the 3-state HMM label
with its filtered probabilities and each state's mean vol as a description.

    .venv\\Scripts\\python.exe ml_research\\regime.py            # validate(): regime.json, regime.md, regime_oos.npz
    .venv\\Scripts\\python.exe ml_research\\regime.py --live     # live() on a fake gateway built from the cache

Features at t use data through the close of t only; targets start at t + 1.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trader  # noqa: E402

OUT = ROOT / "runs" / "ml" / "research"
IV_FILE = ROOT / "data" / "iv" / "SPY.json"
TEST_FROM, START, RETRAIN, MIN_TRAIN, MIN_TRAIN_IV, MIN_POOL = "20250102", 50, 21, 120, 60, 40
HMM_FEATS = ["r1", "lrv5", "lrv20", "breadth", "ldisp5"]
REG_FEATS = ["lrv5", "lrv20", "breadth", "ldisp5", "r20", "r1"]
IV_FEATS = ["liv", "lrv5", "lrv20"]
VARIANTS = ["hmm2", "hmm3", "reg", "reg_hmm3", "reg_iv"]
POSTHOC = ["reg_clip"]     # not pre-registered: reg with test inputs clipped to the training window's 1-99% range
LOG_T = {"vol5": 5, "vol20": 20, "disp5": 5, "disp20": 20}
BIN_T = {"dd3_5": 5, "dd5_5": 5, "dd3_20": 20, "dd5_20": 20, "up5": 5, "up20": 20}
DD = {"dd3_5": -0.03, "dd5_5": -0.05, "dd3_20": -0.03, "dd5_20": -0.05}
LABELS = {2: ["calm", "stressed"], 3: ["calm", "normal", "stressed"]}

TRACK_RECORD = (
    "Out of sample, 2 Jan 2025 to Oct 2026 (walk-forward, monthly refits): no regime model forecast SPY volatility better "
    "than SPY's own implied volatility. The models were 2- and 3-state HMMs and regressions on SPY returns, realized vol, "
    "breadth and dispersion. The vol numbers here are therefore SPY 30-day implied vol, de-biased by its median past "
    "error. 20-session vol: R2 0.28, vs -0.20 for trailing 20-day realized vol. 5-session vol: R2 0.34. Their 10-90% "
    "intervals held 79% and 81% of outcomes. They missed the April 2025 shock (48% coverage in 2025H1) and were too wide "
    "in calm 2026. No model forecast SPY drawdowns better than their historical frequency, so the drawdown probabilities "
    "are base rates that do not change with conditions. No model forecast SPY's direction either. The regime label is a "
    "coarse and overconfident description of current volatility: 'stressed' days were followed by 22% annualized vol, "
    "'calm' days by 10%. It carries no direction signal. Three years of data with one major shock.")


# ---------------------------------------------------------------- data
def load_iv(path=IV_FILE):
    try:
        return {b["date"]: float(b["close"]) for b in json.loads(Path(path).read_text(encoding="utf-8"))["bars"]
                if b.get("close") and b["close"] > 0}
    except (OSError, ValueError, KeyError):
        return {}


def series(d, iv=None):
    """Market-level features (one value per session, through the close of t) and the daily dispersion series."""
    iv = load_iv() if iv is None else iv
    j = d.k["SPY"]
    c, lr = d.c[j], d.lr[j]
    nd = len(d.dates)
    stocks = np.array([s not in d.etf for s in d.sym])
    LR, C = d.lr[stocks], d.c[stocks]
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(LR), axis=0)
        disp = np.where(cnt >= 100, np.nanstd(LR, axis=0, ddof=1), np.nan)
        sma = np.full(C.shape, np.nan)
        cs = np.nancumsum(C, axis=1)
        ok = np.cumsum(~np.isnan(C), axis=1)
        sma[:, 50:] = (cs[:, 50:] - cs[:, :-50]) / 50
        sma[:, 49] = cs[:, 49] / 50
        valid = (ok - np.concatenate([np.zeros((C.shape[0], 50)), ok[:, :-50]], axis=1)) == 50
        above = np.where(valid & ~np.isnan(C), (C > sma).astype(float), np.nan)
        breadth = np.where(np.sum(~np.isnan(above), axis=0) >= 100, np.nanmean(above, axis=0), np.nan)

    def trail(x, n, f=np.mean):
        out = np.full(nd, np.nan)
        for t in range(n, nd):
            w = x[t - n + 1:t + 1]
            if not np.any(np.isnan(w)):
                out[t] = f(w)
        return out
    r2 = lr ** 2
    S = {"r1": lr.copy(), "r20": np.r_[np.full(20, np.nan), np.log(c[20:] / c[:-20])],
         "lrv5": np.log(np.sqrt(252 * trail(r2, 5))), "lrv20": np.log(np.sqrt(252 * trail(r2, 20))),
         "breadth": breadth, "ldisp5": np.log(trail(disp, 5)), "ldisp20": np.log(trail(disp, 20)),
         "liv": np.array([np.log(iv[x]) if x in iv else np.nan for x in d.dates])}
    S["r1"][:1] = np.nan
    return S, disp


def targets(d, disp):
    j = d.k["SPY"]
    c, lr = d.c[j], d.lr[j]
    nd = len(d.dates)
    Y = {k: np.full(nd, np.nan) for k in list(LOG_T) + list(BIN_T)}
    for t in range(nd):
        for h in (5, 20):
            if t + h >= nd:
                continue
            r = lr[t + 1:t + h + 1]
            Y[f"vol{h}"][t] = np.log(np.sqrt(252 * np.mean(r ** 2)))
            Y[f"disp{h}"][t] = np.log(np.mean(disp[t + 1:t + h + 1]))
            Y[f"up{h}"][t] = float(np.sum(r) > 0)
            m = np.min(c[t + 1:t + h + 1]) / c[t] - 1
            for x in (3, 5):
                Y[f"dd{x}_{h}"][t] = float(m < -x / 100)
    return Y


# ---------------------------------------------------------------- Gaussian HMM (diagonal), numpy EM
def _logb(X, mu, var):
    return -0.5 * (np.sum(np.log(2 * np.pi * var), 1)[None] + np.sum((X[:, None, :] - mu[None]) ** 2 / var[None], 2))


def _forward(X, pi, A, mu, var):
    lb = _logb(X, mu, var)
    m = lb.max(1, keepdims=True)
    B = np.exp(lb - m)
    T, K = B.shape
    al, s = np.zeros((T, K)), np.zeros(T)
    a = pi * B[0]
    for t in range(T):
        if t:
            a = (al[t - 1] @ A) * B[t]
        s[t] = a.sum()
        al[t] = a / s[t]
    return al, s, B, float(np.sum(np.log(s)) + np.sum(m))


def hmm_fit(X, K, seed=0, n_init=5, iters=200, floor=1e-3):
    best = None
    rng = np.random.default_rng(seed)
    T, D = X.shape
    for i in range(n_init):
        if i == 0:
            g = np.minimum((np.argsort(np.argsort(X[:, HMM_FEATS.index("lrv20")])) * K) // T, K - 1)
            R = np.eye(K)[g]
        else:
            R = rng.dirichlet(np.ones(K), size=T)
        w = R.sum(0) + 1e-9
        mu = (R.T @ X) / w[:, None]
        var = np.maximum((R.T @ X ** 2) / w[:, None] - mu ** 2, floor)
        pi, A = np.full(K, 1 / K), np.full((K, K), 0.05 / max(K - 1, 1)) + np.eye(K) * (0.95 - 0.05 / max(K - 1, 1))
        prev = -np.inf
        for _ in range(iters):
            al, s, B, ll = _forward(X, pi, A, mu, var)
            be = np.ones((T, K))
            for t in range(T - 2, -1, -1):
                be[t] = A @ (B[t + 1] * be[t + 1]) / s[t + 1]
            ga = al * be
            ga /= ga.sum(1, keepdims=True)
            xi = np.einsum("ti,tj->ij", al[:-1], B[1:] * be[1:] / s[1:, None]) * A
            pi = ga[0] + 1e-6
            pi /= pi.sum()
            A = xi / xi.sum(1, keepdims=True)
            w = ga.sum(0) + 1e-9
            mu = (ga.T @ X) / w[:, None]
            var = np.maximum((ga.T @ X ** 2) / w[:, None] - mu ** 2, floor)
            if ll - prev < 1e-6 * abs(ll):
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, pi, A, mu, var)
    _, pi, A, mu, var = best
    o = np.argsort(mu[:, HMM_FEATS.index("lrv20")])     # order states calm -> stressed
    return {"pi": pi[o], "A": A[np.ix_(o, o)], "mu": mu[o], "var": var[o], "ll": best[0]}


def hmm_filter(X, m):
    return _forward(X, m["pi"], m["A"], m["mu"], m["var"])[0]


# ---------------------------------------------------------------- models
def _ols(X, y):
    Z = np.c_[np.ones(len(X)), X]
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    return lambda Xn: np.c_[np.ones(len(Xn)), Xn] @ b


def _logit(X, y):
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max():
        p = float(y.mean())
        return lambda Xn: np.full(len(Xn), p)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
    return lambda Xn: m.predict_proba((Xn - mu) / sd)[:, 1]


def _state_map(P, y):
    w = P.sum(0)
    rate = (P.T @ y) / np.maximum(w, 1e-9)
    rate = np.where(w > 1e-6, rate, y.mean())
    return lambda Pn: Pn @ rate


_HMM_CACHE = {}


def _fit_hmm(S, rows, K):
    key = (K, int(rows[0]), int(rows[-1]), float(np.nansum(S["r1"][rows])))
    if key in _HMM_CACHE:
        return _HMM_CACHE[key]
    X = np.c_[[S[f][rows] for f in HMM_FEATS]].T
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = hmm_fit((X - mu) / sd, K)
    m["zmu"], m["zsd"] = mu, sd
    _HMM_CACHE[key] = m
    return m


def _hmm_probs(S, m, upto):
    X = np.c_[[S[f][START:upto + 1] for f in HMM_FEATS]].T
    P = np.full((upto + 1, len(m["pi"])), np.nan)
    P[START:] = hmm_filter((X - m["zmu"]) / m["zsd"], m)
    return P


def fit_predict(S, Y, variant, T, te, all_rows=False):
    """Fit `variant` with information available before step T and predict steps te.
    Returns {target: (pred[te], train_resid or None)} and the state probabilities at te (hmm variants)."""
    out, probs, svol = {}, None, None
    feat_rows = np.arange(START, T)
    P = None
    if variant in ("hmm2", "hmm3", "reg_hmm3"):
        K = 2 if variant == "hmm2" else 3
        m = _fit_hmm(S, feat_rows, K)
        P = _hmm_probs(S, m, int(te[-1]))
        probs = P[te]
        i = HMM_FEATS.index("lrv20")
        svol = 100 * np.exp(m["mu"][:, i] * m["zsd"][i] + m["zmu"][i])      # each state's mean realized vol, % a year
    for name in list(LOG_T) + list(BIN_T):
        h = (LOG_T | BIN_T)[name]
        tr = np.arange(START, T if all_rows else T - h)
        y = Y[name][tr]
        if variant.startswith("hmm"):
            ok = ~np.isnan(y) & ~np.isnan(P[tr, 0])
            f = _state_map(P[tr][ok], y[ok])
            pred, fit = f(P[te]), f(P[tr][ok])
        else:
            cols = IV_FEATS if variant == "reg_iv" else REG_FEATS
            Xtr = np.c_[[S[c][tr] for c in cols]].T
            Xte = np.c_[[S[c][te] for c in cols]].T
            if variant == "reg_clip":
                lo, hi = np.nanpercentile(Xtr, [1, 99], axis=0)
                Xte = np.clip(Xte, lo, hi)
            if variant == "reg_hmm3":
                Xtr, Xte = np.c_[Xtr, P[tr][:, 1:]], np.c_[Xte, P[te][:, 1:]]
            ok = ~np.isnan(y) & ~np.any(np.isnan(Xtr), 1)
            need = MIN_TRAIN_IV if variant == "reg_iv" else MIN_TRAIN
            if ok.sum() < need:
                continue
            f = _ols(Xtr[ok], y[ok]) if name in LOG_T else _logit(Xtr[ok], y[ok])
            good = ~np.any(np.isnan(Xte), 1)
            pred = np.full(len(te), np.nan)
            if good.any():
                pred[good] = f(Xte[good])
            fit = f(Xtr[ok])
        out[name] = (pred, (y[ok] - fit) if name in LOG_T else None, float(np.mean(y[ok])))
    return out, probs, svol


# ---------------------------------------------------------------- walk-forward
def walk(S, Y, dates):
    nd = len(dates)
    T0 = next(i for i, x in enumerate(dates) if x >= TEST_FROM)
    names = list(LOG_T) + list(BIN_T)
    R = {v: {n: np.full(nd, np.nan) for n in names} for v in VARIANTS + POSTHOC}
    Q = {v: {n: np.full((nd, 3), np.nan) for n in LOG_T} for v in VARIANTS + POSTHOC}       # train-residual q10, q50, q90
    base = {n: np.full(nd, np.nan) for n in BIN_T}
    probs = {"hmm2": np.full((nd, 2), np.nan), "hmm3": np.full((nd, 3), np.nan), "hmm3_state_vol": np.full((nd, 3), np.nan)}
    for T in range(T0, nd, RETRAIN):
        te = np.arange(T, min(T + RETRAIN, nd))
        for v in VARIANTS + POSTHOC:
            res, p, sv = fit_predict(S, Y, v, T, te)
            if v in probs and p is not None:
                probs[v][te] = p
            if v == "hmm3":
                probs["hmm3_state_vol"][te] = sv
            for n, (pred, resid, mean) in res.items():
                R[v][n][te] = pred
                if n in LOG_T:
                    Q[v][n][te] = np.quantile(resid, [0.1, 0.5, 0.9])
                if n in BIN_T and v == "reg":
                    base[n][te] = mean
        print(dates[T], flush=True)
    return R, Q, base, probs


def pool_q(pred, y, h, dates, T0):
    """Method (b): q10/q90 of earlier realized OOS log residuals (t + h <= T - 1), once >= MIN_POOL exist."""
    nd = len(dates)
    q = np.full((nd, 3), np.nan)
    for T in range(T0, nd, RETRAIN):
        past = np.arange(T0, max(T0, T - h))
        r = (y - pred)[past]
        r = r[~np.isnan(r)]
        if len(r) >= MIN_POOL:
            q[T:T + RETRAIN] = np.quantile(r, [0.1, 0.5, 0.9])
    return q


# ---------------------------------------------------------------- scoring
def r2(y, f):
    return float(1 - np.mean((y - f) ** 2) / np.var(y))


def half(x):
    return x[:4] + ("H1" if x[4:6] <= "06" else "H2")


def brier_block(p, y, base, rw=None):
    from sklearn.metrics import roc_auc_score
    pc = np.clip(p, 0.005, 0.995)
    out = {"n": int(len(y)), "events": int(y.sum()), "rate": float(y.mean()), "brier": float(np.mean((p - y) ** 2)),
           "brier_base": float(np.mean((base - y) ** 2)),
           "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
           "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}
    out["bss_base"] = 1 - out["brier"] / out["brier_base"]
    if rw is not None:
        out["brier_rw"] = float(np.mean((rw - y) ** 2))
        out["bss_rw"] = 1 - out["brier"] / out["brier_rw"]
    return out


def reliability(p, y, edges=(0, 0.05, 0.15, 0.3, 0.5, 1.0001)):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.any():
            rows.append({"bin": f"{a:.2f}-{min(b, 1):.2f}", "n": int(m.sum()), "mean_p": float(p[m].mean()),
                         "freq": float(y[m].mean())})
    return rows


def rw_prob(lrv20, x, h):
    sd = np.exp(lrv20) / np.sqrt(252) * np.sqrt(h)
    return np.minimum(1.0, 2 * norm.cdf(np.log(1 + x) / sd))


def evaluate(S, Y, R, Q, base, probs, dates):
    nd = len(dates)
    T0 = next(i for i, x in enumerate(dates) if x >= TEST_FROM)
    test = np.arange(T0, nd)
    ivw = ~np.isnan(R["reg_iv"]["vol20"])
    res = {"log": {}, "bin": {}, "coverage": {}, "regimes": {}}
    naive = {"vol5": {"rv20": S["lrv20"], "rv5": S["lrv5"], "iv": S["liv"]},
             "vol20": {"rv20": S["lrv20"], "rv5": S["lrv5"], "iv": S["liv"]},
             "disp5": {"disp_now": S["ldisp5"]}, "disp20": {"disp_now": S["ldisp20"]}}
    for n, h in LOG_T.items():
        y = Y[n]
        allv = [R[v][n] for v in VARIANTS if v != "reg_iv"] + [a for k, a in naive[n].items() if k != "iv"]
        cells = test[np.all([~np.isnan(a[test]) for a in allv + [y]], axis=0)]
        e = {"full": {"window": f"{dates[cells[0]]}..{dates[cells[-1]]}", "cells": int(len(cells)),
                      "eff_n": int(len(cells) // h)}}
        for v in VARIANTS:
            if v != "reg_iv":
                e["full"][v] = r2(y[cells], R[v][n][cells])
        for k, a in naive[n].items():
            if k != "iv":
                e["full"]["naive_" + k] = r2(y[cells], a[cells])
        e["full"]["by_half"] = {}
        for hh in sorted({half(dates[t]) for t in cells}):
            cc = np.array([t for t in cells if half(dates[t]) == hh])
            e["full"]["by_half"][hh] = {**{v: r2(y[cc], R[v][n][cc]) for v in VARIANTS if v != "reg_iv"},
                                        **{"naive_" + k: r2(y[cc], a[cc]) for k, a in naive[n].items() if k != "iv"}}
        civ = test[np.all([~np.isnan(a[test]) for a in [R[v][n] for v in VARIANTS] + list(naive[n].values()) + [y]], axis=0)]
        if len(civ) > 30:
            e["iv_era"] = {"window": f"{dates[civ[0]]}..{dates[civ[-1]]}", "cells": int(len(civ)), "eff_n": int(len(civ) // h)}
            for v in VARIANTS:
                e["iv_era"][v] = r2(y[civ], R[v][n][civ])
            for k, a in naive[n].items():
                e["iv_era"]["naive_" + k] = r2(y[civ], a[civ])
        res["log"][n] = e
    for n, h in BIN_T.items():
        y = Y[n]
        rw = rw_prob(S["lrv20"], DD[n], h) if n in DD else None
        cells = test[np.all([~np.isnan(R[v][n][test]) for v in VARIANTS if v != "reg_iv"] + [~np.isnan(y[test]), ~np.isnan(base[n][test])], axis=0)]
        e = {"window": f"{dates[cells[0]]}..{dates[cells[-1]]}", "eff_n": int(len(cells) // h)}
        e["naive_rw"] = brier_block(rw[cells], y[cells], base[n][cells]) if rw is not None else None
        for v in VARIANTS:
            c2 = cells if v != "reg_iv" else test[~np.isnan(R[v][n][test]) & ~np.isnan(y[test]) & ~np.isnan(base[n][test])]
            if len(c2) < 30:
                continue
            e[v] = brier_block(R[v][n][c2], y[c2], base[n][c2], rw[c2] if rw is not None else None)
            e[v]["cells"] = int(len(c2))
            e[v]["reliability"] = reliability(R[v][n][c2], y[c2])
        if rw is not None:
            e["naive_rw"]["reliability"] = reliability(rw[cells], y[cells])
        res["bin"][n] = e
    # interval coverage (10-90%) for both methods
    for n, h in LOG_T.items():
        res["coverage"][n] = {}
        for v in VARIANTS:
            y, p = Y[n], R[v][n]
            qa = Q[v][n]
            qb = pool_q(p, y, h, dates, T0)
            qb = np.where(np.isnan(qb), qa, qb)
            ok = test[~np.isnan(y[test]) & ~np.isnan(p[test]) & ~np.isnan(qa[test, 0])]
            if len(ok) < 30:
                continue
            row = {}
            for meth, q in (("train", qa), ("oos_pool", qb)):
                inside = (y[ok] >= p[ok] + q[ok, 0]) & (y[ok] <= p[ok] + q[ok, 2])
                byh = {}
                for hh in sorted({half(dates[t]) for t in ok}):
                    m = np.array([half(dates[t]) == hh for t in ok])
                    byh[hh] = float(inside[m].mean())
                st = np.nanargmax(probs["hmm3"][ok], axis=1) if not np.all(np.isnan(probs["hmm3"][ok])) else None
                byr = {LABELS[3][k]: float(inside[st == k].mean()) for k in range(3) if st is not None and (st == k).sum() >= 10}
                row[meth] = {"coverage": float(inside.mean()), "below": float(np.mean(y[ok] < p[ok] + q[ok, 0])),
                             "above": float(np.mean(y[ok] > p[ok] + q[ok, 2])),
                             "mean_width_x": float(np.mean(np.exp(q[ok, 2] - q[ok, 0]))), "by_half": byh, "by_regime": byr}
            res["coverage"][n][v] = row
    # what each hmm3 regime meant out of sample
    for K in (2, 3):
        P = probs[f"hmm{K}"]
        ok = test[~np.isnan(P[test, 0])]
        st = np.argmax(P[ok], axis=1)
        tab = {}
        for k in range(K):
            m = ok[st == k]
            if not len(m):
                continue
            g = lambda a: float(np.nanmean(a[m])) if np.any(~np.isnan(a[m])) else None
            tab[LABELS[K][k]] = {"sessions": int(len(m)), "share": float(len(m) / len(ok)),
                                 "rv20_now_pct": 100 * float(np.nanmean(np.exp(S["lrv20"][m]))),
                                 "next_vol20_pct": 100 * float(np.nanmean(np.exp(Y["vol20"][m]))),
                                 "p_dd3_5": g(Y["dd3_5"]), "p_dd5_20": g(Y["dd5_20"]), "p_up20": g(Y["up20"]),
                                 "mean_next20_ret_pct": 100 * float(np.nanmean([np.log(_c(t + 20) / _c(t)) if t + 20 < nd else np.nan for t in m]))}
        res["regimes"][f"hmm{K}"] = tab
    return res


_C = None


def _c(t):
    return _C[t]


# ---------------------------------------------------------------- shipping choice
def choose(ev):
    """Per output, the passing variant with the best OOS score, or the naive that beats every variant."""
    ship = {}
    for n in LOG_T:
        e = ev["log"][n]
        full = e["full"]
        best = max((v for v in VARIANTS if v in full), key=lambda v: full[v])
        nv = max(full[k] for k in full if k.startswith("naive_"))
        passes = full[best] > nv
        iv = e.get("iv_era", {})
        civ = {v: iv[v] for v in VARIANTS if v in iv}
        ivbest = max(civ, key=civ.get) if civ else None
        beats_iv = bool(iv) and civ[ivbest] > iv.get("naive_iv", -9)
        if n.startswith("vol") and iv and iv.get("naive_iv", -9) > max(civ.values()):
            ship[n] = {"use": "naive_iv", "why": "SPY IV beats every variant in the IV era", "fallback": best if passes else "naive_rv20"}
        elif n.startswith("vol") and iv and ivbest == "reg_iv" and beats_iv:
            ship[n] = {"use": "reg_iv", "why": "best in the IV era and above IV", "fallback": best if passes else "naive_rv20"}
        else:
            ship[n] = {"use": best if passes else max((k for k in full if k.startswith("naive_")), key=lambda k: full[k]),
                       "why": "best variant, above every no-fit naive" if passes else "no variant beat the naive"}
    for n in BIN_T:
        e = ev["bin"][n]
        cand = {v: e[v]["brier"] for v in VARIANTS if v in e and v != "reg_iv"}
        best = min(cand, key=cand.get)
        passes = e[best]["bss_base"] > 0
        rw_better = e["naive_rw"] is not None and e["naive_rw"]["brier"] < e[best]["brier"]
        if rw_better and e["naive_rw"]["bss_base"] > 0:
            ship[n] = {"use": "naive_rw", "why": "random-walk probability from current vol beats every variant"}
        elif passes:
            ship[n] = {"use": best, "why": "best variant, Brier skill > 0 vs base rate"}
        else:
            ship[n] = {"use": "base", "why": "nothing beat the base rate"}
    return ship


# ---------------------------------------------------------------- post-hoc (not pre-registered; conservative only)
def bss_ci(p, y, b, h, reps=2000, seed=0):
    """90% moving-block bootstrap interval (block = h sessions) of Brier skill vs the base rate."""
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / h))
    out = []
    for _ in range(reps):
        idx = (rng.integers(0, n - h + 1, nb)[:, None] + np.arange(h)[None]).ravel()[:n]
        bb = np.mean((b[idx] - y[idx]) ** 2)
        out.append(1 - np.mean((p[idx] - y[idx]) ** 2) / bb if bb > 0 else np.nan)
    return [float(x) for x in np.nanquantile(out, [0.05, 0.95])]


def posthoc(S, Y, R, base, ship, dates):
    """Two guards added after seeing the results, both only able to make the tool more conservative:
    (1) a passing probability is served only if its 90% block-bootstrap Brier-skill interval excludes 0;
    (2) vol5's regression extrapolated the +10.5% SPY day of 2025-04-09 into an 8.9% vol forecast (realized: far
        higher), so inputs are clipped to the training window's 1-99% range (reg_clip); it ships only if it still beats
        every naive including IV."""
    T0 = next(i for i, x in enumerate(dates) if x >= TEST_FROM)
    test = np.arange(T0, len(dates))
    res = {"bss_ci": {}, "vol5_clip": {}}
    for n, h in BIN_T.items():
        u = ship[n]["use"]
        v = u if u in R else "reg"
        ok = test[~np.isnan(R[v][n][test]) & ~np.isnan(Y[n][test]) & ~np.isnan(base[n][test])]
        ci = bss_ci(R[v][n][ok], Y[n][ok], base[n][ok], h)
        res["bss_ci"][n] = {"variant": v, "ci90": ci}
        if u in R and ci[0] <= 0:
            ship[n] = {"use": "base", "why": f"{u} passed the pre-registered bar, but its 90% block-bootstrap Brier-skill "
                                              f"interval [{ci[0]:+.3f}, {ci[1]:+.3f}] includes 0 (post-hoc guard)"}
    y = Y["vol5"]
    ok = test[~np.isnan(y[test]) & ~np.isnan(R["reg"]["vol5"][test]) & ~np.isnan(R["reg_clip"]["vol5"][test]) & ~np.isnan(S["liv"][test])]
    i9 = dates.index("20250409")
    res["vol5_clip"] = {"cells": int(len(ok)), "r2_reg": r2(y[ok], R["reg"]["vol5"][ok]),
                        "r2_reg_clip": r2(y[ok], R["reg_clip"]["vol5"][ok]), "r2_naive_iv": r2(y[ok], S["liv"][ok]),
                        "r2_naive_rv20": r2(y[ok], S["lrv20"][ok]), "r2_naive_rv5": r2(y[ok], S["lrv5"][ok]),
                        "20250409_reg_pct": 100 * float(np.exp(R["reg"]["vol5"][i9])),
                        "20250409_reg_clip_pct": 100 * float(np.exp(R["reg_clip"]["vol5"][i9])),
                        "20250409_realized_pct": 100 * float(np.exp(y[i9]))}
    res["vol5_clip"]["by_half"] = {}
    for hh in sorted({half(dates[t]) for t in ok}):
        cc = np.array([t for t in ok if half(dates[t]) == hh])
        res["vol5_clip"]["by_half"][hh] = {"reg": r2(y[cc], R["reg"]["vol5"][cc]), "reg_clip": r2(y[cc], R["reg_clip"]["vol5"][cc]),
                                           "naive_iv": r2(y[cc], S["liv"][cc]), "naive_rv20": r2(y[cc], S["lrv20"][cc])}
    # (3) raw IV is biased high, so "beats IV" is too easy a bar: compare with median-calibrated IV
    #     (IV * exp(median of IV's own embargoed past log residuals), recomputed monthly), the vol outputs' shipped point.
    res["iv_cal"] = {}
    for n, h in (("vol5", 5), ("vol20", 20)):
        y, cal = Y[n], np.full(len(dates), np.nan)
        for T in range(T0, len(dates), RETRAIN):
            r = (y - S["liv"])[:T - h]
            r = r[~np.isnan(r)]
            cal[T:T + RETRAIN] = S["liv"][T:T + RETRAIN] + np.median(r)
        u = "reg_clip" if n == "vol5" and ship[n]["use"] == "reg" else ship[n]["use"]
        cands = {"iv_cal": cal, "naive_iv": S["liv"], "naive_rv20": S["lrv20"], **{v: R[v][n] for v in VARIANTS + POSTHOC}}
        ok = test[~np.isnan(y[test]) & ~np.isnan(cal[test])]
        row = {}
        for k, a in cands.items():
            c2 = ok[~np.isnan(a[ok])]
            row[k] = {"r2": r2(y[c2], a[c2]), "cells": int(len(c2)),
                      "by_half": {hh: r2(y[cc], a[cc]) for hh in sorted({half(dates[t]) for t in c2})
                                  for cc in [np.array([t for t in c2 if half(dates[t]) == hh])]}}
        res["iv_cal"][n] = row
        best_model = max((v for v in VARIANTS + POSTHOC), key=lambda v: row[v]["r2"])
        if u != "naive_iv" and row[u]["r2"] > row["iv_cal"]["r2"]:
            ship[n] = {"use": u, "why": f"{u} beats median-calibrated IV ({row[u]['r2']:.2f} vs {row['iv_cal']['r2']:.2f})"}
        else:
            ship[n] = {"use": "naive_iv", "fallback": "hmm3",
                       "why": f"median-calibrated SPY IV (R2 {row['iv_cal']['r2']:.2f}) beats every model (best: {best_model} "
                              f"{row[best_model]['r2']:.2f}); point = IV * exp(median past residual)"}
    return res


# ---------------------------------------------------------------- validate
def validate():
    global _C
    d = trader.Data()
    _C = d.c[d.k["SPY"]]
    S, disp = series(d)
    Y = targets(d, disp)
    R, Q, base, probs = walk(S, Y, d.dates)
    ev = evaluate(S, Y, R, Q, base, probs, d.dates)
    ship = choose(ev)
    ev["posthoc"] = posthoc(S, Y, R, base, ship, d.dates)
    nd = len(d.dates)
    T0 = d.dates.index(next(x for x in d.dates if x >= TEST_FROM))
    # shipped composite series (strictly OOS), intervals by the method with coverage closer to 80%
    comp, lo, hi, meth, cal = {}, {}, {}, {}, {}
    for n, h in LOG_T.items():
        u = ship[n]["use"]
        if u.startswith("naive_"):
            p = {"naive_rv20": S["lrv20"] if n.startswith("vol") else S["ldisp5" if n == "disp5" else "ldisp20"],
                 "naive_rv5": S["lrv5"], "naive_iv": S["liv"], "naive_disp_now": S["ldisp5" if n == "disp5" else "ldisp20"]}[u].copy()
            p[:T0] = np.nan
            if u == "naive_iv" and ship[n].get("fallback"):        # before IV exists, the fallback
                fb = ship[n]["fallback"]
                alt = R[fb][n] if fb in R else S["lrv20"]
                p = np.where(np.isnan(p), alt, p)
        else:
            p = R[u][n].copy()
            if u == "reg_iv" and ship[n].get("fallback") in R:
                p = np.where(np.isnan(p), R[ship[n]["fallback"]][n], p)
        y = Y[n]
        qb = pool_q(p, y, h, d.dates, T0)
        # method (a) for a naive: its own expanding training residual quantiles (embargoed), the same rule as variants
        # (a) train: a variant's in-sample training-residual quantiles (from walk); a naive's own embargoed history
        if u.startswith("naive_"):
            qa = np.full((nd, 3), np.nan)
            for T in range(T0, nd, RETRAIN):
                r = (y - _naive_hist(S, u, n))[START:T - h]
                r = r[~np.isnan(r)]
                if len(r) >= 30:
                    qa[T:T + RETRAIN] = np.quantile(r, [0.1, 0.5, 0.9])
        else:
            qa = Q[u][n].copy()
        fb = ship[n].get("fallback")
        if fb in Q:
            qa = np.where(np.isnan(qa) & ~np.isnan(Q[fb][n]), Q[fb][n], qa)
        qb = np.where(np.isnan(qb), qa, qb)
        ok = np.arange(T0, nd)
        ok = ok[~np.isnan(y[ok]) & ~np.isnan(p[ok]) & ~np.isnan(qa[ok, 0])]
        cov = {m: float(np.mean((y[ok] >= p[ok] + q[ok, 0]) & (y[ok] <= p[ok] + q[ok, 2]))) for m, q in (("train", qa), ("oos_pool", qb))}
        m = min(cov, key=lambda k: abs(cov[k] - 0.8))
        q = qa if m == "train" else qb
        cal[n] = p + qa[:, 1]                    # median-calibrated point (post hoc): the train-method median
        cc = ok[~np.isnan(cal[n][ok])]
        lr20 = S["lrv20"] if n.startswith("vol") else S["ldisp5" if n == "disp5" else "ldisp20"]
        pt = {"r2_point_raw": r2(y[cc], p[cc]), "r2_point_cal": r2(y[cc], cal[n][cc]), "r2_naive_now": r2(y[cc], lr20[cc]),
              "cells": int(len(cc)), "window": f"{d.dates[cc[0]]}..{d.dates[cc[-1]]}"}
        if n.startswith("vol"):
            civ = cc[~np.isnan(S["liv"][cc])]
            pt["iv_era"] = {"r2_point_raw": r2(y[civ], p[civ]), "r2_point_cal": r2(y[civ], cal[n][civ]),
                            "r2_naive_iv": r2(y[civ], S["liv"][civ]), "cells": int(len(civ))}
        comp[n], lo[n], hi[n], meth[n] = p, p + q[:, 0], p + q[:, 2], {"method": m, "coverage": cov, "cells": int(len(ok)), "point": pt}
        # residual quantiles for live use: all realized OOS residuals (oos_pool) or the latest training quantiles
        r = (y - p)[ok]
        meth[n]["live_q"] = (np.quantile(r, [0.1, 0.5, 0.9]).tolist() if m == "oos_pool" else qa[nd - 1].tolist())
        meth[n]["live_q"][1] = float(qa[nd - 1][1])       # the point's calibration: the train-method median
    for n in BIN_T:
        u = ship[n]["use"]
        comp[n] = (rw_prob(S["lrv20"], DD[n], BIN_T[n]) if u == "naive_rw" else base[n] if u == "base" else R[u][n]).copy()
        comp[n][:T0] = np.nan
    np.savez_compressed(OUT / "regime_oos.npz", dates=np.array(d.dates),
                        **{f"ship_{n}": comp[n] for n in comp}, **{f"lo_{n}": lo[n] for n in lo}, **{f"hi_{n}": hi[n] for n in hi}, **{f"cal_{n}": cal[n] for n in cal},
                        hmm3_probs=probs["hmm3"], hmm2_probs=probs["hmm2"], hmm3_state_vol=probs["hmm3_state_vol"], lrv20=S["lrv20"],
                        **{f"{v}_{n}": R[v][n] for v in VARIANTS + POSTHOC for n in R[v]})
    # calibration of what ships: interval coverage by half-year and by regime label; label behaviour
    test = np.arange(T0, nd)
    P3 = probs["hmm3"]
    shipped = {}
    for n in ("vol5", "vol20"):
        y = Y[n]
        c = test[~np.isnan(y[test]) & ~np.isnan(lo[n][test]) & ~np.isnan(cal[n][test])]
        ins = (y[c] >= lo[n][c]) & (y[c] <= hi[n][c])
        st = P3[c].argmax(1)
        shipped[n] = {"window": f"{d.dates[c[0]]}..{d.dates[c[-1]]}", "cells": int(len(c)), "eff_n": int(len(c) // LOG_T[n]),
                      "r2_point": r2(y[c], cal[n][c]), "r2_naive_rv20": r2(y[c], S["lrv20"][c]),
                      "r2_raw_iv": r2(y[c], S["liv"][c]) if not np.any(np.isnan(S["liv"][c])) else None,
                      "coverage": float(ins.mean()), "below": float(np.mean(y[c] < lo[n][c])), "above": float(np.mean(y[c] > hi[n][c])),
                      "by_half": {hh: float(ins[np.array([half(d.dates[t]) == hh for t in c])].mean()) for hh in sorted({half(d.dates[t]) for t in c})},
                      "by_regime": {LABELS[3][k]: float(ins[st == k].mean()) for k in range(3) if (st == k).sum() >= 10}}
    for n in DD:
        y = Y[n]
        c = test[~np.isnan(y[test]) & ~np.isnan(comp[n][test])]
        shipped[n] = {"use": ship[n]["use"], **brier_block(comp[n][c], y[c], base[n][c], rw_prob(S["lrv20"][c], DD[n], BIN_T[n])),
                      "reliability": reliability(comp[n][c], y[c]), "eff_n": int(len(c) // BIN_T[n])}
    st = P3[test].argmax(1)
    shipped["labels"] = {"share_maxprob_gt_0.95": float(np.mean(P3[test].max(1) > 0.95)), "switches": int(np.sum(st[1:] != st[:-1])),
                         "sessions": int(len(test))}
    out = {"as_of": d.dates[-1], "test_from": d.dates[T0], "ship": ship, "intervals": meth, "shipped": shipped, "eval": ev,
           "median_rv20_pct_full": 100 * float(np.nanmedian(np.exp(S["lrv20"])))}
    (OUT / "regime.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    report(out)
    return out


def _naive_hist(S, u, n):
    return {"naive_rv20": S["lrv20"] if n.startswith("vol") else S["ldisp5" if n == "disp5" else "ldisp20"],
            "naive_rv5": S["lrv5"], "naive_iv": S["liv"],
            "naive_disp_now": S["ldisp5" if n == "disp5" else "ldisp20"]}[u]


# ---------------------------------------------------------------- output (shared by live and sim)
def _sources(ship):
    src = {}
    for n in ("vol20", "vol5"):
        u = ship[n]["use"]
        src[n] = {"naive_iv": "SPY 30-day implied vol, median-calibrated (it beat every model out of sample)",
                  "reg": "regression on SPY realized vol, breadth, dispersion and returns, median-calibrated",
                  "reg_clip": "regression on SPY realized vol, breadth, dispersion and returns (inputs clipped to the training range), median-calibrated",
                  "hmm3": "3-state HMM, median-calibrated", "hmm2": "2-state HMM, median-calibrated",
                  "reg_iv": "regression on SPY implied and realized vol, median-calibrated"}.get(u, u)
    for n in DD:
        u = ship[n]["use"]
        src[n] = ("historical base rate (no model beat it out of sample)" if u == "base" else
                  "random walk at current vol" if u == "naive_rw" else f"{u} (barely above the base rate out of sample)")
    return src


def _package(as_of, probs3, vals, typ, src, notes=(), svol=None):
    """vals: {output: value}; vol outputs in % a year (point is median-calibrated) with lo_/hi_ 10-90% bounds."""
    lab = LABELS[3]
    if probs3 is not None and np.all(np.isfinite(probs3)):
        k = int(np.argmax(probs3))
        label, pl, pr = lab[k], float(probs3[k]), {lab[i]: round(float(probs3[i]), 3) for i in range(3)}
    else:
        label, pl, pr = "unknown", float("nan"), None
    r = lambda x, k=1: None if x is None or not np.isfinite(x) else round(float(x), k)
    out = {"as_of": as_of, "regime": label, "regime_probs": pr,
           "regime_state_vol_pct": ({lab[i]: round(float(svol[i]), 1) for i in range(3)}
                                    if svol is not None and np.all(np.isfinite(svol)) else None),
           "spy_vol20_pct": r(vals["vol20"]), "spy_vol20_10_90_pct": [r(vals["lo_vol20"]), r(vals["hi_vol20"])],
           "spy_vol5_pct": r(vals["vol5"]), "spy_vol5_10_90_pct": [r(vals["lo_vol5"]), r(vals["hi_vol5"])],
           "p_spy_close_down3_within5": r(vals["dd3_5"], 3), "p_spy_close_down5_within5": r(vals["dd5_5"], 3),
           "p_spy_close_down3_within20": r(vals["dd3_20"], 3), "p_spy_close_down5_within20": r(vals["dd5_20"], 3),
           "typical_spy_vol_pct": r(typ), "size_scale_vs_typical": r(typ / vals["vol20"], 2) if vals["vol20"] else None,
           "sources": src, "notes": list(notes)}
    keys = ("vol20", "lo_vol20", "hi_vol20", "dd3_5", "dd5_20")
    if all(isinstance(vals.get(k), (int, float)) for k in keys) and all(np.isfinite([vals[k] for k in keys])):
        out["summary"] = (
            f"Regime {label} (probability {pl:.0%}). Expected SPY volatility over the next 20 sessions {vals['vol20']:.0f}% a year "
            f"(10-90%: {vals['lo_vol20']:.0f}-{vals['hi_vol20']:.0f}%) against a typical {typ:.0f}%: a book held at constant "
            f"risk would run about {typ / vals['vol20']:.2f}x its normal gross exposure. Chance SPY closes 3% below the last "
            f"close within 5 sessions {vals['dd3_5']:.0%}, 5% below within 20 sessions {vals['dd5_20']:.0%} "
            f"({'base rates: no model beat them' if src['dd5_20'].startswith('historical') else 'model'}).")
    return out


# ---------------------------------------------------------------- training simulator
def sim_provider(data):
    """A function of decision session d -> the regime outputs as of the close of d - 1, from strictly OOS predictions
    (runs/ml/research/regime_oos.npz)."""
    z = np.load(OUT / "regime_oos.npz", allow_pickle=True)
    src = _sources(json.loads((OUT / "regime.json").read_text(encoding="utf-8"))["ship"])
    col = {str(x): i for i, x in enumerate(z["dates"])}
    lrv20 = z["lrv20"]
    A = {k: z[k] for k in z.files}

    def at(d):
        as_of = data.dates[d - 1]
        i = col.get(as_of)
        if i is None or not np.isfinite(A["cal_vol20"][i]):
            return None
        vals = {}
        for n in ("vol20", "vol5"):
            vals[n] = float(np.exp(A[f"cal_{n}"][i]) * 100)
            vals["lo_" + n], vals["hi_" + n] = (float(np.exp(A[f"{pre}_{n}"][i]) * 100) for pre in ("lo", "hi"))
        for n in DD:
            vals[n] = float(A[f"ship_{n}"][i])
        typ = 100 * float(np.nanmedian(np.exp(lrv20[:i + 1])))
        return _package(as_of, A["hmm3_probs"][i], vals, typ, src, (), A["hmm3_state_vol"][i])
    return at


# ---------------------------------------------------------------- live
def live(gw=None, symbols=None, data=None, **kw):
    """Market-level regime outputs as of the last cached close, fit on every realized target.
    gw is used for SPY implied vol when data/iv/SPY.json lacks the last close, and to check the cache is current.
    `symbols` is accepted for a uniform interface and ignored: this tool is market-level."""
    info = json.loads((OUT / "regime.json").read_text(encoding="utf-8"))
    ship, meth = info["ship"], info["intervals"]
    d = data or trader.Data()
    iv = load_iv()
    notes = []
    as_of = d.dates[-1]
    if as_of not in iv and gw is not None:
        try:
            st = gw.stats(["SPY"])[0]
            g = st.get("implied_vol")
            g = float(g) if isinstance(g, (int, float)) else float("nan")
            if math.isfinite(g) and 0.03 <= g <= 1.5:       # R1: a fraction, plausible (not percent units, not NaN)
                iv[as_of] = g
                notes.append("SPY implied vol from the live gateway (delayed, intraday), not the close")
        except Exception as e:        # noqa: BLE001
            notes.append(f"gateway stats failed: {e}")
    if gw is not None:
        try:
            b = gw.bars("SPY", duration="1 M", bar_size="1 day")
            done = [x["date"].replace("-", "")[:8] for x in b]
            newer = [x for x in done if x > as_of]
            if len(newer) > 1:    # the last gw bar may be today's partial session
                notes.append(f"stale: the cache ends {as_of} but the gateway has completed sessions after it")
        except Exception as e:        # noqa: BLE001
            notes.append(f"gateway bars failed: {e}")
    S, disp = series(d, iv)
    Y = targets(d, disp)
    nd = len(d.dates)
    t = nd - 1
    te = np.array([t])
    fits = {}
    need = {ship[n]["use"] for n in ship} | {ship[n].get("fallback") for n in ship} | {"hmm3"}
    for v in VARIANTS + POSTHOC:
        if v in need:
            fits[v] = fit_predict(S, Y, v, nd, te, all_rows=True)
    vals = {}
    for n in LOG_T:
        u = ship[n]["use"]
        if u.startswith("naive_"):
            x = float(_naive_hist(S, u, n)[t])
        else:
            x = float(fits[u][0][n][0][0]) if n in fits[u][0] else np.nan
        if not np.isfinite(x):            # R2: no fallback model (it understated vol); say there is no forecast
            vals[n] = vals["lo_" + n] = vals["hi_" + n] = None
            notes.append(f"{n}: no SPY implied vol today")
            continue
        q = meth[n]["live_q"]
        vals[n], vals["lo_" + n], vals["hi_" + n] = (100 * math.exp(x + q[1]), 100 * math.exp(x + q[0]), 100 * math.exp(x + q[2]))
    for n in BIN_T:
        u = ship[n]["use"]
        if u == "naive_rw":
            vals[n] = float(rw_prob(S["lrv20"][t:t + 1], DD[n], BIN_T[n])[0])
        elif u == "base":
            vals[n] = float(np.nanmean(Y[n][START:]))
        else:
            vals[n] = float(fits[u][0][n][0][0])
    typ = 100 * float(np.nanmedian(np.exp(S["lrv20"])))
    out = _package(as_of, fits["hmm3"][1][0], vals, typ, _sources(ship), notes, fits["hmm3"][2])
    if symbols:
        out["symbols_note"] = "market-level tool; per-symbol requests are not used"
    return out


# ---------------------------------------------------------------- report
def report(out):
    """runs/ml/research/regime.md from validate()'s results."""
    ev, sh, ship = out["eval"], out["shipped"], out["ship"]
    ph = ev["posthoc"]
    f = lambda x, k=2: "na" if x is None else f"{x:+.{k}f}"
    L = [f"# Tool \"regime\": market regime for sizing: results (data through {out['as_of']}; written by ml_research/regime.py)", "",
         "Pre-registration: runs/ml/research/regime_prereg.md (5 variants, all reported below). Out of sample from "
         f"{out['test_from']}: walk-forward, monthly retrains on an expanding window, embargo = horizon, features through the "
         "close of t. HMM = numpy Gaussian HMM, diagonal covariance, EM, causal filtered probabilities only.", "",
         "## Conclusion", "",
         "- **No regime model beats SPY implied vol for sizing.** For 20-session SPY vol every variant has OOS R^2 below "
         f"zero, while raw SPY IV reaches {f(ph['iv_cal']['vol20']['naive_iv']['r2'])} and median-calibrated IV "
         f"{f(ph['iv_cal']['vol20']['iv_cal']['r2'])}. For 5-session vol the plain regression (no regime) beat raw IV "
         f"({f(ph['iv_cal']['vol5']['reg']['r2'])} vs {f(ph['iv_cal']['vol5']['naive_iv']['r2'])}), but only because raw IV "
         f"is biased high. Median-calibrated IV ({f(ph['iv_cal']['vol5']['iv_cal']['r2'])}) beats every model. The tool "
         "therefore serves calibrated IV for both vol outputs.",
         "- **Drawdown probabilities: no skill.** No variant has positive Brier skill vs the base rate, except reg on dd3_5 "
         f"(BSS {f(ev['bin']['dd3_5']['reg']['bss_base'], 3)}, 90% block-bootstrap interval "
         f"[{ph['bss_ci']['dd3_5']['ci90'][0]:+.3f}, {ph['bss_ci']['dd3_5']['ci90'][1]:+.3f}], AUC "
         f"{ev['bin']['dd3_5']['reg']['auc']:.2f}). That is noise, so the tool serves base rates and says so. Models trained on "
         "2024 learned that drawdowns follow calm markets (April and July 2024), and the April 2025 crash came after a "
         "stressed market. Their AUCs are often far below 0.5, which is regime instability, not skill. The random walk "
         "from current vol ranks drawdown risk better (AUC 0.6-0.9) but over-predicts it (Brier worse than the base rate).",
         "- **SPY return sign: no skill, as expected.** Every variant's Brier skill vs the base rate is negative for up5 and up20.",
         f"- **Dispersion:** reg's disp5 R^2 is {f(ev['log']['disp5']['full']['reg'])} (the naive scores "
         f"{f(ev['log']['disp5']['full']['naive_disp_now'])}), and every disp20 forecast, naive included, is below zero. "
         "Dispersion is therefore not served.",
         "- **The label:** the 3-state HMM label sorts what followed (table below), but so does current vol. It adds "
         "nothing measurable beyond IV (reg_hmm3 is worse than reg). Its probabilities are overconfident: max probability "
         f"> 0.95 on {sh['labels']['share_maxprob_gt_0.95']:.0%} of days, with {sh['labels']['switches']} label switches in "
         f"{sh['labels']['sessions']} sessions. State meanings drift with each refit (the first, 2024-trained model called "
         "13.9% vol 'stressed'), so the output carries each state's mean vol.", "",
         "## What ships, and its out-of-sample calibration", "", "| output | source | OOS record |", "|---|---|---|"]
    for n in ("vol20", "vol5"):
        x = sh[n]
        L.append(f"| {n} (% a year, 10-90% interval) | {ship[n]['use']}: {ship[n]['why']} | R^2 {x['r2_point']:+.2f} vs "
                 f"{x['r2_naive_rv20']:+.2f} for trailing rv20 and {f(x['r2_raw_iv'])} for raw IV; 10-90% coverage "
                 f"{x['coverage']:.0%} ({x['below']:.0%} below, {x['above']:.0%} above); {x['cells']} sessions, about "
                 f"{x['eff_n']} independent |")
    for n in DD:
        x = sh[n]
        L.append(f"| P({n}) | {ship[n]['use']}: {ship[n]['why']} | Brier {x['brier']:.4f} vs base {x['brier_base']:.4f}; "
                 f"{x['events']} events in {x['n']} sessions (about {x['eff_n']} independent) |")
    L += ["| regime label and probs | hmm3 filtered probabilities | descriptive; see the regime table |", "",
          "Interval coverage of the shipped vol outputs (target 80%):", "",
          "| output | " + " | ".join(sh["vol20"]["by_half"]) + " | calm | normal | stressed |",
          "|---" * (len(sh["vol20"]["by_half"]) + 4) + "|"]
    for n in ("vol20", "vol5"):
        x = sh[n]
        L.append(f"| {n} | " + " | ".join(f"{v:.0%}" for v in x["by_half"].values()) + " | " +
                 " | ".join(f"{x['by_regime'][k]:.0%}" if k in x["by_regime"] else "na" for k in LABELS[3]) + " |")
    L += ["", "vol20's 2025H1 coverage is low because of the April 2025 shock: realized vol rose above the 90% bound. In calm "
          "2026 the interval is too wide.", "", "## Results: every pre-registered variant (OOS R^2 of log targets)", ""]
    for scope in ("full", "iv_era"):
        names = VARIANTS if scope == "iv_era" else [v for v in VARIANTS if v != "reg_iv"]
        L += [f"**{scope}**", "", "| target | window | eff n | " + " | ".join(names) + " | naives |", "|---" * (len(names) + 4) + "|"]
        for n, e in ev["log"].items():
            if scope not in e:
                continue
            x = e[scope]
            nv = ", ".join(f"{k[6:]} {x[k]:+.2f}" for k in x if k.startswith("naive_"))
            L.append(f"| {n} | {x['window']} | {x['eff_n']} | " + " | ".join(f"{x[v]:+.2f}" for v in names) + f" | {nv} |")
        L.append("")
    v5 = ph["vol5_clip"]
    L += ["Post-hoc (not pre-registered):",
          f"- **reg_clip** is reg with test inputs clipped to the training 1-99% range. It was added after reg forecast "
          f"{v5['20250409_reg_pct']:.1f}% 5-session vol the day after SPY's +10.5% day of 2025-04-09 (realized "
          f"{v5['20250409_realized_pct']:.0f}%; clipped: {v5['20250409_reg_clip_pct']:.1f}%).",
          "- **iv_cal** is IV * exp(median of IV's own embargoed past log residuals).", "",
          "| target | iv_cal | raw IV | reg_clip | reg | hmm3 |", "|---|---|---|---|---|---|"]
    for n in ("vol5", "vol20"):
        x = ph["iv_cal"][n]
        L.append(f"| {n} | " + " | ".join(f"{x[k]['r2']:+.2f}" for k in ("iv_cal", "naive_iv", "reg_clip", "reg", "hmm3")) + " |")
    L += ["", "By half-year (R^2):", "", "| target | model | " + " | ".join(ph["iv_cal"]["vol20"]["iv_cal"]["by_half"]) + " |",
          "|---" * (len(ph["iv_cal"]["vol20"]["iv_cal"]["by_half"]) + 2) + "|"]
    for n in ("vol5", "vol20"):
        for k in ("iv_cal", "naive_iv", "reg_clip", "hmm3"):
            L.append(f"| {n} | {k} | " + " | ".join(f"{v:+.2f}" for v in ph["iv_cal"][n][k]["by_half"].values()) + " |")
    L += ["", "## Results: probabilities", "",
          "Each cell is Brier skill vs the base rate / Brier skill vs the random walk at current vol / AUC.", "",
          "| target | window | eff n | events | " + " | ".join(VARIANTS) + " | random walk (vs base / AUC) |",
          "|---" * (len(VARIANTS) + 5) + "|"]
    for n, e in ev["bin"].items():
        first = next(v for v in VARIANTS if v in e)

        def cell(b):
            rw = f"{b['bss_rw']:+.3f}" if "bss_rw" in b else "-"
            auc = f"{b['auc']:.2f}" if b.get("auc") is not None else "-"
            return f"{b['bss_base']:+.3f} / {rw} / {auc}"
        rw = e.get("naive_rw")
        rwc = f"{rw['bss_base']:+.3f} / {rw['auc']:.2f}" if rw else "-"
        L.append(f"| {n} | {e['window']} | {e['eff_n']} | {e[first]['events']} | " +
                 " | ".join(cell(e[v]) if v in e else "na" for v in VARIANTS) + f" | {rwc} |")
    L += ["", "reg_iv is scored on its own, shorter window, because it needs 60 IV-era training rows. Base-rate skill 90% "
          "block-bootstrap intervals for reg: " + "; ".join(f"{n} [{x['ci90'][0]:+.3f}, {x['ci90'][1]:+.3f}]"
                                                             for n, x in ph["bss_ci"].items()) + ".", "",
          "Reliability of the random walk at current vol for dd5_20. It over-predicts, most of all at the highest vol, "
          "which came after the drawdown had already happened:", "",
          "| bin | n | mean p | observed |", "|---|---|---|---|"]
    for r in ev["bin"]["dd5_20"]["naive_rw"]["reliability"]:
        L.append(f"| {r['bin']} | {r['n']} | {r['mean_p']:.3f} | {r['freq']:.3f} |")
    L += ["", "## What each regime label meant out of sample", "",
          "Labels are the argmax of the filtered probabilities, for hmm3 and hmm2.", "",
          "| model | label | sessions | rv20 now | next-20 vol | P(down 3% in 5) | P(down 5% in 20) | P(up over 20) | mean next-20 return |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, tab in ev["regimes"].items():
        for k, x in tab.items():
            L.append(f"| {m} | {k} | {x['sessions']} | {x['rv20_now_pct']:.1f}% | {x['next_vol20_pct']:.1f}% | {x['p_dd3_5']:.1%} | "
                     f"{x['p_dd5_20']:.1%} | {x['p_up20']:.0%} | {x['mean_next20_ret_pct']:+.2f}% |")
    L += ["", "The labels sort future vol and drawdowns because they sort current vol, and calibrated IV does that better. "
          "The stressed label's mean next-20 return is not lower, so the labels carry no direction signal.", "",
          "## How an LLM trader should use it", "",
          "- Read `spy_vol20_pct` and its 10-90% interval as the expected SPY volatility over the next month.",
          "- `size_scale_vs_typical` is typical vol divided by expected vol. To hold risk constant, scale gross exposure by "
          "roughly that factor. It is a number, not an order.",
          "- The drawdown probabilities are historical base rates (2023-26). They do not change with conditions, so a low "
          "number is not a sign of safety.",
          "- The regime label is a coarse description of current vol, not a forecast of direction.", "",
          "## Caveats", "",
          "- Three years of data with one big shock (April 2025). At h = 20 there are only about 21 independent periods.",
          "- The IV series starts 2024-10-17, so the calibrated IV point uses at most 2 years of residuals.",
          "- The universe is today's S&P 500 members, so it is survivor-biased. That affects breadth and dispersion "
          "(features of the losing variants), not SPY vol or IV.",
          "- Three post-hoc guards were added after seeing results, all in the conservative direction: the bootstrap check on "
          "dd3_5, clipped inputs for vol5, and the calibrated-IV bar.",
          "- Live: SPY IV comes from data/iv/SPY.json at the last close. If that file lacks the date, the gateway's delayed "
          "intraday implied_vol is used, and the output notes it.", "",
          "## Track record (tool description text)", "", TRACK_RECORD, ""]
    (OUT / "regime.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if "--live" in sys.argv:
        print(json.dumps(live(), indent=1))
    else:
        o = validate()
        print(json.dumps({"ship": o["ship"], "intervals": o["intervals"]}, indent=1))
