"""Shared, leakage-safe evaluation for the ML arm's research (every family uses it, so results compare).

Arrays are (symbols, steps): a step is a session (daily models) or a 30-minute bar (intraday models).
`labels[t]` is the step's date or timestamp string (YYYYMMDD...), used for the half-year breakdown.

Rules (fixed before any family runs):
  - walk_forward: a model tested from step T is fit only on steps t with t + horizon <= T - 1 (its targets are
    realized before the test starts); retrain every `retrain_every` steps on an expanding window.
  - score: rank IC per step; its t-stat on non-overlapping periods (every `horizon`-th step); top-minus-bottom
    decile spread per period, gross and net of costs (both legs fully turned over each period, cost_bps per
    round trip); long-only top-decile excess; IC per half-year.
  - PASS bar for shipping a return forecast: IC t >= 2.0, net decile spread > 0, and positive mean IC in at least
    2 of the 3 most recent half-years. Volatility/risk forecasts pass on out-of-sample R^2 above the naive baseline.
  - The universe is today's index members (survivor-biased), so every figure is optimistic.
"""

import numpy as np


def spearman(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 20:
        return np.nan
    ra, rb = a[ok].argsort().argsort(), b[ok].argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def walk_forward(features, target, steps_from, test_from, horizon, retrain_every, fit, predict, min_train=250):
    """features: callable(step_indices) -> (n_rows, n_features) with rows ordered step-major, symbol-minor.
    target: (symbols, steps). fit(X, y) -> model; predict(model, X) -> y_hat. Returns predictions (symbols, steps)."""
    nsym, nsteps = target.shape
    pred = np.full(target.shape, np.nan)
    last = nsteps - 1
    for T in range(test_from, last + 1, retrain_every):
        tr = np.arange(steps_from, T - horizon)          # target of step t ends at t + horizon <= T - 1
        if len(tr) < min_train:
            continue
        X, y = features(tr), target[:, tr].T.reshape(-1)
        ok = ~np.isnan(y) & ~np.all(np.isnan(X), axis=1)
        model = fit(X[ok], y[ok])
        te = np.arange(T, min(T + retrain_every, last + 1))
        pred[:, te] = np.asarray(predict(model, features(te))).reshape(len(te), nsym).T
    return pred


def score(pred, target, steps, horizon, labels, cost_bps=10.0):
    """Out-of-sample scorecard for a cross-sectional forecast over `steps`."""
    steps = [t for t in steps if np.sum(~np.isnan(pred[:, t]) & ~np.isnan(target[:, t])) >= 20]
    ic = np.array([spearman(pred[:, t], target[:, t]) for t in steps])
    per = steps[::horizon]
    ic_p = np.array([spearman(pred[:, t], target[:, t]) for t in per])
    spread, top = [], []
    for t in per:
        p, r = pred[:, t], target[:, t]
        ok = ~np.isnan(p) & ~np.isnan(r)
        lo, hi = np.quantile(p[ok], [0.1, 0.9])
        spread.append(np.mean(r[ok][p[ok] >= hi]) - np.mean(r[ok][p[ok] <= lo]))
        top.append(np.mean(r[ok][p[ok] >= hi]) - np.mean(r[ok]))
    spread, top = np.array(spread), np.array(top)
    cost = cost_bps / 1e4
    tstat = lambda x: float(np.nanmean(x) / (np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))))) if np.sum(~np.isnan(x)) > 2 else np.nan
    halves = {}
    for t, v in zip(steps, ic):
        lab = str(labels[t])
        key = lab[:4] + ("H1" if lab[4:6] <= "06" else "H2")
        halves.setdefault(key, []).append(v)
    by_half = {k: float(np.nanmean(v)) for k, v in sorted(halves.items())}
    recent = list(by_half.values())[-3:]
    res = {"steps": len(steps), "periods": len(per), "ic": float(np.nanmean(ic)), "ic_t": tstat(ic_p),
           "ic_pos": float(np.nanmean(ic > 0)), "spread_gross_pct": float(np.mean(spread) * 100), "spread_t": tstat(spread),
           "spread_net_pct": float((np.mean(spread) - 2 * cost) * 100), "top_decile_excess_pct": float(np.mean(top) * 100),
           "top_decile_net_pct": float((np.mean(top) - cost) * 100), "ic_by_half": by_half}
    res["pass"] = bool(res["ic_t"] >= 2.0 and res["spread_net_pct"] > 0 and sum(v > 0 for v in recent) >= 2)
    return res


def r2_oos(pred, target, naive, steps):
    """Out-of-sample R^2 of a forecast and of a naive baseline, over the same (symbol, step) cells."""
    p, y, n = pred[:, steps].ravel(), target[:, steps].ravel(), naive[:, steps].ravel()
    ok = ~np.isnan(p) & ~np.isnan(y) & ~np.isnan(n)
    y, p, n = y[ok], p[ok], n[ok]
    r2 = lambda f: float(1 - np.mean((y - f) ** 2) / np.var(y))
    return {"r2_model": r2(p), "r2_naive": r2(n), "cells": int(ok.sum()), "pass": r2(p) > r2(n)}
