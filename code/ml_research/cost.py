"""ML tool "cost": expected round-trip trading cost and liquidity for a proposed order.

cost (bps, round trip) = spread + 2 x impact (entry and exit) + commission (both legs)
  spread     next-session time-average quoted spread, forecast by a model validated walk-forward against the naive
             "last session's spread" (data/bidask: daily BID_ASK bars, open = time-average bid, close = time-average ask);
             live, blended with the delayed quote's bid/ask (rule chosen on our own quote snapshots, see validate()).
  impact     square-root law per leg, k * sigma_daily * sqrt(notional / ADV$); k is a literature prior (median 0.5,
             band 0.2-1.0), NOT fitted: our paper fills are too small, and IBKR paper fills don't move the market.
  commission $0.005/share, $1 minimum, 1% of trade value maximum, per order; two orders per round trip.

    .venv\\Scripts\\python.exe -m ml_research.cost validate     # walk-forward + calibration -> runs/ml/research/cost.{json,md}
    .venv\\Scripts\\python.exe -m ml_research.cost test         # live() on a fake gateway built from cached bars

Pre-registration: runs/ml/research/cost_prereg.md. Fill/quote evidence: ml_research/cost_fills.py.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "ml" / "research"
BIDASK = ROOT / "data" / "bidask"
MODEL = OUT / "cost_model.json"
OOS = OUT / "cost_oos.npz"
TEST_FROM = "20250102"
RETRAIN = 21
MIN_TRAIN = 20
Q_LO, Q_HI = 0.10, 0.90
K_IMPACT = {"lo": 0.2, "med": 0.5, "hi": 1.0}
COMM_SHARE, COMM_MIN, COMM_MAX_FRAC = 0.005, 1.0, 0.01
DEFAULT_NOTIONAL = 3000.0          # 10% of a $30k sleeve, used when only symbols are given
TIERS = (("A", 500e6, 3.0), ("B", 50e6, 10.0), ("C", 5e6, 50.0))

F_HAR = ["ls1", "ls5", "ls20"]
F_X = ["lmid", "ldv", "lvol", "lpk", "lrv", "etf"]
FEATS = {"har": F_HAR, "harx": F_HAR + F_X, "gbm": F_HAR + F_X, "cold": F_X, "harx_cond": F_HAR + F_X,
         "price_only": ["lmid"]}

TRACK_RECORD = (
    "Round-trip cost = time-average quoted spread + square-root market impact (entry and exit) + commissions ($0.005 a "
    "share, $1 minimum per order). Spread forecast, out of sample 2 Jan 2025 to 16 Oct 2026 on the 523 cached S&P 500 "
    "stocks and ETFs (234k symbol-days): log R2 0.970, vs 0.962 for yesterday's spread and 0.966 for a 5-day average; "
    "median error 0.62 bps vs 0.69. Its 10-90% interval held the realized spread 79.7% of the time (77-83% across "
    "liquidity quintiles). Spreads are very persistent, so this is careful bookkeeping more than prediction. For symbols "
    "outside that universe (scanner small caps) the spread comes from the live quote during market hours, with an "
    "interval taken from 214 of our own quote snapshots. Without a quote it comes from price, volume and volatility, and "
    "on 9 scanner-name quotes that estimate was off by a median factor of 1.6. Impact is a textbook prior (k 0.5, range "
    "0.2-1.0), not fitted: our 114 week-1 paper orders were too small (median $4.3k), and paper fills don't move prices. "
    "Their measured slippage against the arrival price, +1.2 bps per leg (SE 1.8), matches the model's average "
    "half-spread of 1.5 bps.")


# ------------------------------------------------------------------ data
def load_bidask(sym):
    p = BIDASK / f"{sym.replace(' ', '_')}.json"
    if not p.exists():
        return {}
    out = {}
    for b in json.loads(p.read_text(encoding="utf-8")).get("bars", []):
        bid, ask = b.get("open"), b.get("close")
        if bid and ask and bid > 0 and ask > 0:
            out[str(b["date"])[:8]] = (float(bid), float(ask))
    return out


def spread_bps(bid, ask):
    mid = (bid + ask) / 2
    s = (ask - bid) / mid * 1e4
    return s if 0 < s <= 2000 else np.nan


def _rmean(x, n, minv):
    """Trailing nanmean over n columns ending at t (inclusive), needing >= minv valid values."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        w = x[:, max(0, t - n + 1):t + 1]
        cnt = np.sum(~np.isnan(w), 1)
        with np.errstate(all="ignore"):
            m = np.nanmean(w, 1)
        out[:, t] = np.where(cnt >= minv, m, np.nan)
    return out


def _rstd(x, n, minv):
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        w = x[:, max(0, t - n + 1):t + 1]
        cnt = np.sum(~np.isnan(w), 1)
        with np.errstate(all="ignore"):
            m = np.nanstd(w, 1, ddof=1)
        out[:, t] = np.where(cnt >= minv, m, np.nan)
    return out


def feature_arrays(S, mid, o, h, l, c, v, lr, etf):
    """All features as (symbols, steps) arrays from arrays through each column's close. S = spread bps, mid = bid/ask
    mid (unadjusted), the rest daily bars. Shared by validate() and live() so both compute the same thing."""
    with np.errstate(all="ignore"):
        dv = c * v
        F = {"ls1": np.log(S), "ls5": np.log(_rmean(S, 5, 3)), "ls20": np.log(_rmean(S, 20, 10)),
             "lmid": np.log(mid), "ldv": np.log(_rmean(dv, 20, 10)), "lvol": np.log(_rstd(lr, 20, 10)),
             "lpk": 0.5 * np.log(_rmean(np.log(h / l) ** 2 / (4 * np.log(2)), 5, 3)),
             "lrv": np.log(v / _rmean(v, 20, 10)), "etf": np.broadcast_to(np.asarray(etf, float)[:, None], S.shape).copy()}
    for k in F:
        F[k] = np.where(np.isfinite(F[k]), F[k], np.nan)
    return F


def build():
    """Arrays on Data's calendar plus one extra column for bid/ask sessions after the last cached daily bar."""
    import trader
    d = trader.Data()
    n, m = len(d.sym), len(d.dates)
    later = sorted({dt for s in d.sym for dt in load_bidask(s) if dt > d.dates[-1]})[:1]
    cal = list(d.dates) + later
    col = {x: i for i, x in enumerate(cal)}
    S = np.full((n, len(cal)), np.nan)
    mid = np.full((n, len(cal)), np.nan)
    have = np.zeros(n, bool)
    for j, s in enumerate(d.sym):
        ba = load_bidask(s)
        have[j] = bool(ba)
        for dt, (b, a) in ba.items():
            i = col.get(dt)
            if i is not None:
                S[j, i], mid[j, i] = spread_bps(b, a), (a + b) / 2
    etf = [s in d.etf for s in d.sym]
    F = feature_arrays(S[:, :m], mid[:, :m], d.o, d.h, d.l, d.c, d.v, d.lr, etf)
    y = np.full((n, m), np.nan)
    with np.errstate(all="ignore"):
        y[:, :-1] = np.log(S[:, 1:m])
        if later:
            y[:, -1] = np.log(S[:, m])          # session after the last cached bar (target only)
    return d, cal, S, F, y, have


def rows(F, names, idx):
    return np.column_stack([F[k][:, idx].T.reshape(-1) for k in names])


# ------------------------------------------------------------------ models
def ols_fit(X, y):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    b = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return {"kind": "ols", "b": b.tolist()}


def gbm_fit(X, y):
    import lightgbm as lgb
    ok = np.isfinite(y)
    p = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    return {"kind": "gbm", "m": lgb.train(p, lgb.Dataset(X[ok], y[ok]), num_boost_round=300)}


def predict(model, X):
    if model["kind"] == "ols":
        b = np.asarray(model["b"])
        p = b[0] + np.nan_to_num(X) @ b[1:]
        return np.where(np.isfinite(X).all(1), p, np.nan)
    return model["m"].predict(X)


def residual_quantiles(model, X, y, ldv=None, edges=None):
    """In-sample (training-window) residual quantiles, unconditional or by dollar-volume bin."""
    r = y - predict(model, X)
    ok = np.isfinite(r)
    q = {"all": np.quantile(r[ok], [Q_LO, Q_HI]).tolist()}
    if edges is not None:
        b = np.digitize(ldv, edges)
        for k in range(len(edges) + 1):
            sel = ok & (b == k)
            q[str(k)] = np.quantile(r[sel], [Q_LO, Q_HI]).tolist() if sel.sum() > 50 else q["all"]
    return q


def interval(q, pred, ldv=None, edges=None):
    if edges is None or ldv is None:
        return pred + q["all"][0], pred + q["all"][1]
    b = np.digitize(np.nan_to_num(ldv, nan=-99), edges)
    lo = np.array([q[str(k)][0] for k in b]) if np.ndim(b) else q[str(int(b))][0]
    hi = np.array([q[str(k)][1] for k in b]) if np.ndim(b) else q[str(int(b))][1]
    return pred + lo, pred + hi


def walk(F, y, names, kind, t0, start, cond=False):
    """Same rule as ml_eval.walk_forward (fit on steps t with t + 1 <= T - 1, monthly expanding retrains), but also
    returns the interval bounds from each training window's residual quantiles."""
    nsym, nsteps = y.shape
    pred, lo, hi = (np.full(y.shape, np.nan) for _ in range(3))
    for T in range(t0, nsteps, RETRAIN):
        tr = np.arange(start, T - 1)
        if len(tr) < MIN_TRAIN:
            continue
        X, yy = rows(F, names, tr), y[:, tr].T.reshape(-1)
        ok = np.isfinite(yy) & np.isfinite(X).all(1) if kind == "ols" else np.isfinite(yy) & ~np.all(np.isnan(X), 1)
        X, yy = X[ok], yy[ok]
        model = ols_fit(X, yy) if kind == "ols" else gbm_fit(X, yy)
        edges = None
        ldv_tr = rows(F, ["ldv"], tr)[ok, 0]
        if cond:
            edges = np.nanquantile(ldv_tr, [0.2, 0.4, 0.6, 0.8]).tolist()
        q = residual_quantiles(model, X, yy, ldv_tr, edges)
        te = np.arange(T, min(T + RETRAIN, nsteps))
        Xt = rows(F, names, te)
        p = predict(model, Xt)
        a, b = interval(q, p, rows(F, ["ldv"], te)[:, 0], edges)
        for arr, val in ((pred, p), (lo, a), (hi, b)):
            arr[:, te] = np.asarray(val).reshape(len(te), nsym).T
    return pred, lo, hi


def fit_final(F, y, names, kind, start, cond=False):
    tr = np.arange(start, y.shape[1])
    X, yy = rows(F, names, tr), y[:, tr].T.reshape(-1)
    ok = np.isfinite(yy) & np.isfinite(X).all(1)
    model = ols_fit(X[ok], yy[ok])
    ldv = rows(F, ["ldv"], tr)[ok, 0]
    edges = np.nanquantile(ldv, [0.2, 0.4, 0.6, 0.8]).tolist() if cond else None
    return {**model, "names": names, "q": residual_quantiles(model, X[ok], yy[ok], ldv, edges), "edges": edges,
            "rows": int(ok.sum())}


# ------------------------------------------------------------------ scoring
def card(pred, lo, hi, y, naive, steps, ldv, dates):
    P, Y, N = pred[:, steps], y[:, steps], naive[:, steps]
    L, H, D = lo[:, steps], hi[:, steps], ldv[:, steps]
    ok = np.isfinite(P) & np.isfinite(Y) & np.isfinite(N)
    p, yy, nn, l, h = P[ok], Y[ok], N[ok], L[ok], H[ok]
    r2 = lambda f: float(1 - np.mean((yy - f) ** 2) / np.var(yy))
    mae = lambda f: float(np.median(np.abs(np.exp(f) - np.exp(yy))))
    res = {"cells": int(ok.sum()), "r2_log": r2(p), "r2_naive": r2(nn), "medae_bps": mae(p), "medae_naive_bps": mae(nn),
           "median_spread_bps": float(np.median(np.exp(yy))),
           "coverage_10_90": float(np.mean((yy >= l) & (yy <= h))), "below_q10": float(np.mean(yy < l)),
           "above_q90": float(np.mean(yy > h)), "mean_width_ratio": float(np.mean(np.exp(h - l)))}
    # by dollar-volume quintile (cross-sectional within the OOS cells)
    d = D[ok]
    edges = np.nanquantile(d, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(np.nan_to_num(d, nan=-99), edges)
    res["by_dv_quintile"] = {int(k): {"n": int((b == k).sum()), "r2_log": float(1 - np.mean((yy[b == k] - p[b == k]) ** 2) / np.var(yy[b == k])),
                                      "r2_naive": float(1 - np.mean((yy[b == k] - nn[b == k]) ** 2) / np.var(yy[b == k])),
                                      "median_spread_bps": float(np.median(np.exp(yy[b == k]))),
                                      "coverage": float(np.mean((yy[b == k] >= l[b == k]) & (yy[b == k] <= h[b == k])))}
                             for k in range(5)}
    lab = np.broadcast_to(np.array([dates[t] for t in steps])[None, :], P.shape)[ok]
    half = np.array([x[:4] + ("H1" if x[4:6] <= "06" else "H2") for x in lab])
    res["by_half"] = {k: {"n": int((half == k).sum()), "r2_log": float(1 - np.mean((yy[half == k] - p[half == k]) ** 2) / np.var(yy[half == k])),
                          "r2_naive": float(1 - np.mean((yy[half == k] - nn[half == k]) ** 2) / np.var(yy[half == k])),
                          "coverage": float(np.mean((yy[half == k] >= l[half == k]) & (yy[half == k] <= h[half == k])))}
                      for k in sorted(set(half))}
    return res


# ------------------------------------------------------------------ cost arithmetic (shared by live and sim)
def tier(adv_usd, spread):
    for name, a, s in TIERS:
        if adv_usd is not None and adv_usd >= a and spread is not None and spread <= s:
            return name
    return "D"


def commission(qty, price):
    per = min(max(COMM_MIN, COMM_SHARE * qty), COMM_MAX_FRAC * qty * price)
    return 2 * per


def order_cost(price, qty, spread_med, spread_lo, spread_hi, adv_usd, sigma_d, exp_move):
    """Round-trip cost for qty shares at price. sigma_d and exp_move are daily fractions."""
    notional = qty * price
    part = notional / adv_usd if adv_usd and adv_usd > 0 else float("nan")
    imp = {k: (1e4 * kk * sigma_d * math.sqrt(part) if np.isfinite(part) and sigma_d else float("nan"))
           for k, kk in K_IMPACT.items()}
    comm = commission(qty, price) / notional * 1e4 if notional > 0 else float("nan")
    nz = lambda x: x if np.isfinite(x) else 0.0
    med = spread_med + 2 * nz(imp["med"]) + comm
    lo = spread_lo + 2 * nz(imp["lo"]) + comm
    hi = spread_hi + 2 * nz(imp["hi"]) + comm
    r = lambda x, k=1: None if x is None or not np.isfinite(x) else round(float(x), k)
    em = exp_move * 1e4 if exp_move and np.isfinite(exp_move) else float("nan")
    return {"quantity": int(qty), "notional_usd": r(notional, 0),
            "cost_bps": r(med), "cost_bps_10": r(lo), "cost_bps_90": r(hi), "cost_usd": r(med * notional / 1e4, 2),
            "spread_bps": r(spread_med), "spread_bps_10": r(spread_lo), "spread_bps_90": r(spread_hi),
            "impact_bps_roundtrip": r(2 * imp["med"]), "commission_bps_roundtrip": r(comm),
            "pct_adv": r(100 * part, 4), "adv_usd_m": r(adv_usd / 1e6 if adv_usd else float("nan"), 1),
            "tier": tier(adv_usd, spread_med), "exp_move_bps": r(em, 0),
            "breakeven_share_of_daily_move": r(med / em if em and np.isfinite(em) else float("nan"), 2)}


# ------------------------------------------------------------------ validate
def validate():
    t_start = time.time()
    d, cal, S, F, y, have = build()
    m = len(d.dates)
    t0 = next(i for i, x in enumerate(d.dates) if x >= TEST_FROM)
    first_ba = min(i for i in range(m) if np.isfinite(S[:, i]).sum() > 100)
    start = first_ba + 19                      # ls20 needs 10 valid of 20; give the window its first 20 sessions
    steps = list(range(t0, m))
    naive, mean5 = F["ls1"], F["ls5"]
    print(f"symbols with bid/ask {have.sum()}/{len(have)}; bid/ask from {d.dates[first_ba]}; train from {d.dates[start]}; "
          f"test {d.dates[t0]}..{cal[-1]} (target session)", flush=True)
    results, preds = {}, {}
    variants = (("har", "ols", False), ("harx", "ols", False), ("gbm", "gbm", False), ("cold", "ols", False),
                ("harx_cond", "ols", True), ("price_only", "ols", False))
    for name, kind, cond in variants:
        preds[name] = walk(F, y, FEATS[name], kind, t0, start, cond)
        print(f"  fitted {name} ({time.time() - t_start:.0f}s)", flush=True)
    common = np.logical_and.reduce([np.isfinite(preds[k][0]) for k in preds]) & np.isfinite(naive) & np.isfinite(mean5)
    print(f"common OOS cells: {int(common[:, steps].sum())}", flush=True)
    for name, kind, cond in variants:
        p, lo, hi = (np.where(common, a, np.nan) for a in preds[name])
        c = card(p, lo, hi, y, naive, steps, F["ldv"], d.dates)
        c["vs_mean5"] = card(p, lo, hi, y, mean5, steps, F["ldv"], d.dates)["r2_naive"]
        results[name] = c
        print(f"{name:11s} R2 {c['r2_log']:.3f} naive {c['r2_naive']:.3f} mean5 {c['vs_mean5']:.3f} medAE {c['medae_bps']:.2f} "
              f"vs {c['medae_naive_bps']:.2f} bps cov {c['coverage_10_90']:.3f} ({time.time() - t_start:.0f}s)", flush=True)
    # cold vs price-only on the same cells
    results["cold"]["r2_price_only"] = results["price_only"]["r2_log"]
    for name in ("har", "harx", "gbm", "harx_cond"):
        r = results[name]
        r["pass"] = bool(r["r2_log"] > r["r2_naive"] and r["medae_bps"] < r["medae_naive_bps"])
    results["cold"]["pass"] = bool(results["cold"]["r2_log"] > results["price_only"]["r2_log"])
    cand = [k for k in ("har", "harx", "gbm", "harx_cond") if results[k]["pass"]]
    best = max(cand, key=lambda k: results[k]["r2_log"]) if cand else None
    if best and best != "har" and results[best]["r2_log"] - results["har"]["r2_log"] < 0.005 and results["har"]["pass"]:
        best = "har"
    # live and the simulator serve an OLS variant; if gbm were best, its OLS twin harx would be served (and said)
    serve = best if best in ("har", "harx", "harx_cond") else ("harx" if results["harx"]["pass"] else None)
    cond_serve = serve == "harx_cond" or (serve == "harx" and _better_calibrated(results["harx_cond"], results["harx"]))
    names = FEATS[serve] if serve else FEATS["harx"]
    final = {"spread": fit_final(F, y, names, "ols", start, cond=cond_serve),
             "cold": fit_final(F, y, FEATS["cold"], "ols", start, cond=False),
             "served_variant": serve, "best_variant": best, "conditional_intervals": cond_serve,
             "fit_through": cal[-1]}
    final["spread"]["variant"] = serve
    # OOS npz for the simulator: the served variant's walk-forward predictions (cond intervals if chosen)
    p, lo, hi = preds["harx_cond" if cond_serve and serve in ("harx", "harx_cond") else serve]
    with np.errstate(all="ignore"):
        dvol = np.exp(F["ldv"])
        sig = np.exp(F["lvol"])
    np.savez_compressed(OOS, symbols=np.array(d.sym), dates=np.array(d.dates), spread=np.exp(p), spread_lo=np.exp(lo),
                        spread_hi=np.exp(hi), adv_usd=dvol, sigma_d=sig, close=d.c,
                        note=np.array("column t = forecast for session t+1 from data through the close of t"))
    MODEL.write_text(json.dumps(final, indent=1), encoding="utf-8")
    # fill and snapshot evidence
    from ml_research import cost_fills
    ev = cost_fills.evidence(d, S, cal, preds["harx_cond" if cond_serve and serve in ("harx", "harx_cond") else serve], F,
                             cold=preds["cold"])
    ev["thin"] = cost_fills.thin_check(d, final["cold"])
    for k in ("rule", "rule_cold", "rule_interval", "rule_cold_interval"):
        if k in ev["snapshot"]:
            final["snapshot_" + k] = ev["snapshot"][k]
    MODEL.write_text(json.dumps(final, indent=1), encoding="utf-8")
    out = {"prereg": "runs/ml/research/cost_prereg.md", "test": f"{d.dates[t0]}..{cal[-1]}", "train_from": d.dates[start],
           "symbols_with_bidask": int(have.sum()), "results": results, "best": best, "served": serve,
           "conditional_intervals": cond_serve, "evidence": ev, "impact_k": K_IMPACT}
    (OUT / "cost.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    at = sim_provider(d)
    ex = [{"symbol": "SPY", "quantity": 4}, {"symbol": "XLE", "quantity": 10}, {"symbol": "XLE", "quantity": 100},
          {"symbol": "XLV", "quantity": 120}, {"symbol": "META", "quantity": 30}, {"symbol": "MRNA", "quantity": 50},
          {"symbol": "ERIE", "quantity": 125}, {"symbol": "FXE", "quantity": 30}]
    rows_ = [(at(len(d.dates), [o]) or (None, {})) for o in ex]       # one call per order: rows are keyed by symbol
    out["examples"] = {"as_of": next((r[0] for r in rows_ if r[0]), None),
                       "rows": [dict(symbol=o["symbol"], **r[1].get(o["symbol"], {})) for o, r in zip(ex, rows_)]}
    (OUT / "cost.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    write_md(out)
    print(json.dumps(ev, indent=1, default=float)[:3000])
    print(f"done in {time.time() - t_start:.0f}s")
    return out


def _better_calibrated(a, b):
    dev = lambda r: max(abs(v["coverage"] - 0.8) for v in r["by_dv_quintile"].values())
    return dev(a) < dev(b)


# ------------------------------------------------------------------ live
def _model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _series_from_bars(rows_):
    o = np.array([[r["open"] for r in rows_]], float)
    h = np.array([[r["high"] for r in rows_]], float)
    l = np.array([[r["low"] for r in rows_]], float)
    c = np.array([[r["close"] for r in rows_]], float)
    v = np.array([[max(float(r.get("volume") or 0), 0) for r in rows_]], float)
    with np.errstate(all="ignore"):
        lr = np.diff(np.log(c), axis=1, prepend=np.nan)
    return o, h, l, c, v, lr


def _spread_history(gw, sym, dates):
    """Daily time-average spreads for sym on `dates`: from the gateway (BID_ASK bars) if it can, else the cached file."""
    ba, src = {}, None
    try:
        rows_ = gw.bars(sym, duration="2 M", bar_size="1 day", what="BID_ASK")
        ba = {str(r["date"])[:8]: (float(r["open"]), float(r["close"])) for r in rows_ if r.get("open") and r.get("close")}
        src = "gateway"
    except Exception:
        pass
    if not ba:
        ba, src = load_bidask(sym), "cache"
    S = np.array([[spread_bps(*ba[x]) if x in ba else np.nan for x in dates]])
    M = np.array([[sum(ba[x]) / 2 if x in ba else np.nan for x in dates]])
    return S, M, src


def _expected_move(sym, sigma_d):
    try:
        import forecast
        got = forecast.live_provider()
        if got and sym in got[1] and got[1][sym].get("absr1_pct"):
            return got[1][sym]["absr1_pct"] / 100, "forecast absr1"
    except Exception:
        pass
    return (math.sqrt(2 / math.pi) * sigma_d if sigma_d else float("nan")), "sqrt(2/pi) x 20-day vol"


def live(gw, symbols=None, orders=None, **kw):
    """Cost of proposed orders. orders: [{symbol, quantity} or {symbol, notional}]; or symbols only (costed at
    DEFAULT_NOTIONAL). Returns {"orders": [...], "track_record": TRACK_RECORD, "method": ...}."""
    M = _model()
    if orders is None:
        orders = [{"symbol": s, "notional": kw.get("notional", DEFAULT_NOTIONAL)} for s in (symbols or [])]
    if isinstance(symbols, str):                         # C2: input hardening
        symbols = [symbols]
    orders = [o for o in orders if isinstance(o, dict) and str(o.get("symbol") or "").strip()]
    orders = [dict(o, symbol=str(o["symbol"]).upper().strip()) for o in orders]
    syms = list(dict.fromkeys(o["symbol"] for o in orders))
    try:
        quotes = {q.get("symbol", "").upper(): q for q in gw.quotes(syms)}
    except Exception:
        quotes = {}
    per = {}
    for s in syms:
        q = quotes.get(s, {})
        note = []
        try:
            import bars as _bars                     # C1: drop today's partial daily bar (as bars.completed does)
            bars_ = [r for r in _bars.completed(gw.bars(s, duration="3 M", bar_size="1 day")) if r.get("close")]
        except Exception as e:
            bars_ = []
            note.append(f"no daily bars ({str(e)[:60]})")
        if len(bars_) < 12:
            per[s] = {"error": "not enough daily bars to size liquidity", "note": "; ".join(note)}
            continue
        dates = [str(r["date"]).replace("-", "")[:8] for r in bars_]
        o, h, l, c, v, lr = _series_from_bars(bars_)
        S, mid, src = _spread_history(gw, s, dates)
        if np.isnan(mid[0, -1]):
            mid[0, -1] = c[0, -1]
        etf = _is_etf(s)
        F = feature_arrays(S, mid, o, h, l, c, v, lr, [etf])
        sm = M["spread"]
        use_hist = np.isfinite(F["ls1"][0, -1]) and np.isfinite(F["ls20"][0, -1]) and np.isfinite(S[0, -3:]).sum() >= 2
        mdl = sm if use_hist else M["cold"]
        x = np.array([[F[k][0, -1] for k in mdl["names"]]])
        p = float(predict(mdl, x)[0])
        lo, hi = interval(mdl["q"], p, F["ldv"][0, -1], mdl["edges"])
        source = f"model {mdl.get('variant') or 'cold'} (bid/ask history: {src})" if use_hist else "cold model (no bid/ask history)"
        if not np.isfinite(p):
            per[s] = {"error": "could not compute spread features", "note": "; ".join(note)}
            continue
        # live quote blend (rule fixed in validate(): see cost.json evidence.snapshot.rule)
        bid, ask = q.get("bid"), q.get("ask")
        qs = spread_bps(float(bid), float(ask)) if bid and ask and bid > 0 and ask > bid else float("nan")
        key = "snapshot_rule" if use_hist else "snapshot_rule_cold"
        rule = M.get(key, "model")
        if np.isfinite(qs) and not _regular_hours(kw.get("now")):
            note.append(f"quote spread {qs:.1f} bps not used: outside regular hours (pre-market spreads run ~3x the session average)")
        elif np.isfinite(qs) and rule != "model":
            lq = math.log(qs)
            p = lq if rule == "quote" else 0.5 * (lq + p)
            q10, q90 = M[key + "_interval"]
            lo, hi = p + q10, p + q90
            source += f" + delayed quote ({rule}; interval from our quote snapshots)"
        price = float(q.get("last") or q.get("ask") or c[0, -1])
        sigma_d = float(np.exp(F["lvol"][0, -1])) if np.isfinite(F["lvol"][0, -1]) else float("nan")
        em, em_src = _expected_move(s, sigma_d)
        per[s] = {"price": price, "spread": (math.exp(p), math.exp(lo), math.exp(hi)), "adv": float(np.exp(F["ldv"][0, -1])),
                  "sigma_d": sigma_d, "em": em, "quote_spread_bps": None if not np.isfinite(qs) else round(qs, 1),
                  "source": source, "move_source": em_src, "bars_through": dates[-1], "note": "; ".join(note)}
    out = []
    for o in orders:
        s, info = o["symbol"], per.get(o["symbol"], {})
        if "error" in info:
            out.append({"symbol": s, **info})
            continue
        price = info["price"]
        try:
            qty = int(o.get("quantity") or max(1, round(float(o.get("notional", DEFAULT_NOTIONAL)) / price)))
        except (TypeError, ValueError):
            out.append({"symbol": s, "error": "quantity or notional is not a number"})
            continue
        if qty == 0:
            out.append({"symbol": s, "error": "quantity is zero"})
            continue
        r = order_cost(price, abs(qty), *info["spread"], info["adv"], info["sigma_d"], info["em"])
        out.append({"symbol": s, "price": round(price, 2), **r, "quote_spread_bps": info["quote_spread_bps"],
                    "spread_source": info["source"], "move_source": info["move_source"], "bars_through": info["bars_through"],
                    **({"note": info["note"]} if info["note"] else {})})
    return {"orders": out, "track_record": TRACK_RECORD,
            "method": "round trip = spread + 2 x sqrt-law impact (k 0.5, band 0.2-1.0, not fitted) + 2 commissions"}


def _regular_hours(now=None):
    """True between 09:30 and 16:00 ET on a weekday (now: an aware datetime, default the clock)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    t = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    return t.weekday() < 5 and "09:30" <= t.strftime("%H:%M") < "16:00"


_ETFS = None


def _is_etf(s):
    global _ETFS
    if _ETFS is None:
        try:
            import trader
            _ETFS = set(trader.etfs())
        except Exception:
            _ETFS = set()
    return s in _ETFS


# ------------------------------------------------------------------ simulator
def sim_provider(data):
    """A function at(d, orders=None) -> (as_of, {symbol: row}) as of the close of d-1, from strictly out-of-sample
    walk-forward spread forecasts (cost_oos.npz column d-1 = forecast for session d). Without orders each symbol is
    costed at DEFAULT_NOTIONAL; orders = [{symbol, quantity}] costs those."""
    z = np.load(OOS, allow_pickle=True)
    syms, dates = [str(s) for s in z["symbols"]], [str(x) for x in z["dates"]]
    col, row = {x: i for i, x in enumerate(dates)}, {s: j for j, s in enumerate(syms)}
    sp, lo, hi, adv, sig = z["spread"], z["spread_lo"], z["spread_hi"], z["adv_usd"], z["sigma_d"]
    try:
        r = np.load(OUT / "risk_oos.npz", allow_pickle=True)
        rcol = {str(x): i for i, x in enumerate(r["dates"])}
        rrow = {str(s): j for j, s in enumerate(r["symbols"])}
        absr = r["absr1"]
    except Exception:
        absr = None

    def at(d, orders=None):
        as_of = data.dates[d - 1]
        i = col.get(as_of)
        if i is None:
            return None
        want = orders or [{"symbol": s} for s in syms]
        out = {}
        for o in want:
            s = str(o["symbol"]).upper()
            j, k = row.get(s), data.k.get(s)
            if j is None or k is None or not np.isfinite(sp[j, i]) or np.isnan(data.c[k, d - 1]):
                continue
            price = float(data.c[k, d - 1])
            em = math.sqrt(2 / math.pi) * sig[j, i]
            if absr is not None and s in rrow and as_of in rcol and np.isfinite(absr[rrow[s], rcol[as_of]]):
                em = float(absr[rrow[s], rcol[as_of]])
            qty = int(o.get("quantity") or max(1, round(DEFAULT_NOTIONAL / price)))
            out[s] = {"price": round(price, 2), **order_cost(price, abs(qty), float(sp[j, i]), float(lo[j, i]),
                                                              float(hi[j, i]), float(adv[j, i]), float(sig[j, i]), em)}
        return (as_of, out) if out else None
    return at


# ------------------------------------------------------------------ fake gateway test
class FakeGW:
    """Cached daily bars as of the last close; quotes from the last bar with a spread from the cached bid/ask."""

    def __init__(self, data, extra=None):
        self.d, self.extra = data, extra or {}

    def quotes(self, symbols):
        out = []
        for s in symbols:
            j = self.d.k.get(s)
            if j is None:
                if s in self.extra:
                    out.append(self.extra[s]["quote"])
                else:
                    out.append({"symbol": s, "error": "no data"})
                continue
            c = float(self.d.c[j, -1])
            ba = load_bidask(s)
            last = ba.get(self.d.dates[-1])
            bid, ask = (last if last else (c * 0.9999, c * 1.0001))
            out.append({"symbol": s, "last": c, "bid": round(bid, 2), "ask": round(ask, 2), "close": float(self.d.c[j, -2])})
        return out

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", **kw):
        if what != "TRADES":
            raise ValueError("fake gateway: only TRADES bars")
        j = self.d.k.get(symbol)
        if j is None:
            if symbol in self.extra:
                return self.extra[symbol]["bars"]
            raise ValueError(f"200: No security definition has been found for {symbol}")
        D = self.d
        return [{"date": D.dates[i], "open": float(D.o[j, i]), "high": float(D.h[j, i]), "low": float(D.l[j, i]),
                 "close": float(D.c[j, i]), "volume": float(D.v[j, i])} for i in range(len(D.dates) - 63, len(D.dates))
                if not np.isnan(D.c[j, i])]


def test():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    import trader
    d = trader.Data()
    # a thin symbol absent from the cache: synthetic daily bars shaped like a $12 small cap trading ~$4M a day
    rng = np.random.default_rng(1)
    px = 12 * np.exp(np.cumsum(rng.normal(0, 0.06, 63)))
    bars_ = [{"date": d.dates[-63 + i], "open": p * 0.99, "high": p * 1.05, "low": p * 0.95, "close": p,
              "volume": float(rng.integers(200_000, 500_000))} for i, p in enumerate(px)]
    extra = {"THIN": {"quote": {"symbol": "THIN", "last": px[-1], "bid": round(px[-1] * 0.995, 2), "ask": round(px[-1] * 1.005, 2)},
                      "bars": bars_},
             "NOQ": {"quote": {"symbol": "NOQ", "error": "no data"}, "bars": bars_}}
    # a real scanner name absent from the cache: NBIZ's daily bars (data/bars) and the delayed quote rl1 saw on 15 Oct
    from ml_research.cost_fills import daily_bars
    nb = [b for b in daily_bars("NBIZ") if str(b["date"]) < "20261015"][-63:]
    extra["NBIZ"] = {"quote": {"symbol": "NBIZ", "last": 35.11, "bid": 35.08, "ask": 35.24}, "bars": nb}
    gw = FakeGW(d, extra)
    res = live(gw, orders=[{"symbol": "SPY", "quantity": 4}, {"symbol": "XLE", "quantity": 100},
                           {"symbol": "XLE", "notional": 650}, {"symbol": "ERIE", "notional": 30000},
                           {"symbol": "THIN", "quantity": 100}, {"symbol": "NOQ", "notional": 1200},
                           {"symbol": "NBIZ", "quantity": 50},
                           {"symbol": "ZZZZ", "quantity": 10}], now=datetime(2026, 10, 16, 11, 0, tzinfo=ET))
    pre = live(gw, symbols=["XLE", "NBIZ"], now=datetime(2026, 10, 16, 9, 0, tzinfo=ET))
    for r in res["orders"] + pre["orders"]:
        print(json.dumps(r))
    at = sim_provider(d)
    as_of, rows_ = at(len(d.dates) - 1, [{"symbol": "XLE", "quantity": 100}, {"symbol": "SPY", "quantity": 4}])
    print("sim", as_of, json.dumps(rows_))
    (OUT / "cost_live_test.json").write_text(json.dumps({"live_1100ET": res, "live_0900ET": pre, "sim": [as_of, rows_]},
                                                        indent=1, default=float), encoding="utf-8")
    return res


# ------------------------------------------------------------------ report
def write_md(out):
    R, ev = out["results"], out["evidence"]
    f2 = lambda x: "na" if x is None else f"{x:.3f}"
    L = ["# Tool \"cost\": expected round-trip trading cost and liquidity", "",
         f"*Written by `ml_research/cost.py validate`. Pre-registration: `runs/ml/research/cost_prereg.md`. Test "
         f"{out['test']} (target sessions), training from {out['train_from']}, {out['symbols_with_bidask']} symbols with "
         f"bid/ask data.*", "",
         "## Method", "",
         "Round-trip cost in bps = **spread** (one full time-average quoted spread: buy at the ask, sell at the bid) + "
         "**2 x impact** (square-root law per leg, k x daily vol x sqrt(order / ADV$)) + **commission** ($0.005/share, "
         "$1 minimum, 1% maximum, two orders). The spread is the only forecast. Its target is the next session's "
         "time-average relative spread (`data/bidask`: open = average bid, close = average ask). Features come from the "
         "close of t. The walk-forward retrains monthly, with an expanding window and a 1-session embargo. The 10-90% "
         "interval comes from the training window's residual quantiles in log space. Impact is not fitted: k = 0.5, "
         "band 0.2-1.0, a literature prior (see Caveats).", "",
         "## Spread forecast: all pre-registered variants (same out-of-sample cells)", "",
         "| variant | log R2 | naive (last spread) | 5-day mean | median abs err bps | naive | 10-90 coverage | below 10 / above 90 | pass |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in ("har", "harx", "gbm", "harx_cond", "cold", "price_only"):
        r = R[k]
        L.append(f"| {k} | {f2(r['r2_log'])} | {f2(r['r2_naive'])} | {f2(r.get('vs_mean5'))} | {r['medae_bps']:.2f} | "
                 f"{r['medae_naive_bps']:.2f} | {r['coverage_10_90']:.3f} | {r['below_q10']:.3f} / {r['above_q90']:.3f} | "
                 f"{r.get('pass', '-')} |")
    L += ["", f"Cells: {R['har']['cells']:,}; median spread {R['har']['median_spread_bps']:.1f} bps. `cold` uses no "
          f"spread history; its pre-registered bar is the price-only model (R2 {f2(R['price_only']['r2_log'])}).",
          f"**Served:** `{out['served']}` (best passing: `{out['best']}`), conditional intervals: {out['conditional_intervals']}. "
          "For symbols without bid/ask history, `cold` is used.", "",
          "### Calibration of the served model by dollar-volume quintile (0 = least liquid)", "",
          "| quintile | n | median spread bps | log R2 | naive R2 | 10-90 coverage |", "|---|---|---|---|---|---|"]
    srv = R[out["served"]]
    for q, v in srv["by_dv_quintile"].items():
        L.append(f"| {q} | {v['n']:,} | {v['median_spread_bps']:.1f} | {f2(v['r2_log'])} | {f2(v['r2_naive'])} | {v['coverage']:.3f} |")
    L += ["", "| half-year | n | log R2 | naive R2 | 10-90 coverage |", "|---|---|---|---|---|"]
    for h, v in srv["by_half"].items():
        L.append(f"| {h} | {v['n']:,} | {f2(v['r2_log'])} | {f2(v['r2_naive'])} | {v['coverage']:.3f} |")
    L += ["", "Cold model coverage by quintile: " + ", ".join(f"{q}: {v['coverage']:.2f}" for q, v in R["cold"]["by_dv_quintile"].items()), ""]
    sn = ev["snapshot"]
    L += ["## Live quote snapshots vs the session's average spread", ""]
    if "rth" in sn:
        e = sn["rth"]
        L += [f"There are {e['n']} delayed quote-tool snapshots taken during regular hours, on {e['symbols']} cached "
              f"symbols, {', '.join(e['dates'])}. The median absolute log error against that session's time-average "
              f"spread is: quote {e['median_abs_log_err']['quote']:.2f}, model {e['median_abs_log_err']['model']:.2f}, "
              f"blend (log-average) {e['median_abs_log_err']['blend']:.2f}. The median quote/actual ratio is "
              f"{e['median_quote_over_actual']:.2f} and the median model/actual ratio is {e['median_model_over_actual']:.2f}. "
              f"The model's 10-90 interval covered {e['model_interval_coverage']:.2f} of these snapshots. "
              f"**Live rule adopted for symbols with bid/ask history: `{sn['rule']}`** (the delayed quote's spread is still "
              "reported beside it)."]
        if "cold" in e:
            c = e["cold"]["median_abs_log_err"]
            L += ["", f"For symbols without bid/ask history (the `cold` model), the same snapshots give these errors: quote {c['quote']:.2f}, cold "
                  f"model {c['model']:.2f}, blend {c['blend']:.2f}. **Live rule: `{sn['rule_cold']}`.**"]
            if "rule_cold_interval" in sn:
                a, b = sn["rule_cold_interval"]
                L[-1] += (f" Its 10-90% interval is the snapshots' own spread of log(session average / estimate), "
                          f"[{a:+.2f}, {b:+.2f}]. In other words, the session average ran {math.exp(a):.2f}x to "
                          f"{math.exp(b):.2f}x the delayed quote.")
    if "premarket" in sn:
        e = sn["premarket"]
        L += ["", f"Pre-market snapshots (n {e['n']}) have quoted spreads {e['median_quote_over_actual']:.1f}x the "
              "session average, so live() uses the model alone for quotes stamped outside 09:30-16:00."]
    th = ev.get("thin", {})
    if th.get("n"):
        L += ["", f"**Scanner names outside the universe** (an out-of-distribution check of `cold`): {th['n']} snapshots on "
              f"{', '.join(th['symbols'])}. The median log(quote / predicted) is {th['median_log_ratio_quote_over_pred']:+.2f}, "
              f"the median absolute log error is {th['median_abs_log_err']:.2f}, and {th['inside_10_90']:.2f} of the "
              "snapshots fell inside the 10-90 interval.", "",
              "| date | time | sym | quoted bps | cold pred (10-90) | ADV $M |", "|---|---|---|---|---|---|"]
        for r in th["rows"]:
            L.append(f"| {r['date']} | {r['time']} | {r['symbol']} | {r['quote_spread_bps']} | {r['cold_pred_bps']} "
                     f"({r['cold_10']}-{r['cold_90']}) | {r['adv_usd_m']} |")
    fe = ev["fills"]
    st = lambda s: (f"n {s['n']}, mean {s['mean']:+.1f} (SE {s['se']:.1f}), median {s['median']:+.1f}, 10-90% "
                    f"{s['p10']:+.1f} to {s['p90']:+.1f}" if s.get("n", 0) > 1 else f"n {s.get('n', 0)}")
    L += ["", "## Our own paper fills (week 1, 14-16 Oct 2026)", "",
          f"There are {fe['fills']} fills in {fe['orders']} orders, on {fe['symbols']} symbols, with a median notional of "
          f"${fe['median_notional']:,.0f}. That is far too few, and far too small, to fit anything. These names are "
          f"outside the cache: {', '.join(fe['not_in_cache'])}.", "",
          f"- **Slippage vs arrival** (30-minute bar open within 3 minutes of the fill, signed so + = cost, bps per leg): "
          f"{st(fe['arrival_all'])}.",
          f"  - Opening fills: {st(fe['arrival_open'])}.",
          f"  - Intraday fills: {st(fe['arrival_intraday'])}.",
          f"  - The model's half-spread for the same fills: {st(fe['model_half_spread_on_filled'])}.",
          f"- **Slippage vs the delayed price the model saw** (decision price): {st(fe['decision'])}. This is mostly "
          "15 minutes of price drift, not cost.",
          f"- **Commission per order:** {st(fe['commission_bps'])} bps.", "",
          "| sleeve | orders | notional $ | commission $ | est. half-spread $ |", "|---|---|---|---|---|"]
    for k, v in fe["by_sleeve"].items():
        L.append(f"| {k} | {v['orders']} | {v['notional']:,.0f} | {v['commission_usd']:,.2f} | {v['est_half_spread_usd']:,.2f} |")
    exr = (out.get("examples") or {}).get("rows", [])
    if exr:
        L += ["", f"## Worked examples (simulator provider, as of the close of {out['examples']['as_of']})", "",
              "| order | notional $ | cost bps (10-90) | spread | impact | commission | % ADV | tier | expected daily move bps | cost / daily move |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in exr:
            if "cost_bps" not in r:
                continue
            L.append(f"| {r['symbol']} x{r['quantity']} | {r['notional_usd']:,.0f} | {r['cost_bps']} ({r['cost_bps_10']}-{r['cost_bps_90']}) | "
                     f"{r['spread_bps']} | {r['impact_bps_roundtrip']} | {r['commission_bps_roundtrip']} | {r['pct_adv']} | {r['tier']} | "
                     f"{r['exp_move_bps']} | {r['breakeven_share_of_daily_move']} |")
    L += ["", "## Conclusion", "", CONCLUSION, "", "## How an LLM trader should use it", "", USAGE, "", "## Caveats", "", CAVEATS,
          "", "## Track record line (for the tool description)", "", TRACK_RECORD, ""]
    (OUT / "cost.md").write_text("\n".join(L), encoding="utf-8")


CONCLUSION = """- **The spread model passes its pre-registered bar, but it adds little.** All four variants with spread history beat the naive last-session spread on both log R2 and median error. `har`, the simplest, is served under the tie rule: R2 0.970 vs 0.962, median error 0.62 vs 0.69 bps, and only 0.004 better than a plain 5-day average. Spreads are among the most persistent quantities in markets, and the R2 mostly reflects which names are wide, not day-to-day changes. LightGBM and the extra features (price, volume, volatility) add nothing.
- **The intervals are calibrated.** The 10-90% interval covers 79.7% overall, 76.8-83.2% by liquidity quintile, and 76.0-83.0% by half-year. The low end is 2025H1, which includes the April 2025 shock.
- **The cold model is good enough inside the universe, not outside it.** It uses no spread history and beats price-only by a wide margin (R2 0.77 vs 0.14). In-universe coverage is 0.80 overall but ranges from 0.65 (most liquid quintile) to 0.86, so it misses the strict "75-85% in every quintile" definition. On 9 quotes of scanner names outside the universe it was off by a median factor of 1.6, and only 5 of 9 fell inside its 10-90% interval. So live, for those names, the delayed quote's spread is used during market hours: on our snapshots it estimates the session average better than the cold model (error 0.18 vs 0.36).
- **What the tool really adds is arithmetic the LLMs were not doing.**
  - At the sleeves' order sizes, commission is usually the largest cost. A $650 order costs about 31 bps round trip, almost all of it the $1 minimum. A $3k SPY order costs 6.7 bps, and a $6.7k XLE order 4.8 bps.
  - Spreads matter for a handful of wide names: ERIE 59 bps, and the scanner small caps we quoted, at 10-285 bps.
  - Impact is negligible for S&P names at typical sleeve sizes (a few thousand dollars, under 0.05% of ADV). It reaches about 10 bps only for orders like $30k of ERIE (0.2% of ADV).
- **Week 1 shows the churn.** In three sessions, base traded $568k of notional on a $30k sleeve (19x). It paid about $47 in commission plus an estimated $118 in half-spreads, roughly 0.55% of the sleeve. Paper fills behaved like half-spread fills: +1.2 bps per leg against arrival (SE 1.8), vs a model half-spread of 1.5.
- **Whether the tool improves trading is unknown** until the -ml sleeves run against their twins. It only helps if the LLM skips, resizes or converts trades because of it."""

USAGE = """- Before any order you are unsure about, call `cost` with the symbol and the quantity you intend. This matters most for intraday switches, small orders, and names outside the market table.
- Read `cost_bps` (median, with `cost_bps_10`-`cost_bps_90`) and `breakeven_share_of_daily_move`. Only trade if the gain you expect over your holding period clearly exceeds `cost_bps`.
  - A ratio of 0.25 means the round trip eats a quarter of a typical day's move. For a trade you plan to hold for an hour, that is usually a losing trade.
- **Commission has a $1 minimum per order.** Below about $2,000 per order you pay 10+ bps round trip in commission alone, and below $700 you pay 30+ bps. Consolidate small orders, and don't rebalance by a few shares.
- **Hourly rotation between sector ETFs costs a full round trip each time.** That is about 3-8 bps at $3-7k per order, so 20-50 bps a day over six looks. This is more than most edges.
- **Tier C/D, or a spread above about 20 bps:** use limit orders near the mid, not market orders. Avoid market orders in the first minutes after the open and before it (pre-market quotes ran 3.5x the session-average spread). Keep orders under about 1% of ADV, and plan with the 90% figure.
- The tool forecasts cost only, not direction or risk. A low cost-to-move ratio on a volatile small cap does not make it a good trade."""

CAVEATS = """- **Target.** The tool forecasts the session's time-average spread. A market order pays the spread at that minute, which is wider just after the open and before it, and narrower midday. In our paper fills, opening fills were not costlier on average (-3.3 bps, SE 4.8, n 22), but that sample is noise.
- **Impact is not fitted.** k is a literature prior (Toth et al. 2011; Frazzini, Israel & Moskowitz 2018). IBKR paper fills don't move prices, so the paper account's actual cost is roughly spread plus commission. The impact term is what a real market would charge, and it only matters for orders that are a meaningful share of ADV.
- **Universe.** The universe is today's S&P 500 plus liquid ETFs: survivor-biased, and all large and liquid. For scanner small caps the cold model extrapolates (median factor 1.6 off on 9 snapshots). During market hours the live quote replaces it, with an interval from 214 snapshots of liquid names, which may be too narrow for thin ones.
- **Short history.** Bid/ask data begins 17 Oct 2024, so the first training window had about 33 sessions. The walk-forward minimum was set to 20 sessions (stated in the pre-registration).
- **Fill evidence is thin.** It covers 114 orders over 3 sessions. The arrival reference is a 30-minute bar's open within 3 minutes of the fill: a trade price, so it carries half-spread noise. Only 76 orders had a usable reference, and 9 symbols were outside the cache. Slippage against the delayed price the model saw (mean +20 bps, SE 15) is mostly 15 minutes of drift.
- **Commission model.** It uses IBKR fixed pricing ($0.005 a share, $1 minimum, 1% maximum). It ignores the SEC and FINRA fees on sales (about 0.3 bps) and borrow costs for shorts.
- **Expected daily move.** This is the `forecast` tool's absr1 when it is cached, else sqrt(2/pi) x 20-day volatility. It is a typical size, not tomorrow's move.
- **Live data.** Live features come from `gw.bars` (3 months of daily bars) and `gw.bars(what='BID_ASK')` (2 months), falling back to the cached `data/bidask` file. If both are unavailable, the cold model is used."""


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    {"validate": validate, "test": test}[cmd]()
