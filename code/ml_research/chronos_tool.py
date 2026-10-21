"""Tool "chronos": zero-shot Amazon Chronos forecasts of next-session range and 20-session vol, as a challenger to
ml_research/risk.py (pre-registration: runs/ml/research/chronos_prereg.md).

    .venv\\Scripts\\python.exe ml_research\\chronos_tool.py            # validate(): forecasts (cached), scoring, chronos.json/.md
    .venv\\Scripts\\python.exe ml_research\\chronos_tool.py --live    # live() from the cache, prints a few symbols

Each forecast reads only that symbol's own daily series through the close of t (no lookahead, no other symbols).
  range series x_t = (high_t - low_t) / close_{t-1}; forecast x_{t+1} (= risk.py's range1 target)
  vol series   v_t = log trailing 20-session vol (ml.py's definition); forecast v_{t+20} (= risk.py's vol20 target)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml_eval     # noqa: E402
import trader      # noqa: E402

OUT = ROOT / "runs" / "ml" / "research"
TEST_FROM, TEST_TO = "20250102", "20261014"
MODELS = {"bolt": "amazon/chronos-bolt-small", "c2": "amazon/chronos-2"}
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H = {"range1": 1, "vol20": 20}
N_SAMPLE, RETRAIN, BURN = 100, 21, 42
TRACK_RECORD = "(filled in by validate(): see runs/ml/research/chronos.md)"
_PIPES = {}


# ---------------------------------------------------------------- series
def range_series(h, l, c):
    """x_t = (h_t - l_t) / c_{t-1}, for 2-D (symbols, days) or 1-D arrays."""
    h, l, c = (np.atleast_2d(np.asarray(a, float)) for a in (h, l, c))
    x = np.full_like(c, np.nan)
    with np.errstate(all="ignore"):
        x[:, 1:] = (h[:, 1:] - l[:, 1:]) / c[:, :-1]
    return x


def vol_series(c):
    """v_t = log(std(log returns t-19..t, ddof=1) * sqrt(252)); NaN until 20 returns exist (ml.py's lvol20)."""
    c = np.atleast_2d(np.asarray(c, float))
    with np.errstate(all="ignore"):
        lr = np.full_like(c, np.nan)
        lr[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
        v = np.full_like(c, np.nan)
        for t in range(20, c.shape[1]):
            w = lr[:, t - 19:t + 1]
            k = (~np.isnan(w)).sum(1)
            s = np.nanstd(w, axis=1, ddof=1)
            v[:, t] = np.where(k >= 10, np.log(np.maximum(s * np.sqrt(252), 1e-3)), np.nan)
    return v


def targets(d):
    """Targets identical to risk.py: range1 (level fraction of close_t) and vol20 (log annualized, t+1..t+20)."""
    x = range_series(d.h, d.l, d.c)
    y_rng = np.concatenate([x[:, 1:], np.full((x.shape[0], 1), np.nan)], axis=1)
    v = vol_series(d.c)
    y_vol = np.concatenate([v[:, 20:], np.full((v.shape[0], 20), np.nan)], axis=1)
    return {"range1": y_rng, "vol20": y_vol}, {"range1": x, "vol20": v}


# ---------------------------------------------------------------- model
def pipe(key):
    if key not in _PIPES:
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(max(1, min(12, __import__("os").cpu_count() or 4)))
        _PIPES[key] = BaseChronosPipeline.from_pretrained(MODELS[key], device_map="cpu", torch_dtype=torch.float32)
    return _PIPES[key]


def predict(key, contexts, steps, batch=256):
    """contexts: list of 1-D arrays (the series through t). Returns (n, 9) quantiles 0.1..0.9 at `steps` ahead."""
    import torch
    p = pipe(key)
    out = np.full((len(contexts), len(QL)), np.nan)
    for i in range(0, len(contexts), batch):
        ctx = []
        for a in contexts[i:i + batch]:
            a = np.asarray(a, float)
            first = np.argmax(~np.isnan(a)) if (~np.isnan(a)).any() else len(a)
            a = a[first:]
            ctx.append(torch.tensor(a if len(a) else np.array([np.nan]), dtype=torch.float32))
        q, _ = p.predict_quantiles(ctx, prediction_length=steps, quantile_levels=QL)
        if isinstance(q, list):                                   # chronos-2: list of (1, steps, Q)
            q = torch.stack([x[0] for x in q])
        out[i:i + len(ctx)] = q[:, steps - 1, :].float().numpy()
    return out


# ---------------------------------------------------------------- sample
def sample(d):
    """100 symbols stratified by IBKR industry (ETFs their own group) x tercile of trailing vol at 20241231, seed 0."""
    ind = json.loads((ROOT / "lists" / "industries.json").read_text(encoding="utf-8"))
    v = vol_series(d.c)
    t = max(i for i, s in enumerate(d.dates) if s <= "20241231")
    groups = {}
    for j, s in enumerate(d.sym):
        if not np.isfinite(v[j, t]):
            continue
        g = "ETF" if s in d.etf else (ind.get(s, {}).get("industry") or "Other")
        groups.setdefault(g, []).append(j)
    rng = np.random.default_rng(0)
    total = sum(len(x) for x in groups.values())
    pick = []
    for g in sorted(groups):
        js = sorted(groups[g], key=lambda j: v[j, t])
        terc = np.array_split(np.array(js), 3)
        n = max(1, round(N_SAMPLE * len(js) / total))
        for k, cell in enumerate(terc):
            if len(cell) == 0:
                continue
            m = max(1, round(n * len(cell) / len(js)))
            pick += list(rng.choice(cell, size=min(m, len(cell)), replace=False))
    pick = set(int(j) for j in pick)
    rest = sorted(j for g in groups.values() for j in g if j not in pick)      # top up (or trim) to exactly N_SAMPLE
    if len(pick) < N_SAMPLE:
        pick |= set(int(j) for j in rng.choice(rest, size=N_SAMPLE - len(pick), replace=False))
    return sorted(pick)[:N_SAMPLE] if len(pick) <= N_SAMPLE else sorted(rng.choice(sorted(pick), N_SAMPLE, replace=False).tolist())


# ---------------------------------------------------------------- walk-forward helpers
def wf_cells(t_steps, h, fit_fn, pred_fn, burn=BURN):
    """Refit every RETRAIN sessions on cells t with t + h <= T - 1 (expanding, past OOS only). Returns {T: model} blocks."""
    blocks = []
    t0 = t_steps[0]
    for T in range(t0 + burn, t_steps[-1] + 1, RETRAIN):
        tr = [t for t in t_steps if t + h <= T - 1]
        te = [t for t in t_steps if T <= t < T + RETRAIN]
        blocks.append((tr, te))
    return blocks


def ols(X, y):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    return np.linalg.lstsq(A, y[ok], rcond=None)[0]


def ols_apply(b, X):
    return b[0] + X @ b[1:]


def resid(tgt, y, p):
    with np.errstate(all="ignore"):
        return np.log(y / p) if tgt == "range1" else y - p


def apply_q(tgt, p, qr):
    """Point forecast(s) p plus residual quantiles qr -> quantiles."""
    p = np.asarray(p, float)[..., None]
    return p * np.exp(qr) if tgt == "range1" else p + qr


def walk_blend(tgt, Y, comps, steps, names):
    """Walk-forward OLS of y on the named component forecasts (arrays (sym, days)) + residual quantiles.
    Returns point (sym, days) and quantiles (sym, days, 9); NaN before the burn-in ends."""
    h = H[tgt]
    nsym, nd = Y.shape
    P = np.full((nsym, nd), np.nan)
    Q = np.full((nsym, nd, len(QL)), np.nan)
    for tr, te in wf_cells(steps, h, None, None):
        X = np.column_stack([comps[n][:, tr].T.reshape(-1) for n in names])
        y = Y[:, tr].T.reshape(-1)
        b = ols(X, y)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        fit = ols_apply(b, X[ok])
        fit = np.maximum(fit, 1e-4) if tgt == "range1" else fit
        qr = np.nanquantile(resid(tgt, y[ok], fit), QL)
        for t in te:
            Xt = np.column_stack([comps[n][:, t] for n in names])
            pt = ols_apply(b, Xt)
            pt = np.where(np.isfinite(Xt).all(1), np.maximum(pt, 1e-4) if tgt == "range1" else pt, np.nan)
            P[:, t] = pt
            Q[:, t] = apply_q(tgt, pt, qr)
    return P, Q


def calib(Q, y):
    """Hit rate per quantile, 80% coverage, mean pinball loss over the 9 levels (cells with all finite)."""
    ok = np.isfinite(Q).all(-1) & np.isfinite(y)
    Qv, yv = Q[ok], y[ok]
    hits = (yv[:, None] <= Qv).mean(0)
    ql = np.array(QL)
    e = yv[:, None] - Qv
    pin = np.mean(np.maximum(ql * e, (ql - 1) * e))
    cov = np.mean((yv >= Qv[:, 0]) & (yv <= Qv[:, -1]))
    return {"hit_rate": [round(float(x), 3) for x in hits], "cover80": round(float(cov), 3), "pinball": float(pin),
            "cells": int(ok.sum())}


def r2(p, y, cells):
    yy, pp = y[cells], p[cells]
    return float(1 - np.mean((yy - pp) ** 2) / np.var(yy))


def boot_gain(pa, pb, y, cell_mask, dates_idx, reps=2000, block=20, seed=0):
    """R^2(a) - R^2(b) with a moving-block bootstrap over dates. Returns (gain, lo5, hi95)."""
    T = sorted(set(dates_idx))
    sse_a, sse_b, ssy, sy, n = {}, {}, {}, {}, {}
    for t in T:
        m = cell_mask[:, t]
        yy = y[m, t]
        sse_a[t] = np.sum((yy - pa[m, t]) ** 2)
        sse_b[t] = np.sum((yy - pb[m, t]) ** 2)
        sy[t], ssy[t], n[t] = yy.sum(), (yy ** 2).sum(), m.sum()
    A = np.array([[sse_a[t], sse_b[t], sy[t], ssy[t], n[t]] for t in T])

    def g(rows):
        s = A[rows].sum(0)
        var = s[3] / s[4] - (s[2] / s[4]) ** 2
        return (s[1] - s[0]) / s[4] / var
    base = g(np.arange(len(T)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(T) / block))
    draws = []
    for _ in range(reps):
        st = rng.integers(0, len(T) - block + 1, nb)
        rows = np.concatenate([np.arange(s, s + block) for s in st])[:len(T)]
        draws.append(g(rows))
    return float(base), float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def half(s):
    return s[:4] + ("H1" if s[4:6] <= "06" else "H2")


# ---------------------------------------------------------------- forecasting run (cached)
def forecast_all(d, sel, key, tgt, steps):
    f = OUT / f"chronos_raw_{key}_{tgt}.npz"
    if f.exists():
        z = np.load(f)
        if list(z["sel"]) == list(sel) and list(z["steps"]) == list(steps):
            return z["q"], float(z["secs"]), int(z["n"])
    S = targets(d)[1][tgt]
    ctx, idx = [], []
    for t in steps:
        for a, j in enumerate(sel):
            ctx.append(S[j, :t + 1])
            idx.append((a, t))
    t0 = time.time()
    q = predict(key, ctx, H[tgt])
    secs = time.time() - t0
    Qa = np.full((len(sel), len(d.dates), len(QL)), np.nan)
    for (a, t), row in zip(idx, q):
        Qa[a, t] = row
    np.savez_compressed(f, q=Qa, sel=np.array(sel), steps=np.array(steps), secs=secs, n=len(ctx))
    print(f"{key} {tgt}: {len(ctx)} forecasts in {secs:.0f}s ({len(ctx) / secs:.0f}/s)", flush=True)
    return Qa, secs, len(ctx)


def validate():
    d = trader.Data()
    sel = sample(d)
    nd = len(d.dates)
    Y, S = targets(d)
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    t1 = max(i for i, s in enumerate(d.dates) if s <= TEST_TO)
    steps = {tgt: [t for t in range(t0, t1 + 1) if t + H[tgt] <= nd - 1] for tgt in H}
    z = np.load(OUT / "risk_oos.npz", allow_pickle=True)
    assert list(z["dates"]) == list(d.dates) and list(z["symbols"]) == list(d.sym)
    risk = {"range1": z["range1"][sel], "vol20": np.log(z["vol20"][sel])}
    naive = {"range1": np.full((len(sel), nd), np.nan), "vol20": S["vol20"][sel]}
    with np.errstate(all="ignore"):
        rr = (d.h - d.l) / d.c
        for t in range(19, nd):
            naive["range1"][:, t] = np.nanmean(rr[sel, t - 19:t + 1], axis=1)
    res = {"sample": [d.sym[j] for j in sel], "n_symbols": len(sel), "targets": {}, "speed": {}}
    oos = {}
    for tgt in H:
        y = Y[tgt][sel]
        st = steps[tgt]
        Q, P = {}, {}
        for key in MODELS:
            Q[key], secs, n = forecast_all(d, sel, key, tgt, st)
            P[key] = Q[key].mean(-1)
            res["speed"][f"{key}_{tgt}"] = {"forecasts": n, "seconds": round(secs, 1), "per_sec": round(n / secs, 1)}
        comps = {"risk": risk[tgt], "bolt": P["bolt"], "c2": P["c2"]}
        # identical cells: every model, naive and target finite
        cells = np.zeros(y.shape, bool)
        cells[:, st] = True
        for a in (y, risk[tgt], naive[tgt], P["bolt"], P["c2"]):
            cells &= np.isfinite(a)
        # walk-forward recalibrations and the blend (past OOS cells only)
        W = {}
        masked = {k: np.where(cells, v, np.nan) for k, v in comps.items()}
        ym = np.where(cells, y, np.nan)
        W["risk"] = walk_blend(tgt, ym, masked, st, ["risk"])
        W["bolt_recal"] = walk_blend(tgt, ym, masked, st, ["bolt"])
        W["c2_recal"] = walk_blend(tgt, ym, masked, st, ["c2"])
        W["blend"] = walk_blend(tgt, ym, masked, st, ["risk", "bolt", "c2"])
        bw = cells & np.isfinite(W["blend"][0])
        bsteps = [t for t in st if bw[:, t].any()]
        card = {"window_full": f"{d.dates[st[0]]}..{d.dates[st[-1]]}", "cells_full": int(cells.sum()),
                "window_blend": f"{d.dates[bsteps[0]]}..{d.dates[bsteps[-1]]}", "cells_blend": int(bw.sum()),
                "units": "fraction of close_t (levels)" if tgt == "range1" else "log annualized vol"}
        fulls = {"naive": naive[tgt], "risk": risk[tgt], "bolt_raw": P["bolt"], "c2_raw": P["c2"]}
        card["r2_full"] = {k: round(r2(v, y, cells), 4) for k, v in fulls.items()}
        blends = {"naive": naive[tgt], "risk": risk[tgt], "bolt_raw": P["bolt"], "c2_raw": P["c2"],
                  "bolt_recal": W["bolt_recal"][0], "c2_recal": W["c2_recal"][0], "blend": W["blend"][0],
                  "risk_recal": W["risk"][0]}
        card["r2_blend_window"] = {k: round(r2(v, y, bw), 4) for k, v in blends.items()}
        if tgt == "range1":
            with np.errstate(all="ignore"):
                card["r2_log_full"] = {k: round(r2(np.log(v), np.log(y), cells & (y > 0)), 4) for k, v in fulls.items()}
        # by half-year
        hv = {}
        for k, v in blends.items():
            hv[k] = {}
            for hk in sorted({half(d.dates[t]) for t in bsteps}):
                m = bw.copy()
                m[:, [t for t in range(nd) if half(d.dates[t]) != hk]] = False
                hv[k][hk] = round(r2(v, y, m), 4)
        card["r2_by_half_blend_window"] = hv
        hf = {}
        for k, v in fulls.items():
            hf[k] = {}
            for hk in sorted({half(d.dates[t]) for t in st}):
                m = cells.copy()
                m[:, [t for t in range(nd) if half(d.dates[t]) != hk]] = False
                hf[k][hk] = round(r2(v, y, m), 4)
        card["r2_by_half_full"] = hf
        # bootstrap gains vs risk.py
        card["gain_vs_risk_boot"] = {
            "bolt_raw_full": boot_gain(P["bolt"], risk[tgt], y, cells, st),
            "c2_raw_full": boot_gain(P["c2"], risk[tgt], y, cells, st),
            "bolt_recal_blendwin": boot_gain(W["bolt_recal"][0], risk[tgt], y, bw, bsteps),
            "c2_recal_blendwin": boot_gain(W["c2_recal"][0], risk[tgt], y, bw, bsteps),
            "blend_blendwin": boot_gain(W["blend"][0], risk[tgt], y, bw, bsteps),
            "blend_vs_risk_recal_blendwin": boot_gain(W["blend"][0], W["risk"][0], y, bw, bsteps)}
        # quantile calibration
        cal = {"native_full": {k: calib(np.where(cells[..., None], Q[k], np.nan), y) for k in MODELS},
               "native_blend_window": {k: calib(np.where(bw[..., None], Q[k], np.nan), y) for k in MODELS}}
        cal["walkforward_resid_blend_window"] = {k: calib(np.where(bw[..., None], W[k][1], np.nan), y)
                                                 for k in ("risk", "bolt_recal", "c2_recal", "blend")}
        card["calibration"] = cal
        # last blend coefficients (fit on all cells) for live
        X = np.column_stack([comps[n][cells] for n in ("risk", "bolt", "c2")])
        card["blend_coef_all"] = [round(float(b), 5) for b in ols(X, y[cells])]
        card["live_resid_q"] = {}
        bb = ols(X, y[cells])
        fit = ols_apply(bb, X)
        fit = np.maximum(fit, 1e-4) if tgt == "range1" else fit
        card["live_resid_q"]["blend"] = [round(float(x), 5) for x in np.nanquantile(resid(tgt, y[cells], fit), QL)]
        for k in ("bolt", "c2"):
            b = ols(comps[k][cells][:, None], y[cells])
            card[f"{k}_recal_coef_all"] = [round(float(x), 5) for x in b]
            fit = ols_apply(b, comps[k][cells][:, None])
            fit = np.maximum(fit, 1e-4) if tgt == "range1" else fit
            card["live_resid_q"][f"{k}_recal"] = [round(float(x), 5) for x in np.nanquantile(resid(tgt, y[cells], fit), QL)]
        res["targets"][tgt] = card
        oos[f"{tgt}_bolt_q"], oos[f"{tgt}_c2_q"] = Q["bolt"], Q["c2"]
        oos[f"{tgt}_blend"], oos[f"{tgt}_blend_q"] = W["blend"]
        oos[f"{tgt}_bolt_recal"], oos[f"{tgt}_bolt_recal_q"] = W["bolt_recal"]
        oos[f"{tgt}_c2_recal"], oos[f"{tgt}_c2_recal_q"] = W["c2_recal"]
        print(tgt, json.dumps({k: card[k] for k in ("r2_full", "r2_blend_window")}), flush=True)
    res["decision"] = decide(res)
    np.savez_compressed(OUT / "chronos_oos.npz", symbols=np.array([d.sym[j] for j in sel]), dates=np.array(d.dates), **oos)
    (OUT / "chronos.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def decide(res):
    """Pre-registered pass rules: replace / blend / drop, per target."""
    out = {}
    for tgt, c in res["targets"].items():
        hb = c["r2_by_half_blend_window"]
        g = c["gain_vs_risk_boot"]
        rep = {}
        for k in ("bolt_recal", "c2_recal"):
            wins = sum(hb[k][h] > hb["risk"][h] for h in hb["risk"])
            rep[k] = bool(c["r2_blend_window"][k] > c["r2_blend_window"]["risk"] and wins >= 3 and g[f"{k}_blendwin"][1] > 0)
        wins = sum(hb["blend"][h] > hb["risk"][h] for h in hb["risk"])
        cal = c["calibration"]["walkforward_resid_blend_window"]
        blend = bool(c["r2_blend_window"]["blend"] > c["r2_blend_window"]["risk"] and wins >= 3
                     and g["blend_blendwin"][1] > 0 and cal["blend"]["pinball"] <= cal["risk"]["pinball"])
        out[tgt] = {"replace": rep, "blend": blend, "blend_half_wins": f"{wins}/{len(hb['risk'])}",
                    "verdict": "replace" if any(rep.values()) else ("blend" if blend else "drop")}
    return out


# ---------------------------------------------------------------- sim provider (strictly OOS, as of close of d-1)
def sim_provider(data):
    """Decision session d -> (as_of, {symbol: row}) from chronos_oos.npz column d - 1 (sample symbols only), or None
    if the tool was dropped (chronos.json decision) or the date is outside the OOS window."""
    info = json.loads((OUT / "chronos.json").read_text(encoding="utf-8"))
    use = {}
    for t, v in info["decision"].items():
        if v["verdict"] == "replace":
            use[t] = next(k for k, ok in v["replace"].items() if ok)
        elif v["verdict"] == "blend":
            use[t] = "blend"
    if not use:
        return None
    z = np.load(OUT / "chronos_oos.npz", allow_pickle=True)
    syms = [str(s) for s in z["symbols"]]
    col = {str(x): i for i, x in enumerate(z["dates"])}

    def at(d):
        as_of = data.dates[d - 1]
        i = col.get(as_of)
        if i is None:
            return None
        out = {}
        for j, s in enumerate(syms):
            row = {}
            for tgt, k in use.items():
                p, q = z[f"{tgt}_{k}"][j, i], z[f"{tgt}_{k}_q"][j, i]
                if np.isfinite(p):
                    row.update(_fmt(tgt, p, q))
            if row:
                out[s] = row
        return (as_of, out) if out else None
    return at


def _fmt(tgt, p, q):
    if tgt == "range1":
        return {"range1_pct": round(100 * float(p), 2), "range1_p10_p90": [round(100 * float(q[0]), 2), round(100 * float(q[-1]), 2)]}
    return {"vol20_pct": round(100 * float(np.exp(p)), 1),
            "vol20_p10_p90": [round(100 * float(np.exp(q[0])), 1), round(100 * float(np.exp(q[-1])), 1)]}


# ---------------------------------------------------------------- live
def _gw_series(gw, sym):
    b = gw.bars(sym, duration="3 Y", bar_size="1 day")
    b = [x for x in b if x.get("close")]
    h, l, c = (np.array([float(x[k]) for x in b]) for k in ("high", "low", "close"))
    return str(b[-1]["date"]).replace("-", "")[:8], h, l, c


def _risk_range1(d):
    """risk.py's range1 for the last cached close: from forecast.py's cache if current, else risk.today() (~20 s)."""
    as_of = d.dates[-1]
    f = ROOT / "runs" / "ml" / "forecast" / f"risk_{as_of}.json"
    if f.exists():
        fc = json.loads(f.read_text(encoding="utf-8"))["forecasts"]
    else:
        from ml_research import risk
        as_of2, fc = risk.today(d)
        if as_of2 != as_of:
            return {}
    return {s: v["range1_pct"] / 100 for s, v in fc.items() if v.get("range1_pct") is not None}


def live(gw=None, symbols=None, data=None, **kw):
    """Next-session range forecast (fraction of the last close) with a 10-90% interval, as of the last close.
    Shipped variant (chronos.json decision): range1 = walk-forward-validated blend of risk.py's range1_har, chronos-bolt-small
    and chronos-2; intervals = the blend's out-of-sample residual quantiles. The 20-session vol forecast was dropped
    (it lost to risk.py), so it is not served. Symbols outside the cache: series from gw.bars (daily), and the
    bolt-small-only recalibrated forecast (risk.py needs the cached cross-section), flagged in `basis`."""
    info = json.loads((OUT / "chronos.json").read_text(encoding="utf-8"))
    card = info["targets"]["range1"]
    d = data or trader.Data()
    syms = [str(s).upper().strip() for s in (symbols or d.sym)]
    rows, notes, ctx, keys = {}, [], [], []
    for s in syms:
        j = d.k.get(s)
        try:
            if j is not None:
                as_of, h, l, c = d.dates[-1], d.h[j], d.l[j], d.c[j]
            elif gw is not None:
                as_of, h, l, c = _gw_series(gw, s)
                notes.append(f"{s}: not in the cache; series from gateway daily bars")
            else:
                rows[s] = {"note": "not covered"}
                continue
        except Exception as e:      # noqa: BLE001
            rows[s] = {"note": f"not covered ({e})"}
            continue
        c = np.asarray(c, float)
        ctx.append(range_series(h, l, c)[0])
        keys.append((s, j is not None, as_of, float(c[~np.isnan(c)][-1])))
    t0 = time.time()
    if keys:
        pb = predict("bolt", ctx, 1).mean(-1)
        cached = [i for i, k in enumerate(keys) if k[1]]
        pc = np.full(len(keys), np.nan)
        if cached:
            pc[cached] = predict("c2", [ctx[i] for i in cached], 1).mean(-1)
        rr = _risk_range1(d) if cached else {}
        bl, qbl = np.array(card["blend_coef_all"]), np.array(card["live_resid_q"]["blend"])
        a, b = card["bolt_recal_coef_all"]
        qb = np.array(card["live_resid_q"]["bolt_recal"])
        cov = card["calibration"]["walkforward_resid_blend_window"]
        for i, (s, inc, as_of, close) in enumerate(keys):
            r_ = rr.get(s, np.nan)
            if inc and np.isfinite(r_) and np.isfinite(pc[i]):
                p, qr, basis, cv = bl[0] + bl[1] * r_ + bl[2] * pb[i] + bl[3] * pc[i], qbl, "blend", cov["blend"]["cover80"]
            else:
                p, qr, basis, cv = a + b * pb[i], qb, "bolt_recal (risk.py not available for this symbol)", cov["bolt_recal"]["cover80"]
            p = max(float(p), 1e-4)
            rows[s] = {"as_of": as_of, "close": round(close, 2), **_fmt("range1", p, apply_q("range1", p, qr)),
                       "basis": basis, "cover80_oos": cv}
    return {"as_of": d.dates[-1], "verdict": {t: v["verdict"] for t, v in info["decision"].items()},
            "seconds": round(time.time() - t0, 1), "rows": rows, "notes": notes, "track_record": TRACK_RECORD}


def report(res=None):
    """Write the track-record line into chronos.json and the human report runs/ml/research/chronos.md."""
    f = OUT / "chronos.json"
    res = res or json.loads(f.read_text(encoding="utf-8"))
    T, D, sp = res["targets"], res["decision"], res["speed"]
    rg, vo = T["range1"], T["vol20"]
    pct = lambda x: f"{100 * x:.0f}%"
    bw, g = rg["r2_blend_window"], rg["gain_vs_risk_boot"]["blend_blendwin"]
    cb = rg["calibration"]["walkforward_resid_blend_window"]["blend"]
    tr = (f"Next-session range, blend of our range model with the Chronos time-series foundation models (zero-shot, each "
          f"symbol's own daily history). Out of sample {rg['window_blend']} on {res['n_symbols']} stratified S&P 500 symbols "
          f"(survivor-biased): R2 {bw['blend']:.3f} vs {bw['risk']:.3f} for our range model alone and {bw['naive']:.3f} for the "
          f"20-day average range (gain {g[0]:+.3f}, 90% CI {g[1]:+.3f} to {g[2]:+.3f}; ahead in {D['range1']['blend_half_wins']} "
          f"half-years, behind in 2026H2). A small improvement, not a new signal. The 10-90% interval held "
          f"{100 * cb['cover80']:.0f}% of outcomes (nominal 80%). Chronos's 20-session vol forecasts lost to our vol model "
          f"(R2 {vo['r2_full']['c2_raw']:.2f} chronos-2, {vo['r2_full']['bolt_raw']:.2f} bolt vs {vo['r2_full']['risk']:.2f}) "
          f"and are not served.")
    res["track_record"] = tr
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")

    def tab(card, key, cols):
        rows = [f"| model | {' | '.join(cols)} |", "|---" * (len(cols) + 1) + "|"]
        for m in card[key]:
            vals = [str(card[key][m][c]) if isinstance(card[key][m], dict) else "" for c in cols]
            rows.append(f"| {m} | {' | '.join(vals)} |")
        return rows

    L = ["# Tool \"chronos\": zero-shot Chronos vs the shipped risk models", "",
         f"Pre-registration: `chronos_prereg.md`. Code: `ml_research/chronos_tool.py`. Sample: {res['n_symbols']} symbols "
         "stratified by IBKR industry and vol tercile (list in chronos.json).", "",
         "## Verdict", ""]
    for tgt in ("range1", "vol20"):
        L.append(f"- **{tgt}: {D[tgt]['verdict']}** (replace tests {D[tgt]['replace']}; blend passes {D[tgt]['blend']}, "
                 f"blend wins {D[tgt]['blend_half_wins']} half-years)")
    L += ["", "## Track record (tool description text)", "", tr, ""]
    for tgt, card in T.items():
        L += [f"## {tgt} ({card['units']})", "",
              f"Full window {card['window_full']}, {card['cells_full']} identical cells. Blend window {card['window_blend']}, "
              f"{card['cells_blend']} cells (the first {BURN} test sessions only fit the blend and recalibrations).", "",
              "OOS R2, full window:", "", "| model | R2 |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in card["r2_full"].items()]
        if "r2_log_full" in card:
            L += ["", "OOS R2 of log range (diagnostic), full window:", "", "| model | R2 |", "|---|---|"]
            L += [f"| {k} | {v} |" for k, v in card["r2_log_full"].items()]
        L += ["", "OOS R2, blend window (recal = walk-forward a + b * forecast; blend = OLS on risk.py, bolt and c2):", "",
              "| model | R2 |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in card["r2_blend_window"].items()]
        hs = list(next(iter(card["r2_by_half_blend_window"].values())).keys())
        L += ["", "By half-year (blend window):", "", "| model | " + " | ".join(hs) + " |", "|---" * (len(hs) + 1) + "|"]
        L += [f"| {k} | " + " | ".join(str(v[h]) for h in hs) + " |" for k, v in card["r2_by_half_blend_window"].items()]
        L += ["", "R2 gain vs risk.py, moving-block bootstrap over dates (gain, 5%, 95%):", "", "| comparison | gain | 5% | 95% |",
              "|---|---|---|---|"]
        L += [f"| {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |" for k, v in card["gain_vs_risk_boot"].items()]
        L += ["", "Quantile calibration (hit rate of each quantile 0.1..0.9 should equal the level; cover80 nominal 0.80):", "",
              "| source | model | cover80 | pinball | hit rates |", "|---|---|---|---|---|"]
        for src, dd in card["calibration"].items():
            for m, c in dd.items():
                L.append(f"| {src} | {m} | {c['cover80']} | {c['pinball']:.6f} | {' '.join(str(x) for x in c['hit_rate'])} |")
        L.append("")
    L += ["## Speed (CPU, 12 threads, shared machine)", "", "| run | forecasts | seconds | per second |", "|---|---|---|---|"]
    L += [f"| {k} | {v['forecasts']} | {v['seconds']} | {v['per_sec']} |" for k, v in sp.items()]
    L += [""]
    (OUT / "chronos.md").write_text(chr(10).join(L), encoding="utf-8")
    return tr


def _load_track_record():
    global TRACK_RECORD
    f = OUT / "chronos.json"
    if f.exists():
        TRACK_RECORD = json.loads(f.read_text(encoding="utf-8")).get("track_record", TRACK_RECORD)


_load_track_record()

if __name__ == "__main__":
    if "--live" in sys.argv:
        print(json.dumps(live(symbols=["SPY", "AAPL", "NVDA"]), indent=1))
    elif "--report" in sys.argv:
        print(report())
    else:
        print(report(validate()))
