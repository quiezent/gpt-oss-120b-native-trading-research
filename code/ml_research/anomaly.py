"""Tool "anomaly": a pump / anomaly detector that flags abnormal or thin names before an LLM buys them.
Pre-registration: runs/ml/research/anomaly_prereg.md (written before the first run).

    .venv\\Scripts\\python.exe ml_research\\anomaly.py           # validate(): walk-forward OOS, anomaly.json/.md, anomaly_oos.npz
    .venv\\Scripts\\python.exe ml_research\\anomaly.py --fit     # fit the live model on all realized targets (anomaly_model.pkl)
    .venv\\Scripts\\python.exe ml_research\\anomaly.py --test    # live() against a fake gateway built from cached bars

Features at t use bars through the close of t only. Targets start at the close of t (entry there): rel1, rel5
(market-relative log returns) and crash5 (a market-relative drop of more than 2 five-session sigmas within 5 sessions).
Walk-forward like ml_eval.walk_forward: test from 20250102, retrain every 21 sessions on an expanding window, the
training targets end before the test month starts (embargo 5).
"""

import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "ml" / "research"
MODEL = OUT / "anomaly_model.pkl"
OOS = OUT / "anomaly_oos.npz"
ET = ZoneInfo("America/New_York")
TEST_FROM, START, RETRAIN, MIN_TRAIN, H = "20250102", 61, 21, 120, 5
FEATS = ["ret_z", "vol_ratio", "gap_z", "range_ratio", "streak", "ret5_z", "dist20_z", "clv", "log_price", "log_dvol"]
MOVE = FEATS[:8]
EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 99.5, 100.0])
VARIANTS = ("rz_abs", "rz_pump", "iso", "gbm_crash")
BEST = None          # set from validate()'s result (anomaly.json) for live()/sim_provider(); see _best()


# ---------------------------------------------------------------- features (any (symbols, sessions) arrays)
def features(o, h, l, c, v):
    """Feature dict {name: (symbols, sessions)} from daily bars through each session's close."""
    with np.errstate(all="ignore"):
        T = lambda a: pd.DataFrame(np.asarray(a, float).T)          # sessions x symbols for pandas rolling
        lc = np.log(c)
        lr = np.diff(lc, axis=1, prepend=np.nan)
        s60 = T(lr).rolling(60, min_periods=40).std().shift(1).to_numpy().T
        s60 = np.where(s60 > 1e-4, s60, np.nan)
        vv = np.where(np.asarray(v, float) > 0, v, np.nan)
        vmed = T(vv).rolling(60, min_periods=40).median().shift(1).to_numpy().T
        cp = np.concatenate([np.full((c.shape[0], 1), np.nan), c[:, :-1]], axis=1)
        rng = (h - l) / cp
        rmed = T(np.where(rng > 0, rng, np.nan)).rolling(60, min_periods=40).median().shift(1).to_numpy().T
        sg = np.sign(np.nan_to_num(lr))
        streak = np.zeros_like(lr)
        for t in range(1, lr.shape[1]):
            same = (sg[:, t] == np.sign(streak[:, t - 1])) & (sg[:, t] != 0)
            streak[:, t] = np.where(same, streak[:, t - 1] + sg[:, t], sg[:, t])
        streak = np.clip(streak, -10, 10)
        r5 = lc - np.concatenate([np.full((c.shape[0], 5), np.nan), lc[:, :-5]], axis=1)
        m20 = T(c).rolling(20, min_periods=20).mean().to_numpy().T
        dv = T(c * vv).rolling(20, min_periods=10).median().to_numpy().T
        F = {"ret_z": lr / s60, "vol_ratio": np.log(vv / vmed), "gap_z": np.log(o / cp) / s60,
             "range_ratio": np.log(np.where(rng > 0, rng, np.nan) / rmed), "streak": np.where(np.isnan(lr), np.nan, streak),
             "ret5_z": r5 / (s60 * np.sqrt(5)), "dist20_z": np.log(c / m20) / s60,
             "clv": np.where(h > l, (c - l) / (h - l) - 0.5, 0.0), "log_price": lc, "log_dvol": np.log(dv)}
        for k in F:
            F[k] = np.where(np.isfinite(F[k]), F[k], np.nan)
        F["clv"] = np.where(np.isnan(c), np.nan, F["clv"])
        aux = {"lr": lr, "s60": s60, "vmult": vv / vmed, "rmult": np.where(rng > 0, rng, np.nan) / rmed,
               "gap": np.log(o / cp), "r5": r5, "dist20": np.log(c / m20), "dvol": dv}
    return F, aux


def targets(d):
    """rel = log return minus the universe's cross-sectional median; rel1, rel5, mdd5, crash5 from the close of t."""
    with np.errstate(all="ignore"):
        lr = d.lr
        rel = lr - np.nanmedian(lr, axis=0, keepdims=True)
        nsym, nd = lr.shape
        cum = np.full((5, nsym, nd), np.nan)
        for k in range(1, 6):
            acc = np.zeros((nsym, nd - k))
            for i in range(1, k + 1):
                acc = acc + rel[:, i:nd - k + i]
            cum[k - 1, :, :nd - k] = acc
        mdd5 = np.min(cum, axis=0)                           # NaN if any of the 5 is missing
        srel = pd.DataFrame(rel.T).rolling(60, min_periods=40).std().shift(1).to_numpy().T
        crash5 = np.where(np.isnan(mdd5) | np.isnan(srel), np.nan, (mdd5 < -2 * srel * np.sqrt(5)).astype(float))
        rel1 = np.concatenate([rel[:, 1:], np.full((nsym, 1), np.nan)], axis=1)
    return {"rel1": rel1, "rel5": cum[4], "mdd5": mdd5, "crash5": crash5}


def rows(F, names, idx):
    """(len(idx) * symbols, len(names)) rows, step-major, symbol-minor (ml.rows convention)."""
    return np.column_stack([F[n][:, idx].T.reshape(-1) for n in names])


# ---------------------------------------------------------------- models
class Model:
    """One variant fit on a training window: robust-z stats, the scorer, and the score -> probability/band tables."""

    def __init__(self, kind):
        self.kind = kind

    def zs(self, X):
        return np.clip((X - self.med) / self.mad, -10, 10)

    def raw(self, X):
        Z = self.zs(X)
        if self.kind == "rz_abs":
            return np.sqrt(np.nanmean(Z[:, :8] ** 2, axis=1))
        if self.kind == "rz_pump":
            p = lambda n: np.maximum(np.nan_to_num(Z[:, FEATS.index(n)]), 0)
            up = np.nan_to_num(X[:, 0]) > 0
            return p("ret_z") + p("ret5_z") + p("dist20_z") + p("gap_z") + p("streak") + up * (p("vol_ratio") + p("range_ratio"))
        if self.kind == "iso":
            return -self.iso.score_samples(np.nan_to_num(Z))
        return self.gbm.predict(X)

    def fit(self, X, y, rel5, month, rng):
        ok = ~np.isnan(X[:, :8]).all(1)
        X, y, rel5, month = X[ok], y[ok], rel5[ok], month[ok]
        self.med = np.nanmedian(X, axis=0)
        self.mad = 1.4826 * np.nanmedian(np.abs(X - self.med), axis=0)
        self.mad = np.where(self.mad > 0, self.mad, 1.0)
        if self.kind == "iso":
            from sklearn.ensemble import IsolationForest
            sub = rng.choice(len(X), min(200_000, len(X)), replace=False)
            self.iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=0).fit(np.nan_to_num(self.zs(X[sub])))
        if self.kind == "gbm_crash":
            import ml
            lab = ~np.isnan(y)
            self.gbm = ml.fit(X[lab], y[lab], "binary")
        s = self.raw(X)
        self.qs = np.quantile(s, np.linspace(0, 1, 2001))                 # score -> percentile of the training window
        pct = self.pct(s)
        b = np.clip(np.searchsorted(EDGES, pct, side="right") - 1, 0, len(EDGES) - 2)
        lab = ~np.isnan(y)
        self.base = float(np.mean(y[lab]))
        self.rate, self.rlo, self.rhi, self.band = [], [], [], []
        for i in range(len(EDGES) - 1):
            m = (b == i) & lab
            self.rate.append(float(np.mean(y[m])) if m.sum() else self.base)
            mr = [float(np.mean(y[m & (month == mo)])) for mo in np.unique(month[m]) if (m & (month == mo)).sum() >= 20]
            self.rlo.append(float(np.quantile(mr, 0.1)) if len(mr) >= 3 else 0.0)
            self.rhi.append(float(np.quantile(mr, 0.9)) if len(mr) >= 3 else 1.0)
            mr5 = (b == i) & ~np.isnan(rel5)
            self.band.append(np.quantile(rel5[mr5], [0.1, 0.5, 0.9]).tolist() if mr5.sum() > 50 else [np.nan] * 3)
        for a in ("rate", "rlo", "rhi", "band"):
            setattr(self, a, np.array(getattr(self, a)))
        return self

    def pct(self, s):
        return 100 * np.interp(s, self.qs, np.linspace(0, 1, len(self.qs)))

    def predict(self, X):
        """Per row: score, percentile, P(crash5), its 10-90% range, rel5 band (10/50/90)."""
        s = self.raw(X)
        pct = self.pct(s)
        b = np.clip(np.searchsorted(EDGES, pct, side="right") - 1, 0, len(EDGES) - 2)
        prob = s if self.kind == "gbm_crash" else self.rate[b]
        bad = np.isnan(X[:, :8]).all(1) | ~np.isfinite(s)
        nanify = lambda a: np.where(bad if a.ndim == 1 else bad[:, None], np.nan, a)
        return {"score": nanify(s), "pct": nanify(pct), "prob": nanify(prob), "plo": nanify(self.rlo[b]),
                "phi": nanify(self.rhi[b]), "band": nanify(self.band[b])}


# ---------------------------------------------------------------- walk-forward validation
def _month(dates, idx, nsym):
    return np.repeat(np.array([int(dates[t][:6]) for t in idx]), nsym)


def walk(d, F, Y, kind):
    nsym, nd = d.c.shape
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    out = {k: np.full((nsym, nd), np.nan) for k in ("score", "pct", "prob", "plo", "phi")}
    out["band"] = np.full((3, nsym, nd), np.nan)
    rng = np.random.default_rng(0)
    for T in range(t0, nd, RETRAIN):
        tr = np.arange(START, T - H)                     # crash5/rel5 of t end at t + 5 <= T - 1
        if len(tr) < MIN_TRAIN:
            continue
        m = Model(kind).fit(rows(F, FEATS, tr), Y["crash5"][:, tr].T.reshape(-1), Y["rel5"][:, tr].T.reshape(-1),
                            _month(d.dates, tr, nsym), rng)
        te = np.arange(T, min(T + RETRAIN, nd))
        p = m.predict(rows(F, FEATS, te))
        for k in ("score", "pct", "prob", "plo", "phi"):
            out[k][:, te] = p[k].reshape(len(te), nsym).T
        out["band"][:, :, te] = p["band"].reshape(len(te), nsym, 3).transpose(2, 1, 0)
        print(kind, d.dates[T], f"base {m.base:.4f}", flush=True)
    return out, t0


def _auc(p, y):
    ok = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[ok], y[ok]
    npos, nneg = y.sum(), (1 - y).sum()
    if npos < 3 or nneg < 3:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _month_t(vals, months):
    """mean of monthly means and its t-stat (months as clusters)."""
    s = pd.Series(vals).groupby(months).mean()
    if len(s) < 3:
        return float(np.mean(vals)) if len(vals) else np.nan, np.nan, len(s)
    return float(s.mean()), float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s)


def evaluate(d, P, Y, t0, base_rate):
    import ml_eval
    nd = len(d.dates)
    steps = list(range(t0, nd - H))
    S = lambda a: a[:, steps].ravel()
    p, pc, sc, y = S(P["prob"]), S(P["pct"]), S(P["score"]), S(Y["crash5"])
    r1, r5, md, lo, hi = S(Y["rel1"]), S(Y["rel5"]), S(Y["mdd5"]), S(P["plo"]), S(P["phi"])
    b10, b90 = P["band"][0][:, steps].ravel(), P["band"][2][:, steps].ravel()
    br = S(base_rate)
    months = np.repeat(np.array([int(d.dates[t][:6]) for t in steps])[None, :], d.c.shape[0], axis=0).ravel()
    ok = ~np.isnan(p) & ~np.isnan(y)
    res = {"cells": int(ok.sum()), "base_rate_test": float(np.mean(y[ok]))}
    res["auc_pooled"] = _auc(sc[ok], y[ok])
    daily = np.array([_auc(P["score"][:, t], Y["crash5"][:, t]) for t in steps])
    per = daily[::H]
    per = per[~np.isnan(per)]
    res["auc_daily_mean"] = float(np.nanmean(daily))
    res["auc_daily_t"] = float((per.mean() - 0.5) / (per.std(ddof=1) / np.sqrt(len(per))))
    res["brier"] = float(np.mean((p[ok] - y[ok]) ** 2))
    res["brier_base_rate"] = float(np.mean((br[ok] - y[ok]) ** 2))
    res["brier_skill"] = 1 - res["brier"] / res["brier_base_rate"]
    q = np.quantile(p[ok], np.linspace(0, 1, 11))
    rel = []
    for i in range(10):
        m = ok & (p >= q[i]) & ((p < q[i + 1]) if i < 9 else (p <= q[i + 1]))
        if m.sum():
            rel.append({"decile": i + 1, "pred": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4), "n": int(m.sum())})
    res["reliability"] = rel
    flags = {}
    for name, thr in (("pct>=99", 99), ("pct>=95", 95), ("pct>=90", 90), ("pct<50", None)):
        m = ok & ((pc >= thr) if thr else (pc < 50))
        mr1 = m & ~np.isnan(r1)
        mr5 = m & ~np.isnan(r5)
        a1, t1, _ = _month_t(r1[mr1] * 1e4, months[mr1])
        a5, t5, nm = _month_t(r5[mr5] * 1e4, months[mr5])
        flags[name] = {"n": int(m.sum()), "share": round(float(m.sum() / ok.sum()), 4), "crash5_rate": round(float(y[m].mean()), 4),
                       "crash5_vs_base": round(float(y[m].mean() / y[ok].mean()), 2), "pred_prob_mean": round(float(p[m].mean()), 4),
                       "rel1_bps": round(a1, 1), "rel1_t": round(t1, 2), "rel5_bps": round(a5, 1), "rel5_t": round(t5, 2),
                       "mdd5_bps_mean": round(float(np.nanmean(md[m]) * 1e4), 1), "months": nm}
    res["flags"] = flags
    dec = []
    for i in range(10):
        m = ok & (pc >= i * 10) & ((pc < (i + 1) * 10) if i < 9 else (pc <= 100))
        dec.append({"pct_decile": i + 1, "n": int(m.sum()), "crash5": round(float(y[m].mean()), 4),
                    "rel5_bps": round(float(np.nanmean(r5[m]) * 1e4), 1), "mdd5_bps": round(float(np.nanmean(md[m]) * 1e4), 1)})
    res["by_pct_decile"] = dec
    res["rel1_score"] = ml_eval.score(-P["score"], Y["rel1"], steps, 1, d.dates)
    res["rel5_score"] = ml_eval.score(-P["score"], Y["rel5"], steps, 5, d.dates)
    okb = ~np.isnan(r5) & ~np.isnan(b10)
    res["rel5_band_coverage_10_90"] = round(float(np.mean((r5[okb] >= b10[okb]) & (r5[okb] <= b90[okb]))), 4)
    okb2 = okb & (pc >= 95)
    res["rel5_band_coverage_pct95up"] = round(float(np.mean((r5[okb2] >= b10[okb2]) & (r5[okb2] <= b90[okb2]))), 4)
    # interval check: realized monthly crash5 rate per (month, probability bin) inside its 10-90% interval?
    pb = np.clip(np.searchsorted(EDGES, pc, side="right") - 1, 0, len(EDGES) - 2)
    cells, inside = 0, 0
    df = pd.DataFrame({"m": months[ok], "b": pb[ok], "y": y[ok], "lo": lo[ok], "hi": hi[ok]})
    for (_, _), g in df.groupby(["m", "b"]):
        if len(g) < 20:
            continue
        cells += 1
        inside += int(g["lo"].iloc[0] <= g["y"].mean() <= g["hi"].iloc[0])
    res["prob_interval_coverage_month_bins"] = round(inside / max(cells, 1), 4)
    res["prob_interval_cells"] = cells
    res["pass"] = bool(res["auc_pooled"] > 0.5 and res["auc_daily_t"] >= 2 and res["brier_skill"] > 0
                       and dec[-1]["crash5"] >= 1.5 * res["base_rate_test"])
    res["passes_return_bar"] = bool(res["rel1_score"]["pass"] or res["rel5_score"]["pass"])
    return res


def spikes(d, F, Y, t0):
    """Descriptive: upside spikes on volume (ret_z > 3 and volume > 3x the 60-day median), from t0."""
    nd = len(d.dates)
    sl = slice(t0, nd - H)
    m = (F["ret_z"][:, sl] > 3) & (F["vol_ratio"][:, sl] > np.log(3))
    dn = (F["ret_z"][:, sl] < -3) & (F["vol_ratio"][:, sl] > np.log(3))
    months = np.repeat(np.array([int(x[:6]) for x in d.dates[sl]])[None, :], d.c.shape[0], axis=0)
    out = {}
    for name, mask in (("up_spike", m), ("down_spike", dn), ("all", np.ones_like(m))):
        y, r1, r5, md = (Y[k][:, sl] for k in ("crash5", "rel1", "rel5", "mdd5"))
        ok = mask & ~np.isnan(y)
        a1, t1, _ = _month_t(r1[ok] * 1e4, months[ok])
        a5, t5, nm = _month_t(r5[ok] * 1e4, months[ok])
        out[name] = {"n": int(ok.sum()), "crash5": round(float(y[ok].mean()), 4), "rel1_bps": round(a1, 1), "rel1_t": round(t1, 2),
                     "rel5_bps": round(a5, 1), "rel5_t": round(t5, 2), "mdd5_bps": round(float(np.nanmean(md[ok]) * 1e4), 1),
                     "months": nm}
    return out


def validate():
    import trader
    d = trader.Data()
    F, aux = features(d.o, d.h, d.l, d.c, d.v)
    Y = targets(d)
    nsym, nd = d.c.shape
    cards, preds = {}, {}
    t0 = None
    # training base rate as of each test step (the Brier reference), same walk-forward windows
    base = np.full((nsym, nd), np.nan)
    for kind in VARIANTS:
        P, t0 = walk(d, F, Y, kind)
        if kind == VARIANTS[0]:
            for T in range(t0, nd, RETRAIN):
                tr = np.arange(START, T - H)
                base[:, T:T + RETRAIN] = np.nanmean(Y["crash5"][:, tr])
        cards[kind] = evaluate(d, P, Y, t0, base)
        preds[kind] = P
        c = cards[kind]
        print(kind, json.dumps({k: c[k] for k in ("auc_pooled", "auc_daily_t", "brier_skill", "pass")}),
              "rel1 IC t", round(c["rel1_score"]["ic_t"], 2), "rel5 IC t", round(c["rel5_score"]["ic_t"], 2), flush=True)
    passing = [k for k in VARIANTS if cards[k]["pass"]]
    best = max(passing, key=lambda k: cards[k]["brier_skill"]) if passing else max(VARIANTS, key=lambda k: cards[k]["brier_skill"])
    P = preds[best]
    np.savez_compressed(OOS, symbols=np.array(d.sym), dates=np.array(d.dates), variant=best, score=P["score"], pct=P["pct"],
                        prob=P["prob"], plo=P["plo"], phi=P["phi"], band10=P["band"][0], band50=P["band"][1], band90=P["band"][2],
                        **{k: F[k] for k in MOVE}, lr=aux["lr"], vmult=aux["vmult"], rmult=aux["rmult"], gap=aux["gap"],
                        r5=aux["r5"], dist20=aux["dist20"], dvol=aux["dvol"],
                        **{f"pct_{k}": preds[k]["pct"] for k in VARIANTS})
    res = {"test": f"{d.dates[t0]}..{d.dates[nd - 1 - H]} (targets realized)", "train_from": d.dates[START],
           "retrain_every": RETRAIN, "embargo": H, "variants": cards, "best": best, "best_passes": bool(cards[best]["pass"]),
           "spikes": spikes(d, F, Y, t0)}
    res["case_studies"] = case_studies(d, best)
    (OUT / "anomaly.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    write_md(res)
    return res


# ---------------------------------------------------------------- final model (live) and case studies
def fit_final(d=None, kind=None):
    """Fit `kind` (default: the validated best) on every realized target; save anomaly_model.pkl."""
    import trader
    d = d or trader.Data()
    kind = kind or _best()
    F, aux = features(d.o, d.h, d.l, d.c, d.v)
    Y = targets(d)
    nsym, nd = d.c.shape
    tr = np.arange(START, nd - H)
    m = Model(kind).fit(rows(F, FEATS, tr), Y["crash5"][:, tr].T.reshape(-1), Y["rel5"][:, tr].T.reshape(-1),
                        _month(d.dates, tr, nsym), np.random.default_rng(0))
    m.as_of, m.trained_through = d.dates[-1], d.dates[nd - 1 - H]
    m.vol_qs = _vol_qs(F, aux, d, nd - 1)
    MODEL.write_bytes(pickle.dumps(m))
    return m


def _vol_qs(F, aux, d, t):
    """Quantiles of the stocks' 60-day daily vol across the universe at session t (ETFs excluded)."""
    st = np.array([s not in d.etf for s in d.sym])
    v = aux["s60"][st, t]
    return np.quantile(v[np.isfinite(v)], np.linspace(0, 1, 201))


def _best():
    p = OUT / "anomaly.json"
    return BEST or (json.loads(p.read_text(encoding="utf-8"))["best"] if p.exists() else "rz_pump")


def _load_model():
    if MODEL.exists():
        return pickle.loads(MODEL.read_bytes())
    return fit_final()


def case_studies(d, kind):
    """Score the cached out-of-universe names (data/bars, not in trader.Data) with a model fit on the universe through
    20241231 only, then look at what followed in 2025-26 (descriptive; these names were cached because someone looked)."""
    import bars as barsmod
    nsym, nd = d.c.shape
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    F, _ = features(d.o, d.h, d.l, d.c, d.v)
    Y = targets(d)
    tr = np.arange(START, t0 - H)
    m = Model(kind).fit(rows(F, FEATS, tr), Y["crash5"][:, tr].T.reshape(-1), Y["rel5"][:, tr].T.reshape(-1),
                        _month(d.dates, tr, nsym), np.random.default_rng(0))
    F_, aux_ = features(d.o, d.h, d.l, d.c, d.v)
    m.vol_qs = _vol_qs(F_, aux_, d, t0 - 1)
    med = np.nanmedian(d.lr, axis=0)
    ix = {x: i for i, x in enumerate(d.dates)}
    rows_out, named = [], {}
    for p in sorted((ROOT / "data" / "bars").glob("*.json")):
        s = p.stem
        if s in d.k:
            continue
        b = barsmod.load(s)["bars"]
        if len(b) < 80:
            continue
        A = {k: np.array([[r[k] for r in b]], float) for k in ("open", "high", "low", "close", "volume")}
        Fs, aux = features(A["open"], A["high"], A["low"], A["close"], A["volume"])
        pr = m.predict(rows(Fs, FEATS, np.arange(len(b))))
        dates = [r["date"] for r in b]
        lr = aux["lr"][0]
        rel = np.array([lr[i] - med[ix[x]] if x in ix else np.nan for i, x in enumerate(dates)])
        srel = pd.Series(rel).rolling(60, min_periods=40).std().shift(1).to_numpy()
        for i in range(len(b) - H):
            if dates[i] < TEST_FROM or np.isnan(pr["pct"][i]):
                continue
            fw = rel[i + 1:i + 6]
            if np.isnan(fw).any() or np.isnan(srel[i]):
                continue
            cum = np.cumsum(fw)
            rows_out.append((s, pr["pct"][i], pr["prob"][i], float(cum.min() < -2 * srel[i] * np.sqrt(5)), fw[0], cum[-1],
                             aux["dvol"][0, i], int(dates[i][:6])))
        if s in ("NBIZ", "AEHL", "REAX", "MF"):
            named[s] = [{"date": dates[i], "close": b[i]["close"], "ret_pct": round(float(lr[i] * 100), 1),
                         "pct": round(float(pr["pct"][i]), 1), "p_crash5": round(float(pr["prob"][i]), 3),
                         "reason": _reason({k: Fs[k][0, i] for k in FEATS},
                                           {**{k: aux[k][0, i] for k in aux}, "volpct": _volpct(m, aux["s60"][0, i])})}
                        for i in range(max(0, len(b) - 4), len(b))]
    df = pd.DataFrame(rows_out, columns=["sym", "pct", "prob", "crash", "rel1", "rel5", "dvol", "month"])
    out = {"model_trained_through": d.dates[t0 - 1 - H], "names": int(df["sym"].nunique()), "cells": len(df),
           "base_crash5": round(float(df["crash"].mean()), 4)}
    for name, m_ in (("pct>=99", df["pct"] >= 99), ("pct>=95", df["pct"] >= 95), ("pct<50", df["pct"] < 50)):
        g = df[m_]
        _, t5, nm = _month_t(g["rel5"].to_numpy() * 1e4, g["month"].to_numpy())
        _, tc, _ = _month_t(g["crash"].to_numpy() - df["crash"].mean(), g["month"].to_numpy())
        out[name] = {"n": len(g), "names": int(g["sym"].nunique()), "crash5": round(float(g["crash"].mean()), 4),
                     "rel5_t_months": round(t5, 2), "crash5_excess_t_months": round(tc, 2), "months": nm,
                     "pred_prob_mean": round(float(g["prob"].mean()), 4), "rel1_bps": round(float(g["rel1"].mean() * 1e4), 1),
                     "rel5_bps": round(float(g["rel5"].mean() * 1e4), 1)}
    thin = df["dvol"] < 50e6
    out["dvol<50M"] = {"n": int(thin.sum()), "names": int(df[thin]["sym"].nunique()), "base_crash5": round(float(df[thin]["crash"].mean()), 4),
                       "crash5_pct>=95": round(float(df[thin & (df["pct"] >= 95)]["crash"].mean()), 4) if (thin & (df["pct"] >= 95)).any() else None,
                       "n_pct>=95": int((thin & (df["pct"] >= 95)).sum())}
    out["recent"] = named
    return out


# ---------------------------------------------------------------- outputs for one symbol
TIERS = ((500e6, "very liquid (>$500M/day)"), (50e6, "liquid ($50-500M/day)"), (5e6, "moderate ($5-50M/day)"),
         (0.5e6, "thin ($0.5-5M/day)"), (0, "very thin (<$0.5M/day)"))


def tier(dvol):
    if dvol is None or not np.isfinite(dvol):
        return "unknown"
    return next(name for lo, name in TIERS if dvol >= lo)


def _reason(f, a):
    """The one or two features furthest from normal, in plain words."""
    parts = []
    lr, s60 = a.get("lr"), a.get("s60")
    if lr is not None and np.isfinite(lr) and abs(f.get("ret_z", 0) or 0) >= 2:
        parts.append((abs(f["ret_z"]), f"{(np.exp(lr) - 1) * 100:+.1f}% day ({f['ret_z']:+.1f} sigma of its 60-day vol)"))
    vm = a.get("vmult")
    if vm is not None and np.isfinite(vm) and vm >= 2:
        parts.append((np.log(vm) / 0.45, f"volume {vm:.1f}x its 60-day median"))
    g = a.get("gap")
    if g is not None and np.isfinite(g) and abs(f.get("gap_z", 0) or 0) >= 2:
        parts.append((abs(f["gap_z"]), f"opened {(np.exp(g) - 1) * 100:+.1f}% from the previous close"))
    rm = a.get("rmult")
    if rm is not None and np.isfinite(rm) and rm >= 2:
        parts.append((np.log(rm) / 0.4, f"day range {rm:.1f}x typical"))
    r5 = a.get("r5")
    if r5 is not None and np.isfinite(r5) and abs(f.get("ret5_z", 0) or 0) >= 2:
        parts.append((abs(f["ret5_z"]), f"{(np.exp(r5) - 1) * 100:+.1f}% over 5 sessions"))
    st = f.get("streak")
    if st is not None and np.isfinite(st) and abs(st) >= 5:
        parts.append((abs(st) / 2.5, f"{int(abs(st))} {'up' if st > 0 else 'down'} days in a row"))
    d20 = a.get("dist20")
    if d20 is not None and np.isfinite(d20) and abs(f.get("dist20_z", 0) or 0) >= 2.5:
        parts.append((abs(f["dist20_z"]) * 0.8, f"{(np.exp(d20) - 1) * 100:+.1f}% from its 20-day average"))
    cl = f.get("clv")
    if parts and cl is not None and np.isfinite(cl) and (lr or 0) > 0 and cl <= -0.3:
        parts.append((0.1, "closed near the day's low"))
    facts = []                                   # level facts (not scored by the model; see anomaly.md)
    vp, s60, dv = a.get("volpct"), a.get("s60"), a.get("dvol")
    if vp is not None and np.isfinite(vp) and vp >= 98 and s60 is not None and np.isfinite(s60):
        facts.append(f"very volatile: 60-day vol {s60 * np.sqrt(252) * 100:.0f}%/yr, above {min(vp, 99.9):.1f}% of S&P 500 names")
    if dv is not None and np.isfinite(dv) and dv < 5e6:
        facts.append(f"thin: ${dv / 1e6:.1f}M a day")
    if not parts:
        return "; ".join(["nothing unusual vs its own history"] + facts)
    parts.sort(key=lambda x: -x[0])
    txt = [p[1] for p in parts[:2]]
    if "closed near the day's low" in [p[1] for p in parts] and "closed near the day's low" not in txt:
        txt.append("closed near the day's low")
    return "; ".join(txt + facts)


def _volpct(model, s60):
    qs = getattr(model, "vol_qs", None)
    if qs is None or s60 is None or not np.isfinite(s60):
        return None
    return float(100 * np.interp(s60, qs, np.linspace(0, 1, len(qs))))


def _row(model, f, a, as_of, provisional=False):
    X = np.array([[f[k] for k in FEATS]], float)
    p = model.predict(X)
    r = lambda x, k=1: None if x is None or not np.isfinite(x) else round(float(x), k)
    a = {**a, "volpct": _volpct(model, a.get("s60"))}
    return {"as_of": as_of, "provisional_today": provisional, "anomaly_pct": r(p["pct"][0]),
            "pump_risk_p": r(p["prob"][0], 3), "pump_risk_p_10_90": [r(p["plo"][0], 3), r(p["phi"][0], 3)],
            "base_rate": round(model.base, 3),
            "dollar_volume": r(a.get("dvol"), 0), "liquidity_tier": tier(a.get("dvol")),
            "vol60_ann_pct": r(a["s60"] * np.sqrt(252) * 100 if a.get("s60") is not None else None, 0),
            "vol60_vs_universe_pct": r(a["volpct"], 1), "reason": _reason(f, a)}


# ---------------------------------------------------------------- live
def _prev_session(now):
    d = now.date() - timedelta(days=1) if now.hour < 16 else now.date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def _session_fraction(now):
    """Elapsed share of today's regular session (None outside it)."""
    if now.weekday() >= 5:
        return None
    m = now.hour * 60 + now.minute - (9 * 60 + 30)
    return None if m < 0 or m >= 390 else max(m, 15) / 390


def live(gw, symbols=None, intraday=True, now=None, **kw):
    """{symbol: {anomaly_pct, pump_risk_p, pump_risk_p_10_90, rel5_10_50_90_pct, liquidity_tier, reason, ...}}.

    Daily bars: the cache (data/bars) when it reaches the last completed session, else gw.bars(symbol, '1 Y', '1 day')
    (scanner names outside the cache). With intraday=True and the market open, today's session so far is appended from
    gw.quotes as a provisional bar (volume projected to a full day by elapsed time, which overstates early-morning
    volume because trading is front-loaded); the validation used completed days only. symbols=None scores the whole
    cached universe from the cache (no gateway calls) and returns it sorted by anomaly percentile."""
    import bars as barsmod
    model = _load_model()
    now = now or datetime.now(ET)
    prev = _prev_session(now)
    frac = _session_fraction(now) if intraday else None
    if symbols is None:
        import trader
        symbols = trader.universe()
        frac, use_gw = None, False
    else:
        symbols = [str(s).upper().strip() for s in symbols]
        use_gw = gw is not None
    quotes, stats = {}, {}
    if use_gw and frac is not None:
        try:
            quotes = {q["symbol"]: q for q in gw.quotes(symbols) if not q.get("error")}
        except Exception:
            quotes = {}
    out = {}
    for s in symbols:
        doc = barsmod.load(s)
        rows_ = [r for r in (doc["bars"] if doc else []) if r["date"] <= prev]     # completed sessions only
        source = "cache"
        if use_gw and (not rows_ or rows_[-1]["date"] < prev):
            try:
                got = gw.bars(s, duration="1 Y", bar_size="1 day")
                rows_ = [{"date": str(r["date"])[:8], **{k: r[k] for k in ("open", "high", "low", "close", "volume")}} for r in got]
                rows_ = [r for r in rows_ if r["date"] <= prev]
                source = "gateway"
            except Exception as e:
                out[s] = {"error": f"no daily bars: {e}"}
                continue
        provisional = False
        q = quotes.get(s)
        if q and frac is not None and q.get("last") and q.get("open") and rows_ and rows_[-1]["date"] < now.strftime("%Y%m%d"):
            rows_ = rows_ + [{"date": now.strftime("%Y%m%d"), "open": q["open"], "high": max(q.get("high") or q["last"], q["last"]),
                              "low": min(q.get("low") or q["last"], q["last"]), "close": q["last"],
                              "volume": (q.get("volume") or np.nan) / frac}]
            provisional = True
        if len(rows_) < 45:
            out[s] = {"error": f"only {len(rows_)} daily bars; need about 45"}
            continue
        A = {k: np.array([[r[k] for r in rows_]], float) for k in ("open", "high", "low", "close", "volume")}
        F, aux = features(A["open"], A["high"], A["low"], A["close"], A["volume"])
        f = {k: F[k][0, -1] for k in FEATS}
        a = {k: v[0, -1] for k, v in aux.items()}
        if not np.isfinite(a["dvol"] if a["dvol"] is not None else np.nan) and use_gw:
            try:
                st = gw.stats([s])[0]
                if st.get("avg_volume"):
                    a["dvol"] = st["avg_volume"] * rows_[-1]["close"]
            except Exception:
                pass
        row = _row(model, f, a, rows_[-1]["date"], provisional)
        row["bars_from"] = source
        row["stale"] = source == "cache" and rows_[-1]["date"] < prev
        out[s] = row
    if symbols is not None and len(out) > 50:
        out = dict(sorted(out.items(), key=lambda kv: -(kv[1].get("anomaly_pct") or -1)))
    return out


# ---------------------------------------------------------------- training simulator
def sim_provider(data):
    """A function of decision session d -> {symbol: row} as of the close of d-1, from the strictly out-of-sample
    walk-forward predictions in anomaly_oos.npz (NaN before 20250102, so None there)."""
    z = np.load(OOS, allow_pickle=True)
    syms, dates = [str(s) for s in z["symbols"]], [str(x) for x in z["dates"]]
    col = {x: i for i, x in enumerate(dates)}
    A = {k: z[k] for k in z.files if k not in ("symbols", "dates", "variant")}
    base = float(json.loads((OUT / "anomaly.json").read_text(encoding="utf-8"))["variants"][str(z["variant"])]["base_rate_test"]) \
        if (OUT / "anomaly.json").exists() else np.nan
    _, aux = features(data.o, data.h, data.l, data.c, data.v)      # rolling, so column t uses bars through t only
    s60 = aux["s60"]
    st = np.array([s not in data.etf for s in data.sym])

    def at(d):
        as_of = data.dates[d - 1]
        i = col.get(as_of)
        if i is None:
            return None
        out = {}
        r = lambda x, k=1: None if x is None or not np.isfinite(x) else round(float(x), k)
        v = s60[st, d - 1]
        qs = np.quantile(v[np.isfinite(v)], np.linspace(0, 1, 201))
        for j, s in enumerate(syms):
            if not np.isfinite(A["pct"][j, i]) or data.k.get(s) is None:
                continue
            f = {k: A[k][j, i] for k in MOVE}
            a = {"lr": A["lr"][j, i], "vmult": A["vmult"][j, i], "rmult": A["rmult"][j, i], "gap": A["gap"][j, i],
                 "r5": A["r5"][j, i], "dist20": A["dist20"][j, i], "dvol": A["dvol"][j, i], "s60": s60[data.k[s], d - 1]}
            a["volpct"] = float(100 * np.interp(a["s60"], qs, np.linspace(0, 1, len(qs)))) if np.isfinite(a["s60"]) else None
            out[s] = {"as_of": as_of, "provisional_today": False, "anomaly_pct": r(A["pct"][j, i]), "pump_risk_p": r(A["prob"][j, i], 3),
                      "pump_risk_p_10_90": [r(A["plo"][j, i], 3), r(A["phi"][j, i], 3)], "base_rate": round(base, 3),
                      "dollar_volume": r(a["dvol"], 0), "liquidity_tier": tier(a["dvol"]),
                      "vol60_ann_pct": r(a["s60"] * np.sqrt(252) * 100, 0), "vol60_vs_universe_pct": r(a["volpct"]),
                      "reason": _reason(f, a)}
        return out or None
    return at


# ---------------------------------------------------------------- report
def _fmt(x, k=3):
    return "na" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{k}f}"


def write_md(res):
    V = res["variants"]
    L = ["# Tool \"anomaly\": pump / anomaly detector", "",
         f"*17 Oct 2026. Code: `ml_research/anomaly.py`. Pre-registration: `anomaly_prereg.md` (written before the first run). "
         f"Scorecards: `anomaly.json`. OOS predictions: `anomaly_oos.npz` (variant `{res['best']}`).*", "",
         "## Method", "",
         f"- Data: `trader.Data()` daily bars, 523 symbols (today's S&P 500 + ETFs). Out of sample {res['test']}; expanding "
         f"window from {res['train_from']}, retrain every {res['retrain_every']} sessions, embargo {res['embargo']}.",
         "- Features at t (through the close of t): return z vs own 60-day vol, volume / 60-day median, gap z, range / typical, "
         "signed streak, 5-day return z, distance from the 20-day mean, close location in the day's range, log price, log dollar volume.",
         "- Entry at the close of t. Targets: rel1, rel5 (log return minus the universe median); crash5 = market-relative drop "
         "of more than 2 five-session sigmas (own 60-day relative vol x sqrt 5) at any close within 5 sessions.",
         "- Probability: training-window crash5 rate in the score's percentile bin (edges 0..90 by 10, 95, 98, 99, 99.5); "
         "interval = 10-90% of that bin's monthly rates in training. gbm_crash uses its own probability.", "",
         "## Variants (all 4 pre-registered, all reported)", "",
         "| Variant | AUC pooled | daily AUC (t) | Brier skill | top-decile crash5 / base | pct>=99: crash5, rel1, rel5 bps (t) | rel1 IC t | rel5 IC t | Pass (risk) | Pass (return) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in V.items():
        f99 = c["flags"]["pct>=99"]
        L.append(f"| {k} | {_fmt(c['auc_pooled'])} | {_fmt(c['auc_daily_mean'])} ({_fmt(c['auc_daily_t'], 1)}) | {_fmt(c['brier_skill'], 4)} | "
                 f"{c['by_pct_decile'][-1]['crash5']:.3f} / {c['base_rate_test']:.3f} | {f99['crash5_rate']:.3f}, "
                 f"{f99['rel1_bps']:+.0f} ({f99['rel1_t']:+.1f}), {f99['rel5_bps']:+.0f} ({f99['rel5_t']:+.1f}) | "
                 f"{_fmt(c['rel1_score']['ic_t'], 2)} | {_fmt(c['rel5_score']['ic_t'], 2)} | {'yes' if c['pass'] else 'no'} | "
                 f"{'yes' if c['passes_return_bar'] else 'no'} |")
    b = V[res["best"]]
    L += ["", f"## Best: `{res['best']}`" + ("" if res["best_passes"] else " (no variant passed; best Brier skill shown)"), "",
          "Calibration, reliability by predicted-probability decile (out of sample):", "",
          "| decile | predicted | observed | n |", "|---|---|---|---|"]
    L += [f"| {r['decile']} | {r['pred']:.3f} | {r['observed']:.3f} | {r['n']} |" for r in b["reliability"]]
    L += ["", f"- Brier {b['brier']:.4f} vs {b['brier_base_rate']:.4f} for the training base rate (skill {b['brier_skill']:+.4f}).",
          f"- Probability interval: realized monthly crash5 rate inside its 10-90% interval in "
          f"{b['prob_interval_coverage_month_bins']:.0%} of {b['prob_interval_cells']} (month, bin) cells (target about 80%).",
          f"- rel5 10-90% band coverage: {b['rel5_band_coverage_10_90']:.1%} overall, {b['rel5_band_coverage_pct95up']:.1%} for pct>=95 (target 80%).",
          "", "By anomaly percentile decile (out of sample):", "", "| decile | n | crash5 | rel5 bps | mean worst 5-day rel drawdown bps |",
          "|---|---|---|---|---|"]
    L += [f"| {r['pct_decile']} | {r['n']} | {r['crash5']:.3f} | {r['rel5_bps']:+.1f} | {r['mdd5_bps']:+.1f} |" for r in b["by_pct_decile"]]
    L += ["", "Flags (percentile vs the training window's symbol-days; t-stats cluster by month):", "",
          "| flag | share | crash5 | x base | rel1 bps (t) | rel5 bps (t) | worst 5-day rel bps |", "|---|---|---|---|---|---|---|"]
    for n, f in b["flags"].items():
        L.append(f"| {n} | {f['share']:.3f} | {f['crash5_rate']:.3f} | {f['crash5_vs_base']:.2f} | {f['rel1_bps']:+.1f} ({f['rel1_t']:+.1f}) | "
                 f"{f['rel5_bps']:+.1f} ({f['rel5_t']:+.1f}) | {f['mdd5_bps_mean']:+.1f} |")
    sp = res["spikes"]
    L += ["", "## Descriptive: spikes on volume (out-of-sample period, no model)", "",
          "| event | n | crash5 | rel1 bps (t) | rel5 bps (t) | worst 5-day rel bps |", "|---|---|---|---|---|---|"]
    for n, f in sp.items():
        L.append(f"| {n} | {f['n']} | {f['crash5']:.3f} | {f['rel1_bps']:+.1f} ({f['rel1_t']:+.1f}) | {f['rel5_bps']:+.1f} ({f['rel5_t']:+.1f}) | {f['mdd5_bps']:+.1f} |")
    cs = res["case_studies"]
    L += ["", f"## Out-of-universe cached names (descriptive, selection-biased; model fit through {cs['model_trained_through']})", "",
          f"{cs['names']} names, {cs['cells']} symbol-days from 2025, crash5 base {cs['base_crash5']:.3f}.", ""]
    for n in ("pct>=99", "pct>=95", "pct<50"):
        f = cs[n]
        L.append(f"- {n}: n {f['n']} ({f['names']} names), crash5 {f['crash5']:.3f} (predicted {f['pred_prob_mean']:.3f}), "
                 f"rel1 {f['rel1_bps']:+.0f} bps, rel5 {f['rel5_bps']:+.0f} bps.")
    t = cs["dvol<50M"]
    L.append(f"- Names under $50M/day: n {t['n']} ({t['names']} names), crash5 base {t['base_crash5']}, at pct>=95 {t['crash5_pct>=95']} (n {t['n_pct>=95']}).")
    for s, rr in cs["recent"].items():
        L.append(f"- {s}: " + "; ".join(f"{r['date']} {r['ret_pct']:+.1f}% pct {r['pct']} p {r['p_crash5']} ({r['reason']})" for r in rr))
    L += ["", CONCLUSION, ""]
    (OUT / "anomaly.md").write_text("\n".join(L), encoding="utf-8")


TRACK_RECORD = (
    "Out of sample (2 Jan 2025 to 8 Oct 2026, today's S&P 500 stocks and ETFs only, so survivor-biased large caps), this "
    "anomaly score did not pass its pre-registered bar. Its information sits in the extreme tail: on symbol-days in the top "
    "1% the chance of a market-relative drop of more than two 5-session sigmas within 5 sessions was 10% against a 4.3% base "
    "rate (the stated probability averaged 8%); below the top 5% it is close to the base rate, and its 10-90% ranges held "
    "the realized monthly rate only 66% of the time, so they are too narrow. It does not forecast returns: top-1% names "
    "were followed by -12 bps the next day and -13 bps over 5 days, neither significant, and across all names unusual moves "
    "were followed by slightly better, not worse, relative returns. On 70 cached non-S&P names (selection-biased) top-1% "
    "flags were followed by a 15% drop rate, about twice the stated probability, so for small caps treat the probability "
    "as a floor. Liquidity tier, volume multiple and volatility level are facts, not forecasts.")

CONCLUSION = """## Conclusion (honest)

1. **No variant passed the pre-registered pump-risk bar, and none passed the return bar.** By the pre-registered rule
   the reported variant is the best Brier skill, `rz_abs` (+0.0013, essentially zero). `gbm_crash` ranks crash risk best
   within a day (daily AUC 0.555, t 4.4; deciles rise from 2.9% to 5.9%) but its top decile is 1.38x the base rate, short
   of 1.5x, and it over-states the top (10.2% stated vs 8.3% observed at pct >= 99).
2. **The unsupervised scores carry information only in the extreme tail.** Top 1% of symbol-days: crash5 10.3% vs
   4.3% base (2.4x), stated 8.3%; worst 5-day relative drawdown -229 vs -187 bps. Top 5%: 7.3% (1.7x). Deciles 1-8 are
   flat at about 4%, which is why the daily AUC t-stats are only 1.3-1.7.
3. **It does not predict worse returns in large caps.** Top-1% names: rel1 -12 bps (t -1.1), rel5 -13 bps (t -0.5).
   Over the whole cross-section the sign is the opposite of a pump story: unusual names did slightly *better* (rel5 IC
   +0.015 for the score, t 3.2), a short-term reversal of down moves. Upside spikes on volume (+3 sigma, 3x volume,
   n 659) were followed by rel5 -56 bps (t -1.7) and crash5 8.8%, but down spikes were just as crash-prone (10.8%).
   Large-cap spikes are mostly earnings news, not pumps.
4. **Uncertainty is under-stated.** The 10-90% probability intervals contained the realized monthly rate in 66% of
   (month, bin) cells (target 80%), and the rel5 band covered 75%. Read the intervals as about 65-75% ranges.
5. **Small caps (descriptive only).** 70 cached non-S&P names, fit through Dec 2024 and scored in 2025-26: pct >= 99
   was followed by crash5 14.5% (stated 7.3%, excess t 2.8 by month) and rel5 -292 bps (t -1.2); pct >= 95 by 8.1%. These
   names were cached because a scanner or a person looked at them, so the sample is biased, but it points the same way
   as the large caps and suggests the stated probability is too low for speculative names.
6. **The week-1 cases.** AEHL as of the close of 14 Oct (the day before rl1 bought it): pct 99.9, p 0.10 (0.03-0.15),
   "volume 67.6x its 60-day median; day range 15.4x typical; closed near the day's low; thin: $0.6M a day". NBIZ: pct
   about 45, because a +18% day is normal for a stock with 341%/yr volatility; NBIZ's -15% the next day was under one
   daily sigma for it. The NBIZ lesson is position size (see the volatility fact), not an anomaly.

## How an LLM trader should use it

- Treat it as a **fact sheet with a tail flag**, not a forecast. Read the reason, the liquidity tier and the volatility
  level before buying any scanner name. "Thin" means wide spreads and impact; "very volatile" means size down so a
  normal day's move fits the risk budget.
- Only pct >= 99 (about 1% of symbol-days) meaningfully raises the chance of a sharp relative drop (about 2.5x in
  large caps, possibly more in small caps). It says nothing reliable about direction; do not short on it.
- The probability is calibrated on S&P 500 names; for names outside the cache treat it as a floor.
- Intraday, live() appends today's session so far from delayed quotes with volume projected by elapsed time. That
  overstates volume early in the day, and the validation used completed days only.

## Caveats

- Survivorship: the universe is today's S&P 500 and ETFs. Delisted pumps and failed small caps are absent, so the
  evidence about micro-cap pumps is close to none; the small-cap section is descriptive and selection-biased.
- 21 months of out-of-sample data, one regime of crashes (April 2025) included. Overlapping 5-day targets; t-stats
  cluster by month.
- Volume is from adjusted daily bars; split days can distort the volume ratio for a session.
- The volatility-level and "thin" clauses in the reason were added after the validation run. They are facts, not part
  of the scored model.
- A rel5 band per score bin was validated (75% coverage of a 10-90% band) but is not served: it is in log-return units
  pooled across names, so it would be wrong for very volatile names.
"""


if __name__ == "__main__":
    if "--fit" in sys.argv:
        m = fit_final()
        print("fit", m.kind, "through", m.trained_through, "base", round(m.base, 4))
    elif "--test" in sys.argv:
        from ml_research.anomaly_test import main
        main()
    else:
        validate()
