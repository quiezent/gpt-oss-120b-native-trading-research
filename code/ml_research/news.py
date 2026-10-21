"""Tool "news": a pre-filter that points an LLM's limited research budget at the names whose news matters
(pre-registration: runs/ml/research/news_prereg.md; results: runs/ml/research/news.md / news.json).

Headlines are IBKR historical news (BRFG = Briefing.com stock columns, BRFUPDN = Briefing.com analyst actions), cached
by ml_research/news_fetch.py in data/news/<SYM>.json. Each headline gets a FinBERT (ProsusAI/finbert) sentiment, a
novelty (1 - max cosine similarity, all-MiniLM-L6-v2, to the symbol's headlines of the previous 30 days) and, for
analyst actions, a type (upgrade, downgrade, initiate, other). Per-symbol features over a window ending at the decision
time (pre-open: since the previous close; close: the last session's day) predict the next session's |move| and range
as a multiplier on ml_research/risk.py's forecast, and (variant 5) the next session's open-to-close relative return.

    .venv\\Scripts\\python.exe ml_research\\news.py            # validate(): walk-forward OOS, news.json, news.md, news_oos.npz
    .venv\\Scripts\\python.exe ml_research\\news.py --live     # live() on a fake gateway built from the cache (test)

live(gw, symbols) needs a read-only Gateway("trader-research"); it calls only gw.news(). Nothing published at or
after the decision time is used anywhere.
"""

import html
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "ml" / "research"
NEWS = ROOT / "data" / "news"
SCORES = OUT / "news_scores.npz"
MODEL = OUT / "news_model.json"
OOS = OUT / "news_oos.npz"
INDEX = OUT / "news_index.json"
ET = ZoneInfo("America/New_York")
TEST_FROM, MIN_TRAIN, RETRAIN = "20250701", 100, 21
NOV_DAYS = 30
COST_RT = 0.0020                       # 20 bps a round trip per leg (pre-registered for variant 5)
MAX_SYMBOLS = 20
FINBERT, EMBED = "ProsusAI/finbert", "sentence-transformers/all-MiniLM-L6-v2"

V1 = ["any_pre", "l_pre", "any_day", "l_day"]
V2 = V1 + ["l_ud", "brfg_pre", "absnet_max", "nov_max", "abn"]
V3 = ["any_day", "l_day", "l_ud_day", "brfg_day", "absnet_max_day", "nov_max_day", "abn_day", "l5"]
V4 = V2 + ["lbase"]
VARIANTS = {"pre_count": V1, "pre_full": V2, "day_full": V3, "pre_gbm": V4}
NEWS_COL = {"pre_count": ("any_pre", "any_day"), "pre_full": ("any_pre", "any_day"), "day_full": ("any_day",),
            "pre_gbm": ("any_pre", "any_day")}

TRACK_RECORD = "(filled in by validate(); see runs/ml/research/news.md)"
try:
    TRACK_RECORD = json.loads(MODEL.read_text(encoding="utf-8"))["track_record"]
except (OSError, ValueError, KeyError):
    pass


# ---------------------------------------------------------------- headlines
ACT = re.compile(r"\b(upgraded|downgraded|initiated|resumed|reiterated|assumed|reinstated)\b", re.I)


def action(provider, text):
    """'up', 'down', 'init' or 'other' for an analyst action (BRFUPDN); None for a column."""
    if provider != "BRFUPDN":
        return None
    m = ACT.search(text)
    w = m.group(1).lower() if m else ""
    return {"upgraded": "up", "downgraded": "down", "initiated": "init", "resumed": "init", "reinstated": "init",
            "assumed": "init"}.get(w, "other")


def clean(text):
    return text.lstrip("!").strip()


def to_ts(s):
    """IBKR 'YYYY-MM-DDTHH:MM:SSZ' (UTC) -> int seconds since epoch."""
    return int(datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def et_ts(yyyymmdd, hh, mm=0):
    d = datetime(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]), hh, mm, tzinfo=ET)
    return int(d.timestamp())


def kstat(meta):
    """IBKR's sentiment tag K (-1..1) when present, else nan."""
    m = re.search(r"K:(-?[0-9.]+)", meta or "")
    return float(m.group(1)) if m else float("nan")


def load_raw(sym):
    p = NEWS / f"{sym.replace(' ', '_')}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))["headlines"]


def key(sym, h):
    return f"{sym}|{h['provider']}|{h['article_id']}"


# ---------------------------------------------------------------- scoring models (fixed, pre-trained)
class Scorer:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        self.torch = torch
        def load(cls_, name):                       # local cache first (no hub round trip), download if missing
            try:
                return cls_.from_pretrained(name, local_files_only=True)
            except OSError:
                return cls_.from_pretrained(name)
        self.ft, self.fm = load(AutoTokenizer, FINBERT), load(AutoModelForSequenceClassification, FINBERT).eval()
        self.et, self.em = load(AutoTokenizer, EMBED), load(AutoModel, EMBED).eval()
        lab = {v.lower(): k for k, v in self.fm.config.id2label.items()}
        self.ipos, self.ineg = lab["positive"], lab["negative"]

    def score(self, texts, batch=64):
        """-> net (p_pos - p_neg), conf (max prob), emb (unit vectors, float32)."""
        net, conf, emb = [], [], []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                t = [clean(x) or "." for x in texts[i:i + batch]]
                x = self.ft(t, return_tensors="pt", padding=True, truncation=True, max_length=96)
                p = self.torch.softmax(self.fm(**x).logits, -1).numpy()
                net.append(p[:, self.ipos] - p[:, self.ineg])
                conf.append(p.max(1))
                x = self.et(t, return_tensors="pt", padding=True, truncation=True, max_length=96)
                h = self.em(**x).last_hidden_state
                m = x["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
                emb.append(self.torch.nn.functional.normalize(e, dim=1).numpy())
        if not texts:
            return np.zeros(0), np.zeros(0), np.zeros((0, 384), np.float32)
        return np.concatenate(net), np.concatenate(conf), np.concatenate(emb).astype(np.float32)


def score_cache(symbols):
    """FinBERT + embeddings for every cached headline of `symbols`, cached in news_scores.npz by key."""
    have = {}
    if SCORES.exists():
        z = np.load(SCORES, allow_pickle=False)
        have = {k: i for i, k in enumerate(z["keys"])}
        keys, net, conf, emb = list(z["keys"]), list(z["net"]), list(z["conf"]), list(z["emb"])
    else:
        keys, net, conf, emb = [], [], [], []
    todo = [(key(s, h), h["headline"]) for s in symbols for h in load_raw(s) if key(s, h) not in have]
    todo = list(dict(todo).items())
    if todo:
        t0 = time.time()
        n, c, e = Scorer.get().score([t for _, t in todo])
        keys += [k for k, _ in todo]
        net += list(n)
        conf += list(c)
        emb += list(e.astype(np.float16))
        np.savez_compressed(SCORES, keys=np.array(keys), net=np.array(net, np.float32), conf=np.array(conf, np.float32),
                            emb=np.array(emb, np.float16))
        print(f"scored {len(todo)} headlines in {time.time() - t0:.0f}s", flush=True)
    z = np.load(SCORES, allow_pickle=False)
    K, N, C, E = z["keys"], z["net"], z["conf"], z["emb"].astype(np.float32)
    return {k: (float(N[i]), float(C[i]), E[i]) for i, k in enumerate(K)}


def novelty(ts, emb, days=NOV_DAYS):
    """1 - max cosine similarity to the same symbol's headlines strictly earlier and within `days`; 1 if none."""
    nov = np.ones(len(ts))
    for i in range(len(ts)):
        j = (ts < ts[i]) & (ts >= ts[i] - days * 86400)
        if j.any():
            nov[i] = 1.0 - float(np.max(emb[j] @ emb[i]))
    return np.clip(nov, 0, 1)


def symbol_table(sym, raw, sc):
    """Arrays for one symbol, oldest first: ts, provider flags, action, net, conf, novelty, k, text."""
    rows = sorted({key(sym, h): h for h in raw}.values(), key=lambda h: h["time"])
    rows = [h for h in rows if key(sym, h) in sc]
    if not rows:
        return None
    ts = np.array([to_ts(h["time"]) for h in rows], np.int64)
    emb = np.stack([sc[key(sym, h)][2] for h in rows])
    acts = [action(h["provider"], h["headline"]) for h in rows]
    return {"ts": ts, "upd": np.array([h["provider"] == "BRFUPDN" for h in rows]),
            "brfg": np.array([h["provider"] == "BRFG" for h in rows]),
            "up": np.array([a == "up" for a in acts]), "down": np.array([a == "down" for a in acts]),
            "net": np.array([sc[key(sym, h)][0] for h in rows]), "conf": np.array([sc[key(sym, h)][1] for h in rows]),
            "nov": novelty(ts, emb), "k": np.array([kstat(h.get("meta")) for h in rows]),
            "text": [html.unescape(clean(h["headline"])) for h in rows], "time": [h["time"] for h in rows], "prov": [h["provider"] for h in rows]}


# ---------------------------------------------------------------- features
def window(T, lo, hi):
    """Feature summary of a symbol table T over [lo, hi) (epoch seconds)."""
    if T is None:
        return {"n": 0, "idx": np.zeros(0, int)}
    a, b = np.searchsorted(T["ts"], [lo, hi], side="left")
    idx = np.arange(a, b)
    out = {"n": len(idx), "idx": idx}
    if len(idx):
        out.update(n_up=int(T["up"][idx].sum()), n_down=int(T["down"][idx].sum()), n_brfg=int(T["brfg"][idx].sum()),
                   absnet_max=float(np.max(np.abs(T["net"][idx]))), nov_max=float(np.max(T["nov"][idx])),
                   net_sum=float(T["net"][idx].sum()))
    return out


def count(T, lo, hi):
    if T is None:
        return 0
    a, b = np.searchsorted(T["ts"], [lo, hi], side="left")
    return int(b - a)


def feats_from(pre, day, n30, n30c, n5, lbase=np.nan):
    """One feature dict from window summaries (the same function serves validation, sim and live)."""
    g = lambda w, k: w.get(k, 0)
    npd = pre["n"] + day["n"]
    both = [w for w in (pre, day) if w["n"]]
    f = {"any_pre": float(pre["n"] > 0), "l_pre": math.log1p(pre["n"]), "any_day": float(day["n"] > 0),
         "l_day": math.log1p(day["n"]),
         "l_ud": math.log1p(g(pre, "n_up") + g(pre, "n_down") + g(day, "n_up") + g(day, "n_down")),
         "brfg_pre": float(g(pre, "n_brfg") > 0),
         "absnet_max": max([w["absnet_max"] for w in both], default=0.0),
         "nov_max": max([w["nov_max"] for w in both], default=0.0),
         "abn": math.log((npd + 0.5) / (n30 / 21 + 0.5)) if npd else 0.0,
         "l_ud_day": math.log1p(g(day, "n_up") + g(day, "n_down")), "brfg_day": float(g(day, "n_brfg") > 0),
         "absnet_max_day": g(day, "absnet_max"), "nov_max_day": g(day, "nov_max"),
         "abn_day": math.log((day["n"] + 0.5) / (n30c / 21 + 0.5)) if day["n"] else 0.0, "l5": math.log1p(n5),
         "lbase": lbase}
    s = (g(pre, "net_sum") + g(day, "net_sum") + g(pre, "n_up") + g(day, "n_up") - g(pre, "n_down") - g(day, "n_down"))
    f["score"] = s / npd if npd else np.nan
    return f


def bounds(dates):
    close = np.array([et_ts(x, 16) for x in dates], np.int64)
    pre = np.array([et_ts(x, 9) for x in dates], np.int64)
    return close, pre


def panel(d, tables):
    """Feature arrays (symbols, sessions) for session t: pre-open window [close t-1, 09:00 t), day window
    [close t-2, close t-1), n30 to 09:00 t, n30c and n5 to close t-1. Stocks only (ETFs NaN)."""
    nsym, nd = d.c.shape
    close, pre = bounds(d.dates)
    names = sorted(set(V2 + V3 + ["score", "lbase"]))
    F = {k: np.full((nsym, nd), np.nan) for k in names}
    stock = np.array([s not in d.etf for s in d.sym])
    for j, s in enumerate(d.sym):
        if not stock[j]:
            continue
        T = tables.get(s)
        for t in range(6, nd):
            lo_c1, lo_c2 = close[t - 1], close[t - 2]
            p = window(T, lo_c1, pre[t]) if T is not None else {"n": 0}
            dy = window(T, lo_c2, lo_c1) if T is not None else {"n": 0}
            f = feats_from(p, dy, count(T, pre[t] - 30 * 86400, pre[t]), count(T, lo_c1 - 30 * 86400, lo_c1),
                           count(T, close[t - 6], lo_c1))
            for k in names:
                if k != "lbase":
                    F[k][j, t] = f[k]
    return F


def targets(d):
    """Session-t targets: |log c_t/c_{t-1}|, (h_t - l_t)/c_{t-1}, open-to-close and close-to-close relative to the
    mean stock."""
    stock = np.array([s not in d.etf for s in d.sym])
    with np.errstate(all="ignore"):
        absr = np.abs(d.lr)
        prevc = np.concatenate([np.full((d.c.shape[0], 1), np.nan), d.c[:, :-1]], axis=1)
        rng = (d.h - d.l) / prevc
        oc = np.log(d.c / d.o)
        rel_oc = oc - np.nanmean(oc[stock], axis=0)
        rel_cc = d.lr - np.nanmean(d.lr[stock], axis=0)
    for a in (absr, rng, rel_oc, rel_cc):
        a[~stock] = np.nan
    return {"absr1": absr, "range1": rng, "rel_oc": rel_oc, "rel_cc": rel_cc}


def risk_base(d):
    """risk.py's walk-forward OOS forecasts for session t (column t-1 of risk_oos.npz)."""
    z = np.load(OUT / "risk_oos.npz", allow_pickle=True)
    assert [str(s) for s in z["symbols"]] == list(d.sym) and [str(x) for x in z["dates"]] == list(d.dates)
    sh = lambda a: np.concatenate([np.full((a.shape[0], 1), np.nan), a[:, :-1]], axis=1)
    return {"absr1": sh(z["absr1"]), "range1": sh(z["range1"])}


# ---------------------------------------------------------------- models
def ols_fit(X, y, groups=None, news=None):
    """OLS of the winsorized ratio on X; date-clustered covariance; 10/90% quantiles of y / fitted by news group."""
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    groups = groups[ok] if groups is not None else None
    news = news[ok] if news is not None else None
    yw = np.clip(y, *np.nanpercentile(y, [0.1, 99.9]))
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(A, yw, rcond=None)[0]
    fit = A @ b
    model = {"b": b.tolist()}
    if groups is not None:
        e = yw - fit
        inv = np.linalg.pinv(A.T @ A)
        u = np.zeros((len(np.unique(groups)), A.shape[1]))
        _, gi = np.unique(groups, return_inverse=True)
        np.add.at(u, gi, A * e[:, None])
        meat = u.T @ u
        G = len(u)
        model["cov"] = (inv @ meat @ inv * G / max(G - 1, 1)).tolist()
    model["q"] = quantiles(y, fit, news)
    return model


def quantiles(y, fit, news):
    r = y / np.maximum(fit, 1e-3)
    out = {}
    for g, m in (("news", news), ("none", ~news)) if news is not None else (("all", np.ones(len(y), bool)),):
        if m.sum() >= 30:
            out[g] = [float(x) for x in np.nanpercentile(r[m], [10, 90])]
    out.setdefault("all", [float(x) for x in np.nanpercentile(r, [10, 90])])
    return out


def ols_pred(model, X):
    b = np.asarray(model["b"])
    return b[0] + X @ b[1:]


def news_mask(X, cols, names):
    return np.any(np.stack([X[:, names.index(c)] > 0 for c in cols]), axis=0)


def wf_variant(name, F, base, y, d, t0, start):
    """Walk-forward (ml_eval.walk_forward, horizon 1) of base * m; returns final, lo, hi forecasts (symbols, sessions)."""
    import ml
    import ml_eval
    names = VARIANTS[name]
    G = dict(F)
    G["lbase"] = np.log(base)
    ratio = y / base
    feats = lambda idx: ml.rows(G, names, idx)
    ncol = NEWS_COL[name]

    if name == "pre_gbm":
        def fit(X, yy):
            yw = np.clip(yy, *np.nanpercentile(yy, [0.1, 99.9]))
            m = ml.fit(X, yw, "l2")
            fitv = m.predict(X)
            return {"m": m, "q": quantiles(yy, fitv, news_mask(X, ncol, names))}
        mean = lambda M, X: M["m"].predict(X)
    else:
        fit = lambda X, yy: ols_fit(X, yy, news=news_mask(X, ncol, names))
        mean = lambda M, X: ols_pred(M, X)

    def band(i):
        def p(M, X):
            m = mean(M, X)
            nm = news_mask(np.nan_to_num(X), ncol, names)
            qn, q0 = M["q"].get("news", M["q"]["all"]), M["q"].get("none", M["q"]["all"])
            return m * np.where(nm, qn[i], q0[i])
        return p

    models = []
    fit_keep = lambda X, yy: models.append(fit(X, yy)) or models[-1]
    out = {}
    for lab, pr in (("m", mean), ("lo", band(0)), ("hi", band(1))):
        out[lab] = ml_eval.walk_forward(feats, ratio, steps_from=start, test_from=t0, horizon=1, retrain_every=RETRAIN,
                                        fit=fit_keep if lab == "m" else fit, predict=pr, min_train=MIN_TRAIN)
    return {k: base * v for k, v in out.items()}, out["m"], models


def logit_fit(x, y, l2=1e-3):
    """Logistic regression P(y=1) = sigmoid(a + b x), Newton steps, tiny ridge."""
    A = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + l2 * w
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logit_pred(w, x):
    return 1 / (1 + np.exp(-(w[0] + w[1] * x)))


# ---------------------------------------------------------------- scorecards
def half(lab):
    return lab[:4] + ("H1" if lab[4:6] <= "06" else "H2")


def r2_card(final, base, y, steps, dates, news):
    import ml_eval
    c = ml_eval.r2_oos(final, y, base, steps)
    c = {"r2_news_model": c["r2_model"], "r2_risk_base": c["r2_naive"], "cells": c["cells"]}
    # daily MSE gain (Diebold-Mariano style), date-clustered t
    gain = []
    for t in steps:
        ok = ~np.isnan(final[:, t]) & ~np.isnan(y[:, t]) & ~np.isnan(base[:, t])
        if ok.sum():
            gain.append(np.mean((y[ok, t] - base[ok, t]) ** 2 - (y[ok, t] - final[ok, t]) ** 2))
    gain = np.array(gain)
    c["mse_gain_t"] = float(gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain))))
    c["by_half"] = {}
    for h in sorted({half(dates[t]) for t in steps}):
        s = [t for t in steps if half(dates[t]) == h]
        r = ml_eval.r2_oos(final, y, base, s)
        c["by_half"][h] = {"r2_news_model": round(r["r2_model"], 4), "r2_risk_base": round(r["r2_naive"], 4)}
    nm = news[:, steps]
    yy, ff, bb = y[:, steps], final[:, steps], base[:, steps]
    ok = nm & ~np.isnan(yy) & ~np.isnan(ff) & ~np.isnan(bb)
    v = np.var(yy[ok])
    c["news_rows"] = {"cells": int(ok.sum()), "r2_news_model": float(1 - np.mean((yy[ok] - ff[ok]) ** 2) / v),
                      "r2_risk_base": float(1 - np.mean((yy[ok] - bb[ok]) ** 2) / v),
                      "mean_realized_over_base": float(np.mean(yy[ok]) / np.mean(bb[ok])),
                      "mean_forecast_over_base": float(np.mean(ff[ok]) / np.mean(bb[ok]))}
    ok0 = ~nm & ~np.isnan(yy) & ~np.isnan(ff) & ~np.isnan(bb)
    c["no_news_rows"] = {"cells": int(ok0.sum()), "mean_realized_over_base": float(np.mean(yy[ok0]) / np.mean(bb[ok0])),
                         "mean_forecast_over_base": float(np.mean(ff[ok0]) / np.mean(bb[ok0]))}
    halves = [v["r2_news_model"] - v["r2_risk_base"] for v in c["by_half"].values()][-3:]
    c["pass"] = bool(c["r2_news_model"] > c["r2_risk_base"] and c["mse_gain_t"] >= 2 and sum(x > 0 for x in halves) >= 2)
    return c


def coverage(lo, hi, y, steps, dates, news):
    yy, l, h, nm = y[:, steps], lo[:, steps], hi[:, steps], news[:, steps]
    ok = ~np.isnan(yy) & ~np.isnan(l) & ~np.isnan(h)
    inside = (yy >= l) & (yy <= h)
    out = {"all": float(inside[ok].mean()), "news_rows": float(inside[ok & nm].mean()),
           "no_news_rows": float(inside[ok & ~nm].mean()), "below": float((yy < l)[ok].mean()),
           "above": float((yy > h)[ok].mean()), "above_news_rows": float((yy > h)[ok & nm].mean()), "by_half": {}}
    for hh in sorted({half(dates[t]) for t in steps}):
        cols = [i for i, t in enumerate(steps) if half(dates[t]) == hh]
        out["by_half"][hh] = round(float(inside[:, cols][ok[:, cols]].mean()), 3)
    return out


def uplift_reliability(m, y, base, steps, news, bins=(0, 0.95, 1.05, 1.25, 1.5, 2.0, 99)):
    """OOS: rows binned by forecast multiplier; mean realized y / base in each bin, with a date-bootstrap 90% interval."""
    M, Yr, nm = m[:, steps], (y / base)[:, steps], news[:, steps]
    ok = ~np.isnan(M) & ~np.isnan(Yr)
    rng = np.random.default_rng(0)
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = ok & (M >= a) & (M < b)
        if sel.sum() < 30:
            continue
        cols = np.where(sel.any(0))[0]
        per = [(Yr[sel[:, c], c].sum(), sel[:, c].sum()) for c in cols]
        s, n = np.array([p[0] for p in per]), np.array([p[1] for p in per])
        boot = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(cols), len(cols)) for _ in range(400))]
        out.append({"mult_bin": f"{a}-{b}", "n": int(sel.sum()), "news_share": float(nm[sel].mean()),
                    "mean_forecast_mult": round(float(M[sel].mean()), 3), "mean_realized_ratio": round(float(s.sum() / n.sum()), 3),
                    "realized_90ci": [round(float(x), 3) for x in np.percentile(boot, [5, 95])]})
    return out


# ---------------------------------------------------------------- validation
def load_tables(symbols):
    sc = score_cache(symbols)
    return {s: symbol_table(s, load_raw(s), sc) for s in symbols}


def validate():
    import ml_eval
    import trader
    d = trader.Data()
    stocks = [s for s in d.sym if s not in d.etf]
    t_load = time.time()
    tables = load_tables(stocks)
    print(f"tables {time.time() - t_load:.0f}s", flush=True)
    F = panel(d, tables)
    print(f"panel {time.time() - t_load:.0f}s", flush=True)
    Y = targets(d)
    B = risk_base(d)
    nd = len(d.dates)
    t0 = next(i for i, x in enumerate(d.dates) if x >= TEST_FROM)
    start = int(np.where((~np.isnan(B["absr1"])).any(0))[0][0])
    steps = list(range(t0, nd))
    res = {"test": f"{d.dates[t0]}..{d.dates[-1]}", "train_from": d.dates[start], "retrain_every": RETRAIN,
           "decision_times": {"pre": "09:00 ET of session t (window since 16:00 ET of t-1)",
                              "close": "16:00 ET of t-1 (window = session t-1's day)"},
           "variants": {}, "post_hoc": {}}
    # coverage / data description
    n_head = sum(len(T["ts"]) for T in tables.values() if T is not None)
    allk = np.concatenate([T["k"] for T in tables.values() if T is not None])
    allts = np.concatenate([T["ts"] for T in tables.values() if T is not None])
    test_ts = allts >= et_ts(TEST_FROM, 0)
    res["data"] = {"symbols_fetched": sum(1 for s in stocks if (NEWS / f"{s.replace(' ', '_')}.json").exists()),
                   "symbols_with_headlines": sum(1 for T in tables.values() if T is not None), "headlines": int(n_head),
                   "headlines_test_window": int(test_ts.sum()),
                   "ibkr_K_tag_coverage_test": float(np.mean(~np.isnan(allk[test_ts]))),
                   "share_stock_sessions_with_pre_news_test": float(np.nanmean(F["any_pre"][:, steps])),
                   "share_with_pre_or_day_news_test": float(np.nanmean(np.maximum(F["any_pre"], F["any_day"])[:, steps]))}
    print(json.dumps(res["data"]), flush=True)
    news_pre = (np.nan_to_num(F["any_pre"]) + np.nan_to_num(F["any_day"])) > 0
    news_day = np.nan_to_num(F["any_day"]) > 0
    preds = {}
    for name in VARIANTS:
        res["variants"][name] = {}
        for tgt in ("absr1", "range1"):
            out, m, models = wf_variant(name, F, B[tgt], Y[tgt], d, t0, start)
            nm = news_day if name == "day_full" else news_pre
            card = r2_card(out["m"], B[tgt], Y[tgt], steps, d.dates, nm)
            card["interval_10_90_coverage"] = coverage(out["lo"], out["hi"], Y[tgt], steps, d.dates, nm)
            card["uplift_reliability"] = uplift_reliability(m, Y[tgt], B[tgt], steps, nm)
            if name != "pre_gbm":
                card["last_refit_coefs"] = dict(zip(["const"] + VARIANTS[name], [round(x, 4) for x in models[-1]["b"]]))
            res["variants"][name][tgt] = card
            preds[f"{name}_{tgt}"], preds[f"{name}_{tgt}_lo"], preds[f"{name}_{tgt}_hi"] = out["m"], out["lo"], out["hi"]
            preds[f"{name}_{tgt}_mult"] = m
            print(name, tgt, {k: card[k] for k in ("r2_news_model", "r2_risk_base", "mse_gain_t", "pass")},
                  "cov", round(card["interval_10_90_coverage"]["all"], 3), flush=True)
    # post hoc (not a variant): base x training-mean ratio of news rows (pre window)
    for tgt in ("absr1", "range1"):
        G = {"any": news_pre.astype(float)}
        import ml
        m = ml_eval.walk_forward(lambda idx: ml.rows(G, ["any"], idx), Y[tgt] / B[tgt], steps_from=start, test_from=t0,
                                 horizon=1, retrain_every=RETRAIN, fit=lambda X, yy: ols_fit(X, yy),
                                 predict=ols_pred, min_train=MIN_TRAIN)
        c = r2_card(B[tgt] * m, B[tgt], Y[tgt], steps, d.dates, news_pre)
        res["post_hoc"][f"news_dummy_{tgt}"] = {k: c[k] for k in ("r2_news_model", "r2_risk_base", "mse_gain_t", "news_rows", "by_half")}
    res["variants"]["pre_sent_ret"] = direction(F, Y, d, t0, start, steps)
    res["post_hoc"]["iv_control"] = iv_control(d, F, Y, B, t0, start, steps, news_pre)
    res["post_hoc"]["absr1_by_pre_count"] = count_table(F, Y, B, preds["pre_full_absr1"], steps)
    res["post_hoc"]["gap_vs_intraday_by_pre_count"] = gap_table(d, F, Y, B, steps)
    # ship
    res["shipped"] = ship(res, d, F, Y, B, start, tables)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "news.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(OOS, symbols=np.array(d.sym), dates=np.array(d.dates),
                        **{k: v.astype(np.float32) for k, v in preds.items()},
                        base_absr1=B["absr1"].astype(np.float32), base_range1=B["range1"].astype(np.float32))
    return res


def direction(F, Y, d, t0, start, steps):
    """Variant 5: sentiment score -> P(rel_oc > 0) (walk-forward logistic) and a sign long-short."""
    import ml
    import ml_eval
    G = {"score": F["score"]}
    out = {}
    for tgt in ("rel_oc", "rel_cc"):
        y = Y[tgt]
        yb = np.where(np.isnan(y) | np.isnan(F["score"]), np.nan, (y > 0).astype(float))
        p = ml_eval.walk_forward(lambda idx: ml.rows(G, ["score"], idx), yb, steps_from=start, test_from=t0, horizon=1,
                                 retrain_every=RETRAIN, fit=lambda X, yy: logit_fit(X[:, 0], yy),
                                 predict=lambda w, X: logit_pred(w, X[:, 0]), min_train=MIN_TRAIN)
        rate = ml_eval.walk_forward(lambda idx: ml.rows(G, ["score"], idx), yb, steps_from=start, test_from=t0, horizon=1,
                                    retrain_every=RETRAIN, fit=lambda X, yy: float(np.mean(yy)),
                                    predict=lambda m, X: np.where(np.isnan(X[:, 0]), np.nan, m), min_train=MIN_TRAIN)
        P, R, Yb = p[:, steps].ravel(), rate[:, steps].ravel(), yb[:, steps].ravel()
        ok = ~np.isnan(P) & ~np.isnan(Yb) & ~np.isnan(R)
        P, R, Yb = P[ok], R[ok], Yb[ok]
        brier, brier0 = float(np.mean((P - Yb) ** 2)), float(np.mean((R - Yb) ** 2))
        edges = np.quantile(P, np.linspace(0, 1, 6))
        rel = []
        for i in range(5):
            m = (P >= edges[i]) & ((P < edges[i + 1]) if i < 4 else (P <= edges[i + 1]))
            if m.sum():
                rel.append({"bin": i + 1, "pred": round(float(P[m].mean()), 4), "observed": round(float(Yb[m].mean()), 4), "n": int(m.sum())})
        # sign long-short on the raw score
        ls, lab = [], []
        for t in steps:
            s, r = F["score"][:, t], y[:, t]
            pos, neg = (s > 0) & ~np.isnan(r), (s < 0) & ~np.isnan(r)
            if pos.any() and neg.any():
                ls.append(r[pos].mean() - r[neg].mean())
                lab.append(half(d.dates[t]))
        ls = np.array(ls)
        tt = float(ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls))))
        byh = {h: round(float(np.mean([v for v, l in zip(ls, lab) if l == h]) * 100), 4) for h in sorted(set(lab))}
        # event-level mean by sign (all news rows, not only dates with both sides)
        S, Rr = F["score"][:, steps], y[:, steps]
        okk = ~np.isnan(S) & ~np.isnan(Rr)
        ev = {"pos_mean_pct": float(np.mean(Rr[okk & (S > 0)]) * 100), "pos_n": int((okk & (S > 0)).sum()),
              "neg_mean_pct": float(np.mean(Rr[okk & (S < 0)]) * 100), "neg_n": int((okk & (S < 0)).sum()),
              "spearman_pooled": float(ml_eval.spearman(S[okk], Rr[okk]))}
        net = float(ls.mean() - 2 * COST_RT)
        out[tgt] = {"cells": int(ok.sum()), "brier": brier, "brier_base_rate": brier0, "brier_skill": 1 - brier / brier0,
                    "reliability": rel, "longshort_dates": len(ls), "longshort_gross_pct": float(ls.mean() * 100),
                    "longshort_t": tt, "longshort_net_pct": net * 100, "longshort_by_half_pct": byh, "events": ev,
                    "pass": bool(tt >= 2 and net > 0 and sum(v > 0 for v in list(byh.values())[-3:]) >= 2) if tgt == "rel_oc"
                    else "n/a (includes the opening gap: not tradeable from a 09:00 decision)"}
        print("direction", tgt, {k: out[tgt][k] for k in ("brier_skill", "longshort_gross_pct", "longshort_t", "pass")}, flush=True)
    out["note"] = "rel_oc is pre-registered (tradeable from a 09:00 decision); rel_cc includes the opening gap, reported for information."
    return out


def load_iv(sym):
    p = ROOT / "data" / "iv" / f"{sym.replace(' ', '_')}.json"
    try:
        return {b["date"]: float(b["close"]) for b in json.loads(p.read_text(encoding="utf-8"))["bars"] if b.get("close")}
    except (OSError, ValueError, KeyError):
        return {}


def iv_control(d, F, Y, B, t0, start, steps, news_pre):
    """Post hoc: does the pre-open news multiplier still add once the close-of-(t-1) implied vol is a feature?
    (The LLM sees implied vol through the stats tool.) Walk-forward OLS on the same ratio target."""
    import ml
    import ml_eval
    nsym, nd = d.c.shape
    liv = np.full((nsym, nd), np.nan)
    for j, s in enumerate(d.sym):
        iv = load_iv(s)
        for t in range(1, nd):
            v = iv.get(d.dates[t - 1])
            if v and v > 0:
                liv[j, t] = v
    out = {}
    for tgt in ("absr1", "range1"):
        G = dict(F)
        G["liv"] = np.log(liv / np.sqrt(252) / B[tgt])
        cards = {}
        for lab, names in (("iv_only", ["liv"]), ("iv_plus_pre_full", ["liv"] + V2)):
            m = ml_eval.walk_forward(lambda idx: ml.rows(G, names, idx), Y[tgt] / B[tgt], steps_from=start, test_from=t0,
                                     horizon=1, retrain_every=RETRAIN, fit=lambda X, yy: ols_fit(X, yy),
                                     predict=lambda M, X: ols_pred(M, X), min_train=MIN_TRAIN)
            c = r2_card(B[tgt] * m, B[tgt], Y[tgt], steps, d.dates, news_pre)
            cards[lab] = {k: c[k] for k in ("r2_news_model", "r2_risk_base", "cells", "mse_gain_t")}
            cards[lab]["news_rows_r2"] = c["news_rows"]["r2_news_model"]
            preds_m = B[tgt] * m
            cards[lab]["_pred"] = preds_m
        both = ml_eval.r2_oos(cards["iv_plus_pre_full"].pop("_pred"), Y[tgt], cards["iv_only"].pop("_pred"), steps)
        cards["news_beyond_iv"] = {"r2_iv_plus_news": both["r2_model"], "r2_iv_only": both["r2_naive"], "cells": both["cells"]}
        out[tgt] = cards
    out["note"] = "Not a variant (post hoc). Cells differ from the variants where implied vol is missing."
    return out


def count_table(F, Y, B, P, steps):
    """Post hoc, descriptive: OOS realized |move| / risk forecast and the pre_full multiplier, by headlines since the
    prior close (n_pre)."""
    out = []
    npre = np.expm1(F["l_pre"])[:, steps]
    r = (Y["absr1"] / B["absr1"])[:, steps]
    m = P[:, steps] / B["absr1"][:, steps]
    for lab, sel in (("0", npre < 0.5), ("1", (npre > 0.5) & (npre < 1.5)), ("2", (npre > 1.5) & (npre < 2.5)),
                     ("3-4", (npre > 2.5) & (npre < 4.5)), ("5+", npre > 4.5)):
        ok = sel & ~np.isnan(r) & ~np.isnan(m)
        if ok.sum():
            out.append({"n_pre": lab, "cells": int(ok.sum()), "mean_realized_ratio": round(float(r[ok].mean()), 3),
                        "mean_forecast_mult": round(float(m[ok].mean()), 3)})
    return out


def gap_table(d, F, Y, B, steps):
    """Post hoc, descriptive: |gap|, |open-to-close|, |close-to-close| (over risk absr1) and range (over risk range1)
    by headlines since the prior close."""
    prevc = np.concatenate([np.full((d.c.shape[0], 1), np.nan), d.c[:, :-1]], 1)
    with np.errstate(all="ignore"):
        gap, oc = np.abs(np.log(d.o / prevc))[:, steps], np.abs(np.log(d.c / d.o))[:, steps]
    rng, ab, npre = Y["range1"][:, steps], Y["absr1"][:, steps], np.expm1(F["l_pre"])[:, steps]
    b, br = B["absr1"][:, steps], B["range1"][:, steps]
    out = {}
    for lab, sel in (("0", npre < 0.5), ("1", (npre > 0.5) & (npre < 1.5)), ("2+", npre > 1.5)):
        ok = sel & ~np.isnan(gap) & ~np.isnan(oc) & ~np.isnan(b) & ~np.isnan(br)
        out[lab] = {"n": int(ok.sum()), "abs_gap/risk_absr1": round(float(gap[ok].mean() / b[ok].mean()), 3),
                    "abs_open_to_close/risk_absr1": round(float(oc[ok].mean() / b[ok].mean()), 3),
                    "absr1/risk": round(float(ab[ok].mean() / b[ok].mean()), 3),
                    "range/risk_range1": round(float(rng[ok].mean() / br[ok].mean()), 3)}
    return out


def ship(res, d, F, Y, B, start, tables):
    """Fit the shipped pieces on every realized session and write news_model.json (live and sim read it)."""
    import ml
    nd = len(d.dates)
    V = res["variants"]
    passing = [n for n in ("pre_count", "pre_full", "pre_gbm") if all(V[n][t]["pass"] for t in ("absr1", "range1"))]
    ols_pre = [n for n in ("pre_count", "pre_full") if n in passing]
    gain = lambda n: sum(V[n][t]["r2_news_model"] - V[n][t]["r2_risk_base"] for t in ("absr1", "range1"))
    pre_name = max(ols_pre, key=gain) if ols_pre else max(("pre_count", "pre_full"), key=gain)
    day_name = "day_full"
    G = dict(F)
    model = {"pre_variant": pre_name, "pre_passed": pre_name in passing, "day_variant": day_name,
             "day_passed": all(V[day_name][t]["pass"] for t in ("absr1", "range1")),
             "direction_passed": V["pre_sent_ret"]["rel_oc"]["pass"], "fit_through": d.dates[-1], "models": {}}
    tr = np.arange(start, nd)
    for name in (pre_name, day_name):
        names = VARIANTS[name]
        for tgt in ("absr1", "range1"):
            X = ml.rows(G, names, tr)
            y = (Y[tgt] / B[tgt])[:, tr].T.reshape(-1)
            dates = np.repeat(tr, len(d.sym))
            ok = ~np.isnan(y) & np.isfinite(X).all(1)
            M = ols_fit(X[ok], y[ok], groups=dates[ok], news=news_mask(X[ok], NEWS_COL[name], names))
            M["features"] = names
            model["models"][f"{name}_{tgt}"] = M
    sc = F["score"][:, tr].T.reshape(-1)
    yb = Y["rel_oc"][:, tr].T.reshape(-1)
    ok = ~np.isnan(sc) & ~np.isnan(yb)
    model["direction"] = {"w": logit_fit(sc[ok], (yb[ok] > 0).astype(float)).tolist(), "n": int(ok.sum())}
    model["track_record"] = track_record(res, model)
    MODEL.write_text(json.dumps(model, indent=1), encoding="utf-8")
    return {k: v for k, v in model.items() if k != "models"}


def track_record(res, model):
    V, P = res["variants"], res["post_hoc"]
    p = model["pre_variant"]
    A, R = V[p]["absr1"], V[p]["range1"]
    cov = A["interval_10_90_coverage"]
    dr, dc = V["pre_sent_ret"]["rel_oc"], V["pre_sent_ret"]["rel_cc"]
    iv = P["iv_control"]["absr1"]
    verdict = "passed" if model["pre_passed"] else "did NOT pass"
    G = P.get("gap_vs_intraday_by_pre_count", {})
    gp = lambda k: f"{G[k]['abs_gap/risk_absr1'] / G['0']['abs_gap/risk_absr1']:.1f}" if k in G else "?"
    rg = lambda k: f"{G[k]['range/risk_range1'] / G['0']['range/risk_range1']:.2f}" if k in G else "?"
    cnt = "; ".join(f"{b['n_pre']} -> {b['mean_realized_ratio']:.2f}x (forecast {b['mean_forecast_mult']:.2f}x, n {b['cells']})"
                    for b in P.get("absr1_by_pre_count", []))
    return (f"Out of sample {res['test'][:4]}-{res['test'][4:6]} to {res['test'][-8:-4]}-{res['test'][-4:-2]} "
            f"(walk-forward, monthly refits; IBKR's Briefing.com headlines only, about 6% of stock-days have any; today's "
            f"S&P 500, survivor-biased). The news uplift {verdict} our pre-registered bar, which needs both targets. "
            f"Next-session |move|: R2 {A['r2_news_model']:.3f} vs {A['r2_risk_base']:.3f} for the risk forecast alone "
            f"(t {A['mse_gain_t']:.1f}, better in every half-year). Range: {R['r2_news_model']:.3f} vs {R['r2_risk_base']:.3f} "
            f"(t {R['mse_gain_t']:.1f}, not significant). Realized |move| as a multiple of the risk forecast, by headlines "
            f"since the prior close: {cnt}. The extra move is mostly the opening gap: the gap grows "
            f"{gp('1')}x with one headline and {gp('2+')}x with two or more, the intraday range only {rg('1')}x and {rg('2+')}x, "
            f"so by 09:00 part of it already shows in pre-market quotes. Two or more pre-open headlines are usually an analyst "
            f"flurry the morning after earnings. 10-90% move intervals held {cov['all']:.0%} of outcomes "
            f"({cov['news_rows']:.0%} on news names). Implied vol matters more: IV alone gives |move| R2 "
            f"{iv['iv_only']['r2_news_model']:.3f}, and news still adds to it ({iv['iv_plus_pre_full']['r2_news_model']:.3f}). "
            f"Sentiment has no tradeable direction signal. Open-to-close long-short by sentiment sign: "
            f"{dr['longshort_gross_pct']:+.3f}%/day gross (t {dr['longshort_t']:.1f}), {dr['longshort_net_pct']:+.3f}% after "
            f"costs, P(up) Brier skill {dr['brier_skill']:+.3f}. The news is priced by the open: close-to-close, the same sort "
            f"gives {dc['longshort_gross_pct']:+.2f}%/day (t {dc['longshort_t']:.0f}), all of it in the opening gap.")


def refresh_track_record():
    """Rewrite TRACK_RECORD in news_model.json from news.json (no refit)."""
    res = json.loads((OUT / "news.json").read_text(encoding="utf-8"))
    model = _model()
    model["track_record"] = track_record(res, model)
    res["shipped"]["track_record"] = model["track_record"]
    MODEL.write_text(json.dumps(model, indent=1), encoding="utf-8")
    (OUT / "news.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- live
def _model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _mult(M, f):
    x = np.array([f[k] for k in M["features"]], float)
    m = float(ols_pred(M, x[None, :])[0])
    a = np.concatenate([[1.0], x])
    se = float(np.sqrt(max(a @ np.asarray(M["cov"]) @ a, 0)))
    return m, m - 1.645 * se, m + 1.645 * se


def _sessions_before(ts_utc, n=2):
    """The last n weekday 16:00 ET closes at or before ts_utc (holidays not known; see live())."""
    d = datetime.fromtimestamp(ts_utc, ET).date()
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            c = et_ts(d.strftime("%Y%m%d"), 16)
            if c <= ts_utc:
                out.append(c)
        d -= timedelta(days=1)
    return out


def _risk_live(before=None):
    """(as_of, {symbol: {absr1_pct, range1_pct, ...}}) from the newest runs/ml/forecast/fc_*.json or risk_*.json
    (written by forecast.py at the first look), optionally only files as of a session before `before` (YYYYMMDD)."""
    best = None
    for f in (ROOT / "runs" / "ml" / "forecast").glob("*_*.json"):
        if not f.name.startswith(("fc_", "risk_")):
            continue
        tag = f.stem.split("_")[-1]
        if before and tag >= before:
            continue
        if best is None or tag > best[0]:
            best = (tag, f)
    if not best:
        return None, {}
    try:
        d = json.loads(best[1].read_text(encoding="utf-8"))
        return d["as_of"], d["forecasts"]
    except (OSError, ValueError, KeyError):
        return None, {}


def live(gw, symbols=None, mode="symbols", universe=None, now=None, limit=60, pace=1.0, top=15, write_index=None,
         last_session=None, **kw):
    """News pre-filter for up to 20 symbols, or mode="scan" over `universe` (default: the cached stock universe).
    One gw.news() call per symbol (about 1 s each with pacing). Returns a dict; format_text(result) gives the LLM text.
    "Since the close" means since 16:00 ET of the last weekday; pass last_session="YYYYMMDD" after an exchange holiday."""
    t_start = time.time()
    now_ts = int((now or datetime.now(timezone.utc)).timestamp())
    M = _model()
    as_of, risk = _risk_live(datetime.fromtimestamp(now_ts, ET).strftime("%Y%m%d") if now else None)
    closes = _sessions_before(now_ts, 2)
    if last_session:                                # the caller knows the true last session (exchange holidays)
        prevd = datetime.strptime(last_session, "%Y%m%d").date() - timedelta(days=1)
        while prevd.weekday() >= 5:
            prevd -= timedelta(days=1)
        closes = [et_ts(last_session, 16), et_ts(prevd.strftime("%Y%m%d"), 16)]
    last_close, prev_close = closes
    if mode == "scan":
        syms = universe or [s.strip() for s in (ROOT / "data" / "universe523.txt").read_text().splitlines() if s.strip()]
    else:
        syms = list(dict.fromkeys(str(s).upper().strip() for s in (symbols or [])))[:MAX_SYMBOLS]
    rows, scorer = [], None
    cache = _score_cache_read()
    for i, s in enumerate(syms):
        try:
            fresh = gw.news(s, limit=limit)
            err = None
        except Exception as e:
            fresh, err = [], str(e)[:120]
        raw = {key(s, h): h for h in load_raw(s) + list(fresh)}
        raw = [h for h in raw.values() if to_ts(h["time"]) < now_ts]
        recent = [h for h in raw if to_ts(h["time"]) >= now_ts - (NOV_DAYS + 7) * 86400]
        if recent:
            todo = [h for h in recent if key(s, h) not in cache]
            if todo:
                scorer = scorer or Scorer.get()
                n, c, e = scorer.score([h["headline"] for h in todo])
                cache.update({key(s, h): (float(n[k]), float(c[k]), e[k]) for k, h in enumerate(todo)})
            T = symbol_table(s, recent, cache)
        else:
            T = None
        rows.append(_row(s, T, raw, now_ts, last_close, prev_close, M, risk, err))
        if i < len(syms) - 1 and pace:
            time.sleep(pace)
    res = {"generated_utc": datetime.fromtimestamp(now_ts, timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "since_close_et": datetime.fromtimestamp(last_close, ET).strftime("%Y-%m-%d %H:%M"),
           "risk_as_of": as_of, "pre_variant": M["pre_variant"], "uplift_passed_oos": M["pre_passed"],
           "seconds": round(time.time() - t_start, 1), "seconds_per_symbol": round((time.time() - t_start) / max(len(syms), 1), 2),
           "mode": mode, "rows": rows, "track_record": M["track_record"]}
    if mode == "scan":
        res["scanned"] = len(syms)
        res["rows"] = sorted([r for r in rows if r["n_since_close"] > 0],
                             key=lambda r: (-(r["uplift"]["mult"] or 0), -r["n_since_close"]))[:top]
        if write_index is not False:
            INDEX.write_text(json.dumps({**res, "all": rows}, indent=1), encoding="utf-8")
    return res


def _score_cache_read():
    """Scores already computed for cached headlines (FinBERT and embeddings are deterministic)."""
    try:
        z = np.load(SCORES, allow_pickle=False)
        K, N, C, E = z["keys"], z["net"], z["conf"], z["emb"].astype(np.float32)
        return {k: (float(N[i]), float(C[i]), E[i]) for i, k in enumerate(K)}
    except (OSError, ValueError, KeyError):
        return {}


def _row(s, T, raw, now_ts, last_close, prev_close, M, risk, err):
    pre, day = window(T, last_close, now_ts), window(T, prev_close, last_close)
    n30 = count(T, now_ts - 30 * 86400, now_ts)
    n30c = count(T, last_close - 30 * 86400, last_close)
    n5 = len([h for h in raw if last_close - 7 * 86400 <= to_ts(h["time"]) < last_close])
    f = feats_from(pre, day, n30, n30c, n5)
    name = M["pre_variant"]
    out = {"symbol": s, "n_since_close": pre["n"], "n_prev_session": day["n"], "n_30d": n30}
    if err:
        out["error"] = err
    idx = np.concatenate([day["idx"], pre["idx"]]).astype(int) if T is not None else np.zeros(0, int)
    if len(idx):
        net = T["net"][idx]
        spread = (1.645 * net.std(ddof=1) / np.sqrt(len(idx))) if len(idx) > 1 else float(1 - T["conf"][idx][0])
        w = M["direction"]["w"]
        out["novelty"] = round(float(T["nov"][idx].max()), 2)
        out["sentiment"] = {"finbert_net": round(float(net.mean()), 2), "plus_minus": round(float(spread), 2), "n": int(len(idx)),
                            "analyst_up": int(T["up"][idx].sum()), "analyst_down": int(T["down"][idx].sum()),
                            "p_up_open_to_close": round(float(logit_pred(w, f["score"])), 3)}
        order = sorted(idx, key=lambda i: (-T["nov"][i], -T["ts"][i]))[:2]
        out["top"] = [f"{T['time'][i][:16].replace('T', ' ')}Z {T['prov'][i]}: {T['text'][i][:140]}" for i in order]
    else:
        out["novelty"], out["sentiment"] = None, None
        last = max(raw, key=lambda h: h["time"]) if raw else None
        out["top"] = [f"(none since the prior session; latest {last['time'][:10]}: {clean(last['headline'])[:120]})"] if last else []
    um = {}
    for tgt in ("absr1", "range1"):
        m, lo, hi = _mult(M["models"][f"{name}_{tgt}"], f)
        um[tgt] = (m, lo, hi)
    m, lo, hi = um["absr1"]
    out["uplift"] = {"mult": round(m, 2), "ci90": [round(lo, 2), round(hi, 2)]}
    r = risk.get(s) if risk else None
    if r and r.get("absr1_pct") is not None:
        news = pre["n"] + day["n"] > 0
        for tgt, col in (("absr1", "absr1_pct"), ("range1", "range1_pct")):
            Mt = M["models"][f"{name}_{tgt}"]
            q = Mt["q"].get("news" if news else "none", Mt["q"]["all"])
            e = r[col] * um[tgt][0]
            out[f"exp_{tgt}_pct"] = {"risk": r[col], "with_news": round(e, 2), "p10_p90": [round(e * q[0], 2), round(e * q[1], 2)]}
    else:
        out["exp_absr1_pct"] = None
        out["note"] = "no risk forecast for this symbol (not in the cached universe or no forecast file)"
    return out


def format_text(res):
    head = (f"news since the close of {res['since_close_et']} ET (generated {res['generated_utc']} UTC; "
            f"{res['seconds_per_symbol']} s/symbol)")
    if res["mode"] == "scan":
        head += f"; scanned {res['scanned']}, {len(res['rows'])} with news shown"
    lines = [head, ("columns: new = headlines since the close; prev = in the session before; novelty 0-1; FinBERT sentiment -1..1 "
                    "(+- spread); P(up o->c) = calibrated chance of beating the average stock open-to-close (no skill: ~0.50); "
                    "uplift = expected |move| multiple of the risk forecast [90% interval of that average]; exp|move|% "
                    "[10-90% range of outcomes]"),
             "sym new prev novelty sentiment(+-) P(up o->c) uplift[90%] exp|move|% [p10-p90] | top headlines"]
    for r in res["rows"]:
        s = r["sentiment"]
        sent = f"{s['finbert_net']:+.2f}(+-{s['plus_minus']:.2f})" if s else "na"
        pu = f"{s['p_up_open_to_close']:.2f}" if s else "na"
        e = r.get("exp_absr1_pct")
        em = f"{e['with_news']:.2f} [{e['p10_p90'][0]:.2f}-{e['p10_p90'][1]:.2f}]" if e else "na (no risk forecast)"
        u = r["uplift"]
        lines.append(f"{r['symbol']} {r['n_since_close']} {r['n_prev_session']} {r['novelty'] if r['novelty'] is not None else 'na'} "
                     f"{sent} {pu} x{u['mult']:.2f}[{u['ci90'][0]:.2f}-{u['ci90'][1]:.2f}] {em} | " + " || ".join(r["top"])
                     + (f" (error: {r['error']})" if r.get("error") else ""))
    return "\n".join(lines)


def read_index(max_age_min=45):
    """The latest scan written by a background refresher, if fresh enough."""
    try:
        res = json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    age = (datetime.now(timezone.utc) - datetime.strptime(res["generated_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
    return res if age.total_seconds() <= max_age_min * 60 else None


# ---------------------------------------------------------------- training simulator
def sim_provider(data):
    """Function of the decision session d -> (as_of, {symbol: row}) as of the close of d-1: headlines published
    before 16:00 ET of d-1 only, and the day_full multiplier from the walk-forward OOS run (NaN before the test window,
    so rows there carry no uplift). Only symbols with a headline in the last five sessions are listed."""
    z = np.load(OOS, allow_pickle=True)
    syms, dates = [str(s) for s in z["symbols"]], [str(x) for x in z["dates"]]
    col = {x: i for i, x in enumerate(dates)}
    mult, base = z["day_full_absr1_mult"], z["base_absr1"]
    lo, hi = z["day_full_absr1_lo"], z["day_full_absr1_hi"]
    tables = load_tables([s for s in data.sym if s not in data.etf])
    close = {x: et_ts(x, 16) for x in data.dates}
    jx = {s: j for j, s in enumerate(syms)}

    def at(d):
        if d < 6:
            return None
        as_of = data.dates[d - 1]
        c1, c2, c6 = close[data.dates[d - 1]], close[data.dates[d - 2]], close[data.dates[d - 6]]
        i = col.get(data.dates[d])                  # session-d forecast column (built only from data to close d-1)
        out = {}
        for s, T in tables.items():
            if T is None or count(T, c6, c1) == 0:
                continue
            day = window(T, c2, c1)
            idx = day["idx"]
            j = jx.get(s)
            row = {"n_prev_session": day["n"], "n_5d": count(T, c6, c1)}
            if len(idx):
                row["novelty"] = round(float(T["nov"][idx].max()), 2)
                row["finbert_net"] = round(float(T["net"][idx].mean()), 2)
                row["top"] = [f"{T['time'][k][:16]}Z {T['prov'][k]}: {T['text'][k][:140]}"
                              for k in sorted(idx, key=lambda k: (-T["nov"][k], -T["ts"][k]))[:2]]
            if i is not None and j is not None and np.isfinite(mult[j, i]):
                row["uplift_mult"] = round(float(mult[j, i]), 2)
                row["exp_absr1_pct"] = round(float(base[j, i] * mult[j, i] * 100), 2)
                row["p10_p90_pct"] = [round(float(lo[j, i] * 100), 2), round(float(hi[j, i] * 100), 2)]
            out[s] = row
        return (as_of, out) if out else None
    return at


# ---------------------------------------------------------------- the LLM tool
SPEC = {
    "name": "news",
    "description": ("News pre-filter: which names have fresh headlines since the last close, how unusual they are, and how "
                    "much bigger than normal the next move is likely to be. Use it to decide where to spend research calls, "
                    "and for sizing and stop distances. It does not tell you direction: sentiment did not predict "
                    "open-to-close returns, because the news is already in the opening price. Source: IBKR's Briefing.com "
                    "columns and analyst actions only (thin: about 6% of stock-days have any headline). Track record: "
                    + TRACK_RECORD),
    "parameters": {"type": "object", "properties": {
        "symbols": {"type": "array", "items": {"type": "string"}, "description": f"Up to {MAX_SYMBOLS} symbols."},
        "mode": {"type": "string", "enum": ["symbols", "scan"],
                 "description": "scan = the names with news since the close across the universe, biggest uplift first "
                                "(from the background index; up to 15)."}}},
}


def run(gw, args):
    """Tool entry point: text for the LLM."""
    mode = args.get("mode") or ("symbols" if args.get("symbols") else "scan")
    if hasattr(gw, "news_provider"):                 # training simulator: point-in-time rows as of the close of d-1
        got = gw.news_provider()
        if not got:
            return "no news data for this date"
        as_of, rows = got
        want = [str(x).upper() for x in (args.get("symbols") or [])][:MAX_SYMBOLS]
        items = [(k, rows[k]) for k in want if k in rows] if want else sorted(rows.items(), key=lambda kv: -(kv[1].get("uplift_mult") or 0))[:15]
        lines = [f"news through the close of {as_of} (sym: headlines prev session, 5 sessions, novelty, FinBERT, uplift, "
                 f"exp |move|% [p10-p90] | top)"]
        for k, v in items:
            lines.append(f"{k} {v['n_prev_session']} {v['n_5d']} {v.get('novelty', 'na')} {v.get('finbert_net', 'na')} "
                         f"x{v.get('uplift_mult', 'na')} {v.get('exp_absr1_pct', 'na')} {v.get('p10_p90_pct', '')} | "
                         + " || ".join(v.get("top", [])))
        for k in want:
            if k not in rows:
                lines.append(f"{k} 0 0: no headlines in the last five sessions")
        return "\n".join(lines)
    if mode == "scan":
        idx = read_index()
        if not idx:
            return ("no fresh news index (the background scan is not running or is older than 45 minutes); ask for up to "
                    f"{MAX_SYMBOLS} symbols instead")
        return format_text(idx)
    syms = args.get("symbols") or []
    if not syms:
        return "error: give at least one symbol, or mode scan"
    return format_text(live(gw, syms))


# ---------------------------------------------------------------- report
def report(res):
    V = res["variants"]
    L = ["# News tool: out-of-sample results", "",
         f"*Generated by `ml_research/news.py`. Test window {res['test']}. Training starts {res['train_from']}. Monthly "
         f"walk-forward refits, embargo 1 session. The pre-registration is `news_prereg.md`.*", "",
         "## Data", ""]
    for k, v in res["data"].items():
        L.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")
    L += ["", "## Variants 1–4: does news add to risk.py's next-session forecast?", "",
          "| Variant | Target | R² news model | R² risk.py | MSE-gain t | News-row R² (model / risk) | Realized / forecast ratio on news rows | 10–90% coverage (all / news) | Pass |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in VARIANTS:
        for t in ("absr1", "range1"):
            c = V[n][t]
            nr, cv = c["news_rows"], c["interval_10_90_coverage"]
            L.append(f"| {n} | {t} | {c['r2_news_model']:.4f} | {c['r2_risk_base']:.4f} | {c['mse_gain_t']:.2f} | "
                     f"{nr['r2_news_model']:.3f} / {nr['r2_risk_base']:.3f} | {nr['mean_realized_over_base']:.2f} / "
                     f"{nr['mean_forecast_over_base']:.2f} | {cv['all']:.0%} / {cv['news_rows']:.0%} | {'yes' if c['pass'] else 'no'} |")
    L += ["", "### By half-year (R² news model / risk.py)", ""]
    for n in VARIANTS:
        for t in ("absr1", "range1"):
            L.append(f"- {n} {t}: " + "; ".join(f"{h} {v['r2_news_model']:.3f} / {v['r2_risk_base']:.3f}"
                                                 for h, v in V[n][t]["by_half"].items()))
    p = res["shipped"]["pre_variant"]
    L += ["", f"### Uplift calibration ({p}, absr1, OOS): forecast multiplier bin vs realized |move| / risk forecast", "",
          "| Bin | n | News share | Mean forecast | Mean realized | 90% interval (date bootstrap) |", "|---|---|---|---|---|---|"]
    for b in V[p]["absr1"]["uplift_reliability"]:
        L.append(f"| {b['mult_bin']} | {b['n']} | {b['news_share']:.2f} | {b['mean_forecast_mult']} | {b['mean_realized_ratio']} | {b['realized_90ci']} |")
    L += ["", "## Variant 5: direction (sentiment to next-session relative return)", ""]
    for t in ("rel_oc", "rel_cc"):
        c = V["pre_sent_ret"][t]
        L.append(f"- **{t}:** long-short by sentiment sign {c['longshort_gross_pct']:+.3f}% a day gross (t {c['longshort_t']:.2f}, "
                 f"{c['longshort_dates']} dates). Net of costs: {c['longshort_net_pct']:+.3f}%. By half-year: {c['longshort_by_half_pct']}. "
                 f"P(up) Brier {c['brier']:.4f} vs {c['brier_base_rate']:.4f} for the base rate (skill {c['brier_skill']:+.4f}). "
                 f"Pass: {c['pass'] if isinstance(c['pass'], str) else ('yes' if c['pass'] else 'no')}.")
        L.append(f"  - Events: positive score {c['events']['pos_mean_pct']:+.3f}% (n {c['events']['pos_n']}). "
                 f"Negative score {c['events']['neg_mean_pct']:+.3f}% (n {c['events']['neg_n']}). Pooled Spearman {c['events']['spearman_pooled']:.3f}.")
        L.append("  - Reliability: " + "; ".join(f"pred {r['pred']:.3f} obs {r['observed']:.3f} (n {r['n']})" for r in c["reliability"]))
    L += ["", "## Post hoc (not variants)", ""]
    for t in ("absr1", "range1"):
        c = res["post_hoc"][f"news_dummy_{t}"]
        L.append(f"- **News dummy only, {t}:** R² {c['r2_news_model']:.4f} vs {c['r2_risk_base']:.4f}, t {c['mse_gain_t']:.2f}.")
        iv = res["post_hoc"]["iv_control"][t]
        L.append(f"- **Implied vol control, {t}:** IV only R² {iv['iv_only']['r2_news_model']:.4f}. IV plus pre_full news "
                 f"{iv['iv_plus_pre_full']['r2_news_model']:.4f}, vs risk.py {iv['iv_only']['r2_risk_base']:.4f}. On the same cells, news "
                 f"beyond IV: {iv['news_beyond_iv']['r2_iv_plus_news']:.4f} vs {iv['news_beyond_iv']['r2_iv_only']:.4f}.")
    if "absr1_by_pre_count" in res["post_hoc"]:
        L += ["", "### |move| by headlines since the prior close (pre_full, OOS, descriptive)", "",
              "| n_pre | Cells | Realized / risk forecast | Forecast multiplier |", "|---|---|---|---|"]
        L += [f"| {b['n_pre']} | {b['cells']} | {b['mean_realized_ratio']} | {b['mean_forecast_mult']} |"
              for b in res["post_hoc"]["absr1_by_pre_count"]]
    if "gap_vs_intraday_by_pre_count" in res["post_hoc"]:
        L += ["", "### Where the extra move comes from (OOS means as a multiple of the risk forecast, descriptive)", "",
              "| n_pre | Cells | \|gap\| | \|open to close\| | \|close to close\| | Range / range1 |", "|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['abs_gap/risk_absr1']} | {v['abs_open_to_close/risk_absr1']} | {v['absr1/risk']} | "
              f"{v['range/risk_range1']} |" for k, v in res["post_hoc"]["gap_vs_intraday_by_pre_count"].items()]
    L += ["", "## Live tool", "",
          "- **Call:** `live(gw, symbols)` takes up to 20 symbols. `live(gw, mode=\"scan\", universe=[...])` returns the names with news since the close, biggest uplift first. `format_text()` renders either for the LLM, and `run(gw, args)` with `SPEC` is the tool entry point.",
          "- **Each row:** headlines since the close and in the session before, novelty, FinBERT sentiment ± spread, and a calibrated P(beat the average stock open-to-close), which is about 0.50 because there is no skill. Also the uplift multiple of the risk forecast with its 90% interval, the expected |move| and range with 10–90% outcome intervals (when risk.py covers the name), and the top 2 headlines.",
          "- **Cost:**",
          "  - One read-only `gw.news()` call per symbol takes 0.5–0.8 s, and pacing makes it about 1.5 s a symbol. The full fetch did 466 symbols in 706 s.",
          "  - Scoring new headlines is about 0.1 s each, after a one-time model load of 10–30 s, which ran slower under CPU load from parallel jobs. Headlines already in `news_scores.npz` are not rescored.",
          "  - 20 symbols take about 30 s. A universe scan of 464 stocks takes about 12 minutes, too slow to run inside a look.",
          "- **Refresh proposal (not set up):**",
          "  - A separate background process (client `trader-research`, news calls only) loops over the universe at 1 request a second. Each pass takes about 12 minutes.",
          "  - Each pass writes `runs/ml/research/news_index.json` through `live(gw, mode=\"scan\")`. The tool's scan mode serves the index if it is under 45 minutes old.",
          "  - Run one pass from 08:35 ET. Briefing.com's analyst actions land between 07:00 and 08:50 ET, so the 09:00 look sees them.",
          "  - BRFUPDN has no broad tape, so `gw.tape()` cannot replace the per-symbol loop.",
          "  - After an exchange holiday, pass `last_session`, so that \"since the close\" is measured from the true last close.",
          ]
    L += ["", "## Shipped", "", "```", json.dumps(res["shipped"], indent=1), "```", ""]
    (OUT / "news.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- fake gateway test
class FakeGW:
    """Serves cached headlines as they stood at `now` (newest first), plus a synthetic uncached symbol."""

    def __init__(self, now):
        self.now = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.calls = 0

    def news(self, symbol, providers=None, limit=10, before=""):
        self.calls += 1
        if symbol == "ZZZZ":
            raise RuntimeError("200: No security definition has been found for the request")
        if symbol == "NEWCO":
            return [{"time": "2026-10-16T12:30:00Z", "provider": "BRFUPDN", "article_id": "X1", "meta": "",
                     "headline": "!Goldman upgraded NewCo (NEWCO) to Buy with target $50"},
                    {"time": "2026-10-15T14:10:00Z", "provider": "BRFG", "article_id": "X2", "meta": "",
                     "headline": "NewCo shares slide after guidance cut"}]
        rows = [h for h in load_raw(symbol) if h["time"] < self.now]
        return sorted(rows, key=lambda h: h["time"], reverse=True)[:limit]

    def quotes(self, symbols):
        return []


def test_live():
    now = datetime(2026, 10, 16, 13, 0, tzinfo=timezone.utc)            # 09:00 ET, Friday 16 Oct 2026
    gw = FakeGW(now)
    syms = ["AAPL", "NVDA", "JNJ", "XOM", "NEWCO", "ZZZZ"]
    busy = []
    for s in [x.strip() for x in (ROOT / "data" / "universe523.txt").read_text().splitlines() if x.strip()]:
        if any(et_ts("20261015", 16) <= to_ts(h["time"]) < to_ts(gw.now) for h in load_raw(s)):
            busy.append(s)
    syms += busy[:6]
    res = live(gw, syms, now=now, pace=0)
    txt = format_text(res)
    scan = live(gw, mode="scan", universe=busy[:10] + ["AAPL", "ZZZZ"], now=now, pace=0, write_index=False)
    out = {"symbols": res, "scan": scan, "text": txt, "scan_text": format_text(scan), "calls": gw.calls}
    (OUT / "news_livetest.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(txt)
    print(format_text(scan))
    return out


if __name__ == "__main__":
    if "--live" in sys.argv:
        test_live()
    elif "--report" in sys.argv:
        report(refresh_track_record())
    else:
        r = validate()
        report(r)
