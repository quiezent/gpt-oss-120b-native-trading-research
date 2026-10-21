"""Industry family: does IBKR industry structure forecast 5- and 20-session cross-sectional stock returns?

Pre-registered in runs/ml/research/industry_prereg.md (written before any run). Five variants, each scored at h=5 and h=20:
  V1 industry momentum, V2 stock-minus-industry reversal, V3 industry-neutralized simple signals (composite),
  V4 generic LightGBM (ml.FEATS, stocks only), V5 generic + industry-relative features LightGBM.

Timing: every feature uses bars through the close of t; the target is log(close[t+h] / close[t]) demeaned over stocks.
Industry means are equal-weighted and leave-one-out (the stock itself excluded). ETFs are excluded (prediction NaN).

    .venv\\Scripts\\python.exe ml_research\\industry.py run      # all variants -> runs/ml/research/industry.json, _oos.npz
    .venv\\Scripts\\python.exe ml_research\\industry.py today    # best variant's predictions from the latest cached close
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml          # noqa: E402
import ml_eval     # noqa: E402
import trader      # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "runs" / "ml" / "research"
IND_FILE = ROOT / "lists" / "industries.json"
TEST_FROM = "20250102"
MIN_GROUP = 5
HORIZONS = (5, 20)
STEPS_FROM = 130
RETRAIN = 21
IND_FEATS = ["ind_r21_rk", "ind_r63_rk", "ind_r126_rk", "ind_r12_1_rk", "rel_r1_rk", "rel_r5_rk", "rel_r20_rk",
             "rel_vol20_rk", "rel_offhi_rk"]


def groups(d):
    """Group id per symbol row: IBKR category if it has >= MIN_GROUP stocks here, else IBKR industry; -1 for ETFs/unknown."""
    raw = json.loads(IND_FILE.read_text(encoding="utf-8"))
    cat = {s: (raw.get(s) or {}).get("category") for s in d.sym if s not in d.etf}
    ind = {s: (raw.get(s) or {}).get("industry") for s in d.sym if s not in d.etf}
    ncat = {}
    for s, c in cat.items():
        if c:
            ncat[c] = ncat.get(c, 0) + 1
    label = {}
    for s in cat:
        if cat[s] and ncat[cat[s]] >= MIN_GROUP:
            label[s] = "cat:" + cat[s]
        elif ind[s]:
            label[s] = "ind:" + ind[s]
    names = sorted(set(label.values()))
    gid = {n: i for i, n in enumerate(names)}
    g = np.array([gid[label[s]] if s in label else -1 for s in d.sym])
    return g, names, label


def loo_mean(x, g):
    """Leave-one-out equal-weighted group mean of x (symbols, days); NaN when fewer than 2 other members have data."""
    out = np.full_like(x, np.nan)
    for k in np.unique(g[g >= 0]):
        m = g == k
        xs = x[m]
        ok = ~np.isnan(xs)
        s, n = np.nansum(xs, 0), ok.sum(0)
        num, den = s[None, :] - np.where(ok, xs, 0.0), n[None, :] - ok
        out[m] = np.where(den >= 2, num / np.maximum(den, 1), np.nan)
    return out


def lagret(lc, a, b=0):
    """Log return from t-a to t-b."""
    out = np.full_like(lc, np.nan)
    out[:, a:] = lc[:, a - b:lc.shape[1] - b] - lc[:, :lc.shape[1] - a]
    return out


def stock_rank(x, stock):
    """Cross-sectional percentile rank among stocks only (ETF/unclassified rows NaN)."""
    return ml.rank_pct(np.where(stock[:, None], x, np.nan))


def build(d):
    g, names, label = groups(d)
    stock = g >= 0
    lc = np.log(d.c)
    R = {n: lagret(lc, n) for n in (1, 5, 20, 21, 63, 126)}
    R["12_1"] = lagret(lc, 252, 21)
    I = {n: loo_mean(np.where(stock[:, None], r, np.nan), g) for n, r in R.items()}
    vol20 = ml.roll(d.lr, 20, lambda w, axis: np.nanstd(w, axis=axis, ddof=1)) * np.sqrt(252)
    offhi = d.c / ml.roll(d.h, 60, np.nanmax) - 1
    F = {"ind_r21_rk": stock_rank(I[21], stock), "ind_r63_rk": stock_rank(I[63], stock),
         "ind_r126_rk": stock_rank(I[126], stock), "ind_r12_1_rk": stock_rank(I["12_1"], stock),
         "rel_r1_rk": stock_rank(R[1] - I[1], stock), "rel_r5_rk": stock_rank(R[5] - I[5], stock),
         "rel_r20_rk": stock_rank(R[20] - I[20], stock),
         "rel_vol20_rk": stock_rank(vol20 / loo_mean(np.where(stock[:, None], vol20, np.nan), g), stock),
         "rel_offhi_rk": stock_rank(offhi - loo_mean(np.where(stock[:, None], offhi, np.nan), g), stock)}
    raw = {"r5": stock_rank(R[5], stock), "r12_1": stock_rank(R["12_1"], stock), "vol20": stock_rank(vol20, stock)}
    rel12 = stock_rank(R["12_1"] - I["12_1"], stock)
    sig = {
        "V1_industry_momentum": np.nanmean([F["ind_r21_rk"], F["ind_r63_rk"], F["ind_r126_rk"], F["ind_r12_1_rk"]], 0),
        "V2_rel_reversal": -F["rel_r5_rk"],
        "V3_neutral_composite": np.nanmean([rel12, -F["rel_r5_rk"], -F["rel_vol20_rk"]], 0),
    }
    sig["V3_neutral_composite"] = np.where(np.isnan(rel12) | np.isnan(F["rel_r5_rk"]) | np.isnan(F["rel_vol20_rk"]),
                                           np.nan, sig["V3_neutral_composite"])
    sig["V1_industry_momentum"] = np.where(np.isnan(F["ind_r12_1_rk"]), np.nan, sig["V1_industry_momentum"])
    diag = {  # pre-registered V3 components and raw counterparts; V1 components (post-hoc, not selectable)
        "V3c_rel_mom12_1": rel12, "V3c_rel_reversal5": -F["rel_r5_rk"], "V3c_rel_lowvol20": -F["rel_vol20_rk"],
        "raw_mom12_1": raw["r12_1"], "raw_reversal5": -raw["r5"], "raw_lowvol20": -raw["vol20"],
        "V1c_ind_r21": F["ind_r21_rk"], "V1c_ind_r63": F["ind_r63_rk"], "V1c_ind_r126": F["ind_r126_rk"],
        "V1c_ind_r12_1": F["ind_r12_1_rk"], "posthoc_rel_reversal1": -F["rel_r1_rk"],
    }
    return g, names, label, stock, F, sig, diag, lc


def target(lc, h, stock):
    fwd = np.full_like(lc, np.nan)
    fwd[:, :-h] = lc[:, h:] - lc[:, :-h]
    fwd = np.where(stock[:, None], fwd, np.nan)
    return fwd - np.nanmean(fwd, axis=0, keepdims=True)


def lgb_wf(X, names, y, t0, h, stock, keep):
    def fit(Xa, ya):
        lo, hi = np.nanpercentile(ya, [1, 99])
        m = ml.fit(Xa, np.clip(ya, lo, hi), "huber")
        keep.append(m)
        return m
    pred = ml_eval.walk_forward(lambda idx: ml.rows(X, names, idx), y, steps_from=STEPS_FROM, test_from=t0, horizon=h,
                                retrain_every=RETRAIN, fit=fit, predict=lambda m, Xa: m.predict(Xa))
    return np.where(stock[:, None], pred, np.nan)


def run():
    t_start = time.time()
    d = trader.Data()
    n = len(d.dates)
    t0 = next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    g, names, label, stock, F, sig, diag, lc = build(d)
    X, *_ = ml.panel(d)
    X = {**X, **F}
    print(f"built features: {stock.sum()} stocks in {len(names)} groups, {time.time() - t_start:.0f}s", flush=True)
    mask = np.zeros(n, bool)
    mask[t0:] = True
    cards, preds, imp = {}, {}, {}
    for h in HORIZONS:
        y = target(lc, h, stock)
        steps = list(range(t0, n - h))
        for k, s in {**sig, **diag}.items():
            p = np.where(mask[None, :] & stock[:, None], s, np.nan)
            cards[f"{k}_h{h}"] = ml_eval.score(p, y, steps, h, d.dates)
            if k in sig:
                preds[f"{k}_h{h}"] = p
        for k, feats in (("V4_generic_lgbm", ml.FEATS), ("V5_generic_plus_industry_lgbm", ml.FEATS + IND_FEATS)):
            keep = []
            p = lgb_wf(X, feats, y, t0, h, stock, keep)
            cards[f"{k}_h{h}"] = ml_eval.score(p, y, steps, h, d.dates)
            preds[f"{k}_h{h}"] = p
            gain = np.mean([m.feature_importance("gain") for m in keep], 0)
            imp[f"{k}_h{h}"] = {f: round(float(v / gain.sum()), 4) for f, v in sorted(zip(feats, gain), key=lambda z: -z[1])}
            print(f"{k} h{h}: IC {cards[f'{k}_h{h}']['ic']:+.4f} t {cards[f'{k}_h{h}']['ic_t']:+.2f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    variants = [k for k in cards if k.startswith(("V1_", "V2_", "V3_", "V4_", "V5_"))]
    passing = [k for k in variants if cards[k]["pass"]]
    best = max(passing or variants, key=lambda k: cards[k]["ic_t"])
    out = {"test": f"{d.dates[t0]}..{d.dates[n - 1]} (scored through the last realized target)",
           "groups": {"n_groups": len(names), "n_stocks": int(stock.sum()),
                      "sizes": {nm: int((g == i).sum()) for i, nm in enumerate(names)}},
           "variants": {k: cards[k] for k in variants},
           "diagnostics_not_selectable": {k: v for k, v in cards.items() if k not in variants},
           "lgbm_gain_share": imp, "best": best, "best_passes": cards[best]["pass"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "industry.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "industry_oos.npz", pred=preds[best], symbols=np.array(d.sym), dates=np.array(d.dates),
                        variant=np.array(best))
    print("best:", best, "pass:", cards[best]["pass"], f"({time.time() - t_start:.0f}s)")
    return out


def today(variant=None, d=None):
    """Predictions from the latest cached close for the saved best variant (or `variant`, e.g. 'V2_rel_reversal_h5').
    Returns (as_of, {symbol: score}); higher = expected to outperform other stocks over the variant's horizon.
    Fixed-sign signals are percentile ranks in [-0.5, 0.5]; LightGBM variants are refit on all realized targets."""
    variant = variant or json.loads((OUT / "industry.json").read_text(encoding="utf-8"))["best"]
    d = d or trader.Data()
    g, names, label, stock, F, sig, diag, lc = build(d)
    t = len(d.dates) - 1
    name, h = variant.rsplit("_h", 1)
    h = int(h)
    if name in sig:
        p = sig[name][:, t]
    else:
        X, *_ = ml.panel(d)
        X = {**X, **F}
        feats = ml.FEATS if name.startswith("V4") else ml.FEATS + IND_FEATS
        y = target(lc, h, stock)
        tr = np.arange(STEPS_FROM, t - h + 1)
        Xa, ya = ml.rows(X, feats, tr), y[:, tr].T.reshape(-1)
        ok = ~np.isnan(ya) & ~np.all(np.isnan(Xa), axis=1)
        lo, hi = np.nanpercentile(ya[ok], [1, 99])
        m = ml.fit(Xa[ok], np.clip(ya[ok], lo, hi), "huber")
        p = np.where(stock, m.predict(ml.rows(X, feats, [t])), np.nan)
    return d.dates[t], {s: round(float(p[j]), 4) for j, s in enumerate(d.sym) if not np.isnan(p[j])}


if __name__ == "__main__":
    if sys.argv[1:] == ["today"]:
        as_of, sc = today()
        top = sorted(sc.items(), key=lambda z: -z[1])
        print(as_of, len(sc), "stocks; top 10:", top[:10], "bottom 10:", top[-10:])
    else:
        run()
