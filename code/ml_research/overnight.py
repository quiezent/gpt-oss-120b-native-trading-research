"""Overnight vs intraday ("tug of war", Lou-Polk-Skouras) research family. Pre-registration: runs/ml/research/overnight.md.

    python ml_research/overnight.py          # run the five pre-registered variants, write runs/ml/research/overnight*
    python ml_research/overnight.py today    # print today's forecast from the latest cached bars (best variant)

Components per session t: ON[t] = log(open_t / close_{t-1}), ID[t] = log(close_t / open_t).
(a) intraday target close_t/open_t - 1, decision at the open of t (features: closes through t-1 plus the open of t).
(b) next-overnight target open_{t+1}/close_t - 1, decision at the close of t (features: bars through the close of t).
Stocks only (ETFs excluded). Survivor-biased universe, so every figure is optimistic.
"""

import json
import sys
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
TEST_FROM = "20250102"
RETRAIN = 21
FIRST = 25          # 20-session sums and vol20 need history
FEATS_A = ["gap_rk", "on1_rk", "on5_rk", "on20_rk", "id1_rk", "id5_rk", "id20_rk", "gapz_rk", "vol20_rk", "dvol_rk",
           "vol20", "spy_gap", "spy_id1"]
FEATS_B = ["on1_rk", "on5_rk", "on20_rk", "id1_rk", "id5_rk", "id20_rk", "vol20_rk", "dvol_rk", "vol20", "spy_on1", "spy_id1"]


def lag(x, n=1):
    return np.concatenate([np.full((x.shape[0], n), np.nan), x[:, :-n]], axis=1)


def tsum(x, n):
    """Trailing sum over sessions t-n+1..t (NaN if any is missing)."""
    return ml.roll(x, n, np.sum)


def rank(x, mask):
    """Cross-sectional percentile rank in [-0.5, 0.5] among rows in mask; others NaN."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        col = np.where(mask, x[:, t], np.nan)
        ok = ~np.isnan(col)
        if ok.sum() > 10:
            out[ok, t] = col[ok].argsort().argsort() / (ok.sum() - 1) - 0.5
    return out


def build(d):
    """Features and targets for both decision times. Arrays are (symbols, sessions)."""
    st = np.array([s not in d.etf for s in d.sym])
    spy = d.k["SPY"]
    with np.errstate(all="ignore"):
        ON = np.log(d.o / lag(d.c))
        ID = np.log(d.c / d.o)
        vol20 = ml.roll(d.lr, 20, lambda w, axis: np.std(w, axis=axis, ddof=1)) * np.sqrt(252)
        dvol = np.log(ml.roll(d.c * d.v, 20, np.mean))
        yA = d.c / d.o - 1
        yB = np.concatenate([d.o[:, 1:] / d.c[:, :-1] - 1, np.full((d.c.shape[0], 1), np.nan)], axis=1)
    yA[~st], yB[~st] = np.nan, np.nan
    bc = lambda row: np.broadcast_to(row, d.c.shape)
    # (a) at the open of t: past components end at t-1; today's gap is ON[t]
    raw_a = {"gap": ON, "on1": lag(ON), "on5": lag(tsum(ON, 5)), "on20": lag(tsum(ON, 20)), "id1": lag(ID),
             "id5": lag(tsum(ID, 5)), "id20": lag(tsum(ID, 20)), "vol20": lag(vol20), "dvol": lag(dvol)}
    raw_a["gapz"] = ON / (raw_a["vol20"] / np.sqrt(252))
    A = {k + "_rk": rank(v, st) for k, v in raw_a.items()}
    A.update({"vol20": np.where(st[:, None], raw_a["vol20"], np.nan), "spy_gap": bc(ON[spy]), "spy_id1": bc(lag(ID)[spy])})
    # (b) at the close of t: components end at t
    raw_b = {"on1": ON, "on5": tsum(ON, 5), "on20": tsum(ON, 20), "id1": ID, "id5": tsum(ID, 5), "id20": tsum(ID, 20),
             "vol20": vol20, "dvol": dvol}
    B = {k + "_rk": rank(v, st) for k, v in raw_b.items()}
    B.update({"vol20": np.where(st[:, None], vol20, np.nan), "spy_on1": bc(ON[spy]), "spy_id1": bc(ID[spy])})
    return {"st": st, "A": A, "B": B, "yA": yA, "yB": yB, "rawA": raw_a, "rawB": raw_b}


def demean_pct(y):
    return (y - np.nanmean(y, axis=0, keepdims=True)) * 100


def fit(X, y):
    lo, hi = np.nanpercentile(y, [1, 99])
    return ml.fit(X, np.clip(y, lo, hi), "huber")


def lgb_oos(M, names, y, st, t0):
    yd = demean_pct(y)
    pred = ml_eval.walk_forward(lambda idx: ml.rows(M, names, idx), yd, steps_from=FIRST, test_from=t0, horizon=1,
                                retrain_every=RETRAIN, fit=fit, predict=lambda m, X: m.predict(X))
    pred[~st] = np.nan
    return pred


def only_test(sig, t0):
    p = np.array(sig, float, copy=True)
    p[:, :t0] = np.nan
    return p


def run():
    d = trader.Data()
    F = build(d)
    st, t0 = F["st"], next(i for i, s in enumerate(d.dates) if s >= TEST_FROM)
    n = len(d.dates)
    preds = {
        "A_gap_rule": only_test(-F["rawA"]["gap"] * st[:, None], t0),
        "A_tow_rule": only_test(-F["rawA"]["on20"] * st[:, None], t0),
        "A_lgb": lgb_oos(F["A"], FEATS_A, F["yA"], st, t0),
        "B_on_rule": only_test(F["rawB"]["on20"] * st[:, None], t0),
        "B_lgb": lgb_oos(F["B"], FEATS_B, F["yB"], st, t0),
    }
    for p in preds.values():
        p[~st] = np.nan
    res = {}
    for name, p in preds.items():
        y = F["yA"] if name.startswith("A") else F["yB"]
        steps = [t for t in range(t0, n) if not np.all(np.isnan(y[:, t]))]
        res[name] = ml_eval.score(p, y, steps, 1, d.dates)
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}), flush=True)
    passing = [k for k, v in res.items() if v["pass"]]
    best = max(passing, key=lambda k: res[k]["spread_net_pct"]) if passing else max(res, key=lambda k: res[k]["ic_t"])
    out = {"family": "overnight", "test": f"{d.dates[t0]}..{d.dates[-1]}", "universe": "stocks only (464), survivor-biased",
           "cost_bps_round_trip": 10, "variants": res, "best": best, "best_passes": res[best]["pass"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overnight.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "overnight_oos.npz", pred=preds[best], symbols=np.array(d.sym), dates=np.array(d.dates))
    np.savez_compressed(OUT / "overnight_all_oos.npz", symbols=np.array(d.sym), dates=np.array(d.dates), **preds)
    print("best", best, res[best]["pass"])
    return out


def today(name=None, d=None):
    """Forecast from the latest cached bars for the best variant (or `name`). Returns {symbol: score}, higher = better.
    A_* variants need today's open: call after the open with bars that include today's session open."""
    name = name or json.loads((OUT / "overnight.json").read_text())["best"]
    d = d or trader.Data()
    F = build(d)
    t = len(d.dates) - 1
    if name == "A_gap_rule":
        s = -F["rawA"]["gap"][:, t]
    elif name == "A_tow_rule":
        s = -F["rawA"]["on20"][:, t]
    elif name == "B_on_rule":
        s = F["rawB"]["on20"][:, t]
    else:
        M, names, y = (F["A"], FEATS_A, F["yA"]) if name == "A_lgb" else (F["B"], FEATS_B, F["yB"])
        yd = demean_pct(y)
        tr = np.arange(FIRST, t - 1)
        X, yy = ml.rows(M, names, tr), yd[:, tr].T.reshape(-1)
        ok = ~np.isnan(yy) & ~np.all(np.isnan(X), axis=1)
        s = fit(X[ok], yy[ok]).predict(ml.rows(M, names, np.array([t])))
    return {d.sym[j]: float(s[j]) for j in range(len(d.sym)) if F["st"][j] and not np.isnan(s[j])}, d.dates[t]


def live_gap(opens, d=None):
    """Best variant (A_gap_rule) for a live session: `opens` = {symbol: today's open (or first price after the open)};
    previous close = the latest cached close. Returns {symbol: {"gap_pct", "score"}} for stocks, score = -gap
    (higher = more expected intraday rebound relative to other stocks)."""
    d = d or trader.Data()
    t = len(d.dates) - 1
    out = {}
    for s, px in opens.items():
        j = d.k.get(s)
        if j is None or s in d.etf or not px or np.isnan(d.c[j, t]):
            continue
        g = np.log(px / d.c[j, t])
        out[s] = {"gap_pct": float(g * 100), "score": float(-g)}
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "today":
        sc, date = today()
        top = sorted(sc.items(), key=lambda kv: -kv[1])
        print(date, "top", top[:10], "bottom", top[-10:])
    else:
        run()
