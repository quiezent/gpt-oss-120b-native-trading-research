"""Bond, credit and currency ETFs in `forecast` (acceptance criterion F4; pre-registration
runs/ml/research/bondfx_prereg.md, results bondfx.md and bondfx.json).

The pooled range1 and vol20 models carry a constant and market terms worth about 0.2 points of range on every symbol:
negligible on a stock, but on a low-vol bond fund range1 ran about 1.7x and its 10-90 band held 38-40%. These funds get
one pooled recalibration of a base forecast instead (A: the fund's own last 20 sessions; B: the current model's point),
one scale and two residual quantiles over the 8 listed funds, refit walk-forward; a column that no option brings to 70%
coverage on a group is withheld there. forecast.compute() and forecast._ondemand() (live) and forecast.sim_provider()
serve through apply(): the same base, formula and note, with the live fit or with the walk-forward fit as of the close.

    .venv\\Scripts\\python.exe ml_research\\bondfx.py      # validate(): bondfx.json, bondfx_oos.npz
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "ml" / "research"
GROUPS = {"TLT": "Treasury", "IEF": "Treasury", "SHY": "Treasury", "TIP": "Treasury", "LQD": "credit", "HYG": "credit",
          "UUP": "FX", "FXE": "FX"}
ONDEMAND = "on demand"          # a fund outside the cache whose IBKR name says bond, Treasury, credit or currency
COMMODITY = ("GLD", "SLV", "USO", "UNG", "DBC", "BITO", "IBIT", "ETHA")     # reported only, not a gate
# IBKR's long names are cut to 28 characters and abbreviated ("ISHR IBX USD HIYLD CB ETF-UI", "ISHARES 20+ YEAR TREASURY BD")
_BOND = (r"BONDS?|BND|BD|BO|TREASURY|TREASURIES|TREAS|TRSY|T-BILLS?|T-B|CREDIT|CORPORAT\w*|CB|MUNI\w*|HIGH YIELD|HIYLD|"
         r"INVGD|AGGREGATE|TIPS|MBS|FIXED INCOME|FLOATING RATE|SENIOR LOAN|DURATION|MATURIT\w*|MONTH|ULTRA[- ]SHORT")
_CURRENCY = r"CURRENCY\w*|DOLLAR"
BOND = re.compile(rf"\b({_BOND})\b", re.I)                      # bond and credit terms
CURRENCY = re.compile(rf"\b({_CURRENCY})\b", re.I)              # currency terms
BOND_FX = re.compile(rf"\b({_BOND}|{_CURRENCY})\b", re.I)       # either
HEDGED = re.compile(r"\b(HEDGED|HDG)\b", re.I)                  # "currency hedged": a hedge on another fund, not a currency fund
LEVERAGED = re.compile(r"\b(ULTRA(?![ -]SHORT)\w*|[1-4](\.\d+)?X|BULL?|BEAR?|INVERSE|LEVERAGED|VIX|VOLATILITY|"
                       r"DIREXION DAILY|DRX DLY|PROSHARES SHORT)\b", re.I)    # forecast's leveraged/inverse flag
TARGETS = ("range1", "vol20")
H = {"range1": 1, "vol20": 20}
FIT_FROM, TEST_FROM, RETRAIN, MIN_TRAIN = "20250102", "20250303", 21, 10
COVER, FUND_FLOOR, FLOOR = 0.70, 0.60, 1e-4
OPTIONS = ("current", "A", "B")
WITHHELD = "our models ran too high on these and no recalibration passed"
CLAUSE = {"A": "{t} from its own last 20 sessions, recalibrated", "B": "{t} rescaled for these funds",
          "withheld": "{t} withheld: " + WITHHELD}
# why apply() could not serve a column (amendment of 25 Oct): the clause becomes "<t> withheld: <why>"
NO_OWN, NO_FIT, NO_BASE = "no 20-session history", "no recalibration fitted yet", "no model forecast to rescale"
# the pooled models' fallback part of a note: "no implied vol or intraday data: previous model for absr1, vol20" (simulator
# and live, vol2.FALLBACK_NOTE); caches written before 25 Oct say "no implied vol: HAR fallback for absr1, vol20"
FALLBACK = re.compile(r"(?P<why>.+): (?P<what>previous model|HAR fallback) for (?P<cols>\w+(?:, \w+)*)")


def group(sym, long_name=None, stock_type=None):
    """The fund's group: a listed fund by symbol; ONDEMAND for an ETF or ETN outside the list whose IBKR long name has a
    bond or credit term, or a currency term without HEDGED/HDG, and no leveraged or inverse term; None for everything
    else (stocks and other funds keep the pooled models). Leveraged and inverse funds (TMF, TBT) keep forecast's
    leveraged flag (none was tested); a currency-hedged stock fund is not a currency fund."""
    if sym in GROUPS:
        return GROUPS[sym]
    name = long_name or ""
    if (stock_type or "").upper() not in ("ETF", "ETN") or LEVERAGED.search(name):
        return None
    if BOND.search(name) or (CURRENCY.search(name) and not HEDGED.search(name)):
        return ONDEMAND
    return None


def own_base(h, l, c):
    """The fund's own last 20 sessions through the last element of daily h, l, c: range1 = mean (H - L) / C with at least
    10 valid (risk.build's naive range1), vol20 = log annualized std of the 20 daily log returns, ddof 1 (ml.py's lvol20).
    NaN with fewer than 20 sessions."""
    h, l, c = (np.asarray(x, float)[-21:] for x in (h, l, c))
    with np.errstate(all="ignore"):
        rg = ((h - l) / c)[-20:]
        lr = np.diff(np.log(c))[-20:]
    rg, lr = rg[np.isfinite(rg)], lr[np.isfinite(lr)]
    vol = float(np.std(lr, ddof=1)) * math.sqrt(252) if len(lr) >= 2 else 0.0
    short = len(c) < 20
    return {"range1": float(rg.mean()) if len(rg) >= 10 and not short else math.nan,
            "vol20": math.log(vol) if vol > 0 and not short else math.nan}


def fit(y, b, tgt):
    """One pooled fit on training cells. range1: k = sum(y) / sum(b) (mean-unbiased) and the 10/90% quantiles of y / b,
    on b > 0; vol20 (logs): k = mean(y - b) and the quantiles of y - b."""
    with np.errstate(all="ignore"):
        ok = np.isfinite(y) & np.isfinite(b) & ((b > 0) if tgt == "range1" else True)
    y, b = y[ok], b[ok]
    r = y / b if tgt == "range1" else y - b
    k = y.sum() / b.sum() if tgt == "range1" else r.mean()
    q10, q90 = np.quantile(r, (0.1, 0.9))
    return {"k": float(k), "q10": float(q10), "q90": float(q90), "cells": int(ok.sum())}


def serve(b, f, tgt):
    """(point, lo, hi) on the model's scale (range1 a fraction, vol20 a log) from a base and a fit."""
    b = np.asarray(b, float)
    if tgt == "range1":
        return b * f["k"], b * f["q10"], b * f["q90"]
    return b + f["k"], b + f["q10"], b + f["q90"]


def walk(y, b, tgt, dates):
    """Walk-forward as vol2's protocol: refit every RETRAIN sessions from FIT_FROM on an expanding window of the cells
    whose target was realized before the refit (t + h <= T - 1), once MIN_TRAIN sessions have one. Returns the served
    (point, lo, hi) arrays and the fits, each with the first and last session it serves."""
    h, nd = H[tgt], len(dates)
    P, L, U = (np.full(b.shape, np.nan) for _ in range(3))
    fits = []
    for T in range(dates.index(FIT_FROM), nd, RETRAIN):
        yy, bb = y[:, :T - h], b[:, :T - h]
        with np.errstate(all="ignore"):
            ok = np.isfinite(yy) & np.isfinite(bb) & ((bb > 0) if tgt == "range1" else True)
        if ok.any(0).sum() < MIN_TRAIN:
            continue
        f = fit(yy, bb, tgt)
        te = slice(T, min(T + RETRAIN, nd))
        P[:, te], L[:, te], U[:, te] = serve(b[:, te], f, tgt)
        fits.append({"from": dates[T], "to": dates[te.stop - 1], **f})
    return (P, L, U), fits


# ---------------------------------------------------------------- scoring
def score(p, lo, hi, y, m, tgt):
    """Coverage and tails of the band, realized / forecast (vol20 in vol units), R2 (range1 level and log, with
    forecasts and outcomes floored at FLOOR before the log; vol20 is already a log)."""
    if not m.any():
        return {"n": 0}
    Y, P, L, U = y[m], p[m], lo[m], hi[m]
    r2 = lambda a, f: round(float(1 - np.mean((a - f) ** 2) / np.var(a)), 4)
    lg = (lambda x: np.log(np.maximum(x, FLOOR))) if tgt == "range1" else (lambda x: x)
    out = {"n": int(m.sum()), "coverage": round(float(((Y >= L) & (Y <= U)).mean()), 4),
           "below": round(float((Y < L).mean()), 4), "above": round(float((Y > U).mean()), 4),
           "realized_over_forecast": round(float(Y.mean() / P.mean() if tgt == "range1" else
                                                 np.exp(Y).mean() / np.exp(P).mean()), 3), "r2_log": r2(lg(Y), lg(P))}
    if tgt == "range1":
        out["r2_level"] = r2(Y, P)
    return out


def half_of(x):
    return x[:4] + ("H1" if x[4:6] <= "06" else "H2")


def validate():
    """The pre-registered test: options current, A and B for range1 and vol20 on the 8 funds, scored by group, fund and
    half-year on identical cells; the per-group decision; the guards on everything else; the fits served live and in
    the simulator. Writes bondfx.json and bondfx_oos.npz."""
    sys.path.insert(0, str(ROOT))
    import trader
    from ml_research import risk
    z = np.load(OUT / "vol2_oos.npz", allow_pickle=True)
    syms, dates = [str(s) for s in z["symbols"]], [str(x) for x in z["dates"]]
    d = trader.Data(last=dates[-1])
    assert d.sym == syms and d.dates == dates
    F, S, Y, N, V, _ = risk.build(d)
    funds, nd = list(GROUPS), len(dates)
    J = [d.k[s] for s in funds]
    own = [[own_base(d.h[j, :t + 1], d.l[j, :t + 1], d.c[j, :t + 1]) for t in range(nd)] for j in J]
    base = {tgt: np.array([[o[tgt] for o in row] for row in own]) for tgt in TARGETS}
    # the base is exactly the naive series the research used (risk.build, ml.py)
    assert np.allclose(base["range1"], N["range1"][J], equal_nan=True)
    assert np.allclose(base["vol20"], np.where(np.isfinite(V["lvol20"][J]), V["lvol20"][J], np.nan), equal_nan=True)
    t0 = dates.index(TEST_FROM)
    res = {"prereg": "runs/ml/research/bondfx_prereg.md", "as_of": dates[-1], "fit_from": FIT_FROM, "test_from": TEST_FROM,
           "retrain_every": RETRAIN, "min_train_sessions": MIN_TRAIN, "funds": GROUPS, "scores": {}, "by_fund": {},
           "by_half": {}, "decision": {}, "fits": {}, "live": {}, "guards": {}, "diagnostics": {}}
    npz, served = {}, {}
    groups = {g: [i for i, s in enumerate(funds) if GROUPS[s] == g] for g in ("Treasury", "credit", "FX")}
    groups["pooled"] = list(range(len(funds)))
    for tgt in TARGETS:
        h, y = H[tgt], Y[tgt][J]
        cur = tuple(z[k][J] for k in (tgt, f"{tgt}_lo", f"{tgt}_hi"))
        a, fa = walk(y, base[tgt], tgt, dates)
        b, fb = walk(y, cur[0], tgt, dates)
        O = {"current": cur, "A": a, "B": b}
        m = np.zeros(y.shape, bool)
        m[:, t0:nd - h] = True
        m &= np.isfinite(y)
        for p, lo, hi in O.values():
            m &= np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi)
        span = np.flatnonzero(m.any(0))         # the identical cells' span (vol20: the current band starts 29 Apr 2025)
        res["scores"][tgt] = {"window": f"{dates[span[0]]}..{dates[span[-1]]}", "cells": int(m.sum())}
        for g, rows in groups.items():
            mg = np.zeros_like(m)
            mg[rows] = m[rows]
            res["scores"][tgt][g] = {o: score(*O[o], y, mg, tgt) for o in OPTIONS}
        res["by_fund"][tgt] = {}
        for i, s in enumerate(funds):
            mi = np.zeros_like(m)
            mi[i] = m[i]
            res["by_fund"][tgt][s] = {o: score(*O[o], y, mi, tgt) for o in OPTIONS}
        res["by_half"][tgt] = {}
        for hk in sorted({half_of(dates[t]) for t in range(t0, nd - h)}):
            mh = m & np.array([half_of(x) == hk for x in dates])[None, :]
            res["by_half"][tgt][hk] = {g: {o: score(*O[o], y, mh & np.isin(np.arange(len(funds)), rows)[:, None], tgt)
                                           .get("coverage") for o in OPTIONS} for g, rows in groups.items()}
        # the pre-registered rule, per group (pooled = names scored on demand)
        for g, rows in groups.items():
            sc = res["scores"][tgt][g]
            ok = [o for o in OPTIONS if sc[o]["coverage"] >= COVER
                  and min(res["by_fund"][tgt][funds[i]][o]["coverage"] for i in rows) >= FUND_FLOOR]
            pick = max(ok, key=lambda o: (sc[o]["r2_log"], o == "current")) if ok else "withheld"
            res["decision"].setdefault(ONDEMAND if g == "pooled" else g, {})[tgt] = pick
            res["scores"][tgt][g]["passing"] = ok
        res["fits"][tgt] = {"A": fa, "B": fb}
        res["live"][tgt] = {"A": fit(y, base[tgt], tgt), "B": fit(y, cur[0], tgt)}   # every realized cell, as live_q
        for o, (p, lo, hi) in O.items():
            npz[f"{tgt}_{o}"], npz[f"{tgt}_{o}_lo"], npz[f"{tgt}_{o}_hi"] = p, lo, hi
        served[tgt] = O
        print(tgt, json.dumps({g: {o: res["scores"][tgt][g][o].get("coverage") for o in OPTIONS}
                               for g in groups}), flush=True)
    # guards: everything else (S&P stocks, equity and sector ETFs) and all-names R2, current vs new
    etf = trader.etfs()
    eq = [s for s in syms if s in etf and s not in GROUPS and s not in COMMODITY]
    sp = [s for s in syms if s not in etf]
    for tgt in TARGETS:
        h, y = H[tgt], Y[tgt]
        p, lo, hi = (z[k].copy() for k in (tgt, f"{tgt}_lo", f"{tgt}_hi"))
        new = [a.copy() for a in (p, lo, hi)]
        for i, s in enumerate(funds):
            for a, src in zip(new, served[tgt].get(res["decision"][GROUPS[s]][tgt], (None,) * 3)):
                a[d.k[s]] = np.nan if src is None else src[i]          # withheld: no forecast
        win = np.zeros(y.shape, bool)
        win[:, t0:nd - h] = True
        win &= np.isfinite(y)
        g = {}
        for name, ss in (("sp_stocks", sp), ("equity_sector_etfs", eq), ("commodity_crypto_etfs", list(COMMODITY))):
            mk = win & np.isin(np.array(syms), ss)[:, None] & np.isfinite(p) & np.isfinite(lo)
            g[name] = {"current": score(p, lo, hi, y, mk, tgt), "new": score(*new, y, mk & np.isfinite(new[1]), tgt)}
        mall = win & np.isfinite(p) & np.isfinite(new[0])
        g["all_names"] = {"current": score(p, lo, hi, y, mall, tgt), "new": score(*new, y, mall, tgt)}
        res["guards"][tgt] = g
    cov = lambda grp, who: res["guards"]["range1"][grp][who]["coverage"]
    r2 = lambda tgt, k, who: res["guards"][tgt]["all_names"][who][k]
    res["guards"]["pass"] = {
        "sp_stocks_78_82": bool(0.78 <= cov("sp_stocks", "new") <= 0.82),
        "sp_stocks_not_lower_1pt": bool(cov("sp_stocks", "new") >= cov("sp_stocks", "current") - 0.01),
        "equity_etfs_75": bool(cov("equity_sector_etfs", "new") >= 0.75),
        "equity_etfs_not_lower_1pt": bool(cov("equity_sector_etfs", "new") >= cov("equity_sector_etfs", "current") - 0.01),
        "r2_range1_level": bool(r2("range1", "r2_level", "new") >= r2("range1", "r2_level", "current") - 0.01),
        "r2_vol20_log": bool(r2("vol20", "r2_log", "new") >= r2("vol20", "r2_log", "current") - 0.01),
        "r2_spread": "unchanged by construction (0.970, cost.json)"}
    # diagnostics, not gates: absr1 on the funds (unchanged by this work)
    ab, ya = (z["absr1"][J], z["absr1_lo"][J], z["absr1_hi"][J]), Y["absr1"][J]
    ma = np.zeros(ya.shape, bool)
    ma[:, t0:nd - 1] = True
    ma &= np.isfinite(ab[1]) & np.isfinite(ya)
    res["diagnostics"]["absr1_current"] = {g: score(*ab, ya, ma & np.isin(np.arange(len(funds)), rows)[:, None], "range1")
                                           for g, rows in groups.items()}
    # and the shipped vol20 from the first session on which A and B both have a band (6 Mar 2025), which adds the April
    # 2025 shock: the current model's vol20 band starts on 29 Apr 2025, so there is no baseline before then
    yv, Ov, cur = Y["vol20"][J], served["vol20"], served["vol20"]["current"]
    mv = np.zeros(yv.shape, bool)
    mv[:, t0:nd - H["vol20"]] = True
    mv &= np.isfinite(yv)
    for o in ("A", "B"):
        mv &= np.isfinite(Ov[o][0]) & np.isfinite(Ov[o][1]) & np.isfinite(Ov[o][2])
    span = np.flatnonzero(mv.any(0))
    early = mv & ~np.isfinite(cur[1])
    espan = np.flatnonzero(early.any(0))
    dv = {"window": f"{dates[span[0]]}..{dates[span[-1]]}", "cells": int(mv.sum()),
          "before_current_band": f"{dates[espan[0]]}..{dates[espan[-1]]}" if len(espan) else None,
          "groups": {}, "groups_before_current_band": {}, "by_fund": {}}
    for g in ("Treasury", "credit", "FX"):
        o = res["decision"][g]["vol20"]
        if o not in Ov:
            continue
        sel = np.isin(np.arange(len(funds)), groups[g])[:, None]
        dv["groups"][g] = {"option": o, **score(*Ov[o], yv, mv & sel, "vol20")}
        dv["groups_before_current_band"][g] = {"option": o, **score(*Ov[o], yv, early & sel, "vol20")}
        for i in groups[g]:
            dv["by_fund"][funds[i]] = {"option": o, **score(*Ov[o], yv, mv & (np.arange(len(funds)) == i)[:, None],
                                                              "vol20")}
    res["diagnostics"]["vol20_shipped_from_6mar"] = dv
    if not all(v is True or isinstance(v, str) for v in res["guards"]["pass"].values()):
        res["decision"] = {g: {t: "current" for t in TARGETS} for g in res["decision"]}
        res["note"] = "a guard failed: nothing ships"
    np.savez_compressed(OUT / "bondfx_oos.npz", symbols=np.array(funds), dates=np.array(dates), **npz)
    (OUT / "bondfx.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- serving (forecast.py, live and simulator)
_INFO = {}


def info():
    if not _INFO:
        _INFO.update(json.loads((OUT / "bondfx.json").read_text(encoding="utf-8")))
    return _INFO


def fit_at(tgt, opt, as_of=None):
    """The fit served at the close of `as_of`: the live fit when None; else the walk-forward fit whose block holds it
    (the last block's after its end, so strictly earlier data), None before the first."""
    if as_of is None:
        return info()["live"][tgt][opt]
    got = None
    for f in info()["fits"][tgt][opt]:
        if f["from"] <= as_of:
            got = f
    return got


def clause(grp, why=None):
    """The row note's clause for a fund of group `grp` (the same text live and in the simulator); `why` maps a column
    apply() set to None to the reason, which replaces that column's clause."""
    dec, why = info()["decision"][grp], why or {}
    parts = [f"{t} withheld: {why[t]}" if t in why else CLAUSE[dec[t]].format(t=t) for t in TARGETS if dec[t] != "current"]
    s = "bond, credit or currency fund" + (": " + "; ".join(parts) if parts else "")
    return s + (" (validated on 8 listed funds, not on this one)" if grp == ONDEMAND else "")


def drop_fallback(note, cols):
    """`note` with `cols` taken out of the pooled models' fallback part (FALLBACK); a part left with no column goes."""
    out = []
    for part in (note or "").split("; "):
        m = FALLBACK.fullmatch(part)
        if m:
            left = [c for c in m["cols"].split(", ") if c not in cols]
            part = f"{m['why']}: {m['what']} for {', '.join(left)}" if left else ""
        if part:
            out.append(part)
    return "; ".join(out)


def apply(row, grp, own, as_of=None):
    """The bond/FX treatment of one forecast row, live and in the simulator. Per column, the option the validation
    shipped for the group: A serves from `own` (own_base() of the fund), B from the row's current point, with the live
    fit or (simulator) the walk-forward fit as of the close `as_of`; a withheld column becomes None; "current" leaves it.
    A column that cannot be served (no 20-session history, no walk-forward fit yet, nothing to rescale) also becomes None,
    and the clause says why. Columns served from the fund's own history or set to None leave the note's fallback part
    (they no longer come from the pooled models; B keeps it, since it rescales that model's value). Only keys the row
    already has are written. Returns the note clause."""
    dec = info()["decision"][grp]
    why, gone = {}, []
    for tgt in TARGETS:
        opt = dec[tgt]
        if opt == "current":
            continue
        vals = (None, None, None)
        f = fit_at(tgt, opt, as_of) if opt != "withheld" else None
        if opt == "withheld":
            why[tgt] = WITHHELD
        elif f is None:
            why[tgt] = NO_FIT
        else:
            if opt == "A":
                b, miss = (own or {}).get(tgt), NO_OWN
            else:
                x = row.get(f"{tgt}_pct")
                b, miss = None if not x or x <= 0 else (x / 100 if tgt == "range1" else math.log(x / 100)), NO_BASE
            if b is not None and np.isfinite(b):
                p = serve(b, f, tgt)
                vals = tuple(round(float(v) * 100, 2) for v in p) if tgt == "range1" else \
                    tuple(round(math.exp(float(v)) * 100, 1) for v in p)
            else:
                why[tgt] = miss
        if opt == "A" or tgt in why:
            gone.append(tgt)
        keys = (f"{tgt}_pct", f"{tgt}_lo", f"{tgt}_hi")
        row.update({k: v for k, v in zip(keys, vals) if k in row})
    if gone and row.get("note"):
        row["note"] = drop_fallback(row["note"], gone)
    return clause(grp, why)


if __name__ == "__main__":
    validate()
