"""ML3's fit and validation, as pre-registered in runs/ml/research/ml3.md (files on disk only, no gateway).

    .venv\\Scripts\\python.exe ml3_validate.py

gap1: quantile regressions fit on nights whose next open is on or before 2024-12-31, event multipliers fit on that
window's event nights, tested on next opens 2025-01-02 .. 2026-11-06 against the naive 60-night median. size_scale_now:
the fixed rule on SPY's 30-minute bars (from 17 Oct 2025), against the morning size_scale. Writes
runs/ml/research/ml3_model.json (the shipped gap1 parameters) and ml3_validation.json.
"""

import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import forecast3 as f3      # noqa: E402
import trader               # noqa: E402
from ml_research import events, regime    # noqa: E402

RES = ROOT / "runs" / "ml" / "research"
FIT_END, TEST_FROM, TEST_END = "20241231", "20250102", "20261106"
AV_FROM = "20261019"         # next opens from here: earnings from the Alpha Vantage snapshots (earnings_dates.json ends 16 Oct)
NBOOT, SEED = 2000, 0
CHIPS = ["SMH", "AMD", "MU", "LRCX", "NVDA", "QCOM", "AVGO", "INTC", "SPY", "QQQ"]


def iso(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


# ---------------------------------------------------------------- event flags (names x nights; night t = close t to open t+1)
def flags(d):
    n, dates = len(d.dates), d.dates
    ce = np.array([events.close_epoch(x) for x in dates], np.int64)
    oe = np.array([events.et_epoch(x, 9, 30) for x in dates], np.int64)
    de = np.array([events.et_epoch(x, 8, 15) for x in dates], np.int64)      # compute()'s time on day t
    ed = json.loads((ROOT / "data" / "edgar" / "earnings_dates.json").read_text())["symbols"]
    earn, e1 = np.zeros((len(d.sym), n), bool), np.zeros((len(d.sym), n), bool)

    def night(e):
        t = int(np.searchsorted(ce, e, "left")) - 1                          # the last close before e
        return t if 0 <= t < n - 1 and e <= oe[t + 1] else None
    for s, recs in ed.items():
        j = d.k.get(s)
        if j is None:
            continue
        T = np.array(sorted(events._epoch_utc(r["accepted_utc"]) for r in recs), np.int64)
        for e in T:
            t = night(int(e))
            if t is not None and dates[t + 1] < AV_FROM:
                earn[j, t] = True
        for i in range(len(T)):                                              # E1 from the reports known at each night
            p = events.next_pred(T[:i + 1], "E1")
            t = night(int(p))
            if t is not None and T[i] < de[t] and (i + 1 == len(T) or T[i + 1] >= de[t]):
                e1[j, t] = True
    snaps = sorted((ROOT / "data" / "av" / "calendar").glob("*.csv"))
    cal = {}
    for p in snaps:
        with open(p, newline="", encoding="utf-8") as f:
            cal[p.stem] = [(r["symbol"].replace("-", " ").replace(".", " "), r["reportDate"], r.get("timeOfTheDay", ""))
                           for r in csv.DictReader(f)]
    for t in range(n - 1):
        if dates[t + 1] < AV_FROM:
            continue
        ok = [x for x in cal if x <= dates[t]]
        snap = cal[ok[-1] if ok else min(cal)]
        for s, day, when in snap:
            j = d.k.get(s)
            if j is not None and ((day == iso(dates[t]) and when == "post-market") or (day == iso(dates[t + 1]) and when == "pre-market")):
                earn[j, t] = True
    rows = json.loads(f3.MACRO.read_text(encoding="utf-8"))["events"]
    macro = np.array([bool(f3.macro_before_open(iso(dates[t + 1]), rows)) if t < n - 1 else False for t in range(n)])
    return earn, e1, macro


# ---------------------------------------------------------------- panel
def panel(d):
    A = f3.gaps(d.o, d.c)
    l60, l250 = f3.name_feats(A)
    stocks = np.array([s not in d.etf for s in d.sym])
    lmkt5, lv5 = f3.market_feats(A, stocks, d.c[d.k["SPY"]])
    n = len(d.dates)
    with np.errstate(all="ignore"):
        y = np.full(d.c.shape, np.nan)
        y[:, :-1] = np.abs(d.o[:, 1:] / d.c[:, :-1] - 1) * 100            # night t's |gap|, %
    lag = lambda x: np.concatenate([np.full(x.shape[:-1] + (1,), np.nan), x[..., :-1]], axis=-1)
    longn = np.array([(date.fromisoformat(iso(d.dates[t + 1])) - date.fromisoformat(iso(d.dates[t]))).days > 1
                      if t < n - 1 else False for t in range(n)], float)
    L60, L250, MK, V5 = lag(l60), lag(l250), lag(lmkt5), lag(lv5)           # night t uses bars through close t-1
    bc = lambda r: np.broadcast_to(r, d.c.shape)
    X = np.stack([L60, L250, bc(MK), bc(V5), bc(longn)], axis=-1)            # names x nights x FEATS
    Xc = np.stack([L60, L250, bc(MK), bc(lv5), bc(longn)], axis=-1)          # secondary: SPY's vol through close t
    return y, X, Xc, stocks


def pinball(y, q):
    return sum(np.maximum(t * (y - q[:, i]), (t - 1) * (y - q[:, i])) for i, t in enumerate(f3.TAUS))


def cover(y, q):
    return (y >= q[:, 0]) & (y <= q[:, 2])


def boot_diff(loss_a, loss_b, night, seed=SEED):
    """Mean of a - b over name-nights and its 90% interval from a bootstrap over nights."""
    u, inv = np.unique(night, return_inverse=True)
    s = np.bincount(inv, loss_a - loss_b)
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(u), len(u))
        draws.append(s[i].sum() / c[i].sum())
    return float(s.sum() / c.sum()), [float(x) for x in np.percentile(draws, [5, 95])]


def summary(y, q, night, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    if not m.any():
        return {"n": 0}
    cv = cover(y[m], q[m])
    return {"n": int(m.sum()), "coverage": round(float(cv.mean()), 4), "above": round(float((y[m] > q[m, 2]).mean()), 4),
            "below_lo": round(float((y[m] < q[m, 0]).mean()), 4), "below_median": round(float((y[m] < q[m, 1]).mean()), 4),
            "median_ratio": round(float(np.median(y[m] / q[m, 1])), 3), "pinball": round(float(pinball(y[m], q[m]).mean()), 5)}


# ---------------------------------------------------------------- gap1
def gap1(d):
    y, X, Xc, stocks = panel(d)
    earn, e1, macro = flags(d)
    n = len(d.dates)
    nxt = np.array([d.dates[t + 1] if t < n - 1 else "99999999" for t in range(n)])
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)
    J, T = np.nonzero(ok)
    yy, XX, XXc = y[J, T], X[J, T], Xc[J, T]
    ev_e, ev_e1, ev_m = earn[J, T], e1[J, T], macro[T]
    fit, test = nxt[T] <= FIT_END, (nxt[T] >= TEST_FROM) & (nxt[T] <= TEST_END)
    ly = np.log(np.maximum(yy, f3.FLOOR))
    base_rows = fit & ~ev_e & ~ev_m
    M = {"version": f3.VERSION, "features": f3.FEATS, "taus": list(f3.TAUS), "floor_pct": f3.FLOOR,
         "fit": f"nights with the next open {iso(d.dates[int(T[fit].min()) + 1])} .. {iso(FIT_END)}, "
                f"{int(base_rows.sum())} name-nights without an event", "base": {}}
    for t in f3.TAUS:
        m = events.qr_fit(XX[base_rows], ly[base_rows], t)
        M["base"][str(t)] = {"mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "b": m["b"].tolist()}
    qb = f3.quantiles(M, XX)
    qbc = f3.quantiles(M, XXc)
    M["mult"], M["mult_n"] = {}, {}
    for name, rows in (("earn", fit & ev_e & ~ev_m), ("earn_e1", fit & ev_e1 & ~ev_m), ("macro", fit & ev_m & ~ev_e)):
        r = yy[rows][:, None] / qb[rows]
        M["mult"][name] = [round(float(np.quantile(r[:, i], t)), 4) for i, t in enumerate(f3.TAUS)]
        M["mult_n"][name] = int(rows.sum())
    M["ship"] = {"earn": True, "earn_e1": False, "macro": False}         # decided on the test nights below
    lr = np.log(yy[fit] / np.exp(XX[fit, 0]))                            # naive: m60, band from the fit window's ratios
    M["naive"] = [float(np.exp(np.quantile(lr, 0.1))), 1.0, float(np.exp(np.quantile(lr, 0.9)))]
    qn = np.exp(XX[:, 0])[:, None] * np.array(M["naive"])

    def ml3(ship_e1, ship_m, Xq=None):
        q = qb.copy() if Xq is None else Xq.copy()
        mul = np.ones_like(q)
        mul[ev_e] *= np.array(M["mult"]["earn"])
        if ship_m:
            mul[ev_m] *= np.array(M["mult"]["macro"])
        return np.sort(q * mul, axis=1)

    te = test
    night = T
    out = {"model": M}
    # event nights: with and without each multiplier (test window)
    ev = {}
    q_no = qb
    for name, mask, mult in (("earn", te & ev_e, "earn"), ("earn_e1", te & ev_e1, "earn_e1"), ("macro", te & ev_m, "macro")):
        qw = np.sort(qb * np.array(M["mult"][mult]), axis=1)
        a, b = pinball(yy[mask], qw[mask]), pinball(yy[mask], q_no[mask])
        ev[name] = {"with": summary(yy, qw, night, mask), "without": summary(yy, q_no, night, mask),
                    "pinball_diff_with_minus_without": boot_diff(a, b, night[mask]), "multiplier": M["mult"][mult],
                    "fit_n": M["mult_n"][mult]}
    # E1 nights split by whether the release really came that night (realized schedule, through 16 Oct 2026)
    for lab, mask in (("e1_right", te & ev_e1 & ev_e), ("e1_wrong", te & ev_e1 & ~ev_e)):
        qw = np.sort(qb * np.array(M["mult"]["earn_e1"]), axis=1)
        ev[lab] = {"with": summary(yy, qw, night, mask), "without": summary(yy, q_no, night, mask)}
    ev["earn_missed_by_e1"] = {"n": int((te & ev_e & ~ev_e1).sum())}
    out["events"] = ev
    M["ship"]["earn_e1"] = ev["earn_e1"]["pinball_diff_with_minus_without"][0] < 0
    M["ship"]["macro"] = ev["macro"]["pinball_diff_with_minus_without"][0] < 0
    q3 = ml3(False, M["ship"]["macro"])
    q3c = ml3(False, M["ship"]["macro"], qbc)
    # pooled, by group and half-year
    st = stocks[J]
    half = np.array([x[:4] + ("H1" if x[4:6] <= "06" else "H2") for x in nxt[T]])
    res = {"ml3": summary(yy, q3, night, te), "naive": summary(yy, qn, night, te),
           "ml3_no_events": summary(yy, qb, night, te), "ml3_refreshed_at_close": summary(yy, q3c, night, te)}
    res["pinball_diff_ml3_minus_naive"] = boot_diff(pinball(yy[te], q3[te]), pinball(yy[te], qn[te]), night[te])
    res["pinball_diff_refreshed_minus_morning"] = boot_diff(pinball(yy[te], q3c[te]), pinball(yy[te], q3[te]), night[te])
    res["stocks"] = {"ml3": summary(yy, q3, night, te & st), "naive": summary(yy, qn, night, te & st)}
    res["etfs"] = {"ml3": summary(yy, q3, night, te & ~st), "naive": summary(yy, qn, night, te & ~st)}
    res["by_half"] = {h: {"ml3": summary(yy, q3, night, te & (half == h)), "naive": summary(yy, qn, night, te & (half == h))}
                      for h in sorted(set(half[te]))}
    # per-night coverage distribution
    def per_night(q):
        u = np.unique(night[te])
        cv = cover(yy, q)
        c = np.array([cv[te & (night == t)].mean() for t in u])
        return {"nights": int(len(u)), "p10_25_50_75_90": [round(float(x), 3) for x in np.percentile(c, [10, 25, 50, 75, 90])],
                "share_under_60": round(float((c < 0.6).mean()), 3), "share_under_50": round(float((c < 0.5).mean()), 3),
                "worst": sorted(((round(float(x), 3), iso(d.dates[int(t) + 1])) for x, t in zip(c, u)))[:8]}
    res["per_night"] = {"ml3": per_night(q3), "naive": per_night(qn)}
    # median calibration by decile of the forecast median
    qq = q3[te, 1]
    edges = np.quantile(qq, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, qq, "right") - 1, 0, 9)
    res["median_by_decile"] = [{"decile": k + 1, "forecast_median": round(float(np.median(qq[dec == k])), 3),
                                "realized_median": round(float(np.median(yy[te][dec == k])), 3),
                                "below_median": round(float((yy[te][dec == k] < qq[dec == k]).mean()), 3)} for k in range(10)]
    # 5 and 6 Nov 2026 (the gaps at those opens)
    days = {}
    for day in ("20261105", "20261106"):
        m = nxt[T] == day
        rows = {}
        for s in CHIPS:
            j = d.k.get(s)
            i = np.nonzero(m & (J == j))[0]
            if len(i):
                i = i[0]
                rows[s] = {"gap_pct": round(float(yy[i]), 2), "signed_pct": round(float((d.o[j, T[i] + 1] / d.c[j, T[i]] - 1) * 100), 2),
                           "ml3": [round(float(x), 2) for x in q3[i]], "naive": [round(float(x), 2) for x in qn[i]],
                           "ml3_refreshed_at_close": [round(float(x), 2) for x in q3c[i]], "earn": bool(ev_e[i])}
        days[iso(day)] = {"ml3": summary(yy, q3, night, m), "naive": summary(yy, qn, night, m),
                          "ml3_refreshed_at_close": summary(yy, q3c, night, m), "macro_night": bool(ev_m[m][0]) if m.any() else None,
                          "names": rows}
    res["days"] = days
    out["gap1"] = res
    c = res["ml3"]
    out["pass"] = {"coverage_77_83": 0.77 <= c["coverage"] <= 0.83,
                   "pinball_below_naive": res["pinball_diff_ml3_minus_naive"][1][1] < 0,
                   "earn_coverage": ev["earn"]["with"]["coverage"] >= 0.70 and ev["earn"]["with"]["coverage"] > ev["earn"]["without"]["coverage"],
                   "median_45_55": 0.45 <= c["below_median"] <= 0.55}
    out["pass"]["gap1"] = all(out["pass"].values())
    M["ship"]["refresh_in_session"] = res["pinball_diff_refreshed_minus_morning"][0] < 0
    return out, M


# ---------------------------------------------------------------- size_scale_now
def rho(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def intraday(d):
    q1 = json.loads((RES / "regime.json").read_text(encoding="utf-8"))["intervals"]["vol20"]["live_q"][1]
    j = d.k["SPY"]
    gap = {d.dates[i]: math.log(d.o[j, i] / d.c[j, i - 1]) for i in range(1, len(d.dates))}
    nextgap = {d.dates[i]: abs(d.o[j, i + 1] / d.c[j, i] - 1) for i in range(len(d.dates) - 1)}
    sigma = f3.morning_sigma(regime.load_iv(), d.dates, q1)
    days30 = f3.by_day(json.loads((f3.BARS30 / "SPY.json").read_text(encoding="utf-8"))["bars"])
    typ = 12.0                                    # ML2's typical_spy_vol_pct in every live file; ranks don't depend on it
    rows = []
    for day in sorted(days30):
        b = days30[day]
        if len(b) != 13 or day not in gap or not sigma.get(day) or day > TEST_END:
            continue
        G = f3.g_norm(days30, gap, sigma, day)
        if any(g is None for g in G):
            continue
        ss = typ / (sigma[day] * math.sqrt(252) * 100)
        for k in range(13):
            x = f3.so_far(b, k, gap[day])
            now, z = f3.scale_now(ss, sigma[day], x, G[k], k)
            P = b["09:30"]["open"] if k == 0 else b[f3.SLOTS[k - 1]]["close"]
            rest = [b[s] for s in f3.SLOTS[k:]]
            A = math.log(max([P] + [r["high"] for r in rest]) / min([P] + [r["low"] for r in rest]))
            rows.append((day, k, ss, now, z, A, nextgap.get(day, np.nan)))
    days = sorted({r[0] for r in rows})
    di = {x: i for i, x in enumerate(days)}
    R = np.array([(di[r[0]], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows])
    out = {"sessions": len(days), "first": iso(days[0]), "last": iso(days[-1]), "looks": {}}
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(days), len(days)) for _ in range(NBOOT)]
    per = {}
    for k in range(13):
        S = R[R[:, 1] == k]
        S = S[np.argsort(S[:, 0])]
        per[k] = S
        okb = np.isfinite(S[:, 6])
        out["looks"][f3.SLOTS[k]] = {
            "n": int(len(S)), "rest_rho_now": round(rho(-np.log(S[:, 3]), S[:, 5]), 3),
            "rest_rho_morning": round(rho(-np.log(S[:, 2]), S[:, 5]), 3),
            "gap_n": int(okb.sum()), "gap_rho_now": round(rho(-np.log(S[okb, 3]), S[okb, 6]), 3),
            "gap_rho_morning": round(rho(-np.log(S[okb, 2]), S[okb, 6]), 3)}

    def drho(k, col, idx):
        S = per[k][idx]
        ok = np.isfinite(S[:, col])
        return rho(-np.log(S[ok, 3]), S[ok, col]) - rho(-np.log(S[ok, 2]), S[ok, col])
    full = np.arange(len(days))
    mean_a = lambda idx: float(np.mean([drho(k, 5, idx) for k in range(1, 13)]))
    out["A_mean_drho_10_to_1530"] = [round(mean_a(full), 3), [round(float(x), 3) for x in np.percentile([mean_a(i) for i in boots], [5, 95])]]
    for k in range(13):
        out["looks"][f3.SLOTS[k]]["rest_drho_90"] = [round(float(x), 3) for x in np.percentile([drho(k, 5, i) for i in boots], [5, 95])]
        out["looks"][f3.SLOTS[k]]["gap_drho_90"] = [round(float(x), 3) for x in np.percentile([drho(k, 6, i) for i in boots], [5, 95])]
    b1530 = out["looks"]["15:30"]
    out["B_1530"] = [round(b1530["gap_rho_now"] - b1530["gap_rho_morning"], 3), b1530["gap_drho_90"]]
    # R2 of the log target on the log scale (one-variable OLS fit on the test rows: in-sample, for scale only)
    def r2(xc, col):
        v = []
        for k in range(1, 13):
            S = per[k]
            ok = np.isfinite(S[:, col]) & (S[:, col] > 0)
            yv, xv = np.log(S[ok, col]), -np.log(S[ok, xc])
            v.append(float(np.corrcoef(xv, yv)[0, 1] ** 2))
        return round(float(np.mean(v)), 3)
    out["r2_rest_mean_10_to_1530"] = {"now": r2(3, 5), "morning": r2(2, 5)}
    out["r2_gap_mean_10_to_1530"] = {"now": r2(3, 6), "morning": r2(2, 6)}
    nov5 = [r for r in rows if r[0] == "20261105"]
    out["nov5"] = [{"look": f3.SLOTS[r[1]], "size_scale": round(r[2], 2), "size_scale_now": round(r[3], 2), "z": round(r[4], 2),
                    "rest_of_day_range_pct": round(r[5] * 100, 2)} for r in nov5]
    out["nov5_next_gap_pct"] = round(nextgap.get("20261105", float("nan")) * 100, 2)
    a, b = out["A_mean_drho_10_to_1530"], out["B_1530"]
    out["pass"] = {"A": a[0] > 0 and a[1][0] > 0, "B": b[0] > 0 and b[1][0] > -0.05}
    out["pass"]["size_scale_now"] = out["pass"]["A"] and out["pass"]["B"]
    return out


def fit_all(XX, ly, yy, rows, ev_e, ev_e1, ev_m):
    """Base quantiles on `rows` without an event, then the three multipliers on its event nights (as gap1() does)."""
    M = {"version": f3.VERSION, "features": f3.FEATS, "taus": list(f3.TAUS), "floor_pct": f3.FLOOR, "base": {}}
    b = rows & ~ev_e & ~ev_m
    for t in f3.TAUS:
        m = events.qr_fit(XX[b], ly[b], t)
        M["base"][str(t)] = {"mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "b": m["b"].tolist()}
    qb = f3.quantiles(M, XX)
    M["mult"], M["mult_n"] = {}, {}
    for name, r in (("earn", rows & ev_e & ~ev_m), ("earn_e1", rows & ev_e1 & ~ev_m), ("macro", rows & ev_m & ~ev_e)):
        x = yy[r][:, None] / qb[r]
        M["mult"][name] = [round(float(np.quantile(x[:, i], t)), 4) for i, t in enumerate(f3.TAUS)]
        M["mult_n"][name] = int(r.sum())
    with np.errstate(divide="ignore"):
        lr = np.log(yy[rows] / np.exp(XX[rows, 0]))
    M["naive"] = [float(np.exp(np.quantile(lr, 0.1))), 1.0, float(np.exp(np.quantile(lr, 0.9)))]
    M["ship"] = {"earn": True, "earn_e1": True, "macro": True, "refresh_in_session": False}
    return M, qb


def amend1(d):
    """Amendment 1 (post hoc): the same model refit every 21 test nights on an expanding window of known nights."""
    y, X, _, stocks = panel(d)
    earn, e1, macro = flags(d)
    n = len(d.dates)
    nxt = np.array([d.dates[t + 1] if t < n - 1 else "99999999" for t in range(n)])
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=-1)
    J, T = np.nonzero(ok)
    yy, XX = y[J, T], X[J, T]
    ev_e, ev_e1, ev_m = earn[J, T], e1[J, T], macro[T]
    ly = np.log(np.maximum(yy, f3.FLOOR))
    test = (nxt[T] >= TEST_FROM) & (nxt[T] <= TEST_END)
    nights = sorted(set(T[test]))
    q3, qn = np.full((len(yy), 3), np.nan), np.full((len(yy), 3), np.nan)
    qb, me1, mm, mea = (np.full((len(yy), 3), np.nan) for _ in range(4))   # base, and each block's multipliers
    fits = []
    for b0 in range(0, len(nights), 21):
        t0 = nights[b0]
        block = test & (T >= t0) & (T <= nights[min(b0 + 20, len(nights) - 1)])
        known = (nxt[T] <= FIT_END) if b0 == 0 else (T <= t0 - 2)       # the night before last: its open precedes day t0
        M, _ = fit_all(XX, ly, yy, known, ev_e, ev_e1, ev_m)
        q = f3.quantiles(M, XX[block])
        mul = np.ones_like(q)
        mul[ev_e[block]] *= np.array(M["mult"]["earn"])
        mul[ev_m[block]] *= np.array(M["mult"]["macro"])
        q3[block] = np.sort(q * mul, axis=1)
        qb[block], me1[block], mm[block], mea[block] = q, M["mult"]["earn_e1"], M["mult"]["macro"], M["mult"]["earn"]
        qn[block] = np.exp(XX[block, 0])[:, None] * np.array(M["naive"])
        fits.append({"first_night_open": iso(nxt[t0]), "rows": int(known.sum()),
                     "b_median": [round(x, 4) for x in M["base"]["0.5"]["b"]], "mult": M["mult"]})
    M2, _ = fit_all(XX, ly, yy, np.isfinite(yy) & (nxt[T] <= TEST_END), ev_e, ev_e1, ev_m)
    M2["fit"] = f"Amendment 1: every night with the next open 2024-01-22 .. {iso(TEST_END)}"
    pre = json.loads((RES / "ml3_model.json").read_text(encoding="utf-8"))
    qn0 = np.exp(XX[:, 0])[:, None] * np.array(pre["naive"])
    night = T
    st = stocks[J]
    half = np.array([x[:4] + ("H1" if x[4:6] <= "06" else "H2") for x in nxt[T]])
    res = {"ml3_wf": summary(yy, q3, night, test), "naive_fixed": summary(yy, qn0, night, test),
           "naive_wf": summary(yy, qn, night, test)}
    res["pinball_diff_wf_minus_naive_fixed"] = boot_diff(pinball(yy[test], q3[test]), pinball(yy[test], qn0[test]), night[test])
    res["pinball_diff_wf_minus_naive_wf"] = boot_diff(pinball(yy[test], q3[test]), pinball(yy[test], qn[test]), night[test])
    res["stocks"], res["etfs"] = summary(yy, q3, night, test & st), summary(yy, q3, night, test & ~st)
    res["by_half"] = {h: {"ml3_wf": summary(yy, q3, night, test & (half == h)), "naive_wf": summary(yy, qn, night, test & (half == h))}
                      for h in sorted(set(half[test]))}
    u = np.unique(night[test])
    cv = cover(yy, q3)
    c = np.array([cv[test & (night == t)].mean() for t in u])
    res["per_night"] = {"p10_25_50_75_90": [round(float(x), 3) for x in np.percentile(c, [10, 25, 50, 75, 90])],
                        "share_under_60": round(float((c < 0.6).mean()), 3), "share_under_50": round(float((c < 0.5).mean()), 3),
                        "worst": sorted(((round(float(x), 3), iso(d.dates[int(t) + 1])) for x, t in zip(c, u)))[:8]}
    for name, mask in (("earn", ev_e), ("earn_e1", ev_e1), ("macro", ev_m)):
        res[f"{name}_nights"] = summary(yy, q3, night, test & mask)        # q3: the realized schedule's earn, and macro
    srt = lambda q: np.sort(q, axis=1)
    av = nxt[T] >= AV_FROM
    res["events"] = {                                                  # each multiplier alone against the base
        "earn_schedule": {"with": summary(yy, srt(qb * mea), night, test & ev_e & ~av), "without": summary(yy, qb, night, test & ev_e & ~av)},
        "earn_av_calendar": {"with": summary(yy, srt(qb * mea), night, test & ev_e & av), "without": summary(yy, qb, night, test & ev_e & av)},
        "earn_e1": {"with": summary(yy, srt(qb * me1), night, test & ev_e1), "without": summary(yy, qb, night, test & ev_e1)},
        "macro": {"with": summary(yy, srt(qb * mm), night, test & ev_m), "without": summary(yy, qb, night, test & ev_m)}}
    days = {}
    for day in ("20261105", "20261106"):
        m = nxt[T] == day
        rows = {}
        for s in CHIPS:
            i = np.nonzero(m & (J == d.k.get(s)))[0]
            if len(i):
                rows[s] = {"gap_pct": round(float(yy[i[0]]), 2), "ml3_wf": [round(float(x), 2) for x in q3[i[0]]]}
        days[iso(day)] = {"ml3_wf": summary(yy, q3, night, m), "names": rows}
    res["days"] = days
    c = res["ml3_wf"]
    e = res["earn_nights"]
    res["pass"] = {"coverage_77_83": 0.77 <= c["coverage"] <= 0.83,
                   "pinball_below_naive_fixed": res["pinball_diff_wf_minus_naive_fixed"][1][1] < 0,
                   "pinball_below_naive_wf": res["pinball_diff_wf_minus_naive_wf"][1][1] < 0,
                   "earn_coverage_70": e["coverage"] >= 0.70, "median_45_55": 0.45 <= c["below_median"] <= 0.55}
    res["refits"] = fits
    return res, M2


def main():
    if "--amend1" in sys.argv:
        d = trader.Data()
        res, M2 = amend1(d)
        print(json.dumps({k: res[k] for k in ("ml3_wf", "naive_fixed", "naive_wf", "pinball_diff_wf_minus_naive_fixed",
                                               "pinball_diff_wf_minus_naive_wf", "pass")}, indent=1), flush=True)
        (RES / "ml3_amend1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        (RES / "ml3_model_amend1.json").write_text(json.dumps(M2, indent=1), encoding="utf-8")
        return
    d = trader.Data()
    print("data", d.dates[0], d.dates[-1], len(d.sym), flush=True)
    g, M = gap1(d)
    print(json.dumps({k: g["gap1"][k] for k in ("ml3", "naive", "pinball_diff_ml3_minus_naive")}, indent=1), flush=True)
    print("pass gap1", g["pass"], flush=True)
    s = intraday(d)
    print("size_scale_now", json.dumps({k: s[k] for k in ("sessions", "A_mean_drho_10_to_1530", "B_1530", "pass")}), flush=True)
    (RES / "ml3_model.json").write_text(json.dumps(M, indent=1), encoding="utf-8")
    (RES / "ml3_validation.json").write_text(json.dumps({"gap1": g, "size_scale_now": s}, indent=1, default=float),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
