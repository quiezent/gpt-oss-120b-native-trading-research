"""Wave C "macro_feats": scheduled US macro releases as RISK features (never direction).
Pre-registration: runs/ml/research/macro_feats_prereg.md.

    .venv\\Scripts\\python.exe ml_research\\macro_feats.py             # validate(): writes runs/ml/research/macro_feats.json
    .venv\\Scripts\\python.exe ml_research\\macro_feats.py --today     # print the "macro today" line for now (ET)

Input: data/macro/calendar.json (ml_research/macro_fetch.py). No network calls here.

What it tests: flags for a release in the next / last 60 minutes, an 08:30 release this morning, FOMC afternoon, and a
surprise z-score (first release vs the Kalshi pre-release mean), added to
  (a) the intraday range model (ml_research/intraday.py: next60, rest_of_day), and
  (b) vol2's range1 (next-session high-low), cross-section and SPY.
What ships regardless of the result: macro_today(), a one-line calendar text for a tool (data, not a model).
"""

import json
import math
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

CAL = ROOT / "data" / "macro" / "calendar.json"
OUT = ROOT / "runs" / "ml" / "research"
MODEL = OUT / "macro_feats_model.json"
ET = ZoneInfo("America/New_York")
NS = 13
SLOT_MIN = [570 + 30 * k for k in range(NS)]                 # look times 09:30 .. 15:30, minutes after midnight ET
OPEN_MIN, FOMC_MIN = 570, 840
AM_T1 = {"cpi", "nfp", "pce", "gdp"}                          # tier-1 08:30 releases
AM_T2 = {"ppi", "retail", "eci"}                              # tier-2 08:30 releases (claims left out: weekly)
T1 = AM_T1 | {"fomc"}
T2 = AM_T2 | {"jolts", "ism_mfg", "ism_services", "fomc_minutes"}
Z_FIELDS = {"cpi": ["cpi_mom_pct", "core_cpi_mom_pct"], "nfp": ["nfp_change_k", "unemployment_rate"],
            "pce": ["core_pce_mom_pct"], "gdp": ["real_gdp_qoq_saar"]}
Z_MIN_PRIOR, Z_CAP = 6, 4.0
FLAGS = ["am830_t1", "am830_t2", "next60", "last60", "fomc_ahead", "fomc_after"]
STATES = {0: "none", 1: "fomc_after", 2: "fomc_ahead", 3: "am830_t1 09:30-10:00", 4: "am830_t1 10:30-12:00",
          5: "am830_t1 12:30-15:30", 6: "T2 release within 60 min", 7: "am830_t2", 8: "am830_t1 09:30-10:00, |z|>=1"}
DSTATES = {0: "none", 1: "fomc", 2: "t1am", 3: "t2"}
MIN_STATE_DAYS_I, MIN_STATE_DAYS_D = 3, 8
NBOOT = 2000

# Result of validate() on 18 Oct 2026 (runs/ml/research/macro_feats.{md,json}). No variant changes a forecast:
# - intraday: all four variants failed the pre-registered bar (release-morning R2 down 0.005-0.010);
# - range1: D1 (three flags in the HAR) passed the bar as written but is fragile (dR2 +0.0009, 90% interval -0.008 to
#   +0.009, P(dR2 > 0) 0.57) and lowers SPY's release-session R2; recommended not to wire (see the .md).
SHIP = {"intraday": None, "range1": None}
TRACK_RECORD = (
    "Macro calendar: US release times (ET) and first-release values from FRED/ALFRED, pre-release expectations from "
    "Kalshi markets, FOMC from federalreserve.gov. Tested as risk flags against our range forecasts (pre-registered, out "
    "of sample): on the 29 tier-1 release mornings of Jan-Oct 2026 (CPI, payrolls, PCE, GDP at 08:30) the session ranges "
    "already held 80% (next60) and 82% (rest of day) of outcomes in their 10-90% range at the 09:30 and 10:00 looks, and "
    "none of four release-flag variants improved them, so nothing is adjusted. Next-session range1 likewise: the release "
    "flags' small gain came entirely from the FOMC flag (+0.0195 R2 on 13 FOMC sessions), while the CPI/payroll/PCE/GDP-"
    "morning and tier-2 flags slightly hurt, so nothing is adjusted. One weak spot, seen in the breakdown rather than "
    "pre-registered: on FOMC days the session ranges that include the 14:00 statement ran too narrow (6 days in 2026: "
    "10-90% coverage 44-65% for next60 at the 13:30-14:30 looks and about 65% for rest_of_day over all looks, with about "
    "a third of outcomes above the top; median realized/forecast after 14:00 was 1.1-1.7x on every one of the 6 days). "
    "The calendar says when and what was released, not which way prices will go.")


# ---------------------------------------------------------------- calendar
_CAL, _CAL_MTIME = None, None


def load_calendar(path=CAL, refresh=False):
    """Events from calendar.json with derived fields: ymd (YYYYMMDD), tmin (minutes after midnight ET), dt (aware ET).
    Cached until the file changes (live.py rebuilds it each evening)."""
    global _CAL, _CAL_MTIME
    if _CAL is not None and not refresh and path == CAL and CAL.stat().st_mtime == _CAL_MTIME:
        return _CAL
    ev = json.loads(Path(path).read_text(encoding="utf-8"))["events"]
    out = []
    for e in ev:
        if e.get("kind") == "revision_only":
            continue
        e = dict(e)
        e["ymd"] = e["date"].replace("-", "")
        hh, mm = e["time_et"].split(":")
        e["tmin"] = int(hh) * 60 + int(mm)
        e["dt"] = datetime.fromisoformat(e["datetime_et"]).astimezone(ET)
        out.append(e)
    out.sort(key=lambda e: (e["dt"], e["key"]))
    attach_surprises(out)
    if path == CAL:
        _CAL, _CAL_MTIME = out, CAL.stat().st_mtime
    return out


def _expect_ok(x):
    return isinstance(x, dict) and x.get("implied_mean") is not None and not x.get("delayed_release")


def attach_surprises(events):
    """e['z'] = {field: z} for T1 08:30 fields with a usable Kalshi expectation, scaled by 1.4826 x the median absolute
    surprise of that field over strictly earlier releases (>= 6 of them); e['absz'] = max |z| capped at 4 (0 if none)."""
    past = {}
    for e in events:
        e["z"], e["absz"] = {}, 0.0
        if e["key"] not in Z_FIELDS or e.get("status") != "released":
            continue
        fr, ex = e.get("first_release") or {}, e.get("expectation") or {}
        new = {}
        for f in Z_FIELDS[e["key"]]:
            a, x = fr.get(f), ex.get(f)
            if a is None or not _expect_ok(x):
                continue
            s = a - x["implied_mean"]
            hist = past.get(f, [])
            if len(hist) >= Z_MIN_PRIOR:
                sc = 1.4826 * float(np.median(np.abs(hist)))
                if sc > 0:
                    e["z"][f] = round(s / sc, 2)
            new[f] = s
        for f, s in new.items():
            past.setdefault(f, []).append(s)
        if e["z"]:
            e["absz"] = min(Z_CAP, max(abs(v) for v in e["z"].values()))


def _by_day(events, keys):
    out = {}
    for e in events:
        if e["key"] in keys:
            out.setdefault(e["ymd"], []).append(e)
    return out


def _released_only(events):
    return [e for e in events if e.get("status") == "released"]


# ---------------------------------------------------------------- features
def look_flags(day_events, tmin):
    """Flags for one look at minute tmin of a day, from that day's events (T1 + T2)."""
    keys = {e["key"] for e in day_events}
    am = [e for e in day_events if e["tmin"] < OPEN_MIN]
    f = {"am830_t1": float(any(e["key"] in AM_T1 for e in am)),
         "am830_t2": float(any(e["key"] in AM_T2 for e in am)),
         "next60": float(any(tmin <= e["tmin"] < tmin + 60 for e in day_events if e["tmin"] >= OPEN_MIN)),
         "last60": float(any(max(tmin - 60, OPEN_MIN) <= e["tmin"] < tmin for e in day_events)),
         "fomc_ahead": float("fomc" in keys and tmin < FOMC_MIN),
         "fomc_after": float("fomc" in keys and tmin >= FOMC_MIN)}
    f["absz_am"] = float(max([e.get("absz", 0.0) for e in am if e["key"] in AM_T1] or [0.0]))
    return f


def look_state(f, k, split_z=False):
    """Mutually exclusive state for the resq variants (pre-registered order)."""
    if f["fomc_after"]:
        return 1
    if f["fomc_ahead"]:
        return 2
    if f["am830_t1"]:
        if k <= 1:
            return 8 if (split_z and f["absz_am"] >= 1) else 3
        return 4 if k <= 5 else 5
    if f["next60"] or f["last60"]:
        return 6
    if f["am830_t2"]:
        return 7
    return 0


def intraday_flags(dates, events=None):
    """{name: (D, 13)} flags for each day (YYYYMMDD) and look, plus 'state' and 'state_z' (D, 13) ints."""
    events = _released_only(load_calendar() if events is None else events)
    byd = _by_day(events, T1 | T2)
    D = len(dates)
    out = {n: np.zeros((D, NS)) for n in FLAGS + ["absz_am"]}
    out["state"], out["state_z"] = np.zeros((D, NS), int), np.zeros((D, NS), int)
    for i, x in enumerate(dates):
        de = byd.get(str(x), [])
        for k in range(NS):
            f = look_flags(de, SLOT_MIN[k])
            for n in f:
                out[n][i, k] = f[n]
            out["state"][i, k], out["state_z"][i, k] = look_state(f, k), look_state(f, k, True)
    return out


def daily_flags(dates, events=None):
    """{d_t1am, d_fomc, d_t2: (D,)} for the session on each date, and 'dstate' (D,) ints."""
    events = _released_only(load_calendar() if events is None else events)
    byd = _by_day(events, T1 | T2)
    D = len(dates)
    out = {n: np.zeros(D) for n in ("d_t1am", "d_fomc", "d_t2")}
    out["dstate"] = np.zeros(D, int)
    for i, x in enumerate(dates):
        keys = {e["key"] for e in byd.get(str(x), [])}
        out["d_t1am"][i] = float(bool(keys & AM_T1))
        out["d_fomc"][i] = float("fomc" in keys)
        out["d_t2"][i] = float(bool(keys & T2))
        out["dstate"][i] = 1 if "fomc" in keys else 2 if keys & AM_T1 else 3 if keys & T2 else 0
    return out


# ---------------------------------------------------------------- scoring helpers
def _day_sums(y, a, b, lo_a, hi_a, lo_b, hi_b, day):
    """Per-day sufficient statistics for R2 and coverage of two forecasts a, b on the same cells."""
    nd = int(day.max()) + 1 if len(day) else 0
    bc = lambda w: np.bincount(day, weights=w, minlength=nd)
    return {"n": bc(np.ones_like(y)), "sy": bc(y), "syy": bc(y * y), "sea": bc((y - a) ** 2), "seb": bc((y - b) ** 2),
            "ina": bc(((y >= lo_a) & (y <= hi_a)).astype(float)), "inb": bc(((y >= lo_b) & (y <= hi_b)).astype(float)),
            "bla": bc((y < lo_a).astype(float)), "aba": bc((y > hi_a).astype(float))}


def _stats(s, w=None):
    w = np.ones_like(s["n"]) if w is None else w
    g = {k: float(np.sum(v * w)) for k, v in s.items()}
    sst = g["syy"] - g["sy"] ** 2 / g["n"]
    return {"r2_a": 1 - g["sea"] / sst, "r2_b": 1 - g["seb"] / sst, "cov_a": g["ina"] / g["n"], "cov_b": g["inb"] / g["n"],
            "below_a": g["bla"] / g["n"], "above_a": g["aba"] / g["n"]}


def compare_cells(y, a, b, lo_a, hi_a, lo_b, hi_b, day, seed=0, boot=True):
    """a = variant, b = shipped. Point stats plus a day-clustered bootstrap (NBOOT) for dR2 and the variant's coverage."""
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b) & np.isfinite(lo_a) & np.isfinite(hi_a) & np.isfinite(lo_b) & np.isfinite(hi_b)
    if ok.sum() == 0:
        return {"cells": 0}
    _, day_i = np.unique(day[ok], return_inverse=True)
    s = _day_sums(y[ok], a[ok], b[ok], lo_a[ok], hi_a[ok], lo_b[ok], hi_b[ok], day_i)
    st = _stats(s)
    nd = len(s["n"])
    out = {"cells": int(ok.sum()), "days": nd, "r2_variant": round(st["r2_a"], 4), "r2_shipped": round(st["r2_b"], 4),
           "d_r2": round(st["r2_a"] - st["r2_b"], 4), "cov_variant": round(st["cov_a"], 4), "cov_shipped": round(st["cov_b"], 4),
           "below_q10_variant": round(st["below_a"], 4), "above_q90_variant": round(st["above_a"], 4)}
    if boot and nd >= 5:
        rng = np.random.default_rng(seed)
        dr, cv, cb = [], [], []
        for _ in range(NBOOT):
            w = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
            b_ = _stats(s, w)
            dr.append(b_["r2_a"] - b_["r2_b"])
            cv.append(b_["cov_a"])
            cb.append(b_["cov_b"])
        q = lambda x: [round(float(v), 4) for v in np.quantile(x, [0.05, 0.95])]
        out.update({"d_r2_90ci": q(dr), "p_d_r2_pos": round(float(np.mean(np.array(dr) > 0)), 3), "cov_variant_90ci": q(cv),
                    "cov_shipped_90ci": q(cb)})
    return out


# ---------------------------------------------------------------- (a) intraday
def _run_range(I, p, RF, NAV, RT, stocks, tk, names, extra, S, D, t0d):
    """intraday.range_eval's walk-forward for one target with feature list `names` (extra arrays merged into RF).
    Returns log point, lo, hi, level point (S, D, 13) and the realized log target y."""
    F5 = dict(RF)
    F5.update(extra)
    F5["lnaive"] = np.log(NAV[tk])
    y = np.log(np.where(stocks[:, None, None] & (RT[tk] > 0), RT[tk], np.nan))
    y = np.where(np.isfinite(F5["lnaive"]), y, np.nan)
    models = []

    def rfit(X, yy):
        m = I.SlotOLS().fit(X, yy)
        models.append(m)
        return m
    t0 = t0d * NS
    lpred, _, _ = I.run_wf(F5, names, y, rfit, lambda m, X: m.predict(X), NS, t0, S)
    lp3 = lpred.reshape(S, D, NS)
    lo3, hi3, lv3 = (np.full((S, D, NS), np.nan) for _ in range(3))
    blocks = [T for T in range(t0, D * NS, I.RETRAIN_DAYS * NS) if T - NS >= I.MIN_DAYS * NS]
    for m, T in zip(models, blocks):
        te = np.arange(T, min(T + I.RETRAIN_DAYS * NS, D * NS))
        for k in range(NS):
            dd = te[te % NS == k] // NS
            lo3[:, dd, k] = lp3[:, dd, k] + m.q[k][0]
            hi3[:, dd, k] = lp3[:, dd, k] + m.q[k][2]
            lv3[:, dd, k] = np.exp(lp3[:, dd, k]) * m.m[k]
    coefs = {}
    if models:
        last = models[-1]
        for k in range(NS):
            coefs[I.SLOTS[k]] = dict(zip(["const"] + names, np.round(last.b[k], 4).tolist()))
    return {"lp": lp3, "lo": lo3, "hi": hi3, "lv": lv3, "y": y, "coef_last": coefs, "blocks": blocks}


def resq(base, state, t0d, D, nmin=MIN_STATE_DAYS_I, block=21):
    """Variant I3/I4: shift the shipped log point/lo/hi on flagged cells by state-conditional residual quantile
    differences (vs unflagged cells in the same looks), pooled from shipped OOS residuals on days before each block."""
    lp, lo, hi, lv, y = base["lp"], base["lo"], base["hi"], base["lv"], base["y"]
    r = y - lp                                                           # (S, D, 13)
    out = {k: v.copy() for k, v in (("lp", lp), ("lo", lo), ("hi", hi), ("lv", lv))}
    shifts = []
    for T in range(t0d, D, block):
        pool = np.arange(t0d, T)
        te = np.arange(T, min(T + block, D))
        rec = {"block_from": int(T), "states": {}}
        for s in sorted(set(np.unique(state[te])) - {0}):
            ps = state[pool] == s                                         # (len(pool), 13)
            ndays = int(ps.any(1).sum())
            if ndays < nmin:
                rec["states"][STATES[int(s)]] = {"pool_days": ndays, "applied": False}
                continue
            slots = np.where(ps.any(0))[0]
            rs = r[:, pool][:, ps]
            rs = rs[np.isfinite(rs)]
            pn = (state[pool] == 0)
            pn[:, [k for k in range(NS) if k not in slots]] = False
            rn = r[:, pool][:, pn]
            rn = rn[np.isfinite(rn)]
            dq10 = float(np.quantile(rs, 0.1) - np.quantile(rn, 0.1))
            dq90 = float(np.quantile(rs, 0.9) - np.quantile(rn, 0.9))
            dm = float(rs.mean() - rn.mean())
            rec["states"][STATES[int(s)]] = {"pool_days": ndays, "applied": True, "d_q10": round(dq10, 4),
                                             "d_q90": round(dq90, 4), "d_mean": round(dm, 4)}
            for i in te:
                ks = np.where(state[i] == s)[0]
                if len(ks):
                    out["lp"][:, i, ks] += dm
                    out["lo"][:, i, ks] += dq10
                    out["hi"][:, i, ks] += dq90
                    out["lv"][:, i, ks] *= math.exp(dm)
        shifts.append(rec)
    out["y"] = y
    out["shifts"] = shifts
    return out


def final_shifts_intraday(base, state, t0d, D, nmin=MIN_STATE_DAYS_I):
    """State shifts from every OOS residual (for live use, only if a resq variant ships)."""
    r = base["y"] - base["lp"]
    pool = np.arange(t0d, D)
    res = {}
    for s in sorted(set(np.unique(state[pool])) - {0}):
        ps = state[pool] == s
        if int(ps.any(1).sum()) < nmin:
            continue
        slots = np.where(ps.any(0))[0]
        rs = r[:, pool][:, ps]
        rs = rs[np.isfinite(rs)]
        pn = state[pool] == 0
        pn[:, [k for k in range(NS) if k not in slots]] = False
        rn = r[:, pool][:, pn]
        rn = rn[np.isfinite(rn)]
        res[STATES[int(s)]] = {"days": int(ps.any(1).sum()), "d_q10": float(np.quantile(rs, .1) - np.quantile(rn, .1)),
                               "d_q90": float(np.quantile(rs, .9) - np.quantile(rn, .9)), "d_mean": float(rs.mean() - rn.mean())}
    return res


def validate_intraday(log=print):
    from ml_research import intraday as I
    p = I.load_panel()
    syms, dates, etf = [str(s) for s in p["symbols"]], [str(x) for x in p["dates"]], p["etf"].astype(bool)
    S, D = len(syms), len(dates)
    stocks = ~etf
    spy = syms.index("SPY")
    t0d = next(i for i, x in enumerate(dates) if x >= I.TEST_FROM)
    RF, NAV, RT = I.range_features(p, spy)
    fl = intraday_flags(dates)
    bcast = lambda a: np.broadcast_to(a[None], (S, D, NS)).copy()
    extra = {n: bcast(fl[n]) for n in FLAGS + ["absz_am"]}
    stored = np.load(OUT / "intraday_oos.npz", allow_pickle=True)
    assert [str(s) for s in stored["symbols"]] == syms and [str(x) for x in stored["dates"]] == dates
    day_ix = np.broadcast_to(np.arange(D)[None, :, None], (S, D, NS))
    slot_ix = np.broadcast_to(np.arange(NS)[None, None, :], (S, D, NS))
    test = day_ix >= t0d
    am = fl["am830_t1"][None, :, :].astype(bool) & np.ones((S, 1, 1), bool)
    rm = am & (slot_ix <= 1) & test
    other = test & ~rm
    st3 = np.broadcast_to(fl["state"][None], (S, D, NS))
    days_rm = sorted({dates[i] for i in range(t0d, D) if fl["am830_t1"][i, 0]})
    ev = _released_only(load_calendar())
    byd = _by_day(ev, T1 | T2)
    res = {"test_window": [dates[t0d], dates[-1]], "test_days": D - t0d, "release_mornings": days_rm,
           "n_release_mornings": len(days_rm), "state_days_in_test": {}, "targets": {}}
    for s, nm in STATES.items():
        if s:
            col = fl["state_z" if s == 8 else "state"][t0d:]
            res["state_days_in_test"][nm] = int((col == s).any(1).sum())
    log("release mornings in test:", len(days_rm))
    # sub-groups (reported, not pass criteria)
    key_days = lambda keys: np.array([bool({e["key"] for e in byd.get(x, [])} & keys) for x in dates])

    def groups():
        g = {"release_mornings_0930_1000": rm, "other_cells": other}
        g["release_mornings_excl_20260122"] = rm & (np.array(dates) != "20260122")[None, :, None]
        for nm, keys in (("cpi_mornings", {"cpi"}), ("nfp_mornings", {"nfp"}), ("pce_or_gdp_mornings", {"pce", "gdp"})):
            g[nm] = rm & key_days(keys)[None, :, None]
        g["release_days_later_looks_1030_1530"] = am & (slot_ix >= 2) & test
        for s in (1, 2, 6, 7):
            g["state:" + STATES[s]] = (st3 == s) & test
        g["no_release_cells"] = (st3 == 0) & test
        hiz = np.broadcast_to(fl["absz_am"][None] >= 1, (S, D, NS))
        g["release_mornings_absz_ge1"] = rm & hiz
        g["release_mornings_absz_lt1"] = rm & ~hiz
        return g
    G = groups()
    for tk, tname in (("rng60", "next60"), ("rngrod", "rest_of_day")):
        base = _run_range(I, p, RF, NAV, RT, stocks, tk, I.RNG, {}, S, D, t0d)
        lp_st = stored[f"range_{tk}_log"].reshape(S, D, NS).astype(float)
        both = np.isfinite(lp_st) & np.isfinite(base["lp"])
        repro = {"max_abs_diff_log": float(np.nanmax(np.abs(lp_st[both] - base["lp"][both]))),
                 "cells": int(both.sum()), "nan_mismatch": int((np.isfinite(lp_st) != np.isfinite(base["lp"])).sum())}
        log(tname, "reproduction", repro)
        V = {"I1_refit_flags": _run_range(I, p, RF, NAV, RT, stocks, tk, I.RNG + FLAGS, extra, S, D, t0d),
             "I2_refit_flags_z": _run_range(I, p, RF, NAV, RT, stocks, tk, I.RNG + FLAGS + ["absz_am"], extra, S, D, t0d),
             "I3_resq": resq(base, fl["state"], t0d, D),
             "I4_resq_z": resq(base, fl["state_z"], t0d, D)}
        y = base["y"]
        tres = {"reproduction_of_shipped": repro, "variants": {}}
        # shipped alone on each group (coverage and R2 of shipped vs naive is not needed: shipped vs itself = baseline)
        tres["shipped_baseline"] = {g: compare_cells(y[m], base["lp"][m], base["lp"][m], base["lo"][m], base["hi"][m],
                                                     base["lo"][m], base["hi"][m], day_ix[m], boot=True)
                                    for g, m in G.items()}
        for vn, v in V.items():
            card = {}
            for g, m in G.items():
                card[g] = compare_cells(y[m], v["lp"][m], base["lp"][m], v["lo"][m], v["hi"][m], base["lo"][m],
                                        base["hi"][m], day_ix[m], seed=1, boot=g in ("release_mornings_0930_1000", "other_cells") or True)
            # level R2 on release mornings (reported)
            m = rm
            yl = np.exp(y[m])
            card["release_mornings_level_r2"] = compare_cells(yl, v["lv"][m], base["lv"][m], v["lo"][m], v["hi"][m],
                                                              base["lo"][m], base["hi"][m], day_ix[m], boot=False)
            rmc, oc = card["release_mornings_0930_1000"], card["other_cells"]
            c1 = 0.78 <= rmc["cov_variant"] <= 0.85
            c2 = rmc["d_r2"] > 0
            c3 = oc["d_r2"] >= -0.001 and abs(oc["cov_variant"] - oc["cov_shipped"]) <= 0.010
            card["pass_parts"] = {"coverage_78_85": c1, "r2_up": c2, "no_loss_elsewhere": c3}
            card["pass"] = bool(c1 and c2 and c3)
            card["fragile"] = bool(card["pass"] and rmc.get("p_d_r2_pos", 0) < 0.8)
            if "coef_last" in v:
                card["last_fit_flag_coefs"] = {sl: {n: c[n] for n in c if n in FLAGS + ["absz_am"]} for sl, c in v["coef_last"].items()}
            if "shifts" in v:
                card["shifts_by_block"] = v["shifts"]
            tres["variants"][vn] = card
            log(tname, vn, json.dumps({"rm": {k: rmc.get(k) for k in ("cov_variant", "cov_shipped", "d_r2", "d_r2_90ci", "p_d_r2_pos")},
                                       "other": {k: oc.get(k) for k in ("d_r2", "cov_variant", "cov_shipped")}, "pass": card["pass"]}))
        tres["final_shifts_if_resq_ships"] = {"I3": final_shifts_intraday(base, fl["state"], t0d, D),
                                              "I4": final_shifts_intraday(base, fl["state_z"], t0d, D)}
        res["targets"][tname] = tres
    passing = [vn for vn in ("I3_resq", "I4_resq_z", "I1_refit_flags", "I2_refit_flags_z")
               if all(res["targets"][t]["variants"][vn]["pass"] for t in ("next60", "rest_of_day"))]
    if passing:
        best = max(passing, key=lambda vn: np.mean([res["targets"][t]["variants"][vn]["release_mornings_0930_1000"]["r2_variant"]
                                                    for t in ("next60", "rest_of_day")]))
    else:
        best = None
    res["ship"] = best
    return res


# ---------------------------------------------------------------- (b) daily range1
def validate_daily(log=print):
    import trader
    from ml_research import risk, vol2
    d = trader.Data()
    z = np.load(OUT / "vol2_oos.npz", allow_pickle=True)
    assert [str(s) for s in z["symbols"]] == list(d.sym) and [str(x) for x in z["dates"]] == list(d.dates)
    F, Sg, Y, N, V, _ = risk.build(d)
    y = Y["range1"]
    nsym, nd = y.shape
    t0 = next(i for i, s in enumerate(d.dates) if s >= risk.TEST_FROM)
    fl = daily_flags(d.dates)
    nxt = lambda a: np.concatenate([a[1:], [0.0]])             # forecast column t is for session t+1
    M = dict(Sg)
    fcols = {}
    for n in ("d_t1am", "d_fomc", "d_t2"):
        fcols[n] = nxt(fl[n])
        M[n] = np.broadcast_to(fcols[n], y.shape).copy()
    dstate = np.concatenate([fl["dstate"][1:], [0]])
    fit = lambda Xa, ya: risk.ols_fit(Xa, risk.winsor(ya))
    har = risk.wf(M, risk.HAR_LVL, y, 1, t0, fit, risk.ols_pred)
    ship, slo, shi = z["range1"], z["range1_lo"], z["range1_hi"]
    both = np.isfinite(har) & np.isfinite(ship)
    repro = {"max_abs_diff": float(np.nanmax(np.abs(har[both] - ship[both]))), "cells": int(both.sum())}
    log("range1 reproduction", repro)
    p1 = risk.wf(M, risk.HAR_LVL + list(fcols), y, 1, t0, fit, risk.ols_pred)
    lo1, hi1 = vol2.interval_wf(p1, y, 1, True, False, list(range(nd)))
    # D2 resq: ratio residuals of the shipped point, expanding pool realized before the session
    R = y / ship
    p2, lo2, hi2 = ship.copy(), slo.copy(), shi.copy()
    shifts = []
    for T in range(t0, nd):
        pool = np.arange(t0, max(T - 1, t0))                    # target of column t realized at t+1 <= T-1
        s = int(dstate[T])
        if s == 0:
            continue
        ps = pool[dstate[pool] == s]
        pn = pool[dstate[pool] == 0]
        if len(ps) < MIN_STATE_DAYS_D:
            continue
        rs, rn = R[:, ps], R[:, pn]
        rs, rn = rs[np.isfinite(rs)], rn[np.isfinite(rn)]
        mult = float(rs.mean() / rn.mean())
        q10, q90 = np.quantile(rs, [0.1, 0.9])
        p2[:, T] = ship[:, T] * mult
        lo2[:, T] = ship[:, T] * q10
        hi2[:, T] = ship[:, T] * q90
        shifts.append({"date": d.dates[T], "state": DSTATES[s], "pool_days": int(len(ps)), "mult": round(mult, 4),
                       "q10": round(float(q10), 4), "q90": round(float(q90), 4)})
    steps = np.arange(t0, nd - 1)
    rel = np.zeros(nd, bool)
    rel[steps] = (fcols["d_t1am"][steps] + fcols["d_fomc"][steps]) > 0
    col = np.broadcast_to(np.arange(nd)[None], y.shape)
    test = np.zeros(y.shape, bool)
    test[:, steps] = True
    spy = d.k["SPY"]
    rows = {"cross_section": np.ones(nsym, bool), "SPY": np.arange(nsym) == spy}
    res = {"test_window": [d.dates[t0], d.dates[nd - 2]], "reproduction_of_shipped": repro,
           "release_sessions": int(rel.sum()), "state_sessions_in_test": {DSTATES[s]: int((dstate[steps] == s).sum()) for s in (1, 2, 3)},
           "variants": {}}
    groups = {"release_sessions": rel, "other_sessions": ~rel & np.isin(np.arange(nd), steps),
              "fomc_sessions": dstate == 1, "t1am_sessions": dstate == 2, "t2_sessions": dstate == 3}
    for vn, (pv, lv_, hv) in (("D1_refit_flags", (p1, lo1, hi1)), ("D2_resq", (p2, lo2, hi2))):
        card = {}
        for rn_, rmask in rows.items():
            card[rn_] = {}
            for g, gm in groups.items():
                m = test & rmask[:, None] & gm[None, :]
                card[rn_][g] = compare_cells(y[m], pv[m], ship[m], lv_[m], hv[m], slo[m], shi[m], col[m], seed=2,
                                             boot=nsym > 0)
        xs = card["cross_section"]
        a, b = xs["release_sessions"], xs["other_sessions"]
        c1 = a["d_r2"] > 0
        c2 = 0.75 <= a["cov_variant"] <= 0.85 and abs(a["cov_variant"] - 0.8) <= abs(a["cov_shipped"] - 0.8) + 1e-9
        c3 = b["d_r2"] >= -0.001 and abs(b["cov_variant"] - b["cov_shipped"]) <= 0.010
        card["pass_parts"] = {"r2_up": c1, "coverage_75_85_and_closer": c2, "no_loss_elsewhere": c3}
        card["pass"] = bool(c1 and c2 and c3)
        card["spy_flag_lower_r2_on_release"] = bool(card["SPY"]["release_sessions"].get("d_r2", 0) < 0)
        card["fragile"] = bool(card["pass"] and a.get("p_d_r2_pos", 0) < 0.8)
        res["variants"][vn] = card
        log(vn, json.dumps({"rel": {k: a.get(k) for k in ("d_r2", "d_r2_90ci", "cov_variant", "cov_shipped")},
                            "other": {k: b.get(k) for k in ("d_r2", "cov_variant", "cov_shipped")},
                            "spy_rel": {k: card["SPY"]["release_sessions"].get(k) for k in ("d_r2", "cov_variant", "cov_shipped")},
                            "pass": card["pass"]}))
    # D1's fitted flag coefficients on the last training block (full-sample refit for the report)
    tr = np.arange(risk.START, nd - 1)
    import ml
    b = risk.ols_fit(ml.rows(M, risk.HAR_LVL + list(fcols), tr), risk.winsor(y[:, tr].T.reshape(-1)))
    res["D1_full_sample_flag_coefs_pct_points"] = {n: round(float(c) * 100, 4) for n, c in zip(list(fcols), b[-3:])}
    res["D2_last_shifts"] = shifts[-6:]
    res["D2_n_adjusted_sessions"] = len(shifts)
    # descriptive: mean realized/shipped ratio by state over the whole test window
    res["realized_over_shipped_by_state"] = {}
    for s in (0, 1, 2, 3):
        cs = steps[dstate[steps] == s]
        rr = R[:, cs]
        rr = rr[np.isfinite(rr)]
        rs_ = R[spy, cs]
        rs_ = rs_[np.isfinite(rs_)]
        res["realized_over_shipped_by_state"][DSTATES[s]] = {"sessions": int(len(cs)), "xs_mean": round(float(rr.mean()), 4),
                                                             "spy_mean": round(float(rs_.mean()), 4) if len(rs_) else None}
    passing = [v for v in ("D2_resq", "D1_refit_flags") if res["variants"][v]["pass"]]
    res["ship"] = passing[0] if passing else None
    res["ship_after_review"] = None           # synthesis_wave_c.md: D1 is not wired (its gain is all the FOMC flag)
    return res


def validate():
    t = datetime.now(timezone.utc)
    res = {"run_utc": t.isoformat(timespec="seconds"), "prereg": "runs/ml/research/macro_feats_prereg.md",
           "calendar_built_et": json.loads(CAL.read_text(encoding="utf-8"))["meta"].get("built_et")}
    ev = load_calendar()
    res["surprise_z_examples"] = [{"id": e["id"], "z": e["z"]} for e in ev if e["z"]][-8:]
    res["surprise_z_count"] = sum(1 for e in ev if e["z"])
    res["intraday"] = validate_intraday()
    (OUT / "macro_feats.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    res["range1"] = validate_daily()
    (OUT / "macro_feats.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    model = {"fit_from": "intraday_oos.npz shipped residuals, 20260102..20261016",
             "ship": {"intraday": res["intraday"]["ship"], "range1": res["range1"]["ship"]}, "ship_after_review": None,
             "intraday_shifts": {t: res["intraday"]["targets"][t]["final_shifts_if_resq_ships"]
                                 for t in ("next60", "rest_of_day")}}
    MODEL.write_text(json.dumps(model, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- text, live, simulator
def _pct(v, signed=True, nd=1):
    return (f"{v:+.{nd}f}%" if signed else f"{v:.{nd}f}%") if v is not None else "na"


def _val_text(e):
    fr = e.get("first_release") or {}
    k = e["key"]
    g = fr.get
    try:
        if k == "cpi":
            return f"{_pct(g('cpi_mom_pct'))} m/m, core {_pct(g('core_cpi_mom_pct'))}"
        if k == "nfp":
            return f"payrolls {g('nfp_change_k'):+.0f}k, unemployment {g('unemployment_rate'):.1f}%"
        if k == "pce":
            return f"PCE {_pct(g('pce_mom_pct'))} m/m, core {_pct(g('core_pce_mom_pct'))}"
        if k == "gdp":
            n = {1: "advance", 2: "second", 3: "third"}.get(g("estimate_n"), "")
            return f"real GDP {g('real_gdp_qoq_saar'):.1f}% q/q annualized ({n} estimate)"
        if k == "ppi":
            return f"final demand {_pct(g('ppi_fd_mom_pct'))} m/m, core {_pct(g('core_ppi_mom_pct'))}"
        if k == "retail":
            return f"{_pct(g('retail_mom_pct'))} m/m, ex autos {_pct(g('retail_ex_auto_mom_pct'))}"
        if k == "claims":
            return f"{g('initial_claims') / 1000:.0f}k"
        if k == "jolts":
            return f"openings {g('job_openings_k') / 1000:.2f}m"
        if k == "eci":
            return f"{_pct(g('eci_qoq_pct'))} q/q"
        if k == "adp":
            return f"{g('adp_change_k'):+.0f}k"
    except (TypeError, ValueError):
        return None
    vals = [f"{a} {b}" for a, b in fr.items() if isinstance(b, (int, float)) and not isinstance(b, bool)]
    return ", ".join(vals) if vals else None


_EXP_FMT = {"cpi_mom_pct": "{:+.2f}%", "core_cpi_mom_pct": "core {:+.2f}%", "nfp_change_k": "{:+.0f}k",
            "unemployment_rate": "unemployment {:.2f}%", "core_pce_mom_pct": "core {:+.2f}%", "real_gdp_qoq_saar": "{:.1f}%",
            "initial_claims": "{:.0f}", "retail_mom_pct": "{:+.2f}%", "job_openings_k": "{:.0f}k",
            "cpi_yoy_pct": "y/y {:.2f}%", "core_cpi_yoy_pct": "core y/y {:.2f}%", "pce_mom_pct": "{:+.2f}%"}


def _exp_text(e, now_utc):
    ex = e.get("expectation") or {}
    parts = []
    for f, x in ex.items():
        if not isinstance(x, dict) or x.get("delayed_release"):
            continue
        try:
            asof = datetime.fromisoformat(x["as_of_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=timezone.utc)
        if asof > now_utc:
            continue                                            # not known at this decision time
        if f == "rate_change_bps" and x.get("probabilities_bps"):
            pr = sorted(((float(v), int(b)) for b, v in x["probabilities_bps"].items()), reverse=True)
            name = lambda b: "hold" if b == 0 else f"{b:+d}bp"
            parts.append("Kalshi " + ", ".join(f"{name(b)} {v:.0%}" for v, b in pr if v >= 0.05))
        elif x.get("implied_mean") is not None:
            m = x["implied_mean"] / (1000 if f == "initial_claims" else 1)
            fmt = _EXP_FMT.get(f, "{:.2f}")
            if f == "initial_claims":
                fmt = "{:.0f}k"
            parts.append(fmt.format(m))
    if not parts:
        return None
    if parts[0].startswith("Kalshi"):
        return parts[0]
    return "Kalshi expected " + ", ".join(parts)


def _name(e):
    return {"nfp": "jobs report", "fomc": "FOMC statement", "claims": "jobless claims", "ism_mfg": "ISM manufacturing",
            "ism_services": "ISM services", "fomc_minutes": "FOMC minutes", "retail": "retail sales", "pce": "PCE",
            "gdp": "GDP", "cpi": "CPI", "ppi": "PPI", "jolts": "JOLTS", "eci": "ECI"}.get(e["key"], e["name"])


TEXT_KEYS = T1 | T2 | {"claims"}


def macro_today(now=None, events=None, days_ahead=7, value_delay_min=0):
    """One line for a tool: today's scheduled US macro releases (ET) with first-release values once out and the Kalshi
    pre-release expectation, plus the next tier-1 events. Uses only information dated <= now. value_delay_min hides a
    first-release value until that many minutes after the release (the simulator uses 60 to match what the live feed
    can have: FRED posts CPI about 09:08 ET and payrolls about 09:24)."""
    now = _now_et(now)
    now_utc = now.astimezone(timezone.utc)
    ev = load_calendar() if events is None else events
    today = now.strftime("%Y%m%d")
    todays = [e for e in ev if e["ymd"] == today and e["key"] in TEXT_KEYS]
    items = []
    for e in todays:
        t = e["time_et"]
        if e["dt"] <= now:
            known = e.get("status") == "released" and e["dt"] + timedelta(minutes=value_delay_min) <= now
            v = _val_text(e) if known else None
            if e["key"] == "fomc":
                ex = (e.get("expectation") or {}).get("rate_change_bps") or {}
                v = ex.get("settled_outcome") if known else None
                v = f"decision: {v}" if v else "released; decision not yet in our data"
            elif e["key"] in ("ism_mfg", "ism_services", "fomc_minutes") and not v:
                v = "released"
            elif not v:
                v = "released; value not yet in our data"
            ex = _exp_text(e, now_utc) if e["key"] != "fomc" else None
            zt = ""
            if e.get("z") and known:
                zt = "; surprise " + ", ".join(f"{'core ' if 'core' in f else ''}z {z:+.1f}" for f, z in e["z"].items()
                                               if f in ("cpi_mom_pct", "core_cpi_mom_pct", "nfp_change_k", "core_pce_mom_pct", "real_gdp_qoq_saar"))
            unc = " (release time unconfirmed)" if e.get("time_check", {}).get("time_uncertain") else ""
            items.append(f"{t} {_name(e)} {v}" + (f" ({ex}{zt})" if ex else "") + unc)
        else:
            extra = []
            if e["key"] == "fomc":
                fr = e.get("first_release") or {}
                if fr.get("sep_meeting"):
                    extra.append("with projections")
                if fr.get("press_conference_et"):
                    extra.append(f"press conference {fr['press_conference_et']}")
            ex = _exp_text(e, now_utc)
            if ex:
                extra.append(ex)
            if e.get("time_check", {}).get("time_uncertain") or e.get("date_conflict"):
                extra.append("date/time unconfirmed")
            items.append(f"{t} {_name(e)} scheduled" + (f" ({'; '.join(extra)})" if extra else ""))
    head = f"macro today {now.strftime('%a %Y-%m-%d')} ET, as of {now.strftime('%H:%M')}: "
    body = "; ".join(items) if items else "no scheduled US macro releases"
    # next tier-1 events after today
    end = (now + timedelta(days=days_ahead)).strftime("%Y%m%d")
    nxt = [e for e in ev if today < e["ymd"] <= end and e["key"] in T1]
    if nxt:
        days = {}
        for e in nxt:
            days.setdefault(e["dt"].strftime("%a %m-%d"), []).append(f"{e['time_et']} {_name(e)}")
        tail = "; next: " + ", ".join(f"{k} {' + '.join(v)}" for k, v in days.items())
    else:
        tail = f"; no tier-1 release in the next {days_ahead} days"
    return head + body + tail + "."


def _now_et(now):
    if now is None:
        return datetime.now(ET)
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    return now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)


def live(gw=None, symbols=None, now=None, **kw):
    """Macro context for a look at `now` (ET): the one-line text, the flags for this look and for the next session,
    and (only if a variant shipped) the range adjustments. No gateway calls; reads data/macro/calendar.json.
    The calendar must be rebuilt for first-release values to appear (FRED posts CPI about 09:08 ET, NFP about 09:24)."""
    try:
        now = _now_et(now)
    except (TypeError, ValueError, AttributeError):
        return {"error": f"bad time {str(now)[:40]!r}: give an ISO time or a datetime"}
    try:
        ev = load_calendar(refresh=True)
    except (OSError, ValueError, KeyError) as e:
        return {"error": f"macro calendar unreadable: {type(e).__name__}"}
    today = now.strftime("%Y%m%d")
    built = None
    try:
        built = json.loads(CAL.read_text(encoding="utf-8"))["meta"].get("built_et")
    except (OSError, ValueError, KeyError):
        pass
    tmin = now.hour * 60 + now.minute
    day_ev = [e for e in ev if e["ymd"] == today and e["key"] in T1 | T2]
    f = look_flags(day_ev, max(tmin, OPEN_MIN))
    nxt_dates = sorted({e["ymd"] for e in ev if e["ymd"] > today})
    nxt_session = next((x for x in nxt_dates if datetime.strptime(x, "%Y%m%d").weekday() < 5), None)
    out = {"as_of_et": now.isoformat(timespec="minutes"), "calendar_built_et": built, "macro_today": macro_today(now, ev),
           "look_flags": f, "look_state": STATES[look_state(f, max(0, min(NS - 1, (tmin - OPEN_MIN) // 30)))],
           "today_session_flags": {k: float(v) for k, v in zip(("d_t1am", "d_fomc", "d_t2"),
                                                                 _dflags({e["key"] for e in day_ev}))},
           "ship": dict(SHIP), "track_record": TRACK_RECORD}
    if built and built[:10] < (now - timedelta(days=7)).strftime("%Y-%m-%d"):
        out["warning"] = f"calendar built {built[:10]}, more than 7 days old; scheduled dates may have moved"
    return out


def _dflags(keys):
    return (float(bool(keys & AM_T1)), float("fomc" in keys), float(bool(keys & T2)))


def sim_provider(data, look="09:00", value_delay_min=60):
    """For train2.py: a function of decision session d (decided before the open, as at the live 09:00 look) ->
    (as_of, {"macro_today": text, "session_flags": {...}}). A first-release value shows only `value_delay_min` after
    its release (08:30 values are not in the live feed by 09:00), and a Kalshi expectation only if priced before `look`."""
    ev = load_calendar()
    hh, mm = (int(x) for x in look.split(":"))

    def at(d):
        x = data.dates[d]
        now = datetime.strptime(x, "%Y%m%d").replace(hour=hh, minute=mm, tzinfo=ET)
        keys = {e["key"] for e in ev if e["ymd"] == x and e["key"] in T1 | T2}
        return (f"{x} {look} ET", {"macro_today": macro_today(now, ev, value_delay_min=value_delay_min),
                                  "session_flags": dict(zip(("d_t1am", "d_fomc", "d_t2"), _dflags(keys)))})
    return at


# ---------------------------------------------------------------- tape rows (the live tape and simnews share them)
TAPE_DAYS = 7                         # tape lookback, calendar days


def _ts(s):
    """Epoch seconds from an ISO time ('...Z', '+00:00', or naive = UTC)."""
    dt = datetime.fromisoformat(str(s).strip().replace(" ", "T").replace("Z", "+00:00"))
    return int((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp())


def _iso_utc(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _et(t, fmt="%Y-%m-%d %H:%M ET"):
    return datetime.fromtimestamp(t, ET).strftime(fmt)


def _order(x):
    return (x[0], x[1]["provider"], x[1]["article_id"])


AGENCY = {"cpi": "BLS", "nfp": "BLS", "ppi": "BLS", "jolts": "BLS", "eci": "BLS", "claims": "Labor Dept", "pce": "BEA",
          "gdp": "BEA", "retail": "Census", "housing_starts": "Census", "new_home_sales": "Census", "indpro": "Fed",
          "adp": "ADP", "ism_mfg": "ISM", "ism_services": "ISM", "umich": "U. Michigan", "fomc": "Fed", "fomc_minutes": "Fed"}
EXPECT_UNIT = {"cpi_mom_pct": "%", "core_cpi_mom_pct": "%", "cpi_yoy_pct": "%", "core_pce_mom_pct": "%",
               "real_gdp_qoq_saar": "%", "retail_mom_pct": "%", "unemployment_rate": "%", "nfp_change_k": "k",
               "initial_claims": "", "job_openings_k": "k", "ism_mfg_pmi": ""}


def _month(ref):
    try:
        return datetime.fromisoformat(str(ref)[:10]).strftime("%B")
    except ValueError:
        return None


def _quarter(ref):
    try:
        d = datetime.fromisoformat(str(ref)[:10])
        return f"Q{(d.month - 1) // 3 + 1} {d.year}"
    except ValueError:
        return None


def _chg(v, unit="%", p=1):
    """'rose 0.3%', 'fell 0.2%', 'unchanged' from a value rounded to p decimals."""
    r = round(float(v), p)
    if r == 0:
        return "unchanged"
    return f"{'rose' if r > 0 else 'fell'} {abs(r):.{p}f}{unit}"


def _expected(r, field, release_t):
    """' (Kalshi expected 0.57%)' if a market priced this field before the release, else ''."""
    e = (r.get("expectation") or {}).get(field)
    if not e or e.get("implied_mean") is None or e.get("delayed_release"):
        return ""
    try:
        if _ts(e["as_of_utc"]) >= release_t:
            return ""
    except (KeyError, ValueError):
        return ""
    v, u = e["implied_mean"], EXPECT_UNIT.get(field, "")
    txt = f"{v / 1000:,.0f}k" if field == "initial_claims" else f"{v:,.0f}k" if u == "k" else f"{v:.2f}{u}"
    return f" (Kalshi expected {txt})"


def macro_headline(r, release_t):
    """One headline for a released calendar row, or None (revision-only rows, rows with nothing to say)."""
    k, v = r["key"], r.get("first_release") or {}
    if r.get("kind") == "revision_only":
        return None
    mo, q = _month(r.get("reference_period")), _quarter(r.get("reference_period"))
    x = lambda f: v.get(f) is not None
    E = lambda f: _expected(r, f, release_t)
    if k == "cpi":
        parts = []
        if x("cpi_mom_pct"):
            parts.append(f"CPI {_chg(v['cpi_mom_pct'])} in {mo}{E('cpi_mom_pct')}")
        else:
            parts.append(f"CPI for {mo} released (no month-on-month change: the prior month was not published)")
        if x("core_cpi_mom_pct"):
            parts.append(f"core {_chg(v['core_cpi_mom_pct'])}{E('core_cpi_mom_pct')}")
        if x("cpi_yoy_pct"):
            parts.append(f"{v['cpi_yoy_pct']:.1f}% from a year earlier" + (f" (core {v['core_cpi_yoy_pct']:.1f}%)" if x("core_cpi_yoy_pct") else ""))
        h = ", ".join(parts)
    elif k == "nfp":
        if not x("nfp_change_k"):
            return None
        h = f"Payrolls {'rose' if v['nfp_change_k'] >= 0 else 'fell'} {abs(v['nfp_change_k']):,.0f}k in {mo}{E('nfp_change_k')}"
        if x("unemployment_rate"):
            h += f"; unemployment rate {v['unemployment_rate']:.1f}%{E('unemployment_rate')}"
        if x("ahe_mom_pct"):
            h += f"; hourly earnings {_chg(v['ahe_mom_pct'])}"
    elif k == "pce":
        if not x("pce_mom_pct"):
            return None
        h = f"PCE prices {_chg(v['pce_mom_pct'])} in {mo}"
        if x("core_pce_mom_pct"):
            h += f", core {_chg(v['core_pce_mom_pct'])}{E('core_pce_mom_pct')}"
        if x("pce_yoy_pct"):
            h += f"; {v['pce_yoy_pct']:.1f}% from a year earlier" + (f" (core {v['core_pce_yoy_pct']:.1f}%)" if x("core_pce_yoy_pct") else "")
    elif k == "gdp":
        if not x("real_gdp_qoq_saar"):
            return None
        est = {1: "advance", 2: "second", 3: "third"}.get(v.get("estimate_n"), r.get("kind") or "")
        g = v["real_gdp_qoq_saar"]
        h = f"GDP {'grew' if g >= 0 else 'shrank'} at a {abs(g):.1f}% annual rate in {q}, {est} estimate{E('real_gdp_qoq_saar')}"
        if v.get("prior_vintage_value") is not None:
            h += f", revised from {v['prior_vintage_value']:.1f}%"
    elif k == "ppi":
        if not x("ppi_fd_mom_pct"):
            return None
        h = f"PPI final demand {_chg(v['ppi_fd_mom_pct'])} in {mo}" + (f", core {_chg(v['core_ppi_mom_pct'])}" if x("core_ppi_mom_pct") else "")
    elif k == "retail":
        if not x("retail_mom_pct"):
            return None
        h = f"Retail sales {_chg(v['retail_mom_pct'])} in {mo}{E('retail_mom_pct')}" + (
            f", ex autos {_chg(v['retail_ex_auto_mom_pct'])}" if x("retail_ex_auto_mom_pct") else "")
    elif k == "claims":
        if not x("initial_claims"):
            return None
        wk = r.get("reference_period")
        wk = f" in the week to {datetime.fromisoformat(wk[:10]):%d %b}" if wk else ""
        h = f"Initial jobless claims {v['initial_claims'] / 1000:,.0f}k{wk}{E('initial_claims')}"
    elif k == "jolts":
        if not x("job_openings_k"):
            return None
        h = f"Job openings {v['job_openings_k'] / 1000:.2f} million in {mo} (JOLTS){E('job_openings_k')}"
    elif k == "eci":
        if not x("eci_qoq_pct"):
            return None
        h = f"Employment cost index {_chg(v['eci_qoq_pct'])} in {q}"
    elif k == "adp":
        if not x("adp_change_k"):
            return None
        h = f"ADP private payrolls {'rose' if v['adp_change_k'] >= 0 else 'fell'} {abs(v['adp_change_k']):,.0f}k in {mo}"
    elif k == "indpro":
        if not x("indpro_mom_pct"):
            return None
        h = f"Industrial production {_chg(v['indpro_mom_pct'])} in {mo}"
    elif k == "housing_starts":
        if not x("housing_starts_k_saar"):
            return None
        h = f"Housing starts {v['housing_starts_k_saar'] / 1000:.3f} million annualized in {mo}"
    elif k == "new_home_sales":
        if not x("new_home_sales_k_saar"):
            return None
        h = f"New home sales {v['new_home_sales_k_saar']:,.0f}k annualized in {mo}"
    elif k == "umich":
        if not x("umich_sentiment"):
            return None
        h = f"Michigan consumer sentiment {v['umich_sentiment']:.1f} in {mo} (final)"
    elif k == "ism_mfg":
        h = (f"ISM manufacturing PMI {v['ism_mfg_pmi']:.1f}{E('ism_mfg_pmi')}" if x("ism_mfg_pmi")
             else "ISM manufacturing PMI released")
    elif k == "ism_services":
        h = "ISM services PMI released"
    elif k == "fomc":
        e = (r.get("expectation") or {}).get("rate_change_bps") or {}
        out = e.get("settled_outcome")
        h = f"FOMC decision: {out}" if out else "FOMC statement released"
        if v.get("sep_meeting"):
            h += "; new economic projections"
        if v.get("press_conference_et"):
            h += f"; press conference {v['press_conference_et']} ET"
    elif k == "fomc_minutes":
        h = f"Minutes of the FOMC's {r.get('reference_period') or 'last'} meeting released".replace("*", "")
    else:
        return None
    when = "release time unverified" if _uncertain(r) else f"{_et(release_t, '%H:%M')} ET"
    return f"{h} ({AGENCY.get(k, '')}, {when})".replace("(, ", "(")


def _uncertain(r):
    tc = r.get("time_check") or {}
    return bool(tc.get("time_uncertain") or r.get("time_uncertain"))


def release_ts(r):
    """When a row's value became public, conservatively: an unverified time counts as 10:00 ET if that is later."""
    t = _ts(r["datetime_et"])
    if _uncertain(r):
        late = int(datetime.fromisoformat(r["date"]).replace(hour=10, tzinfo=ET).timestamp())
        t = max(t, late)
    return t


def macro_detail(r):
    """article() text for a macro item: every first-release value and any pre-release Kalshi expectation."""
    t = release_ts(r)
    lines = [f"{r['name']} ({AGENCY.get(r['key'], '')}), released {_et(t)}" + (" (time unverified)" if _uncertain(r) else "")]
    if r.get("reference_period"):
        lines.append(f"Reference period: {r['reference_period']}".replace("*", ""))
    for k, v in (r.get("first_release") or {}).items():
        if k != "source" and v is not None:
            lines.append(f"{k}: {v}")
    if r.get("key") == "fomc":
        e = (r.get("expectation") or {}).get("rate_change_bps") or {}
        if e.get("settled_outcome"):
            lines.append(f"Decision: {e['settled_outcome']}")
    for f, e in (r.get("expectation") or {}).items():
        x = _expected(r, f, t)
        if x:
            lines.append(f"Expected {f}: {x.strip()[len('(Kalshi expected '):-1]} (Kalshi implied mean, priced before the release)")
    lines.append("Values are as first published (not later revisions), to 3 decimals; the agency's 1-decimal headline can differ by rounding.")
    return "\n".join(lines)


def tape_items(rows=None):
    """[(t, row)] sorted by time, in the gateway's row format: a headline per released calendar row (provider MACRO,
    article_id MACRO$<id>; the calendar row rides along in '_detail' for article()), and per day a 07:00 ET 'today's
    calendar' line listing that day's releases (released or scheduled) after 07:00. rows: calendar.json events."""
    if rows is None:
        rows = json.loads(CAL.read_text(encoding="utf-8"))["events"]
    out, by_day = [], {}
    for r in rows:
        if r.get("status") not in ("released", "scheduled"):
            continue
        t = release_ts(r)
        by_day.setdefault(r["date"], []).append((t, r))
        h = macro_headline(r, t) if r["status"] == "released" else None
        if h:
            out.append((t, {"time": _iso_utc(t), "provider": "MACRO", "article_id": f"MACRO${r['id']}", "headline": h,
                            "meta": "", "_detail": r}))
    for day, rs in by_day.items():
        t7 = int(datetime.fromisoformat(day).replace(hour=7, tzinfo=ET).timestamp())
        later = sorted(((t, r) for t, r in rs if t > t7), key=lambda x: (x[0], x[1]["key"]))
        if later:
            txt = "; ".join(f"{r['name']} {'(time unverified)' if _uncertain(r) else _et(t, '%H:%M')}" for t, r in later)
            out.append((t7, {"time": _iso_utc(t7), "provider": "MACRO", "article_id": f"MACRO$cal-{day}",
                             "headline": f"Today's US economic calendar (ET): {txt}", "meta": ""}))
    return sorted(out, key=_order)


def on_tape(t, article_id, now):
    """Whether a macro tape item at epoch t shows at `now` (aware, ET): a release from before today, or today's 07:00
    calendar line. Same-day releases are left out because the live calendar is rebuilt each evening (live.py, after
    the 16:15 record), so a value reaches it the evening after the release; the simulator shows the same."""
    if article_id.startswith("MACRO$cal-"):
        return article_id == f"MACRO$cal-{now:%Y-%m-%d}" and t < now.timestamp()
    return t < now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def tape_rows(now=None, items=None, days=TAPE_DAYS):
    """The macro rows on the tape at `now`, newest first, as [(t, row)]: releases of the last `days` days before today
    (with first-release values and the pre-release Kalshi mean), FOMC decisions, and today's calendar line.
    items: tape_items() (read from calendar.json when None)."""
    now = _now_et(now)
    items = tape_items() if items is None else items
    lo = now.timestamp() - days * 86400
    return sorted((x for x in items if x[0] >= lo and on_tape(x[0], x[1]["article_id"], now)), key=_order, reverse=True)


def tape_article(article_id, now=None, items=None):
    """article() text for a MACRO id shown on the tape at or before `now`, else None."""
    now = _now_et(now)
    for t, r in tape_items() if items is None else items:
        if r["article_id"] == article_id and on_tape(t, article_id, now):
            return macro_detail(r["_detail"]) if "_detail" in r else r["headline"]
    return None


if __name__ == "__main__":
    if "--today" in sys.argv:
        print(macro_today())
    elif "--at" in sys.argv:
        print(macro_today(sys.argv[sys.argv.index("--at") + 1]))
    else:
        r = validate()
        print(json.dumps({"intraday_ship": r["intraday"]["ship"], "range1_ship": r["range1"]["ship"]}))
