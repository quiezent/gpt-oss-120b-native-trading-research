"""Evidence for the cost tool from our own records (week 1 paper fills, 14-16 Oct 2026, and the quote snapshots the
models saw in their looks). Descriptive only: nothing here is fitted except the choice of the live quote-blend rule.

  - Paper-fill slippage vs arrival: the 30-minute TRADES bar's open when the fill is within 3 minutes of the bar's
    start (09:30 fills: the opening print). Signed so that positive = cost. Fills aggregated per order (VWAP).
  - Slippage vs decision price: the delayed price the model saw in its look (quote-tool mid if it called quote for
    the symbol, else the market table's `last`).
  - Snapshot spreads: delayed quote bid/ask from the quote tool vs that session's time-average spread (data/bidask).
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEEK1 = ROOT / "runs" / "archive" / "week1"
LIVE = ROOT / "runs" / "live"
BARS30 = ROOT / "data" / "bars30"


def _jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _bars30(sym, cache={}):
    if sym not in cache:
        p = BARS30 / f"{sym.replace(' ', '_')}.json"
        cache[sym] = {b["date"][:14]: b for b in json.loads(p.read_text(encoding="utf-8"))["bars"]} if p.exists() else {}
    return cache[sym]


def _quote_tables(text):
    """Rows of a quote-tool result: {sym: (last, bid, ask, last_time)}."""
    out = {}
    lines = text.splitlines()
    if not lines or not lines[0].startswith("sym last bid ask"):
        return out
    for ln in lines[1:]:
        m = re.match(r"^(.+?) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)$", ln)
        if not m:
            continue
        sym, last, bid, ask = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            out[sym] = (float(last), float(bid), float(ask), m.group(10))
        except ValueError:
            continue
    return out


def _table_last(state_text):
    out, hdr = {}, None
    for ln in state_text.splitlines():
        parts = ln.split()
        if parts[:2] == ["sym", "close"]:
            hdr = parts
            continue
        if hdr and "last" in hdr and len(parts) == len(hdr):
            try:
                out[parts[0]] = float(parts[hdr.index("last")])
            except ValueError:
                pass
        elif hdr and len(parts) != len(hdr) and parts and not re.match(r"^[A-Z]", parts[0]):
            hdr = None
    return out


def looks():
    """[(date 'YYYYMMDD', 'HHMM', sleeve, record)] for every look record, plus opus research files."""
    res = []
    for f in sorted(LIVE.glob("2026-10-1*/*-[0-9][0-9][0-9][0-9].json")):
        m = re.match(r"(.+)-(\d{4})\.json$", f.name)
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        res.append((f.parent.name.replace("-", ""), m.group(2), m.group(1), rec))
    return res


def snapshots(L):
    snaps = {}
    for date, hhmm, sleeve, rec in L:
        for st in rec.get("trace") or []:
            if st.get("tool") == "quote" and isinstance(st.get("result"), str):
                for sym, (last, bid, ask, lt) in _quote_tables(st["result"]).items():
                    snaps[(date, sym, lt)] = (last, bid, ask, sleeve, hhmm)
    for f in sorted(LIVE.glob("2026-10-1*/*/research.jsonl")):
        date = f.parent.parent.name.replace("-", "")
        for r in _jsonl(f):
            if r.get("tool") == "quote" and isinstance(r.get("result"), str):
                for sym, (last, bid, ask, lt) in _quote_tables(r["result"]).items():
                    snaps.setdefault((date, sym, lt), (last, bid, ask, f.parent.name.split("-")[0], f.parent.name[-4:]))
    return snaps


def evidence(d, S, cal, served, F, cold=None):
    from ml_research.cost import spread_bps
    pred, lo, hi = served
    cpred = cold[0] if cold is not None else np.full(pred.shape, np.nan)
    col = {x: i for i, x in enumerate(cal)}
    L = looks()
    # ---------------- snapshots vs the session's time-average spread
    snaps = snapshots(L)
    rows, thin = [], []
    for (date, sym, lt), (last, bid, ask, sleeve, hhmm) in snaps.items():
        qs = spread_bps(bid, ask) if ask > bid > 0 else np.nan
        if not np.isfinite(qs):
            continue
        j, i = d.k.get(sym), col.get(date)
        if j is None:
            thin.append({"date": date, "symbol": sym, "time": lt, "spread_bps": round(qs, 1), "price": last})
            continue
        if i is None or i == 0 or not np.isfinite(S[j, i]):
            continue
        rth = lt >= "09:30" and lt <= "16:00"
        rows.append((date, sym, lt, rth, qs, S[j, i], np.exp(pred[j, i - 1]), np.exp(lo[j, i - 1]), np.exp(hi[j, i - 1]),
                     np.exp(cpred[j, i - 1])))
    snap = {"n_all": len(rows), "not_in_cache": sorted(thin, key=lambda r: -r["spread_bps"])[:15]}
    for label, sel in (("rth", [r for r in rows if r[3]]), ("premarket", [r for r in rows if not r[3]])):
        if not sel:
            continue
        qs, act, mp = (np.array([r[k] for r in sel]) for k in (4, 5, 6))
        lq, la, lm = np.log(qs), np.log(act), np.log(mp)
        blend = 0.5 * (lq + lm)
        mal = lambda x: float(np.median(np.abs(x - la)))
        cov = float(np.mean([(r[7] <= r[5] <= r[8]) for r in sel]))
        snap[label] = {"n": len(sel), "symbols": len({r[1] for r in sel}), "dates": sorted({r[0] for r in sel}),
                       "median_abs_log_err": {"quote": mal(lq), "model": mal(lm), "blend": mal(blend)},
                       "median_quote_over_actual": float(np.median(qs / act)),
                       "median_model_over_actual": float(np.median(mp / act)),
                       "model_interval_coverage": cov, "median_actual_bps": float(np.median(act)),
                       "median_quote_bps": float(np.median(qs))}
        # names without bid/ask history get the cold model: same comparison with the cold model's forecast
        cm = np.array([r[9] for r in sel])
        okc = np.isfinite(cm)
        if okc.sum() > 10:
            lc = np.log(cm[okc])
            snap[label]["cold"] = {"n": int(okc.sum()), "median_abs_log_err": {
                "quote": float(np.median(np.abs(lq[okc] - la[okc]))), "model": float(np.median(np.abs(lc - la[okc]))),
                "blend": float(np.median(np.abs(0.5 * (lq[okc] + lc) - la[okc])))}}
            snap[label]["cold"]["_err"] = {"quote": la[okc] - lq[okc], "model": la[okc] - lc,
                                           "blend": la[okc] - 0.5 * (lq[okc] + lc)}
        snap[label]["_err"] = {"quote": la - lq, "model": la - lm, "blend": la - blend}
    for path, key in (("rule", None), ("rule_cold", "cold")):
        src = snap.get("rth", {}) if key is None else snap.get("rth", {}).get(key, {})
        e = src.get("median_abs_log_err")
        snap[path] = min(e, key=e.get) if e else "model"
        if e and snap[path] != "model":
            # empirical 10-90 interval of log(session average / estimate) for the adopted rule (used live)
            snap[path + "_interval"] = np.quantile(src["_err"][snap[path]], [0.1, 0.9]).tolist()
    for lab in ("rth", "premarket"):
        if lab in snap:
            snap[lab].pop("_err", None)
            if "cold" in snap[lab]:
                snap[lab]["cold"].pop("_err", None)
    # ---------------- paper fills
    fills = _jsonl(WEEK1 / "fills.jsonl")
    orders = {o["ref"]: o for o in _jsonl(WEEK1 / "orders.jsonl") if "ref" in o}
    agg = {}
    for f in fills:
        a = agg.setdefault(f["ref"], {"symbol": f["symbol"], "side": f["side"], "sleeve": f["sleeve"], "shares": 0, "value": 0.0,
                                      "time": f["time"], "comm": 0.0})
        a["shares"] += f["shares"]
        a["value"] += f["shares"] * f["price"]
        a["comm"] += f.get("commission") or 0
    look_ix = {}
    for date, hhmm, sleeve, rec in L:
        look_ix.setdefault((date, sleeve), []).append((hhmm, rec))
    out = []
    for ref, a in agg.items():
        sym, px = a["symbol"], a["value"] / a["shares"]
        sgn = 1 if a["side"] == "BOT" else -1
        t = datetime.strptime(a["time"][:17], "%Y%m%d %H:%M:%S")
        bs = t.replace(minute=0 if t.minute < 30 else 30, second=0)
        b = _bars30(sym).get(bs.strftime("%Y%m%d %H:%M"))
        arr = b["open"] if b and (t - bs) <= timedelta(minutes=3) else None
        o = orders.get(ref, {})
        # decision price: the latest look of that sleeve at or before the order's submit time (ET = UTC-4 in EDT)
        dec, dec_src = None, None
        if o.get("time"):
            ot = datetime.fromisoformat(o["time"]).replace(tzinfo=None) - timedelta(hours=4)
            cands = [(h, r) for h, r in look_ix.get((ot.strftime("%Y%m%d"), a["sleeve"]), []) if h <= ot.strftime("%H%M")]
            if cands:
                h, rec = max(cands, key=lambda x: x[0])
                for st in rec.get("trace") or []:
                    if st.get("tool") == "quote" and isinstance(st.get("result"), str):
                        qt = _quote_tables(st["result"]).get(sym)
                        if qt and qt[2] > qt[1] > 0:
                            dec, dec_src = (qt[1] + qt[2]) / 2, "quote mid"
                if dec is None:
                    dec = _table_last(rec.get("state_text", "")).get(sym)
                    dec_src = "table last" if dec else None
        j = d.k.get(sym)
        i = col.get(t.strftime("%Y%m%d"))
        half = float(np.exp(pred[j, i - 1]) / 2) if j is not None and i else None
        out.append({"ref": ref, "symbol": sym, "sleeve": a["sleeve"], "side": a["side"], "time": a["time"][:17],
                    "shares": a["shares"], "price": round(px, 4), "notional": round(a["value"], 0),
                    "commission_bps": round(a["comm"] / a["value"] * 1e4, 2),
                    "order_type": o.get("order_type"), "tif": o.get("tif"),
                    "slip_arrival_bps": None if arr is None else round(sgn * (px / arr - 1) * 1e4, 2),
                    "slip_decision_bps": None if not dec else round(sgn * (px / dec - 1) * 1e4, 2), "decision_src": dec_src,
                    "model_half_spread_bps": None if half is None or not np.isfinite(half) else round(half, 2),
                    "in_cache": j is not None})
    (ROOT / "runs" / "ml" / "research" / "cost_fills.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    def stats(x):
        x = np.array([v for v in x if v is not None], float)
        if len(x) == 0:
            return {"n": 0}
        return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
                "se": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
                "p10": float(np.quantile(x, 0.1)), "p90": float(np.quantile(x, 0.9))}
    open_ = [r for r in out if r["time"].endswith("09:30:00") or r["time"][9:14] == "09:30"]
    intra = [r for r in out if r not in open_]
    fe = {"fills": len(fills), "orders": len(out), "symbols": len({r["symbol"] for r in out}),
          "not_in_cache": sorted({r["symbol"] for r in out if not r["in_cache"]}),
          "arrival_all": stats([r["slip_arrival_bps"] for r in out]),
          "arrival_open": stats([r["slip_arrival_bps"] for r in open_]),
          "arrival_intraday": stats([r["slip_arrival_bps"] for r in intra]),
          "decision": stats([r["slip_decision_bps"] for r in out]),
          "model_half_spread_on_filled": stats([r["model_half_spread_bps"] for r in out if r["slip_arrival_bps"] is not None]),
          "commission_bps": stats([r["commission_bps"] for r in out]),
          "median_notional": float(np.median([r["notional"] for r in out]))}
    by = {}
    for r in out:
        b = by.setdefault(r["sleeve"], {"orders": 0, "notional": 0.0, "commission_usd": 0.0, "est_half_spread_usd": 0.0})
        b["orders"] += 1
        b["notional"] += r["notional"]
        b["commission_usd"] += r["commission_bps"] * r["notional"] / 1e4
        hs = r["model_half_spread_bps"] if r["model_half_spread_bps"] is not None else 1.0
        b["est_half_spread_usd"] += hs * r["notional"] / 1e4
    fe["by_sleeve"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by.items()}
    return {"snapshot": snap, "fills": fe}


def daily_bars(sym):
    """Daily bars outside the Data universe (data/bars/<SYM>.json, adjusted), as a list of dicts."""
    p = ROOT / "data" / "bars" / f"{sym.replace(' ', '_')}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    b = d.get("bars", d) if isinstance(d, dict) else d
    return b if isinstance(b, list) else [dict(v, date=k) for k, v in sorted(b.items())]


def thin_check(d, cold):
    """Out-of-distribution check of the cold (no bid/ask history) spread model on scanner names outside the universe:
    predicted spread from daily bars through the session before each quote snapshot vs the snapshot's quoted spread."""
    from ml_research.cost import feature_arrays, predict, interval, spread_bps, _series_from_bars
    snaps = snapshots(looks())
    rows = []
    for (date, sym, lt), (last, bid, ask, sleeve, hhmm) in sorted(snaps.items()):
        if sym in d.k or not (ask > bid > 0) or not ("09:30" <= lt <= "16:00"):
            continue
        bars_ = [b for b in daily_bars(sym) if str(b["date"])[:8] < date][-63:]
        if len(bars_) < 21:
            continue
        o, h, l, c, v, lr = _series_from_bars(bars_)
        F = feature_arrays(np.full(c.shape, np.nan), c.copy(), o, h, l, c, v, lr, [False])
        x = np.array([[F[k][0, -1] for k in cold["names"]]])
        p = float(predict(cold, x)[0])
        lo, hi = interval(cold["q"], p, F["ldv"][0, -1], cold["edges"])
        qs = spread_bps(bid, ask)
        rows.append({"date": date, "time": lt, "symbol": sym, "quote_spread_bps": round(qs, 1),
                     "cold_pred_bps": round(float(np.exp(p)), 1), "cold_10": round(float(np.exp(lo)), 1),
                     "cold_90": round(float(np.exp(hi)), 1), "adv_usd_m": round(float(np.exp(F["ldv"][0, -1])) / 1e6, 1),
                     "inside": bool(np.exp(lo) <= qs <= np.exp(hi))})
    if not rows:
        return {"n": 0}
    lr_ = np.array([np.log(r["quote_spread_bps"] / r["cold_pred_bps"]) for r in rows])
    return {"n": len(rows), "symbols": sorted({r["symbol"] for r in rows}), "median_log_ratio_quote_over_pred": float(np.median(lr_)),
            "median_abs_log_err": float(np.median(np.abs(lr_))), "inside_10_90": float(np.mean([r["inside"] for r in rows])),
            "rows": rows}
