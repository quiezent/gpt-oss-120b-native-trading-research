"""A point-in-time stand-in for the TWS Gateway, for rl2's training simulator: the research tools in tools.py
run on it unchanged, but every answer is computed from cached bars as of the decision time (before the open of
session d, so through the close of session d-1). It never sees session d or later.

Covered: quotes, bars (daily only), stats, scan (the daily-bar scans in tools.SCANS; options, lending and dividend
scans aren't available historically and return IBKR-style errors), and an optional ML forecast provider. News,
tape and articles aren't available for past dates, so the training prompt doesn't offer them.

Known limits (inherited from rl1's simulator): prices are today's split- and dividend-adjusted series, so a level
can differ from what traded then (a model that remembers old prices could infer a later split); and the universe is
today's index members, so 2025 scans miss names that later dropped out.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
UNAVAILABLE = {"HIGH_OPT_IMP_VOLAT", "HIGH_OPT_IMP_VOLAT_OVER_HIST", "HIGH_OPT_VOLUME_PUT_CALL_RATIO",
               "LOW_OPT_VOLUME_PUT_CALL_RATIO", "SCAN_utilization_DESC", "HIGH_DIVIDEND_YIELD_IB"}


def _industries():
    p = ROOT / "lists" / "industries.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ema(x, n):
    a = 2 / (n + 1)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        prev = np.where(np.isnan(out[:, t - 1]), x[:, t], out[:, t - 1])
        out[:, t] = np.where(np.isnan(x[:, t]), prev, a * x[:, t] + (1 - a) * prev)
    return out


class Precomputed:
    """Arrays shared by every SimGateway on one Data (EMAs over the whole history are causal)."""

    def __init__(self, data):
        self.ema20, self.ema50, self.ema200 = _ema(data.c, 20), _ema(data.c, 50), _ema(data.c, 200)
        self.industries = _industries()


class SimGateway:
    def __init__(self, data, d, pre, forecast=None):
        self.data, self.d, self.pre, self.t = data, d, pre, d - 1      # t: the last completed session
        self._forecast = forecast

    # ------------------------------------------------------------------ quotes and bars
    def _j(self, s):
        return self.data.k.get(str(s).upper().strip())

    def quotes(self, symbols, primary="", timeout=12.0):
        out, D, t = [], self.data, self.t
        for s in symbols:
            j = self._j(s)
            if j is None or np.isnan(D.c[j, t]):
                out.append({"symbol": str(s).upper(), "error": "no market data for this symbol in this environment"})
                continue
            last = float(D.c[j, t])
            out.append({"symbol": D.sym[j], "last": last, "bid": round(last * 0.9999, 2), "ask": round(last * 1.0001, 2),
                        "open": float(D.o[j, t]), "high": float(D.h[j, t]), "low": float(D.l[j, t]),
                        "close": float(D.c[j, t - 1]), "volume": float(D.v[j, t]), "last_time": None})
        return out

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, end="", primary="", **kw):
        j = self._j(symbol)
        if j is None:
            raise ValueError(f"200: No security definition has been found for {symbol}")
        n = _sessions(duration)
        D, t = self.data, self.t
        if "min" in bar_size or "hour" in bar_size:
            # the cached 30-minute bars are unadjusted while daily bars are adjusted, so past intraday bars aren't offered
            raise ValueError("only '1 day' bars are available for past dates in this environment")
        lo = max(0, t - n + 1)
        return [{"date": D.dates[i], "open": float(D.o[j, i]), "high": float(D.h[j, i]), "low": float(D.l[j, i]),
                 "close": float(D.c[j, i]), "volume": float(D.v[j, i])} for i in range(lo, t + 1) if not np.isnan(D.c[j, i])]

    # ------------------------------------------------------------------ stats
    def stats(self, symbols, primary="", seconds=3.0):
        single = isinstance(symbols, str)
        out, D, t = [], self.data, self.t
        for s in ([symbols] if single else symbols):
            j = self._j(s)
            if j is None:
                out.append({"symbol": str(s).upper(), "error": "200: No security definition has been found for the request"})
                continue
            lr = D.lr[j, max(0, t - 29):t + 1]
            row = {"symbol": D.sym[j], "hist_vol": float(np.nanstd(lr, ddof=1) * np.sqrt(252)), "implied_vol": None,
                   "avg_volume": float(np.nanmean(D.v[j, max(0, t - 89):t + 1])), "shortable": 3.0, "borrow": "available",
                   "shortable_shares": None}
            for w, n in (("13w", 65), ("26w", 130), ("52w", 252)):
                row[f"high_{w}"] = float(np.nanmax(D.h[j, max(0, t - n + 1):t + 1]))
                row[f"low_{w}"] = float(np.nanmin(D.l[j, max(0, t - n + 1):t + 1]))
            out.append(row)
        return out[0] if single else out

    # ------------------------------------------------------------------ scanner
    def scan(self, code, location="STK.US.MAJOR", rows=10, instrument="STK", **filters):
        if code in UNAVAILABLE:
            raise ValueError(f"10359: Scan code {code} is not available for past dates in this environment")
        D, t, P = self.data, self.t, self.pre
        c, o, h, l, v = D.c[:, t], D.o[:, t], D.h, D.l, D.v
        with np.errstate(all="ignore"):
            avgv = np.nanmean(v[:, max(0, t - 59):t + 1], axis=1)
            key = {
                "TOP_PERC_GAIN": c / D.c[:, t - 1] - 1, "TOP_PERC_LOSE": -(c / D.c[:, t - 1] - 1),
                "TOP_OPEN_PERC_GAIN": c / o - 1, "TOP_OPEN_PERC_LOSE": -(c / o - 1),
                "HIGH_OPEN_GAP": o / D.c[:, t - 1] - 1, "LOW_OPEN_GAP": -(o / D.c[:, t - 1] - 1),
                "MOST_ACTIVE": v[:, t], "MOST_ACTIVE_USD": v[:, t] * c, "HOT_BY_VOLUME": v[:, t] / avgv,
                "HIGH_VS_52W_HL": c / np.nanmax(h[:, max(0, t - 251):t + 1], axis=1),
                "LOW_VS_52W_HL": -(c / np.nanmin(l[:, max(0, t - 251):t + 1], axis=1)),
                "HIGH_VS_13W_HL": c / np.nanmax(h[:, max(0, t - 64):t + 1], axis=1),
                "LOW_VS_13W_HL": -(c / np.nanmin(l[:, max(0, t - 64):t + 1], axis=1)),
                "BULLISH_LAST_VS_EMA50": np.where(D.c[:, t - 5] < P.ema50[:, t - 5], c / P.ema50[:, t] - 1, np.nan),
                "BEARISH_LAST_VS_EMA50": np.where(D.c[:, t - 5] > P.ema50[:, t - 5], -(c / P.ema50[:, t] - 1), np.nan),
                "BULLISH_EMA_DIFF_VS_LAST": P.ema20[:, t] / P.ema200[:, t] - P.ema20[:, t - 5] / P.ema200[:, t - 5],
                "BEARISH_EMA_DIFF_VS_LAST": -(P.ema20[:, t] / P.ema200[:, t] - P.ema20[:, t - 5] / P.ema200[:, t - 5]),
                "SCAN_histPerfRatio1m_DESC": c / D.c[:, t - 21] - 1, "SCAN_histPerfRatio1m_ASC": -(c / D.c[:, t - 21] - 1),
                "SCAN_histPerfRatio3m_DESC": c / D.c[:, t - 63] - 1, "SCAN_histPerfRatio3m_ASC": -(c / D.c[:, t - 63] - 1),
            }.get(code)
        if key is None:
            raise ValueError(f"162: Scanner type with code {code} is disabled")
        ok = ~np.isnan(key) & ~np.isnan(c)
        if "priceAbove" in filters:
            ok &= c > float(filters["priceAbove"])
        if "avgVolumeAbove" in filters:
            ok &= avgv > float(filters["avgVolumeAbove"])
        if str(filters.get("stkTypes", "")).startswith("inc:"):
            ok &= np.array([s not in D.etf for s in D.sym])
        ind = filters.get("industryLike")
        if ind:
            ok &= np.array([(P.industries.get(s) or {}).get("industry") == ind for s in D.sym])
        idx = np.where(ok)[0]
        idx = idx[np.argsort(-key[idx])][:max(1, min(int(rows), 50))]
        return [{"rank": r + 1, "symbol": D.sym[j], "primary_exchange": "", "con_id": 0} for r, j in enumerate(idx)]

    # ------------------------------------------------------------------ ML forecasts (ML-arm episodes only)
    def forecast_provider(self):
        return self._forecast(self.d) if self._forecast else None


def _sessions(duration):
    try:
        n, unit = str(duration).split()
        n = int(n)
    except ValueError:
        return 252
    return max(1, n * {"D": 1, "W": 5, "M": 21, "Y": 252}.get(unit.upper()[0], 1))
