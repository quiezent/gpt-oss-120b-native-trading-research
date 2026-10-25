"""rl4's training simulator: the live episode shape on 30-minute bars.

Live (week 2), the sleeves' losses came from what happens between looks: a position built at one look and held, without a
protective order, through later looks and overnight (rl3-ml's MRNA), exits mishandled at later looks, positions misread at
submit time. rl3's simulator never showed any of it: one pre-open decision whose orders then ran untouched for 10 sessions
and were liquidated. Here a sleeve takes several full research looks in one session, as live:

  - The book comes from cheap burn-in decisions on daily bars (train2.burn_decision), so the first look inherits positions,
    open orders and notes.
  - Looks at LOOKS (ET). At each look the model sees trader.state_text with delayed-last marks during the session (the
    close of the last completed 30-minute bar), the goal line and its own notes; it researches with the live tools,
    answered point-in-time at the look time (IntraGateway), and submits once.
  - Orders are checked as live (trader.PaperWorld.act): an order that adds exposure is rejected at once if, filled at its
    limit or stop (else the last price), it would take gross exposure above 2x; a MOO after the open is rejected. They then
    work from the next 30-minute bar: MKT at its open, LMT/STP/TRAIL against its high and low, MOC/LOC at the close.
    Fills, expiries and rejections are reported at the next look in the live wording.
  - After the last look the session finishes on its bars (DAY orders expire at the close), and the book is closed at the
    next session's open: the overnight gap is realized; the days after are charged, not simulated (Ledger).

Ledger computes the trading reward from the sleeve's value before the first look, as reward.branch does: the costed book
return through the liquidation minus a beta-weighted SPY shadow book (each fill hedged at SPY's price at the same point of
the same bar), plus MU x beta exposure, minus an ex-ante risk charge built for week 2's errors (a 90%-of-NAV single name
held without a stop) under the analysis session's criterion 5:
  - variance is split into the market's (GAMMA = 4) and the idiosyncratic part (GAMMA_IDIO = 10: it earns no premium, and
    the charge is quadratic in weight, so it falls on concentration); each holding's idiosyncratic variance is the larger
    of its trailing 60-session residual variance and its forward one (the forecast's vol20, mostly implied vol);
  - intraday, at every 30-minute bar, on the session's (1 - GAP_SHARE) share of daily variance: a working protective order
    (a SELL stop or trail under a long, a BUY one over a short, at its current trigger level) truncates the covered share's
    downside over the rest of the session, scaled by stop_credit (full within one range1 of the price, none beyond two);
  - on the closing book: the overnight GAP_SHARE plus FORWARD more days, with no stop credit (week 2's gaps went through
    stops), and 1 + FORWARD days of premium. rl3's reward simulated 10 more sessions with no decisions; their P&L was noise
    no look could change (criteria v1.1, R5), so here only their expected terms count;
  - KAPPA_CARRY on each name's share above CARRY_FREE carried past the close unprotected (stop_credit-scaled).
The trainer adds the penalties: MALFORMED for a look with no valid submit, a reject penalty per rejected order, and BAD_TURN
per malformed turn that live would have re-asked.

Data: data/bars30 (ml_research/intraday.load_panel) covers 17 Oct 2025 - 16 Oct 2026, scaled each day to the adjusted daily
close (so marks, the table and the quote agree). Known limits: one 30-minute bar is the finest grain (the order of high and
low inside a bar is unknown: ties resolve against the trader, as sim.py); a look's marks are the bar close at the look time,
where live quotes lag about 15 minutes and a look takes several minutes; the universe is the 523 cached names, so an order
for any other symbol is rejected ("no market data"), which live would accept. Such a rejection is not the model's error and
is not charged (IntraWorld.act does not count it in `rejected`), as reward.branch does not count it either. The reward
realizes no overnight or multi-day outcome (the closing book is marked at the close and charged for what follows),
while live carries positions until Monday's flatten: a known mismatch, priced only through the charges.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import reward
import sim
import simgw
import trader
from ml_research import simnews

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
NS = 13                                          # 30-minute bars per full session, starting 09:30 .. 15:30
LOOKS = ["09:00", "10:30", "13:00", "15:30"]
FORWARD = 4                                      # at most this many days of risk and premium charged on the closing book
                                                 # (no simulated P&L); train4 uses the sessions left before Monday's flatten
GAP_SHARE = 0.3                                  # share of a holding's daily variance a stop cannot cut (overnight gaps)
BAD_TURN = 0.1
MAX_GROSS = 2.0


def minutes(hhmm):
    h, m = map(int, hhmm.split(":"))
    return 60 * h + m


def bars_done(hhmm):
    """30-minute bars completed by a look at hhmm (0 before 10:00)."""
    return max(0, min(NS, (minutes(hhmm) - 570) // 30))


# ---------------------------------------------------------------- data
class Intra:
    """30-minute bars on trader.Data's symbols and sessions, scaled each day to the adjusted daily close."""

    def __init__(self, data):
        from ml_research import intraday
        p = intraday.load_panel()
        self.data = data
        syms = [str(s) for s in p["symbols"]]
        dates = [str(x) for x in p["dates"]]
        self.day = {data.ix[x]: i for i, x in enumerate(dates) if x in data.ix}        # data session index -> panel day
        rows = {s: i for i, s in enumerate(syms)}
        idx = np.array([rows.get(s, -1) for s in data.sym])
        S, D = len(data.sym), len(dates)
        A = {}
        for k in ("o", "h", "l", "c", "v"):
            a = np.full((S, D, NS), np.nan)
            ok = idx >= 0
            a[ok] = p[k][idx[ok]]
            A[k] = a
        f = np.full((S, D), np.nan)                     # adjusted daily close / raw last 30-minute close
        for t, i in self.day.items():
            f[:, i] = data.c[:, t] / A["c"][:, i, NS - 1]
        f = np.where(np.isfinite(f) & (f > 0.5) & (f < 2.0), f, np.nan)   # a split inside the window: no intraday bars
        for k in ("o", "h", "l", "c"):
            A[k] = A[k] * f[:, :, None]
        A["v"] = A["v"] / f[:, :, None]
        self.o, self.h, self.l, self.c, self.v = A["o"], A["h"], A["l"], A["c"], A["v"]
        self._rng = None

    def has(self, d):
        return d in self.day

    def bar(self, j, d, k):
        i = self.day.get(d)
        if i is None or not np.isfinite(self.c[j, i, k]):
            return None
        return {"open": self.o[j, i, k], "high": self.h[j, i, k], "low": self.l[j, i, k], "close": self.c[j, i, k],
                "volume": self.v[j, i, k]}

    def last(self, j, d, nb):
        """The mark after nb completed bars of session d: the close of bar nb-1, else None."""
        i = self.day.get(d)
        if i is None or nb <= 0:
            return None
        x = self.c[j, i, nb - 1]
        return float(x) if np.isfinite(x) else None

    def so_far(self, j, d, nb):
        """(open, high, low, volume) of session d over its first nb bars."""
        i = self.day.get(d)
        if i is None or nb <= 0:
            return None
        o, h, l, v = self.o[j, i, 0], np.nanmax(self.h[j, i, :nb]), np.nanmin(self.l[j, i, :nb]), np.nansum(self.v[j, i, :nb])
        return None if not np.isfinite(o) else (float(o), float(h), float(l), float(v))

    def ranges(self):
        """OOS session-range forecasts (runs/ml/research/intraday_oos.npz), loaded once: {symbol: row}, day index, smearing."""
        if self._rng is None:
            z = np.load(ROOT / "runs" / "ml" / "research" / "intraday_oos.npz", allow_pickle=True)
            from ml_research import intraday
            m = intraday._load_model()["range"]
            self._rng = {"rows": {str(s): j for j, s in enumerate(z["symbols"])},
                         "day": {self.data.ix[str(x)]: i for i, x in enumerate(z["dates"]) if str(x) in self.data.ix},
                         "a": {k: z[k] for k in z.files if k.startswith("range_")},
                         "smear": {tk: [m[tk].m[k] for k in range(NS)] for tk in ("rng60", "rngrod")}}
        return self._rng


# ---------------------------------------------------------------- the market
class IntraMarket(sim.Market):
    """sim.Market with a clock inside the session: slot = bars of session `session` already traded (0..NS)."""

    intra = None
    slot = 0

    @classmethod
    def from_market(cls, m, intra):
        x = cls.__new__(cls)
        x.__dict__.update(m.clone().__dict__)
        x.intra, x.slot = intra, 0
        x.max_gross = float("inf")      # rl5: margin is checked at order time only (IntraWorld.act), as IBKR does live;
        return x                        # sim.Market's fill-time rejection never happens on the paper account

    def clone(self):
        x = super().clone()
        x.intra, x.slot = self.intra, self.slot
        return x

    def _j(self, s):
        return self.intra.data.k.get(s)

    def last(self, symbol):
        if self.slot > 0:
            j = self._j(symbol)
            if j is not None:
                for nb in range(self.slot, 0, -1):          # the last completed bar that traded
                    x = self.intra.last(j, self.session, nb)
                    if x is not None:
                        return x
        return super().last(symbol)

    def run_bars(self, to_slot):
        """Trade the session's bars from self.slot up to (not including) to_slot."""
        d = self.session
        for k in range(self.slot, min(to_slot, NS)):
            self.slot = k                              # marks (margin checks) at the bar's start: the last completed bar
            for o in self.open_orders():
                t = o.order_type
                if t == "MOO" and k > 0:               # an opening-auction order that missed the auction
                    o.status, o.message = "Cancelled", "MOO order not filled at the open"
                    continue
                if t in ("MOC", "LOC") and k < NS - 1:
                    continue
                j = self._j(o.symbol)
                b = self.intra.bar(j, d, k) if j is not None else None
                if b is None:
                    continue
                n0 = len(self.fills)
                if t in ("MOC", "LOC"):                    # at the close: _try_fill reads the bar's close
                    self._try_fill(o, b, d)
                else:
                    self._try_fill(o, dict(b), d)
                for f in self.fills[n0:]:
                    f.slot = k
                if o.order_type == "TRAIL" and o.status not in sim.FINISHED:
                    o.trail_ref = max(o.trail_ref, b["close"]) if o.action == "SELL" else min(o.trail_ref, b["close"])
            self.slot = k + 1
        self.slot = max(self.slot, min(to_slot, NS))

    def end_session(self):
        """Trade the rest of the session's bars, expire DAY orders, charge a day of financing, move to the next session."""
        self.run_bars(NS)
        for o in self.open_orders():
            if o.tif == "DAY":
                o.status, o.message = "Expired", "DAY order not filled"
        self.session += 1
        self.slot = 0
        fee = max(0.0, -self.cash) * self.margin_rate / 252
        fee += sum(-q * (self.last(s) or 0.0) for s, (q, _) in self.positions.items() if q < 0) * self.borrow_rate / 252
        self.cash -= fee
        self.financing += fee
        self.realized -= fee


# ---------------------------------------------------------------- the sleeve
def live_wording(event):
    """A burn-in (trader.SimWorld) event in the live wording (paper.events_since_last_look): 'filled BOT 40 AAPL @ 258.22
    (commission $1.00, ref o598-1)' -> 'filled BUY 40 AAPL @ 258.22 (commission $1.00) [o598-1]'."""
    import re
    m = re.match(r"filled (BOT|SLD|BUY|SELL) (.+?) \(commission (\$[\d.,]+), ref (\S+)\)$", event)
    if m:
        return f"filled {'BUY' if m[1] in ('BOT', 'BUY') else 'SELL'} {m[2]} (commission {m[3]}) [{m[4]}]"
    m = re.match(r"(expired|rejected): (\S+) (BUY|SELL) (\S+) (\S+) (\S+(?: LMT)?)(?: \((.*)\))?$", event)
    if m:
        return f"{m[2]} {m[3]} {m[4]} {m[5]} {m[6]} ended {'Cancelled' if m[1] == 'expired' else 'Inactive'}: {m[7] or m[1]}"
    return event


class IntraWorld:
    """A sleeve on IntraMarket, presented and handled as trader.PaperWorld presents and handles the live sleeve."""

    MAX_GROSS = MAX_GROSS

    def __init__(self, sim_world, intra, prefix="tt-rl4"):
        self.data = sim_world.data
        self.m = IntraMarket.from_market(sim_world.m, intra)
        self.opened = dict(sim_world.opened)
        self.events = [live_wording(e) for e in sim_world.events]
        self._seen_fills = len(self.m.fills)
        self._n = 0
        self.prefix = prefix
        self.rejected = 0                         # orders rejected since the last look (for the reward)

    def clone(self):
        import copy
        w = copy.copy(self)
        w.m = self.m.clone()
        w.opened = dict(self.opened)
        w.events = list(self.events)
        return w

    def symbols_in_play(self):
        return sorted(set(self.m.positions) | {o.symbol for o in self.m.open_orders()})

    def marks(self):
        """{symbol: mark} for the market table's last column during the session (every universe name with a bar)."""
        if self.m.slot <= 0:
            return None
        out = {}
        for s, j in self.data.k.items():
            x = self.m.intra.last(j, self.m.session, self.m.slot)
            if x is not None:
                out[s] = x
        return out

    def state(self, d, feats, live=None):
        m = self.m
        value = m.nav()
        pos, gross, net, bnet = [], 0.0, 0.0, 0.0
        for s, (q, avg) in sorted(m.positions.items()):
            px = m.last(s)
            v = q * px
            gross += abs(v)
            net += v
            b = feats.get(s, {}).get("beta")
            bnet += v * (b if b is not None and not math.isnan(b) else 1.0)
            pos.append({"symbol": s, "quantity": int(q), "avg_cost": avg, "close": px, "unrealized": q * (px - avg) + 0.0,
                        "days": d - self.opened.get(s, d)})
        orders = [{"ref": o.ref, "symbol": o.symbol, "action": o.action, "quantity": int(o.quantity),
                   "type": o.order_type, "limit": o.limit, "stop": o.aux if o.order_type in ("STP", "STP LMT") else None,
                   "trail_percent": o.trailing_percent, "tif": o.tif} for o in m.open_orders()]
        ev, self.events = self.events, []
        return {"value": value, "cash": m.cash, "positions": pos, "orders": orders, "events": ev,
                "gross": gross / value if value > 0 else 0, "net": net / value if value > 0 else 0,
                "beta_net": bnet / value if value > 0 else 0}

    def act(self, call):
        """Cancels, then orders in input order, checked as PaperWorld.act checks them; returns the placement notes."""
        m, notes = self.m, []
        for ref in call.get("cancel", []):
            if m.cancel(str(ref)) is None:
                notes.append(f"cancel {ref}: no open order with this ref")
        pos = {s: q for s, (q, _) in m.positions.items()}
        mark = {s: m.last(s) for s in pos}
        nav = m.nav()
        session_open = m.slot > 0
        for raw in call.get("place", []):
            o, err = trader.clean_order(raw)
            if err:
                notes.append(f"order rejected: {err}")
                self.rejected += 1
                continue
            s = o["symbol"]
            if session_open and o["type"] == "MOO":
                notes.append(f"{o['action']} {o['quantity']} {s} MOO: Cancelled MOO orders are accepted only before the open")
                self.rejected += 1
                continue
            q0 = pos.get(s, 0)
            q1 = q0 + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
            p = o["limit"] or o["stop"] or m.last(s) or mark.get(s)
            if p and abs(q1) > abs(q0):
                gross = sum(abs(q) * mark.get(k, p) for k, q in pos.items() if k != s) + abs(q1) * p
                if gross > self.MAX_GROSS * max(nav, 0.0):
                    notes.append(f"{o['action']} {o['quantity']} {s} {o['type']} rejected: insufficient margin (gross exposure "
                                 f"would be {gross / nav if nav > 0 else float('inf'):.2f}x the sleeve, above the "
                                 f"{self.MAX_GROSS:g}x limit)")
                    self.rejected += 1
                    continue
            self._n += 1
            aux = o["stop"] if o["type"] in ("STP", "STP LMT") else None
            od = m.place(s, o["action"], o["quantity"], o["type"], o["limit"], aux, o["tif"],
                         trailing_percent=o["trail_percent"], ref=f"{self.prefix}-{m.session:04d}{self._n:06x}")
            if od.status == "Rejected":
                notes.append(f"{o['action']} {o['quantity']} {s} {o['type']}: Cancelled {od.message}")
                if not od.message.startswith("no market data"):
                    self.rejected += 1
                continue
            pos[s] = q1
            if p:
                mark.setdefault(s, p)
        self.events.extend(notes)
        return notes

    def _record(self, before):
        """Fills and ended orders since `before` ({ref: status}) as live events (paper.events_since_last_look)."""
        for f in self.m.fills[self._seen_fills:]:
            self.events.append(f"filled {'BUY' if f.side == 'BOT' else 'SELL'} {f.shares:g} {f.symbol} @ {f.price:.2f} "
                               f"(commission ${f.commission:.2f}) [{f.ref}]")
            q = self.m.positions.get(f.symbol, [0])[0]
            if q and f.symbol not in self.opened:
                self.opened[f.symbol] = f.session
            if not q:
                self.opened.pop(f.symbol, None)
        self._seen_fills = len(self.m.fills)
        for o in self.m.orders:
            was = before.get(o.ref)
            if was not in (None, "Expired", "Rejected", "Filled", "Cancelled") and o.status in ("Expired", "Rejected", "Cancelled"):
                why = o.message or o.status
                self.events.append(f"{o.ref} {o.action} {o.quantity:g} {o.symbol} {o.order_type} ended "
                                   f"{'Cancelled' if o.status != 'Rejected' else 'Inactive'}: {why}")
                if o.status == "Rejected":
                    self.rejected += 1


# ---------------------------------------------------------------- research tools at the look time
class IntraGateway(simnews.NewsSimGateway):
    """The research tools, answered as of hhmm ET on session d: before 09:30 as simgw (bars through the previous close);
    during the session with today's completed 30-minute bars."""

    def __init__(self, data, d, pre, intra, hhmm, forecast=None, news_archive=None):
        super().__init__(data, d, pre, forecast=forecast, news_archive=news_archive)
        self.intra, self.hhmm, self.nb = intra, hhmm, bars_done(hhmm)
        if minutes(hhmm) >= 570 and self.nb == 0:
            self.nb = 0                              # 09:30 exactly: the open has not printed a bar yet

    @property
    def cutoff(self):
        s = self.data.dates[self.d]
        h, m = map(int, self.hhmm.split(":"))
        return int(datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), h, m, tzinfo=ET).timestamp())

    def _live(self, j):
        """(last, open, high, low, volume so far) during the session, else None."""
        if self.nb <= 0:
            return None
        last = self.intra.last(j, self.d, self.nb)
        sf = self.intra.so_far(j, self.d, self.nb)
        return None if last is None or sf is None else (last,) + sf

    def quotes(self, symbols, primary="", timeout=12.0):
        if self.nb <= 0:
            return super().quotes(symbols, primary, timeout)
        out, D, t = [], self.data, self.d - 1
        h, mm = map(int, self.hhmm.split(":"))
        for s in symbols:
            j = self._j(s)
            x = self._live(j) if j is not None else None
            if x is None:
                out.append({"symbol": str(s).upper(), "error": "no market data for this symbol in this environment"})
                continue
            last, o, hi, lo, v = x
            out.append({"symbol": D.sym[j], "last": last, "bid": round(last * 0.9999, 2), "ask": round(last * 1.0001, 2),
                        "open": o, "high": hi, "low": lo, "close": float(D.c[j, t]), "volume": v,
                        "last_time": f"{D.iso(self.d)}T{h:02d}:{mm:02d}:00-04:00"})
        return out

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, end="", primary="", **kw):
        j = self._j(symbol)
        if j is None:
            raise ValueError(f"200: No security definition has been found for {symbol}")
        bs = str(bar_size).lower()
        if "day" in bs:
            rows = super().bars(symbol, duration, bar_size, what, rth, end, primary, **kw)
            x = self._live(j)
            if x is not None:                          # today's partial bar, as IBKR serves it during the session
                rows.append({"date": self.data.dates[self.d], "open": x[1], "high": x[2], "low": x[3], "close": x[0], "volume": x[4]})
            return rows
        if bs.replace(" ", "") not in ("30mins", "1hour", "1hours"):
            raise ValueError("only '30 mins', '1 hour' and '1 day' bars are available for past dates in this environment")
        n = simgw._sessions(duration)
        out = []
        days = [t for t in range(self.d - n + (1 if self.nb > 0 else 0), self.d + 1) if self.intra.has(t)]
        for t in days:
            last_k = self.nb if t == self.d else NS
            sl = []
            for k in range(last_k):
                b = self.intra.bar(j, t, k)
                if b:
                    m0 = 570 + 30 * k
                    sl.append({"date": f"{self.data.dates[t]} {m0 // 60:02d}:{m0 % 60:02d}:00 US/Eastern",
                               **{x: float(b[x]) for x in b}})
            if "hour" in bs:                           # pairs of 30-minute bars, from 09:30
                sl = [{"date": a["date"], "open": a["open"], "high": max(a["high"], b["high"]) if b else a["high"],
                       "low": min(a["low"], b["low"]) if b else a["low"], "close": (b or a)["close"],
                       "volume": a["volume"] + (b["volume"] if b else 0)}
                      for a, b in ((sl[i], sl[i + 1] if i + 1 < len(sl) else None) for i in range(0, len(sl), 2))]
            out += sl
        return out

    def scan(self, code, location="STK.US.MAJOR", rows=10, instrument="STK", **filters):
        if self.nb <= 0:
            return super().scan(code, location, rows, instrument, **filters)
        if code in simgw.UNAVAILABLE:
            raise ValueError(f"10359: Scan code {code} is not available for past dates in this environment")
        D, t, P = self.data, self.d - 1, self.pre
        S = len(D.sym)
        last, o, vol = np.full(S, np.nan), np.full(S, np.nan), np.full(S, np.nan)
        for j in range(S):
            x = self._live(j)
            if x is not None:
                last[j], o[j], vol[j] = x[0], x[1], x[4]
        prev = D.c[:, t]
        with np.errstate(all="ignore"):
            avgv = np.nanmean(D.v[:, max(0, t - 59):t + 1], axis=1)
            frac = self.nb / NS
            key = {
                "TOP_PERC_GAIN": last / prev - 1, "TOP_PERC_LOSE": -(last / prev - 1),
                "TOP_OPEN_PERC_GAIN": last / o - 1, "TOP_OPEN_PERC_LOSE": -(last / o - 1),
                "HIGH_OPEN_GAP": o / prev - 1, "LOW_OPEN_GAP": -(o / prev - 1),
                "MOST_ACTIVE": vol, "MOST_ACTIVE_USD": vol * last, "HOT_BY_VOLUME": vol / (avgv * frac),
                "HIGH_VS_52W_HL": last / np.fmax(np.nanmax(D.h[:, max(0, t - 251):t + 1], axis=1), last),
                "LOW_VS_52W_HL": -(last / np.fmin(np.nanmin(D.l[:, max(0, t - 251):t + 1], axis=1), last)),
                "HIGH_VS_13W_HL": last / np.fmax(np.nanmax(D.h[:, max(0, t - 64):t + 1], axis=1), last),
                "LOW_VS_13W_HL": -(last / np.fmin(np.nanmin(D.l[:, max(0, t - 64):t + 1], axis=1), last)),
                "BULLISH_LAST_VS_EMA50": np.where(D.c[:, t - 5] < P.ema50[:, t - 5], last / P.ema50[:, t] - 1, np.nan),
                "BEARISH_LAST_VS_EMA50": np.where(D.c[:, t - 5] > P.ema50[:, t - 5], -(last / P.ema50[:, t] - 1), np.nan),
                "BULLISH_EMA_DIFF_VS_LAST": P.ema20[:, t] / P.ema200[:, t] - P.ema20[:, t - 5] / P.ema200[:, t - 5],
                "BEARISH_EMA_DIFF_VS_LAST": -(P.ema20[:, t] / P.ema200[:, t] - P.ema20[:, t - 5] / P.ema200[:, t - 5]),
                "SCAN_histPerfRatio1m_DESC": last / D.c[:, t - 20] - 1, "SCAN_histPerfRatio1m_ASC": -(last / D.c[:, t - 20] - 1),
                "SCAN_histPerfRatio3m_DESC": last / D.c[:, t - 62] - 1, "SCAN_histPerfRatio3m_ASC": -(last / D.c[:, t - 62] - 1),
            }.get(code)
        if key is None:
            raise ValueError(f"162: Scanner type with code {code} is disabled")
        ok = ~np.isnan(key) & ~np.isnan(last)
        if "priceAbove" in filters:
            ok &= last > float(filters["priceAbove"])
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

    def forecast_provider(self):
        return self._forecast(self.d, self.hhmm) if self._forecast else None

    def session_ranges(self, syms):
        """forecast.run's session columns at this look, from the walk-forward OOS forecasts: (header note,
        {symbol: (next60, lo, hi, rest_of_day, lo, hi)} in % of price, full high-low span)."""
        if self.nb <= 0 and minutes(self.hhmm) < 570:
            return "session ranges from 09:30 ET", {}
        R = self.intra.ranges()
        i = R["day"].get(self.d)
        if i is None:
            return "session ranges unavailable (no intraday data)", {}
        k = min(self.nb, NS - 1)
        col = i * NS + k
        out = {}
        for s in syms:
            j = R["rows"].get(s)
            if j is None:
                out[s] = "no session range"
                continue
            vals = []
            for tk in ("rng60", "rngrod"):
                lp, lo, hi = (float(R["a"][f"range_{tk}_{x}"][j, col]) for x in ("log", "lo", "hi"))
                if not all(map(np.isfinite, (lp, lo, hi))):
                    vals = None
                    break
                vals += [round(100 * math.exp(lp) * R["smear"][tk][k], 2), round(100 * math.exp(lo), 2), round(100 * math.exp(hi), 2)]
            out[s] = tuple(vals) if vals else "no session range (ETF or too little history)"
        return f"session ranges at the {self.hhmm} look", out


# ---------------------------------------------------------------- reward
GAMMA_IDIO = 10.0       # idiosyncratic variance earns no premium: charged harder than the market's (reward.GAMMA = 4)
KAPPA_CARRY = 1.0       # R per 1.0x NAV of a name above CARRY_FREE carried past the close unprotected (stop_credit-scaled)
CARRY_FREE = 0.25       # the share of NAV in one name that carries no carry charge


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def truncated_share(x):
    """E[min(max(Z, -x), 0)^2] / E[min(Z, 0)^2] for a standard normal Z: the share of the downside second moment left
    when a stop x standard deviations away truncates the path (0 at x = 0, 1 as x grows)."""
    if x <= 0:
        return 0.0
    return min(1.0, 2 * ((0.5 - _Phi(-x)) - x * _phi(x) + x * x * _Phi(-x)))


def stop_credit(dist, range1):
    """Criterion 5 (24 Oct): full credit for a stop within one range1 (the forecast's next-session high-low span, a
    fraction of price) of the price, none beyond two, linear between."""
    if dist is None or not range1 or range1 <= 0:
        return 0.0
    return max(0.0, min(1.0, 2.0 - dist / range1))


def stop_level(o):
    """The price at which a resting STP / STP LMT / TRAIL order triggers now (a trail from its current reference)."""
    if o.order_type == "TRAIL":
        amt = o.aux if o.aux else (o.trail_ref or 0) * (o.trailing_percent or 0) / 100
        return (o.trail_ref - amt) if o.action == "SELL" else (o.trail_ref + amt)
    return o.aux


def protective(m, s, q, px):
    """(covered share, stop distance as a fraction of px) of the resting protective orders on a holding of q shares: SELL
    STP/STP LMT/TRAIL under a long, BUY over a short, at their current trigger level. The nearest stop among those covering
    the most shares. Only held shares count, so a stop placed ahead of an entry that has not filled earns nothing."""
    best_cov, best_dist = 0.0, None
    for o in m.open_orders():
        if o.symbol != s or o.order_type not in ("STP", "STP LMT", "TRAIL") or (q > 0) != (o.action == "SELL"):
            continue
        lvl = stop_level(o)
        if not lvl:
            continue
        dist = max(0.0, (px - lvl) / px if q > 0 else (lvl - px) / px)
        cov = min(1.0, o.quantity / abs(q))
        if cov > best_cov or (cov == best_cov and best_dist is not None and dist < best_dist):
            best_cov, best_dist = cov, dist
    return best_cov, best_dist


class Risk:
    """Ex-ante risk of a book for the episode's reward: beta exposure and the market and idiosyncratic variance parts."""

    def __init__(self, data, d, feats, sigma_m, fc_rows):
        self.data, self.d, self.feats, self.sigma_m = data, d, feats, sigma_m
        self.fc = fc_rows or {}                  # {symbol: forecast row}: vol20_pct (forward vol) and range1_pct
        self._corr = {}

    def corr(self, syms):
        key = tuple(syms)
        if key not in self._corr:
            C = self.data.resid_cov(self.d, list(syms))
            dg = np.sqrt(np.clip(np.diag(C), 1e-12, None))
            R = np.clip(C / np.outer(dg, dg), -1, 1)
            np.fill_diagonal(R, 1.0)
            self._corr[key] = R
        return self._corr[key]

    def idio_var(self, s):
        """Daily idiosyncratic variance: the larger of the trailing 60-session residual variance and the forward one
        (forecast vol20, mostly implied vol, less its beta share of SPY's variance)."""
        b, r = reward._risk_inputs(self.feats, s)
        var = r ** 2
        f = (self.fc.get(s) or {}).get("vol20_pct")
        if f is not None and np.isfinite(f) and f > 0:
            var = max(var, (f / 100) ** 2 / 252 - (b * self.sigma_m) ** 2)
        return var

    def parts(self, m, rest=None):
        """(e, market variance, idiosyncratic variance) of m's book at its marks, daily units. rest: the share of the
        session still to trade (intraday charge): a protective stop then truncates the covered share's downside over it,
        scaled by stop_credit; None (overnight and tail closes): no credit."""
        nav = m.nav()
        nav = nav if nav > 0 else 1e-9
        syms = sorted(m.positions)
        if not syms:
            return 0.0, 0.0, 0.0
        w, sd = np.zeros(len(syms)), np.zeros(len(syms))
        e = 0.0
        for i, s in enumerate(syms):
            q = m.positions[s][0]
            px = m.last(s) or 0.0
            b, _ = reward._risk_inputs(self.feats, s)
            w[i] = q * px / nav
            e += b * w[i]
            var = self.idio_var(s)
            if rest is not None and px and var > 0:
                cov, dist = protective(m, s, q, px)
                if cov > 0 and dist is not None:
                    r1 = (self.fc.get(s) or {}).get("range1_pct")
                    r1 = r1 / 100 if r1 else 2.5 * math.sqrt(var + (b * self.sigma_m) ** 2)
                    c = stop_credit(dist, r1)
                    x = dist / math.sqrt(max(1e-12, (var + (b * self.sigma_m) ** 2) * (1 - GAP_SHARE) * rest))
                    var *= 1 - cov * c * (1 - truncated_share(x))
            sd[i] = math.sqrt(max(var, 0.0))
        S = np.outer(sd, sd) * self.corr(syms)
        return e, (e * self.sigma_m) ** 2, float(w @ S @ w)

    def charge(self, m, share, rest=None):
        """GAMMA/2 x market variance + GAMMA_IDIO/2 x idiosyncratic variance, over `share` of a day. Returns (charge, e)."""
        e, vm, vi = self.parts(m, rest)
        return share * (0.5 * reward.GAMMA * vm + 0.5 * GAMMA_IDIO * vi), e


class Shadow:
    """reward.branch's SPY shadow book, for intraday fills too (SPY's price at the same point of the same 30-minute bar)."""

    def __init__(self, m, data, intra, feats):
        self.data, self.intra, self.feats = data, intra, feats
        self.spy, self.h, self.hc, self.seen = data.k["SPY"], 0.0, 0.0, len(m.fills)
        c0 = data.c[self.spy, m.session - 1]
        for s, (q, _) in m.positions.items():
            b, _ = reward._risk_inputs(feats, s)
            x = b * q * m.last(s) / c0
            self.h += x
            self.hc -= x * c0

    def _px(self, f):
        D, spy, t = self.data, self.spy, f.session
        k = getattr(f, "slot", None)
        b = self.intra.bar(spy, t, k) if k is not None else None
        if b:
            if f.phase == "close":
                return b["close"]
            if f.phase == "open":
                return b["open"]
            jb = self.intra.bar(D.k[f.symbol], t, k) if f.symbol in D.k else None
            lo, hi = (jb["low"], jb["high"]) if jb else (f.raw, f.raw)
            frac = 0.5 if not hi > lo else min(1.0, max(0.0, (f.raw - lo) / (hi - lo)))
            return b["low"] + frac * (b["high"] - b["low"])
        if f.phase == "close":
            return D.c[spy, t]
        if f.phase == "intraday":
            j = D.k.get(f.symbol)
            lo, hi = (D.l[j, t], D.h[j, t]) if j is not None else (f.raw, f.raw)
            frac = 0.5 if not hi > lo else min(1.0, max(0.0, (f.raw - lo) / (hi - lo)))
            return D.l[spy, t] + frac * (D.h[spy, t] - D.l[spy, t])
        return D.o[spy, t]

    def update(self, m):
        for f in m.fills[self.seen:]:
            px = self._px(f)
            b, _ = reward._risk_inputs(self.feats, f.symbol)
            x = b * (f.shares if f.side == "BOT" else -f.shares) * f.price / px
            self.h += x
            self.hc -= x * px
        self.seen = len(m.fills)


class Ledger:
    """The episode's trading reward, accumulated as the session is traded. Build it before the first look (the sleeve at
    the previous close); call trade_to(world, hhmm) to move to each look and mark(world) for the cumulative reward at it
    (train4's per-look reward-to-go); then close(world) after the last look.

    The episode's P&L is realized and marked at the session's close, less the cost of closing the book (commission and
    slippage, as a liquidation at market would pay); everything after the close is expected, not simulated: the closing
    book is charged its overnight share plus `forward` days of risk and credited 1 + `forward` days of premium. The 24 Oct
    smoke showed why: with the overnight gap realized, it carried 97% of the per-look reward-to-go variance (R5's noise
    rule aborts above 70%), and no look can change a gap, only what is carried into it, which the charges price."""

    def __init__(self, world, data, intra, d, feats, sigma_m, fc_rows, forward=FORWARD):
        self.data, self.intra, self.d, self.forward = data, intra, d, forward
        self.v0 = world.m.nav()
        self.shadow = Shadow(world.m, data, intra, feats)
        self.risk = Risk(data, d, feats, sigma_m, fc_rows)
        self.fc_rows = fc_rows
        self.premium = self.charge_intra = self.charge_close = 0.0

    def trade_to(self, world, hhmm):
        """Trade the session up to a look at hhmm (or "16:00"), charging each bar's intraday risk on the book at its start
        (marks: the last completed bar)."""
        m = world.m
        before = {o.ref: o.status for o in m.orders}
        for k in range(m.slot, bars_done(hhmm)):
            c, _ = self.risk.charge(m, (1 - GAP_SHARE) / NS, rest=(NS - k) / NS)
            self.charge_intra += c
            m.run_bars(k + 1)
        world._record(before)
        self.shadow.update(m)

    def _spy(self, m):
        spy = self.data.k["SPY"]
        x = self.intra.last(spy, m.session, m.slot) if m.slot else None
        return x if x is not None else self.data.c[spy, m.session - 1]

    def mark(self, world):
        """The trading reward so far (percentage points of the start value), at the book's current marks."""
        m = world.m
        hedge = self.shadow.hc + self.shadow.h * self._spy(m)
        return 100 * ((m.nav() - self.v0) / self.v0 - hedge / self.v0 + self.premium - self.charge_intra - self.charge_close)

    def close(self, world):
        """The rest of the session; at its close: the book marked at the close less the cost of closing it, the overnight
        share of a day's risk plus `forward` days (no stop credit), and 1 + `forward` days of premium. Returns the parts."""
        m = world.m
        self.trade_to(world, "16:00")
        before = {o.ref: o.status for o in m.orders}
        m.end_session()                                 # DAY orders expire; marks are the session's close
        world._record(before)
        self.shadow.update(m)
        p_close = self.mark(world)
        at_close, nav_close, carry = book_summary(m), m.nav(), self.carry(m)
        exit_cost = sum(sim.commission(abs(q), m.last(s)) + abs(q) * m.last(s) * m.slippage_bps / 1e4
                        for s, (q, _) in m.positions.items() if m.last(s))
        c, e = self.risk.charge(m, GAP_SHARE + self.forward)
        self.charge_close += c
        self.premium += reward.MU * e * (1 + self.forward)
        det = 100 * (reward.MU * e * (1 + self.forward) - c - exit_cost / self.v0)   # everything after the close
        spy = self.data.k["SPY"]
        hedge = self.shadow.hc + self.shadow.h * self.data.c[spy, m.session - 1]
        r_book = (nav_close - exit_cost) / self.v0 - 1
        r_trade = p_close + det
        return {"book_pp": 100 * r_book, "hedge_pp": 100 * hedge / self.v0, "premium_pp": 100 * self.premium,
                "risk_intra_pp": 100 * self.charge_intra, "risk_close_pp": 100 * self.charge_close, "R_trade": r_trade,
                "P_close": p_close, "close_terms_pp": det, "exit_cost_pp": 100 * exit_cost / self.v0, "overnight_pp": 0.0,
                "gross_close": m.gross() / max(nav_close, 1e-9), "commissions": m.commissions, "fills": len(m.fills),
                "at_close": at_close, "nav_close": nav_close, **carry,
                "carry_charge": KAPPA_CARRY * carry["carry_excess_unprotected"]}

    def carry(self, m):
        """Positions carried past the session's close: how many, how many with no working protective order covering at
        least half (R3, reported), and, per name, the share of NAV above CARRY_FREE left unprotected after its stop's
        stop_credit (the charged part; criteria v1.1 note (a): week 2's losses were the large names)."""
        nav = m.nav()
        nav = nav if nav > 0 else 1e-9
        n = bare = 0
        excess = 0.0
        for s, (q, _) in m.positions.items():
            px = m.last(s) or 0.0
            if not px:
                continue
            cov, dist = protective(m, s, q, px)
            r1 = (self.fc_rows.get(s) or {}).get("range1_pct")
            b, r = reward._risk_inputs(self.risk.feats, s)
            r1 = r1 / 100 if r1 else 2.5 * math.sqrt(r ** 2 + (b * self.risk.sigma_m) ** 2)
            n += 1
            bare += cov < 0.5
            excess += max(0.0, abs(q * px) / nav - CARRY_FREE) * (1 - cov * stop_credit(dist, r1))
        return {"carried": n, "carried_unprotected": bare, "carry_excess_unprotected": excess}


def book_summary(m):
    """The book's concentration and protection at the current marks: the largest holding's weight and whether working
    protective orders cover at least half of it."""
    nav = m.nav()
    rows = []
    for s, (q, _) in m.positions.items():
        px = m.last(s) or 0.0
        cov, _ = protective(m, s, q, px) if px else (0.0, None)
        rows.append((abs(q * px) / nav if nav > 0 else 0.0, s, cov))
    rows.sort(key=lambda r: -r[0])
    top = rows[0] if rows else (0.0, None, 0.0)
    return {"positions": len(rows), "top_weight": round(top[0], 4), "top_symbol": top[1], "top_protected": top[2] >= 0.5,
            "unprotected_75": top[0] >= 0.75 and top[2] < 0.5, "gross": round(sum(r[0] for r in rows), 4)}


