"""A daily-bar paper market for replaying history: orders in, fills and costs out.

Time runs in sessions. Before session d the trader sees bars up to session d-1 and places orders;
advance() then fills them against session d's bar and moves on. No bar is visible before its
session has closed, so a decision can never use prices it would not have had.

Fills (one daily bar per symbol, so intraday order is unknown; ties resolve against the trader):
  MKT / MOO       at the open
  LMT             BUY: open if open <= limit, else limit if low <= limit. SELL mirrored.
  STP             SELL: open if open <= stop, else stop if low <= stop. BUY mirrored.
  STP LMT         triggers like STP, then fills like LMT in the same bar if the bar allows
  TRAIL           stop trails the best close since placement by aux (or trailing_percent)
  MOC / LOC       at the close (LOC only if the close satisfies the limit)
Every market-type fill pays slippage (adverse, in bps), and every fill pays IBKR fixed commission:
$0.005 a share, minimum $1, maximum 1% of the trade value (what the paper account charges).
DAY orders expire after one session; GTC orders stay until filled or cancelled.
A short is allowed. An order that would take gross exposure above `max_gross` x NAV is rejected,
as IBKR would reject it for margin.
"""

import math
from dataclasses import dataclass, field

FINISHED = ("Filled", "Cancelled", "Expired", "Rejected")


def commission(shares, price):
    """IBKR fixed pricing, as charged on this paper account: $0.005 a share, minimum $1, maximum 1% of value."""
    c = max(1.0, 0.005 * abs(shares))
    return min(c, 0.01 * abs(shares) * price)


@dataclass
class Order:
    ref: str
    symbol: str
    action: str          # BUY / SELL
    quantity: float
    order_type: str = "MKT"
    limit: float | None = None
    aux: float | None = None
    tif: str = "DAY"
    trailing_percent: float | None = None
    placed: int = 0      # session index at placement
    status: str = "Submitted"
    triggered: bool = False
    trail_ref: float | None = None
    fill_price: float | None = None
    filled_session: int | None = None
    message: str = ""


@dataclass
class Fill:
    ref: str
    session: int
    date: str
    symbol: str
    side: str            # BOT / SLD
    shares: float
    price: float
    commission: float
    phase: str = "open"      # where in the bar it filled: open, close or intraday
    raw: float = 0.0         # price before slippage (the touched limit/stop, or the open/close)


@dataclass
class Market:
    bars: dict           # symbol -> {date: {"open","high","low","close","volume"}}
    dates: list          # trading calendar (sorted "YYYYMMDD")
    session: int         # index of the next session to trade
    cash: float
    slippage_bps: float = 2.0
    max_gross: float = 2.0
    positions: dict = field(default_factory=dict)   # symbol -> [qty, avg_cost]
    orders: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    realized: float = 0.0
    commissions: float = 0.0
    financing: float = 0.0
    margin_rate: float = 0.05      # a year, on negative cash
    borrow_rate: float = 0.003     # a year, on short market value
    _n: int = 0

    def clone(self):
        """An independent copy that shares the (read-only) bars and calendar."""
        import copy
        m = copy.copy(self)
        m.positions = {k: list(v) for k, v in self.positions.items()}
        m.orders = [copy.copy(o) for o in self.orders]
        m.fills = list(self.fills)
        return m

    # ----- what the trader can see -----
    @property
    def today(self):
        """The date of the last completed session (the trader's 'now')."""
        return self.dates[self.session - 1]

    def bar(self, symbol, d=None):
        d = self.session - 1 if d is None else d
        return self.bars.get(symbol, {}).get(self.dates[d])

    def history(self, symbol, n=260):
        out = []
        for d in range(max(0, self.session - n), self.session):
            b = self.bars.get(symbol, {}).get(self.dates[d])
            if b:
                out.append({"date": self.dates[d], **b})
        return out

    def last(self, symbol):
        for d in range(self.session - 1, -1, -1):
            b = self.bars.get(symbol, {}).get(self.dates[d])
            if b:
                return b["close"]
        return None

    def nav(self):
        v = self.cash
        for s, (q, _) in self.positions.items():
            px = self.last(s)
            v += q * (px if px is not None else 0.0)
        return v

    def gross(self):
        return sum(abs(q) * (self.last(s) or 0.0) for s, (q, _) in self.positions.items())

    def open_orders(self):
        return [o for o in self.orders if o.status not in FINISHED]

    # ----- what the trader can do -----
    def place(self, symbol, action, quantity, order_type="MKT", limit=None, aux=None, tif="DAY",
              trailing_percent=None, ref=None):
        self._n += 1
        o = Order(ref or f"o{self._n}", str(symbol).upper(), str(action).upper(), quantity,
                  str(order_type).upper(), limit, aux, str(tif).upper(), trailing_percent, self.session)
        self.orders.append(o)
        err = None
        if o.action not in ("BUY", "SELL"):
            err = f"action must be BUY or SELL, not {action!r}"
        elif not isinstance(quantity, (int, float)) or quantity <= 0 or quantity != int(quantity):
            err = f"quantity must be a positive whole number of shares, not {quantity!r}"
        elif o.symbol not in self.bars or self.last(o.symbol) is None:
            err = f"no market data for {o.symbol}"
        elif o.order_type not in ("MKT", "MOO", "LMT", "STP", "STP LMT", "TRAIL", "MOC", "LOC"):
            err = f"order type {o.order_type} is not supported here"
        elif o.order_type in ("LMT", "STP LMT", "LOC") and not limit:
            err = f"{o.order_type} needs a limit price"
        elif o.order_type in ("STP", "STP LMT") and not aux:
            err = f"{o.order_type} needs a stop price (aux)"
        elif o.order_type == "TRAIL" and not (aux or trailing_percent):
            err = "TRAIL needs a trailing amount (aux) or trailing_percent"
        elif o.tif not in ("DAY", "GTC"):
            err = f"tif must be DAY or GTC, not {tif!r}"
        if err:
            o.status, o.message = "Rejected", err
        else:
            o.trail_ref = self.last(o.symbol)
        return o

    def cancel(self, ref):
        for o in self.orders:
            if o.ref == ref and o.status not in FINISHED:
                o.status = "Cancelled"
                return o
        return None

    # ----- time -----
    def advance(self, sessions=1):
        """Trade `sessions` sessions: fill against each bar, expire DAY orders, move the clock."""
        for _ in range(sessions):
            if self.session >= len(self.dates):
                return False
            d = self.session
            for o in self.open_orders():
                b = self.bars.get(o.symbol, {}).get(self.dates[d])
                if b:
                    self._try_fill(o, b, d)
                if o.status not in FINISHED and o.tif == "DAY":
                    o.status, o.message = "Expired", "DAY order not filled"
                if o.order_type == "TRAIL" and o.status not in FINISHED and b:
                    o.trail_ref = max(o.trail_ref, b["close"]) if o.action == "SELL" else min(o.trail_ref, b["close"])
            self.session += 1
            fee = max(0.0, -self.cash) * self.margin_rate / 252
            fee += sum(-q * (self.last(s) or 0.0) for s, (q, _) in self.positions.items() if q < 0) * self.borrow_rate / 252
            self.cash -= fee
            self.financing += fee
            self.realized -= fee
        return True

    def _try_fill(self, o, b, d):
        op, hi, lo, cl = b["open"], b["high"], b["low"], b["close"]
        buy = o.action == "BUY"
        px, phase = None, "open"
        t = o.order_type
        if t == "STP LMT" and o.triggered:          # triggered on an earlier bar: now a working limit
            t = "LMT"
        if t in ("MKT", "MOO"):
            px = op
        elif t == "MOC":
            px, phase = cl, "close"
        elif t == "LOC":
            px, phase = (cl if (cl <= o.limit if buy else cl >= o.limit) else None), "close"
        elif t == "LMT":
            px = self._limit_px(buy, o.limit, op, hi, lo)
            phase = "open" if px == op else "intraday"
        elif t in ("STP", "STP LMT", "TRAIL"):
            stop = o.aux
            if t == "TRAIL":
                amt = o.aux if o.aux else o.trail_ref * o.trailing_percent / 100
                stop = o.trail_ref - amt if not buy else o.trail_ref + amt
            gap = op >= stop if buy else op <= stop
            hit = gap or (hi >= stop if buy else lo <= stop)
            if hit:
                trig = op if gap else stop
                phase = "open" if gap else "intraday"
                if t == "STP LMT":
                    o.triggered = True
                    if trig <= o.limit if buy else trig >= o.limit:
                        px = trig
                    else:                        # stop hit but limit not reachable at the trigger: work the limit for the rest of the bar
                        px = self._limit_px(buy, o.limit, trig, hi, lo) if (lo <= o.limit if buy else hi >= o.limit) else None
                        phase = "intraday"
                else:
                    px = trig
        if px is None:
            return
        raw = px
        market_type = t in ("MKT", "MOO", "MOC", "STP", "TRAIL")
        if market_type or phase in ("open", "close"):      # fills at the open/close auction or at market pay slippage
            slip = px * self.slippage_bps / 1e4
            px = px + slip if buy else px - slip
            if not market_type and o.limit:               # ...but a limit order never fills worse than its limit
                px = min(px, o.limit) if buy else max(px, o.limit)
        qty = o.quantity if buy else -o.quantity
        # margin: reject what would push gross exposure past max_gross x NAV
        q0 = self.positions.get(o.symbol, [0, 0.0])[0]
        gross_after = self.gross() - abs(q0) * (self.last(o.symbol) or px) + abs(q0 + qty) * px
        if gross_after > self.max_gross * max(self.nav(), 0.0) + 1e-9 and abs(q0 + qty) > abs(q0):
            o.status, o.message = "Rejected", "insufficient margin (gross exposure would exceed limit)"
            return
        com = commission(o.quantity, px)
        self.cash -= qty * px + com
        self.commissions += com
        self.realized -= com
        q, avg = self.positions.get(o.symbol, [0, 0.0])
        if q == 0 or (q > 0) == (qty > 0):
            avg = (q * avg + qty * px) / (q + qty)
            q += qty
        else:
            closed = min(abs(qty), abs(q)) * (1 if q > 0 else -1)
            self.realized += closed * (px - avg)
            q += qty
            if q != 0 and (q > 0) != (closed > 0):
                avg = px
        if q:
            self.positions[o.symbol] = [q, avg]
        else:
            self.positions.pop(o.symbol, None)
        o.status, o.fill_price, o.filled_session = "Filled", px, d
        self.fills.append(Fill(o.ref, d, self.dates[d], o.symbol, "BOT" if buy else "SLD", o.quantity, px, com, phase, raw))

    @staticmethod
    def _limit_px(buy, limit, op, hi, lo):
        if buy:
            return op if op <= limit else (limit if lo <= limit else None)
        return op if op >= limit else (limit if hi >= limit else None)

    def liquidate(self):
        """Close every position at the next session's open with costs (end of an episode)."""
        for o in self.open_orders():
            o.status = "Cancelled"
        for s, (q, _) in list(self.positions.items()):
            self.place(s, "SELL" if q > 0 else "BUY", abs(q), "MKT", ref=f"liq-{s}")
        return self.advance(1)


def max_drawdown(navs):
    peak, mdd = -math.inf, 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak > 0 else 0.0)
    return mdd
