"""The trader: one decision per trading day, the same code in simulation and on the paper account.

Before each session the model sees its sleeve (cash, positions, open orders, what happened since its
last decision) and a market table built from daily bars through the previous close, and answers with
exactly one `submit` call: orders to place and order refs to cancel. Software places exactly what it
writes. SimWorld runs it on sim.Market; PaperWorld runs it on a sleeve of the IBKR paper account.
"""

import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import bars as barsmod
import sim

ROOT = Path(__file__).resolve().parent
UNIVERSE_FILE = ROOT / "lists" / "universe523.txt"   # today's S&P members that were added by 2024-10-14 (survivor-biased) + ETFs, no leveraged/inverse/volatility ETFs
FIXED = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
TABLE_ROWS = 80
TYPES = ["MKT", "MOO", "LMT", "STP", "STP LMT", "TRAIL", "MOC", "LOC"]
CALL, RET = 200012, 200002

SYSTEM = """You manage a ${size:,} sleeve of an Interactive Brokers paper account that trades US stocks and ETFs. You decide everything: what to buy or sell short, how many shares, which order types and prices, and when to exit. Your orders go to the broker exactly as you write them; nobody reviews or changes them.

Each trading day before the open you see your sleeve and a market table built from daily bars through the previous close, and you answer with one submit call. Orders you place work from that day's open. DAY orders expire at the close; GTC orders work until they fill or you cancel them. To do nothing, submit empty lists.

How you are judged: the costed growth of the sleeve over rolling 10-session periods, compared with holding the same market exposure in SPY. Idle cash gives up the market's expected return, concentrated or very volatile books are charged for their risk, and commissions ($0.005 a share, minimum $1 an order) and slippage count. Missing an opportunity is as much an error as a bad trade.

Order types: MKT (at the open), MOO (in the opening auction), LMT (needs limit), STP (stop; needs stop), STP LMT (needs stop and limit), TRAIL (trailing stop; needs trail_percent), MOC and LOC (at the close; LOC needs limit). Short selling is allowed. Gross exposure may not exceed 2x the sleeve.

Market table columns: close = last close; r1, r5, r20, r60 = % change over 1, 5, 20 and 60 sessions; vol = daily volatility % over 20 sessions; off_hi = % below the 60-session high; beta = 60-session beta to SPY, shrunk one third toward 1. Any symbol in the universe can be traded, not only those shown."""

# Live trading: a look every trading hour (and one before the open), with research tools.
SYSTEM_LIVE = """You manage a ${size:,} sleeve of an Interactive Brokers paper account that trades US stocks and ETFs. You decide everything: what to buy or sell short, how many shares, which order types and prices, and when to exit. Your orders go to the broker exactly as you write them; nobody reviews or changes them.

You get a look every 30 minutes, from 09:00 ET (before the 09:30 open) to 15:30 ET. Each look shows your sleeve and a market table built from daily bars through the previous close; during the session the table also has delayed (about 15 minutes) last prices. Before deciding you may research with the tools you are given (quotes, price history and, where available, news and market scanners), then you call submit exactly once. Orders work from the moment you submit them (from the open, if you submit before it). DAY orders expire at the close; GTC orders work until they fill or you cancel them. To do nothing, submit empty lists.

Your goal: grow the sleeve {goal_pct}% this week, from ${size:,} to ${target:,} by the close on {deadline}: the pace of +200% a year. The sleeve's value counts after commissions ($0.005 a share, minimum $1 an order) and slippage, and it is compared with holding cash and with holding SPY over the same time. Missing an opportunity is as much an error as a bad trade.

Order types: MKT (market), MOO (opening auction; only before the open), LMT (needs limit), STP (stop; needs stop), STP LMT (needs stop and limit), TRAIL (trailing stop; needs trail_percent), MOC and LOC (at the close; LOC needs limit). Short selling is allowed. Gross exposure may not exceed 2x the sleeve.

Market table columns: close = last close; r1, r5, r20, r60 = % change over 1, 5, 20 and 60 sessions; vol = daily volatility % over 20 sessions; off_hi = % below the 60-session high; beta = 60-session beta to SPY, shrunk one third toward 1; last and today (during the session) = delayed last price and its % change from the previous close. The table lists the index and sector ETFs, then the stocks with the most dollars traded, so it leans toward large technology companies; that is a ranking by trading volume, not a recommendation. Every sector is open to you, and any US stock or ETF can be traded, not only those shown.

How to decide: research first, then act.
1. Read today's market news (tape) and look at your positions.
2. Form your own hypothesis: what should move, in which direction, and why (for example a news event, an economic release, a rotation between sectors, a trend, or a price that looks wrong).
3. Test it with the scanner: choose the scan and, if useful, the industry that would find stocks fitting your hypothesis. Don't just take the top of the gainers list.
4. Check every candidate with news and quote; use stats for its volatility (to size the trade) and, before a short, whether shares can be borrowed; bars if the trend matters. Do not trade a symbol you have not checked in this look.
5. Call submit once. Each look starts fresh: only your sleeve, your open orders and the notes you write in submit carry over to the next look, so write down your thesis, plan and exit levels there."""

GOAL_PCT = 2.14   # Clayton, 17 Oct 2026: +200% a year (CAGR), so 3 ** (1/52) - 1 = 2.14% a week; each week runs Monday to Friday


def goal_date(today):
    """The week's last session: its Friday."""
    return today + timedelta(days=4 - today.weekday()) if today.weekday() <= 4 else today


S1_NEWS, S1_NONE = "1. Read today's market news (tape) and look at your positions.", "1. Look at your positions and the market table."
S3_SCAN = "3. Test it with the scanner: choose the scan and, if useful, the industry that would find stocks fitting your hypothesis."
S3_ML = ("3. Test it with the scanner and the forecast tool (machine-learning forecasts; its description gives each model's "
         "out-of-sample track record, so weigh it accordingly): choose the scan and, if useful, the industry that would find "
         "stocks fitting your hypothesis.")
S4_NEWS, S4_NONE = "4. Check every candidate with news and quote;", "4. Check every candidate with quote;"


def live_system(size, today, ml=False, news=True):
    """SYSTEM_LIVE for a sleeve of this size, with this week's goal. ml: the sleeve has the forecast tool;
    news=False: the training simulator, which has no news for past dates."""
    g = goal_date(today)
    text = SYSTEM_LIVE
    if ml:
        text = text.replace(S3_SCAN, S3_ML)
    if not news:
        text = text.replace(S1_NEWS, S1_NONE).replace(S4_NEWS, S4_NONE)
    return (text.replace("${size:,}", f"${size:,.0f}").replace("${target:,}", f"${size * (1 + GOAL_PCT / 100):,.0f}")
            .replace("{goal_pct}", f"{GOAL_PCT:g}").replace("{deadline}", f"{g:%A} {g.day} {g:%B} {g.year}"))


def goal_line(size, value, sessions_left, today):
    target = size * (1 + GOAL_PCT / 100)
    g = goal_date(today)
    gap = (f"{fmt_money(target - value)} to go" if value < target else f"{fmt_money(value - target)} above the goal")
    return (f"Goal: {fmt_money(target)} by the close on {g:%A} {g.day} {g:%B} ({GOAL_PCT:g}% above the "
            f"{fmt_money(size)} start). Sleeve now {fmt_money(value)} ({100 * (value / size - 1):+.1f}% since the start), {gap}; "
            f"{sessions_left} session{'s' if sessions_left != 1 else ''} left including today.")


def live_steps(ml=False, news=True):
    s = LIVE_STEPS
    if ml:
        s = s.replace("with the scanner (choose", "with the forecast tool (machine-learning forecasts), the scanner (choose")
    if not news:
        s = s.replace("(1) call tape for market news; ", "(1) look at your positions; ").replace("call news, quote and stats", "call quote and stats")
    return s


def live_user(text, goal, memo=None, steps=None):
    """The live look's user message: the steps, then the state with the goal line and the model's own notes
    from its last look after the date header."""
    head, rest = text.split("\n", 1)
    if memo is not None:
        goal += (f"\nYour notes from your last look ({memo['when']}): " + (memo["notes"].strip() or "none.")
                 if memo else "\nYour notes from your last look: none yet.")
    return (LIVE_STEPS if steps is None else steps) + head + "\n" + goal + "\n" + rest

# Put right before the state at every live look (prompt v2 in tooltest.py: base researched in 9 of 10 looks
# with it and 0 of 10 without; rl1 10 of 10 vs 0 of 10).
LIVE_STEPS = ("Before you submit: (1) call tape for market news; (2) decide your hypothesis: what should move, which way, and why; "
              "(3) find candidates that fit it with the scanner (choose the scan and industry that test it), the table and your positions; "
              "(4) call news, quote and stats for your candidates; (5) call submit, with notes for your next look and your "
              "honest p_fill and p_win on every order.\n\n")

from tools import NOTES_DESC, P_FILL_DESC, P_WIN_DESC, PROBS_DESC   # shared with mcp_server.py

TOOL = {
    "name": "submit",
    "description": "Submit today's decision: orders to place and open orders to cancel. Call it exactly once. Empty lists mean no change.",
    "parameters": {
        "type": "object",
        "properties": {
            "place": {"type": "array", "description": "New orders.", "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": ["BUY", "SELL"]},
                    "quantity": {"type": "integer", "description": "Whole shares."},
                    "type": {"type": "string", "enum": TYPES},
                    "limit": {"type": "number", "description": "Limit price, for LMT, STP LMT and LOC."},
                    "stop": {"type": "number", "description": "Stop price, for STP and STP LMT."},
                    "trail_percent": {"type": "number", "description": "Trailing distance in %, for TRAIL."},
                    "tif": {"type": "string", "enum": ["DAY", "GTC"]},
                    "p_fill": {"type": "number", "description": P_FILL_DESC},
                    "p_win": {"type": "number", "description": P_WIN_DESC},
                },
                "required": ["symbol", "action", "quantity", "type", "tif"],
            }},
            "cancel": {"type": "array", "description": "Refs of open orders to cancel.", "items": {"type": "string"}},
        },
        "required": ["place", "cancel"],
    },
}

# Live looks are stateless: only the sleeve, its orders and these notes carry over to the next look.
TOOL_LIVE = {**TOOL, "description": "Submit this look's decision: orders to place, open orders to cancel, and notes for your next "
                                    "look. Call it exactly once. Empty lists mean no change. " + PROBS_DESC,
             "parameters": {**TOOL["parameters"], "properties": {**TOOL["parameters"]["properties"],
                                                                 "notes": {"type": "string", "description": NOTES_DESC}}}}


ETFS_FILE = ROOT / "lists" / "etfs.txt"


def etfs():
    return {l.strip() for l in ETFS_FILE.read_text().splitlines() if l.strip()}


def universe():
    return [l.strip() for l in UNIVERSE_FILE.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- market data and features
class Data:
    """Daily bars for the universe on one calendar, plus point-in-time features."""

    def __init__(self, symbols=None, last=None):
        self.symbols = symbols or universe()
        self.bars, dates = barsmod.load_market(self.symbols)
        spy = self.bars["SPY"]
        self.dates = [d for d in dates if d in spy and (last is None or d <= last)]
        self.ix = {d: i for i, d in enumerate(self.dates)}
        n = len(self.dates)
        self.sym = [s for s in self.symbols if s in self.bars]
        self.k = {s: j for j, s in enumerate(self.sym)}
        self.etf = etfs()
        f = lambda key: np.array([[self.bars[s].get(d, {}).get(key, np.nan) for d in self.dates] for s in self.sym], float)
        self.o, self.h, self.l, self.c, self.v = f("open"), f("high"), f("low"), f("close"), f("volume")
        with np.errstate(divide="ignore", invalid="ignore"):
            self.lr = np.diff(np.log(self.c), axis=1, prepend=np.nan)   # daily log return ending at session t

    _simbars = None

    def simbars(self):
        """{symbol: {date: bar}} for sim.Market, built once."""
        if self._simbars is None:
            self._simbars = {s: {self.dates[i]: {"open": self.o[j, i], "high": self.h[j, i], "low": self.l[j, i],
                                                 "close": self.c[j, i], "volume": self.v[j, i]}
                                 for i in range(len(self.dates)) if not np.isnan(self.c[j, i])}
                             for s, j in self.k.items()}
        return self._simbars

    def features(self, d):
        """Features for a decision before session d, from bars through d-1. Returns {symbol: dict}."""
        t = d - 1
        out = {}
        spy = self.lr[self.k["SPY"], t - 59:t + 1]
        sm = np.nanstd(spy, ddof=1)
        for s, j in self.k.items():
            c = self.c[j]
            if t < 60 or np.isnan(c[t]):
                continue
            def ch(n):
                return (c[t] / c[t - n] - 1) * 100 if not np.isnan(c[t - n]) else np.nan
            r = self.lr[j, t - 59:t + 1]
            ok = ~np.isnan(r) & ~np.isnan(spy)
            beta = resid = np.nan
            if ok.sum() >= 40:
                x, y = spy[ok], r[ok]
                vx = np.var(x, ddof=1)
                raw = np.cov(x, y, ddof=1)[0, 1] / vx if vx > 0 else np.nan
                resid = np.std(y - raw * x, ddof=1)
                # Blume shrinkage toward 1 for stocks (60-session betas are noisy); ETFs keep their raw beta
                # (bond, gold and currency ETFs have betas near 0, not near 1)
                beta = raw if s in self.etf else 2 / 3 * raw + 1 / 3
            hi60 = np.nanmax(self.h[j, t - 59:t + 1])
            dv = np.nanmean(c[t - 19:t + 1] * self.v[j, t - 19:t + 1])
            out[s] = {"close": c[t], "r1": ch(1), "r5": ch(5), "r20": ch(20), "r60": ch(60),
                      "vol": np.nanstd(self.lr[j, t - 19:t + 1], ddof=1) * 100, "off_hi": (1 - c[t] / hi60) * 100,
                      "beta": beta, "resid": resid, "dollar_vol": dv}
        return out, sm

    def resid_cov(self, d, symbols):
        """Daily covariance of 60-session residual returns (return minus raw beta x SPY) through d-1."""
        t = d - 1
        spy = self.lr[self.k["SPY"], t - 59:t + 1]
        res = []
        for s in symbols:
            r = self.lr[self.k[s], t - 59:t + 1] if s in self.k else np.full(60, np.nan)
            ok = ~np.isnan(r) & ~np.isnan(spy)
            if ok.sum() >= 40:
                x, y = spy[ok], r[ok]
                b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
                e = np.full(60, np.nan)
                e[ok] = y - b * x
            else:
                e = np.full(60, np.nan)
            res.append(e)
        R = np.array(res)
        n = len(symbols)
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ok = ~np.isnan(R[i]) & ~np.isnan(R[j])
                c = np.cov(R[i, ok], R[j, ok], ddof=1)[0, 1] if ok.sum() >= 40 else (0.02 ** 2 if i == j else 0.0)
                cov[i, j] = cov[j, i] = c
        return cov

    def table_rows(self, d, extra=()):
        feats, _ = self.features(d)
        ranked = sorted((s for s in feats if s not in FIXED), key=lambda s: -(feats[s]["dollar_vol"] or 0))
        rows = [s for s in FIXED if s in feats]
        rows += ranked[:TABLE_ROWS - len(rows)]
        rows += [s for s in extra if s in feats and s not in rows]
        return rows, feats

    def table(self, d, extra=(), live=None):
        """live: {symbol: delayed last price} during the session; adds 'last' and 'today' (% vs previous close)."""
        rows, feats = self.table_rows(d, extra)
        fmt = lambda v, p=1: "na" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{p}f}"
        lines = ["sym close r1 r5 r20 r60 vol off_hi beta" + (" last today" if live else "")]
        for s in rows:
            x = feats[s]
            line = (f"{s} {fmt(x['close'], 2)} {fmt(x['r1'])} {fmt(x['r5'])} {fmt(x['r20'])} {fmt(x['r60'])} "
                    f"{fmt(x['vol'])} {fmt(x['off_hi'])} {fmt(x['beta'], 2)}")
            if live:
                last = live.get(s)
                line += f" {fmt(last, 2)} {fmt((last / x['close'] - 1) * 100 if last else None)}"
            lines.append(line)
        return "\n".join(lines), feats

    next_date = None   # live: the date of the session about to trade (no bar for it yet)

    def iso(self, d):
        if d >= len(self.dates):
            return self.next_date
        s = self.dates[d]
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"


# ---------------------------------------------------------------- prompt, parsing
_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        _tok = get_tokenizer("openai/gpt-oss-120b")
    return _tok


def render(state_text, size, decision_date, effort="low"):
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    r = GptOssRenderer(tokenizer(), use_system_prompt=True, reasoning_effort=effort, current_date=decision_date)
    msgs = r.create_conversation_prefix_with_tools([TOOL], SYSTEM.replace("${size:,}", f"${size:,.0f}"))
    msgs.append({"role": "user", "content": state_text})
    return r.build_generation_prompt(msgs)


SEG = re.compile(r"^(?: to=(?P<r1>[^\s<]+))?<\|channel\|>(?P<ch>analysis|commentary|final)(?: to=(?P<r2>[^\s<]+))?"
                 r"\s*(?:<\|constrain\|>(?P<con>\w+))?<\|message\|>(?P<body>.*?)(?P<term><\|end\|>|<\|call\|>|<\|return\|>)$", re.S)


def parse(tokens, stop_reason):
    """(call_dict, None) or (None, error). The call must be one functions.submit call with valid JSON."""
    if str(stop_reason).split(".")[-1].lower() != "stop" or not tokens or tokens[-1] != CALL:
        return None, "no submit call (output ended without a tool call)"
    segs = tokenizer().decode(tokens).split("<|start|>assistant")
    for k, seg in enumerate(segs):
        m = SEG.match(seg)
        if not m or "<|" in m["body"]:
            return None, f"unparseable output segment {k}"
        rc = m["r1"] or m["r2"]
        if k < len(segs) - 1:
            if m["term"] != "<|end|>" or rc:
                return None, "a tool call or final answer before the end"
            continue
        if m["term"] != "<|call|>" or rc != "functions.submit":
            return None, f"call to {rc!r}, not functions.submit"
        try:
            call = json.loads(m["body"])
        except ValueError as e:
            return None, f"arguments are not valid JSON: {e}"
        if not isinstance(call, dict):
            return None, "arguments must be an object"
        call.setdefault("place", [])
        call.setdefault("cancel", [])
        if not isinstance(call["place"], list) or not isinstance(call["cancel"], list):
            return None, "place and cancel must be lists"
        return call, None
    return None, "empty output"


def analysis_text(tokens):
    t = tokenizer().decode(tokens)
    m = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", t, re.S)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- account text
def fmt_money(x):
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def account_text(world, data, d, feats, header=None, live=None):
    st = world.state(d, feats, live) if live else world.state(d, feats)
    lines = [header or (f"Date: {data.iso(d)} ({date.fromisoformat(data.iso(d)).strftime('%A')}). "
                        f"Orders you submit now work from today's open."),
             f"Sleeve value {'now, at delayed last prices' if live else 'at last close'}: {fmt_money(st['value'])} (cash {fmt_money(st['cash'])}). "
             f"Gross exposure {st['gross']:.2f}x, net {st['net']:+.2f}x, beta-weighted {st['beta_net']:+.2f}x."]
    if getattr(world, "MAX_GROSS", None) and st["gross"] > world.MAX_GROSS:
        lines.append(f"Over the margin limit: gross exposure is {st['gross']:.2f}x the sleeve, above {world.MAX_GROSS:g}x. "
                     f"Orders that add exposure are rejected until you bring it under {world.MAX_GROSS:g}x.")
    if st["positions"]:
        lines.append(f"Positions (sym qty avg_cost {'last' if live else 'close'} unrealized days_held):")
        for p in st["positions"]:
            lines.append(f"  {p['symbol']} {p['quantity']:+d} {p['avg_cost']:.2f} {p['close']:.2f} {p['unrealized']:+,.0f} {p['days']}")
    else:
        lines.append("Positions: none.")
    if st["orders"]:
        lines.append("Open orders (ref sym action qty type limit stop trail% tif):")
        for o in st["orders"]:
            g = lambda k: "-" if o.get(k) in (None, 0) else f"{o[k]:g}"
            lines.append(f"  {o['ref']} {o['symbol']} {o['action']} {o['quantity']} {o['type']} {g('limit')} {g('stop')} {g('trail_percent')} {o['tif']}")
    else:
        lines.append("Open orders: none.")
    lines.append("Since your last decision: " + ("; ".join(st["events"]) if st["events"] else "nothing happened."))
    return "\n".join(lines), st


def state_text(world, data, d, header=None, live=None):
    held = world.symbols_in_play()
    table, feats = data.table(d, extra=held, live=live)
    acct, st = account_text(world, data, d, feats, header, live)
    label = (f"daily bars through the close of {data.iso(d - 1)}" +
             ("; last = delayed last price, today = % change from the previous close" if live else ""))
    return acct + f"\n\nMarket table ({label}):\n" + table, st, feats


def _num(x, name, integer=False):
    if x is None:
        return None, None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None, f"{name} must be a number, not {x!r}"
    if not math.isfinite(x):
        return None, f"{name} must be finite"
    if integer:
        if float(x) != int(x):
            return None, f"{name} must be a whole number of shares"
        return int(x), None
    return float(x), None


def clean_order(o):
    """Normalize one model order to sim/broker fields; returns (fields, error). Nothing is changed beyond
    case and spacing: a missing or non-numeric field is an error, the same in simulation and live. p_fill and
    p_win pass through as floats in [0, 1], else None; they never cause an error and never reach the broker."""
    if not isinstance(o, dict):
        return None, "order is not an object"
    sym = str(o.get("symbol", "")).upper().strip()
    act = str(o.get("action", "")).upper()
    typ = str(o.get("type", "")).upper().replace("_", " ")
    tif = str(o.get("tif") or "DAY").upper()     # IBKR treats an order without a time-in-force as DAY
    q, err = _num(o.get("quantity"), "quantity", integer=True)
    if err or q is None:
        return None, f"{act} {sym}: " + (err or "quantity is required")
    out = {"symbol": sym, "action": act, "quantity": q, "type": typ, "tif": tif}
    for k in ("limit", "stop", "trail_percent"):
        v, err = _num(o.get(k), k)
        if err:
            return None, f"{act} {sym}: {err}"
        out[k] = v
    out["p_fill"], out["p_win"] = prob(o.get("p_fill")), prob(o.get("p_win"))   # recorded and scored only; never an error
    return out, None


def prob(x):
    """A model's probability as a float if it is a number in [0, 1], else None."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 else None


# ---------------------------------------------------------------- worlds
class SimWorld:
    """A sleeve on sim.Market. The market's clock is the session about to trade."""

    MAX_GROSS = 2.0   # sim.Market's max_gross: shown to the model when the book is over it, as in live looks

    def __init__(self, data, d, cash, slippage_bps=2.0):
        self.data = data
        self.m = sim.Market(data.simbars(), data.dates, d, float(cash), slippage_bps=slippage_bps)
        self.start_cash = float(cash)
        self.opened = {}            # symbol -> session index when the position opened
        self.events = []
        self._seen_fills = 0
        self._n = 0

    def clone(self):
        import copy
        w = copy.copy(self)
        w.m = self.m.clone()
        w.opened = dict(self.opened)
        w.events = list(self.events)
        return w

    def symbols_in_play(self):
        return sorted(set(self.m.positions) | {o.symbol for o in self.m.open_orders()})

    def state(self, d, feats):
        m = self.m
        value = m.nav()
        pos = []
        gross = net = bnet = 0.0
        for s, (q, avg) in sorted(m.positions.items()):
            px = m.last(s)
            v = q * px
            gross += abs(v); net += v
            b = feats.get(s, {}).get("beta")
            bnet += v * (b if b is not None and not math.isnan(b) else 1.0)
            pos.append({"symbol": s, "quantity": int(q), "avg_cost": avg, "close": px, "unrealized": q * (px - avg),
                        "days": d - self.opened.get(s, d)})
        orders = [{"ref": o.ref, "symbol": o.symbol, "action": o.action, "quantity": int(o.quantity), "type": o.order_type,
                   "limit": o.limit, "stop": o.aux if o.order_type in ("STP", "STP LMT") else None,
                   "trail_percent": o.trailing_percent, "tif": o.tif} for o in m.open_orders()]
        ev, self.events = self.events, []
        return {"value": value, "cash": m.cash, "positions": pos, "orders": orders, "events": ev,
                "gross": gross / value if value > 0 else 0, "net": net / value if value > 0 else 0,
                "beta_net": bnet / value if value > 0 else 0}

    def act(self, call):
        notes = []
        for ref in call.get("cancel", []):
            if self.m.cancel(str(ref)) is None:
                notes.append(f"cancel {ref}: no such open order")
        for raw in call.get("place", []):
            o, err = clean_order(raw)
            if err:
                notes.append(f"order rejected: {err}")
                continue
            self._n += 1
            aux = o["stop"] if o["type"] in ("STP", "STP LMT") else None
            od = self.m.place(o["symbol"], o["action"], o["quantity"], o["type"], o["limit"], aux, o["tif"],
                              trailing_percent=o["trail_percent"], ref=f"o{self.m.session}-{self._n}")
            if od.status == "Rejected":
                notes.append(f"{o['action']} {o['quantity']} {o['symbol']} {o['type']} rejected: {od.message}")
        self.events.extend(notes)
        return notes

    def advance(self):
        """Trade one session; record fills, expiries and margin rejections as events for the next look."""
        before = {o.ref: o.status for o in self.m.orders}
        self.m.advance(1)
        d = self.m.session - 1
        for f in self.m.fills[self._seen_fills:]:
            self.events.append(f"filled {f.side} {f.shares:g} {f.symbol} @ {f.price:.2f} (commission ${f.commission:.2f}, ref {f.ref})")
            q = self.m.positions.get(f.symbol, [0])[0]
            if q and f.symbol not in self.opened:
                self.opened[f.symbol] = d
            if not q:
                self.opened.pop(f.symbol, None)
        self._seen_fills = len(self.m.fills)
        for o in self.m.orders:
            if before.get(o.ref) not in (None, "Expired", "Rejected", "Filled", "Cancelled") and o.status in ("Expired", "Rejected"):
                self.events.append(f"{o.status.lower()}: {o.ref} {o.action} {o.quantity:g} {o.symbol} {o.order_type}"
                                   + (f" ({o.message})" if o.message else ""))

    def liquidate(self):
        self.m.liquidate()
        return self.m.cash


class PaperWorld:
    """A sleeve of the IBKR paper account through paper.py. Marks come from daily closes, or delayed last
    prices during the session."""

    def __init__(self, gw, sleeve, data):
        import paper
        self.paper, self.gw, self.sleeve, self.data = paper, gw, sleeve, data
        self.dry_run = False

    def symbols_in_play(self):
        b = self.paper.book(self.sleeve)
        return sorted(set(b["positions"]) | {o["symbol"] for o in self.paper.open_orders(self.gw, self.sleeve)})

    def state(self, d, feats, live=None):
        marks = {s: x["close"] for s, x in feats.items()} | (live or {})   # during the session: delayed last prices
        b = self.paper.book(self.sleeve, marks)
        value = b["cash"] + sum(p["quantity"] * (p["mark"] if p["mark"] is not None else p["avg_cost"])
                                for p in b["positions"].values())
        pos, gross, net, bnet = [], 0.0, 0.0, 0.0
        for s, p in sorted(b["positions"].items()):
            px = p["mark"] or p["avg_cost"]
            v = p["quantity"] * px
            gross += abs(v); net += v
            beta = feats.get(s, {}).get("beta")
            bnet += v * (beta if beta is not None and not math.isnan(beta) else 1.0)
            op = (p.get("opened") or "")
            held = d - self.data.ix[op] if op in self.data.ix else 0
            pos.append({"symbol": s, "quantity": int(p["quantity"]), "avg_cost": p["avg_cost"], "close": px,
                        "unrealized": p.get("unrealized", 0.0), "days": held})
        orders = []
        for o in self.paper.open_orders(self.gw, self.sleeve):
            orders.append({"ref": o["ref"], "symbol": o["symbol"], "action": o["action"], "quantity": o["quantity"],
                           "type": o["order_type"], "limit": o.get("limit"), "stop": o.get("aux"), "trail_percent": None,
                           "tif": o.get("tif", "DAY")})
        ev = self.paper.events_since_last_look(self.gw, self.sleeve, commit=not self.dry_run) + self.paper.take_notes(self.sleeve, commit=not self.dry_run)
        return {"value": value, "cash": b["cash"], "positions": pos, "orders": orders, "events": ev,
                "gross": gross / value if value else 0, "net": net / value if value else 0, "beta_net": bnet / value if value else 0}

    MAX_GROSS = 2.0   # each sleeve is its own margin account, as in sim.py: the paper account's margin is shared

    def act(self, call):
        notes = []
        for ref in call.get("cancel", []):
            r = self.paper.cancel(self.gw, self.sleeve, str(ref))
            if r.get("error"):
                notes.append(f"cancel {ref}: {r['error']}")
        orders = []
        for raw in call.get("place", []):
            o, err = clean_order(raw)
            if err:
                notes.append(f"order rejected: {err}")
            else:
                orders.append(o)
        # Margin, as sim.py and IBKR apply it: an order that adds exposure is rejected if, filled at its limit or
        # stop (else the last price), it would take the sleeve's gross exposure above MAX_GROSS x its value.
        syms = sorted(set(self.paper.book(self.sleeve)["positions"]) | {o["symbol"] for o in orders})
        quotes = {q["symbol"]: q for q in self.gw.quotes(syms) if not q.get("error")} if syms else {}
        px = {s: self.paper.mark(q) for s, q in quotes.items()}
        b = self.paper.book(self.sleeve, px)
        pos = {s: p["quantity"] for s, p in b["positions"].items()}
        mark = {s: (px.get(s) or p["avg_cost"]) for s, p in b["positions"].items()}
        nav = b["cash"] + sum(q * mark[s] for s, q in pos.items())
        for o in orders:
            s, q0 = o["symbol"], pos.get(o["symbol"], 0)
            q1 = q0 + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
            p = o["limit"] or o["stop"] or px.get(s) or mark.get(s)
            if p and abs(q1) > abs(q0):
                gross = sum(abs(q) * mark.get(k, p) for k, q in pos.items() if k != s) + abs(q1) * p
                if gross > self.MAX_GROSS * max(nav, 0.0):
                    notes.append(f"{o['action']} {o['quantity']} {s} {o['type']} rejected: insufficient margin (gross exposure "
                                 f"would be {gross / nav if nav > 0 else float('inf'):.2f}x the sleeve, above the {self.MAX_GROSS:g}x limit)")
                    continue
            pos[s] = q1
            if p:
                mark.setdefault(s, p)
            typ, tif, fields = o["type"], o["tif"], {}
            limit = o["limit"] if typ in ("LMT", "STP LMT", "LOC") else None
            aux = o["stop"] if typ in ("STP", "STP LMT") else None
            if typ == "MOO":
                typ, tif = "MKT", "OPG"
            if typ == "TRAIL" and o["trail_percent"] is not None:
                fields["trailingPercent"] = float(o["trail_percent"])
            q = quotes.get(o["symbol"], {})              # arrival quote, logged for the execution model (wave C)
            arrival = {k: q.get(k) for k in ("bid", "ask", "last", "close", "last_time")} if q else None
            r = self.paper.place(self.gw, self.sleeve, o["symbol"], o["action"], o["quantity"], typ, limit, aux, tif,
                                 arrival=arrival, probs={"p_fill": o["p_fill"], "p_win": o["p_win"]}, **fields)
            res = r.get("result") or {}
            msg = r.get("error") or "; ".join(res.get("messages", []) or [])
            status = res.get("status", "error" if r.get("error") else "?")
            if r.get("error") or status in ("Cancelled", "Inactive", "ApiCancelled"):
                notes.append(f"{o['action']} {o['quantity']} {o['symbol']} {o['type']}: {status} {msg}".strip())
        self.paper.add_notes(self.sleeve, notes)
        return notes
