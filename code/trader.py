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

Your target pace: about +{goal_pct}% a week, the pace of +200% a year. The sleeve is judged on its realized, costed results over days to months (after commissions of $0.005 a share, minimum $1 an order, and slippage), against holding cash and against holding SPY over the same time. Missing an opportunity is as much an error as a bad trade.

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


KELLY = """Sizing by the Kelly rule: size each position from your edge and its risk, not from conviction alone. If you expect a position to earn mu over the time you plan to hold it (after costs), and its typical move over that time is sigma, the fraction of the sleeve that grows it fastest is about mu / sigma^2. The market table's vol is a daily figure: multiply it by the square root of the sessions you plan to hold. The stats tool's hist_vol and implied_vol are annualized: divide them by about 16 (the square root of 252) for a daily figure. For example, +0.2% expected over 5 sessions with a 5-session sigma of 5% gives 0.002/0.0025 = 0.8x the sleeve, and +0.4% gives 1.6x: a small error in mu is a large error in size. Betting more than twice the Kelly fraction loses money even when the edge is real, and half of it keeps about three quarters of the growth at half the risk. So:
- No edge, no position: if you can't say why mu is above zero after costs, the Kelly size is zero.
- Use a quarter to a half of the Kelly fraction, because your mu is an estimate.
- The sleeve's limits (2x gross, and the single-shock limit if your state shows one) cap the Kelly size too.
- Positions that move together are one bet: size a group of similar names (several chip stocks, or a sector ETF and its members) as one position, from the group's sigma.
- What you hold overnight carries the gap too: the table's vol is close to close, so it already includes gaps, but a stop doesn't limit a gap, so don't shrink sigma for a stop on a position you hold overnight.
- For each new position, write mu, sigma and the fraction you chose in your notes."""
PACE_END = "Missing an opportunity is as much an error as a bad trade.\n\nOrder types:"


S1_NEWS, S1_NONE = "1. Read today's market news (tape) and look at your positions.", "1. Look at your positions and the market table."
S3_SCAN = "3. Test it with the scanner: choose the scan and, if useful, the industry that would find stocks fitting your hypothesis."
S3_ML = ("3. Test it with the scanner and the forecast tool (machine-learning forecasts; its description gives each model's "
         "out-of-sample track record, so weigh it accordingly): choose the scan and, if useful, the industry that would find "
         "stocks fitting your hypothesis.")
S4_NEWS, S4_NONE = "4. Check every candidate with news and quote;", "4. Check every candidate with quote;"


def live_system(size, today, ml=False, news=True, kelly=False):
    """SYSTEM_LIVE for a sleeve of this size, with the target pace (Clayton, 24 Oct: a pace, not a weekly dollar goal
    with a deadline). ml: the sleeve has the forecast tool; news=False: the training simulator, which has no news;
    kelly: the Kelly sizing paragraph after the pace (Clayton, 9 Nov: "a kelly improved prompt"; sleeves.json
    "kelly_prompt"). It is guidance the model may weigh; nothing enforces it."""
    text = SYSTEM_LIVE
    if kelly:
        assert text.count(PACE_END) == 1
        text = text.replace(PACE_END, PACE_END.replace("\n\nOrder types:", "\n\n" + KELLY + "\n\nOrder types:"))
    if ml:
        text = text.replace(S3_SCAN, S3_ML)
    if not news:
        text = text.replace(S1_NEWS, S1_NONE).replace(S4_NEWS, S4_NONE)
    return text.replace("${size:,}", f"${size:,.0f}").replace("{goal_pct}", f"{GOAL_PCT:g}")


def goal_line(size, value, sessions_left=None, today=None):
    """The sleeve against its start: no dollar target, gap or deadline (Clayton, 24 Oct; week 2 sized from the gap),
    and the weekly reset stated as a fact (Clayton, 24 Oct: positions are still flattened every Monday).
    sessions_left is kept for callers and unused; today (a date or ISO string) dates the next reset."""
    line = f"Sleeve now {fmt_money(value)} ({100 * (value / size - 1):+.1f}% since its {fmt_money(size)} start)."
    if today is not None:
        t = date.fromisoformat(str(today)[:10]) if not isinstance(today, date) else today
        nxt = t + timedelta(days=7 - t.weekday())
        line += (f" All positions are closed at market at the open on Monday {nxt.day} {nxt:%B}, and the sleeve "
                 f"restarts at {fmt_money(size)}.")
    return line


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

    def outside_features(self, d, syms):
        """vol and beta for symbols outside the universe (e.g. held scanner names and leveraged ETFs) from their own
        data/bars (refreshed each morning for held names), so the risk line and the beta-weighted exposure don't fall
        back to "unknown" and 1.0 exactly where risk is largest (week 2: base's UVXY book showed +2.00x, about -6.5x
        in fact). Raw beta, no shrinkage: these are mostly ETFs and small caps. Marked outside=True: no mark comes from
        here (PaperWorld keeps its own rule for those)."""
        t, out = d - 1, {}
        if t < 60 or "SPY" not in self.k:
            return out
        spy = self.lr[self.k["SPY"], t - 59:t + 1]
        for s in syms:
            if s in self.k:
                continue
            try:
                doc = barsmod.load(s) or {}
            except (OSError, ValueError):
                continue
            bars = {str(x.get("date", ""))[:8]: x.get("close") for x in doc.get("bars") or []}
            have = [k for k in bars if k <= self.dates[t]]
            if not have or max(have) < self.dates[t - 5]:            # no bar in the last 5 sessions: say nothing
                continue
            c = np.array([bars.get(k) if bars.get(k) else np.nan for k in self.dates[t - 60:t + 1]], dtype=float)
            lr = np.diff(np.log(c))
            r20 = lr[-20:]
            vol = np.nanstd(r20, ddof=1) * 100 if np.sum(~np.isnan(r20)) >= 15 else np.nan
            ok = ~np.isnan(lr) & ~np.isnan(spy)
            beta = np.nan
            if ok.sum() >= 40 and np.var(spy[ok], ddof=1) > 0:
                beta = np.cov(spy[ok], lr[ok], ddof=1)[0, 1] / np.var(spy[ok], ddof=1)
            out[s] = {"vol": vol, "beta": beta, "outside": True}
        return out

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


# The content type may come as "<|constrain|>json" or as a plain " json" or " code": gpt-oss writes all three, and the plain
# forms were rejected as malformed until 2 Nov (103 of 122 malformed gpt-oss turns in weeks 2-3, every body valid JSON).
SEG = re.compile(r"^(?: to=(?P<r1>[^\s<]+))?<\|channel\|>(?P<ch>analysis|commentary|final)(?: to=(?P<r2>[^\s<]+))?"
                 r"(?:\s*<\|constrain\|>(?P<con>\w+)|\s+(?P<plain>json|code))?\s*<\|message\|>(?P<body>.*?)"
                 r"(?P<term><\|end\|>|<\|call\|>|<\|return\|>)$", re.S)


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


BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
_daily_cache = {}
STOP_TYPES = ("STP", "STP LMT", "TRAIL", "TRAIL LIMIT")


def _daily(sym):
    """{YYYYMMDD: (open, close)} from data/bars (ADJUSTED_LAST, so splits are already in), cached by the file's mtime."""
    p = BARS_DIR / f"{sym.replace(' ', '_')}.json"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    hit = _daily_cache.get(sym)
    if hit and hit[0] == mt:
        return hit[1]
    doc = json.loads(p.read_text(encoding="utf-8"))
    out = {str(b.get("date", ""))[:8]: (b.get("open"), b.get("close")) for b in doc.get("bars") or []
           if b.get("close") and b.get("volume", 1)}       # SHAZ's zero-volume placeholders jumped 92.50 -> 1.85 -> 31.70
    _daily_cache[sym] = (mt, out)
    return out


def book_risk(st, before, sessions=250):
    """One line of facts on the whole book (Clayton, 8 Nov: week 4 was lost on one overnight gap, into 6 Nov, by books
    net short one sector at 1.7-2.1x gross and mostly without stops, which the per-position lines didn't add up). The
    positions as they are now are held fixed over the last `sessions` sessions before `before` (YYYYMMDD) and priced with
    each day's close-to-close and close-to-open moves, so the names' correlation counts. Plus the share of gross with no
    working stop. A fact, not advice: None with no positions. Callable on a stored look's "state" for replays."""
    pos = [p for p in st["positions"] if p["quantity"] and p.get("close")]
    if not pos or st["value"] <= 0:
        return None
    cal = sorted(x for x in _daily("SPY") if x < before)[-(sessions + 1):]
    if len(cal) < 21:
        return None
    day, gap, thin, stale, priced = np.zeros(len(cal) - 1), np.zeros(len(cal) - 1), [], [], 0
    for p in pos:
        b, n, have = _daily(p["symbol"]), p["quantity"] * p["close"], 0
        for i in range(1, len(cal)):
            prev, cur = b.get(cal[i - 1]), b.get(cal[i])
            if prev and cur and prev[1] and cur[0] and cur[1]:
                day[i - 1] += n * (cur[1] / prev[1] - 1)
                gap[i - 1] += n * (cur[0] / prev[1] - 1)
                have += 1
        priced += have > 0
        if b and have and cal[-1] not in b:     # bars that stop early count the latest sessions as no move (TQQQ ends 19 Oct)
            stale.append(f"{p['symbol']} (no bars after {max(x for x in b if x < cal[-1])})")
        elif have < 0.8 * (len(cal) - 1):
            thin.append(p["symbol"])
    if not priced:                              # no held name has any history: no figures rather than "$0"
        return None
    stops = {}
    for o in st.get("orders") or []:
        if o.get("type") in STOP_TYPES and not o.get("cancel_pending"):
            stops[(o["symbol"], o["action"])] = stops.get((o["symbol"], o["action"]), 0) + abs(o["quantity"])
    gross = bare = 0.0
    for p in pos:
        q, v = abs(p["quantity"]), abs(p["quantity"]) * p["close"]
        cover = min(stops.get((p["symbol"], "SELL" if p["quantity"] > 0 else "BUY"), 0), q)
        gross, bare = gross + v, bare + v * (1 - cover / q)
    val = st["value"]
    lost = lambda x: f"{fmt_money(abs(min(x, 0.0)))} ({100 * abs(min(x, 0.0)) / val:.1f}% of the sleeve)"
    line = (f"The whole book, if held as it is: over the last {len(cal) - 1} sessions it would have lost {lost(day.min())} on its "
            f"worst day and {lost(gap.min())} on its worst overnight gap (close to next open); on 1 day in 20 it lost more than "
            f"{lost(float(np.percentile(day, 5)))}. {100 * bare / gross:.0f}% of its gross has no working stop, and a stop doesn't "
            f"limit a gap: it fills at the open price.")
    if thin:
        line += (f" ({', '.join(thin[:5])}{' and others' if len(thin) > 5 else ''} "
                 f"{'has' if len(thin) == 1 else 'have'} less history and count{'s' if len(thin) == 1 else ''} only where there is some.)")
    if stale:
        line += f" (Counted as unmoved since their last bar: {', '.join(stale[:5])}{' and others' if len(stale) > 5 else ''}.)"
    return line


_rets_cache = {}


def _rets(sym):
    """{YYYYMMDD: close-to-close return} on the name's own bars (data/bars), as the Kelly audit's b2_books.rets."""
    b = _daily(sym)
    hit = _rets_cache.get(sym)
    if hit and hit[0] is b:
        return hit[1]
    ds = sorted(b)
    out = {ds[i]: b[ds[i]][1] / b[ds[i - 1]][1] - 1 for i in range(1, len(ds)) if b[ds[i - 1]][1]}
    _rets_cache[sym] = (b, out)
    return out


def _betas(sym, before):
    """(beta to SPY, beta to SMH) on the last 120 daily returns before `before` (YYYYMMDD) that SPY and SMH also have,
    as b2_books.betas; under 60 such returns, (1, 1) as its fallback."""
    r, m, s = _rets(sym), _rets("SPY"), _rets("SMH")
    days = sorted(x for x in r if x < before and x in m and x in s)[-120:]
    if len(days) < 60:
        return 1.0, 1.0
    i, mm, ss = (np.array([d[x] for x in days]) for d in (r, m, s))
    return float(np.cov(i, mm)[0, 1] / np.var(mm, ddof=1)), float(np.cov(i, ss)[0, 1] / np.var(ss, ddof=1))


def shock_losses(notional, nav, before):
    """The Kelly audit's six single shocks (b2_books.py; B1's 15% cap used them) as a loss share of NAV, positive = loss,
    for a book of signed dollar positions {symbol: dollars}: SPY -10%/+10% and SMH -11%/+11% through each name's
    120-session beta, the largest short +30% and the largest long -30%."""
    w = {s: v / nav for s, v in notional.items() if v}
    bm = bs = 0.0
    for s, x in w.items():
        b1, b2 = _betas(s, before)
        bm, bs = bm + x * b1, bs + x * b2
    longs, shorts = [x for x in w.values() if x > 0], [-x for x in w.values() if x < 0]
    return {"SPY falls 10%": 0.10 * bm, "SPY rises 10%": -0.10 * bm, "SMH falls 11%": 0.11 * bs, "SMH rises 11%": -0.11 * bs,
            "its largest short rises 30%": 0.30 * max(shorts, default=0.0), "its largest long falls 30%": 0.30 * max(longs, default=0.0)}


def shock_cap_of(meta):
    """A sleeve's "shock_cap" if it is a number in (0, 1), else None (a string or 15 would break act() or mean 1500%;
    live.py logs a malformed one at each look)."""
    v = meta.get("shock_cap")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1 else None


def shock_rule(cap, st, before):
    """The state's line for a sleeve with a single-shock limit ("shock_cap" in sleeves.json, a sleeve term only Clayton
    sets): the rule, as PaperWorld.act applies it, and the book's worst shock now."""
    rule = (f"Single-shock limit ({100 * cap:g}% of the sleeve): an order that adds exposure is also rejected if the book after "
            f"it would lose more than {100 * cap:g}% of the sleeve in any one of these shocks, and more than it would before the "
            f"order: SPY falling or rising 10%, SMH falling or rising 11% (each name moving by its 120-session beta), its largest "
            f"short rising 30%, its largest long falling 30%. An order that only shrinks a position (without crossing zero) is "
            f"never rejected; one that crosses zero is checked.")
    if st["value"] <= 0 or not st["positions"]:
        return rule + " The book has no positions now." if st["value"] > 0 else rule
    loss = shock_losses({p["symbol"]: p["quantity"] * p["close"] for p in st["positions"]}, st["value"], before)
    name, worst = max(loss.items(), key=lambda kv: kv[1])
    return rule + f" Now the worst is {100 * max(worst, 0.0):.1f}% of the sleeve, if {name}."


def account_text(world, data, d, feats, header=None, live=None, book=False):
    st = world.state(d, feats, live) if live else world.state(d, feats)
    header = header or (f"Date: {data.iso(d)} ({date.fromisoformat(data.iso(d)).strftime('%A')}). "
                        f"Orders you submit now work from today's open.")
    line = shock = None
    if book:                                # a sleeve with "risk_line" in sleeves.json; a failure costs the line, not the look
        try:
            line = book_risk(st, data.iso(d).replace("-", ""))
        except Exception:                   # noqa: BLE001
            line = None
    cap = getattr(world, "SHOCK_CAP", None)
    if cap:                                 # the sleeve term is told whenever it applies; its figure may fail, the rule not
        try:
            shock = shock_rule(cap, st, data.iso(d).replace("-", ""))
        except Exception:                   # noqa: BLE001 - the rule alone, with no claim about the book
            shock = shock_rule(cap, dict(st, value=0.0), "")
    return render_account(st, feats, header, live, getattr(world, "MAX_GROSS", None), book=line, shock=shock), st


def render_account(st, feats, header, live=None, max_gross=2.0, book=None, shock=None):
    """The account part of a look's state text from a state dict (world.state(): value, cash, gross, net, beta_net,
    positions, orders with cancel_pending, events). Callable on a stored look file's "state" for replays. book: the
    book_risk() line, shown after the per-position risk lines. shock: shock_rule()'s line, after the margin line."""
    class _W:
        MAX_GROSS = max_gross
    world = _W
    lines = [header,
             f"Sleeve value {'now, at delayed last prices' if live else 'at last close'}: {fmt_money(st['value'])} (cash {fmt_money(st['cash'])}). "
             f"Gross exposure {st['gross']:.2f}x, net {st['net']:+.2f}x, beta-weighted {st['beta_net']:+.2f}x."]
    if getattr(world, "MAX_GROSS", None):
        cap, used = world.MAX_GROSS * max(st["value"], 0.0), st["gross"] * max(st["value"], 0.0)
        rule = ("Orders in one submit are checked in turn, each as if the earlier ones had filled: an order that only "
                "shrinks a position (without crossing zero) is never rejected for margin; one that adds to a position or "
                "crosses zero is rejected if it would take gross exposure past the limit.")
        if st["value"] <= 0:
            lines.append(f"Buying power: none (the sleeve's value is not positive). {rule}")
        elif st["gross"] > world.MAX_GROSS:
            lines.append(f"Over the margin limit: gross exposure is {st['gross']:.2f}x the sleeve, above {world.MAX_GROSS:g}x. "
                         f"Orders that add exposure are rejected until you bring it under {world.MAX_GROSS:g}x. {rule}")
        else:
            lines.append(f"Buying power: ${cap - used:,.0f} of gross exposure left before the {world.MAX_GROSS:g}x limit "
                         f"(${used:,.0f} used of ${cap:,.0f}). {rule}")
    if shock:
        lines.append(shock)
    if st["positions"]:
        lines.append(f"Positions (sym side shares avg_cost {'last' if live else 'close'} unrealized days_held):")
        for p in st["positions"]:
            side = "LONG" if p["quantity"] > 0 else "SHORT"
            lines.append(f"  {p['symbol']} {side} {abs(p['quantity'])} {p['avg_cost']:.2f} {p['close']:.2f} "
                         f"{p['unrealized']:+,.0f} {p['days']}")
        lines.append("  (To close a LONG, SELL its shares; to close a SHORT, BUY them back.)")
        lines.append("Risk of each position (share of the sleeve; a one-sigma day move from the vol column, in dollars):")
        for p in st["positions"]:                      # Clayton, 24 Oct: dollars at risk, which week 2's models never computed
            v = abs(p["quantity"]) * p["close"]
            share = f"{100 * v / st['value']:.0f}% of the sleeve" if st["value"] > 0 else "sleeve value not positive"
            vol = (feats.get(p["symbol"]) or {}).get("vol")
            if vol is not None and math.isfinite(vol):
                loss = v * vol / 100
                risk = (f"a {vol:.1f}% move = {fmt_money(loss)}" +
                        (f" = {100 * loss / st['value']:.1f}% of the sleeve" if st["value"] > 0 else ""))
            else:
                risk = "volatility unknown"
            lines.append(f"  {p['symbol']} {'LONG' if p['quantity'] > 0 else 'SHORT'}: {share}; {risk}")
        if book:
            lines.append(book)
    else:
        lines.append("Positions: none.")
    if st["orders"]:
        lines.append("Open orders (ref sym action qty type limit stop trail% tif status):")
        for o in st["orders"]:
            g = lambda k: "-" if o.get(k) in (None, 0) else f"{o[k]:g}"
            lines.append(f"  {o['ref']} {o['symbol']} {o['action']} {o['quantity']} {o['type']} {g('limit')} {g('stop')} "
                         f"{g('trail_percent')} {o['tif']} {'CANCEL-PENDING' if o.get('cancel_pending') else 'working'}")
        if any(o.get("cancel_pending") for o in st["orders"]):
            lines.append("  (CANCEL-PENDING: you already asked to cancel it and the broker has not confirmed yet. It can still "
                         "fill until it leaves this list. Cancelling it again does nothing.)")
    else:
        lines.append("Open orders: none.")
    # gpt-oss adds listed fills to positions that already hold them: 13 of its 16 oversized exits, 26 Oct to 2 Nov ("increase from 20 to 40")
    held = " (your positions and cash above already include these fills)" if any(e.startswith("filled ") for e in st["events"]) else ""
    lines.append(f"Since your last decision{held}: " + ("; ".join(st["events"]) if st["events"] else "nothing happened."))
    return "\n".join(lines)


def state_text(world, data, d, header=None, live=None, book=False):
    held = world.symbols_in_play()
    table, feats = data.table(d, extra=held, live=live)
    if hasattr(data, "outside_features"):
        feats = feats | data.outside_features(d, [s for s in held if s not in feats])
    acct, st = account_text(world, data, d, feats, header, live, book=book)
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
            side = "BUY" if f.side in ("BUY", "BOT") else "SELL"     # the same words as live (paper.events_since_last_look)
            self.events.append(f"filled {side} {f.shares:g} {f.symbol} @ {f.price:.2f} (commission ${f.commission:.2f}, ref {f.ref})")
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
        self.SHOCK_CAP = shock_cap_of(paper.sleeves().get(sleeve) or {})   # a sleeve term, set only by Clayton

    def symbols_in_play(self):
        b = self.paper.book(self.sleeve)
        return sorted(set(b["positions"]) | {o["symbol"] for o in self.paper.open_orders(self.gw, self.sleeve)})

    def state(self, d, feats, live=None):
        marks = {s: x["close"] for s, x in feats.items() if not x.get("outside")}
        last = self.data.dates[d - 1] if 0 < d <= len(self.data.dates) else None
        for s in self.paper.book(self.sleeve)["positions"]:   # held outside the universe: the last close from data/bars
            if s not in marks:                                 # (refreshed each morning for every held name), not cost
                try:
                    doc = barsmod.load(s) or {}
                except (OSError, ValueError):
                    doc = {}
                b0 = (doc.get("bars") or [{}])[-1]
                if b0.get("close") and b0.get("date") == last:
                    marks[s] = b0["close"]
        marks |= live or {}                                    # during the session: delayed last prices
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
        asked = self.paper.cancel_requested(self.sleeve)      # still open after our cancel: IBKR has not confirmed it
        sent = self.paper.placed_fields(self.sleeve)
        for o in self.paper.open_orders(self.gw, self.sleeve):
            orders.append({"ref": o["ref"], "symbol": o["symbol"], "action": o["action"], "quantity": o["quantity"],
                           "type": o["order_type"], "stop": o.get("aux"),
                           "trail_percent": (sent.get(o["ref"]) or {}).get("trailingPercent") if o["order_type"] == "TRAIL" else None,
                           # IBKR reports a lmtPrice (stop + 0.03) on plain STP orders; it is not a limit (23 Oct: Opus read
                           # its stops as STP LMT "on the wrong side"), so only order types with a limit show one
                           "limit": o.get("limit") if o["order_type"] in ("LMT", "STP LMT", "LOC") else None,
                           "tif": o.get("tif", "DAY"),
                           "cancel_pending": o.get("status") == "PendingCancel" or o["ref"] in asked})
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
            cap = getattr(self, "SHOCK_CAP", None)
            # The single-shock term, as the state's line states it: an add, or a flip through zero of any size, is checked;
            # a hedge that doesn't raise the worst shock passes however large the book's. The order's own name is priced
            # at the order's price on both sides, so a limit away from the mark can't make a hedge look worse.
            if p and cap and nav > 0 and (abs(q1) > abs(q0) or q0 * q1 < 0):
                try:
                    was = {k: q * (p if k == s else mark.get(k, p)) for k, q in pos.items()}
                    now = dict(was, **{s: q1 * p})
                    day = self.paper.et_today()
                    before = max(shock_losses(was, nav, day).values(), default=0.0)
                    name, after = max(shock_losses(now, nav, day).items(), key=lambda kv: kv[1])
                except Exception:                 # noqa: BLE001 - the check failing never blocks an order; the 2x term still holds
                    after = None
                    notes.append(f"{o['action']} {o['quantity']} {s}: the single-shock check failed, so it was not applied")
                if after is not None and after > cap and after > before + 1e-9:
                    notes.append(f"{o['action']} {o['quantity']} {s} {o['type']} rejected: the single-shock limit (if {name}, "
                                 f"the book after it would lose {100 * after:.1f}% of the sleeve, above the {100 * cap:g}% limit)")
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
            if r.get("error") or status is None or status in ("Cancelled", "Inactive", "ApiCancelled"):   # None: an error before any status (so far always 321)
                notes.append(f"{o['action']} {o['quantity']} {o['symbol']} {o['type']}: {status or 'refused'} {msg}".strip())
        self.paper.add_notes(self.sleeve, notes)
        return notes
