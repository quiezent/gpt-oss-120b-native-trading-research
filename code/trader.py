"""The trader: one decision per trading day, the same code in simulation and on the paper account.

Before each session the model sees its sleeve (cash, positions, open orders, what happened since its
last decision) and a market table built from daily bars through the previous close, and answers with
exactly one `submit` call: orders to place and order refs to cancel. Software places exactly what it
writes. SimWorld runs it on sim.Market; PaperWorld runs it on a sleeve of the IBKR paper account.
"""

import json
import math
import re
import sys
import time as _time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import bars as barsmod
import sim
import tools

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
KELLY_LIMITS = "- The sleeve's limits (2x gross, and the single-shock limit if your state shows one) cap the Kelly size too.\n"
# The gross-exposure term as the system prompt states it, and what replaces it for a sleeve with no software limit
# ("max_gross": null in sleeves.json; Clayton, 14 Nov 2026: "I prefer less intervention from the software framework or a
# hard limit unless IBKR restricts it").
GROSS_RULE = "Gross exposure may not exceed 2x the sleeve."
NO_GROSS_RULE = ("Software sets no limit on your exposure: IBKR's own margin rules apply, to the whole paper account that your "
                 "sleeve is part of, and an order IBKR refuses is reported at your next look.")


S1_NEWS, S1_NONE = "1. Read today's market news (tape) and look at your positions.", "1. Look at your positions and the market table."
S3_SCAN = "3. Test it with the scanner: choose the scan and, if useful, the industry that would find stocks fitting your hypothesis."
S3_ML = ("3. Test it with the scanner and the forecast tool (machine-learning forecasts; its description gives each model's "
         "out-of-sample track record, so weigh it accordingly): choose the scan and, if useful, the industry that would find "
         "stocks fitting your hypothesis.")
S4_NEWS, S4_NONE = "4. Check every candidate with news and quote;", "4. Check every candidate with quote;"
# ML4 (15 Nov 2026): its description gives one accuracy line a column and the full record only on request
# (forecast(track_record=true)), so S3_ML's "its description gives each model's out-of-sample track record" would be false
# for a sleeve on it. S3_ML itself is what ML2's and ML3's sleeves read and is in ML2's manifest: it does not change.
S3_ML4 = ("3. Test it with the scanner and the forecast tool (machine-learning forecasts): choose the scan and, if useful, the "
          "industry that would find stocks fitting your hypothesis.")
S3_BY_TOOL = {"ML4": S3_ML4}          # a sleeve's ml_tool -> its step 3; any other version reads S3_ML


def live_system(size, today, ml=False, news=True, kelly=False, max_gross=2.0, ml_tool=None, own_limit=None):
    """SYSTEM_LIVE for a sleeve of this size, with the target pace (Clayton, 24 Oct: a pace, not a weekly dollar goal
    with a deadline). ml: the sleeve has the forecast tool; news=False: the training simulator, which has no news;
    kelly: the Kelly sizing paragraph after the pace (Clayton, 9 Nov: "a kelly improved prompt"; sleeves.json
    "kelly_prompt"). It is guidance the model may weigh; nothing enforces it. max_gross: the sleeve's gross-exposure
    term (max_gross_of): 2.0 is the text as it always was, another number changes the figure, None says there is no
    software limit. ml_tool: the sleeve's forecast tool version (sleeves.json "ml_tool"); only ML4 changes step 3
    (S3_ML4), every other value reads S3_ML as before. own_limit: the sleeve's own exposure limit (own_limit_of; only with
    max_gross None): its term (own_limit.term) in place of NO_GROSS_RULE; None, the default, is the text as before."""
    text = SYSTEM_LIVE
    if kelly:
        assert text.count(PACE_END) == 1
        text = text.replace(PACE_END, PACE_END.replace("\n\nOrder types:", "\n\n" + KELLY + "\n\nOrder types:"))
    if max_gross != 2.0:
        assert text.count(GROSS_RULE) == 1
        text = text.replace(GROSS_RULE, NO_GROSS_RULE if max_gross is None
                            else GROSS_RULE.replace("2x", f"{max_gross:g}x"))
        if kelly:                           # the Kelly paragraph names the 2x limit too
            assert text.count(KELLY_LIMITS) == 1
            text = text.replace(KELLY_LIMITS, "" if max_gross is None
                                else KELLY_LIMITS.replace("2x", f"{max_gross:g}x"))
    if max_gross is None and own_limit:     # 18 Nov 2026 (Clayton: "Yes, text only"): the sleeve's own limit, told, not enforced
        import own_limit as _ol
        assert text.count(NO_GROSS_RULE) == 1
        text = text.replace(NO_GROSS_RULE, _ol.term(own_limit))
    if ml:
        assert text.count(S3_SCAN) == 1
        text = text.replace(S3_SCAN, S3_BY_TOOL.get(ml_tool, S3_ML))
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


def live_steps(ml=False, news=True, probs=False):
    """The steps line of a live look. probs=True is the text through Fri 13 Nov 2026, which also asked for p_fill and
    p_win on every order (for replays of those looks)."""
    s = LIVE_STEPS_PROBS if probs else LIVE_STEPS
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
              "(4) call news, quote and stats for your candidates; (5) call submit, with notes for your next look.\n\n")
# Through Fri 13 Nov 2026 step 5 also asked for the order probabilities (Clayton, 14 Nov, on p_fill and p_win: "Remove").
LIVE_STEPS_PROBS = LIVE_STEPS.replace("with notes for your next look.", "with notes for your next look and your honest p_fill and "
                                      "p_win on every order.")

from tools import CLOSE_DESC, NOTES_DESC, P_FILL_DESC, P_WIN_DESC, PROBS_DESC, TRAIL_DESC   # shared with mcp_server.py

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
def tool_live(probs=False):
    """The live submit form. From Mon 16 Nov 2026 it does not ask for p_fill and p_win (Clayton, 14 Nov: "Remove"; stated
    p_win was no better than a coin, Brier 0.275 against 0.248). probs=True is the form through Fri 13 Nov, for replays of
    those looks. An order that still carries the two fields is accepted as before: clean_order keeps them, they are
    logged, and they never reach the broker (models trained on the old form write them). The new form also states the
    trailing-stop unit (TRAIL_DESC), in the field and in the tool's own description, since the gpt-oss renderer drops an
    array item field's description (Clayton, 14 Nov: text B only; probs=True stays the form of fd95e2a)."""
    order = TOOL["parameters"]["properties"]["place"]
    fields = {k: v for k, v in order["items"]["properties"].items() if probs or k not in ("p_fill", "p_win")}
    if not probs:
        fields["trail_percent"] = {**fields["trail_percent"], "description": TRAIL_DESC}
        # close-only (Clayton, 17 Nov), text B only: in the field and in the tool's own description (the renderer drops
        # an array item field's description, as for TRAIL_DESC)
        fields["close_only"] = {"type": "boolean", "description": CLOSE_DESC}
    place = {**order, "items": {**order["items"], "properties": fields}}
    return {**TOOL, "description": "Submit this look's decision: orders to place, open orders to cancel, and notes for your next "
                                   "look. Call it exactly once. Empty lists mean no change."
                                   + (" " + PROBS_DESC if probs else " " + TRAIL_DESC + " " + CLOSE_DESC),
            "parameters": {**TOOL["parameters"], "properties": {**TOOL["parameters"]["properties"], "place": place,
                                                                "notes": {"type": "string", "description": NOTES_DESC}}}}


TOOL_LIVE = tool_live()
TOOL_LIVE_PROBS = tool_live(probs=True)      # the form through Fri 13 Nov 2026


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


def max_gross_of(meta):
    """A sleeve's gross-exposure limit as a multiple of its value: 2.0 unless sleeves.json has "max_gross" (a sleeve term
    only Clayton sets). A number above 0 is the limit; null is no software limit, so only IBKR's own checks on the whole
    paper account can refuse an order. Anything else is malformed and keeps 2.0 (live.py logs it at each look)."""
    if "max_gross" not in meta:
        return 2.0
    v = meta["max_gross"]
    if v is None:
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 2.0


def text_of(meta):
    """Which text a sleeve reads: "A" is the text of Sat 14 Nov 2026 (main fd95e2a: the order form with p_fill and p_win,
    no order echo, no cash clause, no account margin line, no check_orders tool), kept on one control sleeve by
    "text": "A" in sleeves.json; "B", the default, is the text from Mon 16 Nov. Clayton approved a second no-tool base
    twin for this A/B on 14 Nov. Changes that alter nothing a model reads go to both.
    The one agreed exception (frozen 15 Nov, monday_changes.md item e): three fixes that repair a malformed call go to both
    texts, so text A's tool results can differ from fd95e2a's for those calls only: agent.unwrap_args (a call wrapped in an
    {"name", "arguments"} envelope runs as the call it wraps), tools.article_ref (a bare or lower-case article id is given
    its provider and upper-cased) and tools.symbol_list ("symbols" given as one string is read as a list of names, not as
    letters). Why both: at fd95e2a those calls ran as garbage (no symbols, the letters of a ticker, a dead article id, or
    a submit rejected as malformed), which no text means to say; leaving the control on them would measure the bug, not
    text B. A well-formed call reads the same in both texts.
    A fourth (16 Nov 2026; the Trading Model Researcher's figures of 15 Nov): agent.gptoss_look's context reserve, for every
    gpt-oss sleeve, text A too. Within RESERVE (2,500) tokens of the 32,768-token context it opens the submit instead of a
    free turn and does not run a tool call whose result would leave less than the reserve (one line says so). A free
    turn that ends in a tool call and leaves less than SUBMIT_ROOM (1,500) is dropped, and the submit is opened at the
    prompt as it was before that turn; a tool's result is cut so that SUBMIT_ROOM is left for the decision. Why both:
    it acts only where a look would otherwise end with no decision ("context full") or with a decision short of room; a
    look whose room never falls under the reserve runs step for step as at fd95e2a.
    Not a text difference, but a change from Saturday's look: the research-call limit is a sleeve setting (max_calls_of),
    and Clayton set 15 on every sleeve from Mon 16 Nov, the text-A control included ("Every sleeve to 15", 15 Nov), so text
    A reads "(research call k of 15)" where fd95e2a read "of 12". Both texts run at the same limit. The forced-submit
    settings (live.forced_mode_of, final_line_of) are not set on the control: it keeps the plain opened submit and the old
    limit line, the Researcher's recommendation, which Clayton followed.
    A fifth that goes to both texts (17 Nov; Clayton, 16 Nov: "Yes, build it"): the missing-price re-ask (agent.gptoss_look,
    tools.missing_prices). A submit holding an order whose type needs a price field it lacks is answered once a look and
    the next submit is taken as written; it repairs a malformed call, as the first three do, and a submit with every price
    field reads and runs as at fd95e2a. IBKR's margin warning (margin_warning) is in text B's account line only: the
    control has no account line.
    A sixth that changes text A too (18 Nov 2026): the sleeve's own exposure limit (own_limit.py; sleeves.json "own_limit"),
    on the eight gpt-oss sleeves, the control included. Clayton, asked whether it should go live on those eight, the
    control named: "Yes, text only". The control gets the system term and the state's limit line (with the positions
    sentence when over); the account sentence and the order check's line are text B's, which alone has those parts.
    A seventh (18 Nov 2026, from the 10:30 look): the last fill line of each symbol in "Since your last decision" ends
    with the sleeve's position now (FILL_POSITION, fill_position_note), on all ten sleeves, the control included, and the
    limit line counts and names the resting orders that would open or add (own_limit.STATE_RESTING). The Researcher's
    decision: Clayton had answered its question about the control "Your decision. You're the trainer ...". Why: at 09:00
    the control and cand2 read Tuesday's covering fills as longs they did not hold.
    An eighth (from the first look of Thu 19 Nov 2026): the per-look limit pieces of the sixth and seventh come out of
    both texts (OWN_LIMIT_LOOK False): the state's limit line, the account sentence and the check's line. The system
    term and the fill-line note stay. Clayton, 18 Nov: "Don't decide for the trader model."; the Researcher's change.
    A ninth that goes to both texts (19 Nov), of the kind of the first three: a stop type spelled out ("STOP", "STOP
    LIMIT") is read as the tool's STP or STP LMT (tools.TYPE_NAMES), so its stop price reaches IBKR; at fd95e2a such an
    order went out with no stop price and IBKR refused it (cand1, 11:30 on 18 Nov). An order typed as listed runs as before."""
    return "A" if meta.get("text") == "A" else "B"


def own_limit_of(meta):
    """A sleeve's own exposure limit, as a multiple of its value, told to the model and never enforced (own_limit.py;
    Clayton, Wed 18 Nov 2026: "Yes, text only"): sleeves.json "own_limit", a number above 0, on a sleeve whose
    "max_gross" is null (no software limit). None otherwise: the sleeve reads the text it read before. A sleeve term
    only Clayton sets."""
    v = meta.get("own_limit")
    if "max_gross" not in meta or meta["max_gross"] is not None:
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < float("inf") else None


def own_limit_bad(meta):
    """Why a sleeve's "own_limit" is ignored, or None: there but not a number above 0, or on a sleeve with a software
    limit (max_gross not null)."""
    if "own_limit" not in meta or meta["own_limit"] is None:
        return None
    v = meta["own_limit"]
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < float("inf")):
        return f"own_limit {v!r} is not a finite number above 0, so the sleeve reads no exposure limit"
    if "max_gross" not in meta or meta["max_gross"] is not None:
        return f"own_limit {v!r} is ignored: it applies only with max_gross null (no software limit)"
    return None


def max_gross_bad(meta):
    """True if the sleeve's "max_gross" is there but is neither null nor a number above 0."""
    v = meta.get("max_gross")
    return "max_gross" in meta and v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)


def max_calls_of(meta):
    """A sleeve's research-call limit for a look (agent.gptoss_look's max_calls): tools.MAX_CALLS (12) unless sleeves.json
    gives "max_calls", a whole number from 1 to tools.MAX_CALLS_TOP. Clayton, 15 Nov 2026, relayed by the Trading Model
    Researcher: "Yes lets go for 15 and pass the info to Trader Training to implement."; then to me, asked whether anything
    else moves to 15 (the text-A control and the Fable sleeves had stayed at 12): "Every sleeve to 15". A gpt-oss sleeve reads
    its limit only in the counter after each research call ("(research call 3 of 15)") and in the research-limit line; a
    Claude sleeve also in its system prompt ("at most 15 calls", agent.claude_look) and the MCP server counts to it
    (TRADER_MAX_CALLS). Anything malformed keeps tools.MAX_CALLS (live.py logs it at each look)."""
    return tools.MAX_CALLS if "max_calls" not in meta or max_calls_bad(meta) else meta["max_calls"]


def max_calls_bad(meta):
    """True if the sleeve's "max_calls" is there but is not a whole number from 1 to tools.MAX_CALLS_TOP."""
    if "max_calls" not in meta:
        return False
    v = meta["max_calls"]
    return not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= tools.MAX_CALLS_TOP)


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


def account_text(world, data, d, feats, header=None, live=None, book=False, cash_clause=False):
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
    account = getattr(world, "ACCOUNT", None) if getattr(world, "TEXT", "A") == "B" else None   # text B only (Clayton, 14 Nov)
    if account is not None:
        st["account"] = account             # kept in the look record's state, for replays
    return render_account(st, feats, header, live, getattr(world, "MAX_GROSS", None), book=line, shock=shock,
                          cash_clause=cash_clause, account=account, log=getattr(world, "log", None),
                          own_limit=look_limit(getattr(world, "OWN_LIMIT", None)),
                          fill_position=FILL_POSITION), st


# Whether the sleeve's own exposure limit is worked out for it at each look: the state's limit line (own_limit.state_line,
# with its room, positions-by-size and resting-orders words), the sentence after the account line (account_sentence) and
# the last line of the check_orders answer (own_limit.check_block). False from Thu 19 Nov 2026: the limit is the system
# text's term alone (own_limit.term, unchanged), and the sleeve reads its own gross on the sleeve-value line. Clayton, 18
# Nov, to the Trading Model Researcher: "Don't decide for the trader model." and "The models need to trade themselves and
# you only adjust the system prompt to guide the boundaries such as margin this time. Don't interfere by trading with the
# 30 minutes look." On 18 Nov the over-limit line's positions by size, with a running total, led the five sleeves over
# their limit at 09:00 to close 28 of the 30 positions it named. The Researcher's one change for 19 Nov: keep the term,
# remove every per-look piece, add nothing in their place. True gives the text of 18 Nov (303b41c2) for replays.
OWN_LIMIT_LOOK = False


def look_limit(limit):
    """The own limit the look's state and check work out (render_account, checks.context): limit when OWN_LIMIT_LOOK is
    on, else None (the state and check as before 18 Nov). The system text's term (live_system) does not go through this."""
    return limit if OWN_LIMIT_LOOK else None


# Whether fill lines end with the sleeve's position now (fill_position_note), on every sleeve, text A and text B alike.
# The Researcher, 18 Nov (about 09:25 ET), after the 09:00 look, where base-90k and cand2 read Tuesday's covering fills
# ("filled BUY 100 NVDA ...") as longs and placed sells that would open shorts: "end every fill line with the position
# now, in words, after the ref" (about 10:00 ET: on the last fill line of each symbol only). About 09:45 ET, relaying
# Clayton ("Your decision. You're the trainer ..."): the text-A control gets it too, "build it for all ten sleeves", since
# the fact is about the sleeve's own book, not a rule. On from the 10:30 look of 18 Nov, after the Researcher's check of
# the built text on that day's records ("Both").
FILL_POSITION = True
_FILL = re.compile(r"^filled (?:BUY|SELL) \S+ (.+?) @ ")


def fill_position_note(event, held):
    """" (you now hold no NVDA)", " (you are now short 22 MU)" or " (you are now long 3 GLD)" for a fill line, from held
    ({symbol: shares}, the state's positions: after all the fills); "" for any other line."""
    m = _FILL.match(event)
    if not m:
        return ""
    s = m.group(1)
    q = held.get(s, 0)
    if not q:
        return f" (you now hold no {s})"
    return f" (you are now {'long' if q > 0 else 'short'} {abs(q)} {s})"     # as the positions table prints it


# The frozen wording of 15 Nov 2026 (monday_changes.md "FROZEN 15 Nov", item c): "your sleeve has spent more than its cash" is
# out, read as a push to hold back (Clayton, 15 Nov: "be careful to not let do nothing creep in").
CASH_NEGATIVE = " Negative cash is already subtracted in the sleeve value."
CASH_SHORTS = " Cash includes the proceeds of your short sales. Your short positions are subtracted in the sleeve value."


def cash_note(st):
    """Text B's clause after "(cash $C)." (Clayton, 14 Nov 2026; from 16 Nov): what a negative cash, or a cash above the
    sleeve's value (short proceeds), already means for the value. Empty otherwise."""
    return CASH_NEGATIVE if st["cash"] < 0 else CASH_SHORTS if st["cash"] > st["value"] else ""


_ET = ZoneInfo("America/New_York")


def _acct_num(a, k):
    try:
        v = a.get(k)
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError, AttributeError):
        return None


def account_line(info, st, live=None):
    """Text B's whole-account line (calculator_tools_spec_v2.md section 3, frozen 15 Nov 2026; Clayton, 14 Nov; from 16 Nov),
    after the sleeve-value line. info: {"account": gw.account()'s row or None if it was not read, "sleeves": the number in the
    sleeves file, "total": their values summed, "unheld": optional, live.unheld_gross()}, read once a look before the looks, so every sleeve sees the same figures
    and their time. Every figure is the whole paper account's, IBKR's own, labelled so; the sleeve's shares are its value
    (at delayed last prices, or the last close) and gross over the account's (IBKR's prices). No BuyingPower: it is the same
    fact as (equity with loan - initial margin) / 0.25, and as a bare dollar figure the one most likely read as the sleeve's.
    The SMA floor is equity with loan minus IBKR's RegTMargin (50% of gross), else minus half of gross."""
    a = (info or {}).get("account")
    if not isinstance(a, dict) or not a:
        return "Whole paper account: IBKR's figures were not read for this look."
    g = lambda k: _acct_num(a, k)
    money = lambda k, name: f"{name} not reported" if g(k) is None else f"{name} ${g(k):,.0f}"
    when = ""
    try:
        t = datetime.strptime(str(a.get("read_time")), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(_ET)
        when = f" at {t:%H:%M:%S} ET"
    except (TypeError, ValueError):
        pass
    n, total = info.get("sleeves"), info.get("total")
    shared = (f"shared by the {n} sleeves" + (f", whose values total {fmt_money(total)}" if total is not None else "")
              if n else "shared by the sleeves")
    ewl, regt, gross, nl, sma, cushion = (g(k) for k in ("EquityWithLoanValue", "RegTMargin", "GrossPositionValue",
                                                          "NetLiquidation", "SMA", "Cushion"))
    floor = ewl - regt if ewl is not None and regt is not None else ewl - 0.5 * gross if ewl is not None and gross is not None else None
    sma_part = ("SMA not reported" if sma is None else f"SMA ${sma:,.0f}") + (
        f", never below equity with loan minus half of gross (${floor:,.0f} now)" if floor is not None else "")
    parts = [money("NetLiquidation", "net liquidation"), money("EquityWithLoanValue", "equity with loan"),
             money("InitMarginReq", "initial margin"), money("ExcessLiquidity", "excess liquidity"),
             "cushion not reported" if cushion is None else f"cushion {100 * cushion:.0f}%", sma_part, money("GrossPositionValue", "gross")]
    # 16 Nov 2026: positions the account lists that no sleeve's ledger holds (live.unheld_gross; the reset's residue), only
    # when there are some; otherwise the line is unchanged
    unheld = info.get("unheld")
    if isinstance(unheld, (int, float)) and not isinstance(unheld, bool) and round(unheld) >= 1:
        parts[-1] += f"; ${round(unheld):,.0f} of the gross is in positions no sleeve holds"
    line = f"Whole paper account ({shared}; IBKR's figures{when}): " + "; ".join(parts) + "."
    if nl and nl > 0:
        mine = sum(abs(p["quantity"]) * p["close"] for p in st.get("positions") or [])
        line += (f" Your sleeve's value ({'at delayed last prices' if live else 'at last close'}) is {100 * st['value'] / nl:.1f}% "
                 "of the account's net liquidation (at IBKR's prices)"
                 + (f"; its gross exposure is {100 * mine / gross:.1f}% of the account's gross." if gross and gross > 0 else "."))
        line += margin_warning(a, st, mine)
    return line


def margin_warning(a, st, mine):
    """IBKR's margin warning, relayed to the sleeves more levered than the whole account (Clayton, 16 Nov: "We need to relay
    the IBKR warning to the most levered sleeves, at least base-90k-b and cand2"; on Mon 16 Nov SMA was below zero for
    hours before the close (the loop logged it from 13:06 ET at least), IBKR refused cand1's order for margin between 15:00
    and 15:30, and at 15:51 ET IBKR closed positions in all ten sleeves). Shown when IBKR's figures put the account in IBKR's own danger zone, SMA below zero or excess liquidity
    under 10% of net liquidation, to a sleeve whose gross is a larger multiple of its value than the account's gross is of
    its net liquidation (on Mon 16 Nov at 15:30: base-90k-b 14.4x, cand2 8.3x, base-ml, base-90k-ml, cand2-ml and cand1,
    against 3.45x; the text-A control has no account line). A sleeve whose value is at or below zero with any gross is
    the most levered of all and gets it too. Facts only: IBKR's figures, its rule, the day it acted, and the two multiples.
    "" otherwise."""
    g = lambda k: _acct_num(a, k)
    nl, xl, sma, gross = g("NetLiquidation"), g("ExcessLiquidity"), g("SMA"), g("GrossPositionValue")
    value = st.get("value")
    if not nl or nl <= 0 or not gross or not isinstance(value, (int, float)) or not mine or mine <= 0:
        return ""
    danger = (sma is not None and sma < 0) or (xl is not None and xl < 0.10 * nl)
    acct_x = gross / nl
    mine_x = mine / value if value > 0 else None          # None: value at or below zero, the most levered of all
    if not danger or (mine_x is not None and mine_x <= acct_x):
        return ""
    # Clayton, Tue 17 Nov ~11:50 ET (the gpt-oss sleeves went on adding after the first wording): "Change the wording to
    # 'Your gross exposure is 7.2x its value; the your account's maximum is 2.9x its net liquidation.'" ("the your" read
    # as "your"). The 2.9x is the account's own gross over its net liquidation at IBKR's read, which he calls its maximum.
    own = (f"Your gross exposure is {mine_x:.1f}x its value" if mine_x is not None
           else f"Your gross exposure is ${mine:,.0f} with its value at or below zero")
    figs = "; ".join(x for x in (f"SMA ${sma:,.0f}" if sma is not None else "", f"excess liquidity ${xl:,.0f}" if xl is not None else "") if x)
    return (f" IBKR margin warning ({figs}): IBKR closes positions in this account, choosing which itself, when excess "
            "liquidity is below zero, and near the close (from 15:50 ET on a full trading day) when SMA is below zero; it did "
            f"so at 15:51 ET on Mon 16 Nov 2026, in every sleeve. {own}; your account's maximum is {acct_x:.1f}x its net "
            "liquidation.")


def render_account(st, feats, header, live=None, max_gross=2.0, book=None, shock=None, cash_clause=False, account=None,
                   log=None, own_limit=None, fill_position=False):
    """The account part of a look's state text from a state dict (world.state(): value, cash, gross, net, beta_net,
    positions, orders with cancel_pending, events). Callable on a stored look file's "state" for replays. book: the
    book_risk() line, shown after the per-position risk lines. shock: shock_rule()'s line, after the margin line.
    cash_clause: text B's cash_note() after the cash figure; False (the default) is the text through 14 Nov. account:
    text B's account_line() info, shown after the sleeve-value line; None (the default) is the text through 14 Nov.
    log: where a failure of the account line is reported (live.py's log, so it reaches live.log; review, 15 Nov), else
    stderr. own_limit: the sleeve's own exposure limit (own_limit_of; 18 Nov): its line (own_limit.state_line) straight
    after the sleeve-value line, and own_limit.ACCOUNT_SENTENCE at the end of the account line when that line carries
    figures; None, the default, is the text as before. fill_position: each fill line ends with the sleeve's position
    now (fill_position_note); False, the default, is the text as before."""
    class _W:
        MAX_GROSS = max_gross
    world = _W
    lines = [header,
             f"Sleeve value {'now, at delayed last prices' if live else 'at last close'}: {fmt_money(st['value'])} (cash {fmt_money(st['cash'])})."
             + (cash_note(st) if cash_clause else "") +
             f" Gross exposure {st['gross']:.2f}x, net {st['net']:+.2f}x, beta-weighted {st['beta_net']:+.2f}x."]
    limit_shown = False
    if own_limit:                           # the sleeve's own limit, told and not enforced (Clayton, 18 Nov: "Yes, text only")
        try:                                # a failure costs the line, never the look (as the account line)
            import own_limit as _ol
            lines.append(_ol.state_line(st, own_limit, _ol.prices_of(feats, live), log))
            limit_shown = True
        except Exception as e:              # noqa: BLE001
            (log or (lambda m: print(m, file=sys.stderr)))(f"exposure-limit line left out: {type(e).__name__}: {e}"[:300])
    if account is not None:
        try:                                # a failure costs the line, never the look
            aline = account_line(account, st, live)
            if limit_shown and aline.startswith("Whole paper account ("):   # the line with IBKR's figures; the sentence
                import own_limit as _ol2                                    # points at the limit line, so only with it
                aline += _ol2.account_sentence(aline)
            lines.append(aline)
        except Exception as e:              # noqa: BLE001
            (log or (lambda m: print(m, file=sys.stderr)))(f"account line left out: {type(e).__name__}: {e}"[:300])
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
                         f"{g('trail_percent')} {o['tif']} {'CANCEL-PENDING' if o.get('cancel_pending') else 'working'}"
                         + (" close_only" if o.get("close_only") else ""))
        if any(o.get("cancel_pending") for o in st["orders"]):
            lines.append("  (CANCEL-PENDING: you already asked to cancel it and the broker has not confirmed yet. It can still "
                         "fill until it leaves this list. Cancelling it again does nothing.)")
            if any(o.get("cancel_pending") and o.get("close_only") for o in st["orders"]):   # 17 Nov: text B only (only it has
                lines.append("  (A close_only order can also be CANCEL-PENDING because the close_only check cancelled it.)")   # them)
    else:
        lines.append("Open orders: none.")
    # gpt-oss adds listed fills to positions that already hold them: 13 of its 16 oversized exits, 26 Oct to 2 Nov ("increase from 20 to 40")
    held = " (your positions and cash above already include these fills)" if any(e.startswith("filled ") for e in st["events"]) else ""
    events = st["events"]
    if fill_position and events:            # the position now after each fill line (the Researcher, 18 Nov); a failure
        try:                                # costs the notes only, never the look (the review of 18 Nov)
            now = {p["symbol"]: p["quantity"] for p in st.get("positions") or []}
            last = {}                       # the last fill line of each symbol carries the note, so "now" is true where
            for i, e in enumerate(events):  # it stands (the Researcher, 18 Nov: on every line, cand1-ml's four AMD buys
                m = _FILL.match(e)          # each read "long 30 AMD")
                if m:
                    last[m.group(1)] = i
            marks = set(last.values())
            events = [e + fill_position_note(e, now) if i in marks else e for i, e in enumerate(events)]
        except Exception as e:              # noqa: BLE001
            events = st["events"]
            (log or (lambda m: print(m, file=sys.stderr)))(f"fill-position notes left out: {type(e).__name__}: {e}"[:300])
    lines.append(f"Since your last decision{held}: " + ("; ".join(events) if events else "nothing happened."))
    return "\n".join(lines)


def state_text(world, data, d, header=None, live=None, book=False, cash_clause=False):
    held = world.symbols_in_play()
    table, feats = data.table(d, extra=held, live=live)
    if hasattr(data, "outside_features"):
        feats = feats | data.outside_features(d, [s for s in held if s not in feats])
    acct, st = account_text(world, data, d, feats, header, live, book=book, cash_clause=cash_clause)
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
    case and spacing, and a stop type spelled out read as the tool's own (tools.order_type: "STOP" is STP, "STOP
    LIMIT" and "STOP LMT" are STP LMT; 19 Nov): a missing or non-numeric field is an error, the same in simulation and
    live. p_fill and p_win pass through as floats in [0, 1], else None; they never cause an error and never reach the
    broker."""
    if not isinstance(o, dict):
        return None, "order is not an object"
    sym = str(o.get("symbol", "")).upper().strip()
    act = str(o.get("action", "")).upper()
    typ = tools.order_type(o.get("type", ""))
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
    co = o.get("close_only")                     # text B's flag (17 Nov): only an explicit true; never an error
    out["close_only"] = co is True or (isinstance(co, str) and co.strip().lower() == "true")
    return out, None


def order_echo(o, price, value, held=0):
    """Text B's line for an order the broker accepted (Clayton, 14 Nov 2026; from 16 Nov; PaperWorld.act writes none for an
    order paper.place returned an error for, or with no status, or Cancelled, Inactive or ApiCancelled: that order has its
    refusal note), shown at the next look after the refusals: its size in dollars at the quote's mark and as a multiple of
    the sleeve's value at placement; for a TRAIL its trail in dollars; for an order on the closing side of a position
    (held: the shares before this submit) larger than it, the position's size. Facts only. None with no price or a sleeve value not above 0."""
    if not price or not value or value <= 0:
        return None
    dollars = o["quantity"] * price
    line = (f"{o['action']} {o['quantity']} {o['symbol']} {o['type']}: about ${dollars:,.0f} at the last price, "
            f"{dollars / value:.2f}x your sleeve's value")
    if o["type"] == "TRAIL" and o.get("trail_percent") is not None:
        line += f", trails ${price * o['trail_percent'] / 100:.2f} behind the price"
    closing = (o["action"] == "SELL" and held > 0) or (o["action"] == "BUY" and held < 0)
    if closing and o["quantity"] > abs(held):
        n = abs(held)
        line += f", your position is {int(n) if float(n).is_integer() else n} shares"     # the book keeps some as floats
    return line + "."


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
        meta = paper.sleeves().get(sleeve) or {}
        self.SHOCK_CAP = shock_cap_of(meta)   # a sleeve term, set only by Clayton
        self.MAX_GROSS = max_gross_of(meta)   # the same: 2.0 unless Clayton set "max_gross"; None = no software limit
        self.OWN_LIMIT = own_limit_of(meta)   # the sleeve's own limit, told and never enforced (18 Nov); None = not told
        self.TEXT = text_of(meta)             # "B" adds the order echo to the next look's notes; "A" (the control) does not

    log = None   # live.py sets its log(): a failure of a text-B feature in act() costs that feature and logs a line
    ACCOUNT = None   # live.py sets text B's account_line() info, read once a look (account_text shows it for text B only)
    OWN_LIMIT = None   # __init__ sets the sleeve's own_limit_of(meta)

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
        closers = self.paper.close_only_refs(self.sleeve)     # 17 Nov: shown as "close_only" after the status
        for o in self.paper.open_orders(self.gw, self.sleeve):
            orders.append({"ref": o["ref"], "symbol": o["symbol"], "action": o["action"], "quantity": o["quantity"],
                           "type": o["order_type"], "stop": o.get("aux"),
                           "trail_percent": (sent.get(o["ref"]) or {}).get("trailingPercent") if o["order_type"] == "TRAIL" else None,
                           # IBKR reports a lmtPrice (stop + 0.03) on plain STP orders; it is not a limit (23 Oct: Opus read
                           # its stops as STP LMT "on the wrong side"), so only order types with a limit show one
                           "limit": o.get("limit") if o["order_type"] in ("LMT", "STP LMT", "LOC") else None,
                           "tif": o.get("tif", "DAY"),
                           "cancel_pending": o.get("status") == "PendingCancel" or o["ref"] in asked,
                           **({"close_only": True} if o["ref"] in closers else {}),
                           # not shown in the text: for the limit line's resting figure (own_limit.resting_figure; 18 Nov),
                           # IBKR's own trail stop and what is left of a partly filled order, where the row has them
                           **({"trail_stop": o["trail_stop"]} if o["order_type"] == "TRAIL" and o.get("trail_stop") is not None else {}),
                           **({"remaining": o["remaining"]} if isinstance(o.get("remaining"), (int, float))
                              and not isinstance(o.get("remaining"), bool) else {})})
        ev = self.paper.events_since_last_look(self.gw, self.sleeve, commit=not self.dry_run) + self.paper.take_notes(self.sleeve, commit=not self.dry_run)
        return {"value": value, "cash": b["cash"], "positions": pos, "orders": orders, "events": ev,
                "gross": gross / value if value else 0, "net": net / value if value else 0, "beta_net": bnet / value if value else 0}

    MAX_GROSS = 2.0   # each sleeve is its own margin account, as in sim.py: the paper account's margin is shared.
    #                   __init__ sets the sleeve's own term over this default (max_gross_of)

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
        # close_only (Clayton, 17 Nov: "a flag the model sets on an order, so a stop it means as protective can never open a
        # new position"): the position is read as of now, with the fills since the look's state (cand2 at 12:00 that day:
        # its old stops filled during the look, and seven new stops then filled as opening buys). Only when the submit
        # holds such an order; if the fills cannot be read, those orders are not sent (their own condition can't be held).
        closers = [o for o in orders if o.get("close_only")]
        co_err, covered = None, {}              # covered: (symbol, side) -> close_only quantity still open at IBKR there
        if closers:                             # (CANCEL-PENDING included), then plus what this submit sends. The open
            try:                                # orders are read before the fills (the review): one that fills between the
                for w in self.paper.close_only_working(self.gw, self.sleeve):   # two reads counts twice, never not at all
                    k = (w["symbol"], w["action"])
                    covered[k] = covered.get(k, 0.0) + self.paper._left(w)
            except Exception as e:              # noqa: BLE001
                co_err = f"your open orders could not be read to check it ({type(e).__name__}: {str(e)[:120]})"
            if co_err is None:
                try:
                    self.paper.sync_fills(self.gw)
                except Exception as e:          # noqa: BLE001
                    co_err = f"your fills could not be read to check it ({type(e).__name__}: {str(e)[:120]})"
        # Margin, as sim.py and IBKR apply it: an order that adds exposure is rejected if, filled at its limit or
        # stop (else the last price), it would take the sleeve's gross exposure above MAX_GROSS x its value. A sleeve
        # with MAX_GROSS None has no such check: its orders go to IBKR, whose refusal is noted below like any other.
        syms = sorted(set(self.paper.book(self.sleeve)["positions"]) | {o["symbol"] for o in orders})
        quotes = {q["symbol"]: q for q in self.gw.quotes(syms) if not q.get("error")} if syms else {}
        px = {s: self.paper.mark(q) for s, q in quotes.items()}
        b = self.paper.book(self.sleeve, px)
        pos = {s: p["quantity"] for s, p in b["positions"].items()}
        mark = {s: (px.get(s) or p["avg_cost"]) for s, p in b["positions"].items()}
        nav = b["cash"] + sum(q * mark[s] for s, q in pos.items())
        held, echoes = dict(pos), []            # the book before this submit's orders, for the echo
        now_pos, sent_syms = dict(pos), set()   # close_only: the position counting fills, re-read after an order in the
        for o in orders:                        # same symbol was sent earlier in this submit (an entry and its stop)
            s, q0 = o["symbol"], pos.get(o["symbol"], 0)
            if o.get("close_only"):
                tag = f"{o['action']} {o['quantity']} {s} {o['type']} (close_only) not sent"
                if co_err is None and s in sent_syms:
                    try:                        # a market entry usually fills within seconds: wait 2 s, then read the open
                        _time.sleep(2)          # close_only orders and the fills again, in that order (the recheck: cover
                        covered = {}            # read before the wait counted a close_only order that filled in it twice)
                        for w in self.paper.close_only_working(self.gw, self.sleeve):
                            k = (w["symbol"], w["action"])
                            covered[k] = covered.get(k, 0.0) + self.paper._left(w)
                        self.paper.sync_fills(self.gw)
                        now_pos = {k: v["quantity"] for k, v in self.paper.book(self.sleeve)["positions"].items()}
                        sent_syms.discard(s)
                    except Exception as e:      # noqa: BLE001
                        notes.append(f"{tag}: your fills could not be read to check it ({type(e).__name__}: {str(e)[:120]})")
                        continue
                h = now_pos.get(s, 0)
                closing = "BUY" if h < 0 else "SELL" if h > 0 else None
                c = covered.get((s, o["action"]), 0.0)
                if co_err is not None:
                    notes.append(f"{tag}: {co_err}")
                    continue
                if o["action"] != closing:
                    notes.append(f"{tag}: your position in {s} is {h:+g} counting fills so far, so it would "
                                 + ("open a position" if not h else "add to it"))
                    continue
                if o["quantity"] > abs(h) - c + 1e-9:
                    notes.append(f"{tag}: your position in {s} is {h:+g} counting fills so far, and your close_only orders "
                                 f"there already cover {c:g}, so it would close more than you hold")
                    continue
            q1 = q0 + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
            p = o["limit"] or o["stop"] or px.get(s) or mark.get(s)
            if p and self.MAX_GROSS and abs(q1) > abs(q0):
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
                                 arrival=arrival, probs={"p_fill": o["p_fill"], "p_win": o["p_win"]},
                                 **({"close_only": True} if o.get("close_only") else {}), **fields)
            res = r.get("result") or {}
            msg = r.get("error") or "; ".join(res.get("messages", []) or [])
            status = res.get("status", "error" if r.get("error") else "?")
            refused = bool(r.get("error")) or status is None or status in ("Cancelled", "Inactive", "ApiCancelled")
            if not refused:                     # close_only (17 Nov): cover counts only what IBKR accepted; an accepted order
                sent_syms.add(o["symbol"])      # makes a later close_only order in its symbol read fills again
                if o.get("close_only"):
                    covered[(o["symbol"], o["action"])] = covered.get((o["symbol"], o["action"]), 0.0) + o["quantity"]
            if refused:                         # None: an error before any status (so far always 321)
                notes.append(f"{o['action']} {o['quantity']} {o['symbol']} {o['type']}: {status or 'refused'} {msg}".strip())
            # The echo only for an order the broker accepted, so it stays a fact about a real order; a refused one has its
            # refusal note above (review, 15 Nov).
            if not refused and getattr(self, "TEXT", "B") == "B":
                try:                            # a failure costs the echo, never the order or the look
                    e = order_echo(o, px.get(s), nav, held.get(s, 0))
                except Exception as ex:         # noqa: BLE001
                    e = None
                    (self.log or (lambda m: print(m, file=sys.stderr)))(
                        f"{self.sleeve}: order echo failed for {o['action']} {o['quantity']} {s}: {type(ex).__name__}: {ex}")
                if e:
                    echoes.append(e)
        notes += echoes                         # refusals keep their place; the echoes follow, in the order sent
        self.paper.add_notes(self.sleeve, notes)
        # This submit's own orders may close a position its close_only orders were for, and no check runs during a look
        # (the recheck): when it sent anything and the sleeve has close_only orders open, wait 2 s, read fills, and cancel
        # those the position no longer covers (their note goes in with the rest). Never raises.
        if sent_syms or any(o.get("close_only") for o in orders):
            try:
                if self.paper.close_only_working(self.gw, self.sleeve):
                    _time.sleep(2)
                    self.paper.sync_fills(self.gw)
                    self.paper.close_only_excess(self.gw, self.sleeve, log=self.log)
            except Exception as e:              # noqa: BLE001
                (self.log or (lambda m: print(m, file=sys.stderr)))(f"{self.sleeve}: close_only check after placing failed: {e}")
        return notes
