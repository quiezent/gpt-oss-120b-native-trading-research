"""check_orders: IBKR's what-if for the orders a text-B sleeve gives, before it submits (calculator_tools_spec_v2.md section 2,
frozen 15 Nov 2026; Clayton, 14 Nov: calculators "Go.", and "Ok, let's go with 3" checks a look, outside the 12 research calls).

Nothing is placed: the framework's whatif_many sets whatIf on every order and raises GatewayError if one ever lost it. Every
margin figure is the whole paper account's (the sleeves share one account), IBKR's own, at IBKR's own last price. The
order's value and the trail are computed here from the look's cached prices (no new quote request). No check blocks,
delays or changes an order: submit is unchanged, and PaperWorld.act runs no what-if.

Used by agent.gptoss_look (live.py hands it one shared client-17 connection, "trader-whatif", for a look) and by
mcp_server.py (the Claude sleeves' process runs its what-ifs on its own research client, 12 to 15: the Gateway Owner's
correction 6 and its README, so no second holder of 17 exists). The text-A control gets neither.
"""

import threading
from datetime import datetime

import tools
import trader

ET = tools.ET
MAX_CHECKS = 3          # Clayton, 14 Nov: 3 a look, not research calls
MAX_ORDERS = 20
WHATIF_TIMEOUT = 10.0   # s, whatif_many's own default

# The description shown to the model: the spec's text with the checker's frozen corrections of 15 Nov (facts only).
CHECK_DESC = (
    "IBKR's what-if for the orders you give; nothing is placed. Give up to 20 orders exactly as for submit. For each order: "
    "IBKR's own figures for the whole paper account if it filled (initial margin after, equity with loan after, commission or "
    "IBKR's range) and IBKR's error text when it gives one (201: initial margin after would exceed equity with loan after, and IBKR "
    "refuses such an order); the order's value as a multiple of your sleeve's value; what it does to your sleeve's position and "
    "to the account's net position in that symbol (all sleeves together, with resting orders on the same side counted as filled, as IBKR "
    "counts them); for TRAIL, the trail in dollars. IBKR's what-if checks each order alone, and prices margin at its own last "
    "price, which can differ from your delayed quote. The all-orders line is computed here from IBKR's figures, not by IBKR, and "
    "leaves out orders that lower initial margin. From 15:50 ET IBKR requires SMA not to be negative: SMA falls by half the value "
    "of each opening trade, long or short, rises by half of each closing trade, and is never below equity with loan minus half "
    "of gross. IBKR's documented rule: when excess liquidity is below zero it can close positions in the account. It did so "
    "on this account at 15:51 ET on Mon 16 Nov 2026. Up to 3 calls a look; they are not research calls.")
# The sentence before "Up to 3 calls" read "It has not happened on this account." through aa3224c; IBKR liquidated at 15:51
# ET on Mon 16 Nov 2026 (book_liquidation.py), so it was no longer true (the review of 16 Nov evening): a fact corrected.
# submit's own words for its place field. Through 2382e0d this read "Orders to check, as for submit.", and gpt-oss took the
# phrase for a key: on 60 held-out looks base wrote its first check plain in 2 of 34 looks with that wording and in 37 of 37
# with this one, and did not check less (34 against 37 looks; the Trading Model Researcher, 15 Nov, models-w3 5fdbbfb). The
# tool's description already says "exactly as for submit". unwrap() stays: a trained model may still wrap.
PLACE_DESC = "New orders."
_place = trader.TOOL_LIVE["parameters"]["properties"]["place"]          # text B's order items, as submit's
CHECK_SPEC = {"name": "check_orders", "description": CHECK_DESC,
              "parameters": {"type": "object", "properties": {"place": {**_place, "description": PLACE_DESC}},
                             "required": ["place"]}}
CLAUDE_LINE = "check_orders: IBKR's what-if, up to 3 calls a look, not research calls."   # the checker's sentence, 15 Nov
CAP_TEXT = f"check limit reached ({MAX_CHECKS} of {MAX_CHECKS}); no further check is run this look."
# Every call counts here, served or not (review, 15 Nov: with uncounted failures unbounded, a gpt-oss look could spend its
# step range on checks and end with no decision). A failed check still does not use one of the 3 (its text says so), but
# no more than MAX_TRIES calls a look are run; past that, the same limit, stated with the counts it has.
MAX_TRIES = MAX_CHECKS + 1


def cap_text(checked):
    """The reply to a check_orders call past the limit: CAP_TEXT after 3 counted checks, else the call limit with its counts."""
    return CAP_TEXT if checked >= MAX_CHECKS else (f"check limit reached ({MAX_TRIES} calls this look, {checked} of {MAX_CHECKS} "
                                                   "counted); no further check is run this look.")
EMPTY_TEXT = "No orders given; the account's figures above are all this check returns."
REFUSAL = "IBKR error 201 (this order alone against the whole account's margin; what-if, nothing was sent): "
# IBKR sends its price-band warning (about 70 words) with every limit-order what-if, whatever the size: one line a check
BAND_KEY = "away from the reference price"
BAND_HEAD = "IBKR's price-band notice, sent with every limit-order what-if (not about this order's size): "
DONE = ("Cancelled", "ApiCancelled", "Filled", "Inactive")
CONN_CODES = (504, 1100)  # the 05:00 ET restart and IBKR's own link: reconnect once (the Gateway Owner's correction 10)
# IBKR error 103, duplicate order id (Mon 16 Nov 2026, the Trading Model Researcher: 87 of 99 what-if rows at 09:00 and
# 09:30): the what-if was sent under an order id IBKR had already seen, so IBKR judged nothing about the order. It is a
# fault of the what-if connection's ids (the TWS Gateway Owner's framework), never an answer on the order. The orders that
# get it are sent once more (each what-if takes a new id); if every order sent still gets it, the check does not count and
# the model is told it is our fault; a single order that still gets it is marked so, and the rest of the check stands.
DUP_ID = 103
DUP_TEXT = ("error: IBKR's what-if could not be run for your orders: our what-if connection sent them under order ids IBKR "
            "had already seen (IBKR error 103, duplicate order id). This is a fault on our side, not IBKR's answer on your "
            "orders, which it has not judged; this check did not count.")
DUP_ROW = ("not checked: our what-if connection sent this order under an order id IBKR had already seen (IBKR error 103, "
           "duplicate order id); a fault on our side, not IBKR's answer on the order")


def supported(gw):
    """True if this framework (a Gateway or its class) has the what-if methods."""
    return all(callable(getattr(gw, m, None)) for m in ("whatif_many", "all_open_orders", "account", "positions"))


READ_TIMEOUT = 5.0     # s, for account(), positions() and all_open_orders() in a check (review, 15 Nov)


class _Skipped(Exception):
    """An account read left out: an earlier one in this look got no reply."""


class Link:
    """One what-if connection shared by every thread of a look (a second connection on one client id is refused, 326), and
    in an MCP process its research Gateway. whatif_many holds the framework's lock only while sending, so what-ifs overlap;
    account(), positions() and all_open_orders() hold it for their whole wait (the framework's own timeout: 10 s on 17,
    30 s on an MCP research client). So read() bounds those waits at READ_TIMEOUT, and after the first "no reply" to one of
    them it skips them for the rest of the look (a Link lives one look; in an MCP process, one look too): the check still
    runs and says which account facts it lacks, and a slow link holds a sleeve about READ_TIMEOUT + whatif_many's 10 s a
    check, and queues the other sleeves' reads behind it for at most READ_TIMEOUT. reconnect(): after the 05:00 ET Gateway
    restart, on GatewayError 504 or 1100, one thread reconnects (gw.connect()) and the others reuse it."""

    def __init__(self, gw):
        self.gw, self.gen, self._lock, self.slow = gw, 0, threading.Lock(), None

    def reconnect(self, gen):
        with self._lock:
            if self.gen == gen:
                self.gw.connect()
                self.gen += 1

    def read(self, name):
        """gw.<name>() with its wait bounded at READ_TIMEOUT; raises _Skipped once a read in this look got no reply. The
        framework reads its timeout per request, so it is lowered for this call under the framework's own lock (which the
        call takes anyway, so no other request on this connection sees the lower timeout)."""
        if self.slow:
            raise _Skipped(self.slow)
        lock = getattr(self.gw, "_lock", None)
        with (lock if lock is not None else self._lock):
            if self.slow:                   # another thread's read got no reply while this one waited
                raise _Skipped(self.slow)
            had = isinstance(getattr(self.gw, "timeout", None), (int, float))
            old = self.gw.timeout if had else None
            if had:
                self.gw.timeout = min(old, READ_TIMEOUT)
            try:
                return getattr(self.gw, name)()
            except Exception as e:          # noqa: BLE001 - re-raised; a no-reply marks the link slow for the look
                if _no_reply(e):
                    self.slow = f"{name}() gave no reply within {READ_TIMEOUT:g} s"
                raise
            finally:
                if had:
                    self.gw.timeout = old


def _no_reply(e):
    return isinstance(e, TimeoutError) or "No reply" in str(e)


def read_account(link, log=None, reconnect=True):
    """account() on the link for text B's account line (one read a look, before the looks): the row, or None (logged). On
    GatewayError 504 or 1100 (the 05:00 ET restart) it reconnects once and reads again, unless reconnect is False (live.py's
    fallback on client 10, whose connection is the look's own). Never raises."""
    say = log or (lambda m: None)
    for attempt in (0, 1):
        gen = link.gen
        try:
            a = link.read("account")
            if isinstance(a, dict) and a:
                return a
            say(f"account line: IBKR's figures not read for this look (account() gave {type(a).__name__})")
            return None
        except Exception as e:              # noqa: BLE001
            if attempt == 0 and reconnect and _conn(e):
                try:
                    link.reconnect(gen)
                    continue
                except Exception as e2:     # noqa: BLE001
                    e = e2
            say(f"account line: IBKR's figures not read for this look ({_gw_msg(e)})")
            return None
    return None


def context(st, feats, live, sleeves, looking, sleeve=None, own_limit=None, marks=False):
    """What a check needs from the look, built once with the sleeve's state: its value and positions (its ledger), the
    look's prices (delayed last during the session, else the last close; no new quote request), the number of sleeves in
    the sleeves file and the number that look in this round. JSON-able, for the Claude path's file. sleeve and
    own_limit (trader.own_limit_of; 18 Nov): with a limit, the check's answer ends with the sleeve's own line
    (own_limit.check_block), which needs the state's gross and the sleeve's name for its resting orders. marks: held
    names the table and quotes lack are priced at the state's own mark ("position mark"), as with a limit; live.py sets
    it from 19 Nov for the sleeves with the term, whose order lines kept these prices when the limit line went (the
    review of 18 Nov evening: they came in with the limit, but they are a price fact, not a limit piece)."""
    prices = {s: [float(x["close"]), "last close"] for s, x in (feats or {}).items()
              if isinstance(x, dict) and isinstance(x.get("close"), (int, float)) and x["close"] == x["close"] and x["close"] > 0}
    prices.update({s: [float(p), "delayed last"] for s, p in (live or {}).items() if isinstance(p, (int, float)) and p > 0})
    out = {"value": float(st["value"]), "positions": {p["symbol"]: p["quantity"] for p in st.get("positions") or []},
           "prices": prices, "sleeves": int(sleeves), "looking": int(looking)}
    if own_limit:
        out.update(own_limit=float(own_limit), gross=float(st["gross"]), sleeve=sleeve)
    if own_limit or marks:
        for p in st.get("positions") or []:  # the state's own marks for held names the table and quotes lack, so the
            c = p.get("close")              # check prices them as the before-figure does (the review of 18 Nov)
            if isinstance(c, (int, float)) and not isinstance(c, bool) and c == c and c > 0:
                out["prices"].setdefault(p["symbol"], [float(c), "position mark"])
    return out


def _limit_line(ctx, orders, resting):
    """The sleeve's own line for a check (own_limit.check_block), from the check's cleaned orders and the account's resting
    orders as read for this check (None if not read), of which only this sleeve's count."""
    import own_limit
    import paper
    mine = filled = None
    if resting is not None:
        mine = [x for x in resting if paper.sleeve_of(x.get("ref")) == ctx.get("sleeve")]
        refs = {x.get("ref") for x in mine}
        filled = {}                         # what a partly filled resting order has filled, by ref (the ledger's fills)
        for f in paper.fills():
            if f.get("ref") in refs:
                filled[f["ref"]] = filled.get(f["ref"], 0.0) + float(f.get("shares") or 0)
    try:
        co = set(paper.close_only_refs(ctx.get("sleeve")))
    except Exception:                       # noqa: BLE001 - an unreadable ledger: no order is taken as close_only
        co = set()
    return own_limit.check_block(ctx, [o for _, o in orders], mine, co, filled)


def whatif_args(o):
    """whatif_many's arguments for a cleaned order, mapped as PaperWorld.act maps it for paper.place: MOO is MKT with tif OPG;
    TRAIL carries trailingPercent; a limit only for LMT, STP LMT and LOC; a stop (aux) only for STP and STP LMT. The model's own
    keys (p_fill, p_win) are left out: a key that is not an ibapi Order field raises ValueError in the framework."""
    typ, tif, fields = o["type"], o["tif"], {}
    limit = o["limit"] if typ in ("LMT", "STP LMT", "LOC") else None
    aux = o["stop"] if typ in ("STP", "STP LMT") else None
    if typ == "MOO":
        typ, tif = "MKT", "OPG"
    if typ == "TRAIL" and o["trail_percent"] is not None:
        fields["trailingPercent"] = float(o["trail_percent"])
    return {"symbol": o["symbol"], "action": o["action"], "quantity": o["quantity"], "order_type": typ, "limit": limit,
            "aux": aux, "tif": tif, **fields}


def _n(x):
    """A price or percent as the model would write it: 750, 712.12, 0.02."""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def _usd(x):
    return "not given" if x is None else f"${x:,.0f}"


def _an(n):
    """The article before "<n>-share": an 18-share, an 8-share, an 11-share, an 800-share, a 1826-share."""
    s = str(int(n))
    lead = f"{int(n):,}".split(",")[0]
    return "an" if s[0] == "8" or lead in ("11", "18") else "a"


def _desc(o):
    t = o["type"]
    px = (f" {_n(o['limit'])}" if t in ("LMT", "LOC") and o["limit"] is not None else
          f" {_n(o['stop'])}" if t == "STP" and o["stop"] is not None else
          f" stop {_n(o['stop'])} limit {_n(o['limit'])}" if t == "STP LMT" and o["stop"] is not None and o["limit"] is not None else
          f" {_n(o['trail_percent'])}%" if t == "TRAIL" and o["trail_percent"] is not None else "")
    return f"{o['action']} {o['quantity']} {o['symbol']} {t}{px} {o['tif']}"


def _price(o, ctx):
    """(price, words) for the order's value: its limit, else its stop, else the look's price for the symbol."""
    a = whatif_args(o)
    if a["limit"] is not None:
        return a["limit"], "at its limit"
    if a["aux"] is not None:
        return a["aux"], "at its stop"
    p = (ctx.get("prices") or {}).get(o["symbol"])
    if p:
        return float(p[0]), f"at the {p[1]} {float(p[0]):.2f}"
    return None, None


def _value(o, ctx):
    """'value $X at ... = R.RRx your sleeve's value ($V)', and for TRAIL '; the trail is $T (P x t / 100)'."""
    p, words = _price(o, ctx)
    v = ctx.get("value")
    if p is None:
        out = f"value not known (no price for {o['symbol']} in this look)"
    elif v is None or v <= 0:
        out = f"value ${o['quantity'] * p:,.0f} {words}; your sleeve's value is not positive ({trader.fmt_money(v or 0.0)})"
    else:
        out = f"value ${o['quantity'] * p:,.0f} {words} = {o['quantity'] * p / v:.2f}x your sleeve's value ({trader.fmt_money(v)})"
    if o["type"] == "TRAIL" and o["trail_percent"] is not None and p is not None:
        tp = o["trail_percent"]
        out += f"; the trail is ${p * tp / 100:.2f} ({p:.2f} x {_n(tp)} / 100)"
    return out


def _effect(q0, dq, ref):
    """What an order of signed size dq does to a position q0; ref(m, side) names the position ("your 60-share long")."""
    n, m = abs(dq), abs(q0)
    side = lambda q: "long" if q > 0 else "short"
    if q0 == 0:
        return f"opens {_an(n)} {n}-share {side(dq)}"
    if q0 * dq > 0:
        return f"adds {n} to {ref(m, side(q0))}"
    if n < m:
        return f"{'sells' if q0 > 0 else 'covers'} {n} of {ref(m, side(q0))}"
    if n == m:
        return f"closes {ref(m, side(q0))}"
    k = n - m
    return f"closes {ref(m, side(q0))} and opens {_an(k)} {k}-share {side(dq)}"


def _num(q):
    q = float(q)
    return int(q) if q.is_integer() else q


def _sleeve(o, ctx):
    q0 = _num((ctx.get("positions") or {}).get(o["symbol"], 0) or 0)
    dq = o["quantity"] if o["action"] == "BUY" else -o["quantity"]
    holds = f"no {o['symbol']}" if q0 == 0 else f"{abs(q0)} {o['symbol']} {'long' if q0 > 0 else 'short'}"
    return f"Your sleeve holds {holds}; if it fills, it {_effect(q0, dq, lambda m, s: f'your {m}-share {s}')}."


def _account(o, held, resting):
    """The account's net position in the symbol (A), the shares in resting orders on the order's side (R), and what IBKR's
    what-if sees: E = A - R for a SELL, A + R for a BUY."""
    if held is None or resting is None:
        return "Account: IBKR's positions or resting orders were not read for this check."
    s, buy = o["symbol"], o["action"] == "BUY"
    a = _num(held.get(s, 0))
    r = _num(sum(x["quantity"] for x in resting if x["symbol"] == s and x["action"] == o["action"]))
    e = a + r if buy else a - r
    net = lambda q: f"net {abs(q)} {s} long" if q > 0 else f"net {abs(q)} {s} short" if q < 0 else f"no net {s} position"
    word = "buys" if buy else "sells"
    rest = f"no resting {s} {word}" if r == 0 else f"{r} shares in resting {s} {word}"
    seen = "" if r == 0 else f" (the resting {word} counted as filled: {net(e)})"
    dq = o["quantity"] if buy else -o["quantity"]
    return f"Account: {net(a)}, {rest}; to IBKR{seen} it {_effect(e, dq, lambda m, side: f'that {side}')}."


def _figures(r):
    return r.get("init_after") is not None or r.get("ewl_after") is not None


def _ibkr(r):
    """IBKR's own figures, commission, any warning other than the price band, and its error text verbatim."""
    ch = r.get("init_change")
    init = _usd(r.get("init_after")) + (f" (change -${-ch:,.2f})" if ch is not None and ch < 0 else "")
    lo, hi, c = r.get("min_commission"), r.get("max_commission"), r.get("commission")
    com = (f"commission ${c:,.2f}" if c is not None else f"commission ${lo:,.2f}-{hi:,.2f}" if lo is not None and hi is not None
           else f"commission ${lo if lo is not None else hi:,.2f}" if lo is not None or hi is not None else "no commission given")
    out = f"IBKR: initial margin after {init}; equity with loan after {_usd(r.get('ewl_after'))}; {com}."
    w = r.get("warning")
    if w and BAND_KEY not in w:
        out += f" IBKR's warning: {w}"
    return out + _error(r)


def _dup(r):
    """IBKR error 103 (duplicate order id) with no IBKR figures: IBKR judged nothing about the order (DUP_ID)."""
    return isinstance(r, dict) and "_bad" not in r and r.get("error_code") == DUP_ID and not _figures(r)


def _error(r):
    code, err = r.get("error_code"), r.get("error")
    if code == 201:
        return " " + REFUSAL + str(err or "")
    if code is not None:
        return f" IBKR error {code}: {err or ''}".rstrip()
    if err and err != "timeout":
        return f" IBKR error: {err}"
    return ""


def _acct_num(a, k):
    try:
        v = a.get(k)
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError, AttributeError):
        return None


def _gw_msg(e):
    """'GatewayError: Disconnected from IB Gateway' (the framework prefixes its message with the code)."""
    msg, code = str(e), getattr(e, "code", None)
    if code is not None and msg.startswith(f"{code}: "):
        msg = msg[len(f"{code}: "):]
    return f"{type(e).__name__}: {msg[:200]}"


def _conn(e):
    return getattr(e, "code", None) in CONN_CODES or "Disconnected" in str(e) or "Not connected" in str(e)


class _Lost(Exception):
    """The connection failed during a check (the check does not count)."""

    def __init__(self, e):
        super().__init__(str(e))
        self.e = e


def unwrap(args):
    """A check call wrapped under one key, repaired: with no "place" key and exactly one key, a value that is an object
    with "place" is the call, and a value that is a non-empty list of objects is the call's place. 16 Nov 2026, the gpt-oss
    dry looks: the first check of each was wrapped under the start of the place field's own description, as {"Orders to
    check": {"place": [...]}} (14 Nov) or {"Orders to check": [...]} (15 Nov, which cost that look a 1,760-token turn).
    Nothing else is repaired (two keys, an inner object without "place", an empty list or a list of anything but objects).
    Both the gpt-oss path and the MCP path come through check(); the trace keeps the call as the model wrote it."""
    if isinstance(args, dict) and "place" not in args and len(args) == 1:
        inner = next(iter(args.values()))
        if isinstance(inner, dict) and "place" in inner:
            return inner
        if isinstance(inner, list) and inner and all(isinstance(o, dict) for o in inner):
            return {"place": inner}
    return args


def check(link, args, ctx, now=None, log=None):
    """One check_orders call: (text, counted, raw). raw holds IBKR's replies for the look's trace. Never raises: a failure
    costs that check only. A lost connection is reconnected once and the check run again; if it fails again the check does
    not count, as a failed research connection does not."""
    say = log or (lambda m: None)
    args = unwrap(args)
    place = args.get("place") if isinstance(args, dict) else None
    if not isinstance(place, list):
        return "error: give place, a list of orders as for submit; this check did not count.", False, None
    if link is None:
        return "error: the what-if connection is not open for this look; this check did not count.", False, None
    if not isinstance(ctx, dict):
        say("check_orders: no sleeve state for the check")
        return "error: your sleeve's state for the check was not read; this check did not count.", False, None
    for attempt in (0, 1):
        gen = link.gen
        try:
            return _check(link, place, ctx, now, say)
        except _Lost as lost:
            e = lost.e
            if attempt == 0 and _conn(e):
                try:
                    link.reconnect(gen)
                    continue
                except Exception as e2:     # noqa: BLE001 - reported below as the connection failure
                    e = e2
            say(f"check_orders: the what-if connection failed: {_gw_msg(e)}")
            if "Refusing to send" in str(e):
                return f"error: IBKR's what-if was not sent ({_gw_msg(e)}); this check did not count.", False, {"error": _gw_msg(e)}
            return f"error: the what-if connection failed ({_gw_msg(e)}); this check did not count.", False, {"error": _gw_msg(e)}
        except Exception as e:              # noqa: BLE001 - a bug here costs this check only
            say(f"check_orders failed: {type(e).__name__}: {e}"[:300])
            return f"error: the order check failed ({type(e).__name__}: {str(e)[:200]}); this check did not count.", False, None
    return "error: the what-if connection failed; this check did not count.", False, None


def _check(link, place, ctx, now, say=None):
    gw = link.gw
    now = now or datetime.now(ET)
    t = f"{now:%H:%M:%S}"
    head = (f"IBKR what-if at {t} ET; nothing was placed. Figures are for the whole paper account, which holds all "
            f"{ctx.get('sleeves')} sleeves; {ctx.get('looking')} of them look in this round, and orders the others send after "
            f"{t} are not in these figures.")
    raw = {"account": None, "whatif": []}
    try:
        acct = link.read("account")
        if not isinstance(acct, dict):
            raise TypeError(f"account() gave {type(acct).__name__}")
        raw["account"] = acct
    except Exception as e:                  # noqa: BLE001
        if _conn(e):
            raise _Lost(e)
        why = ("timeout" if _no_reply(e) else "skipped after a timeout earlier in this look" if isinstance(e, _Skipped)
               else type(e).__name__)
        acct, before = None, f"Account before: IBKR's account figures were not read ({why}); the all-orders line is not computed."
    if acct is not None:
        im, ewl = _acct_num(acct, "InitMarginReq"), _acct_num(acct, "EquityWithLoanValue")
        part = lambda k, name: f"{name} " + ("not reported" if _acct_num(acct, k) is None else f"${_acct_num(acct, k):,.0f}")
        before = "Account before (IBKR): " + "; ".join(part(k, n) for k, n in (
            ("NetLiquidation", "net liquidation"), ("EquityWithLoanValue", "equity with loan"), ("InitMarginReq", "initial margin"),
            ("ExcessLiquidity", "excess liquidity"), ("SMA", "SMA"), ("GrossPositionValue", "gross"))) + "."
    orders, lines = [], []
    for i, raw_o in enumerate(place[:MAX_ORDERS], 1):
        o, err = trader.clean_order(raw_o)
        if not err and not o["symbol"]:
            err = f"{o['action']} : symbol is required"
        if not err and not o["type"]:       # the framework's what-if would check it as MKT, but submit sends it with no type
            err = f"{o['action']} {o['symbol']}: type is required"   # and IBKR refuses it (321): not the same order (review)
        if err:
            lines.append((i, f"{i}. not checked: {err}"))
        else:
            orders.append((i, o))
    if not orders:                          # nothing sent to IBKR: no all-orders line (review, 15 Nov)
        out = [head, before] + [l for _, l in lines]
        if not place:
            out.append(EMPTY_TEXT)
        if len(place) > MAX_ORDERS:
            out.append(f"Orders after the {MAX_ORDERS}th were not checked ({len(place)} given).")
        return "\n".join(out), True, raw
    held = resting = None
    try:
        held = {}
        for p in link.read("positions"):
            if p.get("quantity"):
                held[p["symbol"]] = held.get(p["symbol"], 0) + p["quantity"]
        resting = [x for x in link.read("all_open_orders") if x.get("status") not in DONE and x.get("symbol") and x.get("quantity")]
    except Exception as e:                  # noqa: BLE001
        if _conn(e):
            raise _Lost(e)
        held = resting = None
    syms = {o["symbol"] for _, o in orders}
    raw["positions"] = {s: q for s, q in (held or {}).items() if s in syms}
    raw["resting"] = [x for x in resting or [] if x["symbol"] in syms]
    specs = [whatif_args(o) for _, o in orders]
    try:
        results = gw.whatif_many(specs, timeout=WHATIF_TIMEOUT)
    except ValueError:                      # a malformed order is refused before anything is sent: then one at a time,
        results = []                        # so it costs only itself
        for s in specs:
            try:
                results.append(gw.whatif_many([s], timeout=WHATIF_TIMEOUT)[0])
            except ValueError as e:
                results.append({"_bad": str(e)})
            except Exception as e:          # noqa: BLE001
                raise _Lost(e)
    except Exception as e:                  # noqa: BLE001 - GatewayError: a lost connection, or a what-if that lost its flag
        raise _Lost(e)
    dup = [k for k, r in enumerate(results) if _dup(r)]
    if dup:                                 # IBKR error 103: those orders once more, each under a new id
        raw["whatif_first"] = [dict(r) for r in results]
        try:
            again = gw.whatif_many([specs[k] for k in dup], timeout=WHATIF_TIMEOUT)
        except ValueError:                  # not expected (these orders were accepted for sending once already)
            again = None
        except Exception as e:              # noqa: BLE001
            raise _Lost(e)
        if isinstance(again, list) and len(again) == len(dup):
            for k, r in zip(dup, again):
                results[k] = r
    raw["whatif"] = results
    sent = [r for r in results if "_bad" not in r]
    if sent and all(_dup(r) for r in sent):
        return DUP_TEXT, False, raw
    band = None
    for (i, o), r in zip(orders, results):
        if "_bad" in r:
            lines.append((i, f"{i}. not checked: {r['_bad']}"))
            continue
        w = r.get("warning")
        if w and BAND_KEY in w and band is None:
            band = w.split(". ")[0].rstrip(".") + "."
        if r.get("error") == "timeout" and not _figures(r):
            lines.append((i, f"{i}. {_desc(o)}: {_value(o, ctx)}. IBKR did not answer within {WHATIF_TIMEOUT:g} s: no IBKR figures "
                             "for this order, and it is not in the all-orders line."))
        elif _dup(r):
            lines.append((i, f"{i}. {_desc(o)}: {DUP_ROW}."))
        elif not _figures(r) and r.get("error_code") not in (None, 201):
            lines.append((i, f"{i}. {_desc(o)}:{_error(r)}"))
        else:
            lines.append((i, f"{i}. {_desc(o)}: {_value(o, ctx)}. {_sleeve(o, ctx)} {_account(o, held, resting)} {_ibkr(r)}"))
    out = [head] + ([BAND_HEAD + band] if band else []) + [before] + [l for _, l in sorted(lines)]
    if len(place) > MAX_ORDERS:
        out.append(f"Orders after the {MAX_ORDERS}th were not checked ({len(place)} given).")
    out += _all_orders(acct, [r for r in results if "_bad" not in r])
    if ctx.get("own_limit"):                # 18 Nov (Clayton: "Yes, text only"): the sleeve's own limit, its last line
        try:                                # a failure costs the line, never the check
            rest = []                       # the line counts the whole list, as submit sends it, not only the 20 checked
            for raw_o in place[MAX_ORDERS:]:
                o, err = trader.clean_order(raw_o)
                if not err and o["symbol"] and o["type"]:
                    rest.append((0, o))
            out.append(_limit_line(ctx, orders + rest, resting))
        except Exception as e:              # noqa: BLE001
            raw["own_limit_error"] = f"{type(e).__name__}: {e}"[:300]
            (say or (lambda m: None))(f"check_orders: the sleeve's exposure-limit line left out: {raw['own_limit_error']}")
    return "\n".join(out), True, raw


def _all_orders(acct, results):
    """The gateway owner's formula: IM + the sum of (init_after - IM), EWL + the sum of (ewl_after - ewl_before), over orders
    whose IBKR initial margin after is above IM (a 201 counts). Left out, and counted in the line: every result without IBKR
    figures (a timeout, a refusal with no figures), and one with an error other than 201, labelled by its code (review, 15
    Nov). "The account's figures above stand" only when every order sent came back with figures and none raises initial
    margin. Not computed without account(); the caller gives no line when no order was sent."""
    if acct is None or not results:
        return []
    im, ewl = _acct_num(acct, "InitMarginReq"), _acct_num(acct, "EquityWithLoanValue")
    if im is None or ewl is None:
        return ["All orders: not computed (IBKR's account figures lack initial margin or equity with loan)."]
    counted, missing, coded = [], 0, {}
    for r in results:
        code = r.get("error_code")
        if not _figures(r):
            missing += 1
        elif code not in (None, 201):
            coded[code] = coded.get(code, 0) + 1
        elif r.get("init_after") is not None and r["init_after"] > im:
            counted.append(r)
    if missing == len(results):
        return ["All orders: not computed (no order came back with IBKR figures)."]
    line = "All orders, computed here (not by IBKR): "
    if counted:
        d_ewl = sum(((r["ewl_after"] - r["ewl_before"]) if r.get("ewl_after") is not None and r.get("ewl_before") is not None else
                     r.get("ewl_change") or 0.0) for r in counted)
        line += f"initial margin ${im + sum(r['init_after'] - im for r in counted):,.0f}; equity with loan ${ewl + d_ewl:,.0f}."
    elif missing or coded:
        line += "no order with IBKR figures raises initial margin."
    else:
        line += "no order raises initial margin, so the account's figures above stand."
    n = lambda k: f"{k} order{'s' if k > 1 else ''}"
    parts = ([f"{n(missing)} without IBKR figures"] if missing else []) + [f"{n(k)} with IBKR error {c}" for c, k in coded.items()]
    if parts:
        line += f" ({' and '.join(parts)} {'is' if missing + sum(coded.values()) == 1 else 'are'} not in this line)"
    return [line]
