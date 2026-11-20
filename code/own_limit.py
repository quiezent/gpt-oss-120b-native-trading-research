"""A sleeve's own exposure limit, told to the model in its own text; no software enforces it.

Clayton, Wed 18 Nov 2026 (to the Trading Model Researcher, about 02:13 ET): "You were supposed to look at a better
prompt/warning to direct gpt-oss sleeves to not take on more that x3.0 margins and to trade within their own NAV and not
the entire IBKR account." He had told Trader Training on Tue 17 Nov: "The Researcher has been working on a prompt update
that you will need to implement that directs the gpt-oss sleeves to handle their margins/exposures." Asked whether it
should go live on the eight gpt-oss sleeves, the text-A control included, he answered "Yes, text only" (03:45 ET).

The text is the Researcher's (models-w3 runs/ml/research/week6/limit3/SPEC.md, version 2, and its l3_common.py and
l3m.py), tested on base-90k-b's and cand2's looks of 16-17 Nov (RESULT.md there): five pieces.
  1. term(): the system text's sentence in place of trader.NO_GROSS_RULE (every sleeve with the term).
  2. limit_line(): its own line in the state, straight after the sleeve-value line and before the account line.
  3. ACCOUNT_SENTENCE: at the end of text B's account line, after the margin warning, when the line carries figures.
  4. check_block(): the last line of the check_orders answer, before "(order check N of 3)" (text B, which has the check).
  5. positions_sentence(): appended to the over-limit form of the limit line.
Not built: the Researcher's "leave to send the cut at the first turn" (l3n.FIRST), which the test showed has no effect.
From the first look of Thu 19 Nov 2026 only piece 1, the term, is live: trader.OWN_LIMIT_LOOK is False, so the state
and the check no longer work the limit out for the sleeve (pieces 2 to 5), on the Researcher's one change after Clayton's
"Don't decide for the trader model." (18 Nov). The code below stays for replays of 18 Nov's text.

Where this departs from the Researcher's reference code (two reviews of 18 Nov, 04:00-05:00 ET), all in figures, not in
the tested words except where a case the test never met needs a word:
- Resting orders. The live check reads the sleeve's own (the test's block could not) and counts those that would open
  or add in the stops sentence ("...and your resting orders another $X unless you cancel them"); "Your resting orders are
  not in these figures" is said only when they could not be read. The state's limit line (STATE_RESTING on) takes them
  off the room, says by how much, names the largest three and counts any it could not price, in the Researcher's words
  of 18 Nov ("unless you cancel them", "after those resting orders"); SPEC.md's "your resting orders included" was said
  with no figure behind it and is gone (dc1ff6e7). Names not held are priced as the check prices them (prices_of).
- The walk (check_block). Only orders that fill now move the after-figure, at the look's price; orders at the close add
  but do not cut (they fill after IBKR's 15:50 ET check); orders that fill only at a price (stops, trailing orders,
  limits away from the price) are read along a move of the price one way (worst_adds): a take-profit or a stop
  protecting a position adds nothing, stops stacked on one side are read together, and an entry away from the price is
  counted once with its exits. The reference took every order that is not a stop as filled (base-ml's brackets of Tue
  14:30 read backwards) and read each stop alone.
- close_only orders are read as PaperWorld.act sends them (refused when they would add or when the sleeve's other
  close_only orders already cover the position), and along a path they never pass zero.

A sleeve term in sleeves.json ("own_limit": 3.0, beside "max_gross": null; trader.own_limit_of). Nothing is rejected,
cut or changed by software; max_gross stays null. A sleeve without the term reads the text it read before.
Token cost (o200k_base): the term 137 tokens, 96 more than the sentence it replaces; the limit line 53 under the limit
and 95 over (about 20 more with the resting-orders clause); the positions sentence about 500 on base-90k-b's 51 positions
at Tue 17 Nov 15:30 (the Researcher measured about 330 on its test's looks); the account sentence 40; the check's line
about 120 for each check (up to 3 a look).
"""
import math
import sys

LIMIT = 3.0
# The state line's resting orders (resting_figure): False gives the room with no word on them (dc1ff6e7, after 09:00 on
# 18 Nov showed SPEC.md's "your resting orders included" false: cand1 told $71,192 with $74,583 of resting adds); True
# takes them off the room, says by how much and names the largest. On Mon-Tue's 222 gpt-oss looks, 98 had resting orders
# that would open or add, up to $323,841 on a $97,445 sleeve (base-90k-b, Tue 12:30: buy stops left on names it no
# longer held). The Researcher, 18 Nov about 09:35 ET: hold the switch until names not held are priced (a trailing order
# there was left out silently) and it has read the built text on today's records. About 10:00 ET, its check of the
# built text from fb15974d passed: "Both. Turn STATE_RESTING and FILL_POSITION on for the 10:30 look". On from 10:30.
STATE_RESTING = True

_TERM = ("Your exposure limit: gross exposure may not exceed {x} your sleeve's value. Count your positions and every order, "
         "new or resting, that would open or add to one; a hedge adds to gross too. Trade on your own sleeve's value: the "
         "paper account holds other sleeves too, and its margin, excess liquidity and buying power are not room for your "
         "orders. No software stops an order that breaks this limit; keeping to it is your job at every look. If your "
         "gross exposure is above {x}, your orders at that look must bring it to {x} or less. IBKR may still refuse an "
         "order, and that is reported at your next look.")
ACCOUNT_SENTENCE = (" The dollar figures in this line are the whole account's, for all the sleeves together: none of them "
                    "is room for your orders. Your room is your own exposure limit, in the line above.")
# the margin warning's form for a sleeve whose value is at or below zero names that sleeve's own gross in dollars
# (trader.margin_warning): the sentence then says so (the review of 18 Nov)
OWN_GROSS_IN_WARNING = "with its value at or below zero"
ACCOUNT_SENTENCE_OWN = (" Apart from your own gross exposure in the warning, the dollar figures in this line are the whole "
                        "account's, for all the sleeves together: none of them is room for your orders. Your room is your "
                        "own exposure limit, in the line above.")
OVER_HEAD = "Over your exposure limit: your limit is"


def money(x):
    return f"${x:,.0f}"


def term(limit=LIMIT):
    """Piece 1: the system text's term, for a limit as a multiple of the sleeve's value."""
    return _TERM.format(x=f"{limit:g}x")


def account_sentence(account_line):
    """Piece 3 for an account line that carries IBKR's figures."""
    return ACCOUNT_SENTENCE_OWN if OWN_GROSS_IN_WARNING in account_line else ACCOUNT_SENTENCE


NAMES_SHOWN = 3        # the resting orders' names in the limit line: the largest three, then "and N more names"


def _names(names):
    """ " (by name, largest first: TLT $39,750, IEF $22,550, BABA $6,570, and 2 more names)" from [(symbol, dollars)],
    largest first; "" with none."""
    if not names:
        return ""
    head = ", ".join(f"{s} {money(u)}" for s, u in names[:NAMES_SHOWN])
    more = len(names) - NAMES_SHOWN
    return (f" (by name, largest first: {head}" + (f", and {more} more {_plural(more, 'name', 'names')}" if more > 0 else "")
            + ")")


def _rest_clause(names):
    """What the resting orders would open or add, the words after "another $R"."""
    return f" if they fill, unless you cancel them{_names(names)}"


def _others(unpriced):
    """The Researcher's sentence (18 Nov) for unpriced orders beside resting dollars: its own sentence, so "they" before
    it keeps its plural, and "other" so it is not read as the names' "more"."""
    return f" {unpriced} other resting {_plural(unpriced, 'order', 'orders')} could not be priced."


def _unpriced_alone(unpriced):
    """Under the limit, with no resting dollars: what the room leaves out."""
    return f"{unpriced} of your resting orders could not be priced and {_plural(unpriced, 'is', 'are')} not counted"


def _unpriced_alone_over(unpriced):
    """Over the limit (or no value), with no resting dollars: the Researcher's words of 18 Nov, so "not counted" is not
    read as "does not count against your limit"."""
    if unpriced == 1:
        return ("1 of your resting orders could not be priced; if it would open or add to a position, it counts against "
                "your limit too.")
    return (f"{unpriced} of your resting orders could not be priced; any that would open or add to a position count "
            "against your limit too.")


def limit_line(value, gross_x, limit=LIMIT, resting=None, names=(), unpriced=0):
    """Piece 2, as l3_common.limit_line: value, the sleeve's value; gross_x, its gross exposure as a multiple of that
    value (the state's own figure). resting: the dollars its resting orders would open or add if they fill (worst_adds
    over the state's orders, resting_figure), with names, [(symbol, dollars)] largest first, and unpriced, the number of
    its resting orders with no price at all; or None when they were not read (STATE_RESTING off), and the room is then
    given without a word on them (18 Nov: at 09:00 cand1 was told $71,192 "your resting orders included" while its
    resting orders would open or add $74,583, SPEC.md's words with no figure behind them). The words for resting orders
    are the Researcher's of 18 Nov (about 09:25 ET): "unless you cancel them", the largest three by name, and "after those
    resting orders" in place of "your resting orders included"; and (about 10:00 ET, after the review) "and N more names",
    unpriced orders beside resting dollars in their own sentence ("N other resting orders could not be priced."), and over
    the limit with no resting dollars "...; if it would open or add to a position, it counts against your limit too".
    Exactly at the limit is not over."""
    unread = resting is None
    rest = not unread and resting >= 1
    un = 0 if unread else int(unpriced or 0)
    sentence = ""                                       # the over form's and the no-value form's sentence on them
    if rest:
        sentence = (f" Your resting orders would open or add another {money(resting)}{_rest_clause(names)};"
                    " they count against your limit too." + (_others(un) if un else ""))
    elif un:
        sentence = f" {_unpriced_alone_over(un)}"
    if value <= 0:
        return ("Over your exposure limit: your sleeve's value is not positive, so you have no room. Place only orders "
                "that close or shrink positions." + sentence)
    cap, used = limit * value, gross_x * value
    if used > cap:
        over = used - cap
        return (f"Over your exposure limit: your limit is {limit:g}x your sleeve's value = {money(cap)} of gross exposure, "
                f"and you hold {money(used)} ({gross_x:.2f}x), which is {money(over)} over. Your orders at this look must "
                f"bring your gross exposure to {money(cap)} or less: close or shrink positions by at least {money(over)} "
                "more than you open or add. A stop or trailing order does not bring it down until it fills." + sentence)
    if not rest:
        return (f"Your exposure limit: {limit:g}x your sleeve's value = {money(cap)} of gross exposure. You hold "
                f"{money(used)} ({gross_x:.2f}x). Room left for orders that open or add to positions: "
                f"{money(cap - used)}" + (f"; {_unpriced_alone(un)} in it." if un else "."))
    left = cap - used - resting
    return (f"Your exposure limit: {limit:g}x your sleeve's value = {money(cap)} of gross exposure. You hold {money(used)} "
            f"({gross_x:.2f}x), and your resting orders would open or add another {money(resting)}{_rest_clause(names)}."
            + (_others(un) if un else "") + " Room left for orders that open or add to positions, after those resting "
            "orders: " + (f"{money(left)}." if left >= 0 else f"none; with them you would be {money(-left)} over."))


def ordinal(n):
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def positions_sentence(positions, over):
    """Piece 5, as l3m.positions_sentence: positions, the state's ({"symbol", "quantity", "close"}; close is the price the
    state shows); over, the dollars the sleeve is over its limit. Largest first; the running total is shown up to and
    including the position at which it first reaches the overage; the rest are listed with their size only."""
    ps = sorted(({"s": str(p["symbol"]), "side": "long" if float(p["quantity"]) > 0 else "short",
                  "v": abs(float(p["quantity"])) * float(p["close"])} for p in positions if float(p["quantity"]) != 0),
                key=lambda x: (-x["v"], x["s"]))
    if not ps:
        return ""
    head, rest, total, at = [], [], 0.0, None
    for i, p in enumerate(ps, 1):
        if at is None:
            total += p["v"]
            head.append(f"{p['s']} {p['side']} {money(p['v'])} [{money(total)}]")
            if total >= over:
                at = i
        else:
            rest.append(f"{p['s']} {p['side']} {money(p['v'])}")
    out = ("Your positions by size in dollars at the price shown, largest first, with the running total in brackets: "
           + "; ".join(head) + ".")
    if at is not None:
        out += (f" The running total first reaches the {money(over)} you are over at the {ordinal(at)} position, "
                f"{ps[at - 1]['s']}.")
        if rest:
            out += " The rest: " + "; ".join(rest) + "."
    else:
        out += f" All of them together are less than the {money(over)} you are over."
    return out


def prices_of(feats, live):
    """The look's prices by symbol, as checks.context builds them for the order check: the table's last close, and during
    the session the delayed last over it."""
    out = {s: float(x["close"]) for s, x in (feats.items() if isinstance(feats, dict) else ())
           if isinstance(x, dict) and _px(x.get("close")) is not None}
    out.update({s: float(p) for s, p in (live.items() if isinstance(live, dict) else ()) if _px(p) is not None})
    return out


def resting_figure(st, prices=None):
    """What the state's working orders (CANCEL-PENDING included: they can still fill) would open or add if they fill, by
    worst_adds from the state's positions: {"usd": dollars, "names": [(symbol, dollars)] largest first, "unpriced": orders
    with no price at all}. The look price of a held name is the state's own mark (the before-figure's); of a name not
    held, prices (prices_of: the check's), so a trailing order there is priced (the Researcher, 18 Nov: at 09:00
    base-90k-b's BUY 150 META TRAIL 5% and BUY 100 STX TRAIL 5% were left out, META alone about $93k). A trailing order is
    read at IBKR's own trail stop where the row has it (trail_stop), and an order at what is left to fill (remaining)."""
    pos = {p["symbol"]: float(p["quantity"]) for p in st.get("positions") or []}
    look = {s: _px(p) for s, p in (prices or {}).items()}
    look.update({p["symbol"]: _px(p.get("close")) for p in st.get("positions") or []})
    legs = {}
    for o in st.get("orders") or []:
        n = _norm(o, o.get("type"), o.get("limit"), o.get("stop"), o.get("trail_percent"), o.get("trail_stop"),
                  _left(o, None), bool(o.get("close_only")))
        if n["q"] <= 0 or (n["co"] and not pos.get(n["sym"], 0.0)):
            continue                        # nothing left to fill, or a close_only order with nothing to close: it can
        legs.setdefault(n["sym"], []).append(n)   # add nothing, so it is neither in the dollars nor "could not be priced"
    total, names, unpriced = 0.0, [], 0
    for s, ns in legs.items():
        _, usd, un = worst_adds(pos.get(s, 0.0), ns, look.get(s))
        if not math.isfinite(usd):          # no figure is better than "$inf": the name's orders are counted as unpriced
            usd, un = 0.0, len(ns)
        unpriced += un
        if usd > 0:
            total += usd
            names.append((str(s), usd))
    names.sort(key=lambda x: (-x[1], x[0]))
    return {"usd": total, "names": names, "unpriced": unpriced}


def state_resting(st, prices=None):
    """The dollars of resting_figure."""
    return resting_figure(st, prices)["usd"]


def state_line(st, limit, prices=None, log=None):
    """Pieces 2 and 5 for a look's state (world.state(): value, gross, positions, orders): the limit line with the
    resting orders' dollars (when STATE_RESTING is on; prices: prices_of, for names not held), and on the over-limit form
    the positions sentence after it."""
    v, gx = float(st["value"]), float(st["gross"])
    line = None
    if STATE_RESTING:                       # a failure here costs the resting words only, never the limit line (the
        try:                                # review of 18 Nov): the line is then the switch-off form
            r = resting_figure(st, prices)
            line = limit_line(v, gx, limit, r["usd"], r["names"], r["unpriced"])
        except Exception as e:              # noqa: BLE001
            (log or (lambda m: print(m, file=sys.stderr)))(
                f"resting orders left out of the exposure-limit line: {type(e).__name__}: {e}"[:300])
    if line is None:
        line = limit_line(v, gx, limit, None)
    if line.startswith(OVER_HEAD):
        sent = positions_sentence(st.get("positions") or [], gx * v - limit * v)
        if sent:
            line += " " + sent
    return line


# ------------------------------------------------------------------------------------------------ the order check
STOP_KINDS = ("STP", "TRAIL")          # STP, STP LMT, TRAIL, TRAIL LIMIT: they fill only if the price gets to them


def _type(o):
    return str(o.get("type") or o.get("order_type") or "").upper()


def _is_stop(o):
    return _type(o).startswith(STOP_KINDS)


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and 0 < x < float("inf") else None


MAX_PRICE = 10_000_000.0   # a price per share above this is no price (the Researcher, 18 Nov): IBKR's unset value
#                            1.7976931348623157e308 times any shares overflows to inf without raising


def _px(x):
    """A price: finite, positive and not above MAX_PRICE, else None (the order is then counted as unpriced)."""
    v = _num(x)
    return v if v is not None and v <= MAX_PRICE else None


def adding(q0, dq):
    """The shares of an order of dq (signed) on a position q0 that add exposure: past zero, all that lands on the other
    side; else the rise in the absolute position (the Researcher's week6_counts.adding)."""
    q1 = q0 + dq
    return abs(q1) if q0 * q1 < 0 else max(0.0, abs(q1) - abs(q0))


def cutting(q0, dq):
    """The shares of the order that lower the absolute position (a flip closes all of the old side)."""
    q1 = q0 + dq
    return abs(q0) if q0 * q1 < 0 else max(0.0, abs(q0) - abs(q1))


def _signed(o, q=None):
    q = float(o.get("quantity") or 0) if q is None else q
    return q if str(o.get("action")).upper() == "BUY" else -q


def _through(t, buy, stop, look):
    """A stop the look's price is already at or through: IBKR triggers it at once."""
    return look is not None and stop is not None and (stop <= look if buy else stop >= look)


def kind(o, prices):
    """How the check counts a list order: ("now", price) for one that fills now (MKT, MOO, a limit at or through the
    look's price, a stop the price is already at or through, or a limit with no look price), priced at the look's price,
    else its limit; ("close", price) for an order at the close (MOC, and LOC at or through the look's price), which fills
    after IBKR's 15:50 ET check (the Researcher, 18 Nov): what it adds counts, what it closes does not; ("if", price) for
    one that fills only if the price gets to it (a stop or trailing order, at its stop, else the look's price; a limit
    away from the look's price, LOC included, at its limit). The price is None if none is known."""
    t, p = _type(o), (prices or {}).get(o.get("symbol"))
    look = _num(p[0]) if p else None
    buy = str(o.get("action")).upper() == "BUY"
    if t.startswith(STOP_KINDS):
        if t.startswith("STP") and _through(t, buy, _num(o.get("stop")), look):
            return "now", look
        return "if", _num(o.get("stop")) or look
    lim = _num(o.get("limit"))
    if t in ("LMT", "LOC") and lim is not None and look is not None:
        if (lim < look) if buy else (lim > look):
            return "if", lim
    return ("close" if t in ("MOC", "LOC") else "now"), look or lim or _num(o.get("stop"))


def _norm(o, t, limit, stop, trail_pct, trail_stop, q, close_only):
    """One order as worst_adds reads it."""
    return {"sym": o.get("symbol"), "buy": str(o.get("action")).upper() == "BUY", "q": float(q), "t": str(t or "").upper(),
            "limit": _px(limit), "stop": _px(stop), "trail_pct": _num(trail_pct), "trail_stop": _px(trail_stop),
            "co": bool(close_only), "stop_like": str(t or "").upper().startswith(STOP_KINDS)}


def _leg(n, look):
    """(path, price) of an order that fills only at a price, or at once: "at" fills at once (a market order, an order at
    the open or the close, a limit or stop the price is already through); "down" fills as the price falls to its price
    (a buy limit below, a sell stop below, a sell trailing order at its stop); "up" as the price rises to it. With no
    look price, buys are read on "up" and sells on "down" at their own prices, so orders on one side still stack."""
    t, buy = n["t"], n["buy"]
    if t.startswith("TRAIL"):
        p = n["trail_stop"] or n["stop"] or (look * (1 + n["trail_pct"] / 100 if buy else 1 - n["trail_pct"] / 100)
                                            if look and n["trail_pct"] else look)
        return ("up" if buy else "down"), p
    if t.startswith("STP"):
        p = n["stop"]
        if look is None:
            return ("up" if buy else "down"), p
        if p is None or _through(t, buy, p, look):
            return "at", look
        return ("up" if buy else "down"), p
    if t in ("LMT", "LOC"):
        p = n["limit"]
        if look is None:
            return ("up" if buy else "down"), p
        if p is None or ((p >= look) if buy else (p <= look)):
            return "at", look
        return ("down" if buy else "up"), p
    return "at", look or n["limit"] or n["stop"]


def worst_adds(P, legs, look):
    """(shares, dollars, unpriced) for one symbol: the most that the fills of a move one way from the look's price would
    open or add from the position P. As the price falls, the orders on the falling side fill nearest first; as it rises,
    those on the rising side; the orders that fill at once are in every set. So stops stacked on one side are read
    together, and an entry away from the price with its exits is read once. A round trip (down, then up past the look's
    price, or the reverse) is not read: it would count every bracket's take-profit and stop as both filling, the next
    look sees the first fill, and a close_only exit is cancelled when the position no longer covers it. "Open or add" is
    adding() of the change from P to the position the set leaves (a flip counts the new side, as for one order), priced
    at the highest price among the orders that moved the position that way. A close_only order never passes zero (act
    and close_only_excess cancel what the position no longer covers). Orders with no price at all are left out and
    counted."""
    at, down, up, unpriced = [], [], [], 0
    for n in legs:
        path, px = _leg(n, look)
        if px is None:
            unpriced += 1
            continue
        (at if path == "at" else down if path == "down" else up).append((px, n))
    # orders that fill together (at once, or at one price) are read close_only first, so each is held to the position
    # before the tie (the review of 18 Nov: a close_only and a plain stop at one price read $0 or the flip by list order)
    at.sort(key=lambda x: not x[1]["co"])
    down.sort(key=lambda x: (-x[0], not x[1]["co"]))
    up.sort(key=lambda x: (x[0], not x[1]["co"]))
    close = lambda x: x[1]["t"] in ("MOC", "LOC")
    best_sh = best_usd = 0.0
    for k, j in [(k, 0) for k in range(len(down) + 1)] + [(0, j) for j in range(1, len(up) + 1)]:
        legs = at + down[:k] + up[:j]
        day = [x for x in legs if not close(x)]          # an order at the close fills after the day's fills: the set
        seqs = (day, day + [x for x in legs if close(x)]) if len(day) < len(legs) else (legs,)   # is read both ways
        for seq in seqs:
            pos, moved = P, []
            for px, n in seq:
                dq = n["q"] if n["buy"] else -n["q"]
                if n["co"]:
                    if pos == 0 or pos * dq > 0:
                        continue
                    dq = max(dq, -pos) if pos > 0 else min(dq, -pos)
                pos += dq
                moved.append((px, dq))
            sh = adding(P, pos - P)
            if sh <= 0:
                continue
            way = 1 if pos - P > 0 else -1
            px = max(p for p, dq in moved if dq * way > 0)
            if sh * px > best_usd:
                best_sh, best_usd = sh, sh * px
    return best_sh, best_usd, unpriced


def _left(o, filled):
    """A resting order's shares still to fill: IBKR's remaining when the row has it, else its quantity less the fills
    the ledger holds under its ref (filled: {ref: shares})."""
    r = o.get("remaining")
    if isinstance(r, (int, float)) and not isinstance(r, bool) and r >= 0:
        return float(r)
    return max(0.0, float(o.get("quantity") or 0) - float((filled or {}).get(o.get("ref"), 0.0)))


def _plural(n, one, many):
    return one if n == 1 else many


def check_block(ctx, orders, resting=None, close_only=(), filled=None):
    """Piece 4: the order check's line for a sleeve with the term. ctx: checks.context() (value, gross, positions, prices,
    own_limit); orders: the check's cleaned orders (trader.clean_order), in the order given; resting: this sleeve's
    resting orders at IBKR (all_open_orders rows) or None if they were not read; close_only: refs of this sleeve's
    close_only orders; filled: {ref: shares} of this sleeve's fills, for what is left of a partly filled resting order.

    The after-figure walks, in turn, the orders that fill now (kind()), each as if the earlier ones had filled, priced at
    the look's price as the before-figure is; then the orders at the close, whose adds count and whose closes do not.
    The orders that fill only at a price do not move it: from the position after the walk, worst_adds reads them along a
    move of the price one way, the list's alone (the stops sentence) and with this sleeve's resting orders (the resting
    figure is what they add on top), so a take-profit or a stop protecting a position adds nothing, stops stacked on one
    side are read together, and an entry away from the price is counted once with its exits.
    The wording is SPEC's whenever the list has no limit away from the price and no order at the close that closes; with
    one, the parenthesis and the sentence name it.
    close_only, as PaperWorld.act sends it: an order is sent only if the position is on the other side and covers it
    together with the sleeve's other close_only orders still open there (resting ones at what is left to fill, and those
    earlier in the list that fill later); one act would refuse is left out entirely. Not modelled: act counts an entry
    earlier in the same submit only if it has filled 2 seconds after it was sent, so a close_only exit on an entry made
    before the open, which act refuses, is read here as accepted."""
    limit = float(ctx["own_limit"])
    V, gx = float(ctx["value"]), float(ctx["gross"])
    prices = ctx.get("prices")
    look_of = lambda s: _num((prices or {}).get(s, [None])[0])
    pos = {s: float(q) for s, q in (ctx.get("positions") or {}).items()}
    cover = {}
    for o in resting or []:
        if o.get("ref") in close_only:
            k = (o.get("symbol"), str(o.get("action")).upper())
            cover[k] = cover.get(k, 0.0) + _left(o, filled)
    add = cut = 0.0
    unpriced, later, at_close, away, close_cut, thru = 0, [], [], False, False, False
    for o in orders:
        s, q = o["symbol"], float(o["quantity"])
        dq = _signed(o, q)
        how, px = kind(o, prices)
        if o.get("close_only"):
            h = pos.get(s, 0.0)
            closing = "BUY" if h < 0 else "SELL" if h > 0 else None
            k = (s, str(o["action"]).upper())
            if k[1] != closing or q > abs(h) - cover.get(k, 0.0) + 1e-9:
                continue                    # act() would not send it: it neither moves the position nor counts
            if how != "now":                # still working after act's 2-second re-read: it covers
                cover[k] = cover.get(k, 0.0) + q
        if how == "if":
            later.append(o)
            away = away or not _is_stop(o)
            continue
        if how == "close":
            at_close.append((o, px))
            continue
        q0 = pos.get(s, 0.0)
        pos[s] = q0 + dq
        thru = thru or _is_stop(o)          # a stop the price is already through: counted, and the parenthesis says so
        if px is None:
            unpriced += 1
            continue
        add += adding(q0, dq) * px
        cut += cutting(q0, dq) * px
    pos_day = dict(pos)                     # the position through the day, before the orders at the close
    for o, px in at_close:                  # after the orders that fill now, in turn: what they add counts, not what
        s, dq = o["symbol"], _signed(o)     # they close (they fill after IBKR's 15:50 check)
        q0 = pos.get(s, 0.0)
        if o.get("close_only"):             # never past zero (the review of 18 Nov)
            if q0 == 0 or q0 * dq > 0:
                continue
            dq = max(dq, -q0) if q0 > 0 else min(dq, -q0)
        pos[s] = q0 + dq
        if px is None:
            unpriced += 1
            continue
        add += adding(q0, dq) * px
        close_cut = close_cut or cutting(q0, dq) > 0
    mine, theirs = {}, {}                   # symbol -> legs: the list's orders that fill at a price; this sleeve's resting
    for o in later:
        mine.setdefault(o["symbol"], []).append(_norm(o, o["type"], o.get("limit"), o.get("stop"), o.get("trail_percent"),
                                                      None, float(o["quantity"]), bool(o.get("close_only"))))
    for o in resting or []:
        theirs.setdefault(o.get("symbol"), []).append(_norm(o, o.get("order_type"), o.get("limit"), o.get("aux"), None,
                                                            o.get("trail_stop"), _left(o, filled),
                                                            o.get("ref") in close_only))
    sa = tot = 0.0
    s_stop = s_lim = False
    r_unpriced = 0
    for s in set(mine) | set(theirs):
        look = look_of(s)
        # read against the position through the day and the one after the orders at the close; the larger counts (the
        # review of 18 Nov: a closing MOC applied first hid what a stop would add during the day)
        Ps = {pos_day.get(s, 0.0), pos.get(s, 0.0)}
        both = [(worst_adds(P, mine.get(s, []), look), worst_adds(P, mine.get(s, []) + theirs.get(s, []), look)) for P in Ps]
        (sh_l, usd_l, un_l), _ = max(both, key=lambda b: b[0][1])
        _, (sh_t, usd_t, un_t) = max(both, key=lambda b: b[1][1])
        unpriced += un_l
        r_unpriced += max(0, un_t - un_l)
        sa += usd_l
        tot += max(usd_t, usd_l)
        if usd_l >= 1:
            s_stop = s_stop or any(n["stop_like"] for n in mine[s])
            s_lim = s_lim or any(not n["stop_like"] for n in mine[s])
    ra = tot - sa
    who = ("Stops, trailing orders and limit orders away from the price" if s_stop and s_lim else
           "Limit orders away from the price" if s_lim else "Stops and trailing orders")
    if sa >= 1 and ra >= 1:
        extra = (f" {who} in this list would open or add another {money(sa)} if they fill, and your resting orders "
                 f"another {money(ra)} unless you cancel them; they count against your limit too.")
    elif sa >= 1:
        extra = f" {who} in this list would open or add another {money(sa)} if they fill; they count against your limit too."
    elif ra >= 1:
        extra = (f" Your resting orders would open or add another {money(ra)} if they fill, unless you cancel them; they "
                 "count against your limit too.")
    else:
        extra = ""
    if resting is None:
        extra += " Your resting orders are not in these figures; one that would open or add counts against your limit too."
    un = ""
    if unpriced:
        un += (f" {unpriced} {_plural(unpriced, 'order', 'orders')} could not be priced and "
               f"{_plural(unpriced, 'is', 'are')} not counted.")
    if r_unpriced:
        un += (f" {r_unpriced} of your resting orders could not be priced and {_plural(r_unpriced, 'is', 'are')} "
               "not counted.")
    paren = ("(a stop, a trailing order or a limit order away from the price is not counted until it fills" if away
             else "(a stop or trailing order is not counted until it fills")
    paren += ", except a stop the price is already through" if thru else ""
    paren += "; what an order at the close closes is not counted before the close)" if close_cut else ")"
    if V <= 0:
        return (f"Your sleeve's value is not positive, so it has no room under its exposure limit. These orders open or add "
                f"{money(add)} and close or shrink {money(cut)}. Submit only orders that close or shrink positions."
                + extra + un)
    cap, before = limit * V, gx * V
    after = before + add - cut
    head = (f"Your sleeve, by your own exposure limit ({limit:g}x its value = {money(cap)}): before these orders its gross "
            f"exposure is {money(before)} ({gx:.2f}x). These orders open or add {money(add)} and close or shrink "
            f"{money(cut)} {paren}, so after them it is {money(after)} ({after / V:.2f}x)")
    if after <= cap:
        tail = f": inside your limit, with {money(cap - after)} of room left."
    else:
        tail = (f": {money(after - cap)} over your limit. Before you submit, change the list until this figure is at or "
                f"below {money(cap)}: take out orders that open or add, and put in orders that close or shrink positions.")
    return head + tail + extra + un
