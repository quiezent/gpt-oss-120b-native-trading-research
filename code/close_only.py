"""close_only v2.1: the rule for orders a sleeve marks close_only (runs/analysis/close_only_v2/DESIGN.md).

Clayton, 19 Nov 2026: "Yes if its a fix for future sleeves too" (the close-only fix and linked exits, from Mon 23 Nov, the
platform's standard for every text-B sleeve). CLAUDE.md: software may hold an order to the limits the model put on it,
"an order it marks as a close may not add to the position; exit legs it links cancel together".

Pure functions only (no broker, no ledger), so every case can be tested: paper.close_only_check and trader.PaperWorld.act
read the books and the broker and carry out what these return.

- Orders that could fill at the same moment must together fit the position (groups): stops (STP, STP LMT, TRAIL) with
  stops, limits (LMT) with limits; MOO, MOC, LOC and anything sent outside the session with both; a MKT in the session
  with none (it fills within seconds), so it must fit alone, beside any leg whose cancel the broker has not confirmed.
- A leg is a close_only order at the broker, CANCEL-PENDING included (it can still fill). rem: what it has left to fill.
- At each check every group's legs are kept within the position: the newest leg first is reduced, or cancelled.
- An exit placed beside a working entry is held (not at the broker) and goes out at min(its quantity, the shares its
  entries filled since plus P, the room the position leaves in its groups); while linked it grows with those fills. P
  (the Researcher, 24-25 Nov 2026) is the part of its quantity above what its entries had left to fill at the hold, at
  most the shares held then on its closing side (less its entries' own fills since their snapshots), and it counts only
  once none of its entries still works: nothing goes out for it before then, and at that check the exit goes out or
  grows as far as it fits and is unlinked, so it is used once.
- A leg with no group (a MKT close not yet filled) can fill at any moment, so it counts beside every order (the review of
  20 Nov: two closes, one a working MKT, were both sent); a new MKT in the session counts only such legs and those whose
  cancel is pending, so it goes out beside a full stop (the stop then shrinks at act's check 2 s later).
"""
import os
# The switch (DESIGN E): False keeps the rule of 17-19 Nov, its notes and its words (tools.CLOSE_DESC_V1) unchanged. Set
# True at a day boundary only, once the TWS Gateway Owner has released Gateway.modify with a passed paper test (v2.1 is all
# or nothing: without a modify it would cancel a stop after a partial close). Read at import by tools, trader, paper and
# mcp_server, so a change takes the loop's restart (and a Claude look's new tool server).
_DEFAULT = True    # on for Week 7 (branch week7, live from Mon 23 Nov 2026): the Owner's modify committed (tws 29c5f49, reasons
                   # 4b4269d; its paper test passed without fills, four fill cases untested) and the Researcher's words
                   # decided (models-w3 f1af7fa8). Rollback: TRADER_CLOSE_ONLY_V2=0 in the loop's environment, or False here.
# A process the loop starts (a Claude look's tool server, mcp_server.py) gets the loop's own value through
# TRADER_CLOSE_ONLY_V2 (agent.claude_look), so its words follow the loop's rule even if this file changes mid-run.
V2 = {"1": True, "0": False}.get(os.environ.get("TRADER_CLOSE_ONLY_V2", ""), _DEFAULT)
EPS = 1e-9
STOP_TYPES = frozenset({"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"})
AUCTION_TYPES = frozenset({"MOO", "MOC", "LOC"})
GROUPS = ("stop", "limit")


def groups(order_type, tif="DAY", in_session=True):
    """The groups of orders an order can fill together with (DESIGN, Definitions). order_type as the model or IBKR names it
    (a MOO reaches IBKR as MKT with tif OPG). in_session: False for a new order sent outside 09:30-16:00 ET, which then
    counts with every leg (at the open it can fill beside any stop or limit the price passes). The check calls it with
    in_session True: a leg's groups follow its type."""
    t, f = str(order_type or "").upper(), str(tif or "").upper()
    if t in AUCTION_TYPES or f == "OPG" or not in_session:
        return frozenset(GROUPS)
    if t in STOP_TYPES:
        return frozenset({"stop"})
    if t == "LMT":
        return frozenset({"limit"})
    return frozenset()                                     # MKT in the session


def cover(legs, grp):
    """(shares, group): what the legs that could fill with an order of groups grp already have left, the largest group
    first. A leg with no group (a working MKT) counts in every group. With no group (a new MKT in the session), only legs
    with no group and legs whose cancel is pending count (group None)."""
    if not grp:
        return sum(l["rem"] for l in legs if l.get("pending") or not l["groups"]), None
    return max(((sum(l["rem"] for l in legs if g in l["groups"] or not l["groups"]), g) for g in sorted(grp)),
               key=lambda x: (x[0], x[1]))


def fits(q, h, action, legs, grp):
    """(ok, why, shares, group) for a new close_only order of q shares and groups grp, on a position h (counting fills
    so far), beside legs (same symbol and action). why: None, "side" (not the closing side) or "more" (with the legs that
    could fill with it, it would close more than the position)."""
    closing = "BUY" if h < 0 else "SELL" if h > 0 else None
    if action != closing:
        return False, "side", 0.0, None
    c, g = cover(legs, grp)
    if q > abs(h) - c + EPS:
        return False, "more", c, g
    return True, None, c, g


def room(h, legs, grp, skip=None):
    """Shares the position leaves for one more order of groups grp beside legs (skip: a leg's own ref, left out)."""
    others = [l for l in legs if l.get("ref") != skip]
    return max(0.0, abs(h) - cover(others, grp)[0])


def part_above(q, left, had):
    """P, the part of a held exit's size above what its entries had left to fill at the hold, for shares already held
    (the Trading Model Researcher's rule of 24-25 Nov 2026): min(max(0, q - E), H). left: E as recorded at the hold
    (paper.place's held "left"); had: H, the shares held then on the exit's closing side less its entries' fills since
    their snapshots (held "had"; the review of 25 Nov: without it, a flat or opposite book gave P > 0, and the exit went
    out for another order's shares). Either None (a row held before 25 Nov, which keeps the rule it was placed under:
    never for other shares) gives 0. The caller applies it only once none of the exit's entries still works (the late
    gate, the Researcher's decision of 25 Nov)."""
    if left is None or had is None:
        return 0.0
    try:
        return max(0.0, min(float(q) - float(left), float(had)))
    except (TypeError, ValueError):
        return 0.0


def held_total(q, fe, f, h, action, legs, grp, ref=None, p=0.0):
    """A held (or linked) exit's total at the broker (DESIGN B, with the Researcher's P of 24 Nov 2026): min(q, its
    entries' fills since the hold plus P, its own fills plus the room its groups leave) while h is on its closing side;
    else its own fills (nothing left to work). p: part_above(q, E), the shares already held that the exit's size covers
    beyond its entries (base-30k's HPE on 24 Nov: -20 held, SELL 80 and BUY 100 TRAIL, P 20)."""
    closing = "BUY" if h < 0 else "SELL" if h > 0 else None
    if action != closing:
        return float(f)
    return float(max(f, min(q, fe + p, f + room(h, legs, grp, skip=ref))))


def shrink(h, legs):
    """Reductions that keep every group's legs within the position (DESIGN C.3): [(ref, new_rem)], new_rem 0 meaning a
    cancel. legs: dicts with ref, rem, groups, pending, key (older first); a leg whose cancel is pending is never changed.
    A leg with no group (a working MKT) is a member of every group. In each group the grouped legs are cut first, the
    newest first, and a MKT leg last (it is the model's close of now; a stop beside it is what shrinks). Legs are updated
    in place (rem) so callers see the result."""
    out = {}
    for g in GROUPS:
        members = [l for l in legs if g in l["groups"] or not l["groups"]]
        excess = sum(l["rem"] for l in members) - abs(h)
        for l in sorted(members, key=lambda l: (bool(l["groups"]), l["key"]), reverse=True):
            if excess <= EPS:
                break
            if l.get("pending") or l["rem"] <= EPS:
                continue
            cut = min(excess, l["rem"])
            l["rem"] -= cut
            excess -= cut
            out[l["ref"]] = l["rem"]
    return [(r, max(0.0, v)) for r, v in out.items()]


def fmt(x):
    """Shares as the state prints them: 100, not 100.0."""
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"
