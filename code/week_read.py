"""The week's live comparison: the per-look and per-day counts of runs/ml/research/rl6_criteria.md section 6, named and
committed before the first look of Mon 26 Oct 2026 (09:00 ET). Every count is reported whatever its level; none is a
trigger, and none reaches a model.

    python week_read.py                  # this week: runs/sleeves.json, runs/live/<date>/, runs/*.jsonl
    python week_read.py --week week2     # an archived week (runs/archive/week2): the test set
    python week_read.py --check-port     # the vendored order accounting against train4.py at 0f645b2

Run it after the 07:30 ET refresh that follows the last day read, so that day's closes are in the daily bars. Writes
runs/week_read/<week>/: looks.jsonl (a row a look), days.jsonl (a row a sleeve-day), m3_cases.json and m3_key.json (the
side-misread candidates, blinded for A0's audit, and their key), summary.md.

Definitions (fixed here):
- A look: a look file runs/live/<date>/<sleeve>-<HHMM>.json that is not a dry run. Its slot is the scheduled look time,
  the latest 30-minute mark from 09:00 at or before the file's minute (a delayed look keeps its slot; `started` is the
  file's minute; two files of one sleeve in one slot are both kept and flagged `slot_dup`). Timing, from live.log's line
  for the look (written after its orders were placed): `acts_min`, minutes after the slot when its orders went to the
  broker; for the gpt-oss sleeves, which run one after another, `start_min` (acts less the look's seconds), when its own
  state was built and sampling began. The Claude sleeves see the state built at the slot, run one after another beside
  the gpt-oss ones, and are placed after them.
- retried: the trace entries of agent.sample_once's first timeout (09657d2). framework_failure: no decision, with an
  error the framework raised, not the model (an exception, a second sample timeout, a Claude look that did not finish);
  `failure` says which. place_failed: a Claude decision whose placing raised.
- no_decision: no valid submit. forced: a research call was answered "research limit reached".
- The order accounting is train4.order_stats as of the Researcher's 0f645b2 (models-w3), vendored verbatim below so a
  later training change cannot move it. Inputs: the look's state and call; prices from replay4.prices (the frozen R1
  scorer's reading of the look's table and research results), then each quote result's "SYM: last X" lines in call order
  (the week-3 quote wording, which that reader predates), then the positions' marks; range1 from the forecast file as of
  the previous close; researched symbols by returned_symbols on each research call. It gives: held (positions at the
  look), protective (stop-type orders, not in an offsetting pair, that reduce the position as the same submit's earlier
  non-stop orders leave it: a stop on a same-submit entry counts, one after a same-submit leg that flattens or reverses
  the name does not, so the count depends on leg order; train4's protective_per_held_position numerator), entries,
  entries_no_stop, under_1k (orders under $1,000), token_orders (orders that are not substantive: a few shares, far
  levels, re-placements, offsetting pairs, duplicate stops), unresearched (hard M8: names of exposure-adding orders with
  no data-returning quote, news, stats or forecast call), unresearched_any (M8x: names of any order with none), empty
  (nothing placed or cancelled) and inactive (EMPTY_COST's: no substantive order or cancel). Beside it, protective_held:
  stop-type orders not in an offsetting pair that reduce the position held at the look.
- conc75_unprotected: replay4.state_conc75, at the look the largest position is 75%+ of the sleeve with no stop-type order
  on its closing side working. replay4.py is always the frozen R1 scorer at d59537b (rl6_criteria.md A0), loaded from git.
- margin_rejected: orders the 2x term rejected (placed_notes "insufficient margin").
- m3 (the frozen scorer's M3: replay4.misreads on the notes and the reasoning), m3n (its M3n, the notes alone) and m3q
  (its M3q, an order adding exactly a held position's size): candidates only, counted after A0's blinded audit of
  m3_cases.json, never before. The cases carry no sleeve or model name: order refs read "ref-<hex>".
- gap: GAP_RE's matches (arithmetic on the distance to the goal or pace) in the notes plus the decision text (gpt-oss: the
  analysis before its submit call, or, when that turn was prefilled by a research limit or a malformed answer, the last
  earlier step's analysis; Claude: the CLI's final message, the only text its trace keeps). gap_notes: the notes alone,
  the common basis across models. pace_restated: PACE_RE, the pace or the old goal figure restated, not arithmetic.
- fri_behind: a Friday look at 12:00 ET or later when the sleeve's value at the look is below start_cash x (1 + 0.0214 /
  5 x the sessions it completed before that day); with its entries and their notional.
- -ml sleeves: ml_entries, entries (non-stop orders adding exposure); costed_entries, those at a notional within 10% of the
  one a forecast call in the same look costed for that name (its notional_usd, else the tool's default); dollar_cited, the
  notes or decision text cite move$ or sleeve%; line_cited_entries, entries whose name shares a line or sentence with
  such a citation and had a forecast row returned in the look: candidates for "sized from the dollar line", confirmed
  by a blinded read before they count.
- A sleeve-day (daily.jsonl, written at 16:15): NAV; day and cumulative return on start_cash; cash's 0; SPY over the same
  window (from runs/decks.json's spy_baseline for the week; for a sleeve that joins mid-week, SPY's close before its first
  look); realized, commissions, fills; risk_top: the largest position carried overnight (largest |shares x close| at
  that day's daily-bar close, else the day's last look mark, else its latest fill) and its one-sigma day move in dollars,
  as the next look's risk line shows it: the vol column of trader.Data features through that close (outside names by
  outside_features), else the stats tool's hist_vol% / sqrt(252) from that sleeve-day's latest look that has it;
  `vol_source` says which.
"""

import argparse
import ast
import json
import math
import random
import re
import statistics
import subprocess
import types
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import forecast
import intrasim
import paper
import tools
import trader

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LIVE = RUNS / "live"
OUT = RUNS / "week_read"
PORT = ("0f645b2", "train4.py")      # the vendored order accounting's source
FROZEN = "d59537b"                   # A0's frozen R1 scorer: replay4.py is loaded from this commit, not main's copy
PACE = 0.0214                        # Clayton's pace, a week; the pace line's per-session step is PACE / 5
SEED = 20261026                      # the blinded M3 cases' shuffle
SLOTS = [f"{h:02d}{m:02d}" for h in range(9, 16) for m in (0, 30)]     # live.LOOK_TIMES

# Distance-to-target arithmetic (Analyst, section 6; narrowed on week 2's hand-labelled matches by the review of 25 Oct).
# Week 2 (dollar goal): "the +8% needed in 3 sessions is out of reach", "needs about +$700"; week 3 (pace line):
# "behind the weekly pace", "to get back to pace".
_X = (r"commission|cash|margin|exposure|gross|allocat|buying power|cost|mindful|careful|stop|shares|order|purchase|price|"
      r"invest|quantity|\bkeep\b|\bstay\b|\bunder\b|reduce|consider|decide|\bposition\b|\bfree\b|>\s?0\.2|0\.2\s?%")
_D = r"(?:the |this |our |my )?(?:\+?\d+(?:\.\d+)?\s?% )?(?:weekly |week's |this week's |daily |target )?"
_G = r"(?:pace|goal|target)"
GAP_RE = re.compile(
    r"(?<!\bif )\bneed(?:s|ed)?\b(?:(?!" + _X + r")[^.\n]){0,40}?(?:[+-]?\d+(?:\.\d+)?\s?%|\$\s?\d[\d,]*)"
    r"|\bout of reach\b|\bunreachable\b|\bcan[’']?t be reached\b|\bcannot be reached\b"
    r"|\b\d+ (?:more )?sessions? (?:left|remaining|to go)\b"
    r"|\b(?:behind|ahead of|short of|off|below|under|back on) " + _D + _G + r"\b"
    r"|\b(?:gap|shortfall|distance) (?:to|from) " + _D + _G + r"\b"
    r"|\bcatch(?:ing)? up\b[^.\n]{0,30}?\b" + _G + r"\b|\b" + _G + r"\b[^.\n]{0,30}?\bcatch(?:ing)? up\b"
    r"|\bto (?:reach|hit|make|meet|get back to) " + _D + _G + r"\b"
    r"|\$\s?\d[\d,]* (?:gap|to go)\b|\bstill short of (?:about |~)?\$\s?\d"
    r"|\b(?:would|will) take (?:a |about |~)?\+?\d+(?:\.\d+)?\s?%"
    r"|\bincrease (?:the sleeve|equity) by\b", re.I)
PACE_RE = re.compile(r"(?<![\d.])\+?2\.14\s?%|\$?30,?642\b", re.I)
DOLLAR_RE = re.compile(r"move\$|sleeve%", re.I)
QUOTE_LINE = re.compile(r"^([A-Z][A-Z0-9 .]*): last ([\d.]+)")
BLIND = re.compile(r"\btt-[A-Za-z0-9-]+?-([0-9a-f]{10})\b")     # paper.new_ref's tt-<sleeve>-<hex>, the sleeve dropped

# ---------------------------------------------------------------- train4.py at 0f645b2, verbatim (--check-port)
RESEARCH_TOOLS = ("quote", "news", "stats", "forecast")
STOP_TYPES = ("STP", "STP LMT", "TRAIL")
LINE = {"09:00": 1.0, "10:30": 0.6, "13:00": 0.4, "15:30": 0.2}


def returned_symbols(name, args, out):
    """Symbols one research call returned data on, decided from its full output when it runs (R1's hard M8 rule,
    replay4.researched): the symbols tools.run queried, on a quote, news, stats or forecast call whose result is not an
    error or the research-limit answer, less any symbol whose own line is an error ("SYM error: ...")."""
    r = str(out or "").strip()
    if name not in RESEARCH_TOOLS or not r or r.startswith(("error", "research limit reached")):
        return set()
    a = args["arguments"] if isinstance(args, dict) and isinstance(args.get("arguments"), dict) else args
    bad = {l.split(" error:")[0].strip().upper() for l in r.split("\n") if " error:" in l}
    return set(tools.symbols_in(name, a)) - bad


def researched(trace):
    """Symbols the look's research calls returned data on (returned_symbols, recorded with each call)."""
    return {s for t in trace for s in t.get("researched", ())}


def kind(typ):
    """Order kinds a re-placement must keep: any stop-type order, else the exact type (MKT, MOO and MOC, or LMT and LOC,
    execute at different times)."""
    return "stop" if typ in STOP_TYPES else typ


def line(hhmm):
    """The near-the-market line for a non-protective order at a look, in range1 units: the distance at which a resting
    limit is touched before the close as often (about 15%) as one 1 range1 away from the 09:00 look, on the simulator's
    bars (rl5_line.py). Any time in force: the episode ends at the close. Between looks, the last look's line."""
    t = intrasim.minutes(hhmm) if hhmm else 0
    return min((v for k, v in LINE.items() if intrasim.minutes(k) <= t), default=1.0)


def substantive(typ, action, qty, limit, stop, trail_pct, q0, last, r1, hhmm=None):
    """Whether one order (placed now, or open and being cancelled) counts as acting for EMPTY_COST, and its level's
    distance from the last price in range1 units (None for a market-type order). Counts: a stop-type order covering at
    least half of the held position q0, and at least $1,000 of it, within 2 range1 (the stop credit's reach; a TRAIL by its
    trail %); or an order of at least $1,000 that is marketable (MKT/MOO/MOC) or prices within line() of the last price
    (LMT, LOC, a non-protective STP or STP LMT), in a symbol with market data (last). An STP LMT is judged by its stop, and
    only if its limit is on the fill side of the stop within 1 range1 (below it for a SELL, above for a BUY; else it
    rarely fills when triggered)."""
    if not last:
        return False, None
    r = r1 or 0.03
    q1 = q0 + (qty if action == "BUY" else -qty)
    covered = min(qty, abs(q0))
    protective = (typ in STOP_TYPES and abs(q0) > 0 and abs(q1) < abs(q0) and covered >= 0.5 * abs(q0)
                  and covered * last >= 1000)
    if typ == "TRAIL":
        d = (trail_pct or 0) / 100 / r if trail_pct else None
        return bool(protective and d is not None and d <= 2), d
    lvl = limit if typ in ("LMT", "LOC") else stop if typ in ("STP", "STP LMT") else None
    d = abs(lvl - last) / last / r if lvl else None
    if typ == "STP LMT" and not (limit and stop and 0 <= (stop - limit if action == "SELL" else limit - stop) <= last * r):
        return False, d                     # the limit on the fill side of the stop, within 1 range1 of it
    if protective:
        return d is not None and d <= 2, d
    return qty * (lvl or last) >= 1000 and (d is None or d <= line(hhmm)), d


def level(typ, action, limit, stop, trail_pct, last):
    """An order's working level for matching a re-placed order: a stop-type order's trigger (a TRAIL's from the last
    price), a limit order's limit, None for a market-type order."""
    if typ == "TRAIL":
        return last * (1 - trail_pct / 100 if action == "SELL" else 1 + trail_pct / 100) if trail_pct and last else None
    return stop if typ in ("STP", "STP LMT") else limit if typ in ("LMT", "LOC") else None


def replacements(cleaned, cancelled, last, r1):
    """B3b: (indices of placed orders, refs of cancelled ones) that only re-place what was cancelled. Per symbol, side and
    kind(), the placed orders whose level is within 0.25 range1 of a cancelled one's (or both market) and those cancelled
    orders form a re-placement when their total quantities are within 25% of each other, however the shares are split
    or merged. Such a set leaves the book as it was."""
    swaps, used = set(), set()
    keys = {(c["symbol"], c["action"], kind(c["type"])) for c in cleaned}
    for key in keys:
        s = key[0]
        px, r = last(s), r1.get(s) or 0.03
        P = [(i, level(c["type"], c["action"], c["limit"], c["stop"], c["trail_percent"], px), c["quantity"])
             for i, c in enumerate(cleaned) if (c["symbol"], c["action"], kind(c["type"])) == key]
        C = [(o["ref"], level(o["type"], o["action"], o.get("limit"), o.get("stop"), o.get("trail_percent"), px), o["quantity"])
             for o in cancelled if (o["symbol"], o["action"], kind(o["type"])) == key]
        near = lambda a, b: (a is None and b is None) or (a is not None and b is not None and bool(px)
                                                          and abs(a - b) / px / r <= 0.25)
        p = [x for x in P if any(near(x[1], y[1]) for y in C)]
        c = [y for y in C if any(near(x[1], y[1]) for x in P)]
        qp, qc = sum(x[2] for x in p), sum(y[2] for y in c)
        if p and c and abs(qp - qc) <= 0.25 * qc:
            swaps |= {x[0] for x in p}
            used |= {y[0] for y in c}
    return swaps, used


def offsets(cleaned, last, r1, hhmm=None, skip=()):
    """N1 (Trader Training's launch check): indices of placed orders that offset each other in one submit and so leave
    the book as it was. Per symbol, opposite sides that trigger together, with total quantities within 25% of each other:
    market legs filling at the same moment (MKT with MKT, MOO with MOO, MOC with MOC; before the open MKT and MOO both
    fill at the open), or resting limits and opposite stop-type orders whose levels are within 0.25 range1 (BUY LMT with
    SELL STP, SELL LMT with BUY STP). skip: orders already counted as re-placements."""
    out = set()
    pre = bool(hhmm) and intrasim.minutes(hhmm) < 570
    when = lambda t: "open" if pre and t in ("MKT", "MOO") else t
    for s in {c["symbol"] for c in cleaned}:
        px, r = last(s), r1.get(s) or 0.03
        legs = [i for i, c in enumerate(cleaned) if c["symbol"] == s and i not in skip]
        lv = lambda i: level(cleaned[i]["type"], cleaned[i]["action"], cleaned[i]["limit"], cleaned[i]["stop"],
                             cleaned[i]["trail_percent"], px)
        pairs = [([i for i in legs if when(cleaned[i]["type"]) == m and cleaned[i]["action"] == "BUY"],
                  [i for i in legs if when(cleaned[i]["type"]) == m and cleaned[i]["action"] == "SELL"], False)
                 for m in {when(t) for t in ("MKT", "MOO", "MOC")}]
        pairs += [([i for i in legs if cleaned[i]["type"] == "LMT" and cleaned[i]["action"] == side],
                   [i for i in legs if cleaned[i]["type"] in STOP_TYPES and cleaned[i]["action"] != side], True)
                  for side in ("BUY", "SELL")]
        for a, b, by_level in pairs:
            if by_level:
                near = lambda x, y: (lv(x) is not None and lv(y) is not None and bool(px)
                                     and abs(lv(x) - lv(y)) / px / r <= 0.25)
                a, b = [x for x in a if any(near(x, y) for y in b)], [y for y in b if any(near(x, y) for x in a)]
            qa, qb = sum(cleaned[i]["quantity"] for i in a), sum(cleaned[i]["quantity"] for i in b)
            if a and b and abs(qa - qb) <= 0.25 * max(qa, qb):
                out |= set(a) | set(b)
    return out


def order_stats(call, st, world, trace=(), r1=None, hhmm=None):
    """Process facts about one decision against the book it saw: orders taking a name to 50%+ of the sleeve (M1), orders
    under $1,000 (M9) and entries among them, entries without a protective order for the same name in the same submit,
    protective (stop-type) orders, symbols of exposure-adding orders (stop-type entries included) with no data-returning
    quote/news/stats/forecast call in the look (hard M8) and of any order with no such call (M8x), orders missing p_fill
    or p_win, and two empty flags: `empty`, v1.1's (and rl4's) empty submit, nothing placed and nothing cancelled (a look
    with no valid submit too); and `inactive`, EMPTY_COST's, no substantive order placed and no cancel of an open order
    that would itself be substantive now. A stop is judged against the position held at the look (not one an unfilled
    order of the same submit would make). Not counted: a re-placement (replacements()); opposite orders in one symbol
    that offset each other (offsets(); neither entries nor protective); a protective stop placed while
    the position's protective stops (working, less those cancelled now, plus those placed before it in this submit)
    already cover half of it; cancelling a protective stop while the others still cover half. Token actions (a few
    shares, levels far from the market, unfillable symbols, duplicate or re-placed orders) are inaction in disguise (the
    analysis session and Trader Training, 24 Oct) and count as token_orders. r1: {symbol: range1 as a fraction}."""
    seen = researched(trace)
    r1 = r1 or {}
    nav = st["value"] or 1.0
    held = {p["symbol"]: p["quantity"] for p in st["positions"]}
    pos = dict(held)
    last = world.m.last
    open_by_ref = {o["ref"]: o for o in st["orders"]}
    cancelled = [open_by_ref[r] for r in dict.fromkeys(str(x) for x in ((call or {}).get("cancel") or [])) if r in open_by_ref]
    place = [o for o in (call or {}).get("place", []) if isinstance(o, dict)]
    cleaned = [c for c, err in (trader.clean_order(o) for o in place) if not err]
    swaps, used = replacements(cleaned, cancelled, last, r1)
    offs = offsets(cleaned, last, r1, hhmm, swaps)

    def guards(typ, action, qty, lim, stop, trail, s):
        """A protective stop within reach on the held position."""
        q0 = held.get(s, 0)
        return (typ in STOP_TYPES and abs(q0 + (qty if action == "BUY" else -qty)) < abs(q0)
                and substantive(typ, action, qty, lim, stop, trail, q0, last(s), r1.get(s), hhmm)[0])

    gone = {o["ref"] for o in cancelled}
    cover = {}                               # (symbol, side): shares under protective stops that stay working
    for o in st["orders"]:
        if o["ref"] not in gone and guards(o["type"], o["action"], o["quantity"], o.get("limit"), o.get("stop"),
                                           o.get("trail_percent"), o["symbol"]):
            cover[(o["symbol"], o["action"])] = cover.get((o["symbol"], o["action"]), 0) + o["quantity"]
    real_cancels = 0
    for o in cancelled:
        if o["ref"] in used:
            continue
        s = o["symbol"]
        ok, _ = substantive(o["type"], o["action"], o["quantity"], o.get("limit"), o.get("stop"), o.get("trail_percent"),
                            held.get(s, 0), last(s), r1.get(s), hhmm)
        if ok and guards(o["type"], o["action"], o["quantity"], o.get("limit"), o.get("stop"), o.get("trail_percent"), s):
            ok = cover.get((s, o["action"]), 0) < 0.5 * abs(held.get(s, 0))       # the rest still protect: redundant
        real_cancels += ok
    big = small = small_entries = entries = unprotected_entries = prot = stop_type = no_probs = subst = far = dup = 0
    notional = entry_notional = 0.0
    lim_d = []
    ordered, adding = set(), set()
    stops = {(c["symbol"], c["action"]) for c in cleaned if c["type"] in STOP_TYPES}
    for i, c in enumerate(cleaned):
        s, qty = c["symbol"], c["quantity"]
        ordered.add(s)
        no_probs += c["p_fill"] is None or c["p_win"] is None
        px = c["limit"] or c["stop"] or last(s) or 0.0
        q0 = pos.get(s, 0)
        q1 = q0 + (qty if c["action"] == "BUY" else -qty)
        small += qty * px < 1000
        notional += qty * px
        stopq = held.get(s, 0) if c["type"] in STOP_TYPES else q0
        ok, d = substantive(c["type"], c["action"], qty, c["limit"], c["stop"], c["trail_percent"], stopq, last(s),
                            r1.get(s), hhmm)
        if guards(c["type"], c["action"], qty, c["limit"], c["stop"], c["trail_percent"], s):
            if i not in swaps and cover.get((s, c["action"]), 0) >= 0.5 * abs(held.get(s, 0)):
                ok, dup = False, dup + 1
            cover[(s, c["action"])] = cover.get((s, c["action"]), 0) + qty
        ok = ok and i not in swaps and i not in offs
        subst += ok
        far += not ok
        if d is not None and c["type"] in ("LMT", "LOC", "STP LMT"):
            lim_d.append(d)
        if abs(q1) > abs(q0):
            adding.add(s)
        if c["type"] in STOP_TYPES:
            stop_type += 1
            prot += abs(q1) < abs(q0) and i not in offs
            continue
        if abs(q1) > abs(q0) and i not in offs:
            entries += 1
            small_entries += qty * px < 1000
            entry_notional += qty * px
            unprotected_entries += (s, "SELL" if q1 > 0 else "BUY") not in stops
            big += abs(q1) * px >= 0.5 * nav
        pos[s] = q1
    return {"orders": len(place), "orders_ex_protective": len(place) - prot, "held": len(st["positions"]),
            "m1_big": big, "under_1k": small, "small_entries": small_entries, "entries": entries,
            "entries_no_stop": unprotected_entries, "protective": prot, "stop_type": stop_type, "no_probs": no_probs,
            "unresearched": len(adding - seen), "unresearched_any": len(ordered - seen),
            "empty": not place and not (call or {}).get("cancel"),
            "inactive": call is not None and not subst and not real_cancels, "token_orders": len(cleaned) - subst,
            "stop_replace": sum(1 for i in swaps if cleaned[i]["type"] in STOP_TYPES), "replaced": len(swaps),
            "dup_stops": dup, "offset_orders": len(offs), "far_orders": far, "notional": notional, "entry_notional_nav": entry_notional / nav,
            "limit_dist_r1": lim_d, "notes": bool(str((call or {}).get("notes") or "").strip())}

# ---------------------------------------------------------------- end of the vendored code


replay4 = None                       # frozen_replay4(), set by main()


def frozen_replay4(orders):
    """replay4.py as frozen at FROZEN (rl6_criteria.md A0), from git, with the week's own order ledger (the
    cancel-pending state the account block shows) and today's daily bars."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{FROZEN}:replay4.py"], capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    mod = types.ModuleType("replay4_frozen")
    mod.__file__ = str(ROOT / "replay4.py")
    exec(compile(src, f"replay4.py@{FROZEN}", "exec"), mod.__dict__)
    mod.ORDERS = orders
    mod._DATA[:] = [trader.Data()]
    return mod


def ledgers(week):
    """(sleeves, fills.jsonl, orders.jsonl, daily.jsonl) of a week: the live ones, or an archived week's."""
    if week == "current":
        return (json.loads((RUNS / "sleeves.json").read_text(encoding="utf-8")), RUNS / "fills.jsonl",
                RUNS / "orders.jsonl", LIVE / "daily.jsonl")
    a = RUNS / "archive" / week
    return (json.loads((a / "sleeves.json").read_text(encoding="utf-8")), a / "fills.jsonl", a / "orders.jsonl",
            a / "daily.jsonl")


def et(iso):
    from zoneinfo import ZoneInfo
    return datetime.fromisoformat(iso).astimezone(ZoneInfo("America/New_York"))


def week_days(sleeves):
    """Monday to Friday of the week the sleeves started for (a weekend start counts for the next Monday)."""
    d = min(et(m["start"]).date() for m in sleeves.values())
    monday = d + timedelta(days=(7 - d.weekday()) % 7)
    return [monday + timedelta(days=i) for i in range(5)]


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def slot_of(hhmm):
    """The scheduled look a file's minute belongs to (a delayed look keeps its slot)."""
    return max((s for s in SLOTS if s <= hhmm), default=hhmm)


def log_times():
    """{(date, sleeve, slot): (logged, seconds)} from live.log's per-sleeve lines ("name: ...; 123s"), each written after
    that look's orders were placed; the slot is the one of the latest "look-HHMM starting" line."""
    out, slot = {}, None
    for line in (LIVE / "live.log").read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ET  look-(\d{4}) starting", line)
        if m:
            slot = m.group(2)
            continue
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ET  ([\w-]+): .*?; (\d+)s(?:;|$)", line)
        if m and slot and "DRY RUN" not in line:
            out[(m.group(1)[:10], m.group(2), slot)] = (datetime.fromisoformat(m.group(1)), int(m.group(3)))
    return out


def r1_for(day):
    """{symbol: range1 as a fraction} from the forecast file as of the last close before day."""
    best = None
    for f in (RUNS / "ml" / "forecast").glob("fc_*.json"):
        asof = f.stem[3:]
        if asof < day.strftime("%Y%m%d") and (best is None or asof > best.stem[3:]):
            best = f
    if best is None:
        return {}
    rows = json.loads(best.read_text(encoding="utf-8")).get("forecasts", {})
    return {s: r["range1_pct"] / 100 for s, r in rows.items() if r.get("range1_pct")}


def args_of(t):
    a = t.get("args")
    if isinstance(a, str):
        try:
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            a = {}
    return a if isinstance(a, dict) else {}


def analysis(said):
    return re.sub(r"<\|[^|]+\|>", " ", str(said or "").split("<|channel|>commentary to=")[0])


def reasoning_of(lk):
    """gpt-oss: the submit turn's text before its call, as the frozen scorer's score_live splits it; Claude: the CLI's
    final message (its stored trace keeps no reasoning). M3's input, as frozen."""
    for t in reversed(lk.get("trace") or []):
        if t.get("tool") == "submit":
            said = str(t.get("said") or "").split("<|channel|>commentary to=functions.submit")[0]
            return re.sub(r"<\|[^|]+\|>", " ", said)
        if "cli_result" in t:
            return str(t.get("cli_result") or "")
    return ""


def decision_text(lk):
    """gap's and the dollar line's text: reasoning_of, or when the submit turn was prefilled (a research limit, a
    malformed answer) and so holds no analysis, the last earlier step's analysis."""
    r = reasoning_of(lk)
    if r.strip():
        return r
    prev = [t for t in (lk.get("trace") or []) if "said" in t and t.get("tool") != "submit"]
    return analysis(prev[-1]["said"]) if prev else ""


def failure_of(call, trace):
    """(framework_failure, which) for a look with no decision: an error entry the framework wrote (live.py's, with no
    agent step), not one of the model's own (malformed, prose, context full, no submit within the step limit)."""
    errs = [str(t["error"]) for t in trace if "error" in t and "step" not in t
            and not str(t["error"]).startswith(("placing:", "no submit within the step limit"))]
    if call is not None or not errs:
        return False, None
    e = errs[-1]
    return True, ("timeout" if "TimeoutError" in e else "did not finish" if "did not finish" in e else "exception")


def stats_vols(lk):
    """{symbol: hist_vol %} from the look's stats results."""
    out = {}
    for t in lk.get("trace") or []:
        if t.get("tool") != "stats":
            continue
        lines = str(t.get("result") or "").split("\n")
        if not lines or "hist_vol%" not in lines[0]:
            continue
        for l in lines[1:]:
            x = l.split()
            if len(x) >= 2 and re.fullmatch(r"[\d.]+", x[1]):
                out[x[0]] = float(x[1])
    return out


TOOL_TAIL = re.compile(r"\n\((?:research call \d+ of \d+|the last 60 of \d+ bars)\)\s*$")


def frozen_view(lk):
    """The look as the frozen price reader (replay4.prices at d59537b) expects its tool text: without the lines the
    tools have appended since 26 Oct ("(research call k of 12)", "(the last 60 of N bars)"), and with scanner rows cut to
    their first four fields (rank sym last today%), before the names added then. An input adaptation, like QUOTE_LINE;
    the reader itself is unchanged."""
    out = []
    for t in lk.get("trace") or []:
        r = t.get("result")
        if isinstance(r, str):
            while TOOL_TAIL.search(r):
                r = TOOL_TAIL.sub("", r)
            if t.get("tool") == "scanner":
                r = "\n".join(" ".join(l.split()[:4]) for l in r.split("\n"))
            t = dict(t, result=r)
        out.append(t)
    return dict(lk, trace=out)


def look_row(lk, day, slot, started, times, r1, sleeve_meta, sessions_before):
    call, st = lk.get("call"), lk["state"]
    raw = lk.get("trace") or []
    trace = []
    for t in raw:
        if t.get("tool") in RESEARCH_TOOLS:
            t = dict(t, researched=sorted(returned_symbols(t["tool"], args_of(t), t.get("result"))))
        trace.append(t)
    try:
        px = replay4.prices(frozen_view(lk))
    except Exception:
        px = {}
    for t in raw:                           # week 3's quote wording, which the frozen reader predates
        if t.get("tool") == "quote":
            for l in str(t.get("result") or "").split("\n"):
                m = QUOTE_LINE.match(l)
                if m:
                    px[m.group(1)] = float(m.group(2))
    px |= {p["symbol"]: p["close"] for p in st["positions"] if p.get("close")}
    world = SimpleNamespace(m=SimpleNamespace(last=lambda s: px.get(s)))
    hhmm = f"{started[:2]}:{started[2:]}"               # the model's header showed the actual minute
    os_ = order_stats(call, st, world, trace, r1, hhmm) if call is not None else {}
    notes = str((call or {}).get("notes") or "")
    reasoning, text = reasoning_of(lk), decision_text(lk)
    fail, why = failure_of(call, raw)
    claude = str(lk.get("model") or "").startswith("claude-")
    row = {"date": day.isoformat(), "sleeve": lk["sleeve"], "model": lk.get("model"), "slot": f"{slot[:2]}:{slot[2:]}",
           "started": started, "file": lk["file"], "secs": round(lk.get("seconds") or 0), "value": st["value"],
           "retried": sum("sample timed out" in str(t.get("error", "")) for t in raw),
           "framework_failure": fail, "failure": why,
           "place_failed": any(str(t.get("error", "")).startswith("placing:") for t in raw),
           "no_decision": call is None,
           "forced": any(str(t.get("result", "")).startswith("research limit reached") for t in raw),
           "research_calls": sum(1 for t in raw if t.get("tool") not in (None, "submit")),
           "conc75_unprotected": replay4.state_conc75(lk),
           "margin_rejected": sum("insufficient margin" in str(n) for n in lk.get("placed_notes") or []),
           "gap": len(GAP_RE.findall(notes + "\n" + text)), "gap_notes": len(GAP_RE.findall(notes)),
           "pace_restated": len(PACE_RE.findall(notes + "\n" + text)),
           "dollar_cited": bool(DOLLAR_RE.search(notes + "\n" + text))}
    tm = times.get((day.isoformat(), lk["sleeve"], slot))
    if tm:
        sched = datetime.combine(day, datetime.min.time()) + timedelta(hours=int(slot[:2]), minutes=int(slot[2:]))
        row["acts_min"] = round((tm[0] - sched).total_seconds() / 60, 1)
        if not claude:
            row["start_min"] = round((tm[0] - timedelta(seconds=tm[1]) - sched).total_seconds() / 60, 1)
    row |= {k: os_.get(k) for k in ("orders", "entries", "entries_no_stop", "protective", "held", "under_1k",
                                    "token_orders", "unresearched", "unresearched_any", "empty", "inactive",
                                    "offset_orders", "dup_stops")}
    held = {p["symbol"]: p["quantity"] for p in st["positions"]}
    entries = []                            # (cleaned order, notional) of the non-stop orders adding exposure
    if call is not None:
        cleaned = [c for c, err in (trader.clean_order(o) for o in call.get("place") or [] if isinstance(o, dict)) if not err]
        open_by_ref = {o["ref"]: o for o in st["orders"]}
        cancelled = [open_by_ref[r] for r in dict.fromkeys(str(x) for x in (call.get("cancel") or [])) if r in open_by_ref]
        swaps, _ = replacements(cleaned, cancelled, world.m.last, r1)
        offs = offsets(cleaned, world.m.last, r1, hhmm, swaps)
        row["protective_held"] = sum(1 for i, c in enumerate(cleaned) if c["type"] in STOP_TYPES and i not in offs
                                     and abs(held.get(c["symbol"], 0) + (c["quantity"] if c["action"] == "BUY"
                                                                         else -c["quantity"])) < abs(held.get(c["symbol"], 0)))
        pos = dict(held)
        for c in cleaned:
            q0 = pos.get(c["symbol"], 0)
            q1 = q0 + (c["quantity"] if c["action"] == "BUY" else -c["quantity"])
            if c["type"] not in STOP_TYPES:
                if abs(q1) > abs(q0):
                    pxo = c["limit"] or c["stop"] or px.get(c["symbol"])
                    entries.append((c, c["quantity"] * pxo if pxo else None))
                pos[c["symbol"]] = q1
        try:
            w = replay4.walk(dict(lk, _cancel=call.get("cancel") or []), call, px)
            row["m3"] = sorted(replay4.misreads(lk, call, w, reasoning))     # the frozen scorer's M3
            row["m3n"] = sorted(replay4.misreads(lk, call, w))               # its M3n: the notes alone
            row["m3q"] = replay4.qty_misreads(w)                             # its M3q: a close sent the wrong way
        except Exception as e:
            row["m3_error"] = f"{type(e).__name__}: {e}"
    if day.weekday() == 4 and slot >= "1200":            # Friday afternoon, behind the pace line
        line_ = sleeve_meta["start_cash"] * (1 + PACE / 5 * sessions_before)
        if st["value"] < line_:
            row["fri_behind"] = True
            row["fri_behind_line"] = round(line_, 2)
            row["fri_behind_entries"] = os_.get("entries", 0)
            row["fri_behind_notional"] = round(os_.get("entry_notional_nav", 0) * st["value"], 2)
    if lk["sleeve"].endswith("-ml") and call is not None:   # the forecast's dollar line
        costed, returned = {}, set()
        for t in trace:
            if t.get("tool") == "forecast":
                a = args_of(t)
                for s in t.get("researched", ()):
                    returned.add(s)
                    costed[s] = float(a.get("notional_usd") or forecast.DEFAULT_NOTIONAL)
        segs = [s for s in re.split(r"\n|(?<=[.;])\s", notes + "\n" + text) if DOLLAR_RE.search(s)]
        row["ml_entries"] = len(entries)
        row["costed_entries"] = sum(1 for c, v in entries if c["symbol"] in costed and v
                                    and abs(v / costed[c["symbol"]] - 1) <= 0.10)
        row["line_cited_entries"] = sum(1 for c, _ in entries if c["symbol"] in returned and any(
            re.search(rf"(?<![A-Za-z0-9]){re.escape(c['symbol'])}(?![A-Za-z0-9])", s) for s in segs))
    return row


def risk_top(positions, day, data, marks, svol):
    """The largest position carried overnight and its one-sigma day move in dollars (the risk line's arithmetic)."""
    if not positions:
        return None
    key = day.strftime("%Y%m%d")
    i = data.ix[key] + 1 if key in data.ix else len(data.dates)
    feats = data.features(i)[0] if i <= len(data.dates) else {}
    best = None
    for s, q in positions.items():
        px = None
        p = ROOT / "data" / "bars" / f"{s}.json"
        if p.exists():
            px = next((b["close"] for b in json.loads(p.read_text())["bars"] if b["date"] == key), None)
        px = px or marks.get(s)
        if px and (best is None or abs(q) * px > best[2]):
            best = (s, q, abs(q) * px)
    if best is None:
        return {"symbol": None, "note": "no price"}
    s, q, v = best
    vol, src = (feats.get(s) or {}).get("vol"), "vol column"
    if vol is None or not math.isfinite(vol):
        vol, src = (data.outside_features(i, [s]).get(s) or {}).get("vol"), "outside"
    if (vol is None or not math.isfinite(vol)) and s in svol:
        vol, src = svol[s] / math.sqrt(252), "stats hist_vol"
    if vol is None or not math.isfinite(vol):
        vol, src = None, None
    return {"symbol": s, "side": "LONG" if q > 0 else "SHORT", "value": round(v, 2), "vol_source": src,
            "one_sigma_pct": None if vol is None else round(vol, 2),
            "one_sigma_usd": None if vol is None else round(v * vol / 100, 2)}


def spy_baseline(monday, first_day, daily):
    """SPY's close before the sleeve's first look: decks.json's baseline for a Monday start, else the record's."""
    if first_day == monday:
        decks = json.loads((RUNS / "decks.json").read_text(encoding="utf-8"))
        for d in decks.values():
            if d.get("start") == monday.isoformat():
                return d["spy_baseline"]
    prev = [r for r in daily if r["date"] < first_day.isoformat() and r.get("spy")]
    return prev[-1]["spy"] if prev else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default="current")
    ap.add_argument("--check-port", action="store_true")
    a = ap.parse_args()
    if a.check_port:
        return check_port()
    global replay4
    sleeves, fills_path, orders_path, daily_path = ledgers(a.week)
    days = week_days(sleeves)
    name = a.week if a.week != "current" else f"week-{days[0].isoformat()}"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    replay4 = frozen_replay4(orders_path)
    data = replay4._DATA[0]
    times = log_times()
    daily = list({(r["sleeve"], r["date"]): r for r in jsonl(daily_path)}.values())   # a re-run record: the last row
    fills = jsonl(fills_path)
    looks, marks, svols = [], {}, {}
    for sl, meta in sleeves.items():
        first = None
        for i, day in enumerate(days):
            if day > date.today():
                break
            r1 = r1_for(day)
            mine = []
            for f in sorted((LIVE / day.isoformat()).glob(f"{sl}-[0-9][0-9][0-9][0-9].json")):
                lk = json.loads(f.read_text(encoding="utf-8")) | {"file": f"{day.isoformat()}/{f.name}"}
                if lk.get("dry_run") or lk.get("sleeve") != sl:
                    continue
                first = first or day
                done = sum(1 for d in days[:i] if first <= d)
                mine.append(look_row(lk, day, slot_of(f.stem[-4:]), f.stem[-4:], times, r1, meta, done))
                marks[(sl, day.isoformat())] = {p["symbol"]: p["close"] for p in lk["state"]["positions"]}
                svols.setdefault((sl, day.isoformat()), {}).update(stats_vols(lk))
            for r in mine:
                if sum(1 for x in mine if x["slot"] == r["slot"]) > 1:
                    r["slot_dup"] = True
            looks += mine
        for day in days:                    # a name bought after the day's last look: its latest fill price by then
            got, key, last_fill = marks.setdefault((sl, day.isoformat()), {}), day.strftime("%Y%m%d"), {}
            for f in sorted((f for f in fills if f.get("sleeve") == sl), key=lambda f: paper.et_date(f["time"]) + str(f["time"])):
                if paper.et_date(f["time"]) <= key:
                    last_fill[f["symbol"]] = f["price"]
            for s_, px_ in last_fill.items():
                got.setdefault(s_, px_)
    with (out / "looks.jsonl").open("w", encoding="utf-8") as fh:
        for r in looks:
            fh.write(json.dumps(r, default=str) + "\n")
    rows = []
    for sl, meta in sleeves.items():
        mine = sorted((r for r in daily if r["sleeve"] == sl and days[0].isoformat() <= r["date"] <= days[-1].isoformat()),
                      key=lambda r: r["date"])
        my_looks = [r for r in looks if r["sleeve"] == sl]
        if not mine:
            continue
        first = date.fromisoformat(min(r["date"] for r in my_looks)) if my_looks else date.fromisoformat(mine[0]["date"])
        base = spy_baseline(days[0], first, daily)
        prev_nav, prev_spy = meta["start_cash"], base
        for r in mine:
            day = date.fromisoformat(r["date"])
            row = {"date": r["date"], "sleeve": sl, "nav": round(r["nav"], 2),
                   "day_pct": round(100 * (r["nav"] / prev_nav - 1), 3),
                   "cum_pct": round(100 * (r["nav"] / meta["start_cash"] - 1), 3),
                   "realized": round(r["realized"], 2), "commissions": round(r["commissions"], 2), "fills": r["fills"],
                   "spy_day_pct": round(100 * (r["spy"] / prev_spy - 1), 3) if r.get("spy") and prev_spy else None,
                   "spy_cum_pct": round(100 * (r["spy"] / base - 1), 3) if r.get("spy") and base else None,
                   "risk_top": risk_top(r["positions"], day, data, marks.get((sl, r["date"]), {}),
                                        svols.get((sl, r["date"]), {}))}
            prev_nav, prev_spy = r["nav"], r.get("spy") or prev_spy
            rows.append(row)
    with (out / "days.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    m3_cases(looks, out, sleeves)
    (out / "summary.md").write_text(summary(name, sleeves, looks, rows), encoding="utf-8")
    print(f"{len(looks)} looks, {len(rows)} sleeve-days -> {out}")


def m3_cases(looks, out, sleeves):
    """A0: every M3 or M3n candidate, blinded (no sleeve or model name, order refs as ref-<hex>, shuffled), with its key
    kept apart."""
    cases = []
    for r in looks:
        flags = sorted(set(r.get("m3") or []) | set(r.get("m3n") or []))
        if not flags:
            continue
        lk = json.loads((LIVE / r["file"]).read_text(encoding="utf-8"))
        try:
            account = replay4.parts(lk)[2]
        except Exception:
            account = json.dumps(lk["state"])
        case = {"symbols": flags, "account": account, "notes": (lk.get("call") or {}).get("notes"),
                "orders": (lk.get("call") or {}).get("place"), "cancels": (lk.get("call") or {}).get("cancel"),
                "reasoning": reasoning_of(lk) if set(r.get("m3") or []) - set(r.get("m3n") or []) else None}
        case = json.loads(BLIND.sub(r"ref-\1", json.dumps(case)))
        assert "tt-" not in json.dumps(case), r["file"]
        cases.append({"key": {"file": r["file"], "sleeve": r["sleeve"], "model": r["model"]}, "case": case})
    random.Random(SEED).shuffle(cases)
    names = set(sleeves) | {str(m.get("model")) for m in sleeves.values()}
    for i, c in enumerate(cases):
        hit = [n for n in names if re.search(rf"(?<![A-Za-z0-9-]){re.escape(n)}(?![A-Za-z0-9-])", json.dumps(c["case"]))]
        if hit:
            print(f"m3 case {i}: names {hit} appear in its text (check before the audit)")
    (out / "m3_cases.json").write_text(json.dumps([dict(c["case"], id=i) for i, c in enumerate(cases)], indent=1),
                                       encoding="utf-8")
    (out / "m3_key.json").write_text(json.dumps([dict(c["key"], id=i) for i, c in enumerate(cases)], indent=1),
                                     encoding="utf-8")


def summary(name, sleeves, looks, rows):
    def tot(rs, k):
        return sum(r.get(k) or 0 for r in rs)

    def med_max(xs):
        xs = [x for x in xs if x is not None]
        return f"{statistics.median(xs):.1f} / {max(xs):.1f}" if xs else ""
    L = [f"# {name}: live comparison (rl6_criteria.md section 6)", "",
         "Counts, not triggers. M3 and line-cited candidates count only after a blinded read (m3_cases.json for M3).", "",
         "| sleeve | looks | no decision (framework) | forced | retried | empty | inactive | orders | entries | "
         "entries, no stop | protective / held | protective on held | under $1k | token | unresearched M8 / M8x | "
         "margin rejected | conc75 unprotected | M3 / M3n / M3q cand. | gap looks (matches) | gap in notes | pace restated | "
         "Fri pm behind pace: looks / entries / notional | -ml costed / entries | -ml line-cited cand. | -ml dollar cited | "
         "acts min, median / max |",
         "|" + "---|" * 26]
    for sl in sleeves:
        rs = [r for r in looks if r["sleeve"] == sl]
        if not rs:
            continue
        held = tot(rs, "held")
        cells = [sl, len(rs), f"{tot(rs, 'no_decision')} ({tot(rs, 'framework_failure')})", tot(rs, "forced"),
                 tot(rs, "retried"), tot(rs, "empty"), tot(rs, "inactive"), tot(rs, "orders"), tot(rs, "entries"),
                 tot(rs, "entries_no_stop"),
                 f"{tot(rs, 'protective')}/{held} = {tot(rs, 'protective') / held if held else 0:.2f}",
                 tot(rs, "protective_held"), tot(rs, "under_1k"), tot(rs, "token_orders"),
                 f"{tot(rs, 'unresearched')} / {tot(rs, 'unresearched_any')}", tot(rs, "margin_rejected"),
                 tot(rs, "conc75_unprotected"),
                 f"{sum(len(r.get('m3') or []) for r in rs)} / {sum(len(r.get('m3n') or []) for r in rs)} / {tot(rs, 'm3q')}",
                 f"{sum(1 for r in rs if r.get('gap'))} ({tot(rs, 'gap')})", tot(rs, "gap_notes"), tot(rs, "pace_restated"),
                 f"{tot(rs, 'fri_behind')} / {tot(rs, 'fri_behind_entries')} / ${tot(rs, 'fri_behind_notional'):,.0f}"]
        if sl.endswith("-ml"):
            cells += [f"{tot(rs, 'costed_entries')} / {tot(rs, 'ml_entries')}", tot(rs, "line_cited_entries"),
                      sum(1 for r in rs if r.get("dollar_cited"))]
        else:
            cells += ["", "", ""]
        cells.append(med_max([r.get("acts_min") for r in rs]))
        L.append("| " + " | ".join(str(c) for c in cells) + " |")
    L += ["", "| date | sleeve | NAV | day % | cum % | SPY cum % | realized | commissions | fills | largest overnight "
          "position | its one-sigma day move (source) |", "|" + "---|" * 11]
    for r in rows:
        rt = r["risk_top"] or {}
        top = ("flat" if r["risk_top"] is None else "no price" if not rt.get("symbol")
               else "{} {} ${:,.0f}".format(rt["symbol"], rt["side"], rt["value"]))
        move = ("" if not rt.get("symbol") else "vol unknown" if rt.get("one_sigma_usd") is None
                else "{:.1f}% = ${:,.0f} ({})".format(rt["one_sigma_pct"], rt["one_sigma_usd"], rt["vol_source"]))
        spy = "" if r["spy_cum_pct"] is None else format(r["spy_cum_pct"], "+.2f")
        L.append(f"| {r['date']} | {r['sleeve']} | {r['nav']:,.2f} | {r['day_pct']:+.2f} | {r['cum_pct']:+.2f} | "
                 f"{spy} | {r['realized']:+,.2f} | {r['commissions']:,.2f} | {r['fills']} | {top} | {move} |")
    L += ["", "Paired by scheduled slot (each cell: orders/entries/no-stop entries; acts_min in looks.jsonl says when each "
          "sleeve's orders went out):", ""]
    slots = sorted({(r["date"], r["slot"]) for r in looks})
    names = [s for s in sleeves if any(r["sleeve"] == s for r in looks)]
    L += ["| date | slot | " + " | ".join(names) + " |", "|" + "---|" * (2 + len(names))]
    for d, s in slots:
        cells = []
        for n in names:
            rs = [x for x in looks if x["date"] == d and x["slot"] == s and x["sleeve"] == n]
            cells.append(" + ".join("none" if r["no_decision"] else
                                    f"{r.get('orders') or 0}/{r.get('entries') or 0}/{r.get('entries_no_stop') or 0}"
                                    for r in rs))
        L.append(f"| {d} | {s} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def check_port():
    """The vendored functions against train4.py at PORT, source text for source text."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{PORT[0]}:{PORT[1]}"], capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    mine = Path(__file__).read_text(encoding="utf-8")
    theirs = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    ours = {n.name: ast.get_source_segment(mine, n) for n in ast.parse(mine).body if isinstance(n, ast.FunctionDef)}
    names = ["returned_symbols", "researched", "kind", "line", "substantive", "level", "replacements", "offsets", "order_stats"]
    bad = [n for n in names if ours.get(n) != theirs.get(n)]
    consts = {k: (eval(k, globals()), ast.literal_eval(next(ast.get_source_segment(src, x.value) for x in ast.parse(src).body
              if isinstance(x, ast.Assign) and getattr(x.targets[0], "id", "") == k))) for k in ("RESEARCH_TOOLS", "STOP_TYPES", "LINE")}
    bad += [k for k, (x, y) in consts.items() if x != y]
    print("port identical to", ":".join(PORT) if not bad else f"DIFFERS: {bad}")
    return not bad


if __name__ == "__main__":
    main()
