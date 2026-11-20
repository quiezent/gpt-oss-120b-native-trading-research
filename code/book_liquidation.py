"""Book IBKR's margin liquidation of Mon 16 Nov 2026 into the sleeves' ledgers and tell each model what happened.

What happened: the whole paper account fell below IBKR's margin on Mon 16 Nov 2026 (gross $4.79M on $1.39M of equity at
15:35 ET, after a day with no software exposure limit). Clayton, asked at 15:37 ET: "Leave it to IBKR". At 15:51:03-15:51:12
ET IBKR closed positions on its own: 39 fills in 12 names, client_id 0, no ref, liquidation=1 (the TWS Gateway Owner saved
them read-only: C:\\Claude\\TWS-Gateway-handoff\\liquidation_20261116.json). Such fills never reach client 10, so no
sleeve's ledger has them: each affected sleeve still books the positions IBKR closed. Of the sleeves' orders in those names
only one still rested at IBKR on the evening of 16 Nov (base-ml's SELL 66 XLE TRAIL); the loop reports the others' ends as
usual (paper.events_since_last_look). Positions are taken as each ledger stood one second before IBKR's first fill.

The rule (the TWS Gateway Owner, 16 Nov: "book to the sleeves only up to each sleeve's position in the name, with the
opposite sign. The excess belongs to no sleeve."): per name, IBKR's net quantity closes the sleeves whose own position it
reduces (a net buy closes shorts, a net sell closes longs), up to each sleeve's position; a sleeve on the other side of the
name is not touched. When IBKR's quantity covers every such sleeve (11 of the 12 names), each is closed in full. When it
covers part (SPY: 134 of 216), it is shared in proportion to the positions (largest remainder; ties to the larger
position, then by sleeve name). What IBKR traded beyond the sleeves' positions went into the reset's stale listing (IBKR's
books had closed those at the reset) and is owned by no sleeve: it is written to runs/live/liquidation_20261116_unowned.json.
Each sleeve's fill is one row at IBKR's volume-weighted price for the name and its share of IBKR's commission, at the
time of IBKR's last fill in the name, with exec_id liq-20261116-<SYM>-<sleeve> (so it is booked once) and ref
ibkr-liquidation-20261116-<SYM> (the model sees it in "Since your last decision" as a fill under that ref).
Each affected model also gets one framework note: what IBKR did and why, what it closed in that sleeve, and which of its
orders still rest in those names (they were placed for the positions IBKR closed). Nothing is placed or cancelled.

    python book_liquidation.py           # show the allocation, the notes and the unowned excess; change nothing
    python book_liquidation.py --go      # append the fills to runs/fills.jsonl, the notes to runs/notes.json, write the file
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import paper  # noqa: E402

SRC = Path(r"C:\Claude\TWS-Gateway-handoff\liquidation_20261116.json")
DAY = "20261116"
UNOWNED = ROOT / "runs" / "live" / "liquidation_20261116_unowned.json"
REF = "ibkr-liquidation-" + DAY + "-{sym}"
WHEN = ("Mon 16 Nov", "15:51")                 # the liquidation's day and time, for the model's note


def configure(day, src):
    """Another day's liquidation (17 Nov: written for Monday's, made general for the next): its fills file (a JSON object
    with "fills", rows as the Owner's 16 Nov file: time, client_id 0, liquidation 1, symbol, side, shares, price,
    commission, exec_id) and its day YYYYMMDD; the exec_ids, refs, unowned file and note follow the day."""
    global SRC, DAY, UNOWNED, REF, WHEN
    SRC, DAY = Path(src), str(day)
    UNOWNED = ROOT / "runs" / "live" / f"liquidation_{DAY}_unowned.json"
    REF = "ibkr-liquidation-" + DAY + "-{sym}"
    fills = json.loads(SRC.read_text(encoding="utf-8"))["fills"]
    import reopen_week6
    t = reopen_week6.fill_time(min(f["time"] for f in fills)).astimezone(reopen_week6.ET)
    WHEN = (f"{t:%a %d %b}", f"{t:%H:%M}")


def signed(f):
    return f["shares"] if f["side"] == "BOT" else -f["shares"]


def share_out(n, wants):
    """n shares among {sleeve: |position|} in proportion, whole shares, largest remainder; ties to the larger position,
    then by name. n >= sum(wants) gives each its whole position."""
    total = sum(wants.values())
    if n >= total:
        return dict(wants)
    base = {s: (n * w) // total for s, w in wants.items()}
    left = n - sum(base.values())
    order = sorted(wants, key=lambda s: (-((n * wants[s]) % total), -wants[s], s))
    for s in order[:left]:
        base[s] += 1
    return {s: q for s, q in base.items() if q}


def allocate(fills, positions):
    """fills: IBKR's liquidation rows; positions: {sleeve: {symbol: qty}}. -> (rows to book, unowned {sym: qty}, per-name
    summary). A row: {sleeve, symbol, side, shares, price, commission, time}."""
    by = collections.defaultdict(list)
    for f in fills:
        if f.get("liquidation") != 1 or f.get("client_id") != 0:
            raise ValueError(f"not an IBKR liquidation fill (liquidation 1, client 0): {f.get('exec_id')}")
        by[f["symbol"]].append(f)
    both = sorted(s for s, fs in by.items() if len({f["side"] for f in fs}) > 1)
    if both:                                # one price for the net would not be the net trade's cash: not handled
        raise ValueError(f"IBKR traded both sides of {both}; this booking handles one side per name")
    rows, unowned, summary = [], {}, {}
    for sym in sorted(by):
        fs = by[sym]
        net = sum(signed(f) for f in fs)
        if net == 0:
            continue
        shares = sum(f["shares"] for f in fs)
        vwap = sum(f["shares"] * f["price"] for f in fs) / shares
        com = sum(f.get("commission") or 0.0 for f in fs)
        when = max(f["time"] for f in fs)
        closing = {s: abs(p[sym]) for s, p in positions.items() if p.get(sym) and (p[sym] < 0 if net > 0 else p[sym] > 0)}
        got = share_out(abs(net), closing)
        booked = sum(got.values())
        for s, q in sorted(got.items()):
            rows.append({"sleeve": s, "symbol": sym, "side": "BOT" if net > 0 else "SLD", "shares": q, "price": vwap,
                         "commission": com * q / abs(net), "time": when})
        if abs(net) > booked:
            unowned[sym] = (abs(net) - booked) * (1 if net > 0 else -1)
        summary[sym] = {"ibkr_net": net, "vwap": round(vwap, 4), "commission": round(com, 4), "booked": booked,
                        "sleeves": got, "unowned": unowned.get(sym, 0)}
    return rows, unowned, summary


def fill_record(r):
    return {"exec_id": f"liq-{DAY}-{r['symbol'].replace(' ', '_')}-{r['sleeve']}", "time": r["time"], "symbol": r["symbol"],
            "side": r["side"], "shares": r["shares"], "price": r["price"], "order_id": 0, "perm_id": None,
            "ref": REF.format(sym=r["symbol"]), "exchange": "IBKR liquidation", "client_id": 0, "liquidation": 1,
            "commission": r["commission"], "commission_currency": "USD", "sleeve": r["sleeve"], "synced": paper.now_iso(),
            "booked_by": f"book_liquidation.py (IBKR's margin liquidation of {WHEN[0]} {DAY[:4]}, {WHEN[1]} ET)"}


def note_for(sleeve, rows, positions, resting):
    """One framework note for a sleeve IBKR's liquidation touched (facts only)."""
    parts = []
    for r in sorted(rows, key=lambda r: r["symbol"]):
        had = positions[sleeve][r["symbol"]]
        what = f"{'bought back' if r['side'] == 'BOT' else 'sold'} {r['shares']:g} {r['symbol']} at ${r['price']:,.2f}"
        rest = abs(had) - r["shares"]
        what += (f" (your {'short' if had < 0 else 'long'} {abs(had):g}; you remain {'short' if had < 0 else 'long'} {rest:g})"
                 if rest else f" (your whole {'short' if had < 0 else 'long'} {abs(had):g})")
        parts.append(what)
    from datetime import datetime as _dt
    import reopen_week6 as _r
    text = (f"framework note ({_dt.now(_r.ET):%a %d %b}): on {WHEN[0]} the whole paper account, which all sleeves share, fell "
            f"below IBKR's margin requirement, and at {WHEN[1]} ET IBKR closed positions on its own (a margin liquidation; no model or "
            "software chose them). In your sleeve IBKR " + "; ".join(parts) + ". These are booked to your sleeve at IBKR's "
            f"prices and appear above as fills under refs ibkr-liquidation-{DAY}-<symbol>.")
    names = {r["symbol"] for r in rows}
    mine = [o for o in resting if o["symbol"] in names]
    if mine:
        text += (" Your orders still resting in those names were placed for the positions IBKR closed: "
                 + "; ".join(f"{o['action']} {o['quantity']:g} {o['symbol']} {o['order_type']}"
                             + (f" {o['price_text']}" if o.get("price_text") else "") + f" {o.get('tif') or ''} [{o['ref']}]".replace("  ", " ")
                             for o in mine) + ".")
    return text


def positions_before(fills, sleeves):
    """Each sleeve's positions as its ledger stood one second before IBKR's first liquidation fill (splits applied), not as
    it stands when this runs: a sleeve that trades these names after 15:51 must not change what IBKR closed (the review of
    16 Nov evening). reopen_week6.book_at reads fills up to a time."""
    from datetime import timedelta
    import reopen_week6
    before = reopen_week6.fill_time(min(f["time"] for f in fills)) - timedelta(seconds=1)
    return {s: reopen_week6.book_at(s, before) for s in sleeves}


def resting_orders():
    """Client 10's working orders by sleeve, read-only on client 11 (all_open_orders)."""
    import live  # noqa: F401 - the TWS framework's path
    from tws import Gateway
    with Gateway("trader-research", timeout=30) as gw:
        rows = gw.all_open_orders()
    out = collections.defaultdict(list)
    for o in rows:
        s = paper.sleeve_of(o.get("ref"))
        if not s or o.get("status") in ("Cancelled", "ApiCancelled", "Inactive", "Filled"):
            continue
        px = []
        for k, name in (("limit", "limit"), ("aux", "stop"), ("lmt_price", "limit"), ("aux_price", "stop")):
            v = o.get(k)
            if v not in (None, 0, 0.0) and name not in " ".join(px):
                px.append(f"{name} {v:g}")
        if o.get("trail_stop") not in (None, 0, 0.0):   # the framework gives it on TRAIL rows only (TWS-Gateway 165649d)
            px.append(f"stop now {o['trail_stop']:g}")
        out[s].append({"symbol": o.get("symbol"), "action": o.get("action"), "quantity": o.get("quantity") or 0,
                       "order_type": o.get("order_type"), "tif": o.get("tif"), "ref": o.get("ref"), "price_text": " ".join(px)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--day", help="YYYYMMDD of another day's liquidation (default: Mon 16 Nov's)")
    ap.add_argument("--source", help="that day's fills file (a JSON object with \"fills\")")
    a = ap.parse_args()
    if a.day or a.source:
        if not (a.day and a.source):
            sys.exit("give --day and --source together")
        configure(a.day, a.source)
    src = json.loads(SRC.read_text(encoding="utf-8"))
    fills = src["fills"]
    have = {f["exec_id"] for f in paper.fills()}
    if any(e.startswith(f"liq-{DAY}-") for e in have):
        sys.exit(f"already booked: runs/fills.jsonl holds liq-{DAY}-* rows; nothing done")
    if a.go and list((ROOT / "runs" / "live").glob("LOOKING*")):
        sys.exit("a look is running (runs/live/LOOKING*): the loop writes the ledger and notes.json then; rerun between looks")
    sleeves = paper.sleeves()
    positions = positions_before(fills, sleeves)
    rows, unowned, summary = allocate(fills, positions)
    resting = resting_orders()
    print(f"IBKR liquidation fills: {len(fills)} in {len({f['symbol'] for f in fills})} names")
    for sym, v in summary.items():
        print(f"  {sym:5s} IBKR {v['ibkr_net']:+6d} @ {v['vwap']:.4f}: booked {v['booked']} to {v['sleeves']}; unowned {v['unowned']:+d}")
    by_sleeve = collections.defaultdict(list)
    for r in rows:
        by_sleeve[r["sleeve"]].append(r)
    notes = {s: note_for(s, rs, positions, resting.get(s, [])) for s, rs in sorted(by_sleeve.items())}
    for s, t in notes.items():
        print(f"\nNOTE {s}: {t}")
    print(f"\nunowned (IBKR's trades beyond the sleeves' positions, owned by no sleeve): {unowned}")
    if not a.go:
        print("\nRESULT: dry run; rerun with --go to book")
        return
    for r in rows:
        paper.append(paper.FILLS, fill_record(r))
    for s, t in notes.items():
        paper.add_notes(s, [t])
    UNOWNED.write_text(json.dumps({"source": str(SRC), "unowned": unowned, "summary": summary,
                                   "note": "IBKR's liquidation beyond the sleeves' own positions (into the reset's stale "
                                           "listing); owned by no sleeve. Whether to close it is Clayton's call."},
                                  indent=1), encoding="utf-8")
    print(f"\nRESULT: booked {len(rows)} fills to {len(by_sleeve)} sleeves; {len(notes)} notes; unowned written to {UNOWNED}")


if __name__ == "__main__":
    main()
