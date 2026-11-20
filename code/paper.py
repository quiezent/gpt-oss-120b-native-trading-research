"""The paper account, split into sleeves: one per model under test.

Each order a model places carries the orderRef "tt-<sleeve>-<hex>", so every fill belongs to
exactly one sleeve. A sleeve's cash, positions and realized, costed P&L are computed from its own
fills (runs/fills.jsonl), never from the account-wide positions, which mix all sleeves.

The trading loop calls sync_fills() after every look and after the close. It asks IB Gateway for the last
7 calendar days of executions (the Gateway's day runs 00:00-24:00 ET since 23 Oct), so a missed sync is
caught up later; exec_id dedupes the overlap.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RUNS = Path(__file__).resolve().parent / "runs"
FILLS = RUNS / "fills.jsonl"
ORDERS = RUNS / "orders.jsonl"
SLEEVES = RUNS / "sleeves.json"   # {name: {"start_cash": float, "start": iso, "model": str}}
DIVIDENDS = RUNS / "dividends.jsonl"   # dividends.py: one row per (sleeve, symbol, ex_date) on a position held at the close before
# One row per split: {"symbol", "ex_date": YYYYMMDD (first split-adjusted session), "ratio": new shares per old (3 = 3-for-1)}.
# A position held into ex_date is multiplied by ratio at an average cost divided by it; fills from ex_date on are already
# in new shares (SOXX 3-for-1 on 5 Nov 2026: the books kept pre-split shares all day, against the account's).
SPLITS = RUNS / "splits.jsonl"
# dividends.py books a row itself when the account's quantity and the prices both show the ratio (a reverse split is 1/n, e.g.
# 0.1 for 1-for-10). When a ratio leaves a fraction of a share (1-for-n, 3-for-2), the broker pays it in cash: a book keeps the
# whole shares, toward zero, and is paid the fraction at the row's "cash_price" (the post-split price when it was booked).


def split_qty(q, ratio):
    """(whole shares, fraction of a share left over) of a q-share position after a split of `ratio` new shares per old."""
    from fractions import Fraction
    r = Fraction(ratio).limit_denominator(1000)
    if r.denominator == 1:
        return q * r.numerator, 0.0
    x = Fraction(q) * r
    return int(x), float(x - int(x))


def splits():
    """The rows of SPLITS in ex_date order, each (symbol, ex_date) once: the first row stands, a repeat (a row added by hand
    on top of one dividends.py booked) is not applied again."""
    seen, out = set(), []
    for r in read(SPLITS):
        k = (r["symbol"], str(r["ex_date"]))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return sorted(out, key=lambda r: r["ex_date"])


def et_today():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def et_date(ib_time):
    """IBKR execution time -> US Eastern trading date (YYYYMMDD). The Gateway reports its local time (UTC+8,
    e.g. "20261014  22:01:05"); a time with an explicit zone (e.g. "... US/Eastern") is used as is."""
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    s = " ".join(str(ib_time).split())
    parts = s.split(" ")
    try:
        if len(parts) >= 3:
            dt = datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H:%M:%S").replace(tzinfo=ZoneInfo(parts[2]))
        else:
            dt = datetime.strptime(s, "%Y%m%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8)))
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    except (ValueError, KeyError):
        return s[:8]


def sleeves():
    return json.loads(SLEEVES.read_text(encoding="utf-8")) if SLEEVES.exists() else {}


def add_sleeve(name, start_cash, model, **meta):
    """Add a sleeve mid-week (e.g. ml=True, effort="high"); live.py re-reads sleeves.json at each look."""
    s = sleeves()
    if name in s:
        raise ValueError(f"sleeve {name!r} exists")
    s[name] = {"start_cash": float(start_cash), "start": now_iso(), "model": model} | meta
    RUNS.mkdir(exist_ok=True)
    SLEEVES.write_text(json.dumps(s, indent=1), encoding="utf-8")


def new_ref(sleeve):
    return f"tt-{sleeve}-{uuid.uuid4().hex[:10]}"


def sleeve_of(ref):
    """'tt-<sleeve>-<hex>' -> sleeve; sleeve names may contain '-' (e.g. base-ml)."""
    parts = (ref or "").split("-")
    return "-".join(parts[1:-1]) if len(parts) >= 3 and parts[0] == "tt" else None


ARCHIVE = RUNS / "archive"   # past weeks' ledgers (runs/archive/<week>/fills.jsonl ...); their executions are never re-booked


def sync_fills(gw):
    """Append new executions from the last 7 days (and late commission reports) to runs/fills.jsonl."""
    known = {}
    for p in ARCHIVE.glob("*/fills.jsonl"):
        for f in read(p):
            known[f["exec_id"]] = {**f, "commission": f.get("commission", 0.0)}   # archived: never re-append
    for f in read(FILLS):
        known[f["exec_id"]] = f
    recon = {}   # ref -> latest time booked from IBKR's confirmation after the gateway stopped serving it
    for f in known.values():
        if str(f["exec_id"]).startswith("recon-"):
            recon[f.get("ref")] = max(recon.get(f.get("ref"), ""), f["time"][:17])
    new = 0
    for e in gw.executions(days=7):
        if e.get("ref") in recon and str(e.get("time", ""))[:17] <= recon[e["ref"]]:
            continue   # served again (e.g. after the gateway's time-zone switch) but already booked as recon-<ref>
        old = known.get(e["exec_id"])
        if old is None or (old.get("commission") is None and e.get("commission") is not None):
            append(FILLS, {**e, "sleeve": sleeve_of(e.get("ref")), "synced": now_iso()})
            new += 1
    return new


def fills():
    """Fills by exec_id, the latest record winning (a later one may carry the commission)."""
    out = {}
    for f in read(FILLS):
        out[f["exec_id"]] = f
    return sorted(out.values(), key=lambda f: (f["time"], f["exec_id"]))


def book(sleeve, marks=None):
    """Cash, positions, realized and unrealized P&L for one sleeve, from its own fills.

    Average-cost accounting; shorts allowed. Commission reduces cash and realized P&L. A fill whose
    commission report never arrived is counted at 0 and flagged in missing_commissions.
    """
    meta = sleeves()[sleeve]
    cash, realized, commissions, missing = meta["start_cash"], 0.0, 0.0, 0
    pos = {}   # symbol -> [qty, avg_cost]
    opened = {}  # symbol -> date (YYYYMMDD) the current position was opened
    trades = 0
    pending = splits()

    def split_through(day):                  # every split up to `day` (YYYYMMDD), on the position held into it
        nonlocal cash, realized
        while pending and pending[0]["ex_date"] <= day:
            r = pending.pop(0)
            if r["symbol"] in pos:
                q, avg = pos[r["symbol"]]
                whole, frac = split_qty(q, r["ratio"])
                avg = avg / r["ratio"]
                if frac:                     # paid in cash, as the broker pays it
                    px = r.get("cash_price") or avg
                    cash += frac * px
                    realized += frac * (px - avg)
                pos[r["symbol"]] = [whole, avg if whole else 0.0]

    for f in fills():
        if f.get("sleeve") != sleeve:
            continue
        split_through(et_date(f["time"]))
        trades += 1
        qty = f["shares"] if f["side"] == "BOT" else -f["shares"]
        px, com = f["price"], f.get("commission")
        if com is None:
            missing += 1
            com = 0.0
        cash -= qty * px + com
        commissions += com
        realized -= com
        q, avg = pos.get(f["symbol"], [0, 0.0])
        if q == 0 or (q > 0) == (qty > 0):          # opening or adding
            avg = (q * avg + qty * px) / (q + qty)
            q += qty
        else:                                        # reducing, closing or flipping
            closed = min(abs(qty), abs(q)) * (1 if q > 0 else -1)
            realized += closed * (px - avg)
            q += qty
            if q != 0 and (q > 0) != (closed > 0):   # flipped through zero
                avg = px
        pos[f["symbol"]] = [q, avg if q else 0.0]
        q_before = q - qty
        if q and (q_before == 0 or (q > 0) != (q_before > 0)):   # opened, or flipped through zero
            opened[f["symbol"]] = et_date(f["time"])
        if not q:
            opened.pop(f["symbol"], None)
    split_through(et_today())
    positions ={s: {"quantity": q, "avg_cost": avg, "opened": opened.get(s)} for s, (q, avg) in pos.items() if q}
    dividends = sum(r["amount"] for r in read(DIVIDENDS) if r["sleeve"] == sleeve)   # credited on longs, charged on shorts
    cash += dividends
    realized += dividends
    unrealized, value = 0.0, 0.0
    for s, p in positions.items():
        m = (marks or {}).get(s)
        p["mark"] = m
        if m is not None:
            p["unrealized"] = p["quantity"] * (m - p["avg_cost"])
            unrealized += p["unrealized"]
            value += p["quantity"] * m
    unmarked = [s for s, p in positions.items() if p["mark"] is None]
    return {"sleeve": sleeve, "model": meta["model"], "start": meta["start"], "start_cash": meta["start_cash"],
            "cash": cash, "positions": positions, "realized": realized, "unrealized": unrealized,
            "commissions": commissions, "dividends": dividends, "fills": trades, "missing_commissions": missing,
            "nav": None if unmarked else cash + value, "unmarked": unmarked}


def positions_at(sleeve, day):
    """{symbol: quantity} the sleeve held at the close before `day` (YYYYMMDD): its fills on earlier ET dates."""
    pos = {}
    pending = splits()

    def split_before(d):                         # a split between the fills so far and `d`, on the position held into it
        while pending and pending[0]["ex_date"] < d:
            r = pending.pop(0)
            if r["symbol"] in pos:
                pos[r["symbol"]] = split_qty(pos[r["symbol"]], r["ratio"])[0]

    for fd, f in sorted(((et_date(f["time"]), f) for f in fills() if f.get("sleeve") == sleeve), key=lambda x: x[0]):
        if fd >= day:
            break
        split_before(fd + "~")                   # ex_date <= fd: this fill is in new shares
        pos[f["symbol"]] = pos.get(f["symbol"], 0) + (f["shares"] if f["side"] == "BOT" else -f["shares"])
    split_before(day)
    return {s: q for s, q in pos.items() if q}


def mark(quote):
    """A mark price from a framework quote row: last, else mid, else previous close."""
    if quote.get("last"):
        return quote["last"]
    if quote.get("bid") and quote.get("ask"):
        return (quote["bid"] + quote["ask"]) / 2
    return quote.get("close")


def place(gw, sleeve, symbol, action, quantity, order_type="MKT", limit=None, aux=None, tif="DAY", arrival=None, probs=None,
          close_only=False, **fields):
    """Place one order for a sleeve exactly as given and record IBKR's answer (and the arrival quote and the
    model's p_fill/p_win, if given: logged for calib_report.py, never sent to the broker). close_only (17 Nov): recorded in
    the row (close_only_refs), never sent to the broker (not an ibapi Order field)."""
    ref = new_ref(sleeve)
    row = {"time": now_iso(), "sleeve": sleeve, "ref": ref, "symbol": symbol, "action": action,
           "quantity": quantity, "order_type": order_type, "limit": limit, "aux": aux, "tif": tif, "fields": fields}
    if close_only:
        row["close_only"] = True
    if arrival:
        row["arrival"] = arrival
    if probs is not None:
        row["probs"] = probs
    try:
        out = gw.place(symbol, action, quantity, order_type, limit=limit, aux=aux, tif=tif, ref=ref, **fields)
        row["result"] = out
    except Exception as e:   # IBKR's or the framework's refusal goes back to the model verbatim
        row["error"] = f"{type(e).__name__}: {e}"
    append(ORDERS, row)
    return row


def cancel(gw, sleeve, ref):
    row = {"time": now_iso(), "sleeve": sleeve, "cancel": ref}
    if sleeve_of(ref) != sleeve:
        row["error"] = f"{ref} is not an order of this sleeve"
    else:
        try:
            row["result"] = gw.cancel(ref=ref)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
    append(ORDERS, row)
    return row


def open_orders(gw, sleeve):
    done = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
    return [o for o in gw.orders() if sleeve_of(o.get("ref")) == sleeve and o.get("status") not in done]


def close_only_refs(sleeve):
    """Refs this sleeve placed as close_only (runs/orders.jsonl; Clayton, 17 Nov)."""
    return {r["ref"] for r in read(ORDERS) if r.get("close_only") and r.get("ref") and r.get("sleeve") == sleeve}


def _left(o):
    """An open order's quantity still to fill: IBKR's remaining when it reports one, else the order's quantity."""
    r = o.get("remaining")
    return float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) and r >= 0 else float(o.get("quantity") or 0)


def close_only_working(gw, sleeve, open_rows=None):
    """This sleeve's close_only orders still open at IBKR (orders() on client 10), CANCEL-PENDING ones included: a cancel IBKR
    has not confirmed can still fill (the review of 17 Nov: on 22 Oct some took hours), so it still counts as cover."""
    refs = close_only_refs(sleeve)
    if not refs:                                               # no close_only order ever: no orders() round trip
        return []
    rows = open_orders(gw, sleeve) if open_rows is None else open_rows
    return [o for o in rows if o.get("ref") in refs]


def close_only_recent(days=10):
    """Sleeves that placed a close_only order in the last `days` days (GTC orders may still be working): the loop's
    between-looks guard connects only when there are some."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return sorted({r["sleeve"] for r in read(ORDERS) if r.get("close_only") and r.get("sleeve") and str(r.get("time", "")) >= since})


def close_only_excess(gw, sleeve, log=None, fail_log=None):
    """At each look (live.look, after its fill syncs) and about once a minute between looks (live.close_only_guard): the
    sleeve's close_only orders that its position no longer covers are cancelled. In each symbol, from the oldest order to the
    newest, an order on the closing side is kept while what is left of the position still covers it; every other order
    (the same side as the position, no position, or more than is left) is cancelled. One whose cancel is already pending is
    not cancelled again, and one this check already tried to cancel is not tried again (a failed cancel is not retried
    every minute); each order is named in a note once (the recheck: the guard runs about once a minute). fail_log: where a
    failure of the check itself is said (the guard throttles it). Returns the cancel rows. Never raises."""
    try:
        work = close_only_working(gw, sleeve)
        if not work:
            return []
        pos = {s: p["quantity"] for s, p in book(sleeve)["positions"].items()}
        order_of = {r["ref"]: (r.get("time", ""), i) for i, r in enumerate(read(ORDERS))      # later rows win a tie
                    if r.get("ref") and r.get("sleeve") == sleeve}
        asked = cancel_requested(sleeve)
        tried = {str(r["cancel"]) for r in read(ORDERS) if r.get("cancel") and r.get("sleeve") == sleeve}
        out, said = [], []
        for sym in sorted({o["symbol"] for o in work}):
            q = pos.get(sym, 0)
            closing = "BUY" if q < 0 else "SELL" if q > 0 else None
            room = abs(q) if closing else 0
            for o in sorted((o for o in work if o["symbol"] == sym), key=lambda o: order_of.get(o["ref"], ("", -1))):
                left = _left(o)
                if o["action"] == closing and left <= room + 1e-9:
                    room -= left                                   # the oldest that still fit are kept
                    continue
                if o["ref"] in asked or o["ref"] in tried or o.get("status") == "PendingCancel":
                    continue                                   # already asked for (by the model or this check): named once
                r = cancel(gw, sleeve, o["ref"])
                out.append(r)
                said.append(f"{o['ref']} {o['action']} {left:g} {sym}" + (f" (cancel failed: {r['error']})" if r.get("error") else ""))
        if said:
            add_notes(sleeve, ["framework note (close_only): your position no longer covered these close_only orders, so a "
                               "cancel was sent for each (the oldest that still fit were kept; an order is working until "
                               "the broker confirms it): " + "; ".join(said) + "."])
            if log:
                log(f"{sleeve}: close_only orders cancelled: {said}")
        return out
    except Exception as e:                                     # noqa: BLE001 - a failed check costs nothing but itself
        say = fail_log or log
        if say:
            say(f"{sleeve}: close_only check failed: {type(e).__name__}: {e}")
        return []


def placed_fields(sleeve):
    """{ref: the extra order fields this sleeve sent (e.g. trailingPercent)}, from runs/orders.jsonl: IBKR's open-order
    rows carry no trailing percent, so the state shows it from here."""
    return {r["ref"]: r.get("fields") or {} for r in read(ORDERS) if r.get("ref") and r.get("sleeve") == sleeve}


def cancel_requested(sleeve):
    """Refs of this sleeve's orders whose cancel IBKR accepted (runs/orders.jsonl). An order in this set that is still open
    is a cancel IBKR has not confirmed (22 Oct: some took hours). A cancel that came back with another error (e.g. 504 not
    connected, 161) doesn't count: that order is still working."""
    def accepted(res):
        if res.get("status") in ("PendingCancel", "Cancelled", "ApiCancelled"):
            return True
        msgs = [str(m) for m in res.get("messages") or []]
        return bool(msgs) and all("not confirmed within" in m or m.startswith(("10148", "202")) for m in msgs)
    return {str(r["cancel"]) for r in read(ORDERS)
            if r.get("cancel") and r.get("sleeve") == sleeve and isinstance(r.get("result"), dict) and accepted(r["result"])}


# ---------------------------------------------------------------- what the model hears about at its next look
LOOKS = RUNS / "looks.json"   # {sleeve: {"reported_exec": [...], "reported_orders": [...]}}


def events_since_last_look(gw, sleeve, commit=True):
    """Fills, and orders IBKR cancelled, rejected or expired, not yet reported to this sleeve's model."""
    state = json.loads(LOOKS.read_text()) if LOOKS.exists() else {}
    st = state.setdefault(sleeve, {"reported_exec": [], "reported_orders": []})
    seen_e, seen_o = set(st["reported_exec"]), set(map(str, st["reported_orders"]))
    out = []
    for f in fills():
        if f.get("sleeve") == sleeve and f["exec_id"] not in seen_e:
            com = f.get("commission")
            out.append(f"filled {'BUY' if f['side'] == 'BOT' else 'SELL'} {f['shares']:g} {f['symbol']} @ {f['price']:.2f}"
                       + (f" (commission ${com:.2f})" if com is not None else "") + f" [{f['ref']}]")
            st["reported_exec"].append(f["exec_id"])
    for o in gw.orders():
        if sleeve_of(o.get("ref")) != sleeve or o.get("status") not in ("Cancelled", "ApiCancelled", "Inactive"):
            continue
        key = str(o.get("perm_id"))
        if key in seen_o:
            continue
        why = o.get("completed_status") or "; ".join(o.get("messages", []) or []) or o.get("status")
        out.append(f"{o['ref']} {o['action']} {o['quantity']:g} {o['symbol']} {o['order_type']} ended {o.get('status')}: {why}")
        st["reported_orders"].append(key)
    if commit:
        LOOKS.write_text(json.dumps(state), encoding="utf-8")
    return out


NOTES = RUNS / "notes.json"   # {sleeve: [text, ...]}: placement refusals etc., shown at the next look


def add_notes(sleeve, notes):
    if not notes:
        return
    d = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    d.setdefault(sleeve, []).extend(notes)
    NOTES.write_text(json.dumps(d), encoding="utf-8")


def take_notes(sleeve, commit=True):
    d = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    out = d.get(sleeve, [])
    if commit and out:
        d[sleeve] = []
        NOTES.write_text(json.dumps(d), encoding="utf-8")
    return out
