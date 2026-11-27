"""The paper account, split into sleeves: one per model under test.

Each order a model places carries the orderRef "tt-<sleeve>-<hex>", so every fill belongs to
exactly one sleeve. A sleeve's cash, positions and realized, costed P&L are computed from its own
fills (runs/fills.jsonl), never from the account-wide positions, which mix all sleeves.

The trading loop calls sync_fills() after every look and after the close. It asks IB Gateway for the last
7 calendar days of executions (the Gateway's day runs 00:00-24:00 ET since 23 Oct), so a missed sync is
caught up later; exec_id dedupes the overlap.
"""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import close_only as co

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


IO_TRIES, IO_WAIT_S = 12, 0.25  # a whole-file ledger's read or write is tried this often, this far apart (about 3 s)


def _read_json(path, default):
    """A whole-file JSON ledger (sleeves.json, looks.json, notes.json, live's memo.json and done.json), or `default` if
    it is absent. Windows refuses an in-place write while another process memory-maps the file, which git does when it
    hashes a tracked file that changed (Mon 23 Nov 2026, 09:32 ET: OSError [Errno 22] on runs/looks.json cost ink2-30k-ml
    its 09:30 look; reproduced in a test). Reads and appends are not refused. So an OSError is tried again, IO_TRIES
    times IO_WAIT_S apart; the last one raises as before. A file that reads but does not parse raises at once."""
    if not path.exists():
        return default
    for i in range(IO_TRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:           # it existed and went away (a checkout, a stash): raise as before, never
            raise                           # the default, which would rewrite every sleeve's entry from nothing
        except OSError:
            if i == IO_TRIES - 1:
                raise
            time.sleep(IO_WAIT_S)


def _write_json(path, obj, **kw):
    """Write a whole-file JSON ledger in place, as before, with _read_json's retry on a transient OSError. It is not a
    temp file swapped in: on Windows that swap is refused while any reader has the file open. In the live loop the lanes
    read and write these files only while holding its `ledger` lock, so no lane sees another lane's half-written file."""
    text = json.dumps(obj, **kw)
    for i in range(IO_TRIES):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError:
            if i == IO_TRIES - 1:
                raise
            time.sleep(IO_WAIT_S)


def sleeves():
    return _read_json(SLEEVES, {})


def add_sleeve(name, start_cash, model, **meta):
    """Add a sleeve mid-week (e.g. ml=True, effort="high"); live.py re-reads sleeves.json at each look."""
    s = sleeves()
    if name in s:
        raise ValueError(f"sleeve {name!r} exists")
    s[name] = {"start_cash": float(start_cash), "start": now_iso(), "model": model} | meta
    RUNS.mkdir(exist_ok=True)
    _write_json(SLEEVES, s, indent=1)


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
          close_only=False, held=None, **fields):
    """Place one order for a sleeve exactly as given and record IBKR's answer (and the arrival quote and the
    model's p_fill/p_win, if given: logged for calib_report.py, never sent to the broker). close_only (17 Nov): recorded in
    the row (close_only_refs), never sent to the broker (not an ibapi Order field). held (close_only v2.1, DESIGN B):
    {"after": {entry ref: its fills at the hold}, "left": what those entries had left to fill then (E), "had": the shares
    held then on the exit's closing side, less those entries' fills since their snapshots (H); E and H from 25 Nov, for
    the check's P}: the row is recorded with no broker result and nothing is sent; the
    check (close_only_check) releases it under this ref when its entries fill."""
    ref = new_ref(sleeve)
    row = {"time": now_iso(), "sleeve": sleeve, "ref": ref, "symbol": symbol, "action": action,
           "quantity": quantity, "order_type": order_type, "limit": limit, "aux": aux, "tif": tif, "fields": fields}
    if close_only:
        row["close_only"] = True
    if arrival:
        row["arrival"] = arrival
    if probs is not None:
        row["probs"] = probs
    if held is not None:
        row["held"] = held
        append(ORDERS, row)
        return row
    try:
        out = gw.place(symbol, action, quantity, order_type, limit=limit, aux=aux, tif=tif, ref=ref, **fields)
        row["result"] = out
    except Exception as e:   # IBKR's or the framework's refusal goes back to the model verbatim
        row["error"] = f"{type(e).__name__}: {e}"
    append(ORDERS, row)
    return row


def cancel(gw, sleeve, ref, by=None):
    """Cancel one of the sleeve's orders at IBKR. A held close_only exit (v2.1) that never went to the broker is cancelled
    here, in the ledger: its row has held true and IBKR's form of a cancelled order, so cancel_requested counts it.
    by: who asked, when not the sleeve ("close_only": the check's own cancel, 25 Nov), recorded in the row."""
    row = {"time": now_iso(), "sleeve": sleeve, "cancel": ref, **({"by": by} if by else {})}
    if sleeve_of(ref) != sleeve:
        row["error"] = f"{ref} is not an order of this sleeve"
    elif ref in held_waiting(sleeve):
        row["held"] = True
        row["result"] = {"status": "Cancelled", "messages": ["held: it never went to the broker"]}
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


ET = ZoneInfo("America/New_York")
DONE = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def co_state(sleeve, rows=None):
    """This sleeve's close_only ledger in one pass over runs/orders.jsonl (close_only v2.1): closers (refs placed or held
    close_only), held ({ref: held row}), released ({ref: [release rows]}), unlinked, cancelled (held refs cancelled before
    any release), tried (refs a cancel was sent for), modify ({ref: [modify rows]}), key ({ref: (time, row index)})."""
    rows = read(ORDERS) if rows is None else rows
    st = {"closers": set(), "held": {}, "released": {}, "unlinked": set(), "cancelled": set(), "tried": set(),
          "modify": {}, "key": {}}
    for i, r in enumerate(rows):
        if r.get("sleeve") != sleeve:
            continue
        if r.get("ref"):
            st["key"][r["ref"]] = (r.get("time", ""), i)
            if r.get("close_only"):
                st["closers"].add(r["ref"])
            if r.get("held") is not None:
                st["held"][r["ref"]] = r
        elif r.get("release"):
            st["released"].setdefault(str(r["release"]), []).append(r)
        elif r.get("unlink"):
            st["unlinked"].add(str(r["unlink"]))
        elif r.get("modify"):
            st["modify"].setdefault(str(r["modify"]), []).append(r)
        if r.get("cancel"):
            st["tried"].add(str(r["cancel"]))
            if r.get("held") is True:
                st["cancelled"].add(str(r["cancel"]))
    return st


def held_waiting(sleeve, st=None):
    """{ref: held row} of this sleeve's held close_only exits not yet at the broker (not released, not cancelled)."""
    st = co_state(sleeve) if st is None else st
    return {r: row for r, row in st["held"].items() if r not in st["released"] and r not in st["cancelled"]}


def ref_fills():
    """{ref: shares filled} over runs/fills.jsonl (unsigned)."""
    out = {}
    for f in fills():
        out[f.get("ref")] = out.get(f.get("ref"), 0.0) + abs(float(f.get("shares") or 0))
    return out


def _now():
    """The time in ET (tests set it)."""
    return datetime.now(ET)


# NYSE's 13:00 closes, as live.HALF_DAYS (close_only_v2_test checks they agree; paper cannot import live)
HALF_DAYS = {"2026-11-27", "2026-12-24", "2027-11-26"}


def _close(d):
    return (13, 0) if d.isoformat() in HALF_DAYS else (16, 0)


def _session(now):
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < _close(now.date())


def _expired(row, now):
    """A held DAY exit placed on an earlier ET day, or held past its own day's close."""
    if str(row.get("tif") or "DAY").upper() != "DAY":
        return False
    try:
        placed = datetime.fromisoformat(row["time"]).astimezone(ET)
    except (KeyError, ValueError, TypeError):
        return False
    return placed.date() < now.date() or (now.hour, now.minute) >= _close(now.date())


def _desc(row_or_order):
    """'BUY 100 SMH STP', from a ledger row (order_type) or a broker row."""
    o = row_or_order
    t = o.get("order_type") or o.get("type")
    if t == "MKT" and str(o.get("tif") or "").upper() == "OPG":
        t = "MOO"
    return f"{o['action']} {co.fmt(float(o['quantity']))} {o['symbol']} {t}"


def release(gw, sleeve, row, quantity):
    """Send a held exit to IBKR under its own ref at `quantity` shares (DESIGN B). A release row is written before the
    call, so a ref reaches IBKR at most once even if this process stops in between; a second row carries the result.
    Returns (accepted, message)."""
    ref = row["ref"]
    append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "release": ref, "quantity": quantity})
    out = {"time": now_iso(), "sleeve": sleeve, "release": ref, "quantity": quantity}
    try:
        res = gw.place(row["symbol"], row["action"], quantity, row["order_type"], limit=row.get("limit"), aux=row.get("aux"),
                       tif=row.get("tif") or "DAY", ref=ref, **(row.get("fields") or {}))
        out["result"] = res
    except Exception as e:                                     # noqa: BLE001 - the refusal goes to the model verbatim
        out["error"] = f"{type(e).__name__}: {e}"
    append(ORDERS, out)
    if out.get("error"):
        return False, out["error"]
    status = (out["result"] or {}).get("status")
    msg = "; ".join((out["result"] or {}).get("messages") or [])
    if status is None or status in ("Cancelled", "Inactive", "ApiCancelled"):
        return False, f"{status or 'refused'} {msg}".strip()
    return True, ("" if status != "Unknown" else "the broker has not confirmed it yet")


GONE = ("ended", "not_open")            # GatewayError.reason (tws 4b4269d): the order is no longer one to change
ENDED = ("Filled", "Cancelled", "ApiCancelled", "Inactive", "PendingCancel", "PreCancelled")   # a modify's row status


def resize(gw, sleeve, ref, total, order_id=None):
    """Set a working close_only order's total (filled shares included) by Gateway.modify (tws 29c5f49; its contract, the
    TWS Gateway Owner, 20 Nov). Returns (outcome, message):
    - True: IBKR reported the new size (modified True);
    - False: IBKR refused (modified False; its error_code and error); the order stands as it was;
    - None: sent and not confirmed (modified None: 2102, the order ending while waiting, a timeout); never re-sent at
      this total (_tried_modify);
    - "cancel": nothing was sent because the total is not above the filled shares (GatewayError reason "filled"): the
      caller cancels the remainder;
    - "gone": nothing was sent because the order ended or is no longer open (reasons "ended", "not_open"): nothing to do;
    - "unsent": nothing was sent for any other reason (reasons "size_mismatch", "no_details", a connection loss with
      reason None, a ValueError): the row is marked "sent": False, so the next check may try again (_tried_modify).
    A total that is not a positive finite number is never sent (the Owner: a bug's NaN or None must not reach modify).
    A ref that matches several open orders ("ambiguous") is tried once more by its order_id when the caller has it."""
    import math
    row = {"time": now_iso(), "sleeve": sleeve, "modify": ref, "quantity": total}
    res, reason = None, None
    try:
        if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
            raise ValueError(f"total {total!r} is not a size (not sent)")
        try:
            res = gw.modify(ref=ref, quantity=total)
        except Exception as e:                                 # noqa: BLE001
            if getattr(e, "reason", None) == "ambiguous" and order_id is not None:
                row["by_order_id"] = order_id
                res = gw.modify(order_id=order_id, quantity=total)
            else:
                raise
        row["result"] = res
    except Exception as e:                                     # noqa: BLE001
        reason = getattr(e, "reason", None)
        row.update(error=f"{type(e).__name__}: {e}", reason=reason, sent=False)
    append(ORDERS, row)
    if row.get("error"):
        if reason == "filled":
            return "cancel", row["error"]
        if reason in GONE:
            return "gone", row["error"]
        return "unsent", row["error"]
    done = (res or {}).get("modified")
    msgs = (res or {}).get("messages") or []
    if done is not True and (res or {}).get("status") in ENDED:   # it filled or ended during the modify (the Gateway's
        return "gone", str(msgs[-1]) if msgs else str(res.get("status"))   # None or False after 103/104/134): no claim
    if done is False:                    # IBKR's reason: its error (with the code), else the order's last message
        code, err = (res or {}).get("error_code"), (res or {}).get("error") or (msgs[-1] if msgs else None)
        return False, (str(err) if err and (code is None or str(code) in str(err)) else
                       " ".join(str(x) for x in (code, err) if x) or "refused")
    return (True, "") if done is True else (None, str(msgs[-1]) if msgs else "not confirmed")


def close_only_check(gw, sleeve, log=None, fail_log=None, rows=None, synced=False, now=None, place_ok=True):
    """The close_only check (DESIGN C). With close_only.V2 off: close_only_excess, the rule of 17-19 Nov (and any held
    exit left from a time it was on is cancelled in the ledger, with a note). With it on, at each look, after the look,
    at act's tail, about once a minute between looks and while a look runs: reads the broker's orders first (rows:
    gw.orders() already read by the caller), then syncs fills (synced: the caller did, after reading rows), then the
    book; per symbol: (1) legs on the wrong side or with no position are cancelled; (2) linked exits at the broker grow
    with their entries' fills and are unlinked when the entries end; (3) held exits are released at min(their size,
    their entries' fills since the hold plus P, the room their groups leave; P, the Researcher's rule of 24-25 Nov 2026,
    counts only once none of their entries still works: the part of their size above what their entries had left to
    fill at the hold, at most the shares held then on their closing side), and cancelled in the ledger when the entries end
    with nothing to close or, for DAY, at the close; (4) each group's legs are kept within the position, the newest
    grouped leg first reduced (Gateway.modify) or cancelled, a MKT leg last. place_ok False (HOLD exists): nothing is
    released or grown, only cancelled or reduced. Every change is named once in a note at the next look (written even if
    the check stops part-way). Returns a list of what was done. Never raises."""
    if not co.V2:
        _drop_held(sleeve, log)
        return close_only_excess(gw, sleeve, log=log, fail_log=fail_log)
    said, done = [], []
    try:
        led = read(ORDERS)
        st = co_state(sleeve, led)
        if not st["closers"]:
            return []
        all_rows = gw.orders() if rows is None else rows
        if not synced:
            sync_fills(gw)
        mine = [o for o in all_rows if sleeve_of(o.get("ref")) == sleeve and o.get("status") not in DONE]
        open_refs = {o["ref"] for o in mine}
        waiting = held_waiting(sleeve, st)
        linked = {r: row for r, row in st["held"].items() if r in st["released"] and r not in st["unlinked"]}
        leg_rows = [o for o in mine if o.get("ref") in st["closers"]]
        if not leg_rows and not waiting and not linked:
            return []
        pos = {s: p["quantity"] for s, p in book(sleeve)["positions"].items()}
        rf = ref_fills()
        asked = cancel_requested(sleeve)
        now = (now or _now()).astimezone(ET)

        def say(line):
            said.append(line)
            if log:
                log(f"{sleeve}: close_only: {line}")

        def fe_of(row):
            return sum(max(0.0, rf.get(e, 0.0) - float(snap or 0)) for e, snap in (row["held"].get("after") or {}).items())

        def working(row):
            return [e for e in (row["held"].get("after") or {}) if e in open_refs]

        size = {o["ref"]: float(o.get("quantity") or 0) for o in mine}

        def still(row):                 # its entries still working: listed, and not filled in full by the fills just booked
            return [e for e in working(row) if rf.get(e, 0.0) < size.get(e, 0.0) - co.EPS]

        def p_of(row):                  # the Researcher's P (24-25 Nov 2026, the late gate): once none of its entries still
            if still(row):              # works, min(max(0, q - E), H); 0 while one does, and for a row held before 25 Nov
                return 0.0
            return co.part_above(row["quantity"], row["held"].get("left"), row["held"].get("had"))

        def room_note(t, q, fe, p, h, legs, grp, ref=None):   # the Researcher's clause (25 Nov), only when other close_only
            others = abs(h) - co.room(h, legs, grp, skip=ref)    # legs there cover part of the position and the room they
            if t < min(q, fe + p) - co.EPS and others > co.EPS:   # leave is what holds the total down
                return "; your other close_only orders there that could fill with it already cover the rest of your position"
            return ""

        def send_cancel(l, why):
            """The check's own cancel of a leg, once (a failed one is not sent again every minute, as on 17-19 Nov)."""
            if l["ref"] in st["tried"]:
                return
            st["tried"].add(l["ref"])
            c = cancel(gw, sleeve, l["ref"], by="close_only")
            done.append(("cancel", l["ref"]))
            say(why + (f" (cancel failed: {c['error']})" if c.get("error") else "; it is working until the broker confirms it"))

        for r in [r for r in linked if r not in open_refs]:     # filled or cancelled at the broker: never sent again
            if _unacked(st, r, now):    # a release IBKR has not acknowledged may not be listed yet (the review of 25 Nov:
                continue                # the catch-up's second pass reads the orders at once): kept linked a while
            append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "unlink": r, "why": "ended at the broker"})
            linked.pop(r)
        syms = sorted({o["symbol"] for o in leg_rows} | {row["symbol"] for row in waiting.values()})
        for sym in syms:
            h = float(pos.get(sym, 0) or 0)
            closing = "BUY" if h < 0 else "SELL" if h > 0 else None
            hs = f"{h:+g}"
            legs = []
            for o in sorted((o for o in leg_rows if o["symbol"] == sym), key=lambda o: st["key"].get(o["ref"], ("", -1))):
                total = float(o.get("quantity") or 0)
                f = rf.get(o["ref"], 0.0)
                up = _unconfirmed_up(st, o["ref"], total)   # a raise sent and not confirmed may still be applied: its
                                                            # room counts (review, 20 Nov)
                if total > co.EPS and f >= max(total, up) - co.EPS:   # filled in full by the fills just booked, though the
                    continue    # orders, read first, still list it (24 Nov: MPWR 09:38, ADI and LRCX 12:40, a cancel sent for
                                # a stop that had filled came back "not an open order"): not a leg, nothing to change
                legs.append({"ref": o["ref"], "action": o["action"], "total": total, "f": f,
                             "rem": max(0.0, max(total, up) - f),
                             "groups": co.groups(o.get("order_type"), o.get("tif")), "key": st["key"].get(o["ref"], ("", -1)),
                             "pending": o.get("status") == "PendingCancel" or o["ref"] in asked, "row": o})
            # (1) the wrong side, or no position: cancelled (as on 17-19 Nov)
            for l in [l for l in legs if l["action"] != closing]:
                legs.remove(l)
                if not l["pending"]:
                    send_cancel(l, f"your position in {sym} was {hs} at the check, so a cancel was sent for {l['ref']} {_desc(l['row'])}")
            # (2) linked exits at the broker (the older ones) grow with their entries' fills; unlinked when they end
            for l in [l for l in legs if l["ref"] in linked]:
                row = linked[l["ref"]]
                fe, p = fe_of(row), p_of(row)
                t = int(co.held_total(float(row["quantity"]), fe, l["f"], h, row["action"], legs, l["groups"],
                                      ref=l["ref"], p=p) + co.EPS)
                if place_ok and t > l["total"] + co.EPS and not l["pending"] and not _tried_modify(st, l["ref"], t):
                    ok, msg = resize(gw, sleeve, l["ref"], t, order_id=l["row"].get("order_id"))
                    done.append(("grow", l["ref"], t, ok))
                    if ok is True:
                        say(f"your position in {sym} was {hs} at the check, so {l['ref']} {_desc(row)} went from "
                            f"{co.fmt(l['total'])} to {t} at the broker"
                            + room_note(t, float(row["quantity"]), fe, p, h, legs, l["groups"], ref=l["ref"]))
                        l["total"], l["rem"] = float(t), max(0.0, t - l["f"])
                    elif ok is False:
                        say(f"{l['ref']} {_desc(row)} could not be raised to {t}: {msg}; it stands at {co.fmt(l['total'])}")
                    elif ok is None:
                        say(f"the broker has not confirmed {l['ref']} {_desc(row)} at {t}; it may still stand at "
                            f"{co.fmt(l['total'])} (that size is not sent again)")
                        l["rem"] = max(l["rem"], t - l["f"])   # IBKR may have applied it: the rest of the check counts it
                    # "gone", "unsent", "cancel": nothing reached the broker; the next check sees the order as it is
                if not still(row):      # its entries have ended (the gate's rule): grown as far as it fits, unlinked
                    reached = t <= l["total"] + co.EPS
                    sent = any(r.get("quantity") == t and r.get("sent") is not False for r in st["modify"].get(l["ref"]) or [])
                    sent = sent or any(d[:3] == ("grow", l["ref"], t) and d[3] in (True, False, None) for d in done)
                    gave_up = t > l["total"] + co.EPS and _tried_modify(st, l["ref"], t) and not sent
                    if reached or sent or gave_up:   # else (nothing sent yet, or HOLD): kept linked, tried again
                        if gave_up:
                            say(f"{l['ref']} {_desc(row)} stays at {co.fmt(l['total'])}: the broker could not be asked to "
                                f"raise it to {t}")
                        append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "unlink": l["ref"], "why": "entries ended"})
                        linked.pop(l["ref"], None)
            # (3) held exits: expiry, release, entries ended
            for ref, row in sorted(((r, w) for r, w in waiting.items() if w["symbol"] == sym), key=lambda x: st["key"].get(x[0])):
                grp = co.groups(row.get("order_type"), row.get("tif"))
                if _expired(row, now):
                    append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "cancel": ref, "held": True,
                                    "result": {"status": "Cancelled", "messages": ["held DAY order at the close"]}})
                    done.append(("expired", ref))
                    say(f"your held {ref} {_desc(row)} DAY was cancelled at the close; it never went to the broker")
                    continue
                fe, p = fe_of(row), p_of(row)
                t = int(co.held_total(float(row["quantity"]), fe, 0.0, h, row["action"], legs, grp, p=p) + co.EPS)
                note = room_note(t, float(row["quantity"]), fe, p, h, legs, grp)
                if t >= 1 and place_ok:
                    ok, msg = release(gw, sleeve, row, t)
                    done.append(("release", ref, t))
                    if not ok:
                        say(f"your held {ref} {_desc(row)} was sent to the broker at {t} and refused: {msg}; it will not be "
                            "sent again")
                        if not still(row):  # its entries have ended: never grown later, even if IBKR does have it
                            append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "unlink": ref, "why": "entries ended"})
                        continue
                    legs.append({"ref": ref, "action": row["action"], "total": float(t), "f": 0.0, "rem": float(t),
                                 "groups": grp, "key": st["key"].get(ref, ("", -1)), "pending": False, "row": row})
                    say(f"your position in {sym} was {hs} at the check, so your held {ref} {_desc(row)} went to the broker at {t}"
                        + (f" ({msg})" if msg else "") + note)
                    if still(row):      # an entry still works (the gate's rule): it grows with it; else unlinked now
                        linked[ref] = row
                    else:
                        append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "unlink": ref, "why": "entries ended"})
                    continue
                if not still(row) and t < 1:
                    append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "cancel": ref, "held": True,
                                    "result": {"status": "Cancelled", "messages": ["held: entries ended"]}})
                    done.append(("ended", ref))
                    ents = ", ".join((row["held"].get("after") or {}).keys())
                    if fe < 1 and p < 1:
                        why = f"{ents} ended with no shares filled since your held {ref} {_desc(row)} was placed"
                    elif row["action"] != closing:
                        why = (f"your position in {sym} was {hs} at the check and {ents} ended, so your held {ref} {_desc(row)} "
                               "had nothing to close")
                    else:
                        why = (f"your position in {sym} was {hs} at the check, your other close_only orders there that could fill with "
                               f"your held {ref} {_desc(row)} already cover it, and {ents} ended")
                    say(why + "; it was cancelled and never went to the broker")
            # (4) shrink: orders that could fill together within the position, the newest grouped leg first
            rems = {l["ref"]: l["rem"] for l in legs}   # co.shrink cuts the legs in place
            cuts = co.shrink(h, legs)
            stuck = [l for l in legs if l["ref"] in st["tried"] and not l["pending"]
                     and any(r == l["ref"] and nr <= co.EPS for r, nr in cuts)]
            if stuck:                               # a leg to be cut to 0 whose cancel was already tried (and failed): it
                for l in legs:                      # can't be cancelled again, so it is left as it is and the excess goes
                    l["rem"] = rems[l["ref"]]       # to the next leg (review, 20 Nov); a leg cut above 0 is still reduced
                for l in stuck:
                    l["pending"] = True
                cuts = co.shrink(h, legs)
            for ref, new_rem in cuts:
                l = next(x for x in legs if x["ref"] == ref)
                if new_rem <= co.EPS:
                    send_cancel(l, f"your position in {sym} was {hs} at the check and your other close_only orders there that "
                                   f"could fill with it already cover it, so a cancel was sent for {ref} {_desc(l['row'])}")
                    continue
                total = int(l["f"] + new_rem + co.EPS)
                if total <= l["f"] + co.EPS:            # nothing left above the filled shares: cancel, never a modify
                    send_cancel(l, f"your position in {sym} was {hs} at the check and your other close_only orders there that "
                                   f"could fill with it already cover it, so a cancel was sent for {ref} {_desc(l['row'])}")
                    continue
                if _tried_modify(st, ref, total):       # sent before and still not in force: cancelled, never re-sent
                    send_cancel(l, f"your position in {sym} was {hs} at the check; the broker did not confirm {ref} {_desc(l['row'])} "
                                   f"reduced to {co.fmt(new_rem)}, so a cancel was sent for it")
                    continue
                ok, msg = resize(gw, sleeve, ref, total, order_id=l["row"].get("order_id"))
                done.append(("reduce", ref, total, ok))
                left = co.fmt(new_rem)
                if ok == "cancel":                      # the broker counts more filled than the ledger: the rest goes
                    send_cancel(l, f"your position in {sym} was {hs} at the check, so a cancel was sent for {ref} {_desc(l['row'])}")
                    continue
                if ok in ("gone", "unsent"):            # nothing reached the broker: the next check sees the order as it is
                    continue
                if ok is True:
                    say(f"your position in {sym} was {hs} at the check, so {ref} {_desc(l['row'])} was reduced to {left} left to fill"
                        + (f" (total {total}, {co.fmt(l['f'])} filled)" if l["f"] else ""))
                elif ok is False:
                    send_cancel(l, f"your position in {sym} was {hs} at the check; {ref} {_desc(l['row'])} could not be reduced to "
                                   f"{left} ({msg}), so a cancel was sent for it")
                else:
                    say(f"your position in {sym} was {hs} at the check; the broker has not confirmed {ref} {_desc(l['row'])} reduced "
                        f"to {left} yet")
        return done
    except Exception as e:                                     # noqa: BLE001 - a failed check costs nothing but itself
        say_fail = fail_log or log
        if say_fail:
            say_fail(f"{sleeve}: close_only check failed: {type(e).__name__}: {e}")
        return done
    finally:                                                    # what was done is told even if the check stopped part-way
        if said:
            add_notes(sleeve, ["framework note (close_only): " + "; ".join(said) + "."])


def _drop_held(sleeve, log=None):
    """With close_only.V2 off, a held exit left from a time it was on can never be released: cancelled in the ledger, with
    a note (the review of 20 Nov). Never raises."""
    try:
        waiting = held_waiting(sleeve)
        if not waiting:
            return
        said = []
        for ref, row in waiting.items():
            append(ORDERS, {"time": now_iso(), "sleeve": sleeve, "cancel": ref, "held": True,
                            "result": {"status": "Cancelled", "messages": ["held: the held-exit rule is off"]}})
            said.append(f"your held {ref} {_desc(row)} was cancelled; it never went to the broker")
        add_notes(sleeve, ["framework note (close_only): " + "; ".join(said) + "."])
        if log:
            log(f"{sleeve}: close_only: held exits cancelled (rule off): {sorted(waiting)}")
    except Exception as e:                                     # noqa: BLE001
        if log:
            log(f"{sleeve}: close_only: held exits could not be dropped: {type(e).__name__}: {e}")


UNACKED_S = 300                         # a release IBKR has not acknowledged stays linked this long while not listed


def _unacked(st, ref, now):
    """True while ref's last release row is not a confirmed result (no result row yet, or IBKR's status None or
    'Unknown', not an error) and is less than UNACKED_S old: the orders may not list it yet, so it is not 'ended at the
    broker' (the review of 25 Nov: co_before_state's second pass reads the orders right after a release)."""
    rows = st["released"].get(ref) or []
    if not rows:
        return False
    last = rows[-1]
    if last.get("error"):
        return False
    res = last.get("result")
    if "result" in last and isinstance(res, dict) and res.get("status") not in (None, "Unknown"):
        return False
    try:
        at = datetime.fromisoformat(str(last.get("time")).replace("Z", "+00:00"))
    except ValueError:
        return False
    return (now - at).total_seconds() < UNACKED_S


UNSENT_TRIES = 3                        # a modify that sends nothing ("unsent") is tried again at most this many times


def _unconfirmed_up(st, ref, total):
    """The total of this ref's last modify when it was sent, not confirmed (modified None, not "gone") and above the
    broker's total now: IBKR may still apply it, so its room counts until the broker shows it or the order ends; else 0."""
    rows = st["modify"].get(ref) or []
    if not rows:
        return 0.0
    r = rows[-1]
    res = r.get("result") if isinstance(r.get("result"), dict) else None
    if r.get("sent") is False or res is None or res.get("modified") is not None or res.get("status") in ENDED:
        return 0.0
    q = float(r.get("quantity") or 0)
    return q if q > total + co.EPS else 0.0


def _tried_modify(st, ref, total):
    """This ref's last modify was to this total and was not confirmed: not sent again every minute. A modify that sent
    nothing ("sent": False) does not count until UNSENT_TRIES of them in a row were to this total."""
    rows = st["modify"].get(ref) or []
    if not rows:
        return False
    r = rows[-1]
    if r.get("quantity") != total or (isinstance(r.get("result"), dict) and r["result"].get("modified") is True):
        return False
    if r.get("sent") is False:
        tail = rows[-UNSENT_TRIES:]
        return len(tail) == UNSENT_TRIES and all(x.get("sent") is False and x.get("quantity") == total for x in tail)
    return True


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
    state = _read_json(LOOKS, {})
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
        _write_json(LOOKS, state)
    return out


NOTES = RUNS / "notes.json"   # {sleeve: [text, ...]}: placement refusals etc., shown at the next look


def add_notes(sleeve, notes):
    if not notes:
        return
    d = _read_json(NOTES, {})
    d.setdefault(sleeve, []).extend(notes)
    _write_json(NOTES, d)


def take_notes(sleeve, commit=True):
    d = _read_json(NOTES, {})
    out = d.get(sleeve, [])
    if commit and out:
        d[sleeve] = []
        _write_json(NOTES, d)
    return out
