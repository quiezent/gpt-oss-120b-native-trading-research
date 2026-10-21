"""The paper account, split into sleeves: one per model under test.

Each order a model places carries the orderRef "tt-<sleeve>-<hex>", so every fill belongs to
exactly one sleeve. A sleeve's cash, positions and realized, costed P&L are computed from its own
fills (runs/fills.jsonl), never from the account-wide positions, which mix all sleeves.

IB Gateway reports only today's executions, so sync_fills() must run at least once after the last
fill of each day (the trading loop calls it every look and after the close).
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
FILLS = RUNS / "fills.jsonl"
ORDERS = RUNS / "orders.jsonl"
SLEEVES = RUNS / "sleeves.json"   # {name: {"start_cash": float, "start": iso, "model": str}}


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
    """IBKR execution time -> US Eastern trading date (YYYYMMDD). The Gateway reports its local time (MYT,
    e.g. "20261014  22:01:05"); a time with an explicit zone (e.g. "... US/Eastern") is used as is."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    s = " ".join(str(ib_time).split())
    parts = s.split(" ")
    try:
        if len(parts) >= 3:
            dt = datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H:%M:%S").replace(tzinfo=ZoneInfo(parts[2]))
        else:
            dt = datetime.strptime(s, "%Y%m%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Kuala_Lumpur"))
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    except (ValueError, KeyError):
        return s[:8]


def sleeves():
    return json.loads(SLEEVES.read_text(encoding="utf-8")) if SLEEVES.exists() else {}


def add_sleeve(name, start_cash, model):
    s = sleeves()
    if name in s:
        raise ValueError(f"sleeve {name!r} exists")
    s[name] = {"start_cash": float(start_cash), "start": now_iso(), "model": model}
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
    """Append today's new executions (and late commission reports) to runs/fills.jsonl."""
    known = {}
    for p in ARCHIVE.glob("*/fills.jsonl"):
        for f in read(p):
            known[f["exec_id"]] = {**f, "commission": f.get("commission", 0.0)}   # archived: never re-append
    for f in read(FILLS):
        known[f["exec_id"]] = f
    new = 0
    for e in gw.executions():
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
    for f in fills():
        if f.get("sleeve") != sleeve:
            continue
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
    positions = {s: {"quantity": q, "avg_cost": avg, "opened": opened.get(s)} for s, (q, avg) in pos.items() if q}
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
            "commissions": commissions, "fills": trades, "missing_commissions": missing,
            "nav": None if unmarked else cash + value, "unmarked": unmarked}


def mark(quote):
    """A mark price from a framework quote row: last, else mid, else previous close."""
    if quote.get("last"):
        return quote["last"]
    if quote.get("bid") and quote.get("ask"):
        return (quote["bid"] + quote["ask"]) / 2
    return quote.get("close")


def place(gw, sleeve, symbol, action, quantity, order_type="MKT", limit=None, aux=None, tif="DAY", arrival=None, probs=None,
          **fields):
    """Place one order for a sleeve exactly as given and record IBKR's answer (and the arrival quote and the
    model's p_fill/p_win, if given: logged for calib_report.py, never sent to the broker)."""
    ref = new_ref(sleeve)
    row = {"time": now_iso(), "sleeve": sleeve, "ref": ref, "symbol": symbol, "action": action,
           "quantity": quantity, "order_type": order_type, "limit": limit, "aux": aux, "tif": tif, "fields": fields}
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
