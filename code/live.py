"""Trade the paper account: every sleeve's model takes a look before the open and every hour of the session.

    python live.py                          # run until runs/live/STOP exists
    python live.py --look --dry-run         # one look now for every sleeve, place nothing
    python live.py --look --only opus       # one look now for one sleeve
    python live.py --job refresh|record|macro

Schedule (US Eastern, so DST is automatic): 07:30 refresh daily bars (it takes about 40 minutes; with an ML sleeve it then
starts warm_outside.py, which scores the forecast tool's outside names in the background until 08:55 and whose summary
the loop logs when it ends); a fill sync 3 minutes
before the Gateway's day ends (midnight UTC+8, its login time zone: 12:00 ET, 11:00 ET in winter); looks at 09:00 (before the
open) and every 30 minutes until 15:30; 16:15 record fills and each sleeve's day. At each look a model sees its
sleeve and the market table (with delayed last prices during the session), may research with the
tools in tools.py, then submits orders, which are placed exactly as written. gpt-oss models run on
Tinker (agent.gptoss_look); Claude models run through the Claude CLI with only those tools
(agent.claude_look), in a background thread while the gpt-oss looks run (their research uses client 11; their orders
are placed on client 10 afterwards). Every sleeve reasons at its model's maximum effort (sleeves.json "effort").
Completed jobs are remembered (done.json), so a restart never repeats a look.
Every look is logged in runs/live/<date>/<sleeve>-<HHMM>.json and summarized in runs/live/research.jsonl.

Start the loop detached from any Claude session (a session restart kills processes it started; that lost
the 16 Oct 08:15-11:00 jobs) and with a hidden console (without it Windows Terminal shows the loop's console, and
closing that window stops the loop; 26 Oct), e.g. from PowerShell:
    $si = New-CimInstance -ClassName Win32_ProcessStartup -Property @{ShowWindow = [uint16]0} -ClientOnly
    Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CurrentDirectory='C:\\Claude\\Trader'; ProcessStartupInformation=$si;
        CommandLine='cmd.exe /c ""C:\\Claude\\Trader\\.venv\\Scripts\\python.exe" live.py >> runs\\live\\live.out.log 2>> runs\\live\\live.err.log"'}
"""

import argparse
import collections
import concurrent.futures
import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import tinker

import agent
import bars as barsmod
import checks
import close_only as co
import dividends
import inkling
import ml_tools
import paper
import rates
import regime
import tools
import trader

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "runs" / "live"
STOP = LIVE / "STOP"
HOLD = LIVE / "HOLD"   # while it exists, looks wait (not marked done) and place nothing; refresh/record run
# Present while a look runs, through its post-look fill sync (16 Nov 2026, for the re-open script, which refuses to write the
# ledger while it exists): the look's ISO time and this process's id. It changes nothing a model reads.
LOOKING = LIVE / "LOOKING"
LOOP = LIVE / "LOOP.json"     # {pid, code, looking, started} of the running loop, written when it starts
ET = ZoneInfo("America/New_York")
MCP_CLIENTS = ("trader-mcp-a", "trader-mcp-b", "trader-mcp-c", "trader-mcp-d")   # research clients 12-15, one Claude look on each
HOLD_MODEL = "hold"   # sleeves.json model of a sleeve waiting for an approved model: it gets no looks and stays in cash
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
            "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}
HALF_DAYS = {"2026-11-27", "2026-12-24", "2027-11-26"}   # NYSE closes at 13:00 ET
LOOK_TIMES = [(h, m) for h in range(9, 16) for m in (0, 30)]   # every 30 minutes, 09:00-15:30 (Clayton, 16 Oct)
# The quarter-hour looks (Clayton, 20 Nov 2026: "new base-30k-b change to 15 minutes cadence and add another base-30k. So 2
# base pair at 15 minutes cadence while the others stay at 30 minutes cadence"): 09:15, 09:45 ... 15:45, for the sleeves
# whose sleeves.json "cadence" is 15 only (cadence_of). Every sleeve, 15-minute ones too, looks at :00 and :30 as before.
QUARTER_TIMES = [(h, m) for h in range(9, 16) for m in (15, 45)]
JOBS = ([("refresh", (7, 30))]
        + [(f"look-{h:02d}{m:02d}", (h, m)) for h, m in sorted(LOOK_TIMES + QUARTER_TIMES)] + [("record", (16, 15))])
WINDOW = timedelta(minutes=25)      # a look that can't start within 25 minutes of its time is skipped (looks are 30 minutes apart)
QUARTER_WINDOW = timedelta(minutes=8)    # a quarter-hour look starts by :23 or :53, so its 5-6 minutes end before the next
                                         # half-hour look (review, 20 Nov: at 12 a late one delayed every sleeve's next look)
LAST_START = timedelta(minutes=8)        # no look starts with less than this to the close: its orders would reach IBKR after
                                         # the close (a DAY order then works at the next open; review, 20 Nov)


def _sleeves_or_none(said):
    """paper.sleeves(), or None when sleeves.json can't be read (mid-write, or a bad edit): the loop never dies of it and
    says so at most once a minute (review, 20 Nov)."""
    try:
        return paper.sleeves()
    except Exception as e:                # noqa: BLE001
        now = time.time()
        if now - said.get("t", 0) > 60:
            said["t"] = now
            log(f"sleeves.json not read ({type(e).__name__}: {str(e)[:120]}); quarter-hour looks wait for it")
        return None
GPTOSS_LANES = 20                   # gpt-oss sleeves sampled side by side (Clayton, 11 Nov 2026: "they need to be [parallel]
                                    # because trading is time sensitive"; one after another, the 09:00 look's last sleeve
                                    # decided 21 minutes after its first). 10 until 20 Nov; the Inkling sleeves take lanes in
                                    # the same pool, so from Week 7 (17 such sleeves) up to 20 start together (review, 20 Nov)


def log(msg):
    LIVE.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(ET):%Y-%m-%d %H:%M:%S} ET  {msg}"
    print(line, flush=True)
    with (LIVE / "live.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def reconcile(gw):
    """Log any difference between the account's positions and the sum of the sleeves' books (a fill lost or unattributed)."""
    want = collections.Counter()
    for name in paper.sleeves():
        for s, p in paper.book(name)["positions"].items():
            want[s] += p["quantity"]
    have = collections.Counter({p["symbol"]: p["quantity"] for p in gw.positions() if p.get("quantity")})
    diff = {s: (have[s], want[s]) for s in sorted(set(have) | set(want)) if have[s] != want[s]}
    if diff:
        log(f"POSITION MISMATCH (account, sleeves): {diff}")
        liquidations(gw)                    # did IBKR sell for margin? Such fills are in no sleeve's ledger
    return diff


def account_margin(gw):
    """Log the whole account's margin room after a look. From Mon 16 Nov 2026 software sets no limit on a sleeve's
    exposure (Clayton, 14 Nov), so IBKR's margin on the whole paper account is the only one, and the sleeves share it.
    The TWS Gateway Owner, 15 Nov: order entry needs 25% of long and 30% of short value in available funds for a liquid
    name, but IBKR adds a house margin that grows with the order's size in some stocks (what-ifs of 15 Nov: a NBIS short
    30% of value for 200 shares, 63% for 5,000, 83% for 15,000; NVDA, SPY and QQQ flat at 25 to 31% up to $3.7M). No fixed
    rate holds for every name and size: IBKR's own what-if figure and its 201 are the answer, so no rate is told to a
    model. With about $1.0M of equity one order of 35 to 40 times a $90,000 sleeve is accepted in SPY, QQQ or NVDA. From 15:50 ET
    IBKR checks Reg T (50%) through SMA and may liquidate on its own, and such fills never reach this client (reconcile
    is what shows them). SMA is IBKR's own figure (the framework reports it from c281628); it is never below equity with
    loan minus half of gross, which is shown beside it and used alone when SMA is not reported.
    A log line only: nothing is blocked, and Clayton's word of 15 Nov is to leave a breach to IBKR. Never raises."""
    try:
        a = gw.account()
        nl, gross, elv, xl = (float(a[k]) for k in ("NetLiquidation", "GrossPositionValue", "EquityWithLoanValue", "ExcessLiquidity"))
        floor = elv - 0.5 * gross
        sma = float(a["SMA"]) if a.get("SMA") is not None else None
        room = floor if sma is None else sma
        warn = nl > 0 and (room < 0.10 * nl or xl < 0.25 * nl)
        log(("ACCOUNT MARGIN WARNING: " if warn or nl <= 0 else "account margin: ") +
            f"net liquidation ${nl:,.0f}, gross ${gross:,.0f} ({gross / nl if nl > 0 else float('nan'):.2f}x), "
            f"available funds ${float(a['AvailableFunds']):,.0f}, excess liquidity ${xl:,.0f}, " +
            (f"Reg T room at the close at least ${floor:,.0f}" if sma is None
             else f"SMA ${sma:,.0f} (Reg T room at the close; never below ${floor:,.0f})"))
        return warn
    except Exception:                       # noqa: BLE001
        log("account margin not read: " + traceback.format_exc()[-200:])
        return None


_liquidations_seen = set()


def liquidations(gw, days=2):
    """IBKR's own liquidation fills of the last `days` days, each logged once. They never reach this client's own
    executions (IBKR sends them to the placing client only), so this reads every client's (the framework's
    executions(all_clients=True), c281628) and keeps the rows IBKR marks liquidation=1. Called when the account's
    positions and the sleeves' books differ. Nothing is booked here: the fills are told, for the books to be repaired.
    Never raises."""
    try:
        rows = [e for e in gw.executions(days=days, all_clients=True) if e.get("liquidation")]
    except Exception:                       # noqa: BLE001 - an older framework has no all_clients
        log("liquidation fills not read: " + traceback.format_exc()[-200:])
        return []
    for e in rows:
        if e.get("exec_id") not in _liquidations_seen:
            _liquidations_seen.add(e.get("exec_id"))
            log(f"IBKR LIQUIDATION: {e.get('side')} {e.get('shares')} {e.get('symbol')} @ {e.get('price')} at {e.get('time')} "
                f"(client {e.get('client_id')}, ref {e.get('ref')!r}, exec {e.get('exec_id')})")
    return rows


def trading_day(d):
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS


def close_time(d):
    return dtime(13, 0) if d.isoformat() in HALF_DAYS else dtime(16, 0)


def sessions_left(today):
    """Trading sessions from today through the goal date, both included."""
    n, d = 0, today
    while d <= trader.goal_date(today):
        n += trading_day(d)
        d += timedelta(days=1)
    return n


def gateway():
    from tws import Gateway
    return Gateway("trader-training", timeout=30)


_whatif_off = []   # logged once a process when the framework has no what-ifs


def open_whatif():
    """(link, ok) for text B's check_orders at this look (calculator_tools_spec_v2.md; Clayton, 14 Nov). ok: the framework has
    the what-if methods, so the Claude sleeves' MCP servers can check on their own research clients. link: one shared
    checks.Link on client 17 ("trader-whatif") for the gpt-oss threads, or None. Without the methods the tool is off with one
    log line; a failed connection costs the gpt-oss sleeves' tool for this look and logs. Never raises."""
    try:
        from tws import Gateway
    except Exception:                       # noqa: BLE001
        Gateway = None
    if Gateway is None or not checks.supported(Gateway):
        if not _whatif_off:
            _whatif_off.append(1)
            log("check_orders off: the tws framework has no whatif_many/all_open_orders (the account line still shows)")
        return None, False
    try:
        return checks.Link(Gateway("trader-whatif", timeout=10)), True
    except Exception:                       # noqa: BLE001
        log("check_orders off for the gpt-oss sleeves this look: the what-if connection (trader-whatif, 17) failed: "
            + traceback.format_exc()[-200:])
        return None, True


def read_account(link, gw):
    """account() for text B's account line, once a look: on the what-if link if it is open (its wait bounded at
    checks.READ_TIMEOUT, and reconnected once on GatewayError 504 or 1100 after the 05:00 ET restart, as the framework asks),
    else on client 10 (the same bound, under its lock, so it holds the other sleeves no longer; no reconnect of the look's
    own connection). The row, or None, logged. Never raises: a failure costs this look's account line only.
    link may be a checks.Link on client 10 itself (live.look passes the one it also reads positions() on)."""
    try:
        if link is not None and link.gw is not gw:
            return checks.read_account(link, log)
        return checks.read_account(link if link is not None else checks.Link(gw), log, reconnect=False)
    except Exception:                       # noqa: BLE001
        log("account line: IBKR's figures not read for this look: " + traceback.format_exc()[-200:])
        return None


def unheld_gross(link, data, d, live, gw=None):
    """The dollars of the account's gross in positions no sleeve holds, for the account line (16 Nov 2026: the Gateway may
    still list names IBKR removed at the weekend's cash reset, 70 names and $775,595 of gross on 15 Nov, which no Week 6
    sleeve holds). positions() is read once a look on the account line's connection, with its 5 s bound and skip rule
    (checks.Link.read). Per symbol, the account's signed position A less the sleeves' ledgers' sum L: the part the ledgers
    do not explain is A - L when A is the same sign as L and larger, all of A when L is the other sign (or zero), else
    nothing. Valued at the look's marks (delayed last, else last close), else the row's average cost; summed in absolute
    value. None (one log line) when positions() or a ledger was not read: the clause is then left out.
    24 Nov 2026: with gw (client 10), the ledgers are summed twice: as they stand when positions() is read, and again after
    the fills IBKR has reported since are booked (paper.sync_fills); per symbol the smaller unexplained part counts. At
    09:30 the look syncs at 09:30:45 and reads the account about 35-85 s later, after every held name's quote; the fills in
    between were in positions() and in no ledger, so the clause counted the sleeves' own opening trades ($257,994 on 23 Nov,
    $8,686 on 24 Nov): the second sum explains those. A fill after positions() was read (a stop covering a short) would
    leave the second sum short of the account; the first sum explains that one. A position no sleeve holds is unexplained
    by both and counts in full when no sleeve trades that name in either window. Known limits (the review of 24 Nov): a
    sleeve's fill on such a name in either window can make one sum explain part of it, so the figure can be too low by
    up to that fill; and a fill toward the account's sign before the read with one against it after (in one sleeve or
    across sleeves) is still counted, up to the smaller of the two. Splitting the synced fills by execution time against
    the read would remove both. A failed sync leaves the clause out."""
    def held_now():
        held = {}
        for name in paper.sleeves():
            for s, p in paper.book(name)["positions"].items():
                held[s] = held.get(s, 0) + float(p["quantity"])
        return held

    def unexplained(a, h):
        return a - h if (a * h > 0 and abs(a) > abs(h)) else a if a * h <= 0 else 0.0

    try:
        rows = link.read("positions")
        acct, cost = {}, {}
        for r in rows or []:
            if r.get("quantity"):
                acct[r["symbol"]] = acct.get(r["symbol"], 0) + float(r["quantity"])
                cost.setdefault(r["symbol"], r.get("avg_cost"))
        before = after = held_now()
        if gw is not None:
            paper.sync_fills(gw)
            after = held_now()
        marks = {s: x["close"] for s, x in data.features(d)[0].items()} | (live or {})
        total = 0.0
        for s, a in acct.items():
            u1, u2 = unexplained(a, before.get(s, 0.0)), unexplained(a, after.get(s, 0.0))
            u = u1 if abs(u1) <= abs(u2) else u2       # both have a's sign (or are 0)
            if u:
                px = marks.get(s) if marks.get(s) is not None else cost.get(s)
                total += abs(u) * float(px)
        return total
    except Exception as e:                  # noqa: BLE001
        log(f"account line: positions no sleeve holds not computed ({type(e).__name__}: {str(e)[:150]})")
        return None


def settle_orders(rows, sleeve):
    """An orders() snapshot read before a fill sync, settled by the fills the sync then booked (24 Nov 2026, the review of
    catch_up): each of this sleeve's working rows whose booked fills reach its quantity is marked Filled (so
    paper.open_orders leaves it out); one partly filled has remaining set to its quantity less those fills (or IBKR's
    remaining, if that is smaller). Other sleeves' rows and rows IBKR already ended are left as they are. New dicts; the
    snapshot is not changed. A fill booked can't be undone, so this never hides an order that can still fill."""
    done = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
    booked = {}
    for f in paper.fills():
        if f.get("sleeve") == sleeve and f.get("ref"):
            booked[f["ref"]] = booked.get(f["ref"], 0.0) + float(f.get("shares") or 0)
    out = []
    for r in rows or []:
        q = booked.get(r.get("ref"), 0.0)
        if paper.sleeve_of(r.get("ref")) != sleeve or r.get("status") in done or q <= 0:
            out.append(r)
            continue
        total = float(r.get("quantity") or 0)
        if q >= total:
            out.append({**r, "status": "Filled"})
        else:
            left, rem = total - q, r.get("remaining")
            if isinstance(rem, (int, float)) and not isinstance(rem, bool):
                left = min(left, float(rem))
            out.append({**r, "remaining": left})
    return out


def catch_up_sleeve(gw, name, live=None, dry_run=False):
    """Just before a sleeve's state is built: reads IBKR's orders, then books the fills IBKR has reported since the
    last sync, then quotes this sleeve's held names the look has no mark for. Returns the orders rows, which the
    state is built from (build's orders: its open-orders list, its names in play and the orders IBKR ended in
    "Since your last decision"). Every sleeve, the first gpt-oss sleeve and the Claude sleeves included (24 Nov
    2026: at the 09:30 look on 23 Nov those two were built from the books as at the 09:30:45 sync, about 80 s
    before their states, while their orders were read live as each state was built, so base-90k-ml was shown 14
    shorts its stops had covered at the open, with no stop, and 4.11x gross for 2.94x, and covered them again;
    fable-ml was shown three positions it had closed). Orders first, then fills, the order close_only v2.1
    (DESIGN C) uses, so an order that fills between the two reads (a window that includes waiting for client
    10 behind other sleeves' research calls) is never shown as neither held nor working: once the sync has
    booked its fills, its snapshot row is settled from them (filled in full: shown filled, so not listed;
    partly: its remaining is what the booked fills leave). One that fills later (during the quotes or the
    build) shows as working with no fill yet, as it stood at the sync, and its fill reaches the next look. An
    error raises: it costs this sleeve's look, as a failed re-sync has since 11 Nov.

    25 Nov 2026 (the Researcher's choice of 24 Nov): with close_only v2.1 on and not in a dry run, this sleeve's
    close_only check runs between the fill sync and the settling (co_before_state), on those orders and fills, so
    the state shows each exit as it stands at the broker; if the check changed something at the broker, the orders
    are read and the fills synced again, in that order, and the state is built from that second read."""
    snap = gw.orders()
    paper.sync_fills(gw)
    snap = co_before_state(gw, name, snap, dry_run)
    snap = settle_orders(snap, name)
    if live is not None:            # the look's marks: held names it has none for are quoted now
        missing = sorted(set(paper.book(name)["positions"]) - set(live))
        for q in (gw.quotes(missing) if missing else []):
            m = None if q.get("error") else paper.mark(q)
            if m is not None:
                live[q["symbol"]] = m
    return snap


def co_before_state(gw, sleeve, rows, dry_run=False):
    """A sleeve's close_only check just before its state is built (25 Nov 2026; the Trading Model Researcher's choice of
    24 Nov), inside catch_up_sleeve: rows is IBKR's orders, read before the fill sync that has just run (DESIGN C's
    order). With close_only v2.1 off or in a dry run it does nothing and returns rows. Else the check runs on them, so
    the state shows each exit as it stands at the broker (23 Nov, 09:30: 14 held rows in 9 sleeves were shown "held
    (waits for <ref>)" beside entries already filled; 8 looks cancelled an exit and 4 placed one again). When it changed
    something at the broker (a release or a cancel sent; a raise or reduction confirmed, not yet confirmed, or met by an
    order that had ended), IBKR's orders are read again and the fills synced again, in that order, and those rows are
    returned for the state; else rows. A modify that sent nothing, or a held row cancelled or expired in the ledger, needs
    no new read. One pass (the third review of 25 Nov: a second pass read the orders before IBKR listed an unacknowledged
    release and sent a second exit into its room, and turned an unconfirmed reduction into a cancel within seconds); a
    fill that lands while the check waits for IBKR is booked by that sync and shows as it stood (an entry filled, its
    exit still held), and the next check (act's tail, the minute check) acts on it. The time is logged when the check
    changed something or took 2 s or more (it runs under the look's ledger lock, and IBKR's acknowledgement can take
    seconds). The check never raises (a failure is logged and the state is built as before); a failed read here raises,
    as catch_up_sleeve's own do."""
    if dry_run or not co.V2:
        return rows
    t0, moved = time.time(), []
    try:
        done = paper.close_only_check(gw, sleeve, log=log, fail_log=log, rows=rows, synced=True, place_ok=not HOLD.exists())
        moved = [d for d in done if d[0] in ("cancel", "release")
                 or (d[0] in ("grow", "reduce") and d[3] in (True, None, "gone"))]
    finally:
        secs = time.time() - t0
        if moved or secs >= 2:
            log(f"{sleeve}: close_only check before the state: {len(moved)} change(s), {secs:.1f}s")
    if not moved:
        return rows
    rows = gw.orders()
    paper.sync_fills(gw)
    return rows


class OrdersAt:
    """Client 10 with orders() answered from one snapshot (catch_up's, read before the fills are synced), for one sleeve's
    state build only (24 Nov 2026); every other call goes to the connection itself. build() puts the connection back on
    the world before it returns, so placing and close_only read IBKR's orders live."""

    def __init__(self, gw, rows):
        self._gw, self._rows = gw, list(rows or [])

    def orders(self):
        return [dict(r) for r in self._rows]

    def __getattr__(self, name):
        return getattr(self._gw, name)


def sleeves_total(data, d, live):
    """The sleeves' values summed, each from its ledger at the look's marks (delayed last prices, else the last close, else
    cost), for the account line's "whose values total". None (logged) if it fails."""
    try:
        marks = {s: x["close"] for s, x in data.features(d)[0].items()} | (live or {})
        total = 0.0
        for name in paper.sleeves():
            b = paper.book(name, marks)
            total += b["cash"] + sum(p["quantity"] * (p.get("mark") if p.get("mark") is not None else p["avg_cost"])
                                     for p in b["positions"].values())
        return total
    except Exception:                       # noqa: BLE001
        log("account line: the sleeves' total not computed: " + traceback.format_exc()[-200:])
        return None


def refresh():
    syms = trader.universe()
    held = set()
    for name in paper.sleeves():
        held |= set(paper.book(name)["positions"])
    try:                                                # names traded in the last days too: one sold yesterday still needs
        import carry_record                             # yesterday's bar for the record of its last carried night
        held |= carry_record.recent_names()
    except Exception:                                   # noqa: BLE001
        log("recent names for the bars refresh not read: " + traceback.format_exc()[-200:])
    bad = []
    with gateway() as gw:
        for s in sorted(set(syms) | held):
            try:
                barsmod.refresh(gw, s)
            except Exception as e:
                bad.append(f"{s}: {e}")
    log(f"refreshed bars for {len(set(syms) | held)} symbols; {len(bad)} failed {bad[:5]}")
    book_dividends()                                    # a missed record's dividends, before the first look
    try:                                                # each sleeve's carried book and what the open did to it, for the
        import carry_record                             # nights whose bars are in (carry_record.py; a record, read by no
        log(carry_record.update())                      # model). update() never raises; the import might.
    except Exception:                                   # noqa: BLE001
        log("carry record failed: " + traceback.format_exc()[-300:])
    try:                                                # every sleeve's rate line (rates.py); never fails the refresh
        r = rates.fetch()
        log(f"rates: target {r['target']} as of {r['target_as_of']}, 3-month bill {r['bill3m']} as of {r['bill3m_as_of']}")
    except Exception as e:                              # noqa: BLE001 - ml_research.fred redacts the key
        log(f"rates fetch failed ({type(e).__name__}); the line keeps the last figures until they are {rates.STALE_DAYS} days old")
    if any(m.get("ml") for m in paper.sleeves().values()):
        import fetch_extra                              # inputs of the ML arm's cost and market-vol forecasts
        with gateway() as gw:
            u = sorted(set(syms))
            bad = fetch_extra.update(gw, ["SPY"] + [x for x in u if x != "SPY"], "iv") + fetch_extra.update(gw, u, "bidask")
        log(f"refreshed implied vol and bid/ask history; {len(bad)} failed {bad[:3]}")
        compute_versions()
        warm_outside()                                  # in the background; never fails the refresh
        ml2_guard()
        try:                                            # earnings calendar + history (wave C); never fails the refresh
            import fetch_av
            r = fetch_av.daily(22)
            log(f"Alpha Vantage: calendar rows {r['calendar']}, history +{len(r['history'])}, {r['remaining']} to go, "
                f"{len(r['errors'])} errors {r['errors'][:2]}")
        except Exception as e:                          # noqa: BLE001
            log(f"Alpha Vantage capture failed: {type(e).__name__}")


COMPUTE_RETRY = {"day": None, "versions": [], "last": None}   # versions whose morning compute failed: compute_retry()


def compute_versions(only=None):
    """Each version in use writes its own cache, ML2 first. ML2 is computed whenever any version is in use, on a sleeve
    or not: ML3 and ML4 read ML2's file, and from Week 6 (16 Nov 2026) no sleeve is on ML2, so without this they would
    serve ML2's last file as stale. ML2's failure raises, so the refresh retries as it always has. Another version's is
    logged and goes on COMPUTE_RETRY, and the loop tries that version again alone (compute_retry): it never costs the
    other versions their forecasts, and one missing column (gap1) never re-runs the whole refresh. An unregistered
    version has no look anyway. only: just these versions (the retry). Returns the versions that failed."""
    used = ml_tools.in_use(paper.sleeves())
    order = ([ml_tools.DEFAULT] if used and ml_tools.DEFAULT not in used else []) + used
    if only is not None:
        order = [v for v in order if v in only]
    failed = []
    for v in order:
        if v not in ml_tools.VERSIONS:
            log(f"no forecasts for {v}: not in ml_tools.VERSIONS")
            continue
        try:
            as_of, fc = ml_tools.module(v).compute()
            log(f"forecasts for the ML arm ({ml_tools.module(v).VERSION}): as of {as_of}, {len(fc)} symbols")
        except Exception:                               # noqa: BLE001
            if v == ml_tools.DEFAULT:
                raise
            failed.append(v)
            log(f"{v} forecasts failed: " + traceback.format_exc()[-300:])
    if only is None:                                    # the morning's result; the retry keeps its own
        now = datetime.now(ET)
        COMPUTE_RETRY.update(day=now.date(), versions=failed, last=now)
    return failed


def compute_retry(now=None):
    """Try again, alone, each version whose compute failed at this morning's refresh: every 5 minutes until it succeeds
    or the day's last look time (15:45, a 15-minute sleeve's; 12:45 on a half day) has passed. The loop calls it between jobs, so never during a look; a compute
    takes seconds and a look reads a new file at its next call. Never raises. (16 Nov 2026: with every sleeve on ML4, a
    failed ML4 compute would have left gap1 "unavailable" all day with no second try; the Trading Model Researcher.)"""
    try:
        st = COMPUTE_RETRY
        now = now or datetime.now(ET)
        last = (datetime.combine(now.date(), close_time(now.date())) - timedelta(minutes=15)).time()
        if not st["versions"] or st["day"] != now.date() or now.time() >= last:
            return
        if st["last"] is not None and now - st["last"] < timedelta(minutes=5):
            return
        st["last"] = now
        log(f"forecasts: trying {', '.join(st['versions'])} again")
        st["versions"] = compute_versions(only=list(st["versions"]))
        log("forecasts: the retry succeeded" if not st["versions"] else f"forecasts: {', '.join(st['versions'])} still failing")
    except Exception:                                   # noqa: BLE001
        log("forecasts: the retry failed: " + traceback.format_exc()[-300:])


WARM = None     # the outside-name warm-up's process (warm_outside.py), so two never run at once
WARM_T = ""     # when it started, ET, as warm_outside.jsonl's "t": its own summary line is the one at or after it


def warm_outside(now=None, popen=subprocess.Popen):
    """Start warm_outside.py in the background once the forecasts are computed: before the first look it scores the
    names outside ML2's cached universe (recently asked, then the broad universe) with the look's own functions, so a
    look reads them at once instead of scoring them inside its time cap (Clayton, 9 Nov: "a major issue"; a fix for
    every ML sleeve, Base + ML2.2 included, 9 Nov 13:00 UTC+8). Only on a trading day with a sleeve that has the tool,
    never two at once; never blocks or raises."""
    global WARM, WARM_T
    try:
        now = now or datetime.now(ET)
        if not trading_day(now.date()) or not any(m.get("ml") and m.get("model") != HOLD_MODEL
                                                  for m in paper.sleeves().values()):
            return
        warm_poll()                                     # a run that has ended is logged first
        if WARM is not None:
            log(f"outside-name warm-up still running (pid {WARM.pid}); not started again")
            return
        LIVE.mkdir(parents=True, exist_ok=True)
        with open(LIVE / "warm_outside.out.log", "a", encoding="utf-8") as out:     # the child has its own handle
            out.write(f"=== {now:%Y-%m-%d %H:%M:%S} ET\n")
            out.flush()
            WARM_T = f"{now:%Y-%m-%d %H:%M:%S}"
            WARM = popen([sys.executable, str(ROOT / "warm_outside.py")], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        log(f"outside-name warm-up started (pid {WARM.pid}); per name in runs/live/warm_outside.jsonl")
    except Exception as e:                              # noqa: BLE001
        try:
            log(f"outside-name warm-up not started: {type(e).__name__}: {str(e)[:200]}")
        except Exception:                               # noqa: BLE001
            pass


def warm_poll():
    """Each loop pass, as macro_intraday polls its build: once the warm-up has ended, log its exit code and its summary
    line (stopped, names, ML2 rows kept). Never raises."""
    global WARM
    try:
        code = None if WARM is None else WARM.poll()
        if code is None:
            return
        WARM, s = None, None
        try:
            for ln in reversed((LIVE / "warm_outside.jsonl").read_text(encoding="utf-8").splitlines()):
                d = json.loads(ln) if '"summary"' in ln else {}
                if d.get("summary"):
                    s = d if str(d.get("t", "")) >= WARM_T else None
                    break
        except (OSError, ValueError, AttributeError):
            s = None
        log(f"outside-name warm-up ended (exit {code}): "
            + (f"stopped {s.get('stopped')}, {s.get('names')} names, {s.get('ml2_kept')} ML2 rows kept" if s
               else "no summary line from this run in runs/live/warm_outside.jsonl"))
    except Exception as e:                              # noqa: BLE001
        try:
            log(f"outside-name warm-up poll failed: {type(e).__name__}: {str(e)[:200]}")
        except Exception:                               # noqa: BLE001
            pass


def ml2_guard():
    """Logged, never blocking: an upgrade must not reach ML2 (ml_tools.changed_files)."""
    try:
        moved = ml_tools.changed_files()
        if moved:
            log(f"ML2 CHANGED since its manifest: {moved}")
    except Exception:                                   # noqa: BLE001
        log("ML2 guard failed: " + traceback.format_exc()[-300:])


def book_dividends(gw=None):
    """dividends.book with its result logged, on gw or a new client-10 connection; never raises (F6, Clayton 26 Oct)."""
    try:
        if gw is None:
            with gateway() as g:
                rows = dividends.book(g, log=log)
        else:
            rows = dividends.book(gw, log=log)
        log(f"dividends: {len(rows)} booked" + "".join(f"; {r['sleeve']} {r['symbol']} ex {r['ex_date']} {r['quantity']:+} x "
                                                      f"{r['per_share']} = {r['amount']:+.2f}" for r in rows))
    except Exception:                                   # noqa: BLE001
        log("dividends failed: " + traceback.format_exc()[-300:])


NOT_RUN = "context nearly full ("      # how agent.gptoss_look answers a call it did not run for lack of room


def research_summary(trace, call, table_rows):
    """What a look researched. A call the loop did not run for lack of room is no research and no check, and its symbols
    were not researched (16 Nov 2026: such a call came back with one line, not data); a dropped turn has no "tool" at all.
    A call answered with a limit text is counted as before."""
    ran = lambda t: not str(t.get("result", "")).startswith(NOT_RUN) and not (     # an Inkling reply's later call after
        t.get("tool") != "submit" and isinstance(t.get("not_run"), str))           # the line that opened the submit (20 Nov)
    not_run = sum(1 for t in trace if t.get("tool") and t["tool"] != "submit" and not ran(t))
    trace = [t for t in trace if ran(t)]
    calls = [t for t in trace if t.get("tool") and t["tool"] not in ("submit", "check_orders")]   # checks are not research (14 Nov)
    used = {}
    researched = set()
    for t in calls:
        used[t["tool"]] = used.get(t["tool"], 0) + 1
        researched |= set(tools.symbols_in(t["tool"], t.get("args")))
    traded = sorted({str(o.get("symbol", "")).upper().strip() for o in (call or {}).get("place", []) if isinstance(o, dict)})
    return {"research_calls": len(calls), "tools": used, "researched": sorted(researched), "traded": traded,
            "traded_researched": sorted(set(traded) & researched), "traded_not_in_table": sorted(set(traded) - set(table_rows)),
            "order_checks": sum(1 for t in trace if t.get("tool") == "check_orders"),
            "calls_not_run": not_run, "turns_dropped": sum(1 for t in trace if t.get("dropped"))}


MEMO = LIVE / "memo.json"   # each sleeve's own notes from its last look (submit's `notes`), shown at its next look


def load_memo(name):
    return paper._read_json(MEMO, {}).get(name, {})


NOTES_CAP = {"claude": 8000, "gpt-oss": 2000}   # characters kept for the next look. gpt-oss stays at the 2,000 it was
                                                # trained with; Opus, told its cap, was cut silently at 2,000 in week 2
                                                # (118 memos, a median 28% lost)


def save_memo(name, now, call, model=""):
    if not call:                      # no valid decision: the previous notes stay
        return
    cap = NOTES_CAP["claude" if str(model).startswith("claude-") else "gpt-oss"]
    d = paper._read_json(MEMO, {})
    d[name] = {"when": f"{now:%a %d %b %H:%M} ET", "notes": str(call.get("notes") or "")[:cap]}
    paper._write_json(MEMO, d, indent=1)


def effort(meta):
    """The sleeve's reasoning effort: each model's maximum unless sleeves.json says otherwise (Clayton, 18 Oct)."""
    return meta.get("effort") or ("max" if meta["model"].startswith("claude-") or inkling.is_inkling(meta["model"]) else "high")


def forced_mode_of(meta):
    """How a gpt-oss sleeve's opened submit is written (agent.gptoss_look's forced_mode): the sleeve's "forced_mode" if it is
    one of agent.FORCED_MODES, else agent.FORCED_MODE ("plain" until it is changed). Per sleeve because the Researcher's
    second forced-submit draw (15 Nov, preliminary) read base and the candidates differently, so the setting may differ
    by sleeve; no sleeve sets it until Clayton says which. None for a Claude sleeve (its submit is never opened for it)
    and for an Inkling sleeve (inkling.look names submit in tool_choice instead; 20 Nov)."""
    if str(meta.get("model", "")).startswith("claude-") or inkling.is_inkling(meta.get("model")):
        return None
    v = meta.get("forced_mode")
    return v if v in agent.FORCED_MODES else agent.FORCED_MODE


def final_line_of(meta):
    """Whether a gpt-oss sleeve's limit line that opens the submit states the facts (agent.gptoss_look's final_line): the
    sleeve's "final_line" if it is true or false, else agent.FINAL_LINE (off until it is changed). None for a Claude sleeve."""
    if str(meta.get("model", "")).startswith("claude-"):
        return None
    v = meta.get("final_line")
    return v if isinstance(v, bool) else agent.FINAL_LINE


def cadence_of(meta):
    """A sleeve's minutes between looks: sleeves.json "cadence" 15 (a gpt-oss or Inkling sleeve; it also gets the
    quarter-hour looks), else 30. A Claude sleeve stays at 30 (its CLI lanes are capped by WINDOW; switch_bad says so)."""
    v = meta.get("cadence")
    return 15 if v == 15 and not isinstance(v, bool) and not str(meta.get("model", "")).startswith("claude-") else 30


def quarter_sleeves(sleeves):
    """The sleeves a quarter-hour look runs: those at cadence 15."""
    return sorted(k for k, v in sleeves.items() if cadence_of(v) == 15)


def base_model_of(meta):
    """The base model a base-weight gpt-oss sleeve samples on: sleeves.json "base_model" if it names BASE or a variant of it
    ("openai/gpt-oss-120b:peft:131072"; the Trading Model Researcher, 20 Nov 2026, for the text-B base sleeves from Week 7
    on its decision of Sat 21 Nov), else BASE. None for a checkpoint (it runs on its own trained base), Claude or Inkling
    sleeve."""
    if str(meta.get("model", "")) not in ("base", BASE):
        return None
    v = meta.get("base_model")
    return v if isinstance(v, str) and (v == BASE or variant_context(v)) else BASE


def variant_context(name):
    """n for a base model named exactly "<BASE>:peft:<n>" with n a whole number from 32,768 to 1,048,576, else None."""
    m = re.fullmatch(re.escape(BASE) + r":peft:(\d+)", str(name or ""))
    n = int(m.group(1)) if m else None
    return n if n is not None and agent.CONTEXT <= n <= 1048576 else None


def context_of(meta):
    """The look's context: an Inkling sleeve's (inkling.context_of); a base-weight gpt-oss sleeve on a variant named
    "<BASE>:peft:<n>", its "context" if a whole number up to n, else n; any other gpt-oss sleeve agent.CONTEXT (32,768);
    None for a Claude sleeve. A "context" without such a variant changes nothing (switch_bad says so)."""
    m = str(meta.get("model", ""))
    if m.startswith("claude-"):
        return None
    if inkling.is_inkling(m):
        return inkling.context_of(meta)
    n = variant_context(base_model_of(meta))
    if not n:
        return agent.CONTEXT
    v = meta.get("context")
    return v if isinstance(v, int) and not isinstance(v, bool) and agent.MIN_ROOM < v <= n else n


def next_look_of(meta):
    """Whether the sleeve's system text has trader.NEXT_LOOK: sleeves.json "next_look" true on a text-B sleeve that is not a
    Claude sleeve (the Researcher, 20 Nov: per sleeve, so a sleeve can have a twin without it)."""
    return (meta.get("next_look") is True and trader.text_of(meta) == "B"
            and not str(meta.get("model", "")).startswith("claude-"))


def range_line_of(meta):
    """Whether the sleeve's state has the range line (trader.book_range): sleeves.json "range_line" true, the same sleeves."""
    return (meta.get("range_line") is True and trader.text_of(meta) == "B"
            and not str(meta.get("model", "")).startswith("claude-"))


def switch_bad(meta):
    """The sleeve's malformed forced-submit settings, as log text: a "forced_mode" not in agent.FORCED_MODES, a "final_line"
    that is not true or false, or either on a Claude sleeve (which has no opened submit). [] if none."""
    out, claude = [], str(meta.get("model", "")).startswith("claude-")
    if "cadence" in meta and cadence_of(meta) != meta["cadence"]:
        out.append(f"cadence {meta['cadence']!r} is not used; the sleeve looks every {cadence_of(meta)} minutes "
                   "(15 or 30; a Claude sleeve stays at 30)")
    for key, on in (("next_look", next_look_of), ("range_line", range_line_of)):
        if key in meta and bool(meta[key]) != on(meta):
            out.append(f"{key} {meta[key]!r} is not used: only true, on a text-B sleeve that is not a Claude sleeve")
    if inkling.is_inkling(meta.get("model")):   # an entry copied from a gpt-oss twin (the Researcher's read, 20 Nov)
        try:
            if inkling.effort_value(effort(meta)) < inkling.EFFORT:
                out.append(f"effort {meta.get('effort')!r} runs an Inkling sleeve at {inkling.effort_value(effort(meta))}, "
                           f"under its maximum {inkling.EFFORT}")
        except ValueError as e:
            out.append(f"effort: {e}")
        if not isinstance(meta.get("final_line"), bool):
            out.append(f"final_line is not set on an Inkling sleeve, so the default {agent.FINAL_LINE!r} applies")
    if "forced_mode" in meta and inkling.is_inkling(meta.get("model")):
        out.append(f"forced_mode {meta['forced_mode']!r} is ignored on an Inkling sleeve (its submit is opened by tool_choice)")
    if "base_model" in meta and base_model_of(meta) != meta["base_model"]:
        out.append(f"base_model {meta['base_model']!r} is ignored: only a base-weight gpt-oss sleeve takes one, naming {BASE} or a variant of it")
    if "context" in meta and context_of(meta) != meta["context"]:
        out.append(f"context {meta['context']!r} is not used; the look's context is {context_of(meta)}")
    for key, good, want in (("forced_mode", lambda v: v in agent.FORCED_MODES, f"one of {list(agent.FORCED_MODES)}"),
                            ("final_line", lambda v: isinstance(v, bool), "true or false")):
        if key in meta and claude:
            out.append(f"{key} {meta[key]!r} is ignored on a Claude sleeve (its submit is never opened for it)")
        elif key in meta and not good(meta[key]):
            out.append(f"{key} {meta[key]!r} is not {want}, so the default "
                       f"{agent.FORCED_MODE if key == 'forced_mode' else agent.FINAL_LINE!r} applies")
    return out


def finish(now, name, *a, **k):
    try:                                # a failed record costs that record only: the chain goes on and the Claude orders are placed
        record_look(now, name, *a, **k)
    except Exception:                   # noqa: BLE001
        log(f"{name}: look record failed: " + traceback.format_exc()[-300:])


def words_of(words):
    """(system text hash, tool words hash) for the look record (the Researcher, 20 Nov: nothing in a record showed that twins
    read the same words): sha256 of the system text, inkling.words_sha of the tools offered; None for what is not given."""
    import hashlib
    system, offered = words if words else (None, None)
    return (hashlib.sha256(system.encode()).hexdigest()[:16] if isinstance(system, str) else None,
            inkling.words_sha(offered) if offered else None)


def record_look(now, name, meta, text, st, call, trace, notes, table_rows, dry_run, cost=None, secs=None, words=None):
    if not dry_run:                     # first: later looks read these two, and the rest is only the record
        save_memo(name, now, call, meta["model"])
        if not call:                    # no decision, so the next look's "Since your last decision" must repeat what this one was told
            paper.add_notes(name, st["events"])
    out_dir = LIVE / now.date().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = research_summary(trace, call, table_rows)
    rec = {"time": now.isoformat(timespec="seconds"), "sleeve": name, "model": meta["model"], "effort": effort(meta),
           "seconds": secs, "state_text": text,
           "state": {k: v for k, v in st.items() if k != "events"}, "events": st["events"], "trace": trace, "call": call,
           "placed_notes": notes, "research": summ, "cost": cost, "dry_run": dry_run,
           "ml_tool": ml_tools.version_of(meta), "kelly_prompt": bool(meta.get("kelly_prompt")),
           "shock_cap": trader.shock_cap_of(meta), "max_gross": trader.max_gross_of(meta), "max_calls": trader.max_calls_of(meta),
           "forced_mode": forced_mode_of(meta), "final_line": final_line_of(meta), "own_limit": trader.own_limit_of(meta),
           "regime_line": bool(meta.get("regime_line")), "regime_text": regime.shown(text) if meta.get("regime_line") else None,
           "text": trader.text_of(meta), "base_model": base_model_of(meta), "context": context_of(meta),
           "cadence": cadence_of(meta), "next_look": next_look_of(meta), "range_line": range_line_of(meta),
           "system_sha": words_of(words)[0], "tools_sha": words_of(words)[1]}
    (out_dir / f"{name}-{now:%H%M}.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    if not dry_run:
        paper.append(LIVE / "research.jsonl", {"time": rec["time"], "sleeve": name, "max_calls": rec["max_calls"],
                                               "forced_mode": rec["forced_mode"], "final_line": rec["final_line"],
                                               "own_limit": rec["own_limit"], **summ,
                                               "orders": len(call["place"]) if call else None,
                                               "notes_chars": len(str((call or {}).get("notes") or ""))})
    log(f"{name}: {'DRY RUN ' if dry_run else ''}"
        + (f"{len(call['place'])} orders, {len(call['cancel'])} cancels" if call else "no valid decision")
        + f"; research {summ['research_calls']} {summ['tools'] or ''}; traded {summ['traded']}"
        + (f"; {secs:.0f}s" if secs is not None else "")
        + (f"; notes {notes}" if notes else ""))


def pid_alive(pid):
    """True if process `pid` is running, without touching it: OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION, then
    GetExitCodeProcess == STILL_ACTIVE (259); access denied counts as alive (the TWS Gateway Owner's review, 16 Nov 2026).
    Tests replace it."""
    import ctypes
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return ctypes.get_last_error() == 5     # ERROR_ACCESS_DENIED: it exists
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code)):
            return True                         # it exists; its state could not be read
        return code.value == 259
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))


def own_marker():
    """This process's marker: LOOKING.<pid> beside LOOKING (each process its own file, so one process's look never
    replaces or removes another's: the TWS Gateway Owner's review, 16 Nov 2026)."""
    return LOOKING.with_name(f"{LOOKING.name}.{os.getpid()}")


def marker_files():
    """Every marker file beside LOOKING: the plain LOOKING (written by loops up to 23a7075) and each LOOKING.<pid>."""
    try:
        return sorted(p for p in LOOKING.parent.iterdir() if p.name == LOOKING.name
                      or (p.name.startswith(LOOKING.name + ".") and p.name[len(LOOKING.name) + 1:].isdigit()))
    except OSError:                         # no folder yet
        return []


def marker_pid(path=None):
    """(pid, text) of a marker file ("<ISO time> <pid>"; this process's own if none given); pid None if there is no
    marker or it names no pid."""
    try:
        text = (path or own_marker()).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:                         # none there (or unreadable: then treated as none)
        return None, None
    try:
        return int(text.split()[-1]), text
    except (ValueError, IndexError):
        return None, text


def begin_marker():
    """Write this process's marker (LOOKING.<pid>) for its look. Another process's marker is never read, replaced or
    removed. Never raises."""
    try:
        path = own_marker()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{datetime.now(ET).isoformat()} {os.getpid()}\n", encoding="utf-8")
    except Exception:                       # noqa: BLE001
        log("LOOKING marker not written: " + traceback.format_exc()[-200:])


def end_marker():
    """Remove this process's marker (LOOKING.<pid>), if there; never another process's. Never raises."""
    try:
        own_marker().unlink(missing_ok=True)
    except Exception:                       # noqa: BLE001
        log("LOOKING marker not removed: " + traceback.format_exc()[-200:])


def look(dry_run=False, only=None, marked=False):
    """One look for every sleeve (dry runs too), with this process's marker (LOOKING.<pid>) present from its start through
    its post-look fill sync. marked: loop() has already written it, before marking the look done. The marker is removed
    in a finally (only this process's own file); writing or removing it never fails a look."""
    if not marked:
        begin_marker()
    try:
        _look(dry_run, only)
    finally:
        end_marker()


def _look(dry_run=False, only=None):
    now = datetime.now(ET)
    today = now.date()
    sleeves = {k: v for k, v in paper.sleeves().items() if not only or k in only}
    held = sorted(k for k, v in sleeves.items() if v.get("model") == HOLD_MODEL)
    if held:                                # Clayton, 24 Oct: a sleeve without an approved model stays in cash, no look
        log(f"no look for {', '.join(held)}: model '{HOLD_MODEL}' (no approved model yet); in cash")
        sleeves = {k: v for k, v in sleeves.items() if k not in held}
    ml2_guard()
    for k, v in sleeves.items():            # a malformed sleeve term is ignored by trader.shock_cap_of: say so at every look
        if "shock_cap" in v and trader.shock_cap_of(v) is None:
            log(f"{k}: shock_cap {v['shock_cap']!r} is not a number between 0 and 1, so no single-shock limit applies")
        if trader.max_gross_bad(v):         # the same for the gross term: a malformed one keeps the 2x limit
            log(f"{k}: max_gross {v['max_gross']!r} is neither null nor a number above 0, so the 2x limit applies")
        if trader.max_calls_bad(v):         # and for the research-call limit: a malformed one keeps tools.MAX_CALLS
            log(f"{k}: max_calls {v['max_calls']!r} is not a whole number from 1 to {tools.MAX_CALLS_TOP}, "
                f"so the limit is {tools.MAX_CALLS} research calls")
        for msg in switch_bad(v):           # and for the forced-submit settings: a malformed one keeps the module's default
            log(f"{k}: {msg}")
        if trader.own_limit_bad(v):         # and for the own exposure limit (18 Nov): an ignored one is said at every look
            log(f"{k}: {trader.own_limit_bad(v)}")
    no_tool = ml_tools.unknown(sleeves)
    if no_tool:                             # never another version's tool in its place
        log(f"NO LOOK for {', '.join(no_tool)}: ml_tool not in ml_tools.VERSIONS {sorted(ml_tools.VERSIONS)}")
        sleeves = {k: v for k, v in sleeves.items() if k not in no_tool}
    data = trader.Data(last=(today - timedelta(days=1)).strftime("%Y%m%d"))
    data.next_date = today.isoformat()
    d = len(data.dates)
    close = close_time(today)
    session = dtime(9, 30) <= now.time() < close
    day = f"Date: {today.isoformat()} ({today.strftime('%A')}), {now:%H:%M} ET."
    header = (f"{day} The market is open until {close:%H:%M} ET; prices are delayed about 15 minutes. Orders you submit now work immediately."
              if session else f"{day} Before the open. Orders you submit now work from the 09:30 open.")
    claude_jobs, gptoss_jobs, claude_out = [], [], {}
    try:
        rate_line = rates.line(now)                 # the same facts for every sleeve at this look
    except Exception:                               # noqa: BLE001 - a look never fails on it
        rate_line = ""
        log("rate line failed: " + traceback.format_exc()[-300:])

    def run_claude(lane):                   # week 4's four Claude sleeves run side by side, each lane's tool server on its
        start = time.time()                 # own client, so each decides on a fresh state. WINDOW caps each CLI; placing
        for name, meta, text, st, system, rows, sp in claude_jobs[lane::len(MCP_CLIENTS)]:   # also waits for the gpt-oss chain
            t0 = time.time()
            try:
                call, trace = agent.claude_look(meta["model"], system, text, sp, LIVE / today.isoformat() / f"{name}-{now:%H%M}",
                                                effort(meta), timeout=max(60, min(agent.CLAUDE_TIMEOUT,
                                                                                  start + WINDOW.total_seconds() - t0)),
                                                client=MCP_CLIENTS[lane], ml_tool=meta.get("ml_tool"),
                                                max_gross=trader.max_gross_of(meta), sleeve_value=st["value"],
                                                text=trader.text_of(meta), check_ctx=check_ctxs.get(name) if claude_checks else None,
                                                max_calls=trader.max_calls_of(meta))
            except Exception:
                call, trace = None, [{"error": traceback.format_exc()[-1500:]}]
            claude_out[name] = call, trace, time.time() - t0

    workers = [threading.Thread(target=run_claude, args=(lane,), daemon=True) for lane in range(len(MCP_CLIENTS))]
    if session and now.time() < dtime(9, 31):   # the opening auction's fills are reported 09:30:05-09:30:20. On 6 Nov
        wait = (datetime.combine(today, dtime(9, 30, 45), tzinfo=ET) - datetime.now(ET)).total_seconds()   # base-ml's state
        if wait > 0:                            # was built at 09:30:11, its AMD stop filled at 09:30:19, and it bought 20 to
            time.sleep(wait)                    # cover a short already closed. A fact's timing, not a delay of a decision.
    check_ctxs, claude_checks = {}, False      # text B's check_orders context by sleeve; the Claude path's switch
    with gateway() as gw, contextlib.ExitStack() as whatif_hold:   # the what-if connection closes before client 10's
        co_rows = gw.orders() if co.V2 and not dry_run else None   # v2.1: the broker's orders before the look's sync
        paper.sync_fills(gw)
        reconcile(gw)
        if not dry_run:                     # close_only orders the position no longer covers (Clayton, 17 Nov): cancelled
            co_all(gw, sleeves, rows=co_rows)   # now, newest first, so this look's state shows it and the note says why
        specs = tools.available(gw)
        live, qqq_quote = None, None
        if session:
            held = set()
            for name in (paper.sleeves() if only else sleeves):   # every sleeve's names, at a quarter look too: the account
                held |= set(paper.book(name)["positions"])        # line totals every sleeve (review, 20 Nov)
            rows, _ = data.table_rows(d, extra=sorted(held))
            syms = sorted(set(rows) | held)                 # held names outside the table too, else they show at cost;
            live = {}                                       # at most 90 snapshots at a time (IBKR's 100 market-data lines)
            for i in range(0, len(syms), 90):
                for q in gw.quotes(syms[i:i + 90]):
                    m = None if q.get("error") else paper.mark(q)
                    if m is not None:
                        live[q["symbol"]] = m
                    if q.get("symbol") == "QQQ" and not q.get("error"):
                        qqq_quote = q               # for the time of its last print (the regime line)
        regime_text = ""                    # market facts for sleeves with "regime_line" (candidate 2): the same line for
        if any(m.get("regime_line") for m in sleeves.values()):     # each of them at this look
            try:                            # a failure costs the line, never the look
                regime_text = regime.facts(gw, now, session, data.features(d)[0]["QQQ"]["close"], data.iso(d - 1),
                                           qqq_quote, log)
                if not rate_line:           # candidate 2's rows always have the Rates line before it
                    log("regime line: shown with no Rates line before it")
            except Exception:               # noqa: BLE001
                log("regime line left out: " + traceback.format_exc()[-300:])
        # Text B (Clayton, 14 Nov; calculator_tools_spec_v2.md): check_orders on one what-if connection (client 17) shared
        # by the gpt-oss threads, and the whole-account line from one account() read before the looks, so every sleeve
        # sees the same figures. Text A (the control) gets neither. A failure costs the feature and logs; the looks go on.
        link, acct_info, n_all = None, None, len(paper.sleeves())
        if any(trader.text_of(m) == "B" for m in sleeves.values()):
            link, claude_checks = open_whatif()
            if link is not None:
                whatif_hold.callback(link.gw.close)
            acct_link = link if link is not None else checks.Link(gw)   # account() and positions() on one connection
            acct_info = {"account": read_account(acct_link, gw), "sleeves": n_all}
            if acct_info["account"] is not None:    # no line figures, no clause; a failure leaves the clause out
                acct_info["unheld"] = unheld_gross(acct_link, data, d, live, gw)   # books the fills since the look's
            acct_info["total"] = sleeves_total(data, d, live)   # sync first (24 Nov), so the total counts them too

        def build(name, meta, orders=None):
            world = trader.PaperWorld(gw if orders is None else OrdersAt(gw, orders), name, data)   # catch_up's orders
            world.dry_run, world.log = dry_run, log
            world.OWN_LIMIT = trader.own_limit_of(meta)     # the look-start settings, as the system text, check and record
            world.RANGE_LINE = range_line_of(meta)          # Week 7: the range line, sleeves.json "range_line"
            if trader.text_of(meta) == "B":
                world.ACCOUNT = acct_info
            text, st, feats = trader.state_text(world, data, d, header=header, live=live, book=bool(meta.get("risk_line")),
                                                cash_clause=trader.text_of(meta) == "B")   # Clayton, 14 Nov: text B only
            if trader.text_of(meta) == "B" and (link is not None or claude_checks):
                try:                        # a failure costs this sleeve's check tool, never the look
                    check_ctxs[name] = checks.context(st, feats, live, n_all, len(sleeves), sleeve=name,
                                                      own_limit=trader.look_limit(trader.own_limit_of(meta)),
                                                      marks=trader.own_limit_of(meta) is not None)
                except Exception:           # noqa: BLE001
                    log(f"{name}: check_orders off for this look: " + traceback.format_exc()[-200:])
            ml = bool(meta.get("ml"))           # the ML arm: the same model plus the forecast tool (forecast.py)
            sp = specs + ([ml_tools.module(meta.get("ml_tool")).SPEC] if ml else [])
            # no today: the models aren't told of the Monday close; NAV counts cash and holdings (Clayton, 26 Oct)
            goal = trader.goal_line(meta["start_cash"], st["value"], sessions_left(today))
            goal += f"\n{rate_line}" if rate_line else ""
            if meta.get("regime_line") and regime_text:     # directly after the Rates line
                goal += f"\n{regime_text}"
            text = trader.live_user(text, goal, load_memo(name), steps=trader.live_steps(ml, probs=trader.text_of(meta) == "A"))
            rows, _ = data.table_rows(d, extra=world.symbols_in_play())
            system = trader.live_system(meta["start_cash"], today, ml=ml, ml_tool=meta.get("ml_tool"),
                                        kelly=bool(meta.get("kelly_prompt")), max_gross=trader.max_gross_of(meta),
                                        own_limit=trader.own_limit_of(meta), cadence=cadence_of(meta),
                                        close=close_time(today) if trader.text_of(meta) == "B" else None,
                                        next_look=next_look_of(meta))
            world.gw = gw                   # orders() snapshot for the state only; act() and the rest on client 10 itself
            return world, text, st, system, rows, sp

        def catch_up(name):
            """catch_up_sleeve for this look: its client 10, its marks (live) and its dry_run."""
            return catch_up_sleeve(gw, name, live, dry_run)

        for name, meta in sleeves.items():
            if meta["model"].startswith("claude-"):
                try:                        # an error costs this sleeve's look only
                    snap = catch_up(name)   # 24 Nov: the Claude sleeves' states are built here, before the threads
                    world, text, st, system, rows, sp = build(name, meta, snap)
                except Exception:
                    log(f"{name}: no look: " + traceback.format_exc()[-300:])
                    continue
                claude_jobs.append((name, meta, text, st, system, rows, sp))
            else:
                gptoss_jobs.append((name, meta))
        for w in workers:                   # the Claude looks run meanwhile; their orders are placed below, on client 10
            w.start()
        # The gpt-oss sleeves run side by side, one thread each (GPTOSS_LANES). They share client 10's connection (the tws
        # Gateway serializes each call itself) and the ledgers, which are read and rewritten whole (looks.json, notes.json,
        # memo.json) or appended (fills.jsonl, orders.jsonl). `ledger` is held to build a state, to place and to record.
        # `research` is held for each research call: it guards the tool modules' caches (tools, forecast, livenews,
        # events) and writes no ledger, so a slow call (an SEC retry can wait about 20 minutes) delays only other
        # research, never a placement. The sampling overlaps. Order: never take `ledger` while holding `research`.
        ledger, research = threading.RLock(), threading.Lock()
        service = None
        if gptoss_jobs:
            try:                            # one Tinker session per look, shared, as before: each ServiceClient keeps a
                trader.tokenizer()          # session (heartbeat, pools) that is never closed. The tokenizer is loaded
            except Exception:               # once, before the threads share it; a failure costs only the gpt-oss looks
                log("tokenizer load failed: " + traceback.format_exc()[-300:])
            service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": "live"})

        def run_gptoss(k, name, meta):
            try:                            # an error here costs this sleeve's look only: the rest go on, and the Claude
                with ledger:                # orders below are still placed. Each gpt-oss sleeve sees its book as of its
                    snap = catch_up(name)   # own state (at 09:30 the opening fills arrived after the state was built and
                    world, text, st, system, rows, sp = build(name, meta, snap)   # 7 orders were sent again); 24 Nov:
                                            # the first sleeve too (k 0), not only those after it
            except Exception:
                log(f"{name}: no look: " + traceback.format_exc()[-300:])
                return
            model, t0 = meta["model"], time.time()
            call, trace, prefill, sampled, notes = None, [], 0, 0, []
            submit_tool = trader.TOOL_LIVE_PROBS if trader.text_of(meta) == "A" else trader.TOOL_LIVE
            offered = None                  # the tools this look offers, for the record's words hash (20 Nov)
            try:                            # an error ends this sleeve's look only; the rest and the Claude orders go on
                sampler = lambda svc, m=model: (svc.create_sampling_client(base_model=base_model_of(meta)) if m in ("base", BASE)
                                                else svc.create_sampling_client(model_path=m))
                renew = lambda: sampler(tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": "live"}))
                ctx = check_ctxs.get(name) if link is not None else None
                checker = (lambda args, c=ctx: checks.check(link, args, c, log=lambda m: log(f"{name}: {m}"))) if ctx else None
                if checker is not None:     # 18 Nov: agent._estimate allows for the check's exposure-limit line
                    checker.own_limit = ctx.get("own_limit")
                offered = sp + ([checks.CHECK_SPEC] if checker else []) + [submit_tool]
                if inkling.is_inkling(model):   # Inkling-Small through Tinker's OpenAI-compatible endpoint (20 Nov)
                    call, trace, prefill, sampled = inkling.look(
                        gw, system, text, sp, effort=effort(meta), ml_tool=meta.get("ml_tool"), lock=research,
                        submit_tool=(trader.TOOL_LIVE_PROBS if trader.text_of(meta) == "A" else trader.TOOL_LIVE),
                        sleeve_value=st["value"], look_key=(name, now.isoformat()), checker=checker,
                        final_line=final_line_of(meta), max_calls=trader.max_calls_of(meta), model=model,
                        context=inkling.context_of(meta))
                else:
                    call, trace, prefill, sampled = agent.gptoss_look(sampler(service), gw, system, text, today.isoformat(), sp,
                                                                      effort(meta), renew=renew, ml_tool=meta.get("ml_tool"),
                                                                      lock=research, sleeve_value=st["value"], look=(name, now.isoformat()),
                                                                      submit_tool=(trader.TOOL_LIVE_PROBS if trader.text_of(meta) == "A"
                                                                                   else trader.TOOL_LIVE), checker=checker,
                                                                      max_calls=trader.max_calls_of(meta),
                                                                      forced_mode=forced_mode_of(meta), final_line=final_line_of(meta),
                                                                      context=context_of(meta))
                if call and not dry_run:
                    with ledger:
                        notes = world.act(call)
            except Exception as e:
                if hasattr(e, "inkling_trace"):     # an Inkling look that raised keeps what it had done and paid for
                    trace, (prefill, sampled) = list(e.inkling_trace), e.inkling_cost
                trace = trace + [{"error": traceback.format_exc()[-1500:]}]
            cost = {"prefill_tokens": prefill, "sampled_tokens": sampled}
            if inkling.is_inkling(model):   # the endpoint's own count of cached prompt tokens (the Researcher, 20 Nov)
                cost["cached_tokens"] = next((e["inkling"].get("cached_tokens") for e in reversed(trace)
                                              if isinstance(e, dict) and isinstance(e.get("inkling"), dict)), None)
            with ledger:
                finish(now, name, meta, text, st, call, trace, notes, rows, dry_run, cost=cost, secs=time.time() - t0,
                       words=(system, offered))

        if gptoss_jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(GPTOSS_LANES, len(gptoss_jobs))),
                                                       thread_name_prefix="gptoss") as pool:
                futs = [pool.submit(run_gptoss, k, name, meta) for k, (name, meta) in enumerate(gptoss_jobs)]
                _CO_GUARD["tick"] = datetime.now(ET)    # the first check a minute in, after the sleeves built their states
                while co.V2 and not dry_run:        # close_only v2.1: the check about once a minute while the sleeves run
                    if not concurrent.futures.wait(futs, timeout=20).not_done:
                        break
                    close_only_tick(gw, ledger, datetime.now(ET), dry_run)
                for (name, _), f in zip(gptoss_jobs, futs):
                    try:
                        f.result()
                    except Exception:       # noqa: BLE001 - run_gptoss catches its own; this is the last net
                        log(f"{name}: gpt-oss look thread failed: " + traceback.format_exc()[-300:])
    for w in workers:
        while co.V2 and not dry_run and w.is_alive():   # v2.1: the Claude lanes still run; client 10 is free now
            w.join(timeout=20)
            if w.is_alive():
                close_only_guard(datetime.now(ET))     # its own connection, once a minute in the session (never raises)
        w.join()
    for name, meta, text, st, system, rows, sp in claude_jobs:
        call, trace, secs = claude_out.get(name) or (None, [{"error": "the Claude look did not finish"}], None)
        notes = []
        try:
            with gateway() as gw:
                world = trader.PaperWorld(gw, name, data)
                world.dry_run, world.log = dry_run, log
                notes = world.act(call) if call and not dry_run else []
        except Exception:
            trace = trace + [{"error": "placing: " + traceback.format_exc()[-1500:]}]
        finish(now, name, meta, text, st, call, trace, notes, rows, dry_run, secs=secs)
    if not dry_run:
        try:                                # record this look's fills now, not at the next look
            with gateway() as gw:
                time.sleep(5)
                co_rows = gw.orders() if co.V2 else None    # v2.1: the broker's orders before the sync
                paper.sync_fills(gw)
                reconcile(gw)
                co_all(gw, sleeves, rows=co_rows)   # and after the look's own fills (close_only, 17 Nov)
                account_margin(gw)
        except Exception:
            log("post-look fill sync failed: " + traceback.format_exc()[-300:])


def record():
    today = datetime.now(ET).date().isoformat()
    with gateway() as gw:
        paper.sync_fills(gw)
        reconcile(gw)
        book_dividends(gw)                            # today's ex-dates are in today's bars by now
        acct = gw.account()
        rows = []
        for name, meta in paper.sleeves().items():
            b0 = paper.book(name)
            syms = sorted(set(b0["positions"]) | {"SPY"})
            quotes = {q["symbol"]: paper.mark(q) for q in gw.quotes(syms) if not q.get("error")}
            b = paper.book(name, quotes)
            rows.append({"date": today, "sleeve": name, "model": b["model"], "nav": b["nav"], "cash": b["cash"],
                         "start_cash": b["start_cash"], "realized": b["realized"], "unrealized": b["unrealized"],
                         "commissions": b["commissions"], "fills": b["fills"], "missing_commissions": b["missing_commissions"],
                         "positions": {s: p["quantity"] for s, p in b["positions"].items()},
                         "marks": {s: p["mark"] for s, p in b["positions"].items()},   # the price each name's value
                         # in nav used (None: unmarked, nav None), so a day split reconciles to the cent (the Researcher, 20 Nov)
                         "spy": quotes.get("SPY"),
                         "account_net_liquidation": acct.get("NetLiquidation"),
                         "ml_tool": ml_tools.version_of(meta)})
    if any(m.get("ml") for m in paper.sleeves().values()):     # today's 30-minute bars for the ML arm's intraday models,
        import fetch_extra                                     # with the 4 sessions before: IBKR revises its history
        stats = {}                                             # overnight, and a session a failed fetch missed is filled
        with gateway() as gw:                                  # (fetch_extra.merge_bars30; the Researcher's D-5, 21 Nov)
            bad = fetch_extra.update(gw, sorted(set(trader.universe())), "bars30", duration="5 D", stats=stats)
        log(fetch_extra.bars30_summary(stats, bad, datetime.now(ET).strftime("%Y%m%d")))
    for r in rows:
        paper.append(LIVE / "daily.jsonl", r)
        log(f"{r['sleeve']}: nav {r['nav']}, realized {r['realized']:.2f}, commissions {r['commissions']:.2f}, "
            f"positions {r['positions']}")
    macro_rebuild()                                   # the forecast macro line and every sleeve's tape read it


def macro_rebuild():
    """Rebuild data/macro/calendar.json (ml_research/macro_fetch.py, a few minutes) after the record, so the next
    day's forecast macro line and tape have today's first-release values and any schedule change, and keep tonight's
    scheduled rows as data/macro/schedule/<YYYYMMDD>.json (the schedule as known then; it can't be rebuilt later).
    Logs and never raises: the old calendar stays if the build fails."""
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(ROOT / "ml_research" / "macro_fetch.py")], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=80 * 60)   # up to 35 of them waiting on build.lock
        if r.returncode:
            raise RuntimeError(f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
        cal = json.loads((ROOT / "data" / "macro" / "calendar.json").read_text(encoding="utf-8"))
        sched = [e for e in cal["events"] if e.get("status") == "scheduled"]
        out = ROOT / "data" / "macro" / "schedule" / f"{datetime.now(ET):%Y%m%d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"built_et": cal["meta"].get("built_et"), "events": sched}, indent=1), encoding="utf-8")
        log(f"macro calendar rebuilt in {time.time() - t0:.0f}s: {len(cal['events'])} events, {len(sched)} scheduled "
            f"saved to {out.name}")
    except Exception as e:                              # noqa: BLE001
        log(f"macro calendar rebuild failed after {time.time() - t0:.0f}s: {type(e).__name__}: {str(e)[:300]}")


MACRO_CAL = ROOT / "data" / "macro" / "calendar.json"
FRED_KEYS = {"cpi", "nfp", "gdp", "pce", "ppi", "retail", "jolts", "claims", "eci"}   # tier 1-2 releases valued from FRED


def macro_missing(now, cal=MACRO_CAL):
    """Today's FRED-valued releases at least 40 minutes past their time that calendar.json still has no value for."""
    try:
        ev = json.loads(cal.read_text(encoding="utf-8"))["events"]
    except (OSError, ValueError, KeyError):
        return []
    day = now.date().isoformat()
    out = set()
    for e in ev:
        try:
            if (e.get("date") == day and e.get("key") in FRED_KEYS and not e.get("first_release")
                    and now >= datetime.fromisoformat(e["datetime_et"]).astimezone(ET) + timedelta(minutes=40)):
                out.add(e["key"])
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def macro_busy(lock=ROOT / "data" / "macro" / "build.lock"):
    """Whether a macro build (ml_research/macro_fetch.build_lock) holds its lock now."""
    if not lock.exists():
        return False
    try:
        with open(lock, "a+") as f:
            f.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def macro_intraday(st, now=None, popen=subprocess.Popen):
    """ML2.2: on a release day, rebuild the calendar in the background once FRED has had time to post (from 40 minutes
    after the release, at most every 30 minutes, 6 tries, none started after 15:00 ET or while another build runs), so
    the forecast's macro line shows today's values that day, as the simulator does (value_delay_min=60). Before, a
    release's values reached the calendar only in an evening build. Never blocks a look and never raises; the evening
    rebuild is unchanged."""
    try:
        now = now or datetime.now(ET)
        p = st.get("proc")
        if p is not None and p.poll() is None:           # running: killed after 30 minutes, whatever the day
            if time.time() - st["t0"] > 30 * 60 and not st.get("killed"):
                p.kill()
                st["killed"] = True
                log("intraday macro rebuild killed after 30 minutes")
            return
        if p is not None:
            st["proc"] = None
            st["out"].close()
            still = macro_missing(now)
            log(f"intraday macro rebuild {st['n']} ended (exit {p.returncode}, {time.time() - st['t0']:.0f}s): "
                + (f"still no value for {', '.join(still)}" if still else "today's released values are in"))
        if st.get("day") != now.date():
            st.clear()
            st.update(day=now.date(), n=0, proc=None, last=None)
        if st["n"] >= 6 or now.time() >= dtime(15, 0) or (st["last"] and now - st["last"] < timedelta(minutes=30)):
            return
        keys = macro_missing(now)
        if not keys or macro_busy():
            return
        st.update(n=st["n"] + 1, last=now, t0=time.time(), killed=False)
        out = open(LIVE / "macro_intraday.log", "a", encoding="utf-8")
        try:
            out.write(f"=== {now:%Y-%m-%d %H:%M:%S} ET, try {st['n']}, for {', '.join(keys)}\n")
            out.flush()
            proc = popen([sys.executable, str(ROOT / "ml_research" / "macro_fetch.py")], cwd=ROOT, stdout=out,
                         stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            out.close()
            raise
        st.update(proc=proc, out=out)
        log(f"intraday macro rebuild {st['n']} started for {', '.join(keys)}")
    except Exception as e:                              # noqa: BLE001
        try:
            log(f"intraday macro rebuild error: {type(e).__name__}: {str(e)[:200]}")
        except Exception:                               # noqa: BLE001
            pass


def run_job(name, fn, *a):
    try:
        fn(*a)
        return True
    except Exception:
        log(f"{name} FAILED:\n{traceback.format_exc()}")
        return False


DONE = LIVE / "done.json"   # jobs already run, so a restart never repeats a look


def load_done():
    return {tuple(x) for x in paper._read_json(DONE, [])}


def mark_done(done, key):
    # paper._write_json retries a refused write. Mon 23 Nov 2026: a file another process memory-maps (git hashing a
    # tracked, changed ledger) refuses an in-place write with OSError [Errno 22]; here that would stop the loop.
    done.add(key)
    try:
        paper._write_json(DONE, sorted([list(k) for k in done])[-400:])
    except OSError as e:                    # still refused after paper's retries: the key is kept in memory and goes
        log(f"done.json not written ({type(e).__name__}: {str(e)[:120]}); {key[1]} is kept in memory and reaches "
            f"the file with the next job")  # to disk with the next job; never stop the loop for it


def code_line(run=subprocess.run):
    """The code this loop runs, so a loop left on old code shows in live.log: git describe (with "-dirty" when the tree
    has uncommitted changes) and each registered ML tool's own VERSION."""
    try:
        r = run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        head = (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:                                   # noqa: BLE001
        head = ""
    vs = []
    for name in sorted(ml_tools.VERSIONS):
        try:
            vs.append(f"{name} {ml_tools.module(name).VERSION}")
        except Exception as e:                          # noqa: BLE001
            vs.append(f"{name} unavailable ({type(e).__name__})")
    return f"code {head or 'unknown'}; ML tools: {', '.join(vs)}"


_CO_GUARD = {}


def close_only_guard(now):
    """Close-only orders between looks (Clayton, 17 Nov: "a stop it means as protective can never open a new position";
    the review that day: checked only when sent and at each look, a close_only stop could still fill as an opening order
    after another order closed the position between looks). During the session, about once a minute, and only while some
    sleeve placed a close_only order in the last 10 days: sync client 10's fills and cancel the close_only orders each such
    sleeve's position no longer covers (paper.close_only_excess). It runs in the loop's own thread between jobs, so never
    during a look. A failure is logged once an hour and costs nothing else."""
    if not (dtime(9, 30) <= now.time() < close_time(now.date())):
        return
    last = _CO_GUARD.get("last")
    if last is not None and (now - last).total_seconds() < 60:
        return
    _CO_GUARD["last"] = now
    try:
        names = [n for n in paper.close_only_recent() if n in paper.sleeves()]
        if not names:
            return
        with gateway() as gw:
            if co.V2:                                 # v2.1 (DESIGN C): the broker's orders first, then the fills
                co_all(gw, names, fail_log=lambda m: _co_fail(now, m))
            else:
                paper.sync_fills(gw)
                for name in names:
                    paper.close_only_excess(gw, name, log=log, fail_log=lambda m: _co_fail(now, m))
    except Exception:                                 # noqa: BLE001
        _co_fail(now, "close_only guard failed: " + traceback.format_exc()[-300:])


def close_only_tick(gw, lock, now, dry_run=False):
    """The close_only check while a look runs (v2.1, DESIGN C; critique "fills-races" #8: a take-profit that fills during a
    look left its full-size stop working until the look ended): about once a minute in the session, on the look's own
    client-10 connection, under the look's ledger lock (the sleeves' state builds and acts take it too, so a check never
    runs inside one). Off with close_only.V2 off, in a dry run, and outside the session. Never raises."""
    if dry_run or not co.V2 or not (dtime(9, 30) <= now.time() < close_time(now.date())):
        return
    last = _CO_GUARD.get("tick")
    if last is not None and (now - last).total_seconds() < 60:
        return
    _CO_GUARD["tick"] = now
    try:
        names = [n for n in paper.close_only_recent() if n in paper.sleeves()]
        if not names:
            return
        with lock:
            co_all(gw, names, fail_log=lambda m: _co_fail(now, m))
    except Exception:                                 # noqa: BLE001
        _co_fail(now, "close_only check during the look failed: " + traceback.format_exc()[-300:])


def co_all(gw, names, fail_log=None, rows=None):
    """The close_only check for each sleeve in names. With close_only.V2 on: the broker's orders read once, then the
    fills synced once (DESIGN C's order), then each sleeve; rows: the caller read the orders before its own fill sync
    (look start and after the look), so neither is read again. Nothing is released or grown while HOLD exists (an
    account reset may be under way). Off: each sleeve's check as on 17-19 Nov (paper.close_only_check runs
    close_only_excess)."""
    if not co.V2:
        for name in names:
            paper.close_only_check(gw, name, log=log)
        return
    if rows is None:
        rows = gw.orders()
        paper.sync_fills(gw)
    place_ok = not HOLD.exists()
    for name in names:
        paper.close_only_check(gw, name, log=log, fail_log=fail_log, rows=rows, synced=True, place_ok=place_ok)


def _co_fail(now, msg):
    """A failure of the close_only guard, logged at most once an hour (it runs about once a minute)."""
    said = _CO_GUARD.get("said")
    if said is None or (now - said).total_seconds() > 3600:
        _CO_GUARD["said"] = now
        log(msg + " (logged at most once an hour)")


def loop():
    log(f"live loop started: {code_line()}")
    for path in marker_files():                       # no look runs yet in this process: a marker whose process is dead, is
        try:                                          # this process (a reused pid) or names none is stale; one of another
            pid, stale = marker_pid(path)             # live process (a dry run, another loop) is left alone
            if stale is None:                         # gone since the listing, or it can't be read
                if not path.exists():
                    continue
                stale = "not readable"
            if pid is not None and pid != os.getpid() and pid_alive(pid):
                log(f"LOOKING left in place: process {pid} is alive and looking ({path.name}: {stale[:80]})")
            elif pid is None and time.time() - path.stat().st_mtime < 60:      # a begin_marker may be writing it now
                log(f"LOOKING left in place: it names no pid but was written under a minute ago ({path.name}: {stale[:80]})")
            else:
                path.unlink(missing_ok=True)
                log(f"stale LOOKING marker removed ({stale[:80]}; {path.name})")
        except Exception:                             # noqa: BLE001
            log("stale LOOKING marker not removed: " + traceback.format_exc()[-200:])
    try:        # which loop is running, for scripts that write the ledger beside it (reopen_week6.py --orders; the TWS
        LOOP.write_text(json.dumps({"pid": os.getpid(), "code": code_line(), "looking": 2,     # Gateway Owner, 15 Nov):
                                    "started": datetime.now(ET).isoformat(timespec="seconds")}), encoding="utf-8")
    except Exception:                                 # "looking": 2 says this loop writes LOOKING.<pid> around every look
        log("LOOP.json not written: " + traceback.format_exc()[-200:])
    done = load_done()
    tried = {}                                        # refresh/record: last attempt, for retries
    held = set()
    mac = {}                                          # the intraday macro rebuild's state (ML2.2)
    sleeves_said = {}                                 # when an unreadable sleeves.json was last logged
    while not STOP.exists():
        now = datetime.now(ET)
        if trading_day(now.date()):
            for name, (h, m) in JOBS:
                key = (now.date().isoformat(), name)
                if key in done or (name.startswith("look-") and dtime(h, m) >= close_time(now.date())):
                    continue
                due = now.replace(hour=h, minute=m, second=0, microsecond=0)
                quarter = name.startswith("look-") and m in (15, 45)
                if name.startswith("look-") and datetime.combine(now.date(), close_time(now.date()), tzinfo=ET) - now < LAST_START:
                    continue                      # too near the close for a look to place its orders in the session
                if due <= now < due + (timedelta(hours=2) if name in ("refresh", "record") else QUARTER_WINDOW if quarter else WINDOW):
                    qs = None
                    if quarter:                   # read once, only when the look is due: the same list is checked and run
                        sl = _sleeves_or_none(sleeves_said)
                        qs = quarter_sleeves(sl) if sl is not None else None
                        if not qs:                # no 15-minute sleeve (or no readable file): nothing runs, never
                            continue              # look(only=[]), which is every sleeve
                    if name in ("refresh", "record"):     # no orders: retry every 5 minutes until it succeeds
                        if now - tried.get(key, now - timedelta(hours=1)) < timedelta(minutes=5):
                            continue
                        tried[key] = now
                        log(f"{name} starting")
                        if run_job(name, refresh if name == "refresh" else record):
                            mark_done(done, key)
                        continue
                    if HOLD.exists():             # e.g. an account reset in progress: the look runs if HOLD goes within its window
                        if key not in held:
                            held.add(key)
                            log(f"{name} held: {HOLD} exists; no look until it is removed")
                        continue
                    log(f"{name} starting")
                    begin_marker()                # LOOKING.<pid> first (16 Nov 2026, the TWS Gateway Owner's review): a
                    try:                          # script waiting for "not pending and no LOOKING" never finds the gap
                        mark_done(done, key)      # marked before running: a crash mid-look must not re-place orders
                        run_job(name, look, False, qs, True)     # the look keeps the marker and removes it at its end
                    finally:                      # and if it never ran (mark_done or run_job failing), it goes here
                        end_marker()
                    break                         # a look took time: the next job is judged on a fresh clock (review, 20 Nov)
            macro_intraday(mac)
            compute_retry()                           # a version whose morning compute failed, again, alone (never raises)
            close_only_guard(now)                     # close_only orders between looks, about once a minute (never raises)
        warm_poll()                                   # logs the outside-name warm-up's summary once it ends
        try:                                          # no look runs here: this process's own marker, if end_marker could
            if own_marker().exists():                 # not remove it (another handle held it open, on Windows), goes now;
                end_marker()                          # else it would count as a running look until the next look ends
        except Exception:                             # noqa: BLE001
            pass
        time.sleep(20)
    log("STOP file found; live loop exiting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", action="store_true", help="one look now")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--job", choices=["refresh", "record", "macro"])
    a = ap.parse_args()
    if a.job:
        {"refresh": refresh, "record": record, "macro": macro_rebuild}[a.job]()
    elif a.look:
        look(a.dry_run, a.only)
    else:
        loop()


if __name__ == "__main__":
    main()
