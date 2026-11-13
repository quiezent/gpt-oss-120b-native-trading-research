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
import json
import os
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
import dividends
import ml_tools
import paper
import rates
import tools
import trader

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "runs" / "live"
STOP = LIVE / "STOP"
HOLD = LIVE / "HOLD"   # while it exists, looks wait (not marked done) and place nothing; refresh/record run
ET = ZoneInfo("America/New_York")
MCP_CLIENTS = ("trader-mcp-a", "trader-mcp-b", "trader-mcp-c", "trader-mcp-d")   # research clients 12-15, one Claude look on each
HOLD_MODEL = "hold"   # sleeves.json model of a sleeve waiting for an approved model: it gets no looks and stays in cash
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
            "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}
HALF_DAYS = {"2026-11-27", "2026-12-24", "2027-11-26"}   # NYSE closes at 13:00 ET
LOOK_TIMES = [(h, m) for h in range(9, 16) for m in (0, 30)]   # every 30 minutes, 09:00-15:30 (Clayton, 16 Oct)
JOBS = [("refresh", (7, 30))] +[(f"look-{h:02d}{m:02d}", (h, m)) for h, m in LOOK_TIMES] + [("record", (16, 15))]
WINDOW = timedelta(minutes=25)      # a look that can't start within 25 minutes of its time is skipped (looks are 30 minutes apart)
GPTOSS_LANES = 10                   # gpt-oss sleeves sampled side by side (Clayton, 11 Nov 2026: "they need to be [parallel]
                                    # because trading is time sensitive"; one after another, the 09:00 look's last sleeve
                                    # decided 21 minutes after its first)


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
    return diff


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


def refresh():
    syms = trader.universe()
    held = set()
    for name in paper.sleeves():
        held |= set(paper.book(name)["positions"])
    bad = []
    with gateway() as gw:
        for s in sorted(set(syms) | held):
            try:
                barsmod.refresh(gw, s)
            except Exception as e:
                bad.append(f"{s}: {e}")
    log(f"refreshed bars for {len(set(syms) | held)} symbols; {len(bad)} failed {bad[:5]}")
    book_dividends()                                    # a missed record's dividends, before the first look
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


def compute_versions():
    """Each version in use writes its own cache. ML2's failure raises, so the refresh retries as it always has; another
    version's is logged, so it never costs ML2's sleeves their forecasts. An unregistered version has no look anyway."""
    for v in ml_tools.in_use(paper.sleeves()):
        if v not in ml_tools.VERSIONS:
            log(f"no forecasts for {v}: not in ml_tools.VERSIONS")
            continue
        try:
            as_of, fc = ml_tools.module(v).compute()
            log(f"forecasts for the ML arm ({ml_tools.module(v).VERSION}): as of {as_of}, {len(fc)} symbols")
        except Exception:                               # noqa: BLE001
            if v == ml_tools.DEFAULT:
                raise
            log(f"{v} forecasts failed: " + traceback.format_exc()[-300:])


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


def research_summary(trace, call, table_rows):
    calls = [t for t in trace if t.get("tool") and t["tool"] != "submit"]
    used = {}
    researched = set()
    for t in calls:
        used[t["tool"]] = used.get(t["tool"], 0) + 1
        researched |= set(tools.symbols_in(t["tool"], t.get("args")))
    traded = sorted({str(o.get("symbol", "")).upper().strip() for o in (call or {}).get("place", []) if isinstance(o, dict)})
    return {"research_calls": len(calls), "tools": used, "researched": sorted(researched), "traded": traded,
            "traded_researched": sorted(set(traded) & researched), "traded_not_in_table": sorted(set(traded) - set(table_rows))}


MEMO = LIVE / "memo.json"   # each sleeve's own notes from its last look (submit's `notes`), shown at its next look


def load_memo(name):
    return (json.loads(MEMO.read_text(encoding="utf-8")) if MEMO.exists() else {}).get(name, {})


NOTES_CAP = {"claude": 8000, "gpt-oss": 2000}   # characters kept for the next look. gpt-oss stays at the 2,000 it was
                                                # trained with; Opus, told its cap, was cut silently at 2,000 in week 2
                                                # (118 memos, a median 28% lost)


def save_memo(name, now, call, model=""):
    if not call:                      # no valid decision: the previous notes stay
        return
    cap = NOTES_CAP["claude" if str(model).startswith("claude-") else "gpt-oss"]
    d = json.loads(MEMO.read_text(encoding="utf-8")) if MEMO.exists() else {}
    d[name] = {"when": f"{now:%a %d %b %H:%M} ET", "notes": str(call.get("notes") or "")[:cap]}
    MEMO.write_text(json.dumps(d, indent=1), encoding="utf-8")


def effort(meta):
    """The sleeve's reasoning effort: each model's maximum unless sleeves.json says otherwise (Clayton, 18 Oct)."""
    return meta.get("effort") or ("max" if meta["model"].startswith("claude-") else "high")


def finish(now, name, *a, **k):
    try:                                # a failed record costs that record only: the chain goes on and the Claude orders are placed
        record_look(now, name, *a, **k)
    except Exception:                   # noqa: BLE001
        log(f"{name}: look record failed: " + traceback.format_exc()[-300:])


def record_look(now, name, meta, text, st, call, trace, notes, table_rows, dry_run, cost=None, secs=None):
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
           "shock_cap": trader.shock_cap_of(meta)}
    (out_dir / f"{name}-{now:%H%M}.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    if not dry_run:
        paper.append(LIVE / "research.jsonl", {"time": rec["time"], "sleeve": name, **summ,
                                               "orders": len(call["place"]) if call else None,
                                               "notes_chars": len(str((call or {}).get("notes") or ""))})
    log(f"{name}: {'DRY RUN ' if dry_run else ''}"
        + (f"{len(call['place'])} orders, {len(call['cancel'])} cancels" if call else "no valid decision")
        + f"; research {summ['research_calls']} {summ['tools'] or ''}; traded {summ['traded']}"
        + (f"; {secs:.0f}s" if secs is not None else "")
        + (f"; notes {notes}" if notes else ""))


def look(dry_run=False, only=None):
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
                                                client=MCP_CLIENTS[lane], ml_tool=meta.get("ml_tool"))
            except Exception:
                call, trace = None, [{"error": traceback.format_exc()[-1500:]}]
            claude_out[name] = call, trace, time.time() - t0

    workers = [threading.Thread(target=run_claude, args=(lane,), daemon=True) for lane in range(len(MCP_CLIENTS))]
    if session and now.time() < dtime(9, 31):   # the opening auction's fills are reported 09:30:05-09:30:20. On 6 Nov
        wait = (datetime.combine(today, dtime(9, 30, 45), tzinfo=ET) - datetime.now(ET)).total_seconds()   # base-ml's state
        if wait > 0:                            # was built at 09:30:11, its AMD stop filled at 09:30:19, and it bought 20 to
            time.sleep(wait)                    # cover a short already closed. A fact's timing, not a delay of a decision.
    with gateway() as gw:
        paper.sync_fills(gw)
        reconcile(gw)
        specs = tools.available(gw)
        live = None
        if session:
            held = set()
            for name in sleeves:
                held |= set(paper.book(name)["positions"])
            rows, _ = data.table_rows(d, extra=sorted(held))
            syms = sorted(set(rows) | held)                 # held names outside the table too, else they show at cost;
            live = {}                                       # at most 90 snapshots at a time (IBKR's 100 market-data lines)
            for i in range(0, len(syms), 90):
                for q in gw.quotes(syms[i:i + 90]):
                    m = None if q.get("error") else paper.mark(q)
                    if m is not None:
                        live[q["symbol"]] = m
        def build(name, meta):
            world = trader.PaperWorld(gw, name, data)
            world.dry_run = dry_run
            text, st, feats = trader.state_text(world, data, d, header=header, live=live, book=bool(meta.get("risk_line")))
            ml = bool(meta.get("ml"))           # the ML arm: the same model plus the forecast tool (forecast.py)
            sp = specs + ([ml_tools.module(meta.get("ml_tool")).SPEC] if ml else [])
            # no today: the models aren't told of the Monday close; NAV counts cash and holdings (Clayton, 26 Oct)
            goal = trader.goal_line(meta["start_cash"], st["value"], sessions_left(today))
            goal += f"\n{rate_line}" if rate_line else ""
            text = trader.live_user(text, goal, load_memo(name), steps=trader.live_steps(ml))
            rows, _ = data.table_rows(d, extra=world.symbols_in_play())
            system = trader.live_system(meta["start_cash"], today, ml=ml, kelly=bool(meta.get("kelly_prompt")))
            return world, text, st, system, rows, sp

        for name, meta in sleeves.items():
            if meta["model"].startswith("claude-"):
                try:                        # an error costs this sleeve's look only
                    world, text, st, system, rows, sp = build(name, meta)
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
                    if k:                   # own state (at 09:30 the opening fills arrived after the state was built and
                        paper.sync_fills(gw)    # 7 orders were sent again)
                        if live is not None:
                            missing = sorted(set(paper.book(name)["positions"]) - set(live))
                            for q in (gw.quotes(missing) if missing else []):
                                m = None if q.get("error") else paper.mark(q)
                                if m is not None:
                                    live[q["symbol"]] = m
                    world, text, st, system, rows, sp = build(name, meta)
            except Exception:
                log(f"{name}: no look: " + traceback.format_exc()[-300:])
                return
            model, t0 = meta["model"], time.time()
            call, trace, prefill, sampled, notes = None, [], 0, 0, []
            try:                            # an error ends this sleeve's look only; the rest and the Claude orders go on
                sampler = lambda svc, m=model: (svc.create_sampling_client(base_model=BASE) if m in ("base", BASE)
                                                else svc.create_sampling_client(model_path=m))
                renew = lambda: sampler(tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": "live"}))
                call, trace, prefill, sampled = agent.gptoss_look(sampler(service), gw, system, text, today.isoformat(), sp,
                                                                  effort(meta), renew=renew, ml_tool=meta.get("ml_tool"),
                                                                  lock=research)
                if call and not dry_run:
                    with ledger:
                        notes = world.act(call)
            except Exception:
                trace = trace + [{"error": traceback.format_exc()[-1500:]}]
            with ledger:
                finish(now, name, meta, text, st, call, trace, notes, rows, dry_run,
                       cost={"prefill_tokens": prefill, "sampled_tokens": sampled}, secs=time.time() - t0)

        if gptoss_jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(GPTOSS_LANES, len(gptoss_jobs))),
                                                       thread_name_prefix="gptoss") as pool:
                futs = [pool.submit(run_gptoss, k, name, meta) for k, (name, meta) in enumerate(gptoss_jobs)]
                for (name, _), f in zip(gptoss_jobs, futs):
                    try:
                        f.result()
                    except Exception:       # noqa: BLE001 - run_gptoss catches its own; this is the last net
                        log(f"{name}: gpt-oss look thread failed: " + traceback.format_exc()[-300:])
    for w in workers:
        w.join()
    for name, meta, text, st, system, rows, sp in claude_jobs:
        call, trace, secs = claude_out.get(name) or (None, [{"error": "the Claude look did not finish"}], None)
        notes = []
        try:
            with gateway() as gw:
                world = trader.PaperWorld(gw, name, data)
                world.dry_run = dry_run
                notes = world.act(call) if call and not dry_run else []
        except Exception:
            trace = trace + [{"error": "placing: " + traceback.format_exc()[-1500:]}]
        finish(now, name, meta, text, st, call, trace, notes, rows, dry_run, secs=secs)
    if not dry_run:
        try:                                # record this look's fills now, not at the next look
            with gateway() as gw:
                time.sleep(5)
                paper.sync_fills(gw)
                reconcile(gw)
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
                         "positions": {s: p["quantity"] for s, p in b["positions"].items()}, "spy": quotes.get("SPY"),
                         "account_net_liquidation": acct.get("NetLiquidation"),
                         "ml_tool": ml_tools.version_of(meta)})
    if any(m.get("ml") for m in paper.sleeves().values()):     # today's 30-minute bars for the ML arm's intraday models
        import fetch_extra
        with gateway() as gw:
            bad = fetch_extra.update(gw, sorted(set(trader.universe())), "bars30", duration="1 D")
        log(f"added today's 30-minute bars; {len(bad)} failed {bad[:3]}")
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
    return {tuple(x) for x in json.loads(DONE.read_text())} if DONE.exists() else set()


def mark_done(done, key):
    done.add(key)
    DONE.write_text(json.dumps(sorted([list(k) for k in done])[-400:]), encoding="utf-8")


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


def loop():
    log(f"live loop started: {code_line()}")
    done = load_done()
    tried = {}                                        # refresh/record: last attempt, for retries
    held = set()
    mac = {}                                          # the intraday macro rebuild's state (ML2.2)
    while not STOP.exists():
        now = datetime.now(ET)
        if trading_day(now.date()):
            for name, (h, m) in JOBS:
                key = (now.date().isoformat(), name)
                if key in done or (name.startswith("look-") and dtime(h, m) >= close_time(now.date())):
                    continue
                due = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if due <= now < due + (timedelta(hours=2) if name in ("refresh", "record") else WINDOW):
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
                    mark_done(done, key)          # marked before running: a crash mid-look must not re-place orders
                    run_job(name, look)
            macro_intraday(mac)
        warm_poll()                                   # logs the outside-name warm-up's summary once it ends
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
