"""Trade the paper account: every sleeve's model takes a look before the open and every hour of the session.

    python live.py                          # run until runs/live/STOP exists
    python live.py --look --dry-run         # one look now for every sleeve, place nothing
    python live.py --look --only opus       # one look now for one sleeve
    python live.py --job refresh|record|macro

Schedule (US Eastern, so DST is automatic): 07:30 refresh daily bars (it takes about 40 minutes); a fill sync 3 minutes
before the Gateway's day ends (midnight Kuala Lumpur: 12:00 ET, 11:00 ET in winter); looks at 09:00 (before the
open) and every 30 minutes until 15:30; 16:15 record fills and each sleeve's day. At each look a model sees its
sleeve and the market table (with delayed last prices during the session), may research with the
tools in tools.py, then submits orders, which are placed exactly as written. gpt-oss models run on
Tinker (agent.gptoss_look); Claude models run through the Claude CLI with only those tools
(agent.claude_look), in a background thread while the gpt-oss looks run (their research uses client 11; their orders
are placed on client 10 afterwards). Every sleeve reasons at its model's maximum effort (sleeves.json "effort").
Completed jobs are remembered (done.json), so a restart never repeats a look.
Every look is logged in runs/live/<date>/<sleeve>-<HHMM>.json and summarized in runs/live/research.jsonl.

Start the loop detached from any Claude session (a session restart kills processes it started; that lost
the 16 Oct 08:15-11:00 jobs), e.g. from PowerShell:
    Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CurrentDirectory='C:\\Claude\\Trader';
        CommandLine='cmd.exe /c ""C:\\Claude\\Trader\\.venv\\Scripts\\python.exe" live.py >> runs\\live\\live.out.log 2>> runs\\live\\live.err.log"'}
"""

import argparse
import collections
import json
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
import forecast
import paper
import tools
import trader

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "runs" / "live"
STOP = LIVE / "STOP"
HOLD = LIVE / "HOLD"   # while it exists, looks wait (not marked done) and place nothing; refresh/record run
ET = ZoneInfo("America/New_York")
MYT = ZoneInfo("Asia/Kuala_Lumpur")   # the Gateway's login time zone: its day ends at midnight here
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
            "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}
HALF_DAYS = {"2026-11-27", "2026-12-24", "2027-11-26"}   # NYSE closes at 13:00 ET
LOOK_TIMES = [(h, m) for h in range(9, 16) for m in (0, 30)]   # every 30 minutes, 09:00-15:30 (Clayton, 16 Oct)
JOBS = [("refresh", (7, 30))] +[(f"look-{h:02d}{m:02d}", (h, m)) for h, m in LOOK_TIMES] + [("record", (16, 15))]
WINDOW = timedelta(minutes=25)      # a look that can't start within 25 minutes of its time is skipped (looks are 30 minutes apart)


def log(msg):
    LIVE.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(ET):%Y-%m-%d %H:%M:%S} ET  {msg}"
    print(line, flush=True)
    with (LIVE / "live.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def gateway_cutoff(d):
    """When the Gateway's day ends: midnight in its login time zone (Kuala Lumpur), 12:00 ET in US daylight time and
    11:00 ET otherwise. Executions and completed orders from before it are no longer served (19 Oct: the 11:30 look's
    fills were lost that way), so the "sync" job records fills 3 minutes before it."""
    t = datetime.combine(d + timedelta(days=1), dtime(0, 0), MYT).astimezone(ET) - timedelta(minutes=3)
    return t.hour, t.minute


def sync():
    with gateway() as gw:
        n = paper.sync_fills(gw)
        reconcile(gw)
    log(f"fills synced before the gateway's midnight (Kuala Lumpur): {n} new")


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
    if any(m.get("ml") for m in paper.sleeves().values()):
        import fetch_extra                              # inputs of the ML arm's cost and market-vol forecasts
        with gateway() as gw:
            u = sorted(set(syms))
            bad = fetch_extra.update(gw, ["SPY"] + [x for x in u if x != "SPY"], "iv") + fetch_extra.update(gw, u, "bidask")
        log(f"refreshed implied vol and bid/ask history; {len(bad)} failed {bad[:3]}")
        as_of, fc = forecast.compute()
        log(f"forecasts for the ML arm: as of {as_of}, {len(fc)} symbols")
        try:                                            # earnings calendar + history (wave C); never fails the refresh
            import fetch_av
            r = fetch_av.daily(22)
            log(f"Alpha Vantage: calendar rows {r['calendar']}, history +{len(r['history'])}, {r['remaining']} to go, "
                f"{len(r['errors'])} errors {r['errors'][:2]}")
        except Exception as e:                          # noqa: BLE001
            log(f"Alpha Vantage capture failed: {type(e).__name__}")


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


def save_memo(name, now, call):
    if not call:                      # no valid decision: the previous notes stay
        return
    d = json.loads(MEMO.read_text(encoding="utf-8")) if MEMO.exists() else {}
    d[name] = {"when": f"{now:%a %d %b %H:%M} ET", "notes": str(call.get("notes") or "")[:2000]}
    MEMO.write_text(json.dumps(d, indent=1), encoding="utf-8")


def effort(meta):
    """The sleeve's reasoning effort: each model's maximum unless sleeves.json says otherwise (Clayton, 18 Oct)."""
    return meta.get("effort") or ("max" if meta["model"].startswith("claude-") else "high")


def finish(now, name, meta, text, st, call, trace, notes, table_rows, dry_run, cost=None, secs=None):
    out_dir = LIVE / now.date().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = research_summary(trace, call, table_rows)
    rec = {"time": now.isoformat(timespec="seconds"), "sleeve": name, "model": meta["model"], "effort": effort(meta),
           "seconds": secs, "state_text": text,
           "state": {k: v for k, v in st.items() if k != "events"}, "events": st["events"], "trace": trace, "call": call,
           "placed_notes": notes, "research": summ, "cost": cost, "dry_run": dry_run}
    (out_dir / f"{name}-{now:%H%M}.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    if not dry_run:
        paper.append(LIVE / "research.jsonl", {"time": rec["time"], "sleeve": name, **summ,
                                               "orders": len(call["place"]) if call else None,
                                               "notes_chars": len(str((call or {}).get("notes") or ""))})
        save_memo(name, now, call)
    log(f"{name}: {'DRY RUN ' if dry_run else ''}"
        + (f"{len(call['place'])} orders, {len(call['cancel'])} cancels" if call else "no valid decision")
        + f"; research {summ['research_calls']} {summ['tools'] or ''}; traded {summ['traded']}"
        + (f"; {secs:.0f}s" if secs is not None else "")
        + (f"; notes {notes}" if notes else ""))


def look(dry_run=False, only=None):
    now = datetime.now(ET)
    today = now.date()
    sleeves = {k: v for k, v in paper.sleeves().items() if not only or k in only}
    data = trader.Data(last=(today - timedelta(days=1)).strftime("%Y%m%d"))
    data.next_date = today.isoformat()
    d = len(data.dates)
    close = close_time(today)
    session = dtime(9, 30) <= now.time() < close
    day = f"Date: {today.isoformat()} ({today.strftime('%A')}), {now:%H:%M} ET."
    header = (f"{day} The market is open until {close:%H:%M} ET; prices are delayed about 15 minutes. Orders you submit now work immediately."
              if session else f"{day} Before the open. Orders you submit now work from the 09:30 open.")
    service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": "live"})
    claude_jobs, gptoss_jobs, claude_out = [], [], {}

    def run_claude():                       # one at a time; each look's tool server researches on its own client (12, 13)
        start = time.time()                 # all Claude looks share WINDOW, so their orders aren't placed after the open/close
        for i, (name, meta, text, st, system, rows, sp) in enumerate(claude_jobs):
            t0 = time.time()
            try:
                call, trace = agent.claude_look(meta["model"], system, text, sp, LIVE / today.isoformat() / f"{name}-{now:%H%M}",
                                                effort(meta), timeout=max(60, min(agent.CLAUDE_TIMEOUT,
                                                                                  start + WINDOW.total_seconds() - t0)),
                                                client=("trader-mcp-a", "trader-mcp-b")[i % 2])
            except Exception:
                call, trace = None, [{"error": traceback.format_exc()[-1500:]}]
            claude_out[name] = call, trace, time.time() - t0

    worker = threading.Thread(target=run_claude, daemon=True)
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
            live = {q["symbol"]: paper.mark(q) for q in gw.quotes(rows) if not q.get("error")}
        for name, meta in sleeves.items():
            world = trader.PaperWorld(gw, name, data)
            world.dry_run = dry_run
            text, st, feats = trader.state_text(world, data, d, header=header, live=live)
            ml = bool(meta.get("ml"))           # the ML arm: the same model plus the forecast tool (forecast.py)
            sp = specs + ([forecast.SPEC] if ml else [])
            text = trader.live_user(text, trader.goal_line(meta["start_cash"], st["value"], sessions_left(today), today), load_memo(name),
                                    steps=trader.live_steps(ml))
            rows, _ = data.table_rows(d, extra=world.symbols_in_play())
            system = trader.live_system(meta["start_cash"], today, ml=ml)
            if meta["model"].startswith("claude-"):
                claude_jobs.append((name, meta, text, st, system, rows, sp))
            else:
                gptoss_jobs.append((name, meta, text, st, system, rows, sp, world))
        worker.start()                      # the Claude looks run meanwhile; their orders are placed below, on client 10
        for name, meta, text, st, system, rows, sp, world in gptoss_jobs:
            model, t0 = meta["model"], time.time()
            call, trace, prefill, sampled, notes = None, [], 0, 0, []
            try:                            # an error ends this sleeve's look only; the rest and the Claude orders go on
                sc = service.create_sampling_client(base_model=BASE) if model in ("base", BASE) else service.create_sampling_client(model_path=model)
                call, trace, prefill, sampled = agent.gptoss_look(sc, gw, system, text, today.isoformat(), sp, effort(meta))
                notes = world.act(call) if call and not dry_run else []
            except Exception:
                trace = trace + [{"error": traceback.format_exc()[-1500:]}]
            finish(now, name, meta, text, st, call, trace, notes, rows, dry_run,
                   cost={"prefill_tokens": prefill, "sampled_tokens": sampled}, secs=time.time() - t0)
    worker.join()
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
        try:                                # record this look's fills now, not at the next look: see gateway_cutoff()
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
        acct = gw.account()
        rows = []
        for name in paper.sleeves():
            b0 = paper.book(name)
            syms = sorted(set(b0["positions"]) | {"SPY"})
            quotes = {q["symbol"]: paper.mark(q) for q in gw.quotes(syms) if not q.get("error")}
            b = paper.book(name, quotes)
            rows.append({"date": today, "sleeve": name, "model": b["model"], "nav": b["nav"], "cash": b["cash"],
                         "start_cash": b["start_cash"], "realized": b["realized"], "unrealized": b["unrealized"],
                         "commissions": b["commissions"], "fills": b["fills"], "missing_commissions": b["missing_commissions"],
                         "positions": {s: p["quantity"] for s, p in b["positions"].items()}, "spy": quotes.get("SPY"),
                         "account_net_liquidation": acct.get("NetLiquidation")})
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
    """Rebuild data/macro/calendar.json (ml_research/macro_fetch.py, about 10 minutes) after the record, so the next
    day's forecast macro line and tape have today's first-release values and any schedule change, and keep tonight's
    scheduled rows as data/macro/schedule/<YYYYMMDD>.json (the schedule as known then; it can't be rebuilt later).
    Logs and never raises: the old calendar stays if the build fails."""
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(ROOT / "ml_research" / "macro_fetch.py")], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=40 * 60)
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


def loop():
    log("live loop started")
    done = load_done()
    tried = {}                                        # refresh/record: last attempt, for retries
    held = set()
    while not STOP.exists():
        now = datetime.now(ET)
        if trading_day(now.date()):
            for name, (h, m) in JOBS + [("sync", gateway_cutoff(now.date()))]:
                key = (now.date().isoformat(), name)
                if key in done or (name.startswith("look-") and dtime(h, m) >= close_time(now.date())):
                    continue
                due = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if name == "sync":                # once, and only in the 3 minutes before the cutoff
                    if due <= now < due + timedelta(minutes=3):
                        mark_done(done, key)
                        run_job(name, sync)
                    continue
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
