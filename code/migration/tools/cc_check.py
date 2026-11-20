"""One read-only stall check for the live loop: the decision's half-hour check as a single command.
Prints nothing and exits 0 while healthy; prints ALERT and exits 1 on a stall. Writes nothing. Touches no process.
    python -I cc_check.py            check now
    python -I cc_check.py replay     test the rules on live.log's history
"""
import re, sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

LIVE = Path(r"C:\Claude\Trader\runs\live")
def _chain():
    """The gpt-oss sleeves in look order, from runs/sleeves.json: not Claude, not waiting on model "hold"."""
    import json
    try:
        sl = json.loads((LIVE.parent / "sleeves.json").read_text(encoding="utf-8"))
        return tuple(k for k, m in sl.items() if not str(m.get("model", "")).startswith("claude-") and m.get("model") != "hold")
    except (OSError, ValueError):
        return ("base-ml", "base-100k", "base-100k-ml")


GPTOSS = _chain()
LOOKS = [(h, m) for h in range(9, 16) for m in (0, 30)]
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ET  (.*)$")
START_GRACE = timedelta(minutes=6)      # 129 half-hour pairs since 19 Oct: none later than +3 min; worst single start +5.5 min
HANG = timedelta(minutes=21)            # past two 600 s sample deadlines, so a healthy retry on a new client is not killed


def events(text):
    out = []
    for l in text.splitlines():
        m = LINE.match(l)
        if m:
            out.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2)))
    return out


def verdict(ev, now, err, chain=GPTOSS):
    if re.search(r"_background_thread_func|InvalidStateError", err):
        return "ALERT dead Tinker thread: traceback in live.err.log"
    if now.weekday() > 4 or not (now.hour, now.minute) >= (9, 0) or now.hour >= 16:
        return None
    today = [e for e in ev if e[0].date() == now.date() and e[0] <= now]
    due = [(h, m) for h, m in LOOKS if now.replace(hour=h, minute=m, second=0, microsecond=0) + START_GRACE <= now]
    if due:
        name = f"look-{due[-1][0]:02d}{due[-1][1]:02d}"
        if not any(msg.startswith(f"{name} starting") or msg.startswith(f"{name} held") for _, msg in today):
            return f"ALERT no '{name} starting' or held line by {START_GRACE.seconds // 60} min after its time: the loop is stuck or down"
    starts = [i for i, (_, msg) in enumerate(today) if re.match(r"look-\d{4} starting", msg)]
    if starts:
        after = today[starts[-1]:]
        done = {msg.split(":")[0] for _, msg in after[1:]}
        missing = [n for n in chain if n not in done]
        if missing and now - after[-1][0] > HANG:
            return f"ALERT {after[0][1]} has no line for {missing[0]} and no log line for {int((now - after[-1][0]).seconds / 60)} min: suspected hang"
    return None


def replay():
    ev = [e for e in events((LIVE / "live.log").read_text(encoding="utf-8", errors="replace")) if e[0] >= datetime(2026, 10, 19)]
    days = sorted({e[0].date() for e in ev if e[1].startswith("look-")})
    chain = ("base", "base-ml", "rl5", "rl5-ml")       # weeks 2-3's chain
    false, checked, detail = 0, 0, []
    for d in days:
        if d.weekday() > 4:
            continue
        t = datetime(d.year, d.month, d.day, 9, 0)
        evd = [e for e in ev if e[0].date() == d]
        names = {m.split(":")[0] for _, m in evd}
        day_chain = tuple(n for n in ("base", "base-ml", "rl1", "rl2", "rl3", "rl4", "rl4-ml", "rl5", "rl5-ml") if n in names)   # that day's gpt-oss sleeves
        while t.hour < 16:
            checked += 1
            v = verdict(evd, t, "", day_chain)
            if v:
                false += 1
                detail.append((str(t), v[:90]))
            t += timedelta(minutes=1)
    print(f"history: {len(days)} days, {checked} minute checks, {false} alerts")
    seen = set()
    for t, v in detail:
        if v not in seen:
            seen.add(v)
            print("  first at", t, v)
    # Synthetic stalls on one real full day: cut the log at a chosen line and ask how soon each rule fires
    d = datetime(2026, 10, 28).date()
    evd = [e for e in ev if e[0].date() == d]
    start = next(i for i, e in enumerate(evd) if e[1].startswith("look-1000 starting"))
    for label, n in (("hung in the first sleeve", 0), ("hung after the first sleeve", 1), ("hung after the third", 3)):
        cut = evd[: start + 1 + n]
        t = cut[-1][0]
        while verdict(cut, t, "", chain) is None and t.hour < 16:
            t += timedelta(minutes=1)
        print(f"  {label}: last line {cut[-1][0].time()}, alert {t.time()} -> {verdict(cut, t, '', chain)[:80]}")
    print("  traceback in err log ->", verdict(evd, datetime(2026, 10, 28, 10, 5), "Exception in thread Thread-1 (_background_thread_func):")[:60])


if __name__ == "__main__":
    if sys.argv[1:] == ["replay"]:
        replay()
        sys.exit(0)
    now = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    v = verdict(events((LIVE / "live.log").read_text(encoding="utf-8", errors="replace")), now,
                (LIVE / "live.err.log").read_text(encoding="utf-8", errors="replace"))
    if v:
        print(now.strftime("%H:%M ET"), v)
        sys.exit(1)
