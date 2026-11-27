"""paper._read_json / _write_json: a whole-file ledger read retries a transient OSError, and so does a write in place.
Mon 23 Nov 2026, 09:32 ET: OSError [Errno 22] on runs/looks.json cost ink2-30k-ml its 09:30 look (live.log).
$0, files only, in a temp folder. Run: .venv/Scripts/python.exe -B ledger_io_test.py"""
import json
import pathlib
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  | {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


TMP = Path(tempfile.mkdtemp(prefix="ledger_io_"))
paper.IO_WAIT_S = 0.01
p = TMP / "looks.json"
p.write_text(json.dumps({"a": {"reported_exec": ["x"]}}), encoding="utf-8")

real_read = pathlib.Path.read_text
calls = {"n": 0}


def flaky(fail_first):
    def rt(self, *a, **k):
        if self == p:
            calls["n"] += 1
            if calls["n"] <= fail_first:
                raise OSError(22, "Invalid argument", str(self))
        return real_read(self, *a, **k)
    return rt


calls["n"] = 0
pathlib.Path.read_text = flaky(2)
try:
    got = paper._read_json(p, {})
finally:
    pathlib.Path.read_text = real_read
check("a read refused twice by Windows (Errno 22) is tried again and returns the ledger", got == {"a": {"reported_exec": ["x"]}}
      and calls["n"] == 3, f"{got} after {calls['n']} reads")

calls["n"] = 0
pathlib.Path.read_text = flaky(99)
try:
    paper._read_json(p, {})
    raised = None
except OSError as e:
    raised = e
finally:
    pathlib.Path.read_text = real_read
check(f"a read that keeps failing raises after {paper.IO_TRIES} tries, as before", isinstance(raised, OSError)
      and calls["n"] == paper.IO_TRIES, f"{raised!r} after {calls['n']}")

bad = TMP / "bad.json"
bad.write_text("{not json", encoding="utf-8")
n0 = {"n": 0}


def counting(self, *a, **k):
    if self == bad:
        n0["n"] += 1
    return real_read(self, *a, **k)


pathlib.Path.read_text = counting
try:
    paper._read_json(bad, {})
    perr = None
except ValueError as e:
    perr = e
finally:
    pathlib.Path.read_text = real_read
check("a file that reads but does not parse raises at once (not retried)", isinstance(perr, ValueError) and n0["n"] == 1)
check("an absent ledger gives the default", paper._read_json(TMP / "none.json", {"d": 1}) == {"d": 1})

w = TMP / "notes.json"
paper._write_json(w, {"s": ["one", "two"]}, indent=1)
check("_write_json writes the object (indent passed to json.dumps)", json.loads(w.read_text(encoding="utf-8")) == {"s": ["one", "two"]}
      and "\n" in w.read_text(encoding="utf-8"))

real_write = pathlib.Path.write_text
wc = {"n": 0}


def flaky_write(fail_first):
    def wt(self, *a, **k):
        if self == w:
            wc["n"] += 1
            if wc["n"] <= fail_first:
                raise OSError(22, "Invalid argument", str(self))
        return real_write(self, *a, **k)
    return wt


wc["n"] = 0
pathlib.Path.write_text = flaky_write(2)
try:
    paper._write_json(w, {"s": ["three"]})
finally:
    pathlib.Path.write_text = real_write
check("a write refused twice is tried again and lands", json.loads(w.read_text(encoding="utf-8")) == {"s": ["three"]} and wc["n"] == 3)
wc["n"] = 0
pathlib.Path.write_text = flaky_write(99)
try:
    paper._write_json(w, {"s": ["four"]})
    werr = None
except OSError as e:
    werr = e
finally:
    pathlib.Path.write_text = real_write
check(f"a write that keeps failing raises after {paper.IO_TRIES} tries", isinstance(werr, OSError) and wc["n"] == paper.IO_TRIES)

# the ledgers themselves, pointed at the temp folder
paper.LOOKS, paper.NOTES, paper.SLEEVES = TMP / "L.json", TMP / "N.json", TMP / "S.json"
paper.SLEEVES.write_text(json.dumps({"s1": {"start_cash": 30000.0, "model": "base"}}), encoding="utf-8")
paper.add_notes("s1", ["note a"])
paper.add_notes("s1", ["note b"])
check("add_notes then take_notes round-trips and empties the sleeve's notes",
      paper.take_notes("s1") == ["note a", "note b"] and paper.take_notes("s1") == [])
check("sleeves() reads through _read_json", paper.sleeves() == {"s1": {"start_cash": 30000.0, "model": "base"}})


class GW:
    def orders(self):
        return []


paper.fills = lambda: [{"sleeve": "s1", "exec_id": "e1", "side": "BOT", "shares": 5, "symbol": "AAPL", "price": 100.0,
                        "commission": 1.0, "ref": "tt-s1-1"}]
ev1 = paper.events_since_last_look(GW(), "s1")
ev2 = paper.events_since_last_look(GW(), "s1")
check("events_since_last_look reports a fill once, through _write_json", len(ev1) == 1 and ev2 == []
      and json.loads(paper.LOOKS.read_text(encoding="utf-8"))["s1"]["reported_exec"] == ["e1"], f"{ev1} {ev2}")

# many threads at once under one lock, as the live loop's lanes hold `ledger`: every read sees a whole file and no
# write fails; errors in any thread are counted
lock, stop, seen_bad = threading.Lock(), threading.Event(), []
target = TMP / "race.json"
paper._write_json(target, {"k": list(range(2000))})


def writer(i):
    try:
        while not stop.is_set():
            with lock:
                d = paper._read_json(target, {})
                d["w"] = i
                paper._write_json(target, d)
    except Exception as e:                              # noqa: BLE001
        seen_bad.append(f"writer {i}: {e!r}")


def reader():
    try:
        for _ in range(300):
            with lock:
                d = paper._read_json(target, None)
            if not d or len(d["k"]) != 2000:
                seen_bad.append("short")
    except Exception as e:                              # noqa: BLE001
        seen_bad.append(f"reader: {e!r}")


ws = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
rs = [threading.Thread(target=reader) for _ in range(8)]
for x in ws + rs:
    x.start()
for x in rs:
    x.join()
stop.set()
for x in ws:
    x.join()
check("8 readers and 4 read-modify-write writers under one lock: whole files only, no error in any thread", not seen_bad,
      str(seen_bad[:3]))

# the real cause, reproduced: another process memory-maps the ledger (as git does when hashing a changed tracked file);
# an in-place write is refused with Errno 22 while the map is held, and _write_json lands once it is released
import subprocess  # noqa: E402
mm = TMP / "mapped.json"
paper._write_json(mm, {"v": list(range(300))})
hold = subprocess.Popen([sys.executable, "-c", "import mmap,sys,time;f=open(sys.argv[1],'rb');"
                         "m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);print('mapped',flush=True);time.sleep(0.6)",
                         str(mm)], stdout=subprocess.PIPE, text=True)
hold.stdout.readline()
try:
    mm.write_text("{}", encoding="utf-8")
    plain = None
except OSError as e:
    plain = e.errno
paper.IO_WAIT_S = 0.25
try:
    paper._write_json(mm, {"v": "after"})
    landed = json.loads(mm.read_text(encoding="utf-8")) == {"v": "after"}
except OSError:
    landed = False
paper.IO_WAIT_S = 0.01
hold.wait()
check("a held memory map refuses a plain in-place write (Errno 22), and _write_json lands once it is released",
      plain == 22 and landed, f"plain errno {plain}, landed {landed}")

import live  # noqa: E402
live.DONE = TMP / "done.json"
d = live.load_done()
live.mark_done(d, ("2026-11-23", "look-0930"))
live.mark_done(d, ("2026-11-23", "look-1000"))
check("live.load_done / mark_done round-trip through _read_json / _write_json",
      live.load_done() == {("2026-11-23", "look-0930"), ("2026-11-23", "look-1000")})

# a ledger that existed and went away mid-read raises as before (never the default, which would rebuild it from nothing)
gone = TMP / "gone.json"
gone.write_text("{}", encoding="utf-8")


def vanish(self, *a, **k):
    if self == gone:
        raise FileNotFoundError(2, "No such file", str(self))
    return real_read(self, *a, **k)


pathlib.Path.read_text = vanish
try:
    paper._read_json(gone, {"default": True})
    gerr = None
except FileNotFoundError as e:
    gerr = e
finally:
    pathlib.Path.read_text = real_read
check("a ledger that goes away between exists() and the read raises FileNotFoundError, not the default", gerr is not None)

# mark_done never stops the loop: a write still refused after the retries is logged and the key kept in memory
logged = []
live.log = lambda msg: logged.append(msg)
real_wj = paper._write_json
paper._write_json = lambda *a, **k: (_ for _ in ()).throw(OSError(22, "Invalid argument"))
try:
    dd = set()
    live.mark_done(dd, ("2026-11-23", "look-1045"))
    md_ok = ("2026-11-23", "look-1045") in dd and any("done.json not written" in m for m in logged)
except OSError:
    md_ok = False
finally:
    paper._write_json = real_wj
check("mark_done with a write refused throughout: no exception, the key kept, one log line", md_ok, str(logged))

print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
