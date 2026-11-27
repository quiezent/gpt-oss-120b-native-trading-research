"""Test of intraday._one_scale (one share scale for data/bars30 sessions, from the adjusted daily cache data/bars) and of
_history's gateway month for a name whose cache lacks the latest session (rule v2, round 2, after its two reviews).

    .venv/Scripts/python.exe -B ml_research/intraday_scale_test.py      (from the checkout's root; about 5 minutes)
    ... intraday_scale_test.py --daily <folder>      the same against a stand-in for data/bars, read only (for
    ... intraday_scale_test.py --daily=<folder>       example a copy with BITO's 20 Nov bar, to see Monday's state)

--daily: the folder must exist and hold a daily file for every name in lists/universe523.txt, else the test says so
and exits 2. It replaces data/bars for every daily-bar read in this test: intraday.DAILY (_one_scale in the real, e2e
and fallback checks; the synthetic checks use their own fixtures), the BITO20 flag, the standin_daily and asof_daily helpers (which build
from it), the round 1 module's DAILY, and load_panel's trader.Data() (the test points bars.DATA at it for the three
load_panel calls, so the panel's symbol list, calendar and dividend jumps come from it too). Not replaced: data/bars30
(the real 30-minute files), data/iv, lists/ and the desk's split ledger. The folder in use is printed first and in
the final count line.

$0, files only: no network (an audit hook blocks every socket) and no write outside runs/ml/research/bars30_scale
(the hook blocks those too). Synthetic fixtures go to runs/ml/research/bars30_scale/round2/fixtures, panels to
runs/ml/research/bars30_scale/round2; intraday.BARS30 and intraday.DAILY are pointed at them inside this process. Real
files are read only. Three modules are compared: I = ml_research/intraday.py (v2), R1 = the round 1 module
(runs/ml/research/bars30_scale/round2/intraday_round1.py) and O = the unpatched reference
(runs/ml/research/bars30_scale/build/intraday_orig.py, the module as it was before the fix), the last two imported with
their paths set to this checkout's. Both reference modules are test fixtures kept in git and pinned here by a hash of
their text (line endings ignored), so the test runs on any checkout that has them and imports no other checkout's
code (it reads the desk's split ledger from C:/Claude/Trader/runs/splits.jsonl when that file exists); without them
it says which files to take and exits 2. Gateways are fakes. Prints one line a check and a final count; exits 1
on any failure.

Dated: the real-file checks describe data/bars30 and data/bars as they stood on 21 to 23 Nov 2026 (523 names; SOXX and
BITO the only names that change; SNDK without a 30-minute file; BITO's daily bar for 20 Nov arriving with Monday's
refresh, which the BITO20 flag follows). The next split in the universe, or a change of the universe, makes those
expectations out of date: update them then; the synthetic and fallback checks do not depend on it. The real files
are read as they are, with no snapshot: do not run it across the desk's 07:30 refresh or its 16:15 record. The run leaves
about 90 MB of panels and fixtures under runs/ml/research/bars30_scale/round2: do not add that folder to git.

Round 4 (fallback_r4): an answer that arrives but cannot be read or used is remembered as None (asked once that day),
only a request that raises is retried after GW_FILL_RETRY_MIN; inside one live() call names never asked today take
the asks before retries. A call with no retry candidate is compared with runs/ml/research/bars30_scale/round4/before/
intraday.py (the module before round 4) when that file is there.
"""

import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SAFE = os.path.normcase(os.path.abspath(str(ROOT / "runs" / "ml" / "research" / "bars30_scale")))
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _inside(p):
    p = os.path.normcase(os.path.abspath(os.fsdecode(p)))
    return p == _SAFE or p.startswith(_SAFE + os.sep) or p == os.path.normcase(os.devnull)


def _hook(ev, args):
    if ev in ("socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.bind", "http.client.connect",
              "urllib.Request"):
        raise PermissionError("test guard: network blocked (%s)" % ev)
    if ev == "open" and len(args) >= 3 and isinstance(args[0], (str, bytes, os.PathLike)):
        mode, flags = args[1], args[2]
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
            (mode is None and isinstance(flags, int) and flags & _WFLAGS)
        if w and not _inside(args[0]):
            raise PermissionError("test guard: write blocked %s" % args[0])
    if ev in ("os.mkdir", "os.remove", "os.rmdir", "shutil.rmtree", "os.truncate"):
        if isinstance(args[0], (str, bytes, os.PathLike)) and not _inside(args[0]):
            raise PermissionError("test guard: %s blocked %s" % (ev, args[0]))
    if ev in ("os.rename", "os.replace"):
        for a in args[:2]:
            if isinstance(a, (str, bytes, os.PathLike)) and not _inside(a):
                raise PermissionError("test guard: %s blocked %s" % (ev, a))


sys.addaudithook(_hook)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import numpy as np                       # noqa: E402
from ml_research import intraday as I    # noqa: E402

def _daily_option(argv):
    """The folder of --daily <folder> or --daily=<folder> (a stand-in for data/bars, read only), or None without the
    option. A bare --daily, a folder that does not exist, or one without a daily file per universe name: a message and
    exit 2."""
    val = None
    for i, a in enumerate(argv[1:], 1):
        if a == "--daily":
            val = argv[i + 1] if i + 1 < len(argv) else ""
        elif a.startswith("--daily="):
            val = a[len("--daily="):]
    if val is None:
        return None
    if not val or val.startswith("--"):
        print("--daily needs a folder: --daily <folder> or --daily=<folder> (a stand-in for data/bars, read only)",
              flush=True)
        sys.exit(2)
    p = Path(val).resolve()
    if not p.is_dir():
        print(f"--daily: no such folder: {p}", flush=True)
        sys.exit(2)
    uni = [x.strip() for x in (ROOT / "lists" / "universe523.txt").read_text().splitlines() if x.strip()]
    miss = [s for s in uni if not (p / f"{s.replace(' ', '_')}.json").exists()]
    if miss:
        print(f"--daily: {p} lacks the daily file of {len(miss)} of the {len(uni)} universe names (first: "
              f"{', '.join(miss[:5])}); a stand-in needs them all", flush=True)
        sys.exit(2)
    return p


_DAILY_OPT = _daily_option(sys.argv)
if _DAILY_OPT is not None:
    I.DAILY = _DAILY_OPT
SCALE = ROOT / "runs" / "ml" / "research" / "bars30_scale"
ORIG = SCALE / "build" / "intraday_orig.py"                  # the unpatched copy
ROUND1 = SCALE / "round2" / "intraday_round1.py"              # round 1, unchanged copy
R4BEFORE = SCALE / "round4" / "before" / "intraday.py"        # the module before round 4 (not pinned: its check is
#                                                               skipped with a note when the file is absent)
PINNED = {ORIG: "9a730c064b531cbda35014f1fa191e4ba167b8ce9c3d1a4403834d008b6e74aa",       # sha256 of the text with LF
          ROUND1: "67aee69e6316cb4fc1c0d8424243797ea97e0cf86bc969dd502017701344519f"}     # line endings (text_sha)
WORK = SCALE / "round2"                                       # fixtures and panels this test writes
FIX = WORK / "fixtures"
REAL30, REALD, REALIV = I.BARS30, I.DAILY, I.IVDIR
LEDGER = Path("C:/Claude/Trader/runs/splits.jsonl")           # the desk's split ledger in the live checkout (a worktree's
if not LEDGER.exists():                                       # copy can be behind); else this checkout's
    LEDGER = ROOT / "runs" / "splits.jsonl"
SHIPPED = [I.OUT / "intraday_panel.npz", I.OUT / "intraday_model.pkl"]
# Monday's 07:30 refresh adds a 20 Nov daily bar; the panel path (no level step) changes BITO only then
BITO20 = any(str(b["date"]).replace("-", "")[:8] == "20261120"
             for b in json.loads((REALD / "BITO.json").read_text())["bars"])
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT, m.OUT, m.BARS30, m.IVDIR, m.CACHE, m.MODEL = ROOT, I.OUT, REAL30, REALIV, I.CACHE, I.MODEL
    if hasattr(m, "DAILY"):
        m.DAILY = REALD
    return m


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def text_sha(p):
    """sha256 of a text file with CRLF read as LF, so a checkout's line endings do not change it."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def set30(folder, *mods):
    for m in mods:
        m.BARS30 = folder


def setd(folder, *mods):
    for m in mods:
        if hasattr(m, "DAILY"):
            m.DAILY = folder


# ---------------------------------------------------------------- synthetic fixtures
F30, FD = FIX / "bars30", FIX / "bars"
N = 30
DATES = []
_d = datetime(2026, 9, 1)
while len(DATES) < N:
    if _d.weekday() < 5:
        DATES.append(_d.strftime("%Y%m%d"))
    _d += timedelta(days=1)
DI = {d: i for i, d in enumerate(DATES)}
BEFORE = "20270101"
Q = [100 * np.exp(0.01 * np.sin(i)) for i in range(N)]          # current-scale close
U = [1e6 * (1 + 0.1 * np.cos(i)) for i in range(N)]             # current-scale session volume
OPEN_NEW = Q[-1]                                                # today's open on the current scale (no move overnight)


def session_rows(day, close, vol):
    out = []
    for k, sl in enumerate(I.SLOTS):
        c = close * (1 + 0.001 * (k - 12))
        o = c * (1 - 0.0005)
        out.append({"date": f"{day} {sl}:00 US/Eastern", "open": o, "high": max(o, c) * 1.002, "low": min(o, c) * 0.998,
                    "close": c, "volume": vol / 13})
    return out


def write(sym, tc, tv, dc=None, dv=None, drop30=(), dkeep=None, draw=None, dextra=None, short=None):
    """bars30 as traded (tc, tv per session; short {index: bars kept}); daily (dc, dv) for indices in dkeep;
    draw = raw text of the daily file."""
    F30.mkdir(parents=True, exist_ok=True)
    FD.mkdir(parents=True, exist_ok=True)
    fn = sym.replace(" ", "_") + ".json"
    short = short or {}
    rows = [r for i, d in enumerate(DATES) if i not in drop30 for r in session_rows(d, tc[i], tv[i])[:short.get(i, 13)]]
    (F30 / fn).write_text(json.dumps({"symbol": sym, "what": "TRADES", "bars": rows}), encoding="utf-8")
    p = FD / fn
    if draw is not None:
        p.write_text(draw, encoding="utf-8")
    elif dc is None:
        if p.exists():
            p.unlink()
    else:
        keep = range(N) if dkeep is None else dkeep
        bars = [{"date": DATES[i], "open": dc[i], "high": dc[i], "low": dc[i], "close": dc[i], "volume": dv[i]} for i in keep]
        bars += dextra or []
        p.write_text(json.dumps({"symbol": sym, "what": "ADJUSTED_LAST", "bars": bars}), encoding="utf-8")


def split_case(sym, splits, **kw):
    """splits {index of ex-date: n}: as traded, sessions before an ex-date are on the old scale (price * n, volume / n)."""
    Fs = [float(np.prod([n for s, n in splits.items() if i < s])) for i in range(N)]
    write(sym, [Q[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], Q, list(U), **kw)
    return Fs


def moved_case(sym, splits, moves, **kw):
    """split_case with real overnight moves {index: m} in both series (the daily cache is the moved current scale)."""
    M = [float(np.prod([m for j, m in moves.items() if j <= i])) for i in range(N)]
    Fs = [float(np.prod([n for s, n in splits.items() if i < s])) for i in range(N)]
    write(sym, [Q[i] * M[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], [Q[i] * M[i] for i in range(N)],
          list(U), **kw)
    return Fs


def summ(h):
    return {d: (bb[max(bb)]["close"], sum(b["volume"] for b in bb.values())) for d, bb in h.items()}


def at_scale(h, closes, vols, tol=1e-9):
    s = summ(h)
    return bool(s) and all(abs(c / closes[DI[d]] - 1) < tol and abs(v / vols[DI[d]] - 1) < tol for d, (c, v) in s.items())


def run_case(O, R1, sym, before=BEFORE, op=None, panel=False):
    """{o: unpatched history, r1: round 1 history, p: v2 history (with today's open op), r: v2 _one_scale on the
    unpatched window (before=None if panel), w: that window, nm: window not mutated, seen}."""
    set30(F30, I, O, R1)
    setd(FD, I, R1)
    ho = O._cached_history(sym, before)
    h1 = R1._cached_history(sym, before)
    hp = I._cached_history(sym, before, open_today=op)
    w = O._cached_history(sym, before)
    snap = copy.deepcopy(w)
    r = I._one_scale(sym, w) if panel else I._one_scale(sym, w, before=before, open_today=op)
    return {"o": ho, "r1": h1, "p": hp, "r": r, "w": w, "nm": w == snap, "seen": I._SCALE_SEEN.get(sym, {})}


def unchanged(c):
    return c["r"] is c["w"] and c["p"] == c["o"] and c["nm"]


def fixed(c):
    return at_scale(c["p"], Q, U) and c["r"] is not c["w"] and c["nm"] and c["p"] != c["o"]


def synthetic(O, R1):
    # ---- round 1 checks that v2 keeps
    write("NOSPLIT", Q, U, Q, U)
    c = run_case(O, R1, "NOSPLIT", op=OPEN_NEW)
    check("synthetic no split (today's open given): same objects back, history as read", c["r"] is c["w"]
          and c["p"] == c["o"] and c["nm"] and len(c["p"]) == 25)
    for sym, sp in (("FWD3", {15: 3}), ("REV12", {15: 0.5}), ("S32", {15: 1.5}), ("S20", {15: 20}),
                    ("TWO", {10: 2, 20: 3})):
        split_case(sym, sp)
        c = run_case(O, R1, sym, op=OPEN_NEW)
        b = c["seen"]["boundaries"]
        ok = fixed(c) and len(b) == len(sp) and [x[3] for x in b] == [DATES[s] for s in sorted(sp)]
        check(f"synthetic {sym} {sp}: every session on the current scale, boundary at the ex-date, input not mutated", ok,
              f"boundaries {b}")
    for sym, drop, adj in (("DIV2", 0.02, 0.98), ("DIV6", 0.06, 0.94)):
        tc = [Q[i] * (1 if i < 15 else 1 - drop) for i in range(N)]
        write(sym, tc, U, [tc[i] * (adj if i < 15 else 1) for i in range(N)], U)
        c = run_case(O, R1, sym, op=tc[-1])
        check(f"synthetic {sym} ex-date ({drop:.0%}): no scaling, same objects", unchanged(c))
    tc = [Q[i] * (1 if i < 15 else 0.75) for i in range(N)]
    write("SPEC25", tc, U, [tc[i] * (0.75 if i < 15 else 1) for i in range(N)], U)
    c = run_case(O, R1, "SPEC25")
    rej = c["seen"]["rejected"]
    check("synthetic 25% special dividend, volume unchanged: no scaling", unchanged(c) and rej
          and rej[0][3] == "price changed without the share count", f"rejected {rej}")
    tc = [Q[i] * (1 if i < 15 else 0.7) for i in range(N)]
    tv = [U[i] * (1 if i < 15 else 5) for i in range(N)]
    write("FALL30", tc, tv, tc, tv)
    c = run_case(O, R1, "FALL30", op=tc[-1])
    check("synthetic real 30% fall with 5x volume in both series: no scaling", unchanged(c))
    Fs = [3.2 if i < 15 else 1 for i in range(N)]
    write("ODD32", [Q[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], Q, U)
    c = run_case(O, R1, "ODD32")
    rej = c["seen"]["rejected"]
    check("synthetic step of 3.2 (6.7% from 3): unexplained, no scaling", unchanged(c) and rej
          and rej[0][3] == "unexplained", f"rejected {rej}")
    split_case("GAPREV", {15: 0.5}, drop30={15})
    c = run_case(O, R1, "GAPREV")
    b = c["seen"]["boundaries"]
    check("synthetic reverse 1-for-2 with the ex-date missing from the 30-minute file: boundary at the next stored session",
          at_scale(c["p"], Q, U) and b == [[DATES[14], DATES[16], 0.5, DATES[16]]], f"boundaries {b}")
    split_case("DTAIL", {15: 3}, dkeep=range(0, 26))
    c = run_case(O, R1, "DTAIL", op=OPEN_NEW)
    check("synthetic 3-for-1, daily cache ends 4 sessions before the newest stored one", fixed(c))
    split_case("NODAILY", {15: 3})
    (FD / "NODAILY.json").unlink()
    c = run_case(O, R1, "NODAILY", op=OPEN_NEW)
    check("synthetic no daily file: history unchanged, same objects", unchanged(c) and "error" in c["seen"], str(c["seen"]))
    # ---- round 1 checks v2 makes untrue, changed (round 1's outcome stated in each line)
    split_case("DEARLY", {20: 3}, dkeep=range(0, 19))
    c = run_case(O, R1, "DEARLY")
    check("CHANGED from round 1 (was unchanged): daily cache ends before the split, cache on the current scale: the level "
          "step finds the stored jump after R and fixes it", fixed(c) and c["r1"] == c["o"]
          and c["seen"]["level"] == [DATES[18], 3, DATES[20]], f"level {c['seen']['level']}")
    c = run_case(O, R1, "DEARLY", panel=True)
    check("same, no level step (load_panel's call): unchanged", c["r"] is c["w"])
    split_case("LVL", {N: 3})                   # every stored session on the old scale, cache re-based, newest = N
    c = run_case(O, R1, "LVL")
    check("CHANGED from round 1 (was fixed): level step without today's open and no stored session after R: unchanged",
          unchanged(c) and at_scale(c["r1"], Q, U) and c["seen"]["rejected"][-1][3] == "no as-traded jump",
          f"rejected {c['seen']['rejected']}")
    c = run_case(O, R1, "LVL", op=OPEN_NEW)
    check("level step with today's open on the new scale: fixed, boundary at TODAY", fixed(c)
          and c["seen"]["level"] == [DATES[-1], 3, I.TODAY], f"level {c['seen']['level']}")
    c = run_case(O, R1, "LVL", op=np.float64(OPEN_NEW))
    check("level step with today's open as a numpy float: fixed", fixed(c))
    c = run_case(O, R1, "LVL", op=OPEN_NEW, panel=True)
    check("level step off (load_panel's call, no before): unchanged", c["r"] is c["w"])
    for lab, op in (("None", None), ("zero", 0), ("negative", -5.0), ("NaN", float("nan")), ("inf", float("inf")),
                    ("text", "x"), ("numeric text", str(OPEN_NEW)), ("True", True), ("a list", [OPEN_NEW])):
        try:
            c = run_case(O, R1, "LVL", op=op)
            ok, why = unchanged(c), ""
        except Exception as e:     # noqa: BLE001
            ok, why = False, repr(e)
        check(f"today's open {lab}: no TODAY step, unchanged, no exception", ok, why)
    split_case("LVLOLD", {N: 3}, dkeep=range(0, N - 1))
    c = run_case(O, R1, "LVLOLD")
    check("cache newest one session before the newest stored, both stored on the old scale, no open: unchanged", unchanged(c))
    split_case("LVLNEW", {N: 3}, dextra=[{"date": "20261231", "open": Q[-1], "high": Q[-1], "low": Q[-1],
                                          "close": Q[-1], "volume": U[-1]}])
    c = run_case(O, R1, "LVLNEW")
    check("cache newest after the newest stored session, no open: unchanged", unchanged(c))
    c = run_case(O, R1, "LVLNEW", op=OPEN_NEW)
    check("CHANGED from round 1 (known limit, was unchanged): cache newest after the newest stored session, today's "
          "open on the new scale: fixed", fixed(c) and c["r1"] == c["o"])
    # ---- the five cases of rule v2
    split_case("C12", {N - 1: 0.5}, drop30={N - 1})       # ex-date missing from bars30, cache newest = ex-date (> N)
    c = run_case(O, R1, "C12", op=OPEN_NEW)
    check("case (2) / review 12: 1-for-2, ex-date missing from bars30, cache newest = ex-date, today's open: fixed",
          fixed(c) and c["r1"] == c["o"] and c["seen"]["level"] == [DATES[N - 2], 0.5, I.TODAY], f"level {c['seen']['level']}")
    c = run_case(O, R1, "C12")
    check("case (2) without today's open: unchanged (A3: no stored session after R, no open)", unchanged(c))
    split_case("C13", {N: 3}, short={N - 1: 7})          # previous session a half day (dropped), split ex today
    c = run_case(O, R1, "C13", op=OPEN_NEW)
    check("case (3) / review 13: split ex the session after a half day (half day dropped, cache newest = half day), "
          "today's open: fixed", fixed(c) and c["r1"] == c["o"] and DATES[N - 1] not in c["p"])
    c = run_case(O, R1, "C13")
    check("case (3) without today's open: unchanged", unchanged(c))
    split_case("C04", {N - 1: 3}, dkeep=range(0, N - 1))  # ex-date session stored, its 07:30 refresh failed
    for lab, op in (("without", None), ("with", OPEN_NEW)):
        c = run_case(O, R1, "C04", op=op)
        check(f"case (4): morning after a split, 07:30 refresh failed (N = ex-date, no daily bar), {lab} today's open: "
              "fixed", fixed(c) and c["r1"] == c["o"] and c["seen"]["level"] == [DATES[N - 2], 3, DATES[N - 1]],
              f"level {c['seen']['level']}")
    split_case("C02B", {15: 3}, dkeep=[i for i in range(N) if i != 14])
    c = run_case(O, R1, "C02B")
    b = c["seen"]["boundaries"]
    check("CHANGED from round 1 (was unchanged) case (5) / review 02b: daily bar missing for the last pre-split session: "
          "fixed, that session on the old side", fixed(c) and c["r1"] == c["o"] and b == [[DATES[13], DATES[15], 3, DATES[15]]],
          f"boundaries {b}")
    split_case("C02A", {15: 3}, dkeep=[i for i in range(N) if i != 15])
    c = run_case(O, R1, "C02A")
    check("review 02: daily bar missing for the ex-date session: still fixed", fixed(c), f"boundaries {c['seen']['boundaries']}")
    split_case("C02C", {15: 3}, dkeep=[i for i in range(N) if i not in (14, 15)])
    c = run_case(O, R1, "C02C")
    check("review 02: daily bars missing on both sides: fixed (one matching step in the pair)", fixed(c))
    # ---- half days
    split_case("HALF15", {15: 3}, short={15: 7})
    c = run_case(O, R1, "HALF15")
    check("split ex on a half day inside the window (half day dropped): fixed", fixed(c) and DATES[15] not in c["p"])
    split_case("HALFN", {N - 1: 3}, short={N - 1: 7})
    c = run_case(O, R1, "HALFN", op=OPEN_NEW)
    check("split ex on a half day that is the latest session (dropped; cache newest = half day), today's open: fixed",
          fixed(c) and c["seen"]["level"] == [DATES[N - 2], 3, I.TODAY])
    c = run_case(O, R1, "HALFN")
    check("same without today's open: unchanged", unchanged(c))
    # ---- placement must be exactly one matching step
    F, Cm, V = ([3 if i < 15 else 1 for i in range(N)], [1 / 3 if i >= 17 else 1 for i in range(N)],
                [3 if i >= 17 else 1 for i in range(N)])
    write("TWOSTEP", [Q[i] * F[i] * Cm[i] for i in range(N)], [U[i] / F[i] * V[i] for i in range(N)],
          [Q[i] * Cm[i] for i in range(N)], [U[i] * V[i] for i in range(N)], dkeep=[i for i in range(N) if i not in (15, 16, 17)])
    c = run_case(O, R1, "TWOSTEP")
    rej = c["seen"]["rejected"]
    check("two matching steps in a pair (split at 15, real 1/3 fall at 17, daily missing 15-17): rejected, unchanged",
          unchanged(c) and rej and rej[0][:2] == [DATES[14], DATES[18]] and rej[0][3] == "no as-traded jump", f"rejected {rej}")
    write("OUTSIDE", [Q[i] * F[i] for i in range(N)], [U[i] / F[i] for i in range(N)],
          [Q[i] * F[i] / (3 if i < 18 else 1) for i in range(N)], [U[i] / F[i] * (3 if i < 18 else 1) for i in range(N)])
    c = run_case(O, R1, "OUTSIDE")
    rej = c["seen"]["rejected"]
    check("matching step outside the pair (stored jump at 15, IBKR's break at 18): rejected, unchanged", unchanged(c)
          and rej and rej[0][:2] == [DATES[17], DATES[18]] and rej[0][3] == "no as-traded jump", f"rejected {rej}")
    for lab, cut in (("late (16)", 16), ("early (14)", 14)):
        write("C16", [Q[i] * F[i] for i in range(N)], [U[i] / F[i] for i in range(N)],
              [Q[i] * F[i] / (3 if i < cut else 1) for i in range(N)], [U[i] / F[i] * (3 if i < cut else 1) for i in range(N)])
        c = run_case(O, R1, "C16", op=OPEN_NEW)
        rej = c["seen"]["rejected"]
        check(f"review 16: IBKR applies the 3-for-1 one session {lab}: rejected, unchanged", unchanged(c)
              and rej and rej[0][3] == "no as-traded jump", f"rejected {rej}")
    write("C10", Q, U, [Q[i] / (2 if i < 15 else 1) for i in range(N)], [U[i] * (2 if i < 15 else 1) for i in range(N)])
    c = run_case(O, R1, "C10", op=OPEN_NEW)
    rej = c["seen"]["rejected"]
    check("review 10: daily cache has its own 2-for-1 break, stored file none: rejected, unchanged", unchanged(c)
          and rej and rej[0][3] == "no as-traded jump", f"rejected {rej}")
    split_case("LVL2", {N - 2: 3}, dkeep=range(0, N - 2))     # stored step after R matches, and today's open too
    c = run_case(O, R1, "LVL2", op=Q[-1] / 3)
    check("level step with two matching steps (a stored session after R and today's open): rejected, unchanged",
          unchanged(c) and c["seen"]["rejected"][-1][3] == "no as-traded jump", f"rejected {c['seen']['rejected']}")
    # ---- round 2 review cases that gave a WRONG scale in v2 as built (review_rule p3_break / p3b_wide)
    moved_case("S2F", {10: 2, 20: 3}, {9: 0.55}, dkeep=[i for i in range(N) if i != 9])
    c = run_case(O, R1, "S2F", op=OPEN_NEW)
    check("review S2f: one split's pair rejected (two matching steps), another split fine: any rejection leaves the "
          "history as read (round 1 and v2 as built scaled the older sessions by 3 where the truth is 6)",
          unchanged(c) and c["seen"]["boundaries"] and c["seen"]["rejected"][0][3] == "no as-traded jump"
          and c["r1"] != c["o"], f"seen {c['seen'].get('boundaries')} {c['seen'].get('rejected')}")
    split_case("S2E", {15: 2})
    c = run_case(O, R1, "S2E", op=OPEN_NEW / 3)
    check("review S2e: a split in the window and today's open a 3-for-1 below yesterday's close, cache not re-based: "
          "unchanged (round 1 put every session on yesterday's scale)", unchanged(c)
          and c["seen"]["rejected"][-1][3] == "today's open jumps with no boundary", f"rejected {c['seen'].get('rejected')}")
    c = run_case(O, R1, "S2E", op=OPEN_NEW)
    check("same with a quiet open: fixed", fixed(c))
    write("I4", [Q[i] * (2 if i < 15 else 1) for i in range(N)], [U[i] / (2 if i < 15 else 1) for i in range(N)],
          [Q[i] * (2 / 3 if i < 15 else 1) for i in range(N)], [U[i] * (3 / 2 if i < 15 else 1) for i in range(N)])
    c = run_case(O, R1, "I4", op=OPEN_NEW)
    check("review I4: IBKR's ratio 3 where the stored step is a 2-for-1: the step's own nearest ratio is 2, rejected, "
          "unchanged (round 1 scaled by 3)", unchanged(c) and c["r1"] != c["o"], f"rejected {c['seen'].get('rejected')}")
    moved_case("M5", {15: 2}, {15: 1.3})
    c = run_case(O, R1, "M5", op=OPEN_NEW)
    check("a real +30% move on a 2-for-1 night (step nearest 3-for-2): rejected, unchanged (round 1 fixed it)",
          unchanged(c) and at_scale(c["r1"], [Q[i] * (1.3 if i >= 15 else 1) for i in range(N)], U))
    moved_case("W4B", {26: 2}, {26: 0.6, 28: 0.6}, dkeep=range(0, 26))
    c = run_case(O, R1, "W4B")
    check("review W4b: level step, the split night out of band by a real -40% and a real -40% two sessions later in band: "
          "rejected (another candidate step above 0.2), unchanged", unchanged(c), f"seen {c['seen'].get('level')} "
          f"{c['seen'].get('rejected')}")
    moved_case("W5", {N - 1: 2}, {N - 1: 0.6}, dkeep=range(0, N - 1))
    c = run_case(O, R1, "W5", op=OPEN_NEW * 0.6 * 0.6)
    check("review W5: level step, the stored split step out of band by a real -40% and today's real -40% gap in band: "
          "rejected, unchanged", unchanged(c), f"seen {c['seen'].get('level')} {c['seen'].get('rejected')}")
    c = run_case(O, R1, "LVL", op=np.bool_(True))
    check("today's open np.bool_(True): not a price, unchanged", unchanged(c))
    # ---- replay: before in the past, cache from the future: no level step, as round 1
    split_case("REPLAY", {25: 3})
    c = run_case(O, R1, "REPLAY", before=DATES[20], op=OPEN_NEW)
    check("replay with before in the past against a current cache: no level step even with an open, same as round 1",
          unchanged(c) and c["r1"] == c["o"] and c["seen"]["level"] is None)
    c = run_case(O, R1, "C02A", before=DATES[25])
    check("replay before = session 25 on a split at 15: same as round 1", fixed(c) and c["p"] == c["r1"])
    # ---- noise, malformed files, bad bars (round 1 checks kept)
    for ratio in (2, 1.25):
        for sd in (0.1, 0.3):
            cnt = {"fixed": 0, "unchanged": 0, "WRONG": 0}
            rng = np.random.default_rng(7)
            for _ in range(100):
                Fs = [ratio if i < 15 else 1 for i in range(N)]
                noise = np.exp(rng.normal(0, sd, N))
                write("NOISE", [Q[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], Q,
                      [U[i] * noise[i] for i in range(N)])
                c = run_case(O, R1, "NOISE", op=OPEN_NEW)
                cnt["unchanged" if c["r"] is c["w"] else ("fixed" if at_scale(c["p"], Q, U) else "WRONG")] += 1
            check(f"noisy daily volume, split {ratio}, lognormal {sd}, 100 runs: never WRONG", cnt["WRONG"] == 0, str(cnt))
    good = {"bars": [{"date": d, "close": Q[i], "volume": U[i]} for i, d in enumerate(DATES)]}
    bad = {"bad JSON": "{not json", "no bars key": json.dumps({"symbol": "X"}), "a list": json.dumps(good["bars"]),
           "empty bars": json.dumps({"bars": []}),
           "missing close and volume": json.dumps({"bars": [{"date": d} for d in DATES]}),
           "None closes, zero volumes": json.dumps({"bars": [{"date": d, "close": None, "volume": 0} for d in DATES]}),
           "zero closes, None volumes": json.dumps({"bars": [{"date": d, "close": 0, "volume": None} for d in DATES]}),
           "text values": json.dumps({"bars": [{"date": d, "close": "x", "volume": "y"} for d in DATES]})}
    for lab, txt in bad.items():
        Fs = [3 if i < 15 else 1 for i in range(N)]
        write("MAL", [Q[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], draw=txt)
        try:
            c = run_case(O, R1, "MAL", op=OPEN_NEW)
            ok, why = unchanged(c), ""
        except Exception as e:     # noqa: BLE001
            ok, why = False, repr(e)
        check(f"synthetic malformed daily file ({lab}): unchanged, no exception", ok, why)
    dc, dv = list(Q), list(U)
    dc[8], dv[22] = None, 0
    Fs = [3 if i < 15 else 1 for i in range(N)]
    write("PART", [Q[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], dc, dv)
    c = run_case(O, R1, "PART", op=OPEN_NEW)
    check("synthetic 3-for-1 with a None close and a zero volume on other days: still found", fixed(c))
    write("B30NONE", [Q[i] * Fs[i] for i in range(N)], [U[i] / Fs[i] for i in range(N)], Q, U)
    doc = json.loads((F30 / "B30NONE.json").read_text())
    doc["bars"][13 * 10 + 4]["volume"] = None
    (F30 / "B30NONE.json").write_text(json.dumps(doc))
    lab = "synthetic stored bar with volume None before the split: no exception, split still found, the None kept"
    try:
        c = run_case(O, R1, "B30NONE", op=OPEN_NEW)
        hp, d0, sl0 = c["p"], DATES[10], I.SLOTS[4]
        okp = all(abs(bb[max(bb)]["close"] / Q[DI[d]] - 1) < 1e-9 for d, bb in hp.items())
        okv = all(abs(sum(b["volume"] for b in bb.values()) / U[DI[d]] - 1) < 1e-9 for d, bb in hp.items() if d != d0)
        check(lab, c["r"] is not c["w"] and c["nm"] and okp and okv and hp[d0][sl0]["volume"] is None and "error" not in c["seen"])
    except Exception as e:     # noqa: BLE001
        check(lab, False, repr(e))
    doc = json.loads((F30 / "B30NONE.json").read_text())
    doc["bars"][13 * 15]["open"] = None                       # the ex-date's first bar has no open: its step is unknown
    (F30 / "B30NONE.json").write_text(json.dumps(doc))
    try:
        c = run_case(O, R1, "B30NONE", op=OPEN_NEW)
        ok, why = unchanged(c) and c["seen"]["rejected"][0][3] == "no as-traded jump", str(c["seen"]["rejected"])
    except Exception as e:     # noqa: BLE001
        ok, why = False, repr(e)
    check("synthetic ex-date bar with open None (step unknown): rejected, unchanged, no exception", ok, why)
    split_case("AB C", {15: 3})
    c = run_case(O, R1, "AB C", op=OPEN_NEW)
    check("synthetic symbol with a space (AB C -> AB_C.json): split found", fixed(c)
          and (F30 / "AB_C.json").exists() and (FD / "AB_C.json").exists())
    set30(REAL30, I, O, R1)
    setd(REALD, I, R1)


# ---------------------------------------------------------------- real files (read only)
def standin_daily(sym, folder):
    """The real daily file plus a bar for 20261120 made from the 30-minute session (as the Monday 07:30 refresh would
    add one on the current scale). A stand-in: IBKR's real 20 Nov daily bar may differ a little."""
    doc = json.loads((REALD / f"{sym.replace(' ', '_')}.json").read_text())
    s = [b for b in json.loads((REAL30 / f"{sym.replace(' ', '_')}.json").read_text())["bars"] if b["date"][:8] == "20261120"]
    if s and doc["bars"][-1]["date"][:8] < "20261120":
        doc["bars"].append({"date": "20261120", "open": s[0]["open"], "high": max(b["high"] for b in s),
                            "low": min(b["low"] for b in s), "close": s[-1]["close"], "volume": sum(b["volume"] for b in s)})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{sym.replace(' ', '_')}.json").write_text(json.dumps(doc), encoding="utf-8")


def asof_daily(syms, folder, before):
    """The real daily files cut to bars dated before `before`: the daily cache as a look on that day had it (the 07:30
    refresh brings the previous session's bar). Used by: case (1)'s WITHOUT leg, the pinned before=20261120 check and
    case (2) in real(), missing_more's BITO gateway-month case, and the e2e-names case of fallback_r4's no-retry
    comparison; those do not depend on what the daily folder in use holds for 20 Nov and later. Not every 20 Nov
    replay uses it: the e2e leg 'replay Fri 20 Nov (current daily cache)', the other missing_session replays and the
    other six cases of that comparison read the daily folder in use (REALD), which after Monday's refresh holds a
    20 Nov bar the Friday looks never had; the round 4 test review ran them on today's data/bars, the stand-in and two
    approximations of Monday's folder, and they passed in all four."""
    folder.mkdir(parents=True, exist_ok=True)
    for sym in syms:
        f = f"{sym.replace(' ', '_')}.json"
        doc = json.loads((REALD / f).read_text())
        doc["bars"] = [b for b in doc["bars"] if str(b["date"]).replace("-", "")[:8] < before]
        (folder / f).write_text(json.dumps(doc), encoding="utf-8")
    return folder


REAL_DATES = ["20261017", "20261105", "20261106", "20261119", "20261120", "20261121", "20261123"]


def bito_ok(ho, hp):
    pre = [d for d in ho if d < "20261120"]
    ok_b = all(hp[d][sl][k] == ho[d][sl][k] * 2 for d in pre for sl in ho[d] for k in ("open", "high", "low", "close"))
    ok_v = all(hp[d][sl]["volume"] == ho[d][sl]["volume"] * 0.5 for d in pre for sl in ho[d])
    return ok_b and ok_v and hp.get("20261120") == ho.get("20261120") and "20261119" not in ho and len(pre) == 24


def real(O, R1):
    set30(REAL30, I, O, R1)
    setd(REALD, I, R1)
    names = sorted(p.stem.replace("_", " ") for p in REAL30.glob("*.json"))
    check(f"real files: {len(names)} names read", len(names) == 523)
    for b in REAL_DATES:
        ch, ch1, same_obj, errs, r1_same = [], [], 0, [], True
        for s in names:
            w = O._cached_history(s, b)
            h1 = R1._cached_history(s, b)
            hp = I._cached_history(s, b)
            if hp != w:
                ch.append(s)
            else:
                same_obj += I._one_scale(s, w, before=b) is w
            if h1 != w:
                ch1.append(s)
            if "error" in I._SCALE_SEEN.get(s, {}):
                errs.append(s)
            if b < "20261121" and hp != h1:
                r1_same = False
        print(f"      before={b}: v2 changes {ch}; round 1 changed {ch1}; scale errors {errs[:5]} ({len(errs)})")
        check(f"real files before={b}: every other name gives back the very same objects",
              same_obj == len(names) - len(ch), f"{same_obj} of {len(names) - len(ch)}")
        if b < "20261121":
            check(f"real files before={b}: v2 gives exactly what round 1 gave for all {len(names)} names", r1_same and ch == ch1)
        else:
            check(f"real files before={b}: only BITO and SOXX change (daily cache has 20 Nov: {BITO20})",
                  ch == ["BITO", "SOXX"], f"changed {ch}")
    ho, hp = O._cached_history("SOXX", "20261121"), I._cached_history("SOXX", "20261121")
    ok_b = all(hp[d][sl][k] == ho[d][sl][k] / 3 for d in ho if d < "20261105" for sl in ho[d]
               for k in ("open", "high", "low", "close"))
    ok_v = all(hp[d][sl]["volume"] == ho[d][sl]["volume"] * 3 for d in ho if d < "20261105" for sl in ho[d])
    ok_a = all(hp[d] == ho[d] for d in ho if d >= "20261105")
    nb = sum(d < "20261105" for d in ho)
    check(f"real SOXX: {nb} sessions before 5 Nov have prices / 3 and volume * 3; later ones unchanged",
          ok_b and ok_v and ok_a and hp == R1._cached_history("SOXX", "20261121"))
    led = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
    row = [x for x in led if x["symbol"] == "SOXX"]
    b = I._SCALE_SEEN["SOXX"]["boundaries"]
    check("real SOXX agrees with the desk ledger (ratio 3, ex 20261105)", row and row[0]["ratio"] == 3
          and row[0]["ex_date"] == "20261105" and b == [["20261104", "20261105", 3, "20261105"]],
          f"ledger {[(x['ex_date'], x['ratio']) for x in row]}, detected {b}")
    sn = [x for x in led if x["symbol"] == "SNDK"]
    check("data fact, not a test of the patch: SNDK (ledger 8-for-1, ex 20261120) has no 30-minute file",
          sn and I._cached_history("SNDK", "20261121") == {})
    # case (1): BITO with and without a 20 Nov daily bar, before = Sat 21 and Mon 23 Nov. The WITHOUT leg is the daily
    # folder in use cut before 20 Nov (asof_daily), so it exists whatever that folder holds; the WITH leg adds a 20 Nov
    # bar to it (standin_daily: a copy when it already has one); the third leg is the folder in use as it is
    sd = FIX / "standin_bars"
    standin_daily("BITO", sd)
    no20 = asof_daily(["BITO"], FIX / "daily_asof_20261120", "20261120")
    for lab, folder in (("without a 20 Nov daily bar (daily folder in use cut before 20 Nov)", no20),
                        ("with a 20 Nov daily bar (stand-in)", sd),
                        (f"daily folder in use as it is ({'has' if BITO20 else 'no'} 20 Nov bar)", REALD)):
        for bf in ("20261121", "20261123"):
            I.DAILY = folder
            ho, hp = O._cached_history("BITO", bf), I._cached_history("BITO", bf)
            seen = dict(I._SCALE_SEEN["BITO"])
            check(f"case (1) real BITO, {lab}, before={bf}: sessions through 18 Nov prices * 2, volume * 0.5; 20 Nov "
                  "unchanged", bito_ok(ho, hp), f"boundaries {seen.get('boundaries')}, level {seen.get('level')}")
    # Fri 20 Nov's morning, pinned: the daily cache cut before 20 Nov (it then ended 19 Nov). It would also pass on a
    # folder with the 20 Nov bar, for another reason (no level step: the cache's newest is not before 20 Nov)
    I.DAILY = no20
    ho = O._cached_history("BITO", "20261120")
    hp = I._cached_history("BITO", "20261120")
    rej = I._SCALE_SEEN["BITO"].get("rejected")
    check("real BITO before=20261120 without today's open, daily cache as of that morning: unchanged (the level step "
          "has no stored session after 18 Nov and no open)", hp is not ho and hp == ho
          and I._one_scale("BITO", ho, before="20261120") is ho, f"rejected {rej}")
    I.DAILY = no20                                                              # the daily cache as Friday's looks had it
    hp = I._cached_history("BITO", "20261120", open_today=16.48)
    I.DAILY = REALD
    check("case (2) real BITO on Fri 20 Nov with its open 16.48 (daily cache as of that morning): sessions through 18 Nov "
          "prices * 2, volume * 0.5",
          all(hp[d][sl]["close"] == ho[d][sl]["close"] * 2 and hp[d][sl]["volume"] == ho[d][sl]["volume"] * 0.5
              for d in ho for sl in ho[d]) and I._SCALE_SEEN["BITO"]["level"] == ["20261118", 0.5, I.TODAY])


# ---------------------------------------------------------------- load_panel into this folder
def panel(O, R1):
    set30(REAL30, I, O, R1)
    setd(REALD, I, R1)
    I.CACHE, O.CACHE, R1.CACHE = WORK / "panel_v2.npz", WORK / "panel_orig.npz", WORK / "panel_round1.npz"
    import bars as daily_bars                     # load_panel's trader.Data() reads daily bars through bars.DATA
    data0 = daily_bars.DATA
    daily_bars.DATA = REALD                       # the daily folder in use (data/bars, or the --daily stand-in)
    try:
        t0 = time.time()
        po = O.load_panel(refresh=True)
        t1 = time.time()
        p1 = R1.load_panel(refresh=True)
        t2 = time.time()
        pp = I.load_panel(refresh=True)
        t3 = time.time()
    finally:
        daily_bars.DATA = data0
    I.CACHE, O.CACHE, R1.CACHE = SHIPPED[0], SHIPPED[0], SHIPPED[0]
    print(f"      load_panel's daily bars (trader.Data via bars.DATA) read from {REALD}")
    print(f"      load_panel: original {t1 - t0:.1f} s, round 1 {t2 - t1:.1f} s, v2 {t3 - t2:.1f} s (into {WORK})")
    syms, dates = [str(s) for s in pp["symbols"]], [str(x) for x in pp["dates"]]
    eq = lambda a, b: np.array_equal(a, b, equal_nan=True)
    diff = {k: [syms[j] for j in range(len(syms)) if not eq(po[k][j], pp[k][j])] for k in ("o", "h", "l", "c", "v", "div", "iv")}
    print("      panel rows that differ from the unpatched, by array:", json.dumps(diff))
    check("panel: v2 identical to round 1 in every array", set(p1) == set(pp) and all(
        eq(p1[k], pp[k]) if p1[k].dtype.kind == "f" else np.array_equal(p1[k], pp[k]) for k in pp))
    check("panel: same symbols and dates", list(po["symbols"]) == list(pp["symbols"]) and list(po["dates"]) == list(pp["dates"]))
    ohlcv = set().union(*[diff[k] for k in "ohlcv"])
    allow = {"SOXX", "BRK B", "BF B"} | ({"BITO"} if BITO20 else set())
    check("panel: OHLCV differs only for " + ", ".join(sorted(allow)), ohlcv <= allow and "SOXX" in ohlcv
          and (("BITO" in ohlcv) == BITO20), f"{sorted(ohlcv)}")
    j = syms.index("SOXX")
    pre = np.array([d < "20261105" for d in dates])
    ok = all(eq(pp[k][j][pre], po[k][j][pre] / 3) for k in "ohlc") and eq(pp["v"][j][pre], po["v"][j][pre] * 3) \
        and all(eq(pp[k][j][~pre], po[k][j][~pre]) for k in "ohlcv")
    check("panel SOXX: sessions before 5 Nov prices / 3, volume * 3; later unchanged", ok)
    for s in ("BRK B", "BF B"):
        j = syms.index(s)
        check(f"panel {s}: was all-NaN, now read from {s.replace(' ', '_')}.json (30-minute and iv)",
              np.isnan(po["c"][j]).all() and np.isfinite(pp["c"][j]).sum() > 1000 and np.isfinite(pp["iv"][j]).sum() > 100)
    js, i5 = syms.index("SOXX"), dates.index("20261105")
    check("panel SOXX: no fake dividend on 5 Nov any more", po["div"][js, i5] != 0 and pp["div"][js, i5] == 0)
    jb = syms.index("BITO")
    if not BITO20:
        check("panel BITO unchanged (daily cache has no 20 Nov bar yet; no level step in load_panel)", "BITO" not in ohlcv)
    else:
        pre = np.array([d < "20261119" for d in dates])
        ok = all(eq(pp[k][jb][pre], po[k][jb][pre] * 2) for k in "ohlc") and eq(pp["v"][jb][pre], po["v"][jb][pre] * 0.5) \
            and all(eq(pp[k][jb][~pre], po[k][jb][~pre]) for k in "ohlcv")
        check("panel BITO (daily cache has 20 Nov): sessions before 19 Nov prices * 2, volume * 0.5; later unchanged", ok)


# ---------------------------------------------------------------- fake gateways
class FakeGW:
    """gw.bars('1 D') = the stored session `src` relabelled as `today` (prices * day_k, volume / day_k). '1 M': mode
    'raise' (no gateway history), 'serve' (the real file's sessions through today), 'stale' (through the session before
    `cut`), 'few' / 'nine' / 'ten' (the last 5 / 9 / 10 sessions before today), 'empty' ([]), 'none' (None), 'flaky'
    (raises at its first '1 M' call, then serves). Answers that arrive but cannot be read or used (round 4): 'nodate'
    (one row without 'date'), 'notdicts' (a list of text), 'adict' (a dict, not a list), 'zeroclose' (every close 0:
    the scale comparison raises), 'textvol' (every volume the text 'x'). month_rows {sym: rows} replaces the real
    file's rows for '1 M'; modes {sym: mode} sets one name's mode; seq {sym: [mode, ...]} sets the mode of that name's
    n-th '1 M' call (the last one repeats). ZZZZ is AAPL's bars."""

    def __init__(self, src, today, mode="raise", cut=None, month_rows=None, day_k=1.0, modes=None, seq=None):
        self.src, self.today, self.mode, self.cut, self.calls = src, today, mode, cut, []
        self.month_rows, self.day_k, self.nm = month_rows or {}, day_k, 0
        self.modes, self.seq, self.nms = modes or {}, seq or {}, {}
        self.kws = []                              # (symbol, duration, extra keyword arguments) of every bars call

    def _rows(self, symbol):
        sym = "AAPL" if symbol == "ZZZZ" else symbol
        return json.loads((REAL30 / f"{sym.replace(' ', '_')}.json").read_text())["bars"]

    def bars(self, symbol, duration="1 D", bar_size="30 mins", what="TRADES", rth=True, **kw):
        self.calls.append((symbol, duration))
        self.kws.append((symbol, duration, dict(kw)))
        rows = self._rows(symbol)
        if duration == "1 D":
            k = self.day_k
            return [{**r, "date": self.today + r["date"][8:], "open": r["open"] * k, "high": r["high"] * k,
                     "low": r["low"] * k, "close": r["close"] * k, "volume": r["volume"] / k}
                    for r in rows if r["date"][:8] == self.src]
        if symbol == "ZZZZ":
            return [r for r in rows if "20261016" <= r["date"][:8] < self.today]
        self.nm += 1
        n = self.nms[symbol] = self.nms.get(symbol, 0) + 1
        mode = self.seq[symbol][min(n, len(self.seq[symbol])) - 1] if symbol in self.seq else \
            self.modes.get(symbol, self.mode)
        if mode == "raise" or (mode == "flaky" and self.nm == 1):
            raise ValueError("fake gateway: only today's bars")
        if mode == "empty":
            return []
        if mode == "none":
            return None
        if mode == "notdicts":
            return ["x"] * 13
        rows = self.month_rows.get(symbol, rows)
        month = [r for r in rows if "20261016" <= r["date"][:8] < self.today]
        if mode == "stale":
            month = [r for r in month if r["date"][:8] < self.cut]
        if mode in ("few", "nine", "ten"):
            keep = sorted({r["date"][:8] for r in month})[-{"few": 5, "nine": 9, "ten": 10}[mode]:]
            month = [r for r in month if r["date"][:8] in keep]
        if mode == "adict":
            return {"bars": month}
        if mode == "nodate":
            month = [dict(r) for r in month]
            del month[len(month) // 2]["date"]
        if mode == "zeroclose":
            month = [{**r, "close": 0} for r in month]
        if mode == "textvol":
            month = [{**r, "volume": "x"} for r in month]
        return month

    def stats(self, symbols, **kw):
        return [{"symbol": s, "implied_vol": 0.30, "hist_vol": 0.25} for s in symbols]

    def quotes(self, symbols, **kw):
        return []


def looks(today):
    y, m, d = int(today[:4]), int(today[4:6]), int(today[6:])
    return [datetime(y, m, d, 9, 35, tzinfo=I.ET) + timedelta(minutes=30 * k) for k in range(I.NS)]


def replay(mod, syms, src, today, mode="raise", cut=None, gw=None):
    """live() at all 13 looks; one fake gateway for the whole day unless the caller passes one. Returns (looks' rows,
    the gateway's non-'1 D' calls)."""
    gw = gw or FakeGW(src, today, mode, cut)
    out = []
    for now in looks(today):
        r = mod.live(gw, syms, now=now)
        out.append((r.get("look"), r.get("rows")))
    return out, [c for c in gw.calls if c[1] != "1 D"]


def js(x):
    return json.dumps(x, default=str)


# ---------------------------------------------------------------- B: a name whose cache lacks the latest session
def missing_session(O, R1):
    fb = FIX / "b30_missing"                       # AAPL without its 19 Nov session; SPY and MSFT as stored
    fb.mkdir(parents=True, exist_ok=True)
    for s in ("SPY", "MSFT", "AAPL"):
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s == "AAPL":
            rows = [r for r in rows if r["date"][:8] != "20261119"]
        (fb / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    syms = ["AAPL", "MSFT", "SPY", "ZZZZ"]
    day = "20261120"
    setd(REALD, I, R1)
    set30(REAL30, O)
    full, _ = replay(O, syms, day, day, "serve")                 # unpatched on the complete file
    set30(fb, I, O, R1)
    held, _ = replay(O, syms, day, day, "serve")                 # unpatched on the file missing 19 Nov
    r1, c1 = replay(R1, syms, day, day, "serve")
    reset_b()
    gw = FakeGW(day, day, "serve")
    v2, cv = replay(I, syms, day, day, gw=gw)
    v2b, cvb = replay(I, syms, day, day, gw=gw)                  # a second set of 13 calls on the same day (same gw:
    #                                                              cvb holds both sets' calls)
    a_ok = all(v2[k][1]["AAPL"] == {**full[k][1]["AAPL"], "history": "gateway"} for k in range(I.NS))
    has = all(v2[k][1]["AAPL"].get("rest_of_day_range_pct") is not None and v2[k][1]["AAPL"].get("next60_range_pct")
              is not None for k in range(I.NS))
    before = sum(held[k][1]["AAPL"].get("rest_of_day_range_pct") is None for k in range(I.NS))
    check("B: AAPL missing 19 Nov gets next60 and rest_of_day at all 13 looks, history 'gateway', equal to the unpatched "
          "run on the complete file", a_ok and has, f"unpatched on the gappy file: {before} of 13 looks without a range")
    check("B: round 1 on the gappy file = the unpatched on it (no range)", js(r1) == js(held) and before == 13)
    check("B: the gateway month asked once for AAPL across 26 live() calls on the same day; never for MSFT or SPY",
          [c for c in cvb if c[0] != "ZZZZ"] == [("AAPL", "1 M")], f"calls other than ZZZZ's (the old branch), both "
          f"sets: {[c for c in cvb if c[0] != 'ZZZZ']}")
    check("B: MSFT, SPY and ZZZZ (no cache file: the old gateway branch) identical to the unpatched", all(
        js(v2[k][1][s]) == js(held[k][1][s]) for k in range(I.NS) for s in ("MSFT", "SPY", "ZZZZ"))
        and v2[0][1]["ZZZZ"]["history"] == "gateway")
    check("B: the second day of 13 calls (served from the memo) gives AAPL the same rows as the unpatched on the complete "
          "file", all(v2b[k][1]["AAPL"] == {**full[k][1]["AAPL"], "history": "gateway"} for k in range(I.NS)))
    for mode, cut, lab in (("raise", None, "the gateway raising"), ("stale", "20261119", "the gateway with nothing newer"),
                           ("few", None, "the gateway with fewer than 10 sessions"),
                           ("nine", None, "the gateway with 9 sessions"), ("empty", None, "the gateway returning []"),
                           ("none", None, "the gateway returning None")):
        reset_b()
        v, cl = replay(I, syms, day, day, mode, cut)
        ok = all(js(v[k][1]) == js(held[k][1]) for k in range(I.NS))
        n_a = sum(c == ("AAPL", "1 M") for c in cl)
        want = 13 if mode == "raise" else 1         # an ask that raised is repeated after 30 minutes of look time (looks
        #                                             are 30 minutes apart); an answer that cannot be used, not that day
        check(f"B: {lab}: every row identical to the unpatched on the gappy file (cache kept); asked "
              + ("once a look" if mode == "raise" else "once that day") + f" ({n_a} asks in 13 looks)",
              ok and n_a == want and not I._GW_HISTORY)
    fs = FIX / "b30_spy_missing"                    # SPY itself without 19 Nov: ref = 18 Nov, nothing asked
    fs.mkdir(parents=True, exist_ok=True)
    for s in ("SPY", "MSFT", "AAPL"):
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s == "SPY":
            rows = [r for r in rows if r["date"][:8] != "20261119"]
        (fs / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    set30(fs, I, O)
    reset_b()
    ho, co = replay(O, syms[:3], day, day, "serve")
    v, cl = replay(I, syms[:3], day, day, "serve")
    check("B: SPY's cache missing 19 Nov (ref = 18 Nov): no gateway month for anyone, rows identical to the unpatched",
          cl == [] and co == [] and js(v) == js(ho))
    set30(REAL30, I, O, R1)
    missing_more(O, R1, fb, held, full)


def reset_b():
    I._GW_HISTORY.clear()
    I._GW_FAILED.clear()


def missing_more(O, R1, fb, held, full):
    """More of B: the reviewers' gaps (round 2 review_live SHOULD FIX 3 and 4, SHOULD FIX 1 and 2)."""
    day, syms = "20261120", ["AAPL", "MSFT", "SPY", "ZZZZ"]
    now = looks(day)[5]
    set30(fb, I, O, R1)
    setd(REALD, I, R1)
    reset_b()
    gw = FakeGW(day, day, "serve")
    h, src = I._history(gw, "SPY", day, now, ref="20991231")
    check("B: SPY itself with a later ref: the sym != 'SPY' guard holds, cache, no gateway call", src == "cache"
          and not [c for c in gw.calls if c[1] != "1 D"])
    h, src = I._history(gw, "AAPL", day, now)
    check("B: _history(gw, sym, today, now) with no ref (old signature): cache, no gateway call", src == "cache"
          and max(h) == "20261118" and gw.calls == [])
    reset_b()
    v, cl = replay(I, syms, day, day, "ten")
    check("B: a month of exactly 10 sessions with 19 Nov: used ('gateway' at all 13 looks), asked once",
          all(v[k][1]["AAPL"]["history"] == "gateway" for k in range(I.NS)) and cl.count(("AAPL", "1 M")) == 1)
    reset_b()
    v, cl = replay(I, syms, day, day, "flaky")
    check("B: a failure at the first look, success at the next: 2 calls; look 1 = the unpatched on the gappy file, "
          "looks 2-13 'gateway'", cl.count(("AAPL", "1 M")) == 2 and js(v[0][1]["AAPL"]) == js(held[0][1]["AAPL"])
          and all(v[k][1]["AAPL"] == {**full[k][1]["AAPL"], "history": "gateway"} for k in range(1, I.NS)))
    reset_b()
    gw = FakeGW(day, day, "raise")
    for _ in range(3):
        I.live(gw, syms, now=now)
    check("B: three live() calls at one look after an ask that raised ask once",
          [c for c in gw.calls if c == ("AAPL", "1 M")] == [("AAPL", "1 M")])
    I.live(gw, syms, now=now + timedelta(minutes=15))
    check("B: not asked again 15 minutes later (the 15-minute sleeves' next look)",
          [c for c in gw.calls if c == ("AAPL", "1 M")] == [("AAPL", "1 M")])
    I.live(gw, syms, now=now + timedelta(minutes=30))
    check("B: asked again 30 minutes later", [c for c in gw.calls if c == ("AAPL", "1 M")] == [("AAPL", "1 M")] * 2)
    check("B: the fallback's ask carries timeout=10 s; today's bars and the old branch (ZZZZ, no cache file) carry none",
          all(kw == {"timeout": I.GW_FILL_TIMEOUT_S} for s, d, kw in gw.kws if (s, d) == ("AAPL", "1 M"))
          and I.GW_FILL_TIMEOUT_S == 10.0 and any((s, d) == ("ZZZZ", "1 M") for s, d, kw in gw.kws)
          and all(kw == {} for s, d, kw in gw.kws if (s, d) != ("AAPL", "1 M")))
    # the cap: five names missing 19 Nov in one live() call ask for three months; the other two keep the cache then
    fc = FIX / "b30_cap"
    fc.mkdir(parents=True, exist_ok=True)
    five = ["AAPL", "MSFT", "NVDA", "XOM", "KO"]
    for s in ["SPY"] + five:
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s != "SPY":
            rows = [r for r in rows if r["date"][:8] != "20261119"]
        (fc / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    set30(fc, I, O, R1)
    reset_b()
    gw = FakeGW(day, day, "serve")
    ro = O.live(FakeGW(day, day, "serve"), five, now=now)["rows"]          # unpatched on the gappy files: no range
    set30(REAL30, O)
    rf = O.live(FakeGW(day, day, "serve"), five, now=now)["rows"]          # unpatched on the complete files
    set30(fc, O)
    a = I.live(gw, five, now=now)["rows"]
    asked1 = [c[0] for c in gw.calls if c[1] == "1 M"]
    b = I.live(gw, five, now=now)["rows"]
    asked2 = [c[0] for c in gw.calls if c[1] == "1 M"]
    check("B: the cap: a call naming five names missing 19 Nov asks for three months (GW_FILL_MAX), in the call's order",
          asked1 == five[:3] and I.GW_FILL_MAX == 3, f"asked {asked1}")
    check("B: the cap: the three asked get 'gateway' rows equal to the unpatched on the complete files; the two past the "
          "cap keep the cache, rows identical to the unpatched on the gappy files",
          all(a[s] == {**rf[s], "history": "gateway"} for s in five[:3]) and all(js(a[s]) == js(ro[s]) for s in five[3:]))
    check("B: the cap: the next call asks for the other two and all five are 'gateway'", asked2 == five
          and all(b[s] == {**rf[s], "history": "gateway"} for s in five))
    # the switch: GW_FILL off gives round 1's output and calls on the gappy files
    reset_b()
    I.GW_FILL = False
    try:
        v, cv = replay(I, five, day, day, "serve")
    finally:
        I.GW_FILL = True
    r1, c1 = replay(R1, five, day, day, "serve")
    check("B: the switch: with GW_FILL off, output and calls identical to round 1 on the gappy files (no month asked)",
          js(v) == js(r1) and cv == c1 == [])
    fn = FIX / "b30_missing2"                       # AAPL without 19 and 20 Nov: missing on two days
    fn.mkdir(parents=True, exist_ok=True)
    for s in ("SPY", "MSFT", "AAPL"):
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s == "AAPL":
            rows = [r for r in rows if r["date"][:8] not in ("20261119", "20261120")]
        (fn / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    set30(fn, I)
    reset_b()
    gw = FakeGW(day, day, "serve")
    a, _ = replay(I, syms, day, day, gw=gw)
    gw.today = "20261123"
    b, _ = replay(I, syms, day, "20261123", gw=gw)
    check("B: the next day asks again (AAPL missing 19 and 20 Nov: one call on 20 Nov, one on 23 Nov), 'gateway' both days",
          [c for c in gw.calls if c == ("AAPL", "1 M")] == [("AAPL", "1 M")] * 2 and a[3][1]["AAPL"]["history"] == "gateway"
          and b[3][1]["AAPL"]["history"] == "gateway")
    # round 3's review: four more rules of the memo and the cap, by direct _history calls on the same gappy file
    ref20, ref23 = max(I._cached_history("SPY", day)), max(I._cached_history("SPY", "20261123"))
    t23, one = looks("20261123")[5], ("AAPL", "1 M")

    class NoTimeoutGW(FakeGW):                      # a gateway whose bars() takes no timeout keyword
        def bars(self, symbol, duration="1 D", bar_size="30 mins", what="TRADES", rth=True):
            return FakeGW.bars(self, symbol, duration, bar_size, what, rth)

    reset_b()
    h, src = I._history(NoTimeoutGW(day, day, "serve"), "AAPL", day, now, ref=ref20)
    check("B: a gateway whose bars() takes no timeout keyword: nothing raises, the cache is kept, remembered as an ask "
          "that raised", src == "cache" and max(h) == "20261118"
          and I._GW_FAILED.get(("AAPL", day)) == now.replace(tzinfo=None) and not I._GW_HISTORY)
    reset_b()
    gw, left = FakeGW(day, day, "few"), [2]
    I._history(gw, "AAPL", day, now, ref=ref20, asks=left)                                  # answered, not usable
    I._history(gw, "AAPL", day, now + timedelta(minutes=60), ref=ref20, asks=left)          # may not ask today
    n20, left20 = gw.calls.count(one), list(left)
    gw.today = "20261123"
    I._history(gw, "AAPL", "20261123", t23, ref=ref23, asks=left)
    check("B: an answer that cannot be used is asked for once that day; a call that may not ask does not use the cap; "
          "the next day asks again", n20 == 1 and left20 == [1] and gw.calls.count(one) == 2 and left == [0])
    reset_b()
    gw = FakeGW(day, day, "raise")
    I._history(gw, "AAPL", day, now, ref=ref20)
    I._history(gw, "AAPL", day, now + timedelta(minutes=10), ref=ref20)                     # too soon
    n_soon = gw.calls.count(one)
    I._history(gw, "AAPL", day, now - timedelta(minutes=30), ref=ref20)                     # an earlier look time
    check("B: after an ask that raised, a call 10 minutes later does not ask; a call at an earlier look time (a replay "
          "going back) does", n_soon == 1 and gw.calls.count(one) == 2)
    reset_b()
    fo = FIX / "b30_old"                            # AAPL's cache ends 5 Nov: the old (> 10 days) branch comes first
    fo.mkdir(parents=True, exist_ok=True)
    for s in ("SPY", "MSFT", "AAPL"):
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s == "AAPL":
            rows = [r for r in rows if r["date"][:8] <= "20261105"]
        (fo / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    set30(fo, I, O, R1)
    reset_b()
    ro, co = replay(O, syms, day, day, "serve")
    r1, c1 = replay(R1, syms, day, day, "serve")
    v, cv = replay(I, syms, day, day, "serve")
    check("B: a cache more than 10 days old takes the old branch (a '1 M' at every look), output and calls identical to "
          "round 1 and the unpatched", js(v) == js(r1) == js(ro) and cv == c1 == co and cv.count(("AAPL", "1 M")) == 13)
    fz = FIX / "b30_nospy"                          # SPY without a cache file: ref None
    fz.mkdir(parents=True, exist_ok=True)
    for s in ("MSFT", "AAPL"):
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s == "AAPL":
            rows = [r for r in rows if r["date"][:8] != "20261119"]
        (fz / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    if (fz / "SPY.json").exists():
        (fz / "SPY.json").unlink()
    set30(fz, I, R1)
    reset_b()
    r1, c1 = replay(R1, syms, day, day, "serve")
    v, cv = replay(I, syms, day, day, "serve")
    check("B: SPY without a cache file (ref None): output and call list identical to round 1", js(v) == js(r1) and cv == c1)
    # BITO on Fri 20 Nov with a gateway month holding 19 Nov: new scale used, old scale (not yet adjusted) refused
    set30(REAL30, I, O, R1)
    bito = json.loads((REAL30 / "BITO.json").read_text())["bars"]
    dl = {str(b["date"]).replace("-", "")[:8]: b for b in json.loads((REALD / "BITO.json").read_text())["bars"]}
    sc = lambda rr, k: [{**r, "open": r["open"] * k, "high": r["high"] * k, "low": r["low"] * k, "close": r["close"] * k,
                         "volume": r["volume"] / k} for r in rr]
    s18 = [r for r in bito if r["date"][:8] == "20261118"]
    pre = [r for r in bito if r["date"][:8] < "20261119"]
    new19 = [{**r, "date": "20261119" + r["date"][8:]} for r in sc(s18, 2.0 * dl["20261119"]["close"] / (2 * s18[-1]["close"]))]
    old19 = [{**r, "date": "20261119" + r["date"][8:]} for r in sc(s18, dl["20261119"]["close"] / 2 / s18[-1]["close"])]
    setd(asof_daily(["BITO", "AAPL", "SPY"], FIX / "daily_asof_20261120", "20261120"), I, R1)   # as Friday's looks had it
    for lab, mrows, want in (("new scale", sc(pre, 2.0) + new19, "gateway"), ("old scale (IBKR not yet adjusted)",
                                                                             pre + old19, "cache")):
        reset_b()
        v, cv = replay(I, ["BITO", "AAPL"], day, day, "serve", gw=FakeGW(day, day, "serve", month_rows={"BITO": mrows}))
        r1, _ = replay(R1, ["BITO", "AAPL"], day, day, "serve", gw=FakeGW(day, day, "serve", month_rows={"BITO": mrows}))
        hs = [v[k][1]["BITO"].get("history") for k in range(I.NS)]
        rg = [v[k][1]["BITO"].get("rest_of_day_range_pct") for k in range(I.NS)]
        ok = all(x == want for x in hs) and (all(x is not None for x in rg) if want == "gateway" else
                                             all(x is None for x in rg) and js(v) == js(r1))
        check(f"B: real BITO 20 Nov, gateway month with a {lab} 19 Nov: history '{want}'"
              + (", ranges at all 13 looks" if want == "gateway" else ", rows identical to round 1 (no range)"), ok,
              f"history {hs[:2]}, rest_of_day {rg[:2]}")
    setd(REALD, I, R1)
    fa = FIX / "b30_missing"                        # AAPL missing 19 Nov; a 2-for-1 ex today, month not yet adjusted
    set30(fa, I)
    reset_b()
    v, cv = replay(I, ["AAPL"], day, day, "serve", gw=FakeGW(day, day, "serve", day_k=0.5))
    check("B: today's open half the gateway month's last close (a 2-for-1 ex today, month not adjusted): month refused, "
          "history 'cache'", all(v[k][1]["AAPL"].get("history") == "cache" for k in range(I.NS)))
    set30(REAL30, I, O, R1)


def gappy(folder, drop, base=("SPY",), keep_last=None):
    """A bars30 folder: SPY (and `base`) as stored; each name in `drop` without its 19 Nov session (keep_last {name: n}:
    only its last n full sessions before 19 Nov, so its cache has fewer than 10)."""
    folder.mkdir(parents=True, exist_ok=True)
    for s in list(base) + list(drop):
        rows = json.loads((REAL30 / f"{s}.json").read_text())["bars"]
        if s in drop:
            rows = [r for r in rows if r["date"][:8] != "20261119"]
        if keep_last and s in keep_last:
            ds = sorted({r["date"][:8] for r in rows if r["date"][:8] < "20261119"})[-keep_last[s]:]
            rows = [r for r in rows if r["date"][:8] in ds or r["date"][:8] >= "20261120"]
        (folder / f"{s}.json").write_text(json.dumps({"symbol": s, "what": "TRADES", "bars": rows}))
    return folder


def fallback_r4(O):
    """Round 4: (A) an answer that arrives but cannot be read or used is remembered as None and asked once that day; only
    a request that raises is retried after GW_FILL_RETRY_MIN. (B) inside one live() call, names never asked today take
    the asks first (call's order), retries what is left; with no retry candidate, rows and calls as before round 4."""
    day, syms, one = "20261120", ["AAPL", "MSFT", "SPY", "ZZZZ"], ("AAPL", "1 M")
    fb = gappy(FIX / "b30_missing", ["AAPL"], base=("SPY", "MSFT"))
    setd(REALD, I, O)
    set30(fb, I, O)
    held, _ = replay(O, syms, day, day, "serve")                     # the cache path: unpatched on the gappy file
    key = ("AAPL", day)
    # ---- A: five kinds of answer that arrive but cannot be read or used, each over a whole day of 13 looks
    for mode, lab in (("nodate", "a row without 'date'"), ("notdicts", "a list of text, not bar dicts"),
                      ("adict", "a dict, not a list"), ("zeroclose", "every close 0 (the scale comparison raises)"),
                      ("textvol", "every volume the text 'x' (the panel cannot hold it)")):
        reset_b()
        try:
            v, cl = replay(I, syms, day, day, mode)
            why = ""
        except Exception as e:     # noqa: BLE001
            v, cl, why = None, [], repr(e)
        n_a = cl.count(one)
        ok = v is not None and all(js(v[k][1]) == js(held[k][1]) for k in range(I.NS)) and n_a == 1 \
            and key in I._GW_FAILED and I._GW_FAILED[key] is None and not I._GW_HISTORY
        check(f"A: an answer with {lab}: remembered as None, asked once in 13 looks, every row identical to the cache "
              f"path's, nothing escapes ({n_a} asks)", ok, why)
    # ---- A (final round 4): two answers that are readable but not usable, which the before module used and lost the
    # range with: one infinite high mid-month, and a month whose 13 bars a day sit an hour early (not at the SLOTS)
    real_aapl = json.loads((REAL30 / "AAPL.json").read_text())["bars"]
    inf_rows = [dict(r) for r in real_aapl]
    j = next(i for i, r in enumerate(inf_rows) if r["date"].startswith("20261105 12:00"))
    inf_rows[j]["high"] = float("inf")

    def early(r):
        d, tm = r["date"].split()[:2]
        return {**r, "date": f"{d} {int(tm[:2]) - 1:02d}{tm[2:]} US/Eastern"}
    early_rows = [early(r) for r in real_aapl]
    for rows, lab in ((inf_rows, "one infinite high mid-month (5 Nov 12:00)"),
                      (early_rows, "13 bars a day an hour early, not at the session slots (08:30 .. 14:30)")):
        reset_b()
        try:
            gw = FakeGW(day, day, "serve", month_rows={"AAPL": rows})
            v, cl = replay(I, syms, day, day, gw=gw)
            why = ""
        except Exception as e:     # noqa: BLE001
            v, cl, why = None, [], repr(e)
        n_a = cl.count(one)
        ok = v is not None and all(js(v[k][1]) == js(held[k][1]) for k in range(I.NS)) and n_a == 1 \
            and I._GW_FAILED.get(key, "absent") is None and not I._GW_HISTORY
        check(f"A: a month with {lab}: not used, remembered as None, asked once in 13 looks, every row identical to the "
              f"cache path's ({n_a} asks)", ok, why or f"used {key in I._GW_HISTORY}")
    # ---- kept rule (passes on the module before round 4 too): a request that raises is retried after GW_FILL_RETRY_MIN
    ref20, t0 = max(I._cached_history("SPY", day)), looks(day)[5]
    reset_b()
    gw = FakeGW(day, day, "raise")
    I._history(gw, "AAPL", day, t0, ref=ref20)
    I._history(gw, "AAPL", day, t0 + timedelta(minutes=29), ref=ref20)
    n29 = gw.calls.count(one)
    I._history(gw, "AAPL", day, t0 + timedelta(minutes=30), ref=ref20)
    check("kept rule, not a test of change A: a request that raises is remembered with the look time; not asked 29 "
          "minutes later, asked again at 30",
          n29 == 1 and gw.calls.count(one) == 2 and I._GW_FAILED.get(key) == (t0 + timedelta(minutes=30)).replace(tzinfo=None)
          and I.GW_FILL_RETRY_MIN == 30)
    reset_b()
    gw = FakeGW(day, day, seq={"AAPL": ["raise", "nodate"]})
    for m in (0, 10, 30, 60, 90, 120):
        I.live(gw, syms, now=t0 + timedelta(minutes=m))
    check("A: a request that raised, retried 30 minutes later, gets an unreadable answer: no ask after that the same day "
          "(2 asks over 6 calls in 2 hours)", gw.calls.count(one) == 2 and I._GW_FAILED.get(key, "absent") is None,
          f"{gw.calls.count(one)} asks")
    # ---- B: order under the cap
    five = ["AAPL", "MSFT", "NVDA", "XOM", "KO"]
    fc = gappy(FIX / "b30_cap", five)
    set30(fc, I, O)
    t1 = t0 + timedelta(minutes=35)
    ro = O.live(FakeGW(day, day, "serve"), five, now=t1)["rows"]          # unpatched on the gappy files: no range
    set30(REAL30, O)
    rf = O.live(FakeGW(day, day, "serve"), five, now=t1)["rows"]          # unpatched on the complete files
    set30(fc, O)
    gwr = lambda s: {**rf[s], "history": "gateway"}
    reset_b()
    I.live(FakeGW(day, day, "raise"), five[:2], now=t0)                   # AAPL and MSFT asked at t0; both raised
    pre = dict(I._GW_FAILED)
    gw = FakeGW(day, day, "serve")
    a = I.live(gw, five, now=t1)["rows"]
    asked1 = [c[0] for c in gw.calls if c[1] == "1 M"]
    b = I.live(gw, five, now=t1)["rows"]
    asked2 = [c[0] for c in gw.calls if c[1] == "1 M"]
    t0n = t0.replace(tzinfo=None)
    check("B: five names lacking 19 Nov, the first two raised 35 minutes earlier: the three never asked take the three "
          "asks in the call's order; the two retries keep the cache (rows = the unpatched on the gappy files)",
          pre == {("AAPL", day): t0n, ("MSFT", day): t0n} and asked1 == five[2:]
          and all(a[s] == gwr(s) for s in five[2:]) and all(js(a[s]) == js(ro[s]) for s in five[:2]), f"asked {asked1}")
    check("B: the next call asks the two retries (AAPL, MSFT) and all five are 'gateway'",
          asked2 == five[2:] + five[:2] and all(b[s] == gwr(s) for s in five), f"asked {asked2}")
    three = ["AAPL", "MSFT", "NVDA"]
    reset_b()
    I.live(FakeGW(day, day, "raise"), three[:2], now=t0)
    gw = FakeGW(day, day, "serve")
    a = I.live(gw, three, now=t1)["rows"]
    asked = [c[0] for c in gw.calls if c[1] == "1 M"]
    check("B: two retries and one never-asked name fit the cap of 3: all asked at that call, the never-asked first "
          "(NVDA, AAPL, MSFT); the deferred retries end the call 'gateway' with ranges equal to the unpatched on the "
          "complete files", asked == ["NVDA", "AAPL", "MSFT"] and all(a[s] == gwr(s) for s in three)
          and all(a[s].get("rest_of_day_range_pct") is not None for s in three), f"asked {asked}")
    # a retry whose cache has only 8 sessions ('only 8 past sessions' at the first pass): the deferred month clears it
    f8 = gappy(FIX / "b30_short8", ["AAPL", "NVDA"], keep_last={"AAPL": 8})
    set30(f8, I)
    for mode, want in (("serve", "gateway"), ("few", "error")):
        reset_b()
        r0 = I.live(FakeGW(day, day, "raise"), ["AAPL"], now=t0)["rows"]["AAPL"]
        gw = FakeGW(day, day, "serve", modes={"AAPL": mode})
        a = I.live(gw, ["AAPL", "NVDA"], now=t1)["rows"]
        asked = [c[0] for c in gw.calls if c[1] == "1 M"]
        first = r0.get("error", "").startswith("only 8 past sessions")
        if want == "gateway":
            ok = first and asked == ["NVDA", "AAPL"] and a["AAPL"] == gwr("AAPL") and a["NVDA"] == gwr("NVDA")
            lab = "a usable month: AAPL 'gateway' with a range at that same call, the error cleared"
        else:
            ok = first and asked == ["NVDA", "AAPL"] and a["AAPL"] == r0 and I._GW_FAILED.get(key, "absent") is None
            lab = "an unusable month: AAPL keeps the error row, remembered as None"
        check(f"B: a deferred retry with an 8-session cache, asked after the never-asked NVDA; {lab}", ok,
              f"asked {asked}, first pass {r0.get('error')}, AAPL {a['AAPL'].get('history', a['AAPL'].get('error'))}")
    # today's open reaches the deferred ask: on a day whose open is a split-sized jump from the month (day_k 0.5), a
    # deferred retry is refused exactly as a first-pass ask is (_same_scale's split-sized-open guard)
    set30(fb, I)
    got = {}
    for case in ("first-pass ask", "deferred retry"):
        reset_b()
        if case == "deferred retry":
            I.live(FakeGW(day, day, "raise"), ["AAPL"], now=t0)
        gw = FakeGW(day, day, "serve", day_k=0.5)
        r = I.live(gw, ["AAPL"], now=t1)["rows"]["AAPL"]
        got[case] = (r.get("history", r.get("error")), I._GW_FAILED.get(key, "absent"),
                     [c[0] for c in gw.calls if c[1] == "1 M"])
    check("B: today's open reaches the deferred retry: with a split-sized open (2:1 from the month) the retry is refused "
          "and remembered as None, as a first-pass ask is", got["first-pass ask"] == got["deferred retry"]
          == ("cache", None, ["AAPL"]), f"{got}")
    # ---- B: with no retry candidate, rows and gateway calls (order and arguments) as before round 4
    if not R4BEFORE.exists():
        print(f"      SKIPPED: the no-retry comparison with the module before round 4 ({R4BEFORE} is absent)", flush=True)
    else:
        B4 = load_module("intraday_round4_before", R4BEFORE)
        same, n = [], 0
        cases = [("whole day, AAPL lacking 19 Nov, month served", fb, syms, "serve", REALD)]
        cases += [(f"whole day, answer '{m}'", fb, syms, m, REALD) for m in ("stale", "few", "empty", "none")]
        cases += [("whole day, five names lacking 19 Nov past the cap", fc, five, "serve", REALD),
                  ("whole day, the e2e names on 20 Nov (BITO lacking 19 Nov)", REAL30, E2E, "serve",
                   asof_daily(["BITO", "AAPL", "SPY"], FIX / "daily_asof_20261120", "20261120"))]
        for lab, f30, names, mode, dfold in cases:
            set30(f30, I, B4)
            setd(dfold, I, B4)
            out = []
            for mod in (I, B4):
                mod._GW_HISTORY.clear()
                mod._GW_FAILED.clear()
                gw = FakeGW(day, day, mode, cut="20261119")
                v, _ = replay(mod, names, day, day, gw=gw)
                out.append((js(v), gw.calls, js(gw.kws)))
            n += 1
            same.append(out[0] == out[1])
            if out[0] != out[1]:
                print(f"      differs from the module before round 4: {lab}", flush=True)
        set30(REAL30, I, O, B4)
        setd(REALD, I, O, B4)
        check(f"B: with no retry candidate ({n} whole-day replays of Fri 20 Nov: served, four unusable answers, past the "
              "cap, the e2e names): "
              "rows, gateway calls and their arguments identical to the module before round 4", all(same) and n == 7)
    reset_b()
    set30(REAL30, I, O)
    setd(REALD, I)


# ---------------------------------------------------------------- end to end with a fake gateway
E2E = ["SOXX", "BITO", "SPY", "AAPL", "NVDA", "XOM", "KO", "MSFT", "JPM", "QQQ", "SMH", "TSLA", "AMD", "BRK B", "EWZ",
       "FTNT", "INTC"]
CONTROLS = E2E[2:]


def e2e(O, R1):
    set30(REAL30, I, O, R1)
    sd = FIX / "standin_bars"
    for s in E2E:
        standin_daily(s, sd)
    for lab, src, today, daily in (("replay Thu 12 Nov (current daily cache)", "20261112", "20261112", REALD),
                                   ("replay Fri 20 Nov (current daily cache)", "20261120", "20261120", REALD),
                                   ("stand-in Mon 23 Nov (20 Nov's bars as today, real daily cache, "
                                    + ("with its 20 Nov bar)" if BITO20 else "no 20 Nov bar)"), "20261120", "20261123", REALD),
                                   ("stand-in Mon 23 Nov (20 Nov's bars as today, stand-in daily bar for 20 Nov)",
                                    "20261120", "20261123", sd)):
        setd(daily, I, R1)
        reset_b()
        ro, co = replay(O, E2E, src, today)
        r1, c1 = replay(R1, E2E, src, today)
        rp, cp = replay(I, E2E, src, today)
        setd(REALD, I, R1)
        ok_l = [x[0] for x in rp] == I.SLOTS and [x[0] for x in ro] == I.SLOTS
        ctrl = all(js(ro[k][1][s]) == js(r1[k][1][s]) == js(rp[k][1][s]) for k in range(I.NS) for s in CONTROLS)
        nd = {s: sum(js(ro[k][1][s]) != js(rp[k][1][s]) for k in range(I.NS)) for s in ("SOXX", "BITO")}
        n1 = {s: sum(js(r1[k][1][s]) != js(rp[k][1][s]) for k in range(I.NS)) for s in ("SOXX", "BITO")}
        errs = [(s, rp[k][1][s].get("error")) for k in range(I.NS) for s in E2E if "error" in rp[k][1][s]]
        want = [("BITO", "1 M")] * 13 if today == "20261120" else []
        check(f"e2e {lab}: 13 looks served, no errors; gateway month calls {sorted(set(cp))} x{len(cp)} (expected only BITO "
              "on 20 Nov, one a look); round 1 and the unpatched none", ok_l and not errs and cp == want and co == c1 == [],
              f"{errs[:3]}")
        check(f"e2e {lab}: {len(CONTROLS)} controls byte-identical at all 13 looks (unpatched, round 1, v2), and no extra "
              "gateway call for them", ctrl and not [c for c in cp if c[0] in CONTROLS])
        print(f"      looks that differ from the unpatched {nd}, from round 1 {n1}")
        if today == "20261123" and daily is REALD and not BITO20:      # once the daily cache has 20 Nov, round 1 brings
            #                                                             BITO to one scale too: the else branch
            check(f"e2e {lab}: BITO now differs from round 1 at the looks (round 1 mixed, v2 one scale), SOXX the same",
                  n1["BITO"] > 0 and n1["SOXX"] == 0 and nd["SOXX"] > 0)
        else:
            check(f"e2e {lab}: SOXX and BITO identical to round 1 at all 13 looks", n1 == {"SOXX": 0, "BITO": 0})


def main():
    missing = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ORIG, ROUND1) if not p.exists()]
    if missing:
        print("The reference modules this test compares against are not in this checkout. Take them from the branch "
              "models-w3 (they are test fixtures; no live code reads them): " + ", ".join(missing), flush=True)
        sys.exit(2)
    print(f"daily folder in use: {REALD} ({'the --daily stand-in' if _DAILY_OPT is not None else 'the real data/bars'}; "
          f"BITO's 20 Nov daily bar: {'there' if BITO20 else 'not there'}). It is read by _one_scale, BITO20, the "
          "standin_daily / asof_daily helpers and load_panel's trader.Data(); data/bars30, data/iv, lists/ and the split "
          "ledger are always the real files", flush=True)
    h0 = [sha(p) for p in SHIPPED]
    O = load_module("intraday_orig", ORIG)
    R1 = load_module("intraday_round1", ROUND1)
    check("setup, not a test of the patch: the unpatched copy and the round 1 copy are the pinned files (text hash), the "
          "first without _one_scale, the second with round 1's _jump_ok, both with this checkout's paths",
          all(text_sha(p) == h for p, h in PINNED.items()) and O.BARS30 == REAL30 and not hasattr(O, "_one_scale")
          and hasattr(R1, "_jump_ok") and R1.DAILY == REALD)
    synthetic(O, R1)
    real(O, R1)
    missing_session(O, R1)
    fallback_r4(O)
    e2e(O, R1)
    panel(O, R1)
    check("shipped intraday_panel.npz and intraday_model.pkl byte-identical", [sha(p) for p in SHIPPED] == h0)
    n_ok = sum(ok for _, ok in RES)
    print(f"{n_ok} of {len(RES)} checks passed (daily folder: {REALD})", flush=True)
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()
