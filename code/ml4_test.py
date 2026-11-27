"""Offline tests of ML4.0 (forecast4.py) in this repo: no gateway, no Tinker, no network. The repo is only read; every
folder a tool writes into is a temp folder.

    .venv\\Scripts\\python.exe ml4_test.py      # PYTHONIOENCODING=utf-8, HF_HUB_OFFLINE=1; prints ALL OK and exits 0, else 1

Brought in from the Trading Model Researcher's build4 (runs/ml/research/week6_prep/ml4/build4 in Trader-models,
15 Nov 2026): harness4.py's recorded-call re-rendering (section H below) and forecast4_test.py's checks 0 and a-g, adapted
to run from the repo: modules and the gap1 model come from this folder; ml_tools.py may differ from HEAD only by the ML4
pin; the files the build wrote beside itself (description_ml4.txt, absr1_changes.txt, the log) go to the temp folder.

Added in review (h-o): an independent comparison of ML3.1 and ML4.0 on recorded calls (own parser, own absr1 formula,
ML4's nightly document read from disk); a stale night, a missing ML4 file, a missing model, a name not covered, a failed
quote and failed bars, ML2 with no forecasts, bad arguments: text, never an exception, and no row shifted; track_record
in every form, rows identical with and without it, the legend kept for the look's first rows call; fix_absr1's edge
cases; the morning (compute(), then a look reading its file); ML2 and ML3 unchanged (git diff, ml_tools.changed_files).
"""
import ast
import copy
import glob
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
LIVE = ROOT
DAYS = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
ET = ZoneInfo("America/New_York")
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def _live_stub():
    """live.py imports tinker, the gateway framework and the loop; the tools only read its two holiday sets, taken here from
    the working tree's live.py with ast (nothing of live.py runs)."""
    src = (ROOT / "live.py").read_text(encoding="utf-8")
    vals = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("HOLIDAYS", "HALF_DAYS"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    m = types.ModuleType("live")
    m.HOLIDAYS, m.HALF_DAYS = vals["HOLIDAYS"], vals["HALF_DAYS"]
    m.__file__ = "<stub of live.py: HOLIDAYS and HALF_DAYS only>"
    return m


sys.modules["live"] = _live_stub()
import forecast            # noqa: E402
import forecast3 as f3     # noqa: E402
import forecast4 as f4     # noqa: E402
from ml_research import cost, events, macro_feats   # noqa: E402
import ml_tools            # noqa: E402
import tools               # noqa: E402

assert Path(forecast.__file__).parent == ROOT and Path(f3.__file__).parent == ROOT and Path(f4.__file__).parent == ROOT
assert f3.forecast is forecast and f4.forecast is forecast and f4.f3 is f3

MODEL4 = ROOT / "runs" / "ml" / "research" / "ml4_gap1_model.json"
TMP = Path(tempfile.mkdtemp(prefix="ml4_test_"))
if not os.environ.get("ML4_KEEP_TMP"):                 # ML4_KEEP_TMP=1 keeps the temp folder for inspection
    import atexit
    atexit.register(shutil.rmtree, TMP, ignore_errors=True)
# read-only inputs from the repo; every folder a tool writes into is under TMP
forecast.OUT, forecast.RES = ROOT / "runs" / "ml" / "forecast", ROOT / "runs" / "ml" / "research"
forecast.ONDEMAND = TMP / "ml2_ondemand"
f3.OUT, f3.ONDEMAND = TMP / "ml3", TMP / "ml3" / "ondemand"
f3.MODEL = ROOT / "runs" / "ml" / "research" / "ml3_model_amend1.json"
f3.BARS, f3.BARS30, f3.MACRO = ROOT / "data" / "bars", ROOT / "data" / "bars30", ROOT / "data" / "macro" / "calendar.json"
f4.ROOT = TMP
f4.OUT = TMP / "runs" / "ml" / "forecast4"
f4.ONDEMAND = f4.OUT / "ondemand"
f4.MODEL = MODEL4

# ================================================================ H. harness4.py (build4), adapted
FC2, FC3 = {}, {}
SPY30 = json.loads((f3.BARS30 / "SPY.json").read_text(encoding="utf-8"))["bars"]
VAL = re.compile(r"^[±]?(na|-?[\d.]+)(?:\((-?[\d.]+|na)-(-?[\d.]+|na)\))?$")


def fc2(as_of):
    if as_of not in FC2:
        FC2[as_of] = json.loads((forecast.OUT / f"fc_{as_of}.json").read_text(encoding="utf-8"))
    return FC2[as_of]


def fc3(as_of):
    """ML3's stored document plus ML3's stored on-demand gap1 rows (outside names), the same inputs for both versions."""
    if as_of not in FC3:
        f = ROOT / "runs" / "ml" / "forecast3" / f"fc_{as_of}.json"
        if not f.exists():
            FC3[as_of] = None
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        for p in glob.glob(str(ROOT / "runs" / "ml" / "forecast3" / "ondemand" / as_of / "*.json")):
            try:
                r = json.loads(Path(p).read_text(encoding="utf-8"))
            except ValueError:
                continue
            d["rows"].setdefault(r["symbol"], r["row"])
        FC3[as_of] = d
    return FC3[as_of]


def recorded_calls(days=DAYS, versions=("ML3",)):
    """Every forecast call with a forecasts reply in the look files of the sleeves on these versions."""
    out = []
    for day in days:
        for p in sorted(glob.glob(str(ROOT / "runs" / "live" / day / "*.json"))):
            if not re.search(r"-\d{4}\.json$", p):
                continue
            try:
                d = json.loads(Path(p).read_text(encoding="utf-8"))
            except ValueError:
                continue
            if not isinstance(d, dict) or not str(d.get("ml_tool") or "").startswith(versions):
                continue
            k = 0
            for t in d.get("trace") or []:
                if t.get("tool") != "forecast" or not isinstance(t.get("result"), str) or \
                        not re.match(r"as of the close of \d{4}-\d\d-\d\d", t["result"]):
                    continue
                k += 1
                sl = d.get("sleeve") or ""
                out.append({"file": p, "sleeve": sl, "mc": "claude" if sl.startswith(("opus", "sonnet", "haiku")) else "gptoss",
                            "day": day, "time": d.get("time"), "step": t.get("step"), "args": t.get("args"),
                            "result": t["result"], "k": k, "ml_tool": d.get("ml_tool"),
                            "value": (d.get("state") or {}).get("value")})
    return out


def body(result):
    return re.sub(r"\n\(research call \d+ of \d+\)\s*$", "", result)


def symbols_of(args):
    syms = (args or {}).get("symbols") or []
    if isinstance(syms, str):
        syms = [s for s in re.split(r"[,\s]+", syms) if s]
    return list(dict.fromkeys(str(s).upper().strip() for s in syms if str(s).strip()))[:forecast.MAX_SYMBOLS]


def parse_rows(text, syms):
    """(header columns, {sym: (values, note)}) of a forecast reply."""
    lines = text.split("\n")
    hi = next((i for i, x in enumerate(lines) if x.startswith("sym close range1")), None)
    if hi is None:
        return None, {}
    cols = lines[hi].split(" ")
    nv = len(cols) - 2
    out = {}
    for ln in lines[hi + 1:]:
        s = max((s for s in syms if ln == s or ln.startswith(s + " ")), key=len, default=None)
        if not s:
            continue
        rest = ln[len(s) + 1:]
        if rest.startswith("not covered"):
            out[s] = (None, rest)
            continue
        tk = rest.split(" ")
        out[s] = (tk[:nv], " ".join(tk[nv:]))
    return cols, out


def num(tok):
    m = VAL.match(tok or "")
    if not m:
        return None
    f = lambda x: None if x in (None, "na") else float(x)
    return tuple(f(m.group(i)) for i in (1, 2, 3))


def col_index(cols, name):
    return next((i - 1 for i, c in enumerate(cols) if c.startswith(name)), None)


class GW:
    """SPY's stored 30-minute bars and a fixed pre-market quote; any other request raises (no gateway)."""

    def __init__(self, now, prev):
        self.now, self.prev = now, prev

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, **kw):
        if (symbol, bar_size) != ("SPY", "30 mins"):
            raise RuntimeError("offline test: no gateway")
        day = f"{self.now:%Y%m%d}"
        return [b for b in SPY30 if b["date"][:8] == self.prev or (b["date"][:8] == day and b["date"][9:14] <= f"{self.now:%H:%M}")]

    def quotes(self, symbols, primary="", timeout=12.0):
        return [{"symbol": "SPY", "last": 681.5, "close": 677.0, "bid": 681.4, "ask": 681.6,
                 "last_time": f"{self.now:%Y-%m-%d}T13:58:00Z"}]

    def stats(self, *a, **k):
        raise RuntimeError("offline test: no gateway (stats)")

    def details(self, *a, **k):
        raise RuntimeError("offline test: no gateway (details)")


class BrokenGW(GW):
    """Every request fails: the quote and the bars too."""

    def bars(self, *a, **k):
        raise RuntimeError("offline test: bars failed")

    def quotes(self, *a, **k):
        raise RuntimeError("offline test: quote failed")


def ev_for(as_of, syms):
    if fc3(as_of) is None:
        return {}
    night = fc3(as_of)["night"]
    out = {}
    if len(syms) >= 3:
        out[syms[1]] = {"calendar_day": night[0], "calendar_when": "after close", "calendar_sessions": 0, "calendar_reported": False}
        out[syms[2]] = {"next_earn_day": night[1], "next_earn_when": "before open", "next_earn_sessions": 1, "next_earn_overdue": False}
    return out


def setup(call, canned=False):
    """Stub the inputs of one recorded call; returns (gw, args, as_of, now). Both versions then run on the same inputs."""
    res = body(call["result"])
    lines = res.split("\n")
    as_of = lines[0][19:29].replace("-", "")
    syms = symbols_of(call["args"])
    m = re.search(r"session ranges at the (\d\d):(\d\d) look", lines[0])
    t = datetime.fromisoformat(call["time"]).astimezone(ET)
    now = t.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0) if m else t
    snote = f"session ranges at the {now:%H:%M} look" if m else (
        "session ranges from 09:30 ET" if "session ranges from 09:30 ET" in lines[0] else lines[0].split("; ", 1)[-1].split(";")[0])
    cols, rows = parse_rows(res, syms)
    sess, costs = {}, {}
    if cols:
        i60, irest = col_index(cols, "next60"), col_index(cols, "rest_of_day")
        isp, ico, iti, iad = (col_index(cols, c) for c in ("spread_bps", "cost_bps", "tier", "adv_$m"))
        for s, (vals, note) in rows.items():
            if vals is None:
                continue
            if i60 is not None and len(vals) > irest:
                a, b = num(vals[i60]), num(vals[irest])
                if a and b and None not in a + b:
                    sess[s] = tuple(2 * x for x in a + b)
            sp, co = num(vals[isp]) if len(vals) > isp else None, num(vals[ico]) if len(vals) > ico else None
            if sp and co and None not in sp + co:
                costs[s] = {"symbol": s, "price": float(vals[0]), "spread_bps": sp[0], "spread_bps_10": sp[1],
                            "spread_bps_90": sp[2], "cost_bps": co[0], "cost_bps_10": co[1], "cost_bps_90": co[2],
                            "tier": vals[iti], "adv_usd_m": None if vals[iad] == "na" else float(vals[iad])}
    macro = next((x for x in lines[1:6] if x.startswith("macro")), None)
    doc2 = fc2(as_of)
    forecast.live_provider = lambda a=as_of, d=doc2: (a, {"market": copy.deepcopy(d["market"]), "rows": copy.deepcopy(d["forecasts"])})
    forecast._today_et = lambda n=now: n.date()
    forecast._prev_session = lambda today, a=as_of: a
    forecast._session = lambda gw, ss, iv_hint=None, n=snote, se=sess: (n, {s: se[s] for s in ss if s in se}) \
        if n.startswith("session ranges at") else (n, {})
    for s in syms:
        if s not in doc2["forecasts"]:
            src = ROOT / "runs" / "ml" / "forecast" / "ondemand" / as_of / f"{s.replace(' ', '_')}.json"
            dst = forecast.ONDEMAND / as_of / src.name
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)

    def cost_live(gw, symbols=None, notional=None, **kw):
        out = []
        for s in symbols or []:
            have = (forecast.ONDEMAND / as_of / f"{s.replace(' ', '_')}.json").exists()
            out.append(dict(costs[s]) if s in costs and have else {"symbol": s, "error": "offline test: no live cost",
                                                                   "note": "no stored row"})
        return {"orders": out}
    cost.live = cost_live
    ev = ev_for(as_of, syms) if canned else {}
    events.live = lambda gw, covered, base=None, **kw: {"rows": {s: dict(ev[s]) for s in covered if s in ev}}
    macro_feats.macro_today = lambda *a, **k: macro
    f3._now = lambda n=now: n
    doc3 = fc3(as_of)
    for m_ in (f3, f4):
        m_._MEM.clear()
        if doc3 is not None:
            m_._MEM[as_of] = copy.deepcopy(doc3)
    args = {k: v for k, v in (call["args"] or {}).items()}
    return GW(now, as_of), args, as_of, now


def render(call, canned=False, look_tag=None):
    gw, args, as_of, now = setup(call, canned)
    key = None if look_tag is None else (look_tag, call["sleeve"], call["time"])
    t31 = f3.run(gw, dict(args), sleeve_value=call.get("value"), look=None if key is None else ("ML3.1",) + key)
    gw, args, as_of, now = setup(call, canned)
    t4 = f4.run(gw, dict(args), sleeve_value=call.get("value"), look=None if key is None else ("ML4.0",) + key)
    return t31, t4


# ================================================================ checks
results = []


def say(s=""):
    print(s, flush=True)


def check(name, ok, detail=""):
    say(("ok   " if ok else "FAIL ") + name + ("" if not detail else f": {detail}"))
    results.append(bool(ok))
    return ok


def snapshot(root):
    out = {}
    root = Path(root)
    if not root.exists():
        return out
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = Path(dp) / f
            try:
                st = p.stat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def changed(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


BANNED = [r"\buse (them|it|these) to size\b", r"\bto size positions\b", r"\bskip (trades|the trade|a trade|it)\b",
          r"\bskip trades whose\b", r"stop and limit distances", r"\bset (a |your )?stops?\b", r"stop[- ]distance",
          r"touched by noise", r"none of the models", r"did not beat", r"beat trading costs", r"expected gain from direction",
          r"\bnothing (here |else )?works\b", r"\bnot worth\b", r"\bavoid\b", r"\bhold back\b", r"\btrade less\b",
          r"\b(do not|don't|never) trade\b", r"weigh it accordingly", r"\bshould\b", r"keeps? your risk", r"usual level",
          r"mostly unpredictable", r"careful bookkeeping", r"earnings drift", r"\b(larger|bigger) (size|position|order)s?\b",
          r"\bsize up\b", r"\bincrease (your )?(size|exposure|position)", r"\breduce (your )?(size|exposure|position)",
          r"\bcaution\b", r"\bwarn", r"\bdanger", r"\brisky\b", r"\bbe careful\b", r"\bdo nothing\b", r"\bstay out\b",
          r"\bno edge\b", r"\bnot profitable\b", r"\bwait for\b", r"\byou must\b", r"\bmomentum ranking\b",
          # added in review: probability claims about direction and plain imperatives about trading
          r"as likely down as up", r"coin flip", r"\bconsider\b", r"\brecommend", r"\bbeware\b", r"\bunreliable\b",
          r"\bdon't\b", r"\bdo not\b"]
BANNED_RX = [re.compile(p, re.I) for p in BANNED]


def banned_hits(text):
    return sorted({m.group(0) for rx in BANNED_RX for m in rx.finditer(text or "")})


NON_ROW = ("as of the close of ", "market:", "market now:", "macro", "gap1 events tonight", "gap1 unavailable")


def line_kind(ln):
    if ln.startswith("cost_bps: round trip"):
        return "leg_cost"
    if ln.startswith("move$ ±(90) and sleeve% ±(90):"):
        return "leg_move"
    if ln.startswith("No column forecasts direction"):
        return "leg_dir"
    if ln.startswith("gap1 ±(10-90):"):
        return "leg_gap1"
    if ln.startswith("cost_bps and move$ for a"):
        return "leg_ref"
    for p in NON_ROW:
        if ln.startswith(p):
            return "head:" + p.strip()
    if ln.startswith("sym close range1"):
        return "cols"
    return "row"


def compare(t31, t4, syms, stats):
    """build4's comparison: problems between an ML3.1 and an ML4.0 reply of the same inputs. From ML4.1 (16 Nov) the two
    lines ML4.1 words plainly (forecast4._PLAIN) are mapped back to ML4.0's words first: their only change is the wording."""
    bad = []
    for old, new in getattr(f4, "_PLAIN", ()):
        t4 = t4.replace(new, old)
    if hasattr(f4, "_slot_label"):          # ML4.2 (20 Nov): a quarter-hour call is labelled with its half-hour slot,
        t31 = f4._slot_label(t31)           # its only change to the head line, so ML3.1's label is mapped the same way
    a, b = t31.split("\n"), t4.split("\n")
    ka = [(line_kind(x), x) for x in a]
    kb = [(line_kind(x), x) for x in b]
    if [k for k, _ in ka] != [k for k, _ in kb]:
        return [f"line kinds differ: {[k for k, _ in ka]} vs {[k for k, _ in kb]}"]
    cols_a, rows_a = parse_rows(t31, syms)
    cols_b, rows_b = parse_rows(t4, syms)
    for (k, x), (_, y) in zip(ka, kb):
        if k.startswith("head:") or k in ("cols", "leg_cost", "leg_ref"):
            if k == "head:gap1 unavailable":
                x = x.replace("no ML3 forecasts for this close", "no ML4 forecasts for this close")
            if x != y:
                bad.append(f"{k}: {x!r} vs {y!r}")
        elif k == "leg_move":
            if y != x.split(f4.LOSS_CUT, 1)[0].rstrip():
                bad.append(f"move legend: {y!r}")
        elif k == "leg_dir":
            if y != f4.DIR_LINE:
                bad.append("dir legend")
        elif k == "leg_gap1":
            if y != f4.GAP1_LINE:
                bad.append("gap1 legend")
    if cols_a != cols_b or list(rows_a) != list(rows_b):
        bad.append(f"columns or row order differ: {cols_a} / {list(rows_a)} vs {cols_b} / {list(rows_b)}")
        return bad
    ia = col_index(cols_a, "absr1") if cols_a else None
    i_v20 = col_index(cols_a, "vol20") if cols_a else None
    for s in rows_a:
        (va, na), (vb, nb) = rows_a[s], rows_b[s]
        stats["rows"] += 1
        if va is None or vb is None:
            if (va, na) != (vb, nb):
                bad.append(f"{s}: not-covered rows differ")
            continue
        if len(va) != len(vb):
            bad.append(f"{s}: {len(va)} vs {len(vb)} values")
            continue
        for j, (x, y) in enumerate(zip(va, vb)):
            if j == ia:
                want, clause = f4.fix_absr1(x, va[i_v20])
                if y != want:
                    bad.append(f"{s}: absr1 {x} -> {y}, expected {want}")
                if clause:
                    stats["absr1_changed"] += 1
                    stats["changes"].append((s, x, y, va[i_v20]))
                    exp_note = f"{na}; {clause}" if na else clause
                    if nb != exp_note:
                        bad.append(f"{s}: note {nb!r}, expected {exp_note!r}")
                elif nb != na:
                    bad.append(f"{s}: note differs {na!r} vs {nb!r}")
                continue
            stats["tokens"] += 1
            if x != y:
                bad.append(f"{s}: column {cols_a[j + 1]} {x} vs {y}")
    return bad


def absr1_tokens(text, syms):
    cols, rows = parse_rows(text, syms)
    if not cols:
        return []
    ia = col_index(cols, "absr1")
    return [(s, v[ia]) for s, (v, n) in rows.items() if v is not None and len(v) > ia]


NOW_DATED = re.compile(r"^market now: (SPY [+-][\d.]+% since the last close (as of about \d\d:\d\d ET \(the latest print of "
                       r"the \d\d:\d\d bar|at the 09:30 ET open)|SPY [+-][\d.]+% since the last close \(pre-market quote, "
                       r"(last trade \d\d:\d\d ET|quote asked for at \d\d:\d\d ET)|no session today|the session has closed|"
                       r"no intraday line on half-days|SPY's 30-minute bars unavailable|SPY pre-market quote unavailable)")


# ---------------------------------------------------------------- review helpers (independent of compare above)
def my_rows(text, syms):
    """(columns without sym and note, {sym: (tokens or None, note)}, lines, header index) with my own split."""
    L = text.split("\n")
    hi = next((i for i, x in enumerate(L) if x.startswith("sym close range1")), None)
    if hi is None:
        return None, {}, L, None
    cols = L[hi].split(" ")[1:-1]
    out = {}
    for x in L[hi + 1:]:
        s = max((s for s in syms if x == s or x.startswith(s + " ")), key=len, default=None)
        if s is None:
            continue
        rest = x[len(s) + 1:]
        out[s] = (None, rest) if rest.startswith("not covered") else (rest.split(" ")[:len(cols)], " ".join(rest.split(" ")[len(cols):]))
    return cols, out, L, hi


def my_abs(vol_tok):
    """vol20's one-day E|x| as the review computes it: sqrt(2/pi) * vol20 / sqrt(252), 2 dp."""
    try:
        v = float(vol_tok)
    except ValueError:
        return "na"
    return f"{round(math.sqrt(2 / math.pi) * v / math.sqrt(252), 2):g}"


def ml2_view(t2, t4, syms):
    """ML4's rows read against ML2's own reply of the same inputs: every ML2 token under its own symbol, gap1 inserted after
    absr1, absr1 as fixed. Problems as text (a shifted row shows here)."""
    c2, r2, _, _ = my_rows(t2, syms)
    c4, r4, _, _ = my_rows(t4, syms)
    bad = []
    if c2 is None or c4 is None:
        return ["no header"]
    ia = c2.index("absr1±%")
    if c4 != c2[:ia + 1] + ["gap1±%(10-90)"] + c2[ia + 1:]:
        return [f"header {c2} vs {c4}"]
    if list(r2) != list(r4):
        return [f"rows {list(r2)} vs {list(r4)}"]
    for s in r2:
        (v2, n2), (v4, n4) = r2[s], r4[s]
        if v2 is None or v4 is None:
            if (v2, n2) != (v4, n4):
                bad.append(f"{s}: not covered {n2!r} vs {n4!r}")
            continue
        g = v4[ia + 1]
        v4x = v4[:ia + 1] + v4[ia + 2:]
        want_a, clause = f4.fix_absr1(v2[ia], v2[c2.index("vol20%")])
        if v4x[:ia] + v4x[ia + 1:] != v2[:ia] + v2[ia + 1:] or v4x[ia] != want_a:
            bad.append(f"{s}: {v2} vs {v4x}")
        if n4 != (n2 if not clause else (f"{n2}; {clause}" if n2 else clause)):
            bad.append(f"{s}: note {n2!r} vs {n4!r}")
        bad += [] if g else [f"{s}: no gap1 token"]
    return bad


def gap1_tokens(text, syms):
    c, r, _, _ = my_rows(text, syms)
    if not c or "gap1±%(10-90)" not in c:
        return []
    i = c.index("gap1±%(10-90)")
    return [v[i] for v, _ in r.values() if v is not None]


def main():
    t_start = time.time()
    watch = [ROOT / "runs" / "ml" / "forecast", ROOT / "runs" / "ml" / "forecast3", ROOT / "runs" / "ml" / "research"]
    before_live = {str(w): snapshot(w) for w in watch}

    # ---------------- 0. inputs
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"Trader HEAD {head}")
    st = subprocess.run(["git", "-C", str(ROOT), "status", "--short", "forecast.py", "forecast3.py"],
                        capture_output=True, text=True).stdout.strip()
    def bumped(name):
        """True if the working file is a new revision of its own: its VERSION differs from HEAD's (ML2.4 and ML3.2 came in
        the same commit as ML4's registration, 15 Nov). ML4 itself must never change these files."""
        ver = lambda s: (re.search(r'^VERSION = "([^"]+)"', s, re.M) or [None, None])[1]
        h = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{name}"], capture_output=True).stdout.decode("utf-8")
        return ver(h) != ver((ROOT / name).read_text(encoding="utf-8"))
    touched = [ln.split()[-1] for ln in st.splitlines() if ln.strip()]
    check("0. forecast.py and forecast3.py are HEAD's, or a new revision of their own (VERSION differs from HEAD's)",
          all(bumped(n) for n in touched), st)
    h_ml = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:ml_tools.py"], capture_output=True).stdout.decode("utf-8")
    w_ml = (ROOT / "ml_tools.py").read_bytes().decode("utf-8")
    pin = re.compile(r'^VERSIONS = \{[^\n]*$', re.M)
    norm = lambda s: pin.sub("VERSIONS", s.replace("\r\n", "\n"))
    doc = re.compile(r'\A"""(.*?)"""', re.S)        # the module's docstring, which names ML4's extra parameter from 15 Nov
    check("0. ml_tools.py is HEAD's, or HEAD's with only its VERSIONS line and docstring changed",
          doc.sub("DOC", norm(h_ml), count=1) == doc.sub("DOC", norm(w_ml), count=1))
    check("0. ml_tools.VERSIONS pins ML4 to forecast4, ML2 and ML3 as before",
          ml_tools.VERSIONS == {"ML2": "forecast", "ML3": "forecast3", "ML4": "forecast4"}, str(ml_tools.VERSIONS))
    m3 = (ROOT / "runs" / "ml" / "research" / "ml3_model_amend1.json").read_bytes().replace(b"\r\n", b"\n")
    m4 = MODEL4.read_bytes().replace(b"\r\n", b"\n")
    check("0. ml4_gap1_model.json is a byte copy of ML3's fit (line endings aside)", m3 == m4, hashlib.sha256(m4).hexdigest()[:16])
    # ML4.2 (20 Nov): the slot label at quarter-hour calls; a call in a slot's first half reads as before, byte for byte
    for before, after in (("session ranges at the 10:04 look", "session ranges at the 10:04 look"),
                          ("session ranges at the 09:30 look", "session ranges at the 09:30 look"),
                          ("session ranges at the 10:14 look", "session ranges at the 10:14 look"),
                          ("session ranges at the 09:47 look", "session ranges at the 09:30 look"),
                          ("session ranges at the 10:15 look", "session ranges at the 10:00 look"),
                          ("session ranges at the 15:52 look", "session ranges at the 15:30 look"),
                          ("session ranges at the 09:05 look", "session ranges at the 09:05 look")):
        txt = f"as of the close of 2026-11-20\n{before}\nAAA ..."
        check(f"ML4.2 label: {before[-10:-5]} -> {after[-10:-5]}", f4._slot_label(txt) == txt.replace(before, after), f4._slot_label(txt))
    check("0. versions", forecast.VERSION == "ML2.4" and f3.VERSION == "ML3.2" and f4.VERSION == "ML4.2",
          f"{forecast.VERSION} {f3.VERSION} {f4.VERSION}")

    # ---------------- a. numbers on recorded calls (build4)
    calls = recorded_calls()
    say(f"recorded ML3 forecast calls with a forecasts reply, 9-13 Nov: {len(calls)} "
        f"({sum(c['mc'] == 'gptoss' for c in calls)} gpt-oss, {sum(c['mc'] == 'claude' for c in calls)} Claude)")
    outputs4, outputs31 = [], []
    for canned in (False, True):
        stats = {"rows": 0, "tokens": 0, "absr1_changed": 0, "changes": []}
        n, n_bad, first_bad, errors = 0, 0, [], 0
        legend = Counter()
        now_ok, now_n = 0, 0
        seen_look = set()
        for c in calls:
            syms = symbols_of(c["args"])
            if canned and len(syms) < 3:
                continue
            try:
                t31, t4 = render(c, canned=canned, look_tag=f"a{int(canned)}")
            except Exception:                           # noqa: BLE001
                errors += 1
                if len(first_bad) < 3:
                    first_bad.append(f"{c['file']} k{c['k']}: {traceback.format_exc()[-300:]}")
                continue
            n += 1
            bad = compare(t31, t4, syms, stats)
            if bad:
                n_bad += 1
                if len(first_bad) < 5:
                    first_bad.append(f"{c['file']} k{c['k']}: {bad[:3]}")
            outputs4.append((t4, syms))
            outputs31.append(t31)
            first = (c["sleeve"], c["time"]) not in seen_look
            seen_look.add((c["sleeve"], c["time"]))
            for tag, t in (("ML3.1", t31), ("ML4.0", t4)):
                L = t.split("\n")
                full = any(x.startswith("No column forecasts direction") for x in L) and any(x.startswith("gap1 ±(10-90):") for x in L)
                ref = any(x.endswith(forecast.LEGEND_REF) for x in L)
                legend[(tag, "first" if first else "later", "full" if full and not ref else "ref" if ref and not full else "other")] += 1
            for x in t4.split("\n"):
                if x.startswith("market now:"):
                    now_n += 1
                    now_ok += bool(NOW_DATED.match(x))
        label = "canned earnings events" if canned else "no events"
        check(f"a. [{label}] {n} calls re-rendered, ML4.0 numbers equal ML3.1 row for row ({stats['rows']} rows, "
              f"{stats['tokens']} value tokens; absr1 changed on {stats['absr1_changed']} rows)",
              n >= (30 if not canned else 10) and n_bad == 0 and errors == 0,
              f"{n_bad} calls differ, {errors} errors; {first_bad[:3]}" if (n_bad or errors) else "")
        ok_leg = all(v == 0 for k, v in legend.items() if k[2] == "other") and \
            all(v == 0 for k, v in legend.items() if (k[1], k[2]) in (("first", "ref"), ("later", "full")))
        check(f"a. [{label}] legend once a look in both versions", ok_leg, str(dict(legend)))
        check(f"a. [{label}] market-now line names its print ({now_ok} of {now_n})", now_n > 0 and now_ok == now_n)

    # ---------------- b. description tokens
    import trader
    tok = trader.tokenizer()
    nt = lambda s: len(tok.encode(s, add_special_tokens=False))
    d31, d4, trk = nt(f3.SPEC["description"]), nt(f4.DESCRIPTION), nt(f4.TRACK_RECORD)
    check(f"b. description tokens ML4.0 {d4} <= 1,200 (ML3.1 {d31}; the record on request {trk})", d4 <= 1200)
    (TMP / "description_ml4.txt").write_text(f4.DESCRIPTION, encoding="utf-8")

    # ---------------- c. track record (build4)
    c0 = next(c for c in calls if c["mc"] == "gptoss" and len(symbols_of(c["args"])) >= 2)
    gw, args, as_of, now = setup(c0)
    only = f4.run(gw, {"symbols": [], "track_record": True})
    only2 = f4.run(gw, {"track_record": "true"})
    gw, args, as_of, now = setup(c0)
    both = f4.run(gw, dict(args, track_record=True), sleeve_value=c0["value"], look=("c", 1))
    gw, args, as_of, now = setup(c0)
    plain = f4.run(gw, dict(args), sleeve_value=c0["value"], look=("c", 2))
    gw, args, as_of, now = setup(c0)
    false_ = f4.run(gw, dict(args, track_record=False), sleeve_value=c0["value"], look=("c", 3))
    first_line = f4.TRACK_RECORD.split("\n")[0]
    check("c. track_record=true with no symbols returns the record alone", only == f4.TRACK_RECORD and only2 == f4.TRACK_RECORD)
    check("c. track_record=true with symbols: the forecasts, then the record", both.endswith("\n\n" + f4.TRACK_RECORD)
          and both.startswith("as of the close of ") and "sym close range1" in both)
    check("c. plain calls and track_record=false carry no record",
          first_line not in plain and first_line not in false_ and "241,993" not in plain and plain.startswith("as of the close of "))
    check("c. the description does not hold the record", "241,993" not in f4.DESCRIPTION and "223,155" not in f4.DESCRIPTION
          and first_line not in f4.DESCRIPTION and "track_record=true" in f4.DESCRIPTION)
    removed = forecast.DESCRIPTION.split("**No return or direction forecast is offered.**", 1)[1].split("**Track record**", 1)[0]
    nums = lambda s: set(re.findall(r"\d[\d,]*(?:\.\d+)?", s))
    example = re.search(r"For example, a one-± move.*?\(0\.31%\)\.", forecast.DESCRIPTION).group(0)
    lost = sorted(nums(f3.SPEC["description"]) - nums(removed) - nums(example)
                  - nums(f4.DESCRIPTION + f4.TRACK_RECORD + f4.DIR_LINE))
    check("c. every number of ML3.1's description is kept (outside the removed paragraph and move$'s worked example)",
          not lost, str(lost))
    added = sorted(nums(f4.DESCRIPTION + f4.TRACK_RECORD + f4.DIR_LINE + f4.GAP1_LINE)
                   - nums(f3.SPEC["description"] + (ROOT / "forecast3.py").read_text(encoding="utf-8")
                          + (ROOT / "forecast.py").read_text(encoding="utf-8")))
    label_check = {"87", "20", "0.2", "426", "63", "0.88", "95,624",           # ML4.2's corrections of gap1's "median" on
                   "24", "3", "16", "780", "640", "23", "17", "1,151"}          # nights our estimate names and range1's centre:
    # the Researcher's words of 21 Nov (RESULT.md: 0.8709 / 0.2019 / 0.0023 of 426, 0.631 and 0.883 of 95,624) and of 22 Nov,
    # after Trader Training's two reviews (models-w3 e4f6f93a, labels_check/WORDS_22NOV.md: on 780 report nights our estimate
    # named, 0.2359 / 0.0295 / 0.1603 walk-forward and 0.2346 / 0.0295 / 0.1692 with today's fit, about 640 of them in it;
    # 0 of 1,151 gap1 tables listed "earnings (our estimate)"); numbers already in forecast.py or forecast3.py aren't listed
    check("c. (review) the only numbers new to ML4's texts are the absr1 fix's own figures and ML4.2's label corrections",
          set(x.rstrip(",") for x in added) <= {"35", "10,972", "151", "4,279", "51"} | label_check, str(added))
    s = ("In a replay of nights our estimate named, before the tool had the Alpha Vantage calendar, 87% of gaps fell below "
         "the raised median when no report came that night (a check declared in advance); when one came, 24% did and 16% "
         "went above the raised 90% value (computed after the fact). Through 20 Nov 2026 there is no live record of gap1 "
         "for a name the events line lists as \"earnings (our estimate)\", or of how often a report comes then; see the "
         "track record.")
    check("c. ML4.2: gap1's correction in the description (22 Nov words), once, inside gap1's line",
          f4.DESCRIPTION.count(s) == 1
          and s in next(ln for ln in f4.DESCRIPTION.split("\n") if ln.startswith("- gap1 ±(10-90):")))
    check("c. ML4.2: no ML4 text carries 21 Nov's 149-night figures or says the values ran high (22 Nov words: the live "
          "tool listed no such name)", all(x not in f4.DESCRIPTION + f4.TRACK_RECORD + f4.GAP1_LINE
                                         for x in ("149 name-nights", "ran high")))
    check("c. ML4.2: the note line's estimate figure in the record's terms (22 Nov words)",
          "(last year's report plus 52 weeks; 47% of reports came in the estimated session)" in f4.DESCRIPTION
          and "in the right session 47% of the time" not in f4.DESCRIPTION)
    check("c. ML4.2: the description no longer says none was above the 90% value (that stays in the record, marked)",
          "none was above" not in f4.DESCRIPTION and "none above the 90% value" not in f4.DESCRIPTION)
    for s, col in (("Where our estimate named the night and no report came that night, 87% of gaps fell below the median, 20% below "
                    "the 10% value and 0.2% above the 90% value (426 name-nights, Nov 2025-Aug 2026, a check declared in advance). "
                    "Where our estimate named the night and a report did come that night, 24% of gaps fell below the median, 3% "
                    "below the 10% value and 16% above the 90% value (780 name-nights, Nov 2025-Oct 2026, computed after the fact, "
                    "with multipliers fit on earlier nights only; with today's fit, which used about 640 of these nights, 23%, 3% "
                    "and 17%). Both sets are from a replay of nights before the tool had the Alpha Vantage calendar; the two "
                    "counts cover different windows and give no share of named nights on which a report came. Since 2 "
                    "Nov 2026 a name is listed as \"earnings (our estimate)\" only when the latest Alpha Vantage calendar "
                    "snapshot has no entry for it that is dated today or later and gives a time of day; in the looks of 9-20 Nov "
                    "2026 no forecast table listed one (0 of 1,151 tables with gap1), so there is no live record of gap1 on such "
                    "a night, and how often a report comes on one is not measured.", "- gap1 ("),
                   ("range1 is a mean, not a median: 63% of half-spans fell below it, and half fell below 0.88 of it (95,624 "
                    "name-sessions, Nov 2025-Aug 2026).", "- range1: ")):
        check(f"c. ML4.2: the track record's correction on its {col.strip('- (:')} line, once", f4.TRACK_RECORD.count(s) == 1
              and s in next(ln for ln in f4.TRACK_RECORD.split("\n") if ln.startswith(col)))
    check("c. ML4.2: the 64% report-rate sentence is not in any ML4 text (the Researcher dropped it: a model would read it as "
          "the chance of a report on an (our estimate) row)", "64% of the time" not in f4.DESCRIPTION + f4.TRACK_RECORD + f4.GAP1_LINE)
    check("c. ML4.2: the legend's gap1 line is ML4.1's again (the 21 Nov clause removed, 22 Nov words), and the events "
          "line names its own mark",
          f4.GAP1_LINE == ("gap1 ±(10-90): median and 10-90% range of the size of the gap from today's close to the next "
                           "open, %; it does not change during the day and includes the events line below. Names outside "
                           "the validated universe get no earnings adjustment.")
          and """events.append(f"{s} earnings ({'Alpha Vantage calendar' if e == 'earn' else 'our estimate'})")"""
          in (ROOT / "forecast4.py").read_text(encoding="utf-8"))

    # ---------------- d. banned phrases (build4's list plus the review's additions)
    hits_desc = banned_hits(f4.DESCRIPTION)
    hits_spec = banned_hits(json.dumps(f4.SPEC["parameters"], ensure_ascii=False))
    hits_track = banned_hits(f4.TRACK_RECORD)
    hits_lines = banned_hits("\n".join((f4.DIR_LINE, f4.GAP1_LINE, f4.ABS_NOTE, f4.ABS_NA)))
    hits_out = Counter(h for t, _ in outputs4 for h in banned_hits(t))
    check("d. no banned phrase in the ML4.0 description", not hits_desc, str(hits_desc))
    check("d. no banned phrase in the ML4.0 parameter texts", not hits_spec, str(hits_spec))
    check("d. no banned phrase in the record returned on request", not hits_track, str(hits_track))
    check("d. no banned phrase in ML4.0's own legend and note lines", not hits_lines, str(hits_lines))
    check(f"d. no banned phrase in {len(outputs4)} rendered ML4.0 outputs", not hits_out, str(dict(hits_out)))
    ctrl_d, ctrl_o = banned_hits(f3.SPEC["description"]), Counter(h for t in outputs31 for h in banned_hits(t))
    check("d. control: the same list hits ML3's description; its outputs carry none since ML3.2 (the legend's 'expected gain "
          "from direction' clause went out with ML2.4)", bool(ctrl_d) and not ctrl_o,
          f"ML3.1 description: {ctrl_d}; ML3.1 outputs: {dict(ctrl_o)}")

    # ---------------- e. absr1 (build4)
    neg = [(s, x) for t, ss in outputs4 for s, x in absr1_tokens(t, ss)
           if num(x) and num(x)[0] is not None and num(x)[0] <= 0]
    n_abs = sum(1 for t, ss in outputs4 for s, x in absr1_tokens(t, ss))
    check(f"e. no absr1 <= 0 in the rendered ML4.0 outputs ({n_abs} absr1 values)", n_abs > 0 and not neg, str(neg[:5]))
    lines_out, n_served, bad_e = [], 0, []
    fixed_cost = lambda gw, symbols=None, notional=None, **kw: {"orders": [
        {"symbol": s, "price": 100.0, "spread_bps": 1.0, "spread_bps_10": 0.9, "spread_bps_90": 1.1, "cost_bps": 2.0,
         "cost_bps_10": 1.9, "cost_bps_90": 2.1, "tier": "A", "adv_usd_m": 500.0} for s in symbols or []]}
    for as_of in ("20261106", "20261109", "20261110", "20261111", "20261112"):
        doc2 = fc2(as_of)
        affected = sorted(s for s, r in doc2["forecasts"].items() if r.get("absr1_pct") is not None and r["absr1_pct"] <= 0)
        od = []
        for p in sorted((ROOT / "runs" / "ml" / "forecast" / "ondemand" / as_of).glob("*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            row = r.get("row") or {}
            if row.get("absr1_pct") is not None and row["absr1_pct"] <= 0 and r.get("symbol") not in doc2["forecasts"]:
                od.append(r["symbol"])
        names = affected + sorted(set(od))
        for k in range(0, len(names), 20):
            chunk = names[k:k + 20]
            night0 = fc3(as_of)["night"][0]
            fake = {"result": f"as of the close of {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}; session ranges from 09:30 ET",
                    "args": {"symbols": chunk, "notional_usd": 3000, "sleeve_usd": 30000},
                    "time": f"{night0}T09:15:00-05:00", "sleeve": "absr1-check", "value": 30000.0}
            gw, args, _, _ = setup(fake)
            cost.live = fixed_cost
            t31 = f3.run(gw, dict(args))
            gw, args, _, _ = setup(fake)
            cost.live = fixed_cost
            t4 = f4.run(gw, dict(args))
            a31, a4 = dict(absr1_tokens(t31, chunk)), dict(absr1_tokens(t4, chunk))
            c31, r31 = parse_rows(t31, chunk)
            _, r4 = parse_rows(t4, chunk)
            iv = col_index(c31, "vol20")
            for s in chunk:
                if s not in a31:
                    continue
                n_served += 1
                v4 = num(a4.get(s))
                if not v4 or (v4[0] is not None and v4[0] <= 0):
                    bad_e.append((as_of, s, a4.get(s)))
                if a4.get(s) != my_abs(r31[s][0][iv]) or not r4[s][1].endswith((f4.ABS_NOTE, f4.ABS_NA)):
                    bad_e.append((as_of, s, a31[s], a4.get(s), r31[s][0][iv], r4[s][1][-60:]))
                lines_out.append(f"{as_of} {s:6s} {'cache' if s in affected else 'on-demand'}: absr1 {a31[s]} -> {a4.get(s)} "
                                 f"(vol20 {r31[s][0][iv]}; note: {r4[s][1].split('; ')[-1]})")
    (TMP / "absr1_changes.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    check(f"e. ML4.0 served on all {n_served} stored rows with absr1 <= 0 (6-12 Nov): none <= 0, each the review's "
          f"vol20 equivalent and marked in its note", n_served > 0 and not bad_e, str(bad_e[:5]))

    # ---------------- f. writes (build4)
    for c in [c for c in calls if c["mc"] == "gptoss"][:60]:
        gw, args, as_of, now = setup(c)
        s0 = snapshot(TMP)
        f4.run(gw, dict(args), sleeve_value=c["value"], look=("f", c["file"]))
        s1 = snapshot(TMP)
        for k in changed(s0, s1):
            if not (TMP / k).resolve().is_relative_to(f4.OUT.resolve()):
                check(f"f. run() wrote outside runs/ml/forecast4: {k}", False)
    computed = None
    try:
        s0 = snapshot(TMP)
        f4._MEM.clear()
        as_of, rows = f4.compute(trader.Data(last="20261112"))
        new = changed(s0, snapshot(TMP))
        outside = [k for k in new if not (TMP / k).resolve().is_relative_to(f4.OUT.resolve())]
        check(f"f. compute() wrote only its own fc file ({new})",
              not outside and new == [str(Path("runs/ml/forecast4") / f"fc_{as_of}.json")], str(outside))
        computed = as_of
    except Exception:                                   # noqa: BLE001
        check("f. compute() runs offline", False, traceback.format_exc()[-400:])

    # ---------------- g. interface (build4)
    check("g. ml_tools and tools from this repo", Path(ml_tools.__file__).parent == ROOT and Path(tools.__file__).parent == ROOT)
    check("g. ml_tools.module('ML4') is forecast4 and version_of names ML4.0",
          ml_tools.module("ML4") is f4 and ml_tools.version_of({"ml": True, "ml_tool": "ML4"}) == "ML4.2"
          and ml_tools.unknown({"x": {"ml": True, "ml_tool": "ML4"}}) == []
          and ml_tools.in_use({"a": {"ml": True}, "b": {"ml": True, "ml_tool": "ML4"}, "c": {"ml": True, "ml_tool": "ML3"}}) == ["ML2", "ML3", "ML4"])
    p4, p2 = f4.SPEC["parameters"]["properties"], forecast.SPEC["parameters"]["properties"]
    check("g. SPEC: name forecast; symbols, notional_usd (ML2's text), sleeve_usd (ML4's own text from models-w3 f3b1f01: the "
          "value in the state, no $30,000 fallback named) and track_record",
          f4.SPEC["name"] == "forecast" and set(p4) == {"symbols", "notional_usd", "sleeve_usd", "track_record"}
          and p4["notional_usd"] == p2["notional_usd"] and p4["sleeve_usd"]["type"] == "number"
          and p4["sleeve_usd"]["description"] == ("Your sleeve's value, $, for the sleeve% column. Without it the tool uses "
                                                  "the value in your state.")
          and f4.SPEC["parameters"]["required"] == ["symbols"] and p4["track_record"]["type"] == "boolean")
    check("g. run(gw, args, sleeve_value=None, look=None) and compute(data=None)",
          list(inspect.signature(f4.run).parameters) == ["gw", "args", "sleeve_value", "look"]
          and [p.default for p in inspect.signature(f4.run).parameters.values()][2:] == [None, None]
          and list(inspect.signature(f4.compute).parameters) == ["data"])
    gw, args, as_of, now = setup(c0)
    via_tools = tools.run(gw, "forecast", dict(args), ml_tool="ML4", sleeve_value=c0["value"], look=("g", 1))
    gw, args, as_of, now = setup(c0)
    direct = f4.run(gw, dict(args), sleeve_value=c0["value"], look=("g", 2))
    check("g. tools.run(..., ml_tool='ML4') serves forecast4", via_tools == direct, via_tools[:200])
    gw, args, as_of, now = setup(c0)
    saved = f4.direction_column
    try:
        f4.direction_column = lambda a, ss: {s: "x" for s in ss}
        hooked = f4.run(gw, dict(args), sleeve_value=c0["value"], look=("g", 3))
    finally:
        f4.direction_column = saved
    hdr = next(x for x in hooked.split("\n") if x.startswith("sym close"))
    check("g. direction hook: None now (no 'dir' column, no text); a filled hook adds one column after gap1",
          f4.direction_column("20261110", ["AAPL"]) is None and " dir " not in direct and "gap1±%(10-90) dir " in hdr)

    # ================================================================ review additions
    # ---------------- h. an independent comparison: ML4's nightly document read from disk, my own parser and formula
    random.seed(20261115)
    all_calls = recorded_calls(versions=("ML2", "ML3"))
    sample = random.sample(all_calls, min(150, len(all_calls)))
    sample += [c for c in all_calls if any(t in c["result"] for t in (" -0.", "SGOV", "BOXX", "SHY "))][:60]
    disk = TMP / "h_ml4"
    f4_out = f4.OUT
    f4.OUT = disk
    disk.mkdir(parents=True, exist_ok=True)
    st_h = Counter()
    bad_h = []
    for k, c in enumerate(sample):
        syms = symbols_of(c["args"])
        a_ = body(c["result"])[19:29].replace("-", "")
        if not syms or fc3(a_) is None:
            st_h["skipped"] += 1
            continue
        if not (disk / f"fc_{a_}.json").exists():
            (disk / f"fc_{a_}.json").write_text(json.dumps(fc3(a_)), encoding="utf-8")
        for rep in (1, 2):                              # the look's first call, then a later one
            gw, args, _, _ = setup(c)
            t3 = f3.run(gw, dict(args), sleeve_value=c["value"], look=("h3", k))
            gw, args, _, _ = setup(c)
            f4._MEM.clear()                             # ML4 reads its own file from disk
            t4 = f4.run(gw, dict(args), sleeve_value=c["value"], look=("h4", k))
            for old, new in getattr(f4, "_PLAIN", ()):  # ML4.1's two plainly worded lines: wording only (as compare())
                t4 = t4.replace(new, old)
            if hasattr(f4, "_slot_label"):              # ML4.2's slot label, mapped as compare() maps it
                t3 = f4._slot_label(t3)
            c3, r3, L3, h3 = my_rows(t3, syms)
            c4, r4, L4, h4 = my_rows(t4, syms)
            st_h["calls"] += 1
            if c3 != c4 or list(r3) != list(r4) or len(L3) - h3 != len(L4) - h4 or h3 != h4:
                bad_h.append((c["file"], "shape"))
                continue
            for x, y in zip(L3[:h3], L4[:h4]):
                if x != y:
                    st_h["changed head lines " + y[:14]] += 1
                    if not (y in (f4.DIR_LINE, f4.GAP1_LINE) or x.split(f4.LOSS_CUT, 1)[0].rstrip() == y):
                        bad_h.append((c["file"], x[:60], y[:60]))
            ia, iv = c3.index("absr1±%"), c3.index("vol20%")
            for s in r3:
                (v3, n3), (v4, n4) = r3[s], r4[s]
                st_h["rows"] += 1
                if v3 is None:
                    bad_h += [] if (v3, n3) == (v4, n4) else [(c["file"], s, "not covered")]
                    continue
                try:
                    neg_ = float(v3[ia]) <= 0
                except ValueError:
                    neg_ = False
                for j, (x, y) in enumerate(zip(v3, v4)):
                    if j == ia and neg_:
                        continue
                    st_h["tokens"] += 1
                    if x != y:
                        bad_h.append((c["file"], s, c3[j], x, y))
                if neg_:
                    st_h["absr1 replaced"] += 1
                    want = my_abs(v3[iv])
                    clause = "absr1 from vol20 (the model's value was not above zero)" if want != "na" else \
                        "absr1 na (the model's value was not above zero; no vol20)"
                    if v4[ia] != want or n4 != (f"{n3}; {clause}" if n3 else clause):
                        bad_h.append((c["file"], s, v3[ia], v4[ia], want, n4[-60:]))
                elif n3 != n4:
                    bad_h.append((c["file"], s, "note", n3, n4))
    f4.OUT = f4_out
    check(f"h. independent: {st_h['calls']} renders of {len(sample) - st_h['skipped']} recorded calls (ML2 and ML3 "
          f"sleeves, first and later call), ML4's file read from disk: {st_h['rows']} rows, {st_h['tokens']} tokens equal, "
          f"{st_h['absr1 replaced']} absr1 replaced exactly as stated", st_h["calls"] >= 200 and not bad_h, str(bad_h[:4]))

    # pick calls for the failure paths: one in the session with several names, one before the open
    insess = next(c for c in calls if "session ranges at the" in c["result"][:120] and len(symbols_of(c["args"])) >= 4)
    preopen = next(c for c in calls if "session ranges from 09:30 ET" in c["result"][:120] and len(symbols_of(c["args"])) >= 3)

    def ml2_text(c, gw=None, extra=()):
        g, args, _, _ = setup(c)
        args = dict(args, symbols=symbols_of(args) + list(extra))
        return forecast.run(gw or g, args, sleeve_value=c["value"], look=None), args

    # ---------------- i. a stale night (a failed refresh): the line is added, and no row moves
    for c in (insess, preopen):
        t2, args = ml2_text(c)
        gw, _, as_of, now = setup(c)
        f3._now = lambda n=now + timedelta(days=1): n           # the night of the document is no longer today's
        try:
            t4 = f4.run(gw, dict(args), sleeve_value=c["value"])
            err = None
        except Exception as e:                                    # noqa: BLE001
            t4, err = "", repr(e)
        syms = symbols_of(args)
        probs = ml2_view(t2, t4, syms) if not err else [err]
        stale_line = any(x.startswith("gap1 unavailable (its forecasts are for the night after the ") for x in t4.split("\n"))
        check(f"i. stale night ({c['time'][:16]}): 'gap1 unavailable' line, gap1 na on every row, every ML2 token under its own "
              f"symbol", stale_line and not probs and set(gap1_tokens(t4, syms)) == {"na"}, str(probs[:3]))

    # ---------------- j. no ML4 file for the close (installed after the refresh), and no model file
    t2, args = ml2_text(insess)
    gw, _, as_of, now = setup(insess)
    f4._MEM.clear()
    f4_out, f4.OUT = f4.OUT, TMP / "empty_forecast4"
    try:
        t4 = f4.run(gw, dict(args), sleeve_value=insess["value"])
    finally:
        f4.OUT = f4_out
    syms = symbols_of(args)
    check("j. no ML4 file: 'gap1 unavailable (FileNotFoundError: no ML4 forecasts for this close)', gap1 na, rows in place",
          "gap1 unavailable (FileNotFoundError: no ML4 forecasts for this close)" in t4 and set(gap1_tokens(t4, syms)) == {"na"}
          and not ml2_view(t2, t4, syms), str(ml2_view(t2, t4, syms)[:3]))
    gw, _, as_of, now = setup(insess)
    f4.MODEL = TMP / "no_such_model.json"
    try:
        t4 = f4.run(gw, dict(args), sleeve_value=insess["value"])
    finally:
        f4.MODEL = MODEL4
    check("j. no gap1 model file: 'gap1 unavailable (FileNotFoundError ...)', gap1 na, rows in place",
          "gap1 unavailable (FileNotFoundError" in t4 and set(gap1_tokens(t4, syms)) == {"na"} and not ml2_view(t2, t4, syms))

    # ---------------- k. names not covered, and a gateway that fails everything (quote and bars too)
    extra = ["ZZZQ", "BRK B", "X.Y"]
    for c, label in ((insess, "in session"), (preopen, "before the open")):
        t2, args = ml2_text(c, extra=extra)
        gw, _, as_of, now = setup(c)
        t3 = f3.run(gw, dict(args), sleeve_value=c["value"])
        gw, _, as_of, now = setup(c)
        t4 = f4.run(gw, dict(args), sleeve_value=c["value"])
        syms = symbols_of(args)
        nc = [x for x in t4.split("\n") if x.startswith(("ZZZQ not covered", "X.Y not covered"))]
        brk = [x for x in t4.split("\n") if x.startswith("BRK B ") and "not covered" not in x]   # a covered name with a space
        check(f"k. [{label}] names not covered keep ML2's line, other rows (BRK B too) in place, same as ML3.1",
              len(nc) == 2 and len(brk) == 1 and not ml2_view(t2, t4, syms) and not compare(t3, t4, syms, {"rows": 0, "tokens": 0, "absr1_changed": 0, "changes": []}),
              str((nc, ml2_view(t2, t4, syms)[:2])))
        b2, _ = ml2_text(c, gw=BrokenGW(now, as_of), extra=extra)
        gw, _, as_of, now = setup(c)
        try:
            b4 = f4.run(BrokenGW(now, as_of), dict(args), sleeve_value=c["value"])
            gw, _, as_of, now = setup(c)
            b3 = f3.run(BrokenGW(now, as_of), dict(args), sleeve_value=c["value"])
            err = None
        except Exception as e:                                    # noqa: BLE001
            b4, b3, err = "", "", repr(e)
        mn = next((x for x in b4.split("\n") if x.startswith("market now:")), "")
        want_mn = "SPY pre-market quote unavailable" if label == "before the open" else "SPY's 30-minute bars unavailable"
        check(f"k. [{label}] a gateway that fails every request: text, '{want_mn}', rows in place, same as ML3.1",
              err is None and want_mn in mn and not ml2_view(b2, b4, syms)
              and not compare(b3, b4, syms, {"rows": 0, "tokens": 0, "absr1_changed": 0, "changes": []}), err or mn)

    # ---------------- l. ML2 has no forecasts; bad arguments: text, never an exception
    gw, args, as_of, now = setup(insess)
    forecast.live_provider = lambda: None
    t4 = f4.run(gw, dict(args))
    t4r = f4.run(gw, dict(args, track_record=True))
    check("l. ML2 with no forecasts: its own line, unchanged (with the record after it on request)",
          t4 == "no forecasts are available for this date" and t4r == t4 + "\n\n" + f4.TRACK_RECORD)
    gw, args, as_of, now = setup(insess)
    weird = [None, {}, {"symbols": 5}, {"symbols": "AAPL, MSFT"}, {"symbols": [None, 3, "aapl"]}, {"symbols": ["AAPL"], "notional_usd": "x"},
             {"symbols": ["AAPL"], "sleeve_usd": -5}, {"symbols": ["AAPL"], "track_record": "maybe"}, {"symbols": ["AAPL"], "track_record": [1]},
             {"symbols": [], "track_record": False}, {"symbols": [""] * 30, "track_record": True}, "not a dict"]
    outs, errs = [], []
    for w in weird:
        try:
            o = tools.run(gw, "forecast", copy.deepcopy(w), ml_tool="ML4", sleeve_value=30000.0, look=None)
            outs.append(o)
            if not isinstance(o, str):
                errs.append((w, type(o)))
        except Exception as e:                                    # noqa: BLE001
            errs.append((w, repr(e)))
    check(f"l. {len(weird)} malformed calls through tools.run: each returns text", not errs, str(errs[:3]))
    raised = []
    for w in weird[:-1]:
        try:
            f4.run(gw, copy.deepcopy(w))
        except Exception as e:                                    # noqa: BLE001
            raised.append((w, type(e).__name__))
        try:
            f3.run(gw, dict(w or {}))
        except Exception as e:                                    # noqa: BLE001
            raised.append(("ML3.1", w, type(e).__name__))
    say(f"   (information) direct run() exceptions on malformed args, ML4.0 and ML3.1 alike; tools.run turns them into text: {raised}")

    # ---------------- m. track_record: every form, rows identical with and without it, legend for the first rows call
    gw, args, as_of, now = setup(insess)
    yes = [True, "true", "TRUE", " yes ", 1, 1.0, "1"]
    no = [False, "false", 0, None, "no", 2, "", [True], {"x": 1}]
    rec_only = [f4.run(gw, {"symbols": [], "track_record": v}) for v in yes]
    check("m. track_record true, 'true', 'yes', 1 and '1' with no symbols: the record alone", all(x == f4.TRACK_RECORD for x in rec_only))
    no_rec = [f4.run(gw, {"symbols": [], "track_record": v}) for v in no]
    check("m. false, 0, None, 'no', 2, '' and other values with no symbols: ML2's own error, no record",
          all(x == "error: give at least one symbol" for x in no_rec), str(set(no_rec)))
    plain_rows, with_rec = [], []
    for v in yes[:3]:
        gw, _, as_of, now = setup(insess)
        plain_rows.append(f4.run(gw, dict(args), sleeve_value=insess["value"]))
        gw, _, as_of, now = setup(insess)
        with_rec.append(f4.run(gw, dict(args, track_record=v), sleeve_value=insess["value"]))
    check("m. with symbols, the reply is the plain reply then the record: rows identical with and without it",
          all(w == p + "\n\n" + f4.TRACK_RECORD for p, w in zip(plain_rows, with_rec)))
    for v in no:
        gw, _, as_of, now = setup(insess)
        x = f4.run(gw, dict(args, track_record=v), sleeve_value=insess["value"])
        if x != plain_rows[0]:
            check(f"m. track_record={v!r} with symbols is the plain reply", False, x[:120])
    gw, _, as_of, now = setup(insess)
    key = ("m", "look")
    r0 = f4.run(gw, {"symbols": [], "track_record": True}, sleeve_value=1, look=key)
    gw, _, as_of, now = setup(insess)
    r1 = f4.run(gw, dict(args), sleeve_value=insess["value"], look=key)
    gw, _, as_of, now = setup(insess)
    r2 = f4.run(gw, dict(args), sleeve_value=insess["value"], look=key)
    check("m. a record-only call leaves the legend for the look's first rows call; the next call prints the one-line reference",
          r0 == f4.TRACK_RECORD and f4.DIR_LINE in r1 and f4.GAP1_LINE in r1 and forecast.LEGEND_REF not in r1
          and forecast.LEGEND_REF in r2 and f4.DIR_LINE not in r2 and f4.GAP1_LINE not in r2)
    check("m. sleeve_value is the sleeve% base when sleeve_usd is left out, as in ML3.1",
          "your sleeve's value in your state" in f4.run(setup(insess)[0], {"symbols": symbols_of(args)}, sleeve_value=12345.0)
          and "$12,345 sleeve" in f4.run(setup(insess)[0], {"symbols": symbols_of(args)}, sleeve_value=12345.0))

    # ---------------- n. fix_absr1 edge cases
    cases = [(("0", "10"), ("0.5", f4.ABS_NOTE)), (("-0", "10"), ("0.5", f4.ABS_NOTE)), (("-0.2", "na"), ("na", f4.ABS_NA)),
             (("-0.2", "0"), ("na", f4.ABS_NA)), (("-0.2", "nan"), ("na", f4.ABS_NA)), (("na", "10"), ("na", None)),
             (("0.01", "10"), ("0.01", None)), (("1.3", "na"), ("1.3", None)), (("-0.1", "0.05"), ("0.0025", f4.ABS_NOTE)),
             (("-0.1", None), ("na", f4.ABS_NA)), ((None, "10"), (None, None))]
    got = [(a, f4.fix_absr1(*a)) for a, _ in cases]
    check("n. fix_absr1: <= 0 (and -0) replaced and marked, na without vol20, positive and na left as they are",
          all(g == w for (_, g), (_, w) in zip(got, cases)), str([(a, g) for (a, g), (_, w) in zip(got, cases) if g != w]))

    # ---------------- o. the morning: compute() writes ML4's file; a look the next session reads it from disk
    if computed:
        nxt = next((c for c in all_calls if body(c["result"])[19:29].replace("-", "") == computed
                    and "session ranges at the" in c["result"][:120]), None)
        gw, args, as_of, now = setup(nxt)
        f4._MEM.clear()
        t4 = f4.run(gw, dict(args), sleeve_value=nxt["value"])
        g = gap1_tokens(t4, symbols_of(args))
        check(f"o. after compute() for the {computed} close, a look of {nxt['time'][:10]} reads ML4's own file (gap1 on "
              f"{sum(x != 'na' for x in g)} of {len(g)} rows, no 'gap1 unavailable')",
              "gap1 unavailable" not in t4 and g and sum(x != "na" for x in g) >= len(g) - 1, t4.split("\n")[1][:120])

    # ---------------- p. ML2 and ML3 untouched; nothing in the repo's runs/ml changed
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "HEAD", "--", "forecast.py", "forecast3.py"],
                          capture_output=True, text=True).stdout.strip()
    check("p. git diff HEAD shows nothing for forecast.py and forecast3.py, or only a new revision of their own",
          diff == "" or all(bumped(n) for n in touched), diff)
    check("p. ml_tools.changed_files() is [] (ML2's manifest guard)", ml_tools.changed_files() == [], str(ml_tools.changed_files()))
    after_live = {str(w): snapshot(w) for w in watch}
    moved = {w: changed(before_live[w], after_live[w]) for w in before_live}
    check("p. nothing changed in the repo's runs/ml/forecast, forecast3, research during the test",
          not any(moved.values()), str({w: v[:5] for w, v in moved.items() if v}))

    # ---------------- q. the system text of a sleeve on ML4 (trader.S3_ML4): S3_ML says "its description gives each model's
    # out-of-sample track record, so weigh it accordingly", which is false for ML4 (the record comes on request)
    head_src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:trader.py"], capture_output=True, text=True,
                              encoding="utf-8").stdout
    head = types.ModuleType("trader_head")
    head.__file__ = trader.__file__         # the committed source reads its folder from __file__
    exec(compile(head_src, "trader_head.py", "exec"), head.__dict__)
    day = datetime(2026, 11, 16).date()
    combos = [(ml, news, kelly, mg) for ml in (False, True) for news in (True, False) for kelly in (False, True)
              for mg in (2.0, None, 3.0)]
    base = {c: trader.live_system(90000, day, ml=c[0], news=c[1], kelly=c[2], max_gross=c[3]) for c in combos}
    others = all(trader.live_system(90000, day, ml=c[0], news=c[1], kelly=c[2], max_gross=c[3], ml_tool=t) == base[c]
                 for c in combos for t in (None, "ML2", "ML3", "ML9"))
    check("q. the system text is unchanged for every sleeve not on ML4 (no ml_tool, ML2, ML3, an unknown one), whatever "
          "its other terms", others)
    if hasattr(head, "S3_ML4"):             # once this is committed HEAD has it too: compare the texts ML2 and ML3 read
        check("q. S3_ML is HEAD's", trader.S3_ML == head.S3_ML)
    else:
        check("q. the system text of every sleeve not on ML4 equals HEAD's, byte for byte",
              all(base[c] == head.live_system(90000, day, ml=c[0], news=c[1], kelly=c[2], max_gross=c[3]) for c in combos))
    with4 = {c: trader.live_system(90000, day, ml=c[0], news=c[1], kelly=c[2], max_gross=c[3], ml_tool="ML4") for c in combos}
    check("q. a sleeve on ML4 with the tool reads S3_ML4 once and not S3_ML; nothing else in its system text differs",
          all((with4[c] == base[c].replace(trader.S3_ML, trader.S3_ML4) and with4[c].count(trader.S3_ML4) == 1
               and trader.S3_ML not in with4[c]) if c[0] else with4[c] == base[c] for c in combos))
    check("q. S3_ML4 says nothing of a track record in the description and gives no advice",
          not re.search(r"track record|weigh|should|must|avoid|careful", trader.S3_ML4, re.I)
          and trader.S3_ML4.startswith("3. Test it with the scanner and the forecast tool (machine-learning forecasts):"),
          trader.S3_ML4)
    live_src = (ROOT / "live.py").read_text(encoding="utf-8")
    check("q. live.py gives live_system the sleeve's ml_tool", re.search(
        r"trader\.live_system\([^)]*ml_tool=meta\.get\(\"ml_tool\"\)", live_src, re.S) is not None)

    say(f"elapsed {time.time() - t_start:.0f} s; temp folder {TMP}")
    ok = all(results)
    say("ALL OK" if ok else f"FAILED: {results.count(False)} of {len(results)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
