"""F12 live smoke for the forecast tool before Monday (runs/ml/research/forecast_model_acceptance_criteria.md v1.1),
modelled on runs/ml/research/wave_c_livecheck.py. Read-only research client 11 ("trader-research"): no orders, only what
forecast.run asks for (bars, stats, quotes, details, plus EDGAR and the macro file).

    .venv\\Scripts\\python.exe ml_research\\forecast_livesmoke.py            # gateway up: runs/ml/research/forecast_livesmoke.json
    .venv\\Scripts\\python.exe ml_research\\forecast_livesmoke.py --dry      # no gateway: a stub that refuses every request
    .venv\\Scripts\\python.exe ml_research\\forecast_livesmoke.py --compute  # the Monday 07:30 forecast.compute(), offline

Bars and checks, fixed before the run (forecast_fixes_prereg.md, section 5):
  - each call runs in a fresh Python process, as opus-ml's MCP server is a new process each look, and is timed there
    (forecast.run plus the cost and vol2 imports it would make itself): a 10-symbol call under 20 s; a 20-symbol call with
    eight names outside the cached universe (AGG and JNK bond ETFs) under 45 s. A second call in that process is the warm
    run (the gpt-oss loop has the cache loaded; a model may ask again): for the twenty set, with eight new outside names
    (Amendment 2), and outside_names_scored_total sums the names scored in both runs;
  - every fallback names itself in the note, in one wording ("previous model", never "HAR fallback");
  - every outside name gets risk numbers, AGG and JNK with the on-demand bond treatment; TLT and HYG carry F4's note;
  - opus-ml's MCP tool and gpt-oss's forecast.SPEC carry the same description and parameter descriptions.
Each process scores outside names into its own temp folder (cold, as on Monday when the close is new) and keeps vol2's
live cache in a temp copy, so runs/ml/forecast/ondemand and runs/ml/research/vol2_livecache.json are not touched. Columns
are read by the column line's names. The two sets share no symbol, so the 8-K check is cold for both. The fresh call
imports live.py for its holiday list (forecast._prev_session; about 5.5 s on 24 Oct), as in opus-ml's MCP server; the
gpt-oss loop has live.py loaded already.

What a Sunday run cannot show: the session columns (next60, rest_of_day) are computed only 09:30-16:00 ET on weekdays,
so their gateway time is not in these timings; and the header says "stale" until Monday's 07:30 refresh (the cache
ends Thursday; ml_research/staleness_test.py checks the label itself).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import forecast  # noqa: E402
from ml_research import bondfx  # noqa: E402

ET = ZoneInfo("America/New_York")
RES = ROOT / "runs" / "ml" / "research"
BARS = {"ten": 20.0, "twenty": 45.0}                     # seconds, F12, on the fresh-process time
TEN = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH", "HD", "KO", "SPY", "XLE"]
TWENTY = ["AMZN", "GOOGL", "META", "TSLA", "BAC", "CVX", "PFE", "WMT", "QQQ", "IWM", "TLT", "HYG",   # cached (TLT, HYG bond)
          "AGG", "JNK", "SOFI", "RIVN", "MARA", "RKLB", "IONQ", "CELH"]     # outside: two bond ETFs, two in data/bars, four not
SETS = {"ten": TEN, "twenty": TWENTY}
OUTSIDE = ["AGG", "JNK", "SOFI", "RIVN", "MARA", "RKLB", "IONQ", "CELH"]   # each must get risk numbers
WARM_OUTSIDE = ["LCID", "PLUG", "SOUN", "UPST", "AFRM", "HIMS", "DKNG", "SNOW"]   # the warm run asks for new names
WARM = {"ten": TEN, "twenty": TWENTY[:12] + WARM_OUTSIDE}  # (forecast_fixes_prereg.md Amendment 2)
BOND = {"AGG", "JNK"}                                     # outside bond ETFs: the on-demand bond treatment
F4 = ("TLT", "HYG")                                       # cached bond funds: F4's note
RISK = ("range1±%(10-90)", "absr1±%", "vol20%")
HEAD_FALLBACKS = ("(stale: not from the last session)", "events unavailable", "macro calendar unavailable",
                  "session ranges unavailable", "no session ranges on half-days", "market: no SPY implied vol today")


def timed(fn, *a):
    t = time.time()
    out = fn(*a)
    return out, round(time.time() - t, 2)


def description_check():
    """The description opus-ml's MCP server serves for `forecast` (listed through the server itself, as the Claude CLI
    sees it) against forecast.SPEC, which live.py hands gpt-oss: the text and every parameter's description."""
    import asyncio
    tmp = Path(tempfile.mkdtemp())
    os.environ.update(TRADER_SUBMIT_FILE=str(tmp / "submit.json"), TRADER_LOG_FILE=str(tmp / "log.jsonl"),
                      TRADER_HAS_FORECAST="1")
    import mcp_server                                      # registers the tools; connects only on a research call
    tool = next(t for t in asyncio.run(mcp_server.server.list_tools()) if t.name == "forecast")
    a, b = tool.description, forecast.SPEC["description"]
    props = tool.input_schema.get("properties", {})
    params = {k: {"spec": v.get("description"), "mcp": (props.get(k) or {}).get("description")}
              for k, v in forecast.SPEC["parameters"]["properties"].items()}
    return {"identical": a == b, "chars_mcp": len(a), "chars_spec": len(b),
            "first_difference_at": None if a == b else next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))),
            "parameters_differ": {k: v for k, v in params.items() if v["spec"] != v["mcp"]}}


class DryGW:
    """--dry: the real Gateway's methods, each refusing at once (no connection is made); other attributes are missing,
    as on the real gateway, so forecast.run takes its live path."""

    def __init__(self):
        try:
            from tws import Gateway
            self._names = {n for n in dir(Gateway) if not n.startswith("_") and callable(getattr(Gateway, n))}
        except Exception:                                  # noqa: BLE001
            self._names = {"bars", "stats", "quotes", "details", "news", "article", "tape", "scan"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        if name.startswith("_") or name not in self.__dict__.get("_names", ()):
            raise AttributeError(name)

        def refuse(*a, **k):
            raise ConnectionError(f"dry run: no gateway ({name})")
        return refuse


def table(txt):
    """(column names, header text, {symbol: {column name: token, ..., "note": rest}}) from one forecast result, cut by the
    column line's names (every column is one token, the note is the rest); a "not covered" line gives {"not covered": why}."""
    lines = txt.splitlines()
    k = next((i for i, ln in enumerate(lines) if ln.startswith("sym close ")), None)
    if k is None:
        return None, txt, {}
    names = lines[k].split(" ")
    n = len(names) - 1                                     # columns before the note
    rows = {}
    for ln in lines[k + 1:]:
        tok = ln.split(" ")
        if ln.startswith(f"{tok[0]} not covered"):
            rows[tok[0]] = {"not covered": ln[len(tok[0]) + 1:]}
        else:
            rows[tok[0]] = dict(zip(names[:n], tok[:n]), note=" ".join(tok[n:]))
    return names, "\n".join(lines[:k]), rows


def pooled_fallbacks(model, ship, grp):
    """absr1 / vol20 served by the previous model where the shipped one is not, less the columns F4 serves from the fund's
    own history or withholds (bondfx.apply drops those from the note's fallback part; B keeps it)."""
    fb = [t for t in ("absr1", "vol20") if ship.get(t) != "har" and model.get(t) == "har"]
    if grp:
        dec = bondfx.info()["decision"][grp]
        fb = [t for t in fb if dec.get(t) in (None, "current", "B")]
    return fb


def fallback_check(txt, syms, cached, scored, ship, costs):
    """Every fallback in one forecast result names itself. cached: the cache's rows; scored: the rows _ondemand produced
    in this call (with vol2's `model`); ship: vol2's shipped variant per target (a target shipped as HAR is no fallback);
    costs: cost.live's rows for the outside names (a "not covered" line must not hide a gateway failure in their note)."""
    names, header, rows = table(txt)
    if names is None:
        return {"ok": False, "problems": ["no column line"], "seen": [], "rows": {}, "wording": {}}
    seen = [p for p in HEAD_FALLBACKS if p in header]
    problems, kinds = [], {}
    for s in syms:
        r = rows.get(s)
        if r is None:
            problems.append(f"{s}: no line")
            continue
        if "not covered" in r:
            kinds[s] = "not covered"
            if not re.fullmatch(r"not covered \(.+\)", r["not covered"]):
                problems.append(f"{s}: not covered without a reason")
            n = (costs.get(s) or {}).get("note") or ""
            if n and "time cap" not in n and n not in r["not covered"]:
                problems.append(f"{s}: not covered, the line hides cost.live's note ({n})")
            continue
        note = r["note"]
        kind = kinds[s] = "cached" if s in cached else "scored" if s in scored else "cost only"
        need = []
        if any(r[c].startswith("na") for c in RISK) and not note:
            need.append("a reason for the missing risk columns")
        if r["spread_bps(10-90)"].startswith("na") and not note:
            need.append("a reason for the missing cost columns")
        if r.get("next60±%(10-90)") == "na" and not note:
            need.append("a reason for the missing session columns")
        if kind == "cached" and cached[s].get("note") and cached[s]["note"] not in note:
            need.append("the cache's note")
        if kind == "cost only" and "cost from live data (rough)" not in note:
            need.append("'cost from live data (rough)'")
        if kind == "scored":
            if "outside the validated universe" not in note:
                need.append("'outside the validated universe'")
            grp = bondfx.ONDEMAND if "bond, credit or currency fund" in note else bondfx.group(s)
            fb = pooled_fallbacks(scored[s].get("model") or {}, ship, grp)
            if fb and "previous model for" not in note:
                need.append(f"the fallback for {', '.join(fb)}")
        if need:
            problems.append(f"{s} ({kind}): note lacks {'; '.join(need)}")
    body = "\n".join(txt.splitlines()[len(header.splitlines()) + 1:])
    return {"ok": not problems, "problems": problems, "seen": seen, "rows": kinds,
            "wording": {w: body.count(w) for w in ("HAR fallback", "previous model for")}}


def outside_check(txt, syms):
    """Every outside name gets risk numbers (range1 not na); AGG and JNK with the on-demand bond treatment (the note holds
    bondfx.clause("on demand") as this code writes it, and vol20 is withheld); TLT and HYG carry F4's note."""
    _, _, rows = table(txt)
    out, f4, on, n = [], [], bondfx.clause(bondfx.ONDEMAND), 0
    for s in (x for x in OUTSIDE + WARM_OUTSIDE if x in syms):
        r = rows.get(s) or {"not covered": "no line"}
        if "not covered" in r or r["range1±%(10-90)"].startswith("na"):
            out.append(f"{s}: no risk numbers ({r.get('not covered') or r.get('note')})")
            continue
        n += 1
        if s in BOND and (on not in r["note"] or r["vol20%"] != "na"):
            out.append(f"{s}: no on-demand bond treatment (vol20 {r['vol20%']}; note: {r['note']})")
    for s in (x for x in F4 if x in syms):
        want = bondfx.clause(bondfx.group(s))
        if want not in (rows.get(s) or {}).get("note", ""):
            f4.append(f"{s}: note lacks F4's clause \"{want}\" (a cache written before F4: run forecast.compute())")
    return {"outside_ok": not out, "outside_scored": n, "outside_problems": out, "f4_ok": not f4, "f4_problems": f4}


def one_call(name, dry, out):
    """One set in this process, which the parent started fresh: forecast.run timed cold, then warm (WARM: for the twenty
    set, eight outside names the process has not seen, as Amendment 2 asks). The
    rows behind each text (cost.live's and _ondemand's) are kept for the checks; results go to `out` (JSON)."""
    import shutil
    t = time.time()
    from ml_research import cost, vol2                     # forecast.run would import these itself: timed with it
    import_s = time.time() - t
    syms, bar = SETS[name], BARS[name]
    tmp = Path(tempfile.mkdtemp())
    forecast.ONDEMAND = tmp / "ondemand"                   # cold on-demand rows; the live folder is not touched
    if vol2.LIVECACHE.exists():                            # IV and 30-minute history as live has it, written to a copy
        shutil.copy(vol2.LIVECACHE, tmp / "vol2_livecache.json")
    vol2.LIVECACHE = tmp / "vol2_livecache.json"
    got = forecast.live_provider()
    if not got:
        raise SystemExit(f"no forecast cache in {forecast.OUT}")
    as_of, fc = got
    cached, ship = fc["rows"], vol2._info()["ship"]
    scored, costs, ondemand, cost_live = {}, {}, forecast._ondemand, cost.live

    def spy(*a, **k):                                      # the rows behind the text, for fallback_check
        rows, why = ondemand(*a, **k)
        scored.update(rows)
        return rows, why

    def cost_spy(*a, **k):
        o = cost_live(*a, **k)
        costs.update({x["symbol"]: x for x in o.get("orders", [])})
        return o
    forecast._ondemand, cost.live = spy, cost_spy
    if dry:
        from ml_research import events

        def no_edgar(*a, **k):
            raise ConnectionError("dry run: no EDGAR")
        events.live = no_edgar
        ctx = DryGW()
    else:
        from tws import Gateway
        ctx = Gateway("trader-research", timeout=60)
    res = {"set": name, "symbols": syms, "cache_as_of": as_of, "cache_file": f"fc_{as_of}.json", "bar_s": bar,
           "import_s": round(import_s, 2), "outside": [s for s in syms if s not in cached]}
    with ctx as gw:
        for run in ("fresh", "warm"):
            scored.clear()
            costs.clear()
            call = syms if run == "fresh" else WARM[name]
            t = time.time()
            txt = forecast.run(gw, {"symbols": call})
            sec = round(time.time() - t + (import_s if run == "fresh" else 0), 2)
            res[run] = {"symbols": call, "seconds": sec, "under_bar": sec < bar, "text": txt,
                        "fallbacks": fallback_check(txt, call, cached, dict(scored), ship, dict(costs)),
                        **outside_check(txt, call)}
            print(txt, f"\n=== {name} {run}: {sec} s (bar {bar}), fallbacks ok {res[run]['fallbacks']['ok']}",
                  res[run]["fallbacks"]["problems"], res[run]["outside_problems"], res[run]["f4_problems"], flush=True)
    Path(out).write_text(json.dumps(res), encoding="utf-8")


def smoke(dry=False):
    res = {"started_et": f"{datetime.now(ET):%Y-%m-%d %H:%M:%S}", "dry": dry, "client": "trader-research (11)",
           "bars_s": BARS, "description": description_check()}
    tmp = Path(tempfile.mkdtemp())
    for name in SETS:                                      # each set in a fresh process
        out = tmp / f"{name}.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--call", name, "--out", str(out)]
                       + (["--dry"] if dry else []), check=True, cwd=str(ROOT))
        res[name] = json.loads(out.read_text(encoding="utf-8"))
    runs = [res[n][r] for n in SETS for r in ("fresh", "warm")]
    d = res["description"]
    res["pass"] = {"ten_under_20s_fresh": res["ten"]["fresh"]["under_bar"],
                   "twenty_under_45s_fresh": res["twenty"]["fresh"]["under_bar"],
                   "fallbacks_name_themselves": all(r["fallbacks"]["ok"] for r in runs),
                   "one_fallback_wording": not any(r["fallbacks"]["wording"].get("HAR fallback") for r in runs),
                   "outside_names_scored_fresh": res["twenty"]["fresh"]["outside_ok"],
                   "outside_names_scored_warm": res["twenty"]["warm"]["outside_ok"],
                   "outside_names_scored_total": res["twenty"]["fresh"]["outside_scored"] + res["twenty"]["warm"]["outside_scored"],
                   "tlt_hyg_carry_f4_note": all(res["twenty"][r]["f4_ok"] for r in ("fresh", "warm")),
                   "same_description": d["identical"] and not d["parameters_differ"]}
    res["warm"] = {**{f"{n}_s": res[n]["warm"]["seconds"] for n in SETS},
                   **{f"{n}_under_bar": res[n]["warm"]["under_bar"] for n in SETS}}
    if dry:
        res["pass"].update(ten_under_20s_fresh=None, twenty_under_45s_fresh=None, outside_names_scored_fresh=None,
                           outside_names_scored_warm=None, outside_names_scored_total=None)
        res["warm"] = {k: None for k in res["warm"]}
        res["note"] = "dry run: every gateway request and EDGAR refused, so the timings and rows exercise the harness only"
    out = RES / ("forecast_livesmoke_dry.json" if dry else "forecast_livesmoke.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"pass": res["pass"], "warm": res["warm"], "description": res["description"]}, indent=1), "\nwrote", out)
    return res


def refresh_log():
    """The 07:30 refresh job's times from runs/live/live.log: start, forecasts cached, job end (the Alpha Vantage line)."""
    p = ROOT / "runs" / "live" / "live.log"
    out = {}
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) ET  (refresh starting|forecasts for the ML arm|Alpha Vantage)", ln)
        if m:
            out.setdefault(m[1], {})[{"refresh starting": "start", "forecasts for the ML arm": "forecasts",
                                      "Alpha Vantage": "end"}[m[3]]] = m[2]
    return {d: v for d, v in out.items() if "forecasts" in v}


def compute_check():
    """The Monday 07:30 path offline: forecast.compute() on data/ as it stands, with no fitted vol2 models on disk for the
    close (on Monday the close is new, so they are fitted then) and every output in a temp folder. Also: the rows
    against the cached file for the same close, every previous-model fallback named in its note, and the rows whose spread came
    from cost's cold model (compute adds no note for that)."""
    import shutil
    import numpy as np
    from ml_research import cost, regime, vol2
    ref, tmp = forecast.OUT, Path(tempfile.mkdtemp())
    (tmp / "vol2").mkdir()
    shutil.copy(vol2.OUT / "vol2.json", tmp / "vol2")
    if vol2.LIVECACHE.exists():
        shutil.copy(vol2.LIVECACHE, tmp / "lc.json")
    forecast.OUT, vol2.OUT, vol2.LIVECACHE = tmp / "forecast", tmp / "vol2", tmp / "lc.json"
    parts, got = {}, {}

    def clocked(mod, name):
        f = getattr(mod, name)

        def w(*a, **k):
            t, key = time.time(), f"{mod.__name__.split('.')[-1]}.{name}"
            got[key] = f(*a, **k)
            parts[key] = round(time.time() - t, 1)
            return got[key]
        setattr(mod, name, w)
    for mod, name in ((vol2, "live"), (cost, "build"), (regime, "live")):
        clocked(mod, name)
    (as_of, rows), seconds = timed(forecast.compute)
    print(f"forecast.compute: {seconds} s, as of {as_of}, {len(rows)} rows, parts {parts}", flush=True)
    v = got["vol2.live"]
    ship = v["ship"]
    unnamed, har, none = [], 0, 0
    for s, r in rows.items():
        m = (v["rows"].get(s) or {}).get("model") or {}
        fb = pooled_fallbacks(m, ship, bondfx.group(s))
        nn = [x for x in m if m[x] == "none"]
        har, none = har + bool(fb), none + bool(nn)
        if (fb and "previous model for" not in r["note"]) or (nn and not r["note"]):
            unnamed.append(s)
    d, cal, S, F, y, have = got["cost.build"]
    sm = cost._model()["spread"]
    p = cost.predict(sm, cost.rows(F, sm["names"], [len(d.dates) - 1]))
    cold = [s for j, s in enumerate(d.sym) if s in rows and not np.isfinite(p[j])]
    diff, maxd, rp = [], 0.0, ref / f"fc_{as_of}.json"
    if rp.exists():
        old = json.loads(rp.read_text(encoding="utf-8"))["forecasts"]
        for s in sorted(set(old) | set(rows)):
            a, b = old.get(s) or {}, rows.get(s) or {}
            if a != b:
                diff.append(s)
                maxd = max([maxd] + [abs(a[x] - b[x]) for x in set(a) & set(b)
                                     if isinstance(a[x], (int, float)) and isinstance(b[x], (int, float))])
    res = {"run_et": f"{datetime.now(ET):%Y-%m-%d %H:%M:%S}", "data": str(ROOT / "data"), "as_of": as_of, "rows": len(rows),
           "seconds": seconds, "parts_s": parts, "bar": "the 07:30 job ends before 09:00 ET",
           "har_fallback_rows": har, "no_value_rows": none, "fallback_rows_unnamed": unnamed,
           "spread_cold_model_rows": len(cold), "spread_cold_model_examples": cold[:10],
           "vs_cached_file": {"file": str(rp), "exists": rp.exists(), "rows_differing": len(diff), "examples": diff[:10],
                              "max_abs_numeric_diff": round(maxd, 6)},
           "refresh_log": refresh_log()}
    (RES / "forecast_livesmoke_compute.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no gateway: a stub that refuses every request")
    ap.add_argument("--compute", action="store_true", help="time forecast.compute() offline on data/ instead")
    ap.add_argument("--call", choices=sorted(SETS), help=argparse.SUPPRESS)      # one set in this process (smoke's child)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.call:
        one_call(a.call, a.dry, a.out)
    else:
        compute_check() if a.compute else smoke(dry=a.dry)
