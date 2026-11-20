"""ML2.3 and ML3.1 (the ML3 review's fixes 1-4, 15 Nov 2026; Clayton, 14 Nov: removing the sizing advice is a fix, so it
reaches ML2 too). Offline: no gateway, no Tinker, no network; the stored forecast files are only read (runs/ml/forecast,
runs/ml/forecast3, data/bars30/SPY.json), and anything the tools would write goes to a temp folder.

    set PYTHONIOENCODING=utf-8 & .venv\\Scripts\\python.exe forecast_fixes_test.py     # ALL OK and exit 0, else exit 1

1. Numbers: the committed modules (git show fd95e2a:forecast.py, fd95e2a:forecast3.py, loaded from a temp folder) and the new
   ones on the same stored inputs (5 closes x 3 symbol lists, live path with session ranges, events and the market-now
   line; the simulator path; a pre-market look) print the same row lines byte for byte, the same header, market, macro
   and events lines, the same numbers in the legend lines and the same SPY move and range in the market-now line.
2. Fix 1: sleeve_value is the sleeve% base when sleeve_usd is left out, sleeve_usd still wins, the default stays $30,000,
   and only the sleeve% tokens change.
3. Fix 2: the new description is the old one less exactly the advice sentences (and the sleeve% line), nothing else.
4. Fix 3: the market-now line names the time of its print. Fix 4: with a look key the legend prints once per look.
5. ML3.1: a stale night no longer shifts the rows (each row keeps its own symbol).
6. Token counts of the description before and after (gpt-oss tokenizer, offline).
"""
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the tokenizer from the local cache only
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import forecast             # noqa: E402
import forecast3 as f3      # noqa: E402
from ml_research import events, macro_feats    # noqa: E402

ET = ZoneInfo("America/New_York")
TMP = Path(tempfile.mkdtemp(prefix="forecast_fixes_test_"))
results = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or not detail else f": {detail}"))
    results.append(bool(ok))


REF = "fd95e2a"


def head_module(path, name, replace=()):
    # fd95e2a, the commit before the fixes: HEAD holds them since 7b363da (15 Nov), so it is no longer the old code
    src = subprocess.run(["git", "show", f"{REF}:{path}"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout
    assert "def run(" in src, path
    for a, b in replace:
        assert a in src, a
        src = src.replace(a, b, 1)
    p = TMP / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


old2 = head_module("forecast.py", "forecast_head")
old3 = head_module("forecast3.py", "forecast3_head", [("\nimport forecast\n", "\nimport forecast_head as forecast\n")])
# the committed modules sit in TMP, so their ROOT-relative paths are pointed back at the repo (read-only use); every
# folder either one would write into goes to TMP
old2.ROOT, old2.OUT, old2.RES = forecast.ROOT, forecast.OUT, forecast.RES
old3.ROOT, old3.OUT, old3.MODEL, old3.BARS, old3.BARS30, old3.MACRO = f3.ROOT, f3.OUT, f3.MODEL, f3.BARS, f3.BARS30, f3.MACRO
for m in (old2, forecast):
    m.ONDEMAND = TMP / f"{m.__name__}_ondemand"
for m in (old3, f3):
    m.ONDEMAND = TMP / f"{m.__name__}_ondemand"
check("committed versions loaded", old2.VERSION == "ML2.2" and old3.VERSION == "ML3.0" and old3.forecast is old2,
      f"{old2.VERSION} {old3.VERSION}")
check("new versions", forecast.VERSION == "ML2.4" and f3.VERSION == "ML3.2", f"{forecast.VERSION} {f3.VERSION}")

DATES = ["20261106", "20261109", "20261110", "20261111", "20261112"]
LISTS = [["AAPL", "NVDA", "HYG", "TLT", "SPY", "BRK B"], ["AMD", "XLU", "SHY", "PTC", "QQQ"], ["MSFT", "GOOGL", "LQD", "IEF"]]
FC2 = {d: json.loads((forecast.OUT / f"fc_{d}.json").read_text(encoding="utf-8")) for d in DATES}
FC3 = {d: json.loads((f3.OUT / f"fc_{d}.json").read_text(encoding="utf-8")) for d in DATES}
SPY30 = json.loads((f3.BARS30 / "SPY.json").read_text(encoding="utf-8"))["bars"]
NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)*")


def sess_of(syms):
    """Deterministic session ranges (high-low spans, %), one per name, the same for both versions."""
    return {s: tuple(round(0.4 + 0.1 * (i + k), 2) for k in range(6)) for i, s in enumerate(syms)}


def patch(as_of, now, ev):
    """Both versions' live path on stored inputs: ML2's cache file for as_of, ML3's for as_of, fixed clock, canned 8-K
    check and macro line, deterministic session ranges."""
    for m in (old2, forecast):
        m.live_provider = lambda a=as_of: (a, {"market": copy.deepcopy(FC2[a]["market"]), "rows": copy.deepcopy(FC2[a]["forecasts"])})
        m._today_et = lambda n=now: n.date()
        m._prev_session = lambda today, a=as_of: a
        m._session = lambda gw, syms, iv_hint=None, n=now: (f"session ranges at the {n:%H:%M} look", sess_of(syms))
    for m in (old3, f3):
        m._now = lambda n=now: n
        m._MEM.clear()
        m._MEM[as_of] = copy.deepcopy(FC3[as_of])
    events.live = lambda gw, covered, base=None, **kw: {"rows": {s: dict(ev[s]) for s in covered if s in ev}}
    macro_feats.macro_today = lambda *a, **k: "macro today: no scheduled US releases"


class GW:
    """SPY's stored 30-minute bars for the previous session and today up to `now`, and a pre-market quote."""

    def __init__(self, now, prev):
        self.now, self.prev = now, prev

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, **kw):
        if (symbol, bar_size) != ("SPY", "30 mins"):
            return []
        day = f"{self.now:%Y%m%d}"
        return [b for b in SPY30 if b["date"][:8] == self.prev or (b["date"][:8] == day and b["date"][9:14] <= f"{self.now:%H:%M}")]

    def quotes(self, symbols, primary="", timeout=12.0):
        return [{"symbol": "SPY", "last": 681.5, "close": 677.0, "bid": 681.4, "ask": 681.6, "last_time": f"{self.now:%Y-%m-%d}T13:58:00Z"}]


class SimGW:
    def __init__(self, as_of):
        self.as_of = as_of

    def forecast_provider(self):
        return self.as_of, {"market": dict(FC2[self.as_of]["market"], macro="macro today: none"),
                            "rows": copy.deepcopy(FC2[self.as_of]["forecasts"])}

    def session_ranges(self, syms):
        return "session ranges at the 11:00 look", sess_of(syms)


LEGEND_STARTS = ("cost_bps: round trip", "move$ ±(90) and sleeve%", "No column forecasts direction", "gap1 ±(10-90):",
                 "cost_bps and move$ for a")


def parts(text):
    """{kind: [lines]}: legend lines, the market-now line, and every other line (rows, header, market, macro, events)."""
    out = {"legend": [], "now": [], "data": []}
    for ln in text.split("\n"):
        out["legend" if ln.startswith(LEGEND_STARTS) else "now" if ln.startswith("market now:") else "data"].append(ln)
    return out


def now_numbers(lines):
    """The SPY move and range in a market-now line (the times and the delay are new text, not forecast numbers)."""
    got = []
    for ln in lines:
        mv = re.search(r"SPY ([+-][\d.]+)% since the last close", ln)
        rg = re.search(r"range since the close ([\d.]+)%", ln)
        got.append((mv and mv.group(1), rg and rg.group(1), "unavailable" in ln))
    return got


GAIN_0 = ": the expected gain from direction is 0"      # ML2.4 (15 Nov 2026): this clause of the direction line is out


def legend_numbers(lines):
    lines = [ln.replace(GAIN_0, "", 1) for ln in lines]     # the committed text's clause and its 0 are not a forecast number
    return [NUM.findall(ln) for ln in lines if not ln.startswith("gap1 ±(10-90):")] + \
           [NUM.findall(ln) for ln in lines if ln.startswith("gap1 ±(10-90):")]


def same_numbers(name, old, new):
    """Every number the old text prints, the new one prints the same: data lines byte for byte, legend numbers, the
    market-now move and range."""
    a, b = parts(old), parts(new)
    ok = a["data"] == b["data"] and legend_numbers(a["legend"]) == legend_numbers(b["legend"]) \
        and now_numbers(a["now"]) == now_numbers(b["now"])
    diff = next((f"{x!r}\n   vs {y!r}" for x, y in zip(a["data"], b["data"]) if x != y), "") or \
        f"legend {legend_numbers(a['legend'])} vs {legend_numbers(b['legend'])}; now {a['now']} vs {b['now']}"
    check(name, ok, diff)
    return ok


def ev_for(as_of, syms):
    """A calendar earnings release tonight for the list's second name and our estimate for its third (the gap1
    multipliers and the events line), as ml3_test.py stubs them."""
    night = FC3[as_of]["night"]
    return {syms[1]: {"calendar_day": night[0], "calendar_when": "after close", "calendar_sessions": 0, "calendar_reported": False},
            syms[2]: {"next_earn_day": night[1], "next_earn_when": "before open", "next_earn_sessions": 1, "next_earn_overdue": False}}


def main():
    # ---------------------------------------------------------------- 1. numbers: committed vs new, same inputs
    n_cmp = 0
    for as_of in DATES:
        night0 = FC3[as_of]["night"][0]
        for k, syms in enumerate(LISTS):
            for hhmm in ("10:32", "14:05"):
                now = datetime.strptime(f"{night0} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
                args = {"symbols": syms, "notional_usd": [5000, 2500, 12000][k]} | ({"sleeve_usd": 90000} if k == 1 else {})
                patch(as_of, now, ev_for(as_of, syms))
                gw = GW(now, as_of)
                o2, n2 = old2.run(gw, args), forecast.run(gw, args)
                o3, n3 = old3.run(gw, args), f3.run(gw, args)
                same_numbers(f"ML2 {as_of} list {k} {hhmm}", o2, n2)
                same_numbers(f"ML3 {as_of} list {k} {hhmm}", o3, n3)
                n_cmp += 2
                if as_of == DATES[-1] and k == 0 and hhmm == "10:32":
                    print("\n---- ML3.0 (committed)\n" + o3 + "\n---- ML3.1 (new)\n" + n3 + "\n----")
    # the simulator path, and a pre-market look (09:05: the quote branch)
    for as_of in DATES[:2]:
        args = {"symbols": LISTS[0] + ["NOPE"], "notional_usd": 4000}
        for m in (old3, f3):
            m._MEM.clear()
            m._MEM[as_of] = copy.deepcopy(FC3[as_of])
        same_numbers(f"ML2 sim {as_of}", old2.run(SimGW(as_of), args), forecast.run(SimGW(as_of), args))
        same_numbers(f"ML3 sim {as_of}", old3.run(SimGW(as_of), args), f3.run(SimGW(as_of), args))
        night0 = FC3[as_of]["night"][0]
        now = datetime.strptime(f"{night0} 09:05", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        patch(as_of, now, {})
        same_numbers(f"ML3 pre-market {as_of}", old3.run(GW(now, as_of), args), f3.run(GW(now, as_of), args))
        n_cmp += 3
    print(f"     ({n_cmp} committed/new pairs compared)")

    # ---------------------------------------------------------------- 2. fix 1: the sleeve's own value
    as_of, syms = DATES[-1], LISTS[0]
    night0 = FC3[as_of]["night"][0]
    now = datetime.strptime(f"{night0} 11:02", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    patch(as_of, now, {})
    gw, a = GW(now, as_of), {"symbols": syms, "notional_usd": 5000}
    base = forecast.run(gw, a)
    check("no sleeve value: the $30,000 default, said so", "of a $30,000 sleeve (the default: no sleeve value given)" in base)
    check("no sleeve value: the committed numbers", parts(base)["data"] == parts(old2.run(gw, a))["data"])
    st = forecast.run(gw, a, sleeve_value=90000.0)
    check("sleeve_value: rows equal the committed code given sleeve_usd 90,000",
          parts(st)["data"] == parts(old2.run(gw, dict(a, sleeve_usd=90000)))["data"])
    check("sleeve_value: named as the state's", "of a $90,000 sleeve (your sleeve's value in your state)" in st)
    tok = re.compile(r" ±[\d.]+\([\dna.]+\) (±[\d.]+\([\dna.]+\)|na)(?= |$)")
    strip_pct = lambda t: [tok.sub(" SLEEVE", ln) for ln in parts(t)["data"]]
    check("sleeve_value: only the sleeve% tokens change", strip_pct(st) == strip_pct(base) and st != base)
    aapl = lambda t: next(ln for ln in t.split("\n") if ln.startswith("AAPL "))
    pct = lambda t: float(re.search(r"±[\d.]+\([\d.na]+\) ±([\d.]+)\(", aapl(t)).group(1))
    check("sleeve% at $90,000 is a third of the default's", abs(pct(base) / 3 - pct(st)) <= 0.006, f"{pct(base)} {pct(st)}")
    passed = forecast.run(gw, dict(a, sleeve_usd=45000), sleeve_value=90000.0)
    check("sleeve_usd still wins", "of a $45,000 sleeve (the sleeve_usd you passed)" in passed
          and parts(passed)["data"] == parts(old2.run(gw, dict(a, sleeve_usd=45000)))["data"])
    for bad in ("abc", True, 0, -5, float("nan"), float("inf")):
        t = forecast.run(gw, dict(a, sleeve_usd=bad), sleeve_value=90000.0)
        check(f"unusable sleeve_usd {bad!r}: the state's value", "of a $90,000 sleeve (your sleeve's value in your state)" in t)
        t = forecast.run(gw, dict(a, sleeve_usd=bad), sleeve_value=bad)
        check(f"unusable both {bad!r}: the default, as before", "of a $30,000 sleeve (the default" in t
              and parts(t)["data"] == parts(old2.run(gw, dict(a, sleeve_usd=bad)))["data"])
    t = forecast.run(gw, a, sleeve_value=5e8)
    check("sleeve_value clamped as sleeve_usd is", "of a $10,000,000 sleeve" in t)
    t3 = f3.run(gw, a, sleeve_value=90000.0)
    check("ML3 passes sleeve_value to ML2", "of a $90,000 sleeve (your sleeve's value in your state)" in t3)
    P = forecast.SPEC["parameters"]["properties"]["sleeve_usd"]["description"]
    check("sleeve_usd's description says what is true with or without the caller's value",
          "current value from your state when the tool is given it" in P and "30,000" in P and "default 30,000" not in P, P)
    check("description's sleeve% line", "pass your sleeve's current value" not in forecast.DESCRIPTION
          and "your sleeve's current value from your state when the tool is given it, else $30,000" in forecast.DESCRIPTION)

    # ---------------------------------------------------------------- 3. fix 2: the advice sentences, and only they, are gone
    ADVICE = [" Use them to size positions, to set stop and limit distances, and to skip trades whose cost eats much of the "
              "expected move.",
              ", which is about how much gross exposure keeps your risk at its usual level",
              " A stop closer than twice the rest_of_day ± value's 10% figure is likely to be touched by noise."]
    OLD_SLEEVE = "- sleeve% ±(90): the same two moves as % of sleeve_usd (default $30,000, a sleeve's starting value: pass your sleeve's current value)."
    NEW_SLEEVE = next(ln for ln in forecast.DESCRIPTION.split("\n") if ln.startswith("- sleeve% ±(90):"))
    rebuilt = old2.DESCRIPTION
    for s in ADVICE:
        check(f"advice in the committed text: {s[:40]!r}", s in rebuilt)
        rebuilt = rebuilt.replace(s, "", 1)
    rebuilt = rebuilt.replace(OLD_SLEEVE, NEW_SLEEVE, 1)
    # ML2.4 (15 Nov 2026; Clayton: "Apply to all sleeves running ML."): the rest of the paragraph after "No return or
    # direction forecast is offered." is out of ML2's and ML3's description
    COSTS = (" Out of sample, none of the models we tested beat trading costs. That covers overnight gaps, intraday momentum, "
             "reversal, volume and time-of-day, news sentiment, industry, factor timing, momentum ranking and market timing. "
             "Our anomaly flag did not predict drops. Earnings drift was tested and did not beat trading costs: over 1,821 "
             "earnings releases (Oct 2025-Oct 2026), the move from 10:30 ET on the reaction day to the close two sessions "
             "later, in the direction of the first reaction, averaged -9 bps against SPY, against round-trip costs of 14-18 bps.")
    check("ML2.4's cut is in the committed text, once, right after its first sentence",
          rebuilt.count("**No return or direction forecast is offered.**" + COSTS + "\n") == 1)
    rebuilt = rebuilt.replace(COSTS, "", 1)
    check("new description = committed less the advice and ML2.4's cost paragraph (sleeve% line aside), byte for byte",
          rebuilt == forecast.DESCRIPTION)
    check("ML2.4: the paragraph's first sentence stays, alone on its line; nothing of the rest is left in ML2 or ML3",
          "\n**No return or direction forecast is offered.**\n" in forecast.DESCRIPTION
          and not any(s in d for d in (forecast.SPEC["description"], f3.SPEC["description"])
                      for s in ("none of the models we tested", "beat trading costs", "1,821", "14-18 bps")))
    for s in ("Use them to size", "keeps your risk", "likely to be touched by noise", "pass your sleeve's"):
        check(f"gone from ML2 and ML3: {s!r}", s not in forecast.SPEC["description"] and s not in f3.SPEC["description"])
    check("kept: the measured stop record and size_scale's definition",
          "a stop at the band edge (one ± value away) was touched in about 40% of sessions" in forecast.DESCRIPTION
          and "size_scale = typical SPY vol / expected vol." in forecast.DESCRIPTION)
    old3_rest = old3.SPEC["description"].replace(old2.DESCRIPTION, "", 1)
    new3_rest = f3.SPEC["description"].replace(forecast.DESCRIPTION, "", 1)
    check("ML3's own section: only the market-now time added",
          new3_rest == old3_rest.replace("with its range since the close. It is a fact",
                                         "with its range since the close and the time (ET) of the delayed print it uses. "
                                         "It is a fact", 1), new3_rest[:300])
    check("plain ASCII in what changed", all(ord(c) < 128 for c in "".join([P, NEW_SLEEVE.replace("±", ""), forecast.LEGEND_REF])))

    # ---------------------------------------------------------------- 4. fix 3: the market-now time; fix 4: legend once a look
    patch(as_of, now, {})
    nl = next(ln for ln in f3.run(GW(now, as_of), a).split("\n") if ln.startswith("market now:"))
    check("market-now names its print's time (11:02 look: the 10:30 bar, about 10:47 ET)",
          "as of about 10:47 ET (the latest print of the 10:30 bar, delayed about 15 minutes)" in nl and "look" not in nl, nl)
    late = datetime.strptime(f"{night0} 15:40", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    patch(as_of, late, {})
    nl = next(ln for ln in f3.run(GW(late, as_of), a).split("\n") if ln.startswith("market now:"))
    check("a bar already complete: its end (15:40 look: the 15:00 bar, 15:25 ET)", "as of about 15:25 ET (the latest print of the 15:00 bar" in nl, nl)
    op = datetime.strptime(f"{night0} 09:41", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    patch(as_of, op, {})
    nl = next(ln for ln in f3.run(GW(op, as_of), a).split("\n") if ln.startswith("market now:"))
    check("before the first bar ends: the 09:30 open", "since the last close at the 09:30 ET open (delayed about 15 minutes)" in nl, nl)
    pm = datetime.strptime(f"{night0} 09:05", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    patch(as_of, pm, {})
    nl = next(ln for ln in f3.run(GW(pm, as_of), a).split("\n") if ln.startswith("market now:"))
    check("pre-market: the quote's last trade time", "(pre-market quote, last trade 08:58 ET, delayed about 15 minutes)" in nl, nl)

    patch(as_of, now, ev_for(as_of, syms))
    gw = GW(now, as_of)
    full = f3.run(gw, a, look=("base-ml3", "2026-11-13 11:00"))
    second = f3.run(gw, dict(a, symbols=LISTS[1], notional_usd=2000), sleeve_value=90000.0, look=("base-ml3", "2026-11-13 11:00"))
    other = f3.run(gw, a, look=("haiku-ml", "2026-11-13 11:00"))
    nolook = [f3.run(gw, a) for _ in range(2)]
    L = lambda t: parts(t)["legend"]
    check("first call of a look: the full legend", len(L(full)) == 4 and forecast.LEGEND_REF not in full, L(full))
    check("first call: the same text as no look at all", full == nolook[0])
    check("later call: one reference line with its own size and sleeve",
          L(second) == [f"cost_bps and move$ for a $2,000 order, sleeve% of a $90,000 sleeve (your sleeve's value in your state); "
                        f"{forecast.LEGEND_REF}"], L(second))
    check("later call: rows, market, events and market-now lines stay", parts(second)["data"] == parts(old3.run(gw, dict(
        a, symbols=LISTS[1], notional_usd=2000, sleeve_usd=90000)))["data"] and len(parts(second)["now"]) == 1)
    check("another sleeve's look: its own full legend", other == full)
    check("no look key: the full legend every call (callers that pass none yet)", nolook[0] == nolook[1] == full)
    err = forecast.run(gw, {"symbols": []}, look="L-err")
    check("a call that fails does not use up the look's legend", err.startswith("error") and
          len(parts(forecast.run(gw, a, look="L-err"))["legend"]) == 3)
    ml2_second = forecast.run(gw, a, look="L2") and forecast.run(gw, a, look="L2")
    check("ML2 alone: later call short too", parts(ml2_second)["legend"][-1].endswith(forecast.LEGEND_REF))
    try:
        import trader
        tk = trader.tokenizer()
        count, unit = (lambda s: len(tk.encode(s))), "gpt-oss tokens"
    except Exception as e:                                  # noqa: BLE001
        count, unit = len, f"characters (tokenizer unavailable: {type(e).__name__})"
    saved = count("\n".join(L(full))) - count("\n".join(L(second)))
    print(f"     legend: {count(chr(10).join(L(full)))} {unit} on a look's first call, {count(chr(10).join(L(second)))} on later "
          f"calls ({saved} saved a later call)")
    check("legend once saves something on a later call", saved > 0)

    # ---------------------------------------------------------------- 5. ML3.1: a stale night keeps each row's own symbol
    stale_now = datetime.strptime(f"{FC3[DATES[-2]]['night'][1]} 10:05", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    patch(DATES[-2], stale_now, {})
    gw = GW(stale_now, DATES[-2])
    n_st, o_st, ml2 = f3.run(gw, a), old3.run(gw, a), forecast.run(gw, a)
    strip = lambda t: [ln for ln in parts(t)["data"] if not ln.startswith(("gap1 unavailable", "sym close"))]
    no_gap = lambda t: [" ".join(ln.split(" ")[:1 + (ln.startswith("BRK B"))] + [x for j, x in enumerate(ln.split(" ")[1 + (ln.startswith("BRK B")):]) if j != 3])
                        if ln.split(" ")[0] in {s.split(" ")[0] for s in syms} else ln for ln in strip(t)]
    check("stale night: gap1 unavailable, rows as ML2's with na", "gap1 unavailable (its forecasts are for the night after" in n_st
          and no_gap(n_st) == strip(ml2), "\n".join(no_gap(n_st)[:8]) + "\n----\n" + "\n".join(strip(ml2)[:8]))
    check("stale night: the committed code shifted the rows (the bug fixed)", no_gap(o_st) != strip(ml2))

    # ---------------------------------------------------------------- 6. the description's size
    for nm, old, new in (("ML2", old2.SPEC, forecast.SPEC), ("ML3", old3.SPEC, f3.SPEC)):
        j = lambda s: s["description"] + json.dumps(s["parameters"])
        print(f"     {nm} description + parameters: {count(j(old))} -> {count(j(new))} {unit}")

    # ---------------------------------------------------------------- the absr1 finding (reported, not changed): a size
    # cannot be negative, but vol2's absr1 is a linear (OLS) model on levels, so very quiet names come out below zero
    neg = sorted({(s, r["absr1_pct"]) for d in DATES for s, r in FC2[d]["forecasts"].items()
                  if r.get("absr1_pct") is not None and r["absr1_pct"] < 0})
    print(f"     negative absr1 in the 5 stored closes (unchanged by this fix): {len(neg)} rows, e.g. {neg[:6]}")
    check("nothing written into the stored forecast folders", not any(p.stat().st_mtime > START for d in (forecast.OUT, f3.OUT)
                                                                      for p in d.glob("fc_*.json")))


START = __import__("time").time()
if __name__ == "__main__":
    try:
        main()
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        results.append(False)
    print("ALL OK" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)
