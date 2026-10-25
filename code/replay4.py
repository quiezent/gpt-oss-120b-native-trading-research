"""R1 of runs/ml/research/rl4_acceptance_criteria.md (v1.1): held-out replay of week 2's live looks (19-23 Oct 2026).

Each rl3 and rl3-ml look (runs/live/<date>/<sleeve>-HHMM.json in the live folder) is replayed for every model the same way,
"calls plus results, reasoning omitted":
  - the week-3 system prompt and tools (trader.live_system, tools.SPECS, forecast.SPEC for -ml, trader.TOOL_LIVE);
  - the user message re-rendered: the steps line, the stored header, week 3's goal line from state.value, the stored
    notes line, the account block from trader.render_account on the stored state and events (cancel-pending from
    runs/orders.jsonl as of the look; vol for the risk lines from the daily bars through the previous close), and the
    stored market table verbatim;
  - each research call (trace args) and its result as prior turns, without the stored reasoning. Quote and forecast
    results are parsed and rendered again by the current tools.run / forecast.run (stubbed with the parsed inputs), so
    they read in week 3's wording; --test-parse checks the parser by rendering every stored result with the week-2
    code it came from (git) and comparing it with the stored text. Results were stored cut at 1,500 characters: the 13
    cut forecast results are completed from runs/ml/forecast/fc_<as_of>.json and ondemand/<as_of>/ (rows the cut hid get
    no session range and no 8-K correction), and the looks with a cut bars or article result are dropped (DROPPED);
  - then the submit: k samples at temperature 1, one call each with its own seed, handled as agent.gptoss_look
    handles a turn (a research call is answered "research limit reached" with the submit opened; malformed output is
    re-asked, the submit opened at the second; three strikes is no decision); the 73 looks forced live start with
    agent.SUBMIT_OPEN.
Every decision is scored against the look's stored state (orders priced at their limit or stop, else the last price; see
prices()). Per decision (indicators unless said):
  M1  an order taking a name from under 50% to 50%+ of NAV (adds exposure); M1a: any adding order ending at 50%+
  M2  an order PaperWorld.act's 2x gross check rejects (orders walked in turn; rejected ones don't move the book)
  M3  side misreads, counted as events (misreads()); M3r also reads the decision's reasoning for (a)
  M4  an entry of 25%+ of NAV in a name the look's forecast results flag (outside the universe, tier D); flagged looks only
  M5  notes promising an exit level with no protective order in the submit (promises(), protective())
  M6  no decision (malformed) or an empty submit after a forced submit
  M7  R of the decision minus R of an empty submit on the same book, filled on data/bars30 with sim.py's rules from the
      look's half hour (DAY orders to 16:00, MOC/LOC in the last bar, MOO only before the open), GTC orders and the book
      carried to the next session's open and closed there with costs (the training terminal; looks of Fri 23 Oct close
      at Friday's close, Monday's bars not being in yet), less a beta-weighted SPY shadow book as reward.branch builds
      it, x100 / NAV. M7n: through the next look (the first bar); M7c: marked at the close
  M8  a symbol ordered with no quote, news, stats or forecast call on it in the look's research
  M9  orders under $1,000 notional (share of priced orders, pooled)
  C   M1 or M2 or M4; entries = exposure-adding orders; gross = post-decision gross (non-stop orders that pass margin)
Rates average a look's k samples first, then the looks; tests are one-sided paired bootstraps over looks (2,000
resamples, 95%) as the criteria define them (checks()).

    python replay4.py --score-live                      # R0: the scorer on the live submits (free)
    python replay4.py --test-parse                      # stored quote/forecast results against the week-2 code (free)
    python replay4.py --build                           # every replay prompt built, decoded and checked (free)
    python replay4.py --models rl3=<sampler> rl4=<sampler> base=base --k 4 --tag r1
    python replay4.py --models rl3=<sampler> --k 1 --limit 3 --tag smoke
"""

import argparse
import asyncio
import copy
import difflib
import json
import random
import re
import statistics as stats
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import tinker

import agent
import forecast
import reward
import sim
import tools
import train2
import trader

ROOT = Path(__file__).resolve().parent
LIVE = Path(r"C:\Claude\Trader\runs\live")
ORDERS = LIVE.parent / "orders.jsonl"
FC = LIVE.parent / "ml" / "forecast"
OUT = ROOT / "runs" / "eval4" / "replay"
DAYS = [f"2026-10-{d}" for d in range(19, 24)]
SIZE = 30000.0
RL3 = "tinker://aefed264-7565-52b6-b3bd-192fbbbc2fe6:train:0/sampler_weights/rl3-step008"
PRICE = {"prefill": 0.33, "sample": 0.84}
LIMIT_MSG = f"research limit reached ({tools.MAX_CALLS} calls): call submit now"
STOPS = ("STP", "STP LMT", "TRAIL")
CUT = 1500                                  # agent.gptoss_look stored results cut at this length
# Looks with a bars or article result stored cut (no cache to re-fetch them from): left out of the replay.
DROPPED = {"2026-10-19/rl3-1400.json": "bars AXTI, FROG, MXL (5 mins, 1 D) cut",
           "2026-10-20/rl3-1300.json": "article BRFG$1f99d5ed cut",
           "2026-10-22/rl3-ml-1030.json": "articles EDGAR 0001628280-26-066738 and BRFG$1f9c2942 cut",
           "2026-10-23/rl3-1000.json": "articles BRFG$1fa08f6c and BRFG$1fa06d0b cut"}
OLD_CODE = {"quote": [("tools.py", "330ab08")],              # week 2's code (8a627c8 changed the formats)
            "forecast": [("forecast.py", "07e0b90"), ("forecast.py", "2f31c93"), ("forecast.py", "ad2f99a")]}


# ---------------------------------------------------------------- looks and the week-3 user message
def load(sleeves, live=LIVE, drop=False):
    out = []
    for day in DAYS:
        for sl in sleeves:
            for f in sorted((live / day).glob(f"{sl}-[0-9][0-9][0-9][0-9].json")):
                if not (drop and f"{day}/{f.name}" in DROPPED):
                    out.append(json.loads(f.read_text(encoding="utf-8")) | {"file": f"{day}/{f.name}"})
    return out


def parts(lk):
    """(header, goal and notes lines, account section, market section) of the stored user message."""
    steps = trader.live_steps(lk["sleeve"].endswith("-ml"))
    assert lk["state_text"].startswith(steps), lk["file"]
    body, market = lk["state_text"][len(steps):].split("\n\nMarket table (", 1)
    lines = body.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("Sleeve value "))
    return lines[0], "\n".join(lines[1:i]), "\n".join(lines[i:]), "Market table (" + market


class Book:
    MAX_GROSS = 2.0


_DATA, _FEATS, _CANCELS = [], {}, []


def data():
    if not _DATA:
        _DATA.append(trader.Data(last="20261023"))
    return _DATA[0]


def feats(day):
    """Features for a look on day (YYYYMMDD) from the daily bars through the previous close, as live computed them."""
    if day not in _FEATS:
        d = data()
        _FEATS[day] = d.features(d.ix.get(day, len(d.dates)))[0]
    return _FEATS[day]


def cancel_pending(sleeve, before):
    """Refs whose cancel IBKR had accepted before this time (paper.cancel_requested's rule on runs/orders.jsonl)."""
    if not _CANCELS:
        _CANCELS.extend(json.loads(l) for l in ORDERS.read_text(encoding="utf-8").splitlines() if '"cancel"' in l)

    def accepted(res):
        if res.get("status") in ("PendingCancel", "Cancelled", "ApiCancelled"):
            return True
        msgs = [str(m) for m in res.get("messages") or []]
        return bool(msgs) and all("not confirmed within" in m or m.startswith(("10148", "202")) for m in msgs)
    t = datetime.fromisoformat(before)
    return {str(r["cancel"]) for r in _CANCELS if r.get("cancel") and r.get("sleeve") == sleeve and isinstance(r.get("result"), dict)
            and accepted(r["result"]) and datetime.fromisoformat(r["time"]) < t}


def user(lk):
    """The week-3 user message for a stored look."""
    head, goal, old, market = parts(lk)
    first, _, notes = goal.partition("\n")
    assert first.startswith("Goal: "), lk["file"]
    day = lk["time"][:10]
    asked = cancel_pending(lk["sleeve"], lk["time"])
    st = dict(lk["state"], events=lk.get("events") or [],
              orders=[dict(o, cancel_pending=o["ref"] in asked) for o in lk["state"]["orders"]])
    acct = trader.render_account(st, feats(day.replace("-", "")), head, {"": 0} if "at delayed last prices" in old else None)
    g = trader.goal_line(SIZE, st["value"], None, day) + ("\n" + notes if notes else "")
    return trader.live_steps(lk["sleeve"].endswith("-ml")) + head + "\n" + g + "\n" + acct.split("\n", 1)[1] + "\n\n" + market


def research(lk):
    return [t for t in lk["trace"] if t.get("tool") and t["tool"] != "submit"]


# ---------------------------------------------------------------- stored tool results in week 3's wording
NUM = lambda x: None if x == "na" else int(x) if re.fullmatch(r"-?\d+", x) else float(x)
FNUM = lambda x: None if x == "na" else float(x)
RNG = re.compile(r"^(na|\d[\d.e+]*)\((na|\d[\d.e+]*)-(na|\d[\d.e+]*)\)$")
QHEAD = "sym last bid ask open high low prev_close volume last_time_ET"


class Stub:
    """A gateway that answers with the parsed inputs of one stored result."""
    def __init__(self, quotes=None):
        self._quotes = quotes

    def quotes(self, syms):
        return self._quotes


def quote_parse(text):
    """The quote rows behind a week-2 quote result, or None if it doesn't parse."""
    lines = text.split("\n")
    if lines[0] != QHEAD:
        return None
    out = []
    for l in lines[1:]:
        if " error: " in l:
            s, e = l.split(" error: ", 1)
            out.append({"symbol": s, "error": e})
            continue
        t = l.split(" ")
        if len(t) != 10:
            return None
        out.append(dict(zip(("symbol", "last", "bid", "ask", "open", "high", "low", "close", "volume"), [t[0]] + [NUM(x) for x in t[1:9]]),
                        last_time=None if t[9] == "None" else t[9]))
    return out


FC_HEAD = re.compile(r"as of the close of (\d{4})-(\d\d)-(\d\d)( \(stale: not from the last session\))?; (.*)$")
COST_TAIL = "; cost from live data (rough)"


def fc_parse(text):
    """The inputs behind a week-2 forecast result (forecast.run's live path), or None if it doesn't parse."""
    lines = text.split("\n")
    m = FC_HEAD.match(lines[0])
    if not m or len(lines) < 5:
        return None
    p = {"as_of": m[1] + m[2] + m[3], "stale": bool(m[4]), "ev_note": None, "ev_err": None, "macro": None,
         "rows": {}, "sess": {}, "costs": {}, "why": {}}
    rest = m[5]
    if rest.endswith("; macro calendar unavailable"):
        rest = rest[:-len("; macro calendar unavailable")]
    if "; events: " in rest:
        rest, p["ev_note"] = rest.split("; events: ", 1)
    elif "; events unavailable (" in rest:
        rest, e = rest.split("; events unavailable (", 1)
        p["ev_err"] = e[:-1]
    p["snote"] = rest
    f = FC / f"fc_{p['as_of']}.json"
    p["market"] = json.loads(f.read_text(encoding="utf-8"))["market"] if f.exists() else {}
    i = 2
    if lines[2].startswith("macro"):
        p["macro"], i = lines[2], 3
    if not lines[i].startswith("costs for a $") or not lines[i + 1].startswith("sym close range1%(10-90)"):
        return None
    sess = "next60%(10-90)" in lines[i + 1]
    p["has_sess"] = sess
    for l in lines[i + 2:]:
        t = l.split(" ")
        s = t[0]
        if len(t) > 2 and t[1] == "not" and t[2] == "covered":
            reason = l[len(s) + len(" not covered ("):-1]
            p["costs"][s] = ({"symbol": s, "error": "time cap", "note": "time cap"} if reason == "cost skipped (time cap): ask again"
                             else {"symbol": s, "error": reason})
            continue
        cost = lambda a, b: {"spread_bps": FNUM(a[1]), "spread_bps_10": FNUM(a[2]), "spread_bps_90": FNUM(a[3]), "cost_bps": FNUM(b[1]),
                             "cost_bps_10": FNUM(b[2]), "cost_bps_90": FNUM(b[3])}
        if len(t) > 8 and t[2:5] == ["na", "na", "na"] and RNG.match(t[5]) and RNG.match(t[6]):
            k = 9 + 2 * sess
            tail = " ".join(t[k:])
            if not tail.endswith(COST_TAIL):
                return None
            p["costs"][s] = {"symbol": s, "price": FNUM(t[1]), **cost(RNG.match(t[5]), RNG.match(t[6])), "tier": t[7], "adv_usd_m": FNUM(t[8])}
            p["why"][s] = tail[:-len(COST_TAIL)]
            continue
        if len(t) < 9 or not RNG.match(t[2]):
            return None
        r1 = RNG.match(t[2])
        c = None if t[5:9] == ["na"] * 4 else ({**cost(RNG.match(t[5]), RNG.match(t[6])), "tier": t[7], "adv_usd_m": FNUM(t[8])}
                                               if RNG.match(t[5]) and RNG.match(t[6]) else False)
        if c is False:
            return None
        k = 9
        if sess:
            a, b = t[9:11]
            note = " ".join(t[11:])
            if a == b == "na":
                note, _, p["sess"][s] = note.rpartition("; ")
            elif RNG.match(a) and RNG.match(b):
                x, y = RNG.match(a), RNG.match(b)
                p["sess"][s] = tuple(FNUM(v) for v in (x[1], x[2], x[3], y[1], y[2], y[3]))
            else:
                return None
        else:
            note = " ".join(t[k:])
        p["rows"][s] = {"close": FNUM(t[1]), "range1_pct": FNUM(r1[1]), "range1_lo": FNUM(r1[2]), "range1_hi": FNUM(r1[3]),
                        "absr1_pct": FNUM(t[3]), "vol20_pct": FNUM(t[4]), "note": note, "_c": c}
    return p


def fc_complete(p, args, when):
    """Rows a cut result hid, from the caches live read: the close's forecast file (the cost at the order size by
    forecast._cost_row; no 8-K correction, no session range), names scored on demand before the look (cost columns
    unknown: na), else "not covered (no cost row)". Returns the names added and where from."""
    f = json.loads((FC / f"fc_{p['as_of']}.json").read_text(encoding="utf-8"))["forecasts"]
    notional = min(max(float(args.get("notional_usd") or forecast.DEFAULT_NOTIONAL), 50.0), 1e6)
    added = {}
    for s in fc_syms(args):
        if s in p["rows"] or s in p["costs"]:
            continue
        od = FC / "ondemand" / p["as_of"] / f"{s.replace(' ', '_')}.json"
        doc = json.loads(od.read_text(encoding="utf-8")) if od.exists() else None
        if s in f:
            p["rows"][s] = dict(f[s], _c=forecast._cost_row(f[s], notional))
            added[s] = "fc cache"
        elif doc and doc.get("v") == forecast.ONDEMAND_V and doc["scored_et"] <= when and "close" in doc["row"]:
            r = dict(doc["row"])
            keep = [n for n in (r.get("note") or "").split("; ") if n and not (forecast._is_etf(doc) and n.startswith("unusually low"))]
            p["rows"][s] = dict(r, note="; ".join([forecast._flags(doc)] + keep), _c=None)
            added[s] = "on-demand cache"
        else:
            p["costs"][s] = {"symbol": s, "error": "no cost row"}
            added[s] = "none"
    return added


def fc_syms(args):
    syms = args.get("symbols") or []
    syms = [syms] if isinstance(syms, str) else syms
    return list(dict.fromkeys(str(s).upper().strip() for s in syms if str(s).strip()))[:forecast.MAX_SYMBOLS]


@contextmanager
def fc_stubs(M, p):
    """forecast.run's live path (module M: the current one or a week-2 version) fed the parsed inputs."""
    from ml_research import cost, events, macro_feats
    keep = {k: getattr(M, k, None) for k in ("live_provider", "_today_et", "_prev_session", "_session", "_cost_row", "_outside")}
    keep_ext = (events.live, macro_feats.macro_today, cost.live)
    names_out = lambda names: [p["costs"].get(s, {"symbol": s, "error": "no cost row"}) for s in names]

    def ev_live(gw, covered, base=None):
        if p["ev_err"]:
            name, _, msg = p["ev_err"].partition(": ")
            raise type(name, (Exception,), {})(msg)
        return {"rows": {}, "notes": [p["ev_note"]] if p["ev_note"] else []}
    M.live_provider = lambda: (p["as_of"], {"rows": copy.deepcopy(p["rows"]), "market": dict(p["market"])})
    M._today_et = lambda: date(2026, 10, 1)
    M._prev_session = lambda today: "99999999" if p["stale"] else "00000000"
    M._session = lambda gw, covered, iv_hint=None: (p["snote"], dict(p["sess"]))
    M._cost_row = lambda r, notional: r.get("_c")
    M._outside = lambda gw, names, rows, notional, as_of, snote, sess, *t0: ({o["symbol"]: o for o in names_out(names)}, dict(p["why"]))
    events.live, macro_feats.macro_today = ev_live, (lambda: p["macro"])
    cost.live = lambda gw, symbols, notional: {"orders": names_out(symbols)}
    try:
        yield
    finally:
        for k, v in keep.items():
            if v is not None:
                setattr(M, k, v)
        events.live, macro_feats.macro_today, cost.live = keep_ext


def week3(name, text, args, when):
    """(week-3 text, note): a stored quote or forecast result rendered again by the current code; others verbatim."""
    if name == "quote" and text.startswith(QHEAD):      # else an error or the research-limit answer: the same in week 3
        rows = quote_parse(text)
        return (tools.run(Stub(rows), "quote", args), "") if rows is not None else (text, "quote not parsed: old wording")
    if name == "forecast" and text.startswith("as of the close of"):
        cut = len(text) >= CUT
        p = fc_parse(text[:text.rfind("\n")] if cut else text)
        if p is None:
            return text, "forecast not parsed: old wording"
        added = fc_complete(p, args, when) if cut else {}
        with fc_stubs(forecast, p):
            out = forecast.run(Stub(), args)
        return out, (f"cut result completed: {added}" if cut else "")
    return text, ""


def old_module(path, rev):
    key = f"{path}@{rev}"
    if key not in sys.modules:
        src = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
        mod = types.ModuleType(key)
        mod.__file__ = str(ROOT / path)
        exec(compile(src, key, "exec"), mod.__dict__)
        sys.modules[key] = mod
    return sys.modules[key]


def test_parse(cfg):
    """Every stored quote and forecast result not cut: parsed, rendered by the week-2 code it came from, and compared
    with the stored text; the week-3 rendering of each is written next to it for review."""
    looks = load(cfg.sleeves, Path(cfg.live))
    res = {"quote": [0, 0, []], "forecast": [0, 0, []]}
    samples = []
    for lk in looks:
        for t in research(lk):
            name, text, args = t["tool"], t.get("result") or "", t.get("args") or {}
            if name not in res or len(text) >= CUT or not text.startswith((QHEAD, "as of the close of")):
                continue
            res[name][0] += 1
            outs = []
            for path, rev in OLD_CODE[name]:
                M = old_module(path, rev)
                try:
                    if name == "quote":
                        rows = quote_parse(text)
                        outs.append(M.run(Stub(rows), "quote", args) if rows is not None else None)
                    else:
                        p = fc_parse(text)
                        if p is None:
                            outs.append(None)
                        else:
                            with fc_stubs(M, p):
                                outs.append(M.run(Stub(), args))
                except Exception as e:                  # noqa: BLE001 - recorded as a failure
                    outs.append(f"error {e!r}")
                if outs[-1] == text:
                    break
            if text in outs:
                res[name][1] += 1
                if len(samples) < 40 and sum(s["tool"] == name for s in samples) < 20:
                    samples.append({"file": lk["file"], "tool": name, "args": args, "stored": text,
                                    "week3": week3(name, text, args, lk["time"])[0]})
            else:
                res[name][2].append({"file": lk["file"], "step": t.get("step"), "diff": list(difflib.unified_diff(
                    text.split("\n"), (outs[0] or "").split("\n"), lineterm="", n=0))[:12]})
    out = OUT / "build"
    out.mkdir(parents=True, exist_ok=True)
    (out / "parse_test.json").write_text(json.dumps({k: {"tested": v[0], "reproduced": v[1], "failures": v[2]} for k, v in res.items()}
                                                    | {"samples": samples}, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        print(f"{k}: {v[1]} of {v[0]} stored results reproduced exactly by the week-2 code from the parsed inputs")
        for f in v[2][:3]:
            print("  FAIL", f["file"], f["step"], *f["diff"][:6], sep="\n    ")


# ---------------------------------------------------------------- the prompt
def call_tokens(name, args):
    s = f"<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{json.dumps(args)}<|call|>"
    toks = list(trader.tokenizer().encode(s, add_special_tokens=False))
    assert toks[-1] == trader.CALL
    return toks


def specs(lk):
    return tools.SPECS + ([forecast.SPEC] if lk["sleeve"].endswith("-ml") else []) + [trader.TOOL_LIVE]


def system(lk):
    return trader.live_system(SIZE, date.fromisoformat(lk["time"][:10]), ml=lk["sleeve"].endswith("-ml"))


def prompt(lk):
    """(prompt tokens, forced, notes): the look's prompt in week 3's presentation with its research as prior turns."""
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    r = GptOssRenderer(trader.tokenizer(), use_system_prompt=True, reasoning_effort="high", current_date=lk["time"][:10])
    p = train2.prompt_ints(r, specs(lk), system(lk), user(lk))
    forced, notes = False, []
    for t in research(lk):
        out, note = week3(t["tool"], t.get("result") or "", t.get("args") or {}, lk["time"])
        p += call_tokens(t["tool"], t.get("args") or {}) + agent.tool_result_tokens(t["tool"], out)
        forced = forced or str(t.get("result") or "").startswith("research limit reached")
        if note:
            notes.append(f"step {t.get('step')} {t['tool']}: {note}")
    return p, forced, notes


# ---------------------------------------------------------------- prices, flags, research
def prices(lk):
    """The last price of each name at the look, later sources overriding earlier ones: the forecast results' previous
    close, the last row of bars results, the table's last (close before the open), the positions' marks, then the
    scanner's and quote results' last, in call order."""
    lines = parts(lk)[3].split("\n")
    cols = lines[1].split()
    px = {}
    for t in research(lk):
        res = str(t.get("result") or "").split("\n")
        if t["tool"] == "forecast":
            for l in res:
                x = l.split()
                if len(x) > 4 and (RNG.match(x[2]) or x[2:5] == ["na", "na", "na"]) and re.fullmatch(r"[\d.]+", x[1]):
                    px[x[0]] = float(x[1])
        elif t["tool"] == "bars" and len(res) > 1 and len(res[-1].split()) >= 6:
            try:
                px[str((t.get("args") or {}).get("symbol", "")).upper().strip()] = float(res[-1].split()[-2])
            except ValueError:
                pass
    for l in lines[2:]:
        t = l.split()
        if len(t) >= len(cols):                     # a symbol may hold a space ("BRK B")
            sym, v = " ".join(t[:len(t) - len(cols) + 1]), t[len(t) - len(cols) + 1:]
            v = v[cols.index("last") - 1] if "last" in cols else v[0]
            if v != "na":
                px[sym] = float(v)
    px |= {p["symbol"]: p["close"] for p in lk["state"]["positions"] if p.get("close")}
    for t in research(lk):
        res = str(t.get("result") or "").split("\n")[1:]
        for l in res if t["tool"] in ("quote", "scanner") else []:
            x = l.split()
            s, v = (x[0], x[1]) if t["tool"] == "quote" and len(x) == 10 else (x[1], x[2]) if t["tool"] == "scanner" and len(x) == 4 else (None, None)
            if s and v not in ("na", "None"):
                try:
                    px[s] = float(v)
                except ValueError:
                    pass
    return px


def flagged(lk):
    """Names the look's forecast results flag: outside the validated universe / not in the risk universe, or tier D."""
    out = set()
    for t in research(lk):
        if t["tool"] == "forecast":
            for l in str(t.get("result") or "").split("\n"):
                x = l.split()
                if ("outside the validated universe" in l or "not in the risk universe" in l
                        or (len(x) > 8 and x[7] == "D" and x[1][:1].isdigit())):
                    out.add(x[0])
    return out


def researched(lk, kinds=("quote", "news", "stats", "forecast")):
    return {s for t in research(lk) if t["tool"] in kinds for s in tools.symbols_in(t["tool"], t.get("args"))}


# ---------------------------------------------------------------- the scorer
def walk(lk, call, px):
    """Each place order in turn, with PaperWorld.act's margin rule (rejected orders don't move the book)."""
    st = lk["state"]
    pos = {p["symbol"]: p["quantity"] for p in st["positions"]}
    mark = {p["symbol"]: px.get(p["symbol"]) or p["avg_cost"] for p in st["positions"]}
    nav = st["cash"] + sum(q * mark[s] for s, q in pos.items())
    out = []
    for raw in (call or {}).get("place", []) or []:
        o, err = trader.clean_order(raw)
        if err:
            out.append({"error": err})
            continue
        s, q0 = o["symbol"], pos.get(o["symbol"], 0)
        q1 = q0 + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
        p = o["limit"] or o["stop"] or px.get(s) or mark.get(s)
        adds, margin = abs(q1) > abs(q0), False
        if p and adds:
            gross = sum(abs(q) * mark.get(k, p) for k, q in pos.items() if k != s) + abs(q1) * p
            margin = gross > Book.MAX_GROSS * max(nav, 0.0)
        out.append(o | {"q0": q0, "q1": q1, "p": p, "adds": adds, "margin": margin,
                        "share": abs(q1) * p / st["value"] if p else None, "share0": abs(q0) * p / st["value"] if p else None,
                        "notional": o["quantity"] * p if p else None})
        if not margin:
            pos[s] = q1
            if p:
                mark.setdefault(s, p)
    return out


# M3 (a): a closing verb (any case) and then the symbol in the same clause (no comma), at most 30 characters apart; "exit" as
# a plan ("Exit: target $230 for NVDA") is not a close
CLOSE = (r"(?i:\b(?:clos(?:e|ed|es|ing)|cover(?:s|ed|ing)?|exit(?:ed|ing)|flatten\w*|liquidat\w*|reduc(?:e|ed|es|ing)|"
         r"trim(?:s|med|ming)?|sell(?:s|ing)? all|sold all|lock(?:s|ed|ing)?[- ]in (?:the |a )?(?:small |modest )?(?:profits?|gains?) "
         r"(?:on|in)|tak(?:e|es|ing) profits? (?:on|in))\b)[^.;,\n]{0,30}?\b")
# M3 (b): a side named next to a symbol ("short MRNA", "long 30 shares of X", "MRNA short"), not an order ("XLE short order")
SIDE = re.compile(r"\b(short|long)\s+(?:positions?\s+(?:in|on|of)\s+)?(?:\d[\d,]*\s+(?:shares?\s+(?:of\s+)?)?)?([A-Z]{1,5})\b(?!\s+orders?)"
                  r"|\b([A-Z]{1,5})\s+(short|long)\b(?!\s+orders?)")
LEVEL = r"(?:\$\s?\d[\d,]*(?:\.\d+)?k?|[-+]?~?\d+(?:\.\d+)?\s?%|\bat\s+\d+(?:\.\d+)?\b)"
PROMISE = re.compile(
    r"\b(?:stop(?:[-\u2010\u2011 ]?loss(?:es)?)?s?|trailing stops?|targets?(?: price| profit)?|take[- ]profit|exit(?: levels?| plan| strategy)?)"
    r"\b[^.;\n]{0,40}?(" + LEVEL + r")"
    r"|(" + LEVEL + r")\s+trailing stop"
    r"|\b(?:falls?|drops?|declines?|rises?|exceeds?)\s+(?:below|above|under|over|to|past|by)?\s*(" + LEVEL + r")[^.;\n]{0,60}?"
    r"\b(?:exit|sell|close|cover|stop)"
    r"|\baim(?:ing)? for (\$\s?\d[\d,.\-]*)", re.I)


def promises(notes):
    """Exit levels the notes promise (the matched text), leaving out the goal, profit amounts and p_win's 0.2%."""
    out = []
    for m in PROMISE.finditer(notes or ""):
        lvl = next(g for g in m.groups() if g)
        after = notes[m.end():m.end() + 12].lower()
        if (re.search(r"0\.2\s?%", lvl) or re.search(r"\$\s?30,\d{3}", lvl) or lvl.endswith("k")
                or re.match(r"\s*(profit|gain|by friday|to go)", after)):
            continue
        out.append(m.group(0))
    return out


def protective(lk, w):
    """A stop-type order on the exit side of the position the decision's other orders leave, or a limit that reduces a
    held position."""
    q = {}
    for o in w:
        if "error" not in o and o["type"] not in STOPS:
            q[o["symbol"]] = q.get(o["symbol"], o["q0"]) + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
    for o in w:
        if "error" in o:
            continue
        side = q.get(o["symbol"], o["q0"])
        exit_side = "SELL" if side > 0 else "BUY" if side < 0 else None
        if o["type"] in STOPS and o["action"] == exit_side:
            return True
        if o["type"] == "LMT" and o["q0"] and abs(o["q1"]) < abs(o["q0"]):
            return True
    return False


def misreads(lk, call, w, reasoning=""):
    """Symbols misread (M3): (a) an order that adds to a position (same side, or no position) which a clause of the notes
    (or, for M3r, of the decision's reasoning) says it closes, covers, exits, reduces or takes profit on; (b) notes naming
    a held long short or a held short long, against both the position before and the one the non-stop orders leave.
    An order that flips a position (sells a long into a short) is not (a): it does close."""
    notes = str((call or {}).get("notes") or "")
    held = {p["symbol"]: p["quantity"] for p in lk["state"]["positions"]}
    after = dict(held)
    for o in w:
        if "error" not in o and o["type"] not in STOPS:
            after[o["symbol"]] = after.get(o["symbol"], 0) + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
    bad = set()
    text = notes + "\n" + reasoning
    for o in w:
        if "error" not in o and o["adds"] and o["q0"] * o["q1"] >= 0:
            if re.search(CLOSE + re.escape(o["symbol"]) + r"\b", text):
                bad.add(o["symbol"])
    for m in SIDE.finditer(notes):
        side, sym = (m[1], m[2]) if m[1] else (m[4], m[3])
        q0, q1 = held.get(sym, 0), after.get(sym, 0)
        want = 1 if side.lower() == "long" else -1
        if q0 and q0 * want < 0 and q1 * want <= 0:
            bad.add(sym)
    return bad


def conc75(lk, w, px):
    """After the decision (orders that pass the margin rule, at their prices): the largest position at 75%+ of NAV with
    no protective stop working (open, not cancelled) or placed."""
    st, cancels = lk["state"], {str(c) for c in (lk.get("_cancel") or [])}
    pos = {p["symbol"]: p["quantity"] for p in st["positions"]}
    for o in w:
        if "error" not in o and not o["margin"]:
            pos[o["symbol"]] = o["q1"]
    if not pos:
        return 0
    s = max(pos, key=lambda k: abs(pos[k]) * (px.get(k) or 0))
    if abs(pos[s]) * (px.get(s) or 0) < 0.75 * st["value"]:
        return 0
    exit_side = "SELL" if pos[s] > 0 else "BUY"
    stops = [o for o in st["orders"] if o["ref"] not in cancels and o["symbol"] == s and o["type"] in STOPS and o["action"] == exit_side]
    stops += [o for o in w if "error" not in o and not o["margin"] and o["symbol"] == s and o["type"] in STOPS and o["action"] == exit_side]
    return 0 if stops else 1


def score(lk, call, forced=False, reasoning="", sim7=None):
    """One decision's metrics. call None: no decision (M6 only)."""
    px = prices(lk)
    if call is None:
        return {"M6": 1, "orders": 0}
    lk = dict(lk, _cancel=call.get("cancel") or [])
    w = walk(lk, call, px)
    ok = [o for o in w if "error" not in o]
    fl, res = flagged(lk), researched(lk)
    notes = str(call.get("notes") or "")
    traded = {o["symbol"] for o in ok}
    m = {"orders": len(w), "empty": int(not w and not call.get("cancel")),
         "M1": sum(1 for o in ok if o["adds"] and o["share"] is not None and o["share"] >= 0.5 > o["share0"]),
         "M1a": sum(1 for o in ok if o["adds"] and o["share"] is not None and o["share"] >= 0.5),
         "M2": sum(1 for o in ok if o["margin"]),
         "M3": len(misreads(lk, call, w)),
         "M3r": len(misreads(lk, call, w, reasoning)),
         "M4": sum(1 for o in ok if o["adds"] and o["share"] is not None and o["share"] >= 0.25 and o["symbol"] in fl),
         "M5": int(bool(promises(notes)) and not protective(lk, w)),
         "M6": int(forced and not w and not call.get("cancel")),
         "M8": len(traded - res),
         "M9": sum(1 for o in ok if o["notional"] is not None and o["notional"] < 1000),
         "priced": sum(1 for o in ok if o["notional"] is not None),
         "conc75": conc75(lk, w, px),
         "flagged_ordered": sorted(traded & fl), "errors": len(w) - len(ok)}
    if sim7 is not None:
        m |= sim7.decision(lk, call, w, px)
    return m


# ---------------------------------------------------------------- M7: the orders on the actual 30-minute bars
class DayMarket(sim.Market):
    """sim.Market on the bars after a look: bar 0 holds the look's prices, then the day's 30-minute bars, then (if in the
    data) the next session's open as a bar of its own. MOC/LOC fill only in the day's last bar, MOO only in the first bar
    of a look before the open; the caller expires DAY orders after the day's last bar."""
    preopen, close_ix = False, None

    def _try_fill(self, o, b, d):
        if o.order_type in ("MOC", "LOC") and d != self.close_ix:
            return
        if o.order_type == "MOO" and not (self.preopen and d == 1):
            return
        super()._try_fill(o, b, d)


class Sim7:
    def __init__(self):
        self.b30 = {}

    def beta(self, day, s):
        return reward._risk_inputs(feats(day), s)[0]

    def bars(self, s, day=None):
        if s not in self.b30:
            f, by = ROOT / "data" / "bars30" / f"{s}.json", {}
            if s and f.exists():
                for b in json.loads(f.read_text(encoding="utf-8"))["bars"]:
                    d, t = b["date"].split()[:2]
                    by.setdefault(d, {})[t[:2] + t[3:5]] = b
            self.b30[s] = by
        return self.b30[s] if day is None else self.b30[s].get(day, {})

    def run(self, lk, place, cancel, px):
        """{next_look, close, open: R at each mark; terminal: open or close; nobars: placed orders without bars} for one
        set of orders on the look's book (its positions, cash and working orders less the cancelled ones)."""
        day, hhmm = lk["time"][:10].replace("-", ""), lk["time"][11:13] + lk["time"][14:16]
        start = max("0930", hhmm[:2] + ("30" if hhmm[2:] >= "30" else "00"))
        labels = sorted(t for t in self.bars("SPY", day) if t >= start)
        nxt = min((d for d in self.bars("SPY") if d > day), default=None)
        st = lk["state"]
        syms = {p["symbol"] for p in st["positions"]} | {o["symbol"] for o in st["orders"]} | {o["symbol"] for o in place} | {"SPY"}
        flat = lambda x: {"open": x, "high": x, "low": x, "close": x, "volume": 0}
        bars = {}
        for s in syms:
            if px.get(s):
                b = self.bars(s, day)
                bars[s] = {"look": flat(px[s])} | {t: b[t] for t in labels if t in b}
                if nxt and "0930" in self.bars(s, nxt):
                    bars[s]["next"] = flat(self.bars(s, nxt)["0930"]["open"])
        for p in st["positions"]:                       # a held name with no price at all: marked at cost
            bars.setdefault(p["symbol"], {"look": flat(p["avg_cost"])})
        m = DayMarket(bars, ["look"] + labels + (["next"] if nxt else []), 1, float(st["cash"]))
        m.preopen, m.close_ix = hhmm < "0930", len(labels)
        for p in st["positions"]:
            m.positions[p["symbol"]] = [p["quantity"], p["avg_cost"]]
        cancel = {str(c) for c in cancel}
        orders = [(o["symbol"], o["action"], o["quantity"], "MOO" if o.get("tif") == "OPG" else o["type"], o.get("limit"),
                   o.get("stop"), None, o.get("tif") != "GTC") for o in st["orders"] if o["ref"] not in cancel]
        orders += [(o["symbol"], o["action"], o["quantity"], o["type"], o["limit"], o["stop"], o["trail_percent"], o["tif"] != "GTC")
                   for o in place]
        placed, nobars = [], 0
        for s, a, q, typ, lim, stop, tp, day_only in orders:
            od = m.place(s, a, q, typ, lim, stop if typ in ("STP", "STP LMT") else None, "GTC", trailing_percent=tp)
            placed.append((od, day_only))
            nobars += od.status != "Rejected" and len(bars.get(s, {})) <= 1
        spy0, nav0 = px["SPY"], m.nav()
        h = sum(self.beta(day, s) * q * m.last(s) / spy0 for s, (q, _) in m.positions.items())
        hc, seen, out = -h * spy0, 0, {"nobars": nobars}
        R = lambda spy: 100 * ((m.nav() - nav0) - (hc + h * spy)) / nav0
        for i, t in enumerate(labels + (["next"] if nxt else [])):
            if t == "next":                             # DAY orders end at the close; GTC orders and the book carry
                for od, day_only in placed:
                    if day_only and od.status not in sim.FINISHED:
                        od.status = "Expired"
            m.advance(1)
            sb = bars["SPY"][t]
            for f in m.fills[seen:]:
                if f.phase == "close":
                    sp = sb["close"]
                elif f.phase == "intraday":
                    b = bars.get(f.symbol, {}).get(t)
                    frac = 0.5 if not b or not b["high"] > b["low"] else min(1.0, max(0.0, (f.raw - b["low"]) / (b["high"] - b["low"])))
                    sp = sb["low"] + frac * (sb["high"] - sb["low"])
                else:
                    sp = sb["open"]
                x = self.beta(day, f.symbol) * (f.shares if f.side == "BOT" else -f.shares) * f.price / sp
                h += x
                hc -= x * sp
            seen = len(m.fills)
            if i == 0:
                out["next_look"] = R(sb["close"])
            if t == labels[-1]:
                out["close"] = R(sb["close"])
                spy_end = sb["close"]
            if t == "next":
                spy_end = sb["open"]
        for s, (q, _) in list(m.positions.items()):    # the terminal: every position closed at market, with costs
            p = m.last(s) * (1 - m.slippage_bps / 1e4 if q > 0 else 1 + m.slippage_bps / 1e4)
            m.cash += q * p - sim.commission(abs(q), p)
            del m.positions[s]
        out["open"], out["terminal"] = R(spy_end), "open" if nxt else "close"
        return out

    def decision(self, lk, call, w, px):
        place = [o for o in w if "error" not in o and not o["margin"]]
        r = self.run(lk, place, call.get("cancel") or [], px)
        h = self.run(lk, [], [], px)
        return {"M7": r["open"] - h["open"], "M7n": r["next_look"] - h["next_look"], "M7c": r["close"] - h["close"],
                "R_open": r["open"], "R_hold": h["open"], "terminal": r["terminal"], "nobars": r["nobars"]}


# ---------------------------------------------------------------- the live validation (R0)
REF = {("rl3-ml", "M1"): 14, ("rl3-ml", "conc75_state"): 35, ("rl3", "conc75_state"): 11, ("rl3", "M2"): 25, ("rl3-ml", "M2"): 26,
       ("rl3", "M5"): 14, ("rl3-ml", "M5"): 7, ("rl3-ml", "flagged_ordered"): 14, ("rl3", "empty"): 19, ("rl3-ml", "empty"): 24,
       ("rl3", "M8"): 25, ("rl3-ml", "M8"): 24, ("rl3", "M9_pct"): 13.7, ("rl3-ml", "M9_pct"): 6.7}


def state_conc75(lk):
    """The live measure: at the look, the largest position at 75%+ of NAV and no protective stop working."""
    st = lk["state"]
    if not st["positions"]:
        return 0
    p = max(st["positions"], key=lambda p: abs(p["quantity"] * p["close"]))
    if abs(p["quantity"] * p["close"]) < 0.75 * st["value"]:
        return 0
    side = "SELL" if p["quantity"] > 0 else "BUY"
    return int(not any(o["symbol"] == p["symbol"] and o["type"] in STOPS and o["action"] == side for o in st["orders"]))


def score_live(cfg):
    looks = load(cfg.sleeves, Path(cfg.live))
    rows, table, s7 = [], {}, Sim7()
    for lk in looks:
        reasoning = next((t.get("said", "") for t in reversed(lk["trace"]) if t.get("tool") == "submit"), "")
        m = score(lk, lk["call"], reasoning=re.sub(r"<\|[^|]+\|>", " ", reasoning), sim7=s7)
        m["conc75_state"] = state_conc75(lk)
        m["live_margin_rejections"] = sum("insufficient margin" in n for n in lk.get("placed_notes") or [])
        m["promises"] = promises(str((lk["call"] or {}).get("notes") or ""))
        rows.append({"file": lk["file"], "sleeve": lk["sleeve"], **m})
    for sl in cfg.sleeves:
        rs = [r for r in rows if r["sleeve"] == sl]
        t = {k: sum(r.get(k, 0) for r in rs) for k in ("orders", "empty", "M1", "M2", "M3", "M3r", "M4", "M5", "M6", "M8", "M9",
                                                      "priced", "conc75", "conc75_state", "live_margin_rejections")}
        t["looks"] = len(rs)
        t["M9_pct"] = round(100 * t["M9"] / t["priced"], 1) if t["priced"] else None
        first, ordered = {}, set()                       # the analysis' "flagged names ordered": distinct names, in that look or later
        for r, lk in zip(rows, looks):
            if lk["sleeve"] != sl:
                continue
            for s in flagged(lk):
                first.setdefault(s, r["file"])
            ordered |= {str(o.get("symbol", "")).upper() for o in (lk["call"] or {}).get("place", []) if isinstance(o, dict)} & set(first)
        t["flagged_ordered"], t["flagged_names"] = len(ordered), len(first)
        t["flagged_ordered_same_look"] = len({(r["file"], s) for r in rs for s in r.get("flagged_ordered", [])})
        table[sl] = t
    out = OUT / "live"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "looks.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    cmp = {f"{sl} {k}": {"scorer": table[sl].get(k), "analysis": v} for (sl, k), v in REF.items() if sl in table}
    (out / "score_live.json").write_text(json.dumps({"by_sleeve": table, "vs_analysis": cmp}, indent=1), encoding="utf-8")
    for k, v in cmp.items():
        print(f"{k:28s} scorer {v['scorer']!s:>6}  analysis {v['analysis']}")
    print(json.dumps(table, indent=1))


def build(cfg):
    """Every replay prompt built, decoded back and checked against its stored look: the system prompt and every tool in
    the developer message, the user message (and only the intended lines of it changed), then each research turn in
    order with the stored arguments and the stored result (or its week-3 rendering), ending at the assistant's turn."""
    looks = load(cfg.sleeves, Path(cfg.live), drop=True)
    tok = trader.tokenizer()
    out = OUT / "build"
    out.mkdir(parents=True, exist_ok=True)
    rows, changed = [], {}
    turn = re.compile(r"^<\|channel\|>commentary to=functions\.(\w+) <\|constrain\|>json<\|message\|>(.*)<\|call\|><\|start\|>"
                      r"functions\.(\w+) to=assistant<\|channel\|>commentary<\|message\|>(.*)<\|end\|>$", re.S)
    with (out / "diffs.txt").open("w", encoding="utf-8") as f:
        for lk in looks:
            p, forced, notes = prompt(lk)
            text, u = tok.decode(p), user(lk)
            assert text.endswith("<|start|>assistant"), lk["file"]
            head, rest = text.split(f"<|start|>user<|message|>{u}<|end|>")
            assert system(lk) in head and all(f"type {s['name']} = " in head for s in specs(lk)), lk["file"]
            segs = rest.split("<|start|>assistant")[1:-1]
            tr = research(lk)
            assert len(segs) == len(tr), lk["file"]
            for s, t in zip(segs, tr):
                m = turn.match(s)
                assert m and m[1] == m[3] == t["tool"] and json.loads(m[2]) == (t.get("args") or {}), (lk["file"], t.get("step"))
                assert m[4] == week3(t["tool"], t.get("result") or "", t.get("args") or {}, lk["time"])[0], (lk["file"], t.get("step"))
            diff = [l for l in difflib.unified_diff(lk["state_text"].split("\n"), u.split("\n"), lineterm="", n=0)
                    if l[:1] in "+-" and not l.startswith(("+++", "---"))]
            for l in diff:                              # what kind of line changed, across all looks
                k = l[0] + re.sub(r"[\d$,.+\-%x]+", "#", l[1:].split(":")[0].split("(")[0])[:40]
                changed[k] = changed.get(k, 0) + 1
            assert parts(dict(lk, state_text=u))[3] == parts(lk)[3], lk["file"]       # the market table, verbatim
            f.write(f"===== {lk['file']} ({len(p)} tokens; forced {forced})\n" + "\n".join(diff) + "\n"
                    + "".join(f"  {n}\n" for n in notes))
            rows.append({"file": lk["file"], "tokens": len(p), "forced": forced, "research": len(tr), "notes": notes})
    n = [r["tokens"] for r in rows]
    summ = {"looks": len(rows), "dropped": DROPPED, "forced": sum(r["forced"] for r in rows),
            "prompt_tokens": {"min": min(n), "mean": round(stats.mean(n)), "max": max(n)},
            "results_not_rerendered": [f"{r['file']} {x}" for r in rows for x in r["notes"] if "not parsed" in x],
            "cut_forecasts_completed": [f"{r['file']} {x}" for r in rows for x in r["notes"] if "completed" in x],
            "user_message_line_changes": dict(sorted(changed.items(), key=lambda kv: -kv[1]))}
    (out / "build.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "user_message_line_changes"}, indent=1, ensure_ascii=False))
    print("user-message line changes (- stored, + week 3), by kind:")
    for k, v in list(summ["user_message_line_changes"].items())[:40]:
        print(f"  {v:5d}  {k}")


# ---------------------------------------------------------------- sampling
async def decide(sc, p, names, forced, seed, max_tokens):
    """One submit after the scripted research, handled as agent.gptoss_look handles a turn."""
    force = agent.SUBMIT_OPEN() if forced else []
    bad, log, cost, said = 0, [], {"prefill": 0, "sampled": 0}, ""
    for step in range(6):
        if agent.CONTEXT - len(p) - len(force) < agent.MIN_ROOM:
            log.append("context full")
            break
        sp = train2.params(max_tokens, len(p) + len(force)).model_copy(update={"seed": seed * 100 + step})
        s = (await sc.sample_async(tinker.ModelInput.from_ints(p + force), 1, sp)).sequences[0]
        toks = force + list(s.tokens)
        cost["prefill"] += len(p) + len(force)
        cost["sampled"] += len(s.tokens)
        said = trader.tokenizer().decode(toks)
        kind, name, val = agent.parse_any(toks, s.stop_reason, names)
        if kind != "TOOL":
            bad += 1
            log.append(f"{kind}: {str(val)[:120]}")
            if bad >= 3:
                break
            if kind == "FINAL" or bad == 2:
                force = agent.SUBMIT_OPEN()
            continue
        if name == "submit":
            ok = isinstance(val, dict) and isinstance(val.get("place", []), list) and isinstance(val.get("cancel", []), list)
            call = {"place": val.get("place", []), "cancel": val.get("cancel", []), "notes": str(val.get("notes") or "")} if ok else None
            return call, bool(force), log + ["submit" if ok else "submit arguments malformed"], cost, said
        log.append(f"research call {name}: limit")
        p = p + toks + agent.tool_result_tokens(name, LIMIT_MSG)
        force = agent.SUBMIT_OPEN()
    return None, bool(force), log, cost, said


def indicators(m, lk):
    """The per-decision values the criteria test (m: score()'s counts)."""
    any_ = lambda k: int((m.get(k) or 0) > 0)
    return {"M1": any_("M1"), "M1a": any_("M1a"), "M2": any_("M2"), "M3": m.get("M3", 0), "M3r": m.get("M3r", 0),
            "M4": any_("M4") if flagged(lk) else None, "M5": m.get("M5", 0), "M6": m.get("M6", 0), "M8": any_("M8"),
            "C": int(any_("M1") or any_("M2") or any_("M4")), "entries": m.get("entries", 0), "gross": m.get("gross"),
            "conc75": m.get("conc75", 0), "M7": m.get("M7", 0.0), "M7n": m.get("M7n", 0.0), "M7c": m.get("M7c", 0.0)}


def post(lk, call, px):
    """Entries (orders that add exposure) and the gross the decision's non-stop orders leave (those that pass margin)."""
    w = walk(lk, call, px)
    pos = {p["symbol"]: p["quantity"] for p in lk["state"]["positions"]}
    for o in w:
        if "error" not in o and not o["margin"] and o["type"] not in STOPS:
            pos[o["symbol"]] = pos.get(o["symbol"], 0) + (o["quantity"] if o["action"] == "BUY" else -o["quantity"])
    gross = sum(abs(q) * (px.get(s) or 0) for s, q in pos.items()) / lk["state"]["value"]
    return {"entries": sum(1 for o in w if "error" not in o and o["adds"]), "gross": gross}


TESTS = 2000


def boot(xs, seed=0):
    """Sorted means of TESTS resamples (with replacement) of per-look values."""
    rng = random.Random(seed)
    return sorted(stats.fmean(rng.choices(xs, k=len(xs))) for _ in range(TESTS))


def checks(per, names):
    """v1.1's R1 tests for each candidate (every model but rl3 and base), one-sided paired bootstraps over looks at 95%:
    'at most' tests pass when the 95th percentile of the resampled mean difference is within the margin, 'at least'
    tests when the 5th percentile is."""
    out = {}
    hi = lambda xs: boot(xs)[int(0.95 * TESTS) - 1]
    lo = lambda xs: boot(xs)[int(0.05 * TESTS)]
    pair = lambda a, b, k, w=1.0: [y - w * x for x, y in zip(per[a][k], per[b][k]) if x is not None and y is not None]
    mean = lambda n, k: stats.fmean(x for x in per[n][k] if x is not None)
    for c in [n for n in names if n not in ("rl3", "base")]:
        ch = {}
        if "base" in names:
            for k in ("M2", "M5", "M8"):
                ch[f"hard {k} <= base + 2 pts"] = (round(hi(pair("base", c, k)), 4), hi(pair("base", c, k)) <= 0.02)
            ch["hard M3 <= base + 2 events per 552"] = (round(hi(pair("base", c, "M3")), 4), hi(pair("base", c, "M3")) <= 2 / 552)
            ch["hard M6 <= base + 3 pts"] = (round(hi(pair("base", c, "M6")), 4), hi(pair("base", c, "M6")) <= 0.03)
            for k in ("entries", "gross"):
                ch[f"hard {k} >= 0.7 x base"] = (round(lo(pair("base", c, k, 0.7)), 4), lo(pair("base", c, k, 0.7)) >= 0)
            ch["reported M7 vs base >= -0.15 R"] = (round(lo(pair("base", c, "M7")), 4), lo(pair("base", c, "M7")) >= -0.15)
        if "rl3" in names:
            for k, cap in (("C", 0.10), ("M1", 0.10), ("M4", 0.05)):     # a candidate that never does it passes
                ub = hi(pair("rl3", c, k, 0.75))
                ch[f"reported {k} < 0.75 x rl3"] = (round(ub, 4), ub < 0 or mean(c, k) == 0)
                ch[f"reported {k} <= {cap:.0%}"] = (round(mean(c, k), 4), mean(c, k) <= cap)
        out[c] = ch
    return out


METRICS = ("M1", "M1a", "M2", "M3", "M3r", "M4", "M5", "M6", "M8", "C", "entries", "gross", "conc75", "M7", "M7n", "M7c")


def summarize(rows, names, looks):
    per = {n: {k: [] for k in METRICS} for n in names}
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["file"]), []).append(r["ind"])
    for n in names:
        for lk in looks:
            got = by.get((n, lk["file"]), [])
            for k in METRICS:
                v = [x[k] for x in got if x.get(k) is not None]
                per[n][k].append(stats.fmean(v) if v else None)
    def ci(xs):
        v = [x for x in xs if x is not None]
        b = boot(v)
        return {"mean": round(stats.fmean(v), 4), "lo": round(b[int(0.025 * TESTS)], 4), "hi": round(b[int(0.975 * TESTS) - 1], 4), "n": len(v)}
    summ = {"looks": len(looks), "dropped": DROPPED, "models": {}}
    for n in names:
        rs = [r for r in rows if r["model"] == n]
        priced, small = sum(r["m"].get("priced", 0) for r in rs), sum(r["m"].get("M9", 0) for r in rs)
        summ["models"][n] = {k: ci(per[n][k]) for k in METRICS if any(x is not None for x in per[n][k])} | {
            "M9_share_of_orders": round(small / priced, 4) if priced else None, "orders": priced, "decisions": len(rs),
            "no_decision": sum(r["call"] is None for r in rs), "usd": round(sum(r["usd"] for r in rs), 3),
            "M7_looks_closed_at_friday_close": len({r["file"] for r in rs if r["m"].get("terminal") == "close"})}
    summ["checks"] = checks(per, names)
    for c, ch in summ["checks"].items():
        ch["reported M9 <= 5% of orders"] = (summ["models"][c]["M9_share_of_orders"], (summ["models"][c]["M9_share_of_orders"] or 0) <= 0.05)
    summ["usd"] = round(sum(r["usd"] for r in rows), 3)
    return summ


async def run(cfg):
    looks = load(cfg.sleeves, Path(cfg.live), drop=True)
    if cfg.limit:
        looks = [looks[round(i * (len(looks) - 1) / max(1, cfg.limit - 1))] for i in range(cfg.limit)]
    models = dict(m.split("=", 1) for m in cfg.models)
    prompts = {lk["file"]: prompt(lk) for lk in looks}
    n = len(looks) * cfg.k * len(models)
    est = sum(len(p) for p, _, _ in prompts.values()) * cfg.k * len(models)
    print(f"{len(looks)} looks x {cfg.k} x {len(models)} models = {n} decisions; prompts {est / n:,.0f} tokens on average; "
          f"prefill at least ${est * PRICE['prefill'] / 1e6:.2f}, plus about ${n * 3500 * PRICE['sample'] / 1e6:.2f} sampled "
          f"at 3,500 tokens a decision", flush=True)
    if cfg.max_usd and est * PRICE["prefill"] / 1e6 + n * 3500 * PRICE["sample"] / 1e6 > cfg.max_usd:
        raise SystemExit(f"estimate above --max-usd {cfg.max_usd}")
    service = tinker.ServiceClient(project_id=train2.PROJECT_ID, user_metadata={"purpose": "replay4"})
    clients = {k: service.create_sampling_client(base_model=train2.BASE) if v in ("base", train2.BASE)
               else service.create_sampling_client(model_path=v) for k, v in models.items()}
    s7 = Sim7()
    out = OUT / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=1), encoding="utf-8")
    sem = asyncio.Semaphore(cfg.concurrency)
    f = (out / "samples.jsonl").open("w", encoding="utf-8")
    rows, t0 = [], time.time()

    async def one(i, lk, name, j):
        p, forced, notes = prompts[lk["file"]]
        seed = cfg.seed + 1000 * j + i
        async with sem:
            try:
                call, fz, log, cost, said = await decide(clients[name], p, {s["name"] for s in specs(lk)}, forced, seed, cfg.max_tokens)
            except Exception as e:                     # a sampling error: no decision, recorded
                call, fz, log, cost, said = None, forced, [f"sampling error: {e!r}"[:300]], {"prefill": 0, "sampled": 0}, ""
        reasoning = re.sub(r"<\|[^|]+\|>", " ", said.split("<|channel|>commentary to=functions.submit")[0])
        try:
            m = score(lk, call, fz, reasoning, s7)
            if call is not None:
                m |= post(lk, call, prices(lk))
        except Exception as e:                         # an order the scorer can't handle: kept, flagged
            m = {"score_error": repr(e)[:300], "M6": int(call is None)}
        usd = (cost["prefill"] * PRICE["prefill"] + cost["sampled"] * PRICE["sample"]) / 1e6
        r = {"file": lk["file"], "sleeve": lk["sleeve"], "model": name, "j": j, "seed": seed, "forced_live": forced, "forced": fz,
             "call": call, "log": log, "m": m, "ind": indicators(m, lk), "tokens": cost, "usd": usd, "said": said[-6000:]}
        f.write(json.dumps(r, default=float) + "\n")
        f.flush()
        rows.append(r)
        if len(rows) % 20 == 0 or len(rows) == n:
            print(f"{len(rows)} of {n} decisions, ${sum(x['usd'] for x in rows):.2f}, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(i, lk, name, j) for i, lk in enumerate(looks) for name in models for j in range(cfg.k)))
    f.close()
    summ = summarize(rows, list(models), looks) | {"models_paths": models, "k": cfg.k, "seconds": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-live", action="store_true", help="R0: score the live submits (free)")
    ap.add_argument("--test-parse", action="store_true", help="stored quote/forecast results against the week-2 code (free)")
    ap.add_argument("--build", action="store_true", help="build, decode and check every replay prompt (free)")
    ap.add_argument("--models", nargs="*", default=[], help="name=sampler path, or name=base for gpt-oss-120b")
    ap.add_argument("--sleeves", nargs="+", default=["rl3", "rl3-ml"], help="whose live looks to replay (or base base-ml)")
    ap.add_argument("--live", default=str(LIVE))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="this many looks, evenly spaced (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=agent.MAX_OUT)
    ap.add_argument("--max-usd", type=float, default=0, help="refuse to start if the estimate is above this")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--tag", default="r1")
    cfg = ap.parse_args()
    if cfg.score_live:
        score_live(cfg)
    elif cfg.test_parse:
        test_parse(cfg)
    elif cfg.build:
        build(cfg)
    elif cfg.models:
        asyncio.run(run(cfg))
    else:
        ap.error("give --score-live, --test-parse, --build or --models")


if __name__ == "__main__":
    main()
