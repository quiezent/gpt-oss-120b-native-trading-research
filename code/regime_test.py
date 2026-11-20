"""The market-facts line ("regime_line" in sleeves.json; candidate 2, Week 6). Offline: no gateway, no Tinker, no ledger.
- The Researcher's four examples, byte for byte, and regime.py's copy of its two functions against its file.
- regime.facts picks the form from what the look can see, takes no quote before the open, asks for the day's facts
  once, never shows a stale VIX or a QQQ figure that is not a print, falls back to the closes-only form when anything
  after the closes is missing or fails, and stops asking for a request that failed twice in a day.
- live.look() on fakes at 09:00, 09:31 and 13:00: the line sits directly after the Rates line for a flagged sleeve
  (the same text as the Researcher's insert_after_rates gives), and a sleeve without the flag is unchanged; a failing
  gateway costs the line or its detail, never a look.

    python regime_test.py
"""
import contextlib
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
import agent  # noqa: E402
import live  # noqa: E402
import paper  # noqa: E402
import regime  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

ET = ZoneInfo("America/New_York")
ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


# 1. the text
EX = [((None, 18.26, 679.35, None, None, "Mon"), "Market: VIX closed Mon at 18.26. QQQ closed Mon at 679.35."),
      ((19.10, 18.26, 679.35, None, 673.00, "Mon"),
       "Market: VIX 19.10, +0.84 from Mon's close of 18.26. QQQ 673.00 before the open, -0.93% from Mon's close of 679.35."),
      ((19.46, 18.26, 679.35, 672.10, 667.97, "Mon"),
       "Market: VIX 19.46, +1.20 from Mon's close of 18.26. QQQ opened at 672.10, -1.07% from Mon's close of 679.35; "
       "now 667.97, -0.61% since the open."),
      ((25.82, 16.61, 691.95, 683.09, 652.18, "Wed"),
       "Market: VIX 25.82, +9.21 from Wed's close of 16.61. QQQ opened at 683.09, -1.28% from Wed's close of 691.95; "
       "now 652.18, -4.53% since the open.")]
CLOSES, SESSION = EX[0][1], EX[2][1]
PREOPEN = EX[1][1].replace("VIX 19.10, +0.84", "VIX 19.46, +1.20")      # the fake VIX snapshot reads 19.46
for args, want in EX:
    check(f"example byte for byte: {want[:46]}...", regime.regime_line(*args) == want, regime.regime_line(*args))
check("the line is plain ASCII", all(ord(c) < 128 for _, w in EX for c in w))
THEIRS = Path(r"C:\Claude\Trader-models\runs\ml\research\week6_prep\c2\regime\regime_line_final.py")
cut = lambda s, a, b: s[s.index(a):s.index(b)].strip()
mine = cut(Path(regime.__file__).read_text(encoding="utf-8"), "def _pct(a, b):", "# ---- end of the copy ----")
if THEIRS.exists():
    check("regime.py's copy is the Researcher's two functions, character for character",
          mine == cut(THEIRS.read_text(encoding="utf-8"), "def _pct(a, b):", 'if __name__ == "__main__":'))
else:
    print("skip the Researcher's file is not on this machine")
state = "Date: h\nSleeve now $30,000 (+0.0% since its $30,000 start).\nRates: r.\nYour notes from your last look: none yet.\nSleeve value"
with_line = regime.insert_after_rates(state, "Market: m.")
check("insert_after_rates puts it right after the Rates line", with_line.split("\n")[2:4] == ["Rates: r.", "Market: m."])
check("shown() finds the line of a message, and None when there is none", regime.shown(with_line) == "Market: m." and regime.shown(state) is None)
check("shown() does not take a Market line from the model's own notes",
      regime.shown(state.replace("none yet.", "Plan.\nMarket: my own view.")) is None)

# 2. regime.facts on a fake gateway
at = lambda h, m, s=0: datetime(2026, 11, 10, h, m, s, tzinfo=ET)       # a Tuesday; the prior session is Mon 9 Nov
VIX_ROW = {"symbol": "VIX", "last": 19.46, "close": 18.26, "last_time": "2026-11-10T17:44:00Z"}


class GW:
    """VIX daily bars, QQQ five-minute bars and the VIX snapshot, counting each request; any of them can raise."""
    def __init__(self, daily=None, five=None, vix=None, fail=()):
        self.daily = [{"date": "20261106", "close": 25.64}, {"date": "20261109", "close": 18.26}] if daily is None else daily
        self.five = ([{"date": "20261110 09:30:00 US/Eastern", "open": 672.10}, {"date": "20261110 09:35:00 US/Eastern", "open": 671.0}]
                     if five is None else five)
        self.vix = dict(VIX_ROW) if vix is None else vix
        self.fail, self.n, self.timeouts = set(fail), {"daily": 0, "five": 0, "vix": 0, "other": 0}, []

    def bars(self, symbol, duration="", bar_size="", **k):
        self.timeouts.append(k.get("timeout"))
        if symbol == regime.VIX and bar_size == "1 day":
            self.n["daily"] += 1
            if "daily" in self.fail:
                raise RuntimeError("No reply from IB Gateway to the historical data request")
            return self.daily
        if symbol == "QQQ" and bar_size == "5 mins" and k.get("rth") is True:
            self.n["five"] += 1
            if "five" in self.fail:
                raise RuntimeError("No reply from IB Gateway to the historical data request")
            return self.five
        self.n["other"] += 1
        return []

    def quotes(self, symbols, **k):
        if symbols == [regime.VIX]:
            self.timeouts.append(k.get("timeout"))
            self.n["vix"] += 1
            if "vix" in self.fail:
                raise RuntimeError("Disconnected from IB Gateway")
            return [dict(self.vix)]
        self.n["other"] += 1
        return []


def fresh(gw=None):
    regime._day.clear()
    return gw or GW()


PRE = {"symbol": "QQQ", "last": 673.0, "last_time": "2026-11-10T14:15:40Z"}       # 09:15:40 ET: a pre-market print
RTH = {"symbol": "QQQ", "last": 667.97, "last_time": "2026-11-10T17:45:10Z"}      # 12:45:10 ET
logs = []
facts = lambda g, now, session, quote=None: regime.facts(g, now, session, 679.35, "2026-11-09", quote, logs.append)
g = fresh()
check("09:00, before the open: closes only, and no quote is taken",
      facts(g, at(9, 0, 10), False) == CLOSES and g.n == {"daily": 1, "five": 0, "vix": 0, "other": 0}, g.n)
g = fresh()
got = facts(g, at(9, 31), True, PRE)
check("09:31, the delayed last is pre-market: the before-the-open form, and no bars asked for the open", got == PREOPEN and g.n["five"] == 0, got)
g = fresh()
got = facts(g, at(13, 0, 10), True, RTH)
check("13:00: the session form, the open from the 09:30 bar", got == SESSION, got)
facts(g, at(13, 30), True, RTH)
check("the day's prior close and open are asked for once", g.n["daily"] == 1 and g.n["five"] == 1 and g.n["vix"] == 2, g.n)
check("every request has a short timeout", g.timeouts and all(t is not None and t <= 10 for t in g.timeouts), g.timeouts)
g2 = GW(daily=[{"date": "20261109", "close": 18.26}, {"date": "20261110", "close": 20.0}])
regime.facts(g2, datetime(2026, 11, 11, 9, 0, tzinfo=ET), False, 675.0, "2026-11-10")
check("a new day asks again and takes that day's prior close", g2.n["daily"] == 1 and regime._day["vix_prev"] == ("20261110", 20.0))
g = fresh(GW(daily=[{"date": "20261110", "close": 99.0}, {"date": "20261109", "close": 18.26}, {"date": "20261106", "close": 25.64}]))
check("a daily bar of today is not the prior close, whatever the bars' order", facts(g, at(13, 0), True, RTH) == SESSION)
g = fresh(GW(daily=[{"date": "20261105", "close": 28.18}, {"date": "20261106", "close": 25.64}]))
try:
    facts(g, at(9, 0), False)
    check("VIX's prior close from another session than the table's QQQ close: raises", False)
except ValueError as e:
    check("VIX's prior close from another session than the table's QQQ close: raises", "20261106" in str(e), str(e))
g.daily = [{"date": "20261106", "close": 25.64}, {"date": "20261109", "close": 18.26}]
check("another session's close is not kept: once the right bar is served, the next look has the line",
      facts(g, at(9, 31), True, PRE) == PREOPEN and g.n["daily"] == 2, g.n)
g = fresh(GW(daily=[]))
try:
    facts(g, at(9, 0), False)
    check("no VIX daily bar: raises", False)
except ValueError:
    check("no VIX daily bar: raises", True)
g = fresh(GW(fail=["daily"]))
errs = []
for _ in range(4):
    try:
        facts(g, at(9, 0), False)
    except Exception as e:                                            # noqa: BLE001
        errs.append(type(e).__name__)
check("a request that failed twice today is not asked a third time", g.n["daily"] == 2 and len(errs) == 4, (g.n, errs))
g = fresh(GW(vix={"symbol": "VIX", "error": "no data"}))
check("no VIX last in the session: closes only, and it is logged", facts(g, at(13, 0), True, RTH) == CLOSES and "closes only" in logs[-1], logs[-1:])
g = fresh(GW(fail=["vix"]))
check("the VIX snapshot raising: closes only, and it is logged", facts(g, at(13, 0), True, RTH) == CLOSES and "VIX snapshot" in logs[-1], logs[-1:])
g = fresh(GW(fail=["five"]))
check("the bars for QQQ's open raising: closes only, and it is logged", facts(g, at(13, 0), True, RTH) == CLOSES and "QQQ's open" in logs[-1], logs[-1:])
facts(g, at(13, 30), True, RTH)
facts(g, at(14, 0), True, RTH)
check("and the open is not asked a third time that day", g.n["five"] == 2, g.n)
g = fresh(GW(vix={"symbol": "VIX", "last": 19.30, "close": 18.26, "last_time": "2026-11-09T21:15:01Z"}))
check("a VIX last left from an earlier session is not shown as today's: closes only",
      facts(g, at(9, 31), True, PRE) == CLOSES and "not of today" in logs[-1], logs[-1:])
g = fresh(GW(vix={"symbol": "VIX", "last": 19.30, "close": 18.26}))
check("a VIX last with no time can't be told from a stale one: closes only", facts(g, at(13, 0), True, RTH) == CLOSES)
g = fresh()
check("a QQQ quote with no last (the table would show its mid or close): closes only",
      facts(g, at(13, 0), True, {"symbol": "QQQ", "close": 688.17, "bid": 667.9, "ask": 668.0}) == CLOSES)
check("no QQQ quote at all: closes only", facts(fresh(), at(13, 0), True, None) == CLOSES)
g = fresh(GW(five=[{"date": "20261109 09:30:00 US/Eastern", "open": 662.63}]))
check("the session is on but the 09:30 bar of today is not in the bars: closes only, never a false 'before the open'",
      facts(g, at(13, 0), True, RTH) == CLOSES and "09:30 bar" in logs[-1])
g.five = [{"date": "20261110 09:30:00 US/Eastern", "open": 672.10}]
check("a bar that was only late is picked up at the next look", facts(g, at(13, 30), True, RTH) == SESSION and g.n["five"] == 2, g.n)
g = fresh()
check("a quote with no time: the clock decides, pre-open until 09:46",
      "before the open" in facts(g, at(9, 45, 30), True, {"symbol": "QQQ", "last": 673.0}))
g = fresh()
check("a quote with no time: the session form from 09:46", "opened at 672.10" in facts(g, at(9, 46, 5), True, {"symbol": "QQQ", "last": 673.0}))
g = fresh()
late = {"symbol": "QQQ", "last": 672.4, "last_time": "2026-11-10T14:36:00Z"}      # a late 09:30 look: the quote is already of 09:36 ET
check("a late look whose delayed quote is already in the session gets the session form", "opened at 672.10" in facts(g, at(9, 51), True, late))
g = fresh(GW(vix=dict(VIX_ROW, close=25.64)))
logs.clear()
facts(g, at(13, 0), True, RTH)
check("a snapshot close that is not the prior session's is logged, and the daily bar is used", logs and "daily bar is used" in logs[0], logs[:1])

# 3. live.look() on fakes: where the line lands, and only for the flagged sleeve
texts, lines, NOW, RATE = {}, [], [None], ["Rates: r."]


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW[0]


class LookGW(GW):
    def quotes(self, symbols, **k):
        if symbols == [regime.VIX]:
            return super().quotes(symbols, **k)
        return [dict(QQQ_ROW[0])] if "QQQ" in symbols else []

    def __getattr__(self, name):
        return lambda *a, **k: []


QQQ_ROW, LOOK_GW = [PRE], [None]


@contextlib.contextmanager
def fake_gateway():
    yield LOOK_GW[0]


class FakeData:
    dates = ["20261106", "20261109"]

    def __init__(self, **k):
        pass

    def features(self, d):
        return {"QQQ": {"close": 679.35}}, None

    def table_rows(self, d, extra=None):
        return ["QQQ"], None

    def iso(self, d):
        return "2026-11-09" if d == 1 else "2026-11-10"


class FakeWorld:
    def __init__(self, gw, name, data):
        self.name = name

    def symbols_in_play(self):
        return []


def fake_state_text(world, data, d, header=None, live=None, book=False, cash_clause=False):   # 16 Nov: text B's cash clause
    return header + "\nSleeve value at last close: $30,000 (cash $30,000).\nPositions: none.", {"value": 30000.0, "events": []}, {}


MARKER = Path(tempfile.mkdtemp()) / "LOOKING"     # live.LOOKING, patched (16 Nov 2026): the real one is never touched
marker_seen = []


def fake_finish(now, name, meta, text, st, call, trace, notes, rows, dry_run, cost=None, secs=None):
    texts[name] = text
    marker_seen.append(live.own_marker().exists())  # this process's LOOKING.<pid> beside MARKER (16 Nov 2026)


class FakeService:
    def __init__(self, **k):
        pass

    def create_sampling_client(self, **k):
        return object()


sleeves = {"c2": {"model": "m-c2", "start_cash": 90000, "regime_line": True},
           "plain": {"model": "m-plain", "start_cash": 90000}}
patches = [
    (paper, "sleeves", lambda: sleeves), (paper, "sync_fills", lambda gw: None), (paper, "book", lambda *a, **k: {"positions": {}}),
    (paper, "mark", lambda q: q.get("last")),
    (live, "gateway", fake_gateway), (live, "reconcile", lambda gw: {}), (live, "ml2_guard", lambda: None),
    (live, "open_whatif", lambda: (None, False)),     # 16 Nov 2026: no what-if connection (check_orders_test covers it)
    (live, "LOOKING", MARKER),
    (live, "finish", fake_finish), (live, "log", lambda m: lines.append(m)), (live, "load_memo", lambda n: {}),
    (live, "sessions_left", lambda t: 5), (live.ml_tools, "unknown", lambda s: []), (live.rates, "line", lambda now: RATE[0]),
    (live.tinker, "ServiceClient", FakeService), (tools, "available", lambda gw: []),
    (agent, "gptoss_look", lambda *a, **k: (None, [], 0, 0)), (live, "datetime", FakeDT),
    (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld), (trader, "state_text", fake_state_text),
    (trader, "tokenizer", lambda: None),
]
saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
for m, a, v in patches:
    setattr(m, a, v)


def run(when, quote, gw):
    NOW[0], QQQ_ROW[0], LOOK_GW[0] = when, quote, gw
    regime._day.clear()
    texts.clear()
    lines.clear()
    live.look(dry_run=True)
    return texts.get("c2", "").split("\n"), texts.get("plain", "").split("\n")


try:
    for when, quote, want, label in ((FakeDT(2026, 11, 10, 9, 0, 10, tzinfo=ET), PRE, CLOSES, "09:00"),
                                     (FakeDT(2026, 11, 10, 9, 31, 0, tzinfo=ET), PRE, PREOPEN, "09:31"),
                                     (FakeDT(2026, 11, 10, 13, 0, 10, tzinfo=ET), RTH, SESSION, "13:00")):
        c2, plain = run(when, quote, LookGW())
        i = c2.index("Rates: r.") if "Rates: r." in c2 else -1
        check(f"look at {label}: the flagged sleeve has the line directly after the Rates line", i >= 0 and c2[i + 1] == want,
              c2[i + 1] if i >= 0 else texts.get("c2", "")[:200])
        check(f"look at {label}: without the line the two sleeves' texts are the same, and it is what insert_after_rates gives",
              [l for l in c2 if l != want] == plain and "\n".join(c2) == regime.insert_after_rates("\n".join(plain), want))
        check(f"look at {label}: the sleeve without the flag has no Market line, and shown() reads each text right",
              not any(l.startswith("Market:") for l in plain) and regime.shown("\n".join(c2)) == want and regime.shown("\n".join(plain)) is None)
    one = FakeDT(2026, 11, 10, 13, 0, 10, tzinfo=ET)
    c2, plain = run(one, RTH, LookGW(daily=[]))
    check("a failing line is left out and logged, and both sleeves still get their look",
          set(texts) == {"c2", "plain"} and c2 == plain and any("regime line left out" in m for m in lines), [m[:60] for m in lines if "regime" in m])
    c2, plain = run(one, RTH, LookGW(fail=["five", "vix"]))
    check("a gateway that fails after the closes: the flagged sleeve gets the closes-only line, and both get their look",
          set(texts) == {"c2", "plain"} and CLOSES in c2 and [l for l in c2 if l != CLOSES] == plain)
    c2, plain = run(one, RTH, LookGW(fail=["daily"]))
    check("a gateway that fails on the daily bars: no line, both sleeves get their look", set(texts) == {"c2", "plain"} and c2 == plain)
    RATE[0] = ""
    c2, plain = run(one, RTH, LookGW())
    i = next((k for k, l in enumerate(c2) if l.startswith("Sleeve now ")), -1)
    check("with no Rates line the line follows the sleeve line, and that is logged",
          i >= 0 and c2[i + 1] == SESSION and any("no Rates line" in m for m in lines), c2[i:i + 2])
    RATE[0] = "Rates: r."
    del sleeves["c2"]
    c2, plain = run(one, RTH, LookGW())
    check("with no flagged sleeve no VIX or open request is made", LOOK_GW[0].n["daily"] == 0 and LOOK_GW[0].n["vix"] == 0 and LOOK_GW[0].n["five"] == 0,
          LOOK_GW[0].n)
    sleeves["c2"] = {"model": live.HOLD_MODEL, "start_cash": 90000, "regime_line": True}
    c2, plain = run(one, RTH, LookGW())
    check("a flagged sleeve on hold asks for nothing either", LOOK_GW[0].n["daily"] == 0 and LOOK_GW[0].n["vix"] == 0, LOOK_GW[0].n)
    check("LOOKING: a dry-run look writes the marker (present at each sleeve's record) and removes it",
          marker_seen and all(marker_seen) and not live.own_marker().exists(), (len(marker_seen), live.own_marker().exists()))
finally:
    for m, a, v in saved:
        setattr(m, a, v)
print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
