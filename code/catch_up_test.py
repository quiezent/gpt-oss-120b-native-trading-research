"""live.look's catch-up before each sleeve's state (24 Nov 2026), on fakes: no gateway, no Tinker, no real ledger, log or
look record (paper's ledgers are pointed at a temporary folder; the gateway, the models and the records are fakes).

At the 09:30 look on Mon 23 Nov the look synced fills at 09:30:45 and read IBKR's account at 09:32:08, after every held
name's quote. The first gpt-oss sleeve (base-90k-ml, k 0) and the Claude sleeve (fable-ml) were then built from the books
as at 09:30:45: stops that filled at the open in between were Filled at IBKR (so not in the open-orders list) but in no
ledger (so the position, gross and net still had them). base-90k-ml was shown 14 shorts it had covered and 4.11x gross for
2.94x, and covered them again. The account line told every sleeve "$257,994 of the gross is in positions no sleeve holds".

Here three sleeves, in sleeves.json order: g0 (the first gpt-oss sleeve), c0 (a Claude sleeve) and g1 (a second gpt-oss
sleeve). Each is short 100 A with a BUY STP covering it; g1 also has a BUY 20 B LMT with a SELL 20 B LMT take-profit, and
c0 a BUY 10 D LMT and a BUY 30 E LMT. The three stops and g1's B entry fill while the look fetches its quotes (as the
open's fills land after the look's first sync). More fills fall inside a sleeve's own catch-up (the reviews of 24 Nov):
c0's D entry in full and 12 of its 30 E between its orders read and its fill sync (booked, and the snapshot settled from
them: D not listed, E listed with 18 remaining; never shown as neither), and g1's take-profit during its catch-up quote
of B (shown as working with B still held, as it stood at the sync). Checks, for each
sleeve's state: its position, its open-orders list, its "Since your last decision" lines, its gross and net and its marks
agree (the Trading Model Researcher's three cases); the account line's clause counts only the name no sleeve holds; the
sleeves' total counts the open's fills; and with the account line's read failing, the states are still caught up.

    python catch_up_test.py
"""
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.dont_write_bytecode = True
import agent  # noqa: E402
import live  # noqa: E402
import paper  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


PX = {"A": 49.0, "B": 10.0, "D": 20.0, "E": 5.0, "R": 7.0}             # delayed last prices; R is the name no sleeve holds (a reset's residue)
OPEN_AT = "20261124 09:30:55 US/Eastern"


def ex(eid, ref, symbol, side, shares, price, when):
    return {"exec_id": eid, "time": when, "symbol": symbol, "side": side, "shares": shares, "price": price, "order_id": 1,
            "perm_id": 1, "ref": ref, "exchange": "IBKRATS", "client_id": 10, "liquidation": 0, "commission": 1.0,
            "commission_currency": "USD", "realized_pnl": 0.0}


PRIOR = [ex(f"e-{s}-0", f"tt-{s}-000000000{i}", "A", "SLD", 100, 50.0, "20261123 15:00:00 US/Eastern")
         for i, s in enumerate(("g0", "c0", "g1"))]
STOPS = {s: f"tt-{s}-aaaaaaaa0{i}" for i, s in enumerate(("g0", "c0", "g1"))}
LMT_B, TP_B, LMT_D, LMT_E = "tt-g1-bbbbbbbb04", "tt-g1-cccccccc05", "tt-c0-dddddddd06", "tt-c0-eeeeeeee07"
OPENING = ([ex(f"e-{s}-1", ref, "A", "BOT", 100, 49.0, OPEN_AT) for s, ref in STOPS.items()]
           + [ex("e-g1-2", LMT_B, "B", "BOT", 20, 10.0, OPEN_AT)])
D_FILL = ex("e-c0-3", LMT_D, "D", "BOT", 10, 20.0, "20261124 09:31:31 US/Eastern")     # inside c0's catch-up
E_FILL = ex("e-c0-5", LMT_E, "E", "BOT", 12, 5.0, "20261124 09:31:31 US/Eastern")      # 12 of 30, the same moment
TP_FILL = ex("e-g1-4", TP_B, "B", "SLD", 20, 10.5, "20261124 09:31:33 US/Eastern")     # inside g1's catch-up quote


class FakeGW:
    """Client 10. The open's four fills happen on the look's first quotes() call: before it, executions() has only
    yesterday's sales and orders() lists the orders working; after it, both show the fills. c0's D entry fills just after
    the first orders() call after the open (c0's catch-up reads orders first; nothing else reads them before it in a dry
    run), and g1's take-profit when B is quoted (g1's catch-up: B was bought at the open and is not in the look's quotes)."""
    _lock, timeout = None, None                    # checks.Link.read looks for these (the framework's lock and wait)

    def __init__(self):
        self.opened, self.calls, self.lock = False, [], threading.RLock()
        self.d_filled = self.tp_filled = False
        self.d_at = self.tp_at = None

    def _log(self, what):
        self.calls.append(what)

    def executions(self, days=7):
        self._log("executions")
        return [dict(e) for e in PRIOR + (OPENING if self.opened else []) + ([D_FILL, E_FILL] if self.d_filled else [])
                + ([TP_FILL] if self.tp_filled else [])]

    def orders(self):
        self._log("orders")
        st = "Filled" if self.opened else "Submitted"
        rows = [{"ref": ref, "symbol": "A", "action": "BUY", "quantity": 100, "order_type": "STP", "aux": 51.0, "tif": "GTC",
                 "status": st, "perm_id": 10 + i} for i, ref in enumerate(STOPS.values())]
        rows += [{"ref": LMT_B, "symbol": "B", "action": "BUY", "quantity": 20, "order_type": "LMT", "limit": 10.0,
                  "tif": "DAY", "status": st, "perm_id": 20},
                 {"ref": TP_B, "symbol": "B", "action": "SELL", "quantity": 20, "order_type": "LMT", "limit": 10.5,
                  "tif": "GTC", "status": "Filled" if self.tp_filled else "Submitted", "perm_id": 21},
                 {"ref": LMT_D, "symbol": "D", "action": "BUY", "quantity": 10, "order_type": "LMT", "limit": 20.0,
                  "tif": "DAY", "status": "Filled" if self.d_filled else "Submitted", "perm_id": 22},
                 {"ref": LMT_E, "symbol": "E", "action": "BUY", "quantity": 30, "order_type": "LMT", "limit": 5.0,
                  "tif": "DAY", "status": "Submitted", "remaining": 18 if self.d_filled else 30, "perm_id": 23}]
        if not self.d_filled and self.opened:      # the first orders() read after the open: c0's catch-up
            self.d_filled, self.d_at = True, len(self.calls)
        return rows

    def quotes(self, syms):
        self._log(("quotes", tuple(syms)))
        self.opened = True
        if tuple(syms) == ("B",) and not self.tp_filled:
            self.tp_filled, self.tp_at = True, len(self.calls)
        return [{"symbol": s, "last": PX.get(s, 1.0)} for s in syms]

    def positions(self):
        self._log("positions")
        a = -300 if not self.opened else 0
        rows = [{"symbol": "R", "quantity": 10, "avg_cost": 6.0}]
        if a:
            rows.append({"symbol": "A", "quantity": a, "avg_cost": 50.0})
        if self.opened:
            rows.append({"symbol": "B", "quantity": 20, "avg_cost": 10.0})
        return rows

    def __getattr__(self, name):                   # anything else the look asks for: nothing
        def call(*a, **k):
            self._log(name)
            return []
        return call


class FakeData:
    dates = ["20261123"]
    ix = {"20261123": 0}

    def __init__(self, **k):
        pass

    def table_rows(self, d, extra=None):
        return ["A"], None

    def features(self, d):
        return {"A": {"close": 50.0}}, None

    def iso(self, d):
        return "2026-11-23"


WORLDS = []


class FakeWorld:
    def __init__(self, gw, name, data):
        self.gw, self.name = gw, name
        WORLDS.append(self)

    def symbols_in_play(self):
        return []

    def act(self, call):
        return []


class FakeService:
    def __init__(self, **k):
        pass

    def create_sampling_client(self, base_model=None, model_path=None):
        return None


class FakeDT(live.datetime):                       # Tue 24 Nov 2026, 09:31:30 ET: the 09:30 look, after its 09:30:45 wait
    @classmethod
    def now(cls, tz=None):
        return live.datetime(2026, 11, 24, 9, 31, 30, tzinfo=live.ET)


def scenario(account_ok):
    """One dry-run look on fresh ledgers. Returns what each sleeve's state was built from, the account info, the gateway's
    calls, the syncs and the log."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "archive").mkdir()
    sleeves = {"g0": {"model": "m-g0", "start_cash": 30000, "start": "2026-11-23"},
               "c0": {"model": "claude-test", "start_cash": 30000, "start": "2026-11-23"},
               "g1": {"model": "m-g1", "start_cash": 30000, "start": "2026-11-23"}}
    (tmp / "sleeves.json").write_text(json.dumps(sleeves), encoding="utf-8")
    (tmp / "fills.jsonl").write_text("".join(json.dumps({**e, "sleeve": paper.sleeve_of(e["ref"]),
                                                         "synced": "2026-11-23T20:05:00+00:00"}) + "\n" for e in PRIOR),
                                     encoding="utf-8")
    (tmp / "looks.json").write_text(json.dumps({s: {"reported_exec": [f"e-{s}-0"], "reported_orders": []} for s in sleeves}),
                                    encoding="utf-8")
    gw, seen, lines, syncs = FakeGW(), {}, [], []
    real_sync = paper.sync_fills

    def counting_sync(g):
        syncs.append(len(gw.calls))
        return real_sync(g)

    def fake_state_text(world, data, d, header=None, live=None, book=False, cash_clause=False):
        name = world.name
        b = paper.book(name, live)
        pos = {s: p["quantity"] for s, p in b["positions"].items()}
        vals = [p["quantity"] * (p["mark"] if p["mark"] is not None else p["avg_cost"]) for p in b["positions"].values()]
        opens = paper.open_orders(world.gw, name)
        seen[name] = {"pos": pos, "gross": sum(abs(v) for v in vals), "net": sum(vals),
                      "unmarked": sorted(s for s, p in b["positions"].items() if p["mark"] is None),
                      "open": sorted(o["ref"] for o in opens),
                      "remaining": {o["ref"]: o.get("remaining") for o in opens},
                      "on_snapshot": isinstance(world.gw, sys.modules["live"].OrdersAt),   # `live` here is the marks
                      "events": paper.events_since_last_look(world.gw, name, commit=False),
                      "account": getattr(world, "ACCOUNT", None), "value": b["cash"] + sum(vals)}
        return "state", {"value": b["cash"] + sum(vals), "events": seen[name]["events"]}, None

    @__import__("contextlib").contextmanager
    def fake_gateway():
        yield gw

    patches = [
        (paper, "FILLS", tmp / "fills.jsonl"), (paper, "ORDERS", tmp / "orders.jsonl"), (paper, "SLEEVES", tmp / "sleeves.json"),
        (paper, "DIVIDENDS", tmp / "dividends.jsonl"), (paper, "SPLITS", tmp / "splits.jsonl"), (paper, "ARCHIVE", tmp / "archive"),
        (paper, "LOOKS", tmp / "looks.json"), (paper, "NOTES", tmp / "notes.json"), (paper, "sync_fills", counting_sync),
        (live, "gateway", fake_gateway), (live, "reconcile", lambda g: {}), (live, "ml2_guard", lambda: None),
        (live, "open_whatif", lambda: (None, False)), (live, "datetime", FakeDT),
        (live, "read_account", (lambda link, g: {"NetLiquidation": 1.0}) if account_ok else (lambda link, g: None)),
        (live, "LOOKING", tmp / "LOOKING"), (live, "finish", lambda *a, **k: None), (live, "log", lines.append),
        (live, "load_memo", lambda n: {}), (live, "sessions_left", lambda t: 4), (live.ml_tools, "unknown", lambda s: []),
        (live.rates, "line", lambda now: ""), (live.tinker, "ServiceClient", FakeService), (tools, "available", lambda g: []),
        (agent, "gptoss_look", lambda *a, **k: (None, [], 0, 0)), (agent, "claude_look", lambda *a, **k: (None, [])),
        (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld), (trader, "state_text", fake_state_text),
        (trader, "tokenizer", lambda: None), (trader, "goal_line", lambda *a, **k: "goal"),
        (trader, "live_user", lambda *a, **k: "user"), (trader, "live_steps", lambda ml, **k: ""),
        (trader, "live_system", lambda *a, **k: "system"),
    ]
    saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
    for m, a, v in patches:
        setattr(m, a, v)
    try:
        live.look(dry_run=True)
    finally:
        for m, a, v in saved:
            setattr(m, a, v)
        shutil.rmtree(tmp, ignore_errors=True)
    return seen, gw, syncs, lines


WANT = {   # position, open orders, fill lines, gross (= net) of each state
    "g0": ({}, [], [f"filled BUY 100 A @ 49.00 (commission $1.00) [{STOPS['g0']}]"], 0.0,
           "A covered at the open; its stop gone from the orders"),
    "c0": ({"D": 10, "E": 12}, [LMT_E], [f"filled BUY 100 A @ 49.00 (commission $1.00) [{STOPS['c0']}]",
                                         f"filled BUY 10 D @ 20.00 (commission $1.00) [{LMT_D}]",
                                         f"filled BUY 12 E @ 5.00 (commission $1.00) [{LMT_E}]"], 260.0,
           "A covered; D's entry and 12 of E's 30, filled between its orders read and its sync, booked; D's order not "
           "listed, E's listed"),
    "g1": ({"B": 20}, [TP_B], [f"filled BUY 100 A @ 49.00 (commission $1.00) [{STOPS['g1']}]",
                               f"filled BUY 20 B @ 10.00 (commission $1.00) [{LMT_B}]"], 200.0,
           "A covered, 20 B bought; the take-profit that filled during its catch-up quote still working, B still held"),
}


def per_sleeve(seen, tag):
    for s in ("g0", "c0", "g1"):
        v = seen.get(s)
        if v is None:
            check(f"{tag} {s}: a state was built", False, sorted(seen))
            continue
        pos, open_, ev, g, why = WANT[s]
        check(f"{tag} {s}: the position ({why})", v["pos"] == pos, v["pos"])
        check(f"{tag} {s}: the open-orders list agrees with it", v["open"] == open_, v["open"])
        check(f"{tag} {s}: \"Since your last decision\" has the fills booked, and only those", v["events"] == ev, v["events"])
        check(f"{tag} {s}: gross and net from the same book (gross {g:,.0f}, net {g:,.0f}), every held name marked",
              abs(v["gross"] - g) < 1e-9 and abs(v["net"] - g) < 1e-9 and not v["unmarked"], (v["gross"], v["net"], v["unmarked"]))
        check(f"{tag} {s}: built on its catch-up's orders snapshot", v["on_snapshot"] is True)
    c0 = seen.get("c0") or {}
    check(f"{tag} c0: E's order, 12 of 30 booked, listed with 18 remaining (settled from the booked fills)",
          (c0.get("remaining") or {}).get(LMT_E) == 18, c0.get("remaining"))


seen, gw, syncs, lines = scenario(account_ok=True)
per_sleeve(seen, "account read:")
acct = (seen.get("g0") or {}).get("account") or {}
check("account read: the clause counts only the name no sleeve holds (10 R at its average cost, 6.00: no sleeve holds it, "
      "so the look has no mark), not the sleeves' own fills (20 B, $200)",
      abs((acct.get("unheld") or -1) - 60.0) < 1e-9, acct.get("unheld"))
want_total = 30098.0 + 30098.0 + 30097.0     # each sleeve's value with the open's fills and before c0's D and g1's take-profit
check("account read: the sleeves' total counts the open's fills (computed after the clause's sync)",
      acct.get("total") is not None and abs(acct["total"] - want_total) < 1e-6, (acct.get("total"), want_total))
check("account read: c0's D entry filled right after its catch-up read the orders, g1's take-profit during its catch-up "
      "quote of B (the two windows are exercised)", gw.d_at is not None and gw.tp_at is not None
      and gw.calls[gw.d_at - 1] == "orders" and gw.calls[gw.d_at] == "executions"
      and gw.calls[gw.tp_at - 1] == ("quotes", ("B",)), (gw.d_at, gw.tp_at, gw.calls))
check("account read: every world is on client 10 itself after its build (act and close_only read IBKR's orders live)",
      WORLDS and all(w.gw is gw for w in WORLDS), [type(w.gw).__name__ for w in WORLDS])
q_at = next(i for i, c in enumerate(gw.calls) if isinstance(c, tuple) and c[0] == "quotes")
check("account read: the look's first sync came before the open's fills, and every later sync after them (the clause's, "
      "then one before each sleeve's state)", len(syncs) == 5 and syncs[0] < q_at and all(x > q_at for x in syncs[1:]),
      (syncs, q_at))
check("account read: B (bought at the open, not in the look's quote list) was quoted for g1's state",
      any(isinstance(c, tuple) and c[0] == "quotes" and c[1] == ("B",) for c in gw.calls), [c for c in gw.calls if isinstance(c, tuple)])
check("account read: nothing logged as a failure", not any("no look" in m or "failed" in m or "not computed" in m for m in lines), lines)

WORLDS.clear()
seen2, gw2, syncs2, lines2 = scenario(account_ok=False)
per_sleeve(seen2, "account not read:")
check("account not read: no clause (the account line's failure), and the states still caught up before each was built",
      "unheld" not in ((seen2.get("g0") or {}).get("account") or {}) and len(syncs2) == 4, (syncs2, (seen2.get("g0") or {}).get("account")))
print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
