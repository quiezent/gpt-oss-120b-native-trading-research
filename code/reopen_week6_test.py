"""reopen_week6.py, offline: a fake broker that answers per client (client 18 for the re-open, client 10 for base-ml's own
orders: orders() and find() cover only the caller's client, executions(all_clients=True) and all_open_orders() every
client), a fake clock, and temp copies of the ledgers (base-ml's real fills and orders, Week 6's sleeves). Covers the
framework owner's ten findings and N1-N10 of 15 Nov 2026.

    python reopen_week6_test.py
"""
import atexit
import contextlib
import io
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import paper
import reopen_week6 as R

REAL = Path(__file__).resolve().parent / "runs"
SNAP = json.loads(R.SNAP.read_text(encoding="utf-8"))
TAKEN = datetime.fromisoformat(SNAP["taken"])
WANT = {s: q for s, q in SNAP["books"]["base-ml"].items() if q}
STALE = R.stale_net(SNAP)
SAVED = [o for o in SNAP["orders"] if o.get("sleeve") == "base-ml"]
ENTRIES = {"CRWV", "LSCC", "USDE"}
IDS = {"trader-reopen": 18, "trader-training": 10}
OK = []


def check(name, good, detail=""):
    OK.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + ("" if good else f": {str(detail)[:1500]}"))


class GwError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(f"{code}: {msg}" if code is not None else msg)
        self.code = code


class Broker:
    """The account: every client's orders and fills."""
    def __init__(self):
        self.base = {}                      # what positions() lists besides the fills (stale names)
        self.ords, self.execs, self.placed, self.found = [], [], [], []
        self.answer = {}                    # symbol or (symbol, order_type) -> {"status", "messages"} | Exception
        self.short, self.cash, self.bar_rows = {}, 1_000_000.0, []
        self.unknown_arrives, self.perm, self.connect_error = False, 9000, {}
        self.dies_after_ack = set()         # (symbol, order_type): IBKR cancels it right after a Submitted ack
        self.trail_stop = None              # IBKR's own stop for a TRAIL (None: echo what was sent)
        self.find_error = None
        self.after_place = None             # callback(broker) after each place()
        self.on_positions = None            # callback(broker) inside positions(), before it answers
        self.on_settle = None               # callback(broker) when the script waits for IBKR's later openOrder
        self.stats_calls = []
        self.stats_error = {}               # symbol -> IBKR's error for that symbol
        self.interrupt = {}                 # (symbol, order_type) -> "before" (IBKR never gets it) | "after" (it does)

    def fill(self, ref, shares=None, client=None, when="20261116 09:30:01 US/Eastern", price=10.0):
        o = next(o for o in self.ords if o["ref"] == ref)
        n = o["remaining"] if shares is None else shares
        self.add_exec(o["symbol"], "BOT" if o["action"] == "BUY" else "SLD", n, ref, o["client_id"], when, price, o["perm_id"])
        o["filled"] += n
        o["remaining"] -= n
        if not o["remaining"]:
            o["status"] = "Filled"

    def add_exec(self, sym, side, n, ref, client, when="20261116 09:30:01 US/Eastern", price=10.0, perm=0):
        self.execs.append({"exec_id": f"x{len(self.execs)}", "time": when, "symbol": sym, "side": side, "shares": n,
                           "price": price, "ref": ref, "perm_id": perm, "client_id": client, "liquidation": 0})
        self.cash -= (n if side == "BOT" else -n) * price

    def cancel_ref(self, ref):
        next(o for o in self.ords if o["ref"] == ref)["status"] = "Cancelled"

    def fill_all_reopens(self, skip=()):
        for o in list(self.ords):
            if o["ref"].startswith(R.PREFIX) and o["status"] not in R.DONE and o["symbol"] not in skip:
                self.fill(o["ref"])

    def reopens(self):
        return [p for p in self.placed if str(p["ref"]).startswith(R.PREFIX)]

    def reentries(self):
        return [p for p in self.placed if str(p["ref"]).startswith("tt-base-ml-")]


class FakeGW:
    """One client's connection to the broker."""
    def __init__(self, broker, client_id):
        self.b, self.cid = broker, client_id

    def close(self):
        self.b.closed = getattr(self.b, "closed", 0) + 1

    def positions(self):
        if self.b.on_positions:
            self.b.on_positions(self.b)
        pos = dict(self.b.base)
        for e in self.b.execs:
            pos[e["symbol"]] = pos.get(e["symbol"], 0) + (e["shares"] if e["side"] == "BOT" else -e["shares"])
        return [{"symbol": s, "quantity": q} for s, q in pos.items() if q]

    def account(self):
        return {"TotalCashValue": self.b.cash}

    def orders(self):                       # this client's own only
        return [{k: v for k, v in o.items() if k != "client_id"} for o in self.b.ords if o["client_id"] == self.cid]

    def all_open_orders(self):              # every client's resting orders, gateway._resting's keys (no filled)
        keys = ("client_id", "order_id", "perm_id", "ref", "symbol", "action", "quantity", "order_type", "limit", "aux",
                "trail_stop", "tif", "status")
        return [{k: o.get(k) for k in keys} for o in self.b.ords if o["status"] not in R.DONE]

    def executions(self, days=None, all_clients=False):
        return [dict(e) for e in self.b.execs if all_clients or e["client_id"] == self.cid]

    def find(self, ref):                    # this client's orders and today's fills
        self.b.found.append((self.cid, ref))
        if self.b.find_error:
            raise self.b.find_error
        return {"ref": ref, "orders": [o for o in self.orders() if o["ref"] == ref],
                "executions": [e for e in self.executions() if e["ref"] == ref]}

    def stats(self, syms):
        self.b.stats_calls.append(syms)
        if isinstance(syms, str):                   # gateway.stats: a single symbol's error is raised
            if syms in self.b.stats_error:
                raise GwError(self.b.stats_error[syms], 200)
            return {"symbol": syms, "shortable_shares": self.b.short.get(syms, 100000)}
        return [{"symbol": s, "error": self.b.stats_error[s]} if s in self.b.stats_error else
                {"symbol": s, "shortable_shares": self.b.short.get(s, 100000)} for s in syms]

    def bars(self, symbol, duration="1 Y", bar_size="1 day", **k):
        self.b.bars_asked = (symbol, duration, bar_size)
        return list(self.b.bar_rows)

    def place(self, symbol, action, quantity, order_type="MKT", limit=None, aux=None, tif="DAY", ref=None, **fields):
        self.b.placed.append(dict(symbol=symbol, action=action, quantity=quantity, order_type=order_type, limit=limit,
                                  aux=aux, tif=tif, ref=ref, client=self.cid, **fields))
        a = self.b.answer.get((symbol, order_type), self.b.answer.get(symbol, {"status": "PreSubmitted", "messages": []}))
        if isinstance(a, Exception):
            raise a
        if self.b.interrupt.get((symbol, order_type)) == "before":
            raise KeyboardInterrupt
        self.b.perm += 1
        o = {"order_id": self.b.perm, "perm_id": self.b.perm, "ref": ref, "symbol": symbol, "action": action,
             "quantity": quantity, "order_type": order_type, "limit": limit, "aux": aux, "tif": tif,
             "status": a["status"], "filled": 0, "remaining": quantity, "client_id": self.cid}
        if order_type == "TRAIL":
            o["trail_stop"] = self.b.trail_stop if self.b.trail_stop is not None else fields.get("trailStopPrice")
        if a["status"] != "Unknown" or self.b.unknown_arrives:
            if a["status"] == "Unknown":
                o["status"] = "PreSubmitted"
            elif a["status"] is None:
                o["status"] = "Inactive"
            self.b.ords.append(o)
        out = {k: v for k, v in o.items() if k != "client_id"}
        out.update(status=a["status"], messages=list(a.get("messages") or []))
        if (symbol, order_type) in self.b.dies_after_ack:
            o["status"] = "Cancelled"
        if self.b.after_place:
            self.b.after_place(self.b)
        if self.b.interrupt.get((symbol, order_type)) == "after":
            raise KeyboardInterrupt
        return out


def at(h, m, s=0, day=16):
    return datetime(2026, 11, day, h, m, s, tzinfo=R.ET)


class Env:
    def __init__(self, old_lineup=False, hold=True, held=True):
        self.d = Path(tempfile.mkdtemp(prefix="reopen6_"))
        atexit.register(shutil.rmtree, self.d, True)        # every run's temp folders go at exit
        # base-ml's real ledger and order log as they stood at Saturday's snapshot: the real re-open of Mon 16 Nov wrote its
        # re-entered orders and later fills (and the liquidation's rows) to the real files, which a test of the re-open from
        # the start must not see (it found them "sent earlier" and could not confirm them; 16 Nov evening)
        taken = datetime.fromisoformat(SNAP["taken"])
        rows = [r for r in paper.read(REAL / "fills.jsonl") if r.get("sleeve") == "base-ml" and R.fill_time(r["time"]) <= taken]
        (self.d / "fills.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        rows = [r for r in paper.read(REAL / "orders.jsonl") if r.get("sleeve") == "base-ml"
                and datetime.fromisoformat(r["time"]) <= taken]
        (self.d / "orders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        if (REAL / "splits.jsonl").exists():
            shutil.copy2(REAL / "splits.jsonl", self.d / "splits.jsonl")
        real = json.loads((REAL / "sleeves.json").read_text(encoding="utf-8"))
        w6 = json.loads((REAL / "week6_sleeves.json").read_text(encoding="utf-8"))
        sl = {"base-ml": real["base-ml"]}
        for k, v in w6.items():
            sl[k] = {"ml": False, "effort": "high", **v, "start_cash": float(v["start_cash"]), "start": "2026-11-16T11:00:00+00:00"}
        if old_lineup:
            sl["c1"] = {"start_cash": 30000.0, "start": "2026-11-09T11:00:00+00:00", "model": "base"}
        (self.d / "sleeves.json").write_text(json.dumps(sl, indent=1), encoding="utf-8")
        self.sleeves0 = json.loads((self.d / "sleeves.json").read_text(encoding="utf-8"))
        (self.d / "looks.json").write_text(json.dumps({"base-ml": {"reported_exec": [], "reported_orders": []}}), encoding="utf-8")
        (self.d / "notes.json").write_text("{}", encoding="utf-8")
        paper.RUNS = self.d
        paper.FILLS, paper.ORDERS, paper.SLEEVES = self.d / "fills.jsonl", self.d / "orders.jsonl", self.d / "sleeves.json"
        paper.LOOKS, paper.NOTES, paper.SPLITS = self.d / "looks.json", self.d / "notes.json", self.d / "splits.jsonl"
        paper.DIVIDENDS, paper.ARCHIVE = self.d / "dividends.jsonl", self.d / "archive"
        paper.now_iso = lambda: R.NOW().astimezone(timezone.utc).isoformat(timespec="seconds")
        R.HOLD_FILE, R.LOG, R.LOOKING = self.d / "HOLD", self.d / "reopen.jsonl", self.d / "LOOKING"
        R.LOOP, R.DONE_FILE = self.d / "LOOP.json", self.d / "done.json"
        R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 1, "started": "2026-11-16T04:00:00-05:00"}),
                          encoding="utf-8")
        self.alive = True                   # the loop's pid
        R.PID_ALIVE = lambda pid: self.alive and int(pid) == 4242
        R.PID_CREATED = lambda pid: None    # creation time unknown: the pid alone decides (the F2 checks fake it)
        self.done = [[day, job] for day in ("2026-11-16", "2026-11-17") for job in
                     ["refresh", "record"] + [f"look-{h:02d}{m:02d}" for h, m in R.LOOK_TIMES]]
        self.write_done()
        self.slept = []
        R.SLEEP = self.sleep
        R.HOLD_FILE.write_text("", encoding="utf-8")
        self.b = Broker()
        self.clients = []
        R.GATEWAY = self.factory
        self.when = at(9, 10)
        if held:
            code, t = self.run("--hold", "--go", when=at(6, 0))
            assert code == 0 and self.sleeves()["base-ml"]["model"] == "hold", t
            self.clients.clear()
        if not hold:
            R.HOLD_FILE.unlink()

    def write_done(self):
        R.DONE_FILE.write_text(json.dumps(self.done), encoding="utf-8")

    def undo(self, job, day="2026-11-16"):
        self.done = [d for d in self.done if d != [day, job]]
        self.write_done()

    def sleep(self, secs):
        self.slept.append(secs)
        if self.b.on_settle:
            self.b.on_settle(self.b)

    def factory(self, client):
        self.clients.append(client)
        if client in self.b.connect_error:
            raise self.b.connect_error[client]
        return FakeGW(self.b, IDS[client])

    def run(self, *args, when=None):
        if when is not None:
            self.when = when
        R.NOW = (lambda: self.when(self.b)) if callable(self.when) else (lambda: self.when)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = R.main(list(args))
        return code, buf.getvalue()

    def sleeves(self):
        return json.loads(paper.SLEEVES.read_text(encoding="utf-8"))

    def notes(self):
        return json.loads(paper.NOTES.read_text(encoding="utf-8")).get("base-ml", [])

    def held(self):
        return self.sleeves()["base-ml"]["model"] == "hold"


def result(text):
    return ([l for l in text.splitlines() if l.startswith("RESULT:")] or [""])[-1]


def ready(e):
    """--go at 09:10 and all 24 re-opens filled."""
    e.run("--go", when=at(9, 10))
    e.b.fill_all_reopens()


# ------------------------------------------------------------------ the constants match live.py and the framework
import live   # noqa: E402  (slow: the test only)
check("HOLD_MODEL and HOLD are live.py's", R.HOLD_MODEL == live.HOLD_MODEL
      and (Path(R.__file__).resolve().parent / "runs" / "live" / "HOLD") == live.HOLD.resolve())
clients = json.loads(Path(r"C:\Claude\TWS-Gateway\clients.json").read_text(encoding="utf-8"))
check("trader-reopen is client 18 and trader-training client 10 in the framework's clients.json",
      clients.get(R.REOPEN_CLIENT) == 18 and clients.get(R.ORDERS_CLIENT) == 10 == R.ORDERS_CLIENT_ID, clients)
check("the plan is the snapshot's 24 names and 11 orders; the stale net 70 names", len(WANT) == 24 and len(SAVED) == 11
      and len(STALE) == 70)
check("refs reopen-20261116-<SYM>[-n] map to no sleeve", paper.sleeve_of("reopen-20261116-SMCI") is None
      and paper.sleeve_of("reopen-20261116-SMCI-2") is None)

# ------------------------------------------------------------------ N1: which client each mode uses
e = Env(held=False)
code, t = e.run("--hold", when=at(6, 0))
code2, t2 = e.run("--hold", "--go", when=at(6, 0))
code3, t3 = e.run("--release", when=at(6, 1))
check("--hold, --hold --go and --release connect to no Gateway", e.clients == [] and code == code2 == code3 == 0, (e.clients, t2))
e.run(when=at(9, 10))
e.run("--go", when=at(9, 10))
check("status and --go connect on trader-reopen (18), and the re-opens are placed on client 18",
      e.clients == ["trader-reopen", "trader-reopen"] and {p["client"] for p in e.b.reopens()} == {18}, e.clients)
e.clients.clear()
e.b.fill_all_reopens()
e.run("--orders", when=at(9, 45))
e.run("--orders", "--go", when=at(9, 45))
check("--orders connects on trader-training (10), and base-ml's orders are placed on client 10",
      e.clients == ["trader-training", "trader-training"] and {p["client"] for p in e.b.reentries()} == {10}, e.clients)

# N1: client 10's windows
for h, m, blocked in ((7, 22, False), (7, 23, True), (7, 28, True), (8, 14, True), (8, 15, False), (8, 51, False),
                      (8, 52, True), (8, 57, True), (9, 11, True), (9, 12, False), (9, 21, False), (9, 22, True),
                      (9, 41, True), (9, 42, False), (9, 51, False), (9, 52, True), (9, 58, True), (10, 11, True),
                      (10, 12, False), (15, 41, True), (15, 42, False)):
    e = Env()
    code, t = e.run("--orders", when=at(h, m))
    check(f"--orders at {h:02d}:{m:02d} {'refuses without connecting' if blocked else 'connects'}",
          (code == 1 and e.clients == [] and "client 10" in result(t)) if blocked else e.clients == ["trader-training"], t)
e = Env()
code, t = e.run("--orders", "--force", when=at(10, 5))
check("--force overrides the clock windows", e.clients == ["trader-training"] and code == 0, t)
R.LOOKING.write_text("2026-11-16T10:19:00-05:00 4242", encoding="utf-8")
code, t = e.run("--orders", "--force", when=at(10, 20))
check("runs/live/LOOKING refuses client 10 even with --force", code == 1 and e.clients == ["trader-training"]
      and "LOOKING" in t and "does not override" in t, t)
R.LOOKING.unlink()
e = Env()
code, t = e.run("--go", when=at(10, 5))
check("the re-open (client 18) is not held to client 10's windows: --go at 10:05 sends", code == 0 and len(e.b.reopens()) == 24, t)
e = Env()
e.b.connect_error["trader-training"] = GwError("Unable to connect as the client id is already in use.", 326)
code, t = e.run("--orders", when=at(9, 20))
check("error 326 on client 10 says the loop holds it", code == 1 and "live loop" in t and "Nothing was sent" in t, t)
e = Env()
e.b.connect_error["trader-reopen"] = GwError("Unable to connect as the client id is already in use.", 326)
code, t = e.run("--go", when=at(9, 10))
check("error 326 on client 18 says another run holds it", code == 1 and "client trader-reopen is in use" in t
      and not e.b.placed, t)

# N1 + the coordinator: per client. Client 10 can't see client 18's orders; it must not judge them gone or resend them
e = Env()
e.b.answer["TLT"] = {"status": "Unknown", "messages": []}
e.b.unknown_arrives = True                               # IBKR did get it: it works on client 18
e.run("--go", when=at(9, 10))
gw10 = FakeGW(e.b, 10)
check("(the fake) client 10's find() and orders() of a client-18 ref come back empty; all_open_orders lists it",
      not gw10.find(R.PREFIX + "TLT")["orders"] and not [o for o in gw10.orders() if o["ref"].startswith(R.PREFIX)]
      and [o for o in gw10.all_open_orders() if o["ref"] == R.PREFIX + "TLT"])
e.b.found.clear()
e.b.fill_all_reopens(skip={"TLT"})
code, t = e.run("--orders", "--go", when=at(9, 45))
check("--orders on client 10 sees TLT's re-open still working (all_open_orders), does not verify it, sends no re-open",
      "not yet ['TLT']" in t and len(e.b.reopens()) == 24 and not [f for f in e.b.found if f[1].startswith(R.PREFIX)]
      and e.held(), t)
code, t = e.run("--go", when=at(9, 46))
check("the client-18 rerun sees TLT working and does not resend it", len(e.b.reopens()) == 24, t)
e = Env()
e.b.answer["IEF"] = {"status": "Unknown", "messages": []}
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens()
code, t = e.run("--orders", "--go", when=at(9, 45))
check("an uncertain re-open ref is never judged from client 10: IEF waits, its stop is not re-entered, nothing logged",
      "waiting on reopen-20261116-IEF" in t and "client 10 cannot see" in t and "IEF" not in {p["symbol"] for p in e.b.reentries()}
      and not [r for r in paper.read(R.LOG) if r.get("find_empty")], t)
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens()
code, t = e.run("--orders", when=at(9, 45))
check("fills on client 18 verify every name from client 10 (executions with all_clients=True)", "verified 24 of 24" in t, t)

# ------------------------------------------------------------------ N2: the hold first
e = Env(held=False)
code, t = e.run("--go", when=at(9, 10))
check("--go refuses unless base-ml is held, and sends nothing", code == 1 and not e.b.placed and "--hold --go first" in t, t)
code, t = e.run(when=at(9, 10))
check("the status says the HOLD file comes off only once base-ml shows held", R.HOLD_SAYS in t and "hold: free" in t, t)
e.b.connect_error["trader-reopen"] = RuntimeError("no Gateway")
code, t = e.run("--hold", when=at(5, 0))
check("--hold without --go changes nothing", code == 0 and not e.held(), t)
code, t = e.run("--hold", "--go", when=at(5, 1, day=17))
check("--hold --go refuses on another date", code == 1 and not e.held() and "not 2026-11-16" in t, t)
code, t = e.run("--hold", "--go", "--date", "2026-11-17", when=at(5, 1, day=17))
check("--hold --go with --date (tests) at 05:01, inside the restart: no Gateway needed", code == 0 and e.held(), t)
e = Env(held=False)
code, t = e.run("--hold", "--go", when=at(0, 30))
s = e.sleeves()
check("--hold --go: model 'hold', the real model in model_held, nothing else changed",
      code == 0 and s["base-ml"]["model"] == "hold" and s["base-ml"]["model_held"] == e.sleeves0["base-ml"]["model"]
      and {k: v for k, v in s.items() if k != "base-ml"} == {k: v for k, v in e.sleeves0.items() if k != "base-ml"}
      and {k: v for k, v in s["base-ml"].items() if k not in ("model", "model_held")}
      == {k: v for k, v in e.sleeves0["base-ml"].items() if k != "model"} and not list(e.d.glob("*.tmp")), s)
code, t = e.run("--hold", "--go", when=at(0, 31))
check("--hold --go again leaves it as it is", code == 0 and e.sleeves() == s and "base-ml is held" in t, t)
e = Env(held=False)
e.run("--go", "--date", "2026-11-16", when=at(9, 10))
e.b.fill_all_reopens()
code, t = e.run("--orders", "--go", when=at(9, 45))
check("--orders --go refuses unless base-ml is held", code == 1 and not e.b.reentries() and "--hold --go first" in t, t)
e = Env()
s = e.sleeves()
s["base-ml"].pop("model_held")
paper.SLEEVES.write_text(json.dumps(s, indent=1), encoding="utf-8")
code, t = e.run("--hold", "--go", when=at(6, 0))
check("a base-ml already 'hold' without model_held: --hold --go refuses to touch it", code == 1 and "hold-unknown" in t, t)

# ------------------------------------------------------------------ finding 1: reruns
e = Env()
code, t = e.run(when=at(9, 10))
check("status: exit 0, places nothing, shows 24 names", code == 0 and not e.b.placed and "24 names" in t, t)
code, t = e.run("--go", when=at(9, 10))
rp = e.b.reopens()
check("--go at 09:10 places all 24 as MKT OPG under reopen-20261116-<SYM>", code == 0 and len(rp) == 24
      and all(p["tif"] == "OPG" and p["order_type"] == "MKT" and p["ref"] == R.PREFIX + p["symbol"] for p in rp)
      and {p["symbol"]: (p["quantity"] if p["action"] == "BUY" else -p["quantity"]) for p in rp} == WANT, t)
code, t = e.run("--go", when=at(9, 15))
check("a rerun while the 24 are working places nothing", code == 0 and len(e.b.reopens()) == 24, t)
e.b.fill_all_reopens()
code, t = e.run("--go", when=at(9, 35))
check("a rerun after all filled places nothing", code == 0 and len(e.b.reopens()) == 24 and "complete" in t, t)
e = Env()
e.run("--go", when=at(9, 10))
four = ["TLT", "COIN", "U", "QYLD"]
for s in four:
    e.b.cancel_ref(R.PREFIX + s)
e.b.fill_all_reopens()
code, t = e.run("--go", when=at(9, 40))
new = e.b.reopens()[24:]
check("after 20 of 24 filled and 4 cancelled, a rerun places only the 4, as -2, MKT DAY",
      code == 0 and sorted(p["symbol"] for p in new) == sorted(four)
      and all(p["ref"] == f"{R.PREFIX}{p['symbol']}-2" and p["tif"] == "DAY" and p["quantity"] == abs(WANT[p["symbol"]])
              for p in new), t)
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill(R.PREFIX + "SMCI", 100)
e.b.cancel_ref(R.PREFIX + "SMCI")
e.b.fill_all_reopens()
code, t = e.run("--go", when=at(9, 40))
new = e.b.reopens()[24:]
check("SMCI filled 100 of 191 then cancelled: a rerun sells only 91 as reopen-20261116-SMCI-2",
      code == 0 and len(new) == 1 and new[0]["ref"] == R.PREFIX + "SMCI-2" and new[0]["quantity"] == 91, t)
e.b.cancel_ref(R.PREFIX + "SMCI-2")
e.run("--go", when=at(9, 50))
check("and if -2 is cancelled too, the next is -3 for the same 91", e.b.reopens()[-1]["ref"] == R.PREFIX + "SMCI-3"
      and e.b.reopens()[-1]["quantity"] == 91)

# ------------------------------------------------------------------ N3: orders before executions; done = max per ref
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens()
for o in e.b.ords:                                       # the order rows still say working, nothing filled; the fills are in
    if o["symbol"] == "SMCI":
        o.update(status="Submitted", filled=0, remaining=191)
code, t = e.run("--go", when=at(9, 31, 6))
check("a fill IBKR's order row doesn't show yet is counted from executions: nothing resent", len(e.b.reopens()) == 24, t)
e = Env()
e.run("--go", when=at(9, 10))
o = next(o for o in e.b.ords if o["symbol"] == "SMCI")
o.update(status="Filled", filled=191, remaining=0)       # the row says filled; the execution hasn't been served yet
e.b.fill_all_reopens(skip={"SMCI"})
code, t = e.run("--go", when=at(9, 31, 6))
check("a fill only the order row shows (filled) is counted too: nothing resent", len(e.b.reopens()) == 24, t)
e = Env()
code, t = e.run("--go", when=at(9, 30, 30))
check("09:30:30 refuses (MKT DAY starts at 09:31:00)", code == 1 and not e.b.placed and "09:31:00" in t, t)
e = Env()
code, t = e.run("--go", when=at(9, 31, 0))
check("09:31:00 sends MKT DAY", code == 0 and {p["tif"] for p in e.b.reopens()} == {"DAY"}, t)
e = Env()
code, t = e.run("--go", when=at(9, 24, 59))
check("09:24:59 sends MKT OPG", code == 0 and {p["tif"] for p in e.b.reopens()} == {"OPG"}, t)
e = Env()
code, t = e.run("--go", when=at(9, 26))
check("09:26 refuses and sends nothing", code == 1 and not e.b.placed, t)
e = Env()
code, t = e.run("--go", when=lambda b: at(9, 24, 58) if len(b.placed) < 5 else at(9, 25, 1))
rp = e.b.reopens()
check("the tif is chosen per order: the clock passing 09:25 mid-run stops the rest, said plainly",
      code == 2 and len(rp) == 5 and {p["tif"] for p in rp} == {"OPG"} and "too late for the opening auction" in t, t)

# ------------------------------------------------------------------ finding 2 and N4: what counts as placed; 'Unknown'
e = Env()
e.b.answer["COIN"] = {"status": "Inactive", "messages": ["201: Order rejected - reason: margin"]}
e.b.answer["IEF"] = {"status": "PreSubmitted", "messages": ["404: Order held while securities are located"]}
e.b.answer["XLU"] = {"status": None, "messages": ["321: Error validating request"]}
e.b.answer["XLI"] = GwError("Not connected", 504)
e.b.answer["NVTS"] = {"status": "PreSubmitted", "messages": ["399: Order held until the open"]}
code, t = e.run("--go", when=at(9, 10))
check("Inactive/201, PreSubmitted with 404, None/321 and an exception are failed; 399 is placed; exit not 0",
      code != 0 and all(f"failed    {a}" in t for a in ("SELL 20 COIN", "SELL 130 IEF", "BUY 38 XLU", "BUY 13 XLI"))
      and "not sent: GwError" in t and "placed    BUY 5 NVTS" in t and "201: Order rejected" in t, t)

e = Env()
e.b.answer["TLT"] = {"status": "Unknown", "messages": ["No acknowledgement within 30s"]}
code, t = e.run("--go", when=at(9, 10))
check("'Unknown' is uncertain and sent once in that run; exit not 0",
      code != 0 and len([p for p in e.b.reopens() if p["symbol"] == "TLT"]) == 1 and "uncertain" in t, t)
code, t = e.run(when=at(9, 13))
check("a status run (no --go) asks find() but records nothing", not [r for r in paper.read(R.LOG) if r.get("find_empty")]
      and (18, R.PREFIX + "TLT") in e.b.found and "waiting on reopen-20261116-TLT" in t, t)
code, t = e.run("--go", when=at(9, 13))
tl = [p for p in e.b.reopens() if p["symbol"] == "TLT"]
check("3 minutes after: find() empty, recorded; 'waiting on' it, nothing sent for TLT", len(tl) == 1
      and "waiting on reopen-20261116-TLT" in t and len([r for r in paper.read(R.LOG) if r.get("find_empty")]) == 1, t)
code, t = e.run("--go", when=at(9, 15))
check("5 minutes after the send, but only 2 after the first empty find(): still waiting",
      len([p for p in e.b.reopens() if p["symbol"] == "TLT"]) == 1 and "needs 3" in t, t)
e.b.ords.append({"order_id": 1, "perm_id": 77, "ref": R.PREFIX + "TLT", "symbol": "TLT", "action": "SELL", "quantity": 30,
                 "order_type": "MKT", "status": "PreSubmitted", "filled": 0, "remaining": 30, "client_id": 1})
code, t = e.run("--go", when=at(9, 17))
check("listed by all_open_orders (another client's view), not by find(): found, not resent",
      len([p for p in e.b.reopens() if p["symbol"] == "TLT"]) == 1, t)
e.b.ords.pop()
code, t = e.run("--go", when=at(9, 17))
tl = [p for p in e.b.reopens() if p["symbol"] == "TLT"]
check("sent 7 minutes ago, two empty find()s 4 minutes apart, not listed: gone; the remainder goes as -2",
      len(tl) == 2 and tl[1]["ref"] == R.PREFIX + "TLT-2" and "gone" in t, t)
e = Env()
e.b.answer["TLT"] = {"status": "Unknown", "messages": []}
e.b.unknown_arrives = True
e.run("--go", when=at(9, 10))
code, t = e.run("--go", when=at(9, 30, 59))
check("an 'Unknown' that IBKR does have is never resent", len([p for p in e.b.reopens() if p["symbol"] == "TLT"]) == 1, t)

e = Env()
e.b.short["SMCI"] = 100
code, t = e.run("--go", when=at(9, 10))
check("a short with too few shortable shares (SMCI 100 < 191) is not sent and reported; exit not 0",
      code != 0 and "SMCI" not in [p["symbol"] for p in e.b.reopens()] and len(e.b.reopens()) == 23
      and "shortable_shares 100 < 191" in t, t)
check("stats was asked for the names to be shorted only, one symbol per call (F4)",
      all(isinstance(c, list) and len(c) == 1 for c in e.b.stats_calls)
      and sorted(c[0] for c in e.b.stats_calls) == sorted(s for s, q in WANT.items() if q < 0),
      e.b.stats_calls)

# ------------------------------------------------------------------ N5, N6: positions per symbol, from IBKR
e = Env()
e.b.base = dict(STALE)
code, t = e.run("--go", when=at(9, 10))
check("the stale 70 without --stale-ok refuses and sends nothing", code == 1 and not e.b.placed and "still listed" in t, t)
code, t = e.run("--go", "--stale-ok", when=at(9, 10))
check("the stale 70 with --stale-ok passes and places 24", code == 0 and len(e.b.reopens()) == 24, t)
e = Env()
e.b.base = {s: q for i, (s, q) in enumerate(sorted(STALE.items())) if i % 2}
code, t = e.run("--go", "--stale-ok", when=at(9, 10))
half = [s for i, s in enumerate(sorted(STALE)) if not i % 2]
check("a partial drop (35 of the 70 still listed) passes with --stale-ok and prints the dropped names",
      code == 0 and len(e.b.reopens()) == 24 and "dropped 35" in t and all(s in t for s in half[:5]), t)
e = Env()
e.b.base = {s: q for i, (s, q) in enumerate(sorted(STALE.items())) if i % 2}
code, t = e.run("--go", when=at(9, 10))
check("the same without --stale-ok refuses", code == 1 and not e.b.placed and "35 stale names are still listed" in t, t)
e = Env()
e.b.base = dict(STALE)
e.b.base["AAPL"] += 1
code, t = e.run("--go", "--stale-ok", when=at(9, 10))
check("a stale name at neither value refuses and is printed", code == 1 and not e.b.placed and "'AAPL': (16, 0, 15)" in t, t)
e = Env()
e.b.base = {"SPY": 3}
code, t = e.run("--go", "--stale-ok", when=at(9, 10))
check("a position at neither value refuses", code == 1 and not e.b.placed, t)
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens()
for s in ("TLT", "COIN"):
    next(o for o in e.b.ords if o["ref"] == R.PREFIX + s).update(status="Cancelled", filled=0, remaining=0)
e.b.execs = [x for x in e.b.execs if x["ref"] not in (R.PREFIX + "TLT", R.PREFIX + "COIN")]
e.b.add_exec("NVDA", "BOT", 7, "tt-cand1-0001", 10, "20261116 09:35:00 US/Eastern", price=180.0)    # a Week-6 sleeve's buy
e.b.add_exec("XLE", "SLD", 30, "tt-base-ml-0002", 10, "20261116 09:36:00 US/Eastern", price=70.0)   # base-ml's own
code, t = e.run("--go", when=at(9, 40))
check("Week-6 and base-ml tt- fills since the snapshot are taken from IBKR's executions: positions pass, only TLT and COIN go",
      code == 0 and sorted(p["symbol"] for p in e.b.reopens()[24:]) == ["COIN", "TLT"], t)
e.b.add_exec("AMD", "BOT", 3, "tt-c1-0003", 10, "20261116 09:41:00 US/Eastern")                      # a sleeve not in the lineup
code, t = e.run("--go", when=at(9, 42))
check("a fill whose ref maps to no current sleeve is not explained: refuse", code == 1 and "AMD" in t, t)

# ------------------------------------------------------------------ N7: the cash check after the open
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens()
e.b.add_exec("SPY", "BOT", 100, "tt-cand2-0001", 10, "20261116 09:35:00 US/Eastern", price=650.0)     # $65,000 spent
e.b.base = {"SPY": 0}
code, t = e.run(when=at(9, 40))
check("after the Week-6 sleeves buy $65,000, the cash check (fills taken out) still passes", code == 0 and "below" not in t
      and e.b.cash < 990000, t)
e.b.cash -= 20000
code, t = e.run(when=at(9, 41))
check("$20,000 missing that no fill explains refuses", code == 1 and "below 990,000" in t, t)

# ------------------------------------------------------------------ finding 5 and 9: snapshot and lineup
e = Env()
paper.append(paper.FILLS, {"exec_id": "later1", "time": "20261116 10:05:00 US/Eastern", "symbol": "XLE", "side": "SLD",
                           "shares": 30, "price": 70.0, "ref": "tt-base-ml-later", "sleeve": "base-ml", "commission": 1.0})
code, t = e.run(when=at(9, 10))
check("a base-ml ledger fill after the snapshot doesn't stop the plan", "differ" not in t, t)
e = Env()
paper.append(paper.FILLS, {"exec_id": "early1", "time": "20261113 15:00:00 US/Eastern", "symbol": "XLE", "side": "BOT",
                           "shares": 1, "price": 70.0, "ref": "tt-base-ml-early", "sleeve": "base-ml", "commission": 1.0})
code, t = e.run("--go", when=at(9, 10))
check("a ledger that differs from the snapshot at its time refuses", code == 1 and not e.b.placed and "differ" in t, t)
e = Env(old_lineup=True)
code, t = e.run("--go", when=at(9, 10))
check("last week's sleeves.json refuses, with the reset_week.py reason", code == 1 and not e.b.placed
      and "reset_week.py must run first" in t, t)
e = Env(hold=False)
code, t = e.run("--go", when=at(8, 50))
check("no runs/live/HOLD before 09:00 refuses", code == 1 and not e.b.placed and "HOLD" in t, t)
code, t = e.run("--go", when=at(9, 10))
check("after 09:00 HOLD isn't required", code == 0 and len(e.b.reopens()) == 24, t)
e = Env()
code, t = e.run("--go", when=at(9, 10, day=17))
check("the wrong ET date refuses", code == 1 and not e.b.placed and "not 2026-11-16" in t, t)
e = Env()
code, t = e.run("--go", when=at(5, 0))
check("05:00 ET (the Gateway's restart) refuses without connecting", code == 1 and not e.clients, t)

# ------------------------------------------------------------------ finding 3, 7, 8, N8, N9: the orders and the note
e = Env()
e.run("--go", when=at(9, 10))
code, t = e.run("--orders", "--go", when=at(9, 14))
re1 = e.b.reentries()
check("before the re-open fills, only CRWV, LSCC and USDE are re-entered; base-ml stays held",
      {p["symbol"] for p in re1} == ENTRIES and len(re1) == 3 and code == 2 and e.held(), t)
n1 = e.notes()
check("the first note lists the 3 with their new refs and the 8 waiting", len(n1) == 1 and all(p["ref"] in n1[0] for p in re1)
      and "Not re-entered yet" in n1[0] and "broker refused" not in n1[0], n1)
lk = json.loads(paper.LOOKS.read_text(encoding="utf-8"))["base-ml"]["reported_orders"]
check("only the 3 dealt-with originals are marked as told", sorted(lk) == sorted(str(o["perm_id"]) for o in SAVED if o["symbol"] in ENTRIES))
e.b.fill_all_reopens(skip={"NVDA"})
e.b.bar_rows = [{"date": "20261112 14:00:00 US/Eastern", "high": 99.0, "low": 60.0},      # ends 14:30, before 14:34 ET
                {"date": "20261112 14:30:00 US/Eastern", "high": 66.0, "low": 64.0},
                {"date": "20261113 10:00:00 US/Eastern", "high": 68.0, "low": 65.0},
                {"date": "20261116 09:30:00 US/Eastern", "high": 80.0, "low": 66.0}]      # Monday: after the last close
e.b.trail_stop = 64.75                                                                    # IBKR's own figure
code, t = e.run("--orders", "--go", when=at(9, 45))
re2 = e.b.reentries()[3:]
xle = [p for p in re2 if p["symbol"] == "XLE"]
check("with NVDA not filled its stop waits; the other 7 go in", code == 2 and len(re2) == 7 and "NVDA" not in {p["symbol"] for p in re2})
check("the XLE trail carries trailStopPrice 64.60 (95% of 68.00) with its 5%", len(xle) == 1
      and xle[0].get("trailStopPrice") == 64.6 and xle[0].get("trailingPercent") == 5.0 and e.b.bars_asked == ("XLE", "10 D", "30 mins"), xle)
log_ts = [r for r in paper.read(R.LOG) if r.get("ref") == xle[0]["ref"] and "trail_stop" in r]
check("IBKR's stop is read back from orders() and logged", log_ts and log_ts[0]["trail_stop"] == 64.75
      and log_ts[0]["sent_trail_stop"] == 64.6, log_ts)
n2 = e.notes()
check("the note carries IBKR's own stop figure", len(n2) == 2 and "follow-up" in n2[1]
      and "with IBKR's stop at 64.75 (sent as 64.60: 5% below XLE's highest regular-hours trade (68.00)" in n2[1], n2[-1:])
orig = {o["symbol"]: o for o in SAVED}
check("the re-entered orders keep the saved terms", all(
    p["action"] == orig[p["symbol"]]["action"] and p["quantity"] == orig[p["symbol"]]["quantity"]
    and p["order_type"] == orig[p["symbol"]]["order_type"] and p["tif"] == "GTC"
    and (p["aux"] == orig[p["symbol"]]["aux"] if p["order_type"] == "STP" else
         p["limit"] == orig[p["symbol"]]["limit"] if p["order_type"] == "LMT" else True) for p in re2))
e.b.fill_all_reopens()
code, t = e.run("--orders", "--go", when=at(9, 50))
check("once NVDA fills, its stop goes in and base-ml is released (exit 0)", code == 0
      and [p["symbol"] for p in e.b.reentries()[10:]] == ["NVDA"] and "released base-ml" in t and e.sleeves() == e.sleeves0, t)
lk = json.loads(paper.LOOKS.read_text(encoding="utf-8"))["base-ml"]["reported_orders"]
check("all 11 originals are now marked as told, once each", sorted(lk) == sorted(str(o["perm_id"]) for o in SAVED), lk)
code, t = e.run("--orders", "--go", when=at(9, 55, 0))
check("(09:55 is inside the 10:00 look's window: refused)", code == 1, t)
code, t = e.run("--orders", when=at(10, 15))
check("after the release, --orders shows nothing to do", code == 0 and "nothing to do" in t, t)
code, t = e.run("--orders", "--go", when=at(10, 15))
check("and --orders --go refuses (base-ml is free: its own orders are the model's again), re-enters nothing, writes no note",
      code == 1 and len(e.b.reentries()) == 11 and len(e.notes()) == 3 and "not held" in t, t)
code, t = e.run("--go", when=at(10, 16))
check("a rerun of --go after everything: nothing to send, exit 0", code == 0 and len(e.b.reopens()) == 24, t)

# N8: a re-entered order IBKR cancels after its ack is not placed
e = Env()
ready(e)
e.b.dies_after_ack.add(("RBLX", "STP"))
code, t = e.run("--orders", "--go", when=at(9, 45))
n = e.notes()[-1]
check("an order IBKR cancels right after a Submitted ack is not counted placed; base-ml stays held",
      code == 2 and e.held() and "Sent, not working: BUY 30 RBLX STP stop 55 GTC" in n and "IBKR later reported Cancelled" in n, n)
e = Env()
ready(e)
e.run("--orders", "--go", when=at(9, 45))
code, t = e.run("--release", "--go", when=at(9, 46))             # pretend released early, then hold again for the check
e.run("--hold", "--go", when=at(9, 46))
next(o for o in e.b.ords if o["ref"] == next(p["ref"] for p in e.b.reentries() if p["symbol"] == "IEF"))["status"] = "Cancelled"
code, t = e.run("--orders", "--go", when=at(9, 50))
check("a placed re-entry IBKR later cancels is re-read by ref: not confirmed, re-entered once, then released",
      [p["symbol"] for p in e.b.reentries()[11:]] == ["IEF"] and code == 0 and not e.held()
      and "IBKR later reported Cancelled" in e.notes()[-1], t)

# the note's wording; Inactive, an exception and an Unknown re-entry
e = Env()
ready(e)
e.b.answer[("AMD", "LMT")] = {"status": "Inactive", "messages": ["201: Order rejected - reason: x"]}
e.b.answer[("INTC", "LMT")] = RuntimeError("socket closed")
e.b.answer[("PLTR", "LMT")] = {"status": "Unknown", "messages": ["No acknowledgement within 30s"]}
paper.ORDERS.write_text("".join(json.dumps(r) + "\n" for r in paper.read(paper.ORDERS)
                                if r.get("ref") != "tt-base-ml-bc6b934dbc"), encoding="utf-8")   # the XLE trail's own row gone
code, t = e.run("--orders", "--go", when=at(9, 45))
n = e.notes()[-1]
check("the note: Inactive with IBKR's 201 as 'Sent, not working', the exception as 'Not sent', Unknown as unconfirmed",
      code == 2 and e.held() and "Sent, not working: BUY 2 AMD LMT limit 500 GTC" in n and "201: Order rejected" in n
      and "Not sent: BUY 5 INTC LMT limit 55 GTC: RuntimeError: socket closed" in n
      and "BUY 2 PLTR" in n.split("Sent, unconfirmed")[1] and "broker refused" not in n, n)
check("the XLE trail without its placement row goes in without a stop price, and the note says so",
      "trailStopPrice" not in [p for p in e.b.placed if p["symbol"] == "XLE" and p["order_type"] == "TRAIL"][0]
      and "could not be restored" in n, n)
lk = json.loads(paper.LOOKS.read_text(encoding="utf-8"))["base-ml"]["reported_orders"]
pltr = str(next(o["perm_id"] for o in SAVED if o["symbol"] == "PLTR"))
check("the Unknown one's original is not marked as told; the dealt-with 10 are", pltr not in lk and len(lk) == 10, lk)
code, t = e.run("--orders", "--go", when=at(9, 46))
again = sorted(p["symbol"] for p in e.b.reentries()[11:])
check("a rerun 1 minute later: AMD and INTC are tried again; PLTR waits (sent 1 minute ago)",
      again == ["AMD", "INTC"] and "waiting on" in t and (10, e.b.reentries()[4]["ref"]) in e.b.found, (again, t))
code, t = e.run("--orders", "--go", when=at(9, 51))
check("6 minutes after the send and after the first empty find(): PLTR is gone and re-entered once (AMD, INTC again)",
      sorted(p["symbol"] for p in e.b.reentries()[13:]) == ["AMD", "INTC", "PLTR"] and "last try gone" in t, t)

# ------------------------------------------------------------------ N10: every Gateway error ends with a RESULT line
e = Env()
e.b.answer["TLT"] = {"status": "Unknown", "messages": []}
e.run("--go", when=at(9, 10))
e.b.find_error = GwError("Not connected to IB Gateway", 504)
code, t = e.run("--go", when=at(9, 20))
check("find() failing (a dropped connection) ends the run with a RESULT line and exit 1, no traceback",
      code == 1 and result(t).startswith("RESULT: stopped by an error (GwError") and "Traceback" not in t, t)
e = Env()
ready(e)
e.b.find_error = GwError("Not connected", 504)
e.b.answer[("PLTR", "LMT")] = {"status": "Unknown", "messages": []}
code, t = e.run("--orders", "--go", when=at(9, 45))
code, t = e.run("--orders", "--go", when=at(9, 46))
check("the same on client 10 (--orders)", code == 1 and result(t).startswith("RESULT: stopped by an error"), t)

# --release
e = Env()
code, t = e.run("--release", when=at(9, 20))
check("--release without --go changes nothing", code == 0 and e.held(), t)
code, t = e.run("--release", "--go", when=at(9, 20))
check("--release --go restores base-ml's model exactly", code == 0 and e.sleeves() == e.sleeves0, t)

# finding 10: the note waits when a look window opens mid-run
e = Env()
ready(e)
code, t = e.run("--orders", "--go", when=lambda b: at(10, 21, 50) if len(b.reentries()) < 11 else at(10, 27, 5))
check("a look window opening mid-run: the orders go in but the note waits, and base-ml stays held",
      code == 2 and len(e.b.reentries()) == 11 and not e.notes() and "NOT WRITTEN" in t and e.held(), t)
code, t = e.run("--orders", "--go", when=at(10, 45))
check("the rerun writes the note for the 11 (none re-sent) and releases",
      code == 0 and len(e.b.reentries()) == 11 and len(e.notes()) == 1 and e.notes()[0].count("[tt-base-ml-") == 11
      and e.sleeves() == e.sleeves0, e.notes())

# ------------------------------------------------------------------ F1: LOOP.json and the loop's pid
import os  # noqa: E402
check("pid_alive (the real check, OpenProcess on Windows): this process is alive; an unused pid and None are not",
      R.pid_alive(os.getpid()) and not R.pid_alive(4_000_001) and not R.pid_alive(None))
e = Env()
R.LOOP.unlink()
code, t = e.run("--orders", when=at(9, 20))
check("no runs/live/LOOP.json: --orders refuses without connecting", code == 1 and not e.clients and "LOOP.json is missing" in t, t)
R.LOOP.write_text(json.dumps({"pid": 4242, "code": "old"}), encoding="utf-8")
code, t = e.run("--orders", when=at(9, 20))
check("LOOP.json without \"looking\" (an old loop): refuses", code == 1 and not e.clients and "no \"looking\"" in t, t)
R.LOOP.write_text("{not json", encoding="utf-8")
code, t = e.run("--orders", "--force", when=at(9, 20))
check("an unreadable LOOP.json refuses, even with --force", code == 1 and not e.clients, t)
e = Env()
e.alive = False
R.LOOKING.write_text("2026-11-16T10:00:01 4242", encoding="utf-8")
code, t = e.run("--orders", when=at(10, 15))
check("the loop not running (pid gone) at 10:15, outside every window: connects, says so; its stale LOOKING is ignored",
      code == 0 and e.clients == ["trader-training"] and "not running" in t and "stale: its pid 4242 is gone" in t, t)
e = Env()
R.LOOKING.write_text("2026-11-16T09:44:00 4242", encoding="utf-8")
code, t = e.run("--orders", when=at(9, 44))
check("the loop running and LOOKING present (a Claude lane at 09:44): refuses", code == 1 and not e.clients and "LOOKING" in t, t)

# ------------------------------------------------------------------ F2: due jobs, the record window, the re-check per place()
e = Env()
e.undo("look-1000")
code, t = e.run("--orders", "--force", when=at(10, 15))
check("look-1000 not yet in done.json at 10:15 (within its 25 minutes): refuses, even with --force",
      code == 1 and not e.clients and "look-1000 is due" in t, t)
e.done.append(["2026-11-16", "look-1000"]); e.write_done()
code, t = e.run("--orders", when=at(10, 15))
check("once look-1000 is in done.json, 10:15 connects", code == 0 and e.clients == ["trader-training"], t)
e = Env()
e.undo("refresh")
code, t = e.run("--orders", when=at(9, 20))
check("the refresh not done at 09:20 (it retries until 09:30): refuses", code == 1 and "refresh is due" in t, t)
code, t = e.run("--orders", when=at(9, 45))
check("after 09:30 the undone refresh doesn't block", code == 0, t)
e = Env()
code, t = e.run("--orders", when=at(16, 20))
check("16:20 (the record window 16:13-16:45) refuses", code == 1 and "16:15 record" in t, t)
code, t = e.run("--orders", when=at(16, 46))
check("16:46 with the record done connects", code == 0, t)
e.undo("record")
code, t = e.run("--orders", when=at(16, 50))
check("16:50 with the record not yet done refuses", code == 1 and "record is due" in t, t)
e = Env()
ready(e)
e.b.after_place = lambda b: (R.LOOKING.write_text(f"{R.NOW().isoformat()} 4242", encoding="utf-8")
                             if len(b.reentries()) == 3 else None)
code, t = e.run("--orders", "--go", when=at(9, 45))
check("a look starting mid-run (LOOKING after the 3rd place): stops at once, 3 re-entered, nothing written, held, exit 2",
      code == 2 and len(e.b.reentries()) == 3 and not e.notes() and e.held() and "STOPPED before the next place()" in t
      and result(t).startswith("RESULT: incomplete") and "3 re-entered in this run, 8 not sent" in result(t), t)
R.LOOKING.unlink()
e.b.after_place = None
code, t = e.run("--orders", "--go", when=at(9, 46))
check("the rerun continues: the other 8 only, then one note for all 11, released",
      code == 0 and len(e.b.reentries()) == 11 and len(e.notes()) == 1 and e.notes()[0].count("[tt-base-ml-") == 11
      and not e.held(), t)
e = Env()
ready(e)
code, t = e.run("--orders", "--go", when=lambda b: at(9, 51, 50) if len(b.reentries()) < 5 else at(9, 57, 5))
check("the clock entering a window mid-run also stops it before the next place()", code == 2 and len(e.b.reentries()) == 5
      and "10:00 look" in t and not e.notes(), t)

# ------------------------------------------------------------------ F3: executions on both sides of positions and account
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens(skip={"SMCI"})
def land_once(b):
    if not getattr(b, "landed", False):
        b.landed = True
        b.fill(R.PREFIX + "SMCI")
e.b.on_positions = land_once
code, t = e.run("--go", when=at(9, 31, 5))
check("a fill landing between the executions and positions reads: read again, then nothing to refuse or resend",
      code == 0 and len(e.b.reopens()) == 24 and "reading again" in t and "complete" in t, t)
e = Env()
e.run("--go", when=at(9, 10))
e.b.fill_all_reopens(skip={"SMCI"})
e.b.on_positions = lambda b: b.add_exec("SPY", "BOT", 1, f"tt-cand1-{len(b.execs)}", 10, "20261116 09:31:00 US/Eastern")
code, t = e.run("--go", when=at(9, 31, 5))
check("fills arriving during all 3 reads: refuse, nothing sent", code == 1 and len(e.b.reopens()) == 24
      and "executions changed during each of 3 reads" in t, t)

# ------------------------------------------------------------------ F5: the HOLD file and the reminder
e = Env(held=False, hold=False)
code, t = e.run("--hold", "--go", when=at(6, 0))
check("--hold --go without runs/live/HOLD refuses", code == 1 and not e.held() and "HOLD" in t and "does not exist" in t, t)
R.HOLD_FILE.write_text("", encoding="utf-8")
code, t = e.run("--hold", "--go", when=at(6, 0))
check("--hold --go's RESULT line reminds the operator to remove HOLD before 09:00", code == 0 and R.REMIND in result(t), t)
code, t = e.run("--go", when=at(9, 10))
check("--go's RESULT line (success) carries the reminder", code == 0 and R.REMIND in result(t), t)
e = Env(hold=False)
code, t = e.run("--go", when=at(8, 50))
check("--go's RESULT line (refusal: no HOLD before 09:00) carries it too", code == 1 and R.REMIND in result(t), t)
code, t = e.run("--orders", when=at(9, 45))
check("--orders' RESULT line does not", R.REMIND not in result(t), t)
check("the docstring says the same (HOLD for --hold --go and --go before 09:00; off by 09:00 as soon as held)",
      "runs/live/HOLD must exist for --hold --go, and for --go before 09:00 ET" in R.__doc__
      and "removes runs/live/HOLD by 09:00 ET at the latest, as soon as base-ml shows held (step 1), whether or\n     not "
          "the re-open orders are sent" in R.__doc__ and "from 09:00 ET --go does not need HOLD" in R.__doc__)

# ------------------------------------------------------------------ F6: harmless codes; a fresh orders() read 3 s later
check("classify: 2137 and code-less warningText on PreSubmitted are placed; 404 and 201 are not",
      R.classify({"status": "PreSubmitted", "messages": ["2137: The order is held"]})[0] == "placed"
      and R.classify({"status": "Submitted", "messages": ["Order will be held until the security is available"]})[0] == "placed"
      and R.classify({"status": "PreSubmitted", "messages": ["10167: delayed data"]})[0] == "placed"
      and R.classify({"status": "PreSubmitted", "messages": ["404: held for locate"]})[0] == "failed"
      and R.classify({"status": "Inactive", "messages": ["201: rejected"]})[0] == "failed")
e = Env()
ready(e)
e.b.answer[("AMD", "LMT")] = {"status": None, "messages": ["no status yet, no code"]}
e.b.answer[("INTC", "LMT")] = {"status": "Submitted", "messages": ["2109: outside RTH"]}
def settle(b):
    for o in b.ords:
        if o["symbol"] == "AMD" and o["order_type"] == "LMT":
            o["status"] = "PreSubmitted"            # IBKR's later openOrder
        if o["order_type"] == "TRAIL":
            o["trail_stop"] = 64.7
e.b.on_settle = settle
e.b.bar_rows = [{"date": "20261113 10:00:00 US/Eastern", "high": 68.0, "low": 65.0}]
code, t = e.run("--orders", "--go", when=at(9, 45))
n = e.notes()[-1]
check("after the place()s the script waits 3 s and re-reads orders(): a first answer with no status becomes placed",
      e.slept == [3.0] and code == 0 and "Sent, not working" not in n and "BUY 2 AMD LMT limit 500 GTC [" in n, (e.slept, n))
check("and the trail stop that came only in the later openOrder is the one told (64.70)", "IBKR's stop at 64.70" in n, n)

# ------------------------------------------------------------------ F9: the note's facts
check("the note: cash reset over the weekend, orders cancelled after it, positions re-entered on Monday, orders under new refs",
      "the paper account's cash was reset over the weekend, which removes the positions held at the broker" in n
      and "cancelled after it, so that none could trigger against a position the broker no longer held" in n
      and "re-entered at the broker on Monday" in n and "re-entered with the same terms under new refs" in n
      and "before the reset" not in n, n)

# ------------------------------------------------------------------ G1: a dead loop pid keeps the due-job checks and windows
e = Env()
e.alive = False
e.undo("look-1000")
code, t = e.run("--orders", when=at(10, 15))
check("loop pid dead, look-1000 not yet done at 10:15: refuses (a restarted loop would run it at once)",
      code == 1 and not e.clients and "look-1000 is due" in t and "not running" in t, t)
e = Env()
e.alive = False
code, t = e.run("--orders", when=at(10, 5))
check("loop pid dead at 10:05 (the 10:00 look's window): refuses", code == 1 and not e.clients and "10:00 look" in t, t)

# ------------------------------------------------------------------ G2: --force never waives T-3..T; the 5-minute lead
e = Env()
code, t = e.run("--orders", "--force", when=at(9, 58))
check("--force at 09:58 (2 minutes before 10:00): refuses", code == 1 and not e.clients and "3 minutes before it" in t, t)
code, t = e.run("--orders", "--force", when=at(9, 57))
check("--force at 09:57 (T-3): refuses", code == 1 and not e.clients, t)
code, t = e.run("--orders", "--force", when=at(9, 53))
check("--force at 09:53: refuses (T-3 opens within 5 minutes)", code == 1 and not e.clients and "within 5 minutes" in t, t)
code, t = e.run("--orders", "--force", when=at(9, 51))
check("--force at 09:51 connects (T-3 is 6 minutes away)", code == 0 and e.clients == ["trader-training"], t)
e = Env()
code, t = e.run("--orders", "--force", when=at(10, 5))
check("--force at 10:05 still waives the after-look window", code == 0 and e.clients == ["trader-training"], t)
code, t = e.run("--orders", when=at(9, 53))
check("without --force at 09:53: refuses (09:57 opens within 5 minutes)", code == 1 and "opens at 09:57" in t, t)

# ------------------------------------------------------------------ G6: a LOOKING is stale by its pid or its age
e = Env()
R.LOOKING.write_text("2026-11-16T10:14:00-05:00 999", encoding="utf-8")
code, t = e.run("--orders", when=at(10, 15))
check("LOOKING whose pid is dead (a reboot): stale, said so, ignored", code == 0 and "stale: its pid 999 is gone" in t, t)
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)       # N6: the age rule is only for a pid that is not the loop's
R.LOOKING.write_text("2026-11-16T09:30:00-05:00 5555", encoding="utf-8")
code, t = e.run("--orders", when=at(10, 15))
check("LOOKING 45 minutes old with a live pid that is not the loop's (a reused pid): stale, ignored",
      code == 0 and "written 45 minutes ago" in t, t)
R.PID_ALIVE = lambda pid: e.alive and int(pid) == 4242
R.LOOKING.write_text("2026-11-16T10:10:00-05:00 4242", encoding="utf-8")
code, t = e.run("--orders", "--force", when=at(10, 15))
check("LOOKING 5 minutes old with a live pid: a look is running; refuses even with --force", code == 1 and "a look is running" in t, t)
e = Env()
e.alive = False
R.PID_ALIVE = lambda pid: int(pid) == 5555
R.LOOKING.write_text("2026-11-16T10:13:00-05:00 5555", encoding="utf-8")
code, t = e.run("--orders", when=at(10, 15))
check("loop dead, but a LOOKING of a live process (a dry-run look elsewhere): refuses", code == 1 and "pid 5555 alive" in t, t)

# ------------------------------------------------------------------ G3: a record before each send; a killed run
e = Env()
ready(e)
e.b.interrupt[("IEF", "STP")] = "after"                  # IBKR has it; the run dies before paper.place writes its row
code, t = e.run("--orders", "--go", when=at(9, 45))
ief_ref = next(o["ref"] for o in e.b.ords if o["symbol"] == "IEF" and o["order_type"] == "STP")
sends = [r for r in paper.read(R.LOG) if r.get("sending") and r.get("reenter")]
check("Ctrl-C during place(): closed, a RESULT line, exit 2, held, nothing written", code == 2 and e.held() and not e.notes()
      and result(t).startswith("RESULT: incomplete: interrupted (Ctrl-C) during the place() of BUY 130 IEF STP") and
      "sent, no record: uncertain" in t, t)
check("each send was logged first with the saved order's perm_id and terms", len(sends) == len(e.b.reentries())
      and all(r["terms"]["symbol"] and "time" in r for r in sends)
      and sends[-1]["reenter"] == next(o["perm_id"] for o in SAVED if o["symbol"] == "IEF")
      and not [r for r in paper.read(paper.ORDERS) if r.get("ref") == ief_ref], sends[-1:])
code, t = e.run("--orders", "--go", when=at(9, 46))
ief = [p for p in e.b.reentries() if p["symbol"] == "IEF"]
check("the rerun finds IEF's order at IBKR by its terms, adopts it, and does not send it again",
      len(ief) == 1 and f"adopted {ief_ref}" in t and ief_ref in e.notes()[-1], t)
check("the adoption is logged, and the rest go in", [r for r in paper.read(R.LOG) if r.get("adopt") == ief_ref]
      and len(e.b.reentries()) == 11 and code == 0 and not e.held(), t)

e = Env()
ready(e)
e.b.interrupt[("IEF", "STP")] = "before"                 # killed before IBKR got it
code, t = e.run("--orders", "--go", when=at(9, 45))
e.b.interrupt.clear()
code, t = e.run("--orders", "--go", when=at(9, 46))
check("a send with no record and nothing at IBKR: 1 minute later it waits, IEF is not sent again",
      len([p for p in e.b.reentries() if p["symbol"] == "IEF"]) == 1 and "waiting on the re-entry of BUY 130 IEF STP" in t
      and e.held(), t)
code, t = e.run("--orders", "--go", when=at(9, 48))
check("3 minutes after the send, 2 after the first empty look-up: still waiting",
      len([p for p in e.b.reentries() if p["symbol"] == "IEF"]) == 1 and "waiting on the re-entry" in t, t)
code, t = e.run("--orders", "--go", when=at(9, 51))
check("6 minutes after the send, two empty look-ups 5 minutes apart: sent again, once",
      len([p for p in e.b.reentries() if p["symbol"] == "IEF"]) == 2 and code == 0 and not e.held(), t)
e = Env()
ready(e)
e.b.interrupt[("IEF", "STP")] = "before"
e.run("--orders", "--go", when=at(9, 45))
e.b.interrupt.clear()
code, t = e.run("--orders", when=at(9, 46))
check("a status run (no --go) records no look-up", not [r for r in paper.read(R.LOG) if r.get("lookup_empty")], t)

# ------------------------------------------------------------------ G8: the verified count in every note
e = Env()
e.run("--go", when=at(9, 10))
e.run("--orders", "--go", when=at(9, 14))
n0 = e.notes()[0]
check("the first note before the fills: now 0 of 24 filled, positions 'are being re-entered'",
      "now 0 of 24 names filled in full per IBKR" in n0 and "are being re-entered" in n0 and "were re-entered" not in n0, n0)
e.b.fill_all_reopens(skip={"NVDA"})
e.run("--orders", "--go", when=at(9, 45))
n1 = e.notes()[1]
check("a follow-up note heads with now 23 of 24, 'are being re-entered'", n1.startswith(
    "framework note, follow-up on your open orders after the weekend reset (Mon 16 Nov): now 23 of 24 names filled in full "
    "per IBKR; your positions are being re-entered"), n1)
e.b.fill_all_reopens()
e.run("--orders", "--go", when=at(10, 15))
n2 = e.notes()[2]
check("the last follow-up: now 24 of 24, 'were re-entered'", "now 24 of 24 names filled in full per IBKR; your positions were "
      "re-entered" in n2, n2)

# ------------------------------------------------------------------ G9: the HOLD wording
check("the reminder: remove HOLD by 09:00 at the latest, as soon as base-ml shows held, whether or not the orders are sent",
      "by 09:00 ET at the latest, as soon as base-ml shows held, whether or not the re-open orders are sent" in R.REMIND
      and "from 09:00 ET --go does not need HOLD" in R.REMIND and "are in" not in R.REMIND)
e = Env(hold=False)
code, t = e.run("--go", when=at(9, 0, 30))
check("from 09:00 --go doesn't need HOLD, and its RESULT carries the reminder", code == 0 and R.REMIND in result(t), t)

# ------------------------------------------------------------------ S1: stats([sym]); an error row is not sent
e = Env()
e.b.stats_error["SMCI"] = "200: No security definition has been found for the request"
code, t = e.run("--go", when=at(9, 10))
check("a stats error for SMCI: not raised, SMCI not sent and reported 'shortable shares unknown'; the other 23 go",
      code == 2 and len(e.b.reopens()) == 23 and "SMCI" not in [p["symbol"] for p in e.b.reopens()]
      and "shortable shares unknown (IBKR: 200: No security definition" in t and "stopped by an error" not in t, t)

# ------------------------------------------------------------------ the TWS Gateway Owner's fifth review (16 Nov 2026)
def safe_run(e, *args, when=None):
    """e.run, with a Ctrl-C that escapes the script (a traceback) reported instead of ending the test."""
    try:
        return e.run(*args, when=when)
    except KeyboardInterrupt:
        return None, "KeyboardInterrupt escaped the script (a traceback, no RESULT line)"


def marker(name, text):
    (R.LOOKING.parent / name).write_text(text, encoding="utf-8")


# N1, the reader: every marker file counts, the old plain LOOKING and LOOKING.<pid> alike
e = Env()
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
marker("LOOKING", "2026-11-16T10:14:00-05:00 999")
marker("LOOKING.5555", "2026-11-16T10:13:00-05:00 5555")
code, t = e.run("--orders", "--force", when=at(10, 15))
check("N1: two markers, the plain one stale and LOOKING.5555 live: refuses (any live marker is a running look)",
      code == 1 and not e.clients and "LOOKING.5555 exists" in t and "a look is running" in t, t)
(R.LOOKING.parent / "LOOKING.5555").unlink()
marker("LOOKING.999", "2026-11-16T10:14:00-05:00 999")
code, t = e.run("--orders", when=at(10, 15))
check("N1: two stale markers (plain LOOKING and LOOKING.999, both pid 999 dead): connects, each named with its reason",
      code == 0 and e.clients == ["trader-training"] and t.count("stale: its pid 999 is gone") == 2
      and "LOOKING.999 is stale" in t, t)
R.LOOKING.unlink()
(R.LOOKING.parent / "LOOKING.999").unlink()
e = Env()
marker("LOOKING", "2026-11-16T10:13:00-05:00 4242")             # the running loop at 23a7075 writes the plain file
code, t = e.run("--orders", when=at(10, 15))
check("N1: the old plain LOOKING of the running loop still refuses", code == 1 and not e.clients and "a look is running" in t, t)
R.LOOKING.unlink()
e = Env()
R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 2, "started": "2026-11-16T04:00:00-05:00"}),
                  encoding="utf-8")
marker("LOOKING.4242", "2026-11-16T10:13:00-05:00 4242")
code, t = e.run("--orders", when=at(10, 15))
check("N1: LOOP.json \"looking\": 2 is read, and the loop's LOOKING.4242 refuses", code == 1 and not e.clients
      and "LOOKING.4242 exists" in t and "no \"looking\"" not in t, t)
(R.LOOKING.parent / "LOOKING.4242").unlink()
code, t = e.run("--orders", when=at(10, 15))
check("N1: with LOOKING.4242 gone it connects", code == 0 and e.clients == ["trader-training"], t)
e = Env()
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
marker("LOOKING.5555", "2026-11-16T09:30:00-05:00 5555")
code, t = e.run("--orders", when=at(10, 15))
check("N1/N6: LOOKING.5555 45 minutes old, live but not the loop's pid: stale by its age, ignored",
      code == 0 and "LOOKING.5555 is stale: written 45 minutes ago" in t, t)
(R.LOOKING.parent / "LOOKING.5555").unlink()

# N6: the loop's own marker is live whatever its age (the 15:30 look runs past 40 minutes with no window over it)
e = Env()
marker("LOOKING.4242", "2026-11-16T09:30:00-05:00 4242")
code, t = e.run("--orders", "--force", when=at(10, 15))
check("N6: the loop's pid (LOOP.json), its marker 45 minutes old: a look is running, refuses", code == 1 and not e.clients
      and "a look is running" in t and "the live loop" in t, t)
(R.LOOKING.parent / "LOOKING.4242").unlink()
marker("LOOKING", "2026-11-16T15:30:00-05:00 4242")
code, t = e.run("--orders", "--force", when=at(16, 25))
check("N6: the loop's 15:30 look still marked at 16:25 (55 minutes): refuses", code == 1 and not e.clients
      and "a look is running" in t, t)
R.LOOKING.unlink()
e = Env()
R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 2, "started": "2026-11-16T10:00:00-05:00"}),
                  encoding="utf-8")
marker("LOOKING.4242", "2026-11-16T09:50:00-05:00 4242")         # 25 minutes old, before this loop started
code, t = e.run("--orders", when=at(10, 15))
check("N6: the loop's pid on a marker written before LOOP.json's \"started\" (a previous process, same pid): stale",
      code == 0 and e.clients == ["trader-training"] and "written before the loop (pid 4242) started" in t, t)
(R.LOOKING.parent / "LOOKING.4242").unlink()

# N1: a marker removed between the listing and the read is no marker
e = Env()
gone = R.LOOKING.parent / "LOOKING.6666"
real_files = getattr(R, "marker_files", None)
R.marker_files = lambda: [gone]
try:
    code, t = e.run("--orders", when=at(10, 15))
    vanished = code == 0 and e.clients == ["trader-training"] and "LOOKING.6666" not in t
except Exception as ex:                                          # noqa: BLE001
    vanished, t = False, f"{type(ex).__name__}: {ex}"
finally:
    if real_files:
        R.marker_files = real_files
    else:
        del R.marker_files
check("N1: a marker file that vanishes between the listing and the read is no marker, and nothing raises", vanished, t)

# N3: an adopted TRAIL is told with its stop as sent and IBKR's own stop
e = Env()
ready(e)
e.b.bar_rows = [{"date": "20261113 10:00:00 US/Eastern", "high": 68.0, "low": 65.0}]
e.b.trail_stop = 64.75
e.b.interrupt[("XLE", "TRAIL")] = "after"                        # IBKR has it; no orders.jsonl row
code, t = e.run("--orders", "--go", when=at(9, 45))
e.b.interrupt.clear()
code, t = e.run("--orders", "--go", when=at(9, 46))
xle_ref = next(o["ref"] for o in e.b.ords if o["symbol"] == "XLE" and o["order_type"] == "TRAIL")
n = e.notes()[-1] if e.notes() else ""
xle_said = next((x for x in n.split("; ") if xle_ref in x), "")
check("N3: the adopted XLE trail is told with IBKR's stop and the stop it was sent with, not as 'could not be restored'",
      f"adopted {xle_ref}" in t and "trail 5%" in xle_said
      and "with IBKR's stop at 64.75 (sent as 64.60: 5% below XLE's highest regular-hours trade (68.00)" in xle_said
      and "could not be restored" not in n, (xle_said, n))

# N8: a send with no record that is waiting is told as unconfirmed and labelled so in RESULT
e = Env()
ready(e)
e.b.interrupt[("IEF", "STP")] = "before"                         # killed before IBKR got it
e.run("--orders", "--go", when=at(9, 45))
e.b.interrupt.clear()
code, t = e.run("--orders", "--go", when=at(9, 46))
n = e.notes()[-1] if e.notes() else ""
check("N8: the waiting IEF send is in the note as interrupted, not known whether IBKR received it, with no ref",
      len(e.notes()) == 1 and "Sending interrupted, not known whether IBKR received it (its ref was not recorded): "
      "BUY 130 IEF STP" in n and "Sent, unconfirmed" not in n, n)
check("N8: and RESULT's 'still missing' gives its waiting, not 'waiting for its re-open'", code == 2
      and "BUY 130 IEF STP" in t and "waiting for its re-open" not in t.split("still missing")[-1]
      and "waiting on the re-entry of BUY 130 IEF STP" in t.split("still missing")[-1], t)
code, t = e.run("--orders", "--go", when=at(9, 48))
check("N8: a rerun while it still waits does not tell it again", len(e.notes()) == 1, e.notes())
code, t = e.run("--orders", "--go", when=at(9, 51))
check("N8: once it is sent again and placed, the note says so with its ref, and base-ml is released",
      code == 0 and len(e.notes()) == 2 and "re-entered with the same terms" in e.notes()[-1]
      and "BUY 130 IEF STP" in e.notes()[-1] and not e.held(), (t, e.notes()[-1:]))

# N4: Ctrl-C after the placements (here in the 3-second wait): closed, RESULT, exit 2, no traceback
e = Env()
ready(e)
e.b.on_settle = lambda b: (_ for _ in ()).throw(KeyboardInterrupt())
e.b.closed = 0
code, t = safe_run(e, "--orders", "--go", when=at(9, 45))
check("N4: Ctrl-C in the wait after the placements: closed, exit 2, a RESULT line of what was sent and that no note was written",
      code == 2 and e.b.closed >= 1 and result(t).startswith("RESULT: incomplete: interrupted (Ctrl-C) after the placements; "
                                                             "11 re-entered in this run; no note was written")
      and e.held() and not e.notes() and "Traceback" not in t, t)
e.b.on_settle = None
code, t = e.run("--orders", "--go", when=at(9, 46))
check("N4: the rerun re-sends nothing, writes one note for the 11 and releases", code == 0 and len(e.b.reentries()) == 11
      and len(e.notes()) == 1 and e.notes()[0].count("[tt-base-ml-") == 11 and not e.held(), t)

# N4: Ctrl-C between queuing the note and its 'noted' row: the rerun does not queue it again
e = Env()
ready(e)
real_add = paper.add_notes
def add_then_stop(sleeve, notes):
    real_add(sleeve, notes)
    raise KeyboardInterrupt
paper.add_notes = add_then_stop
try:
    code, t = safe_run(e, "--orders", "--go", when=at(9, 45))
finally:
    paper.add_notes = real_add
check("N4: Ctrl-C right after the note is queued: exit 2, RESULT says it was recorded and queued, the note is there once",
      code == 2 and "the note was recorded and queued; the rerun completes its record without queuing it again" in result(t)
      and len(e.notes()) == 1 and e.held(), t)
code, t = e.run("--orders", "--go", when=at(9, 46))
check("N4: the rerun finds it in the pending notes and does not queue it again; it records it, marks all 11 and releases",
      code == 0 and len(e.notes()) == 1 and "already in base-ml's pending notes, not queued again" in t and not e.held()
      and len(json.loads(paper.LOOKS.read_text(encoding="utf-8"))["base-ml"]["reported_orders"]) == 11, (t, e.notes()))

# N4: Ctrl-C after the note is recorded, before it is queued: the rerun queues it (nothing lost), once
e = Env()
ready(e)
paper.add_notes = lambda sleeve, notes: (_ for _ in ()).throw(KeyboardInterrupt())
try:
    code, t = safe_run(e, "--orders", "--go", when=at(9, 45))
finally:
    paper.add_notes = real_add
check("N4: Ctrl-C before the note is queued: exit 2, RESULT says recorded and not queued, no note yet",
      code == 2 and "the note was recorded and not queued; the rerun completes its record and queues it" in result(t)
      and not e.notes(), t)
code, t = e.run("--orders", "--go", when=at(9, 46))
check("N4: the rerun queues the recorded note, once, and writes no second note for the same orders",
      code == 0 and len(e.notes()) == 1 and e.notes()[0].count("[tt-base-ml-") == 11
      and "not in base-ml's pending notes, queued now" in t and not e.held(), (t, e.notes()))
e = Env()
ready(e)
paper.add_notes = lambda sleeve, notes: (_ for _ in ()).throw(KeyboardInterrupt())
try:
    safe_run(e, "--orders", "--go", when=at(9, 45))
finally:
    paper.add_notes = real_add
code, t = e.run("--release", "--go", when=at(9, 46))
check("N4: --release --go after such a stop queues the recorded note first, then releases", code == 0 and not e.held()
      and len(e.notes()) == 1 and "queued now" in t, (t, e.notes()))

# N4: Ctrl-C in the re-open's placing loop (client 18)
e = Env()
e.b.interrupt[("SMCI", "MKT")] = "after"
e.b.closed = 0
code, t = safe_run(e, "--go", when=at(9, 10))
sent18 = len(e.b.reopens())
check("N4: Ctrl-C during a re-open place(): closed, exit 2, a RESULT line, no traceback", code == 2 and e.b.closed >= 1
      and result(t).startswith("RESULT: incomplete: interrupted (Ctrl-C)") and "1 sent with no answer (uncertain)" in result(t)
      and f"{24 - sent18} not sent" in result(t), t)
e.b.interrupt.clear()
code, t = e.run("--go", when=at(9, 12))
check("N4: the rerun finds SMCI's order by its ref and sends only the rest: 24 re-opens, SMCI once",
      code == 0 and len(e.b.reopens()) == 24 and len([p for p in e.b.reopens() if p["symbol"] == "SMCI"]) == 1, t)

# N4: Ctrl-C in the reads before any place() (guarded)
e = Env()
e.b.on_positions = lambda b: (_ for _ in ()).throw(KeyboardInterrupt())
e.b.closed = 0
code, t = safe_run(e, "--go", when=at(9, 10))
check("N4: Ctrl-C in the reads before any place(): closed, exit 2, a RESULT line, nothing sent", code == 2
      and e.b.closed >= 1 and result(t).startswith("RESULT: incomplete: interrupted (Ctrl-C)") and not e.b.placed, t)

# ------------------------------------------------------------------ the second author's round (three reviewers, 15 Nov)
# the loop's marker written just before LOOP.json's "started" (the clock set back a little): still a running look
e = Env()
R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 2, "started": "2026-11-16T10:00:05-05:00"}),
                  encoding="utf-8")
marker("LOOKING.4242", "2026-11-16T10:00:03-05:00 4242")
code, t = e.run("--orders", "--force", when=at(10, 5))
check("N6: the loop's marker 2 s before LOOP.json's \"started\" (a clock set back): a look is running, refuses",
      code == 1 and not e.clients and "a look is running" in t, t)
(R.LOOKING.parent / "LOOKING.4242").unlink()

# N8: a send with no record told as interrupted, then judged gone: told so, and the resend's answer beside it
e = Env()
ready(e)
e.b.interrupt[("IEF", "STP")] = "before"
e.run("--orders", "--go", when=at(9, 45))
e.b.interrupt.clear()
e.run("--orders", "--go", when=at(9, 46))
e.b.answer[("IEF", "STP")] = {"status": "Inactive", "messages": ["201: Order rejected - reason: x"]}
code, t = e.run("--orders", "--go", when=at(9, 51))
n = e.notes()[-1] if len(e.notes()) == 2 else ""
check("N8: once judged gone, the send with no record is told as not found at IBKR (once), beside the resend's answer",
      code == 2 and "Sending interrupted, not found at IBKR (its ref was not recorded): BUY 130 IEF STP" in n
      and "no order of these terms at IBKR on two look-ups" in n and "Sent, not working: BUY 130 IEF STP" in n
      and n.count("BUY 130 IEF STP") == 2, (e.notes(), t))
e.b.answer.clear()
code, t = e.run("--orders", "--go", when=at(10, 15))
check("N8: a later run does not tell the gone send again", len(e.notes()) == 3 and "not found at IBKR" not in e.notes()[-1],
      e.notes()[-1:])

# N4: --release --go finishes a recorded note only when client 10 is free (the note and looks.json are a look's files)
e = Env()
ready(e)
paper.add_notes = lambda sleeve, notes: (_ for _ in ()).throw(KeyboardInterrupt())
try:
    safe_run(e, "--orders", "--go", when=at(9, 45))
finally:
    paper.add_notes = real_add
looks0 = paper.LOOKS.read_text(encoding="utf-8")
marker("LOOKING.4242", "2026-11-16T10:00:00-05:00 4242")
code, t = e.run("--release", "--go", when=at(10, 3))
check("N4: --release --go during the loop's look with a recorded note unfinished: refuses, held, nothing written",
      code == 1 and e.held() and not e.notes() and paper.LOOKS.read_text(encoding="utf-8") == looks0
      and result(t).startswith("RESULT: refused, base-ml stays held: a note recorded by a stopped --orders --go"), t)
(R.LOOKING.parent / "LOOKING.4242").unlink()
paper.add_notes = lambda sleeve, notes: (_ for _ in ()).throw(KeyboardInterrupt())
try:
    code, t = safe_run(e, "--release", "--go", when=at(10, 20))
finally:
    paper.add_notes = real_add
check("N4: a Ctrl-C while --release --go finishes the note: a RESULT line, exit 2, held", code == 2 and e.held()
      and result(t).startswith("RESULT: incomplete: interrupted (Ctrl-C) while finishing a recorded note"), t)
code, t = e.run("--release", "--go", when=at(10, 20))
check("N4: outside the look it finishes the note (queued once) and releases", code == 0 and not e.held()
      and len(e.notes()) == 1, (t, e.notes()))
e = Env()
marker("LOOKING.4242", "2026-11-16T10:00:00-05:00 4242")
code, t = e.run("--release", "--go", when=at(10, 3))
check("--release --go with no recorded note still releases during a look (it touches only sleeves.json)",
      code == 0 and not e.held(), t)
(R.LOOKING.parent / "LOOKING.4242").unlink()

# N4: the STOPPED and RESULT lines name an earlier run's note this run finished
for how in ("place", "settle"):
    e = Env()
    ready(e)
    e.b.answer[("IEF", "STP")] = {"status": "Inactive", "messages": ["201: Order rejected - reason: x"]}
    paper.add_notes = lambda sleeve, notes: (_ for _ in ()).throw(KeyboardInterrupt())
    try:
        safe_run(e, "--orders", "--go", when=at(9, 45))
    finally:
        paper.add_notes = real_add
    e.b.answer.clear()
    if how == "place":
        e.b.interrupt[("IEF", "STP")] = "before"
    else:
        e.b.on_settle = lambda b: (_ for _ in ()).throw(KeyboardInterrupt())
    code, t = safe_run(e, "--orders", "--go", when=at(9, 46))
    e.b.on_settle = None
    said = "by an earlier stopped run was finished at this run's start (queued now)"
    check(f"N4: Ctrl-C in the {how} after this run finished an earlier run's note: the lines say so, not 'nothing written'",
          code == 2 and len(e.notes()) == 1 and said in result(t) and "nothing written" not in t
          and ("this run's own note: no note was written" in result(t) if how == "settle" else True), t)

# N3: a TRAIL first told on a rerun (its run stopped in the wait after the placements) gets its stop's reason
e = Env()
ready(e)
e.b.bar_rows = [{"date": "20261113 10:00:00 US/Eastern", "high": 68.0, "low": 65.0}]
e.b.trail_stop = 64.75
e.b.on_settle = lambda b: (_ for _ in ()).throw(KeyboardInterrupt())
safe_run(e, "--orders", "--go", when=at(9, 45))
e.b.on_settle = None
code, t = e.run("--orders", "--go", when=at(9, 46))
xle_ref = next(o["ref"] for o in e.b.ords if o["symbol"] == "XLE" and o["order_type"] == "TRAIL")
n = e.notes()[-1] if e.notes() else ""
xle_said = next((x for x in n.split("; ") if xle_ref in x), "")
check("N3: a TRAIL told first on a rerun gives the reason from its 'sending' row, not 'as sent earlier'",
      "with IBKR's stop at 64.75 (sent as 64.60: 5% below XLE's highest regular-hours trade (68.00)" in xle_said
      and "as sent earlier" not in n, (xle_said, n))

# N4: the re-open's Ctrl-C RESULT counts every name not sent, a no-borrow one included
e = Env()
e.b.short["SMCI"] = 0                                            # SMCI (18th) not sent: no borrow
e.b.interrupt[("STX", "MKT")] = "before"                         # Ctrl-C during the 21st name's send
code, t = safe_run(e, "--go", when=at(9, 10))
check("N4: Ctrl-C during a re-open send after a no-borrow name: RESULT counts SMCI and the rest as not sent",
      code == 2 and f"{24 - len(e.b.reopens())} not sent" in result(t) and 24 - len(e.b.reopens()) == 4
      and "'SMCI': 'not sent: shortable_shares 0 < 191'" in t, t)

# ------------------------------------------------------------------ the TWS Gateway Owner's sixth review (e3c8b66)
# F1: done.json is read before the markers. live.py writes LOOKING.<pid> and then marks the look done; a fake loop that
# does both between the script's two reads must still be seen.
def loop_starts_look(job, at_text, pid=4242, day="2026-11-16"):
    """What live.py does at a look's start: begin_marker, then mark_done."""
    marker(f"LOOKING.{pid}", f"{at_text} {pid}")
    done = json.loads(R.DONE_FILE.read_text(encoding="utf-8"))
    R.DONE_FILE.write_text(json.dumps(done + [[day, job]]), encoding="utf-8")


def racing(mod, first, job="look-1030", at_text="2026-11-16T10:45:00-05:00"):
    """Wrap mod.<first> (the first of client10_block's two reads) so the fake loop starts its look right after it."""
    real = getattr(mod, first)
    def wrapped(now):
        got = real(now)
        loop_starts_look(job, at_text)
        return got
    setattr(mod, first, wrapped)
    return real


e = Env()
R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 2, "started": "2026-11-16T04:00:00-05:00"}),
                  encoding="utf-8")
e.undo("look-1030")                                              # look-1030 starts late, at 10:45 (after an overrunning 10:00)
real = racing(R, "pending_jobs")
try:
    blk, _ = R.client10_block(at(10, 45), False, start=True)
finally:
    R.pending_jobs = real
check("F1: the loop writes its marker and marks look-1030 done between the two reads: refused (the marker, read second)",
      blk is not None and "LOOKING.4242 exists" in blk and "a look is running" in blk, blk)
(R.LOOKING.parent / "LOOKING.4242").unlink()
check("F1: and with the look over (marker gone, done) client 10 is free at 10:45", R.client10_block(at(10, 45), False)[0] is None,
      R.client10_block(at(10, 45), False))
e.undo("look-1030")
real_mf = R.marker_files
def starts_then_lists():
    loop_starts_look("look-1030", "2026-11-16T10:45:00-05:00")
    return real_mf()
R.marker_files = starts_then_lists
try:
    blk, _ = R.client10_block(at(10, 45), False)                 # the look starts just before the markers are listed
finally:
    R.marker_files = real_mf
check("F1: the look starts after done.json was read, as the markers are listed: refused", blk is not None and "a look is running"
      in blk, blk)
(R.LOOKING.parent / "LOOKING.4242").unlink()

# F1, the old order: 2382e0d's client10_block, loaded from a temp folder outside the repo, with the same fake loop acting
# between its two reads (the markers first, then done.json): it sees neither.
import importlib.util                                            # noqa: E402
import subprocess                                                # noqa: E402
try:
    src = subprocess.run(["git", "show", "2382e0d:reopen_week6.py"], cwd=Path(__file__).resolve().parent,
                         capture_output=True, text=True, encoding="utf-8", timeout=60, check=True).stdout
except Exception as ex:                                          # noqa: BLE001
    src = None
    print(f"skip F1 old order: git show 2382e0d:reopen_week6.py failed ({type(ex).__name__}: {ex})")
if src:
    old_dir = Path(tempfile.mkdtemp(prefix="reopen6_old_"))
    atexit.register(shutil.rmtree, old_dir, True)
    (old_dir / "reopen_week6_2382e0d.py").write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("reopen_week6_2382e0d", old_dir / "reopen_week6_2382e0d.py")
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    e = Env()
    R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 2, "started": "2026-11-16T04:00:00-05:00"}),
                      encoding="utf-8")
    OLD.LOOP, OLD.LOOKING, OLD.DONE_FILE, OLD.PID_ALIVE = R.LOOP, R.LOOKING, R.DONE_FILE, R.PID_ALIVE
    e.undo("look-1030")
    blk_plain, _ = OLD.client10_block(at(10, 45), False)
    racing(OLD, "looking_state")
    blk, _ = OLD.client10_block(at(10, 45), False, start=True)
    seen_after = R.client10_block(at(10, 45), False)[0]
    check("F1: at 2382e0d the same fake loop between the reads (markers, then done.json) is missed: client 10 looks free",
          "look-1030 is due" in (blk_plain or "") and blk is None and "a look is running" in (seen_after or ""),
          (blk_plain, blk, seen_after))
    (R.LOOKING.parent / "LOOKING.4242").unlink()

# F2: a pid is the loop's only while its process was created no later than LOOP.json's "started" (+ 2 minutes)
def created(table):
    R.PID_CREATED = lambda pid: table.get(int(pid))


def loop2(started="2026-11-16T04:00:00-05:00"):
    R.LOOP.write_text(json.dumps({"pid": 4242, "code": "test", "looking": 2, "started": started}), encoding="utf-8")


e = Env()
loop2()
created({4242: at(3, 59)})                                      # the loop's process, created just before it wrote "started"
marker("LOOKING.4242", "2026-11-16T09:30:00-05:00 4242")         # 45 minutes old
code, t = e.run("--orders", "--force", when=at(10, 15))
check("F2: the loop's pid alive and created before \"started\": the live loop, its 45-minute marker refuses",
      R.loop_state()[0] == "alive" and code == 1 and not e.clients and "a look is running" in t and ", the live loop" in t, t)
created({4242: at(4, 10)})                                      # 10 minutes after "started": another process now
code, t = e.run("--orders", "--force", when=at(10, 15))
check("F2: the loop's pid created 10 minutes after \"started\": the loop is dead, said why; its marker judged by age (stale)",
      R.loop_state()[0] == "dead" and code == 0 and e.clients == ["trader-training"]
      and "not running (pid 4242 now belongs to a process created at 04:10:00, after the loop's start at 04:00:00)" in t
      and "LOOKING.4242 is stale: written 45 minutes ago" in t and ", the live loop" not in t, t)
marker("LOOKING.4242", "2026-11-16T10:13:00-05:00 4242")         # 2 minutes old, written by the new process: fails safe
e.clients.clear()
code, t = e.run("--orders", "--force", when=at(10, 15))
check("F2: and a fresh marker of that pid (written after its process was created) still refuses, by the general rules",
      code == 1 and not e.clients and "a look is running" in t and ", the live loop" not in t, t)
(R.LOOKING.parent / "LOOKING.4242").unlink()

e = Env()
loop2()
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
created({4242: at(3, 59), 5555: at(10, 10)})
marker("LOOKING.5555", "2026-11-16T10:05:00-05:00 5555")         # its pid's process created 5 minutes after it
code, t = e.run("--orders", when=at(10, 15))
check("F2: a marker whose pid is alive but was created 5 minutes after the marker: stale, the pid was reused",
      code == 0 and e.clients == ["trader-training"] and "LOOKING.5555 is stale: its pid 5555 now belongs to a process "
      "created at 10:10:00, after the marker was written" in t and "the pid was reused" in t, t)
created({4242: at(3, 59), 5555: at(10, 6)})                     # 1 minute after: inside the margin
code, t = e.run("--orders", when=at(10, 15))
check("F2: created 1 minute after the marker (inside the 2-minute margin): unchanged, a look is running",
      code == 1 and "LOOKING.5555 exists" in t and "a look is running" in t and "reused" not in t, t)
(R.LOOKING.parent / "LOOKING.5555").unlink()

# F2, the zones (the TWS Gateway Owner, 15 Nov): GetProcessTimes is UTC, "started" carries -05:00 and a marker its own ISO
# time. The same instants must give the same verdict whichever zone each is written in, just inside and just outside the
# 2-minute margin; a naive local-against-UTC compare would be 5 hours off.
from datetime import timedelta as _td, timezone as _tz      # noqa: E402
t_started = datetime.fromisoformat("2026-11-16T04:00:00-05:00")
verdicts = []
for started_txt in ("2026-11-16T04:00:00-05:00", "2026-11-16T09:00:00+00:00"):      # one instant, written in ET and in UTC
    for zone in (_tz.utc, R.ET):
        for secs, want in ((119, "alive"), (121, "dead")):
            e = Env()
            loop2(started_txt)
            created({4242: (t_started + _td(seconds=secs)).astimezone(zone)})
            verdicts.append((started_txt[-6:], str(zone), secs, R.loop_state()[0], want))
check("F2 zones: the loop's pid created 1:59 after \"started\" is the loop and 2:01 after is not, with \"started\" and the "
      "creation time each written in ET or in UTC", len(verdicts) == 8 and all(got == want for *_, got, want in verdicts),
      str(verdicts))
t_mark = datetime.fromisoformat("2026-11-16T10:05:00-05:00")
verdicts = []
for mark_txt in ("2026-11-16T10:05:00-05:00", "2026-11-16T15:05:00+00:00"):          # one instant, written in ET and in UTC
    for zone in (_tz.utc, R.ET):
        for secs, want in ((119, "running"), (121, "stale")):
            e = Env()
            loop2()
            R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
            created({4242: at(3, 59), 5555: (t_mark + _td(seconds=secs)).astimezone(zone)})
            marker("LOOKING.5555", f"{mark_txt} 5555")
            state = R.looking_state(at(10, 15))
            verdicts.append((mark_txt[-6:], str(zone), secs, state[0] if state else None, want,
                             "reused" in str(state[1] if state else "")))
            (R.LOOKING.parent / "LOOKING.5555").unlink()
check("F2 zones: a marker's pid created 1:59 after the marker's time is a running look and 2:01 after is stale as reused, "
      "with the marker's time and the creation time each written in ET or in UTC",
      len(verdicts) == 8 and all(got == want and reused == (want == "stale") for *_, got, want, reused in verdicts), str(verdicts))
made_here = R.pid_created(os.getpid())
check("F2 zones: the real creation time is timezone-aware (UTC ticks turned into ET), so it compares with any aware time",
      made_here is not None and made_here.utcoffset() is not None
      and abs((made_here - datetime.now(_tz.utc)).total_seconds()) < 3600, str(made_here))

e = Env()
loop2()
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
created({})                                                     # PID_CREATED None: as at 2382e0d
marker("LOOKING.4242", "2026-11-16T09:30:00-05:00 4242")
code, t = e.run("--orders", "--force", when=at(10, 15))
ok1 = R.loop_state()[0] == "alive" and code == 1 and ", the live loop" in t
(R.LOOKING.parent / "LOOKING.4242").unlink()
marker("LOOKING.5555", "2026-11-16T09:30:00-05:00 5555")
code, t2 = e.run("--orders", when=at(10, 15))
ok2 = code == 0 and "LOOKING.5555 is stale: written 45 minutes ago" in t2
(R.LOOKING.parent / "LOOKING.5555").unlink()
marker("LOOKING.5555", "2026-11-16T10:05:00-05:00 5555")
code, t3 = e.run("--orders", when=at(10, 15))
ok3 = code == 1 and "LOOKING.5555 exists" in t3
(R.LOOKING.parent / "LOOKING.5555").unlink()
check("F2: creation time unknown (None): the loop alive by its pid, its marker live at any age, others by age, as at 2382e0d",
      ok1 and ok2 and ok3 and "the pid was reused" not in t + t2 + t3 and "now belongs" not in t + t2 + t3
      # the verdicts are 2382e0d's; since the wording pass each running marker's text adds that a reuse can't be ruled out
      and "a reused pid can't be ruled out" in t and "a reused pid can't be ruled out" in t3, (t, t2, t3))

# F2, the reviewer's scenario: the loop is killed hard during its 10:00 look (LOOKING.4242 left behind), Windows gives pid
# 4242 to a new process at 10:08, the loop is not restarted.
e = Env()
loop2()
ready(e)
marker("LOOKING.4242", "2026-11-16T10:00:05-05:00 4242")
created({})
code, t = e.run("--orders", "--go", when=at(10, 20))
check("F2 scenario, creation time unknown (as at 2382e0d): refused as a look of the live loop, nothing re-entered",
      code == 1 and "a look is running" in t and ", the live loop" in t and not e.b.reentries(), t)
created({4242: at(10, 8)})
blk, note = R.client10_block(at(10, 20), False, start=True)
check("F2 scenario, the reuse seen: client10_block does not refuse (a dead loop alone is no refusal), and says why",
      blk is None and "not running (pid 4242 now belongs to a process created at 10:08:00" in note
      and "LOOKING.4242 is stale: its pid 4242 now belongs to a process created at 10:08:00" in note, (blk, note))
code, t = e.run("--orders", "--go", when=at(10, 20))
check("F2 scenario, the reuse seen: --orders --go re-enters the 11 on client 10", code == 0
      and len(e.b.reentries()) == 11 and "a look is running" not in t, t)
e = Env()
loop2()
created({4242: at(10, 8)})
marker("LOOKING.4242", "2026-11-16T10:00:05-05:00 4242")
e.undo("look-1030")
code, t = e.run("--orders", when=at(10, 35))
check("F2 scenario: with the loop dead its due jobs still refuse (look-1030 not done at 10:35)",
      code == 1 and not e.clients and "look-1030 is due" in t and "now belongs to a process created at 10:08:00" in t, t)
(R.LOOKING.parent / "LOOKING.4242").unlink()

# The second author, review of the two fixes: (1) a creation time after now (the clock set back since that process
# started) is not taken as a reused pid; (2) a half-written done.json refuses cleanly instead of raising.
e = Env()
loop2()
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
created({4242: at(3, 59), 5555: at(10, 0)})                     # a hand-started look, created at 10:00:00 ...
marker("LOOKING.5555", "2026-11-16T09:55:01-05:00 5555")         # ... then the clock set back 5 minutes before its marker
try:
    blk, note = R.client10_block(at(9, 56), True)
except Exception as ex:                                          # noqa: BLE001
    blk, note = None, f"raised {type(ex).__name__}: {ex}"
check("review 1: a marker whose process was created after now (the clock set back after it started): not taken as reused, "
      "a look is running", blk is not None and "LOOKING.5555 exists" in blk and "a look is running" in blk
      and "reused" not in (blk + str(note)), (blk, note))
created({4242: at(3, 59), 5555: at(9, 50)})                     # created before now and 5 minutes after the marker: reused
marker("LOOKING.5555", "2026-11-16T09:45:00-05:00 5555")
blk, note = R.client10_block(at(9, 56), True)
check("review 1: created before now and more than 2 minutes after the marker: still stale, the pid was reused",
      blk is None and "LOOKING.5555 is stale: its pid 5555 now belongs to a process created at 09:50:00" in note, (blk, note))
(R.LOOKING.parent / "LOOKING.5555").unlink()

e = Env()
loop2()
R.DONE_FILE.write_text('[["2026-11-16", "look-0', encoding="utf-8")   # mark_done caught mid-write
try:
    blk, note = R.client10_block(at(10, 45), False)
except Exception as ex:                                          # noqa: BLE001
    blk, note = f"raised {type(ex).__name__}: {ex}", None
check("review 2: done.json half-written: client10_block refuses (can't be read), does not raise",
      blk is not None and "done.json can't be read" in blk and "raised" not in blk, (blk, note))
marker("LOOKING.4242", "2026-11-16T10:45:00-05:00 4242")
try:
    blk, note = R.client10_block(at(10, 45), False)
except Exception as ex:                                          # noqa: BLE001
    blk, note = f"raised {type(ex).__name__}: {ex}", None
check("review 2: done.json half-written and a look's marker up: the running look is named first",
      blk is not None and "LOOKING.4242 exists" in blk and "a look is running" in blk, (blk, note))
(R.LOOKING.parent / "LOOKING.4242").unlink()
e = Env()
loop2()
ready(e)
def half_written_after_first(b):
    """10:20; once the first re-entry is placed, done.json is caught mid-write by the next check."""
    if b.reentries():
        R.DONE_FILE.write_text('[["2026-11-16", "look-0', encoding="utf-8")
    return at(10, 20)
code, t = e.run("--orders", "--go", when=half_written_after_first)
check("review 2: done.json half-written before the second place(): the run stops cleanly (no 'stopped by an error'), one "
      "re-entered", code == 2 and len(e.b.reentries()) == 1 and "stopped by an error" not in t
      and "RESULT: incomplete:" in t and "done.json can't be read" in result(t), (code, len(e.b.reentries()), t[-1200:]))

# The TWS Gateway Owner's three wording points on c1fa13d (15 Nov): none changes the logic
e = Env()
loop2()
R.DONE_FILE.write_text('[["2026-11-16", "look-0', encoding="utf-8")
blk, note = R.client10_block(at(10, 45), False)
check("wording 1: an unreadable done.json: the refusal gives the error, says the loop may be writing it, and where to look "
      "if it says so again", blk is not None and "done.json can't be read (JSONDecodeError: " in blk
      and "The loop may be writing it: rerun in a moment" in blk and "look at" in blk and str(R.DONE_FILE) in blk
      and "--force does not override this" in blk, blk)
jobs = [["2026-11-16", f"look-{h:02d}{m:02d}"] for h, m in R.LOOK_TIMES if (h, m) <= (10, 30)] + [["2026-11-16", "refresh"]]
R.DONE_FILE.write_bytes(b"\xef\xbb\xbf" + json.dumps(jobs).encode("utf-8"))      # saved with a BOM (Windows PowerShell 5.1)
blk, note = R.client10_block(at(10, 45), False)
check("wording 1: a done.json saved with a BOM is read: no refusal for it, and its done jobs are not due",
      R.pending_jobs(at(10, 45)) == [] and (blk is None or "done.json" not in blk), (blk, note))
R.PID_ALIVE = lambda pid: int(pid) in (4242, 5555)
marker("LOOKING.5555", "2026-11-16T10:40:00-05:00 5555")
created({4242: at(3, 59)})                                       # 5555's creation time can't be read (not ours to query)
blk, note = R.client10_block(at(10, 45), False)
check("wording 2: a live marker whose pid's creation time can't be read: refused as before, and the message says a reused "
      "pid can't be ruled out and to restart the loop if no look is on",
      blk is not None and "a look is running" in blk and "The creation time of pid 5555 could not be read" in blk
      and "a reused pid can't be ruled out" in blk and "restart the loop" in blk, blk)
created({4242: at(3, 59), 5555: at(10, 39)})                     # read, and before the marker: the plain message
blk, note = R.client10_block(at(10, 45), False)
check("wording 2: with the creation time read (before the marker) the message has no such sentence",
      blk is not None and "a look is running" in blk and "could not be read" not in blk
      and note == "the live loop is running (pid 4242)", (blk, note))
(R.LOOKING.parent / "LOOKING.5555").unlink()
created({})                                                      # the loop's own creation time can't be read
blk, note = R.client10_block(at(10, 45), False)
check("wording 2: the loop alive by its pid but its creation time unread: the note says a reused pid can't be ruled out",
      note is not None and note.startswith("the live loop is running (pid 4242; its process's creation time could not be read, "
                                           "so a reused pid can't be ruled out)"), (blk, note))
doc = R.__doc__ or ""
helptext = subprocess.run([sys.executable, str(Path(R.__file__)), "--help"], capture_output=True, text=True).stdout
check("wording 3: the module's text and --force's help name the creation-time rule and the done.json refusal",
      "the pid was reused" in doc and "can't be read (the loop rewrites it in place)" in doc
      and "done.json" in " ".join(helptext.split()) and "3 minutes before a look" in " ".join(helptext.split()), helptext[-600:])

# F2: the real call, on this test's own process
import os                                                        # noqa: E402
mine = R.pid_created(os.getpid())
wall = datetime.now(R.ET)
check("F2: pid_created(this process) is a time within the last hour and not in the future",
      mine is not None and mine.tzinfo is not None and wall - timedelta(hours=1) < mine <= wall, (mine, wall))
check("F2: pid_created of a bad pid is None and does not raise", R.pid_created(0) is None and R.pid_created("x") is None
      and R.pid_created(None) is None)

print("ALL OK" if all(OK) else f"{OK.count(False)} FAILED")
sys.exit(0 if all(OK) else 1)
