"""The sleeve's own exposure limit (own_limit.py; Clayton, Wed 18 Nov 2026: "Yes, text only"). Offline: no IB connection,
no Tinker; nothing under runs/ is read or written (paper.sleeves and paper.close_only_refs are replaced in-process).

- the texts: the term, the limit line's three forms (exactly at the limit is not over), the positions sentence;
- the system text: the term in place of NO_GROSS_RULE only with max_gross null; a sleeve without the term reads the old text;
- own_limit_of / own_limit_bad;
- the state: the line straight after the sleeve-value line and before the account line; the account sentence at the end
  of the account line, after the margin warning, only when the line carries IBKR's figures; text A gets the line alone;
  a sleeve without the term reads the state as before;
- the order check: the walk, stops alone against the position after the list's other orders, close_only orders, this
  sleeve's resting orders (others' and close_only ones left out), the resting orders not read, the over form, a value at
  or below zero, the line last in the answer, a failure costing the line only, and context()'s fields;
- PaperWorld and live.py's wiring.

    python own_limit_test.py
"""
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import checks  # noqa: E402
import live  # noqa: E402
import own_limit as O  # noqa: E402
import paper  # noqa: E402
import trader  # noqa: E402

DEFAULTS = (O.STATE_RESTING, trader.FILL_POSITION)   # both on from the 10:30 look of 18 Nov; the sections below set
O.STATE_RESTING = False                               # the switch they test and restore it at the end

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


# ------------------------------------------------------------------------------------------------ the texts
SPEC_TERM = ("Your exposure limit: gross exposure may not exceed 3x your sleeve's value. Count your positions and every "
             "order, new or resting, that would open or add to one; a hedge adds to gross too. Trade on your own sleeve's "
             "value: the paper account holds other sleeves too, and its margin, excess liquidity and buying power are not "
             "room for your orders. No software stops an order that breaks this limit; keeping to it is your job at every "
             "look. If your gross exposure is above 3x, your orders at that look must bring it to 3x or less. IBKR may "
             "still refuse an order, and that is reported at your next look.")
check("the term is SPEC.md's, word for word", O.term(3.0) == SPEC_TERM, O.term(3.0))
check("the term follows the limit", "above 2.5x" in O.term(2.5) and "exceed 2.5x" in O.term(2.5) and "3x" not in O.term(2.5))
check("the account sentence is SPEC.md's", O.ACCOUNT_SENTENCE == (
    " The dollar figures in this line are the whole account's, for all the sleeves together: none of them is room for your "
    "orders. Your room is your own exposure limit, in the line above."))

under = O.limit_line(100000.0, 1.5)
check("under the limit (the Researcher's form of 18 Nov: no word on resting orders)",
      under == ("Your exposure limit: 3x your sleeve's value = $300,000 of gross exposure. You hold $150,000 (1.50x). Room "
                "left for orders that open or add to positions: $150,000."), under)
check("exactly at the limit is not over", O.limit_line(100000.0, 3.0).startswith("Your exposure limit:")
      and O.limit_line(100000.0, 3.0).endswith("positions: $0."))
over = O.limit_line(100000.0, 5.0)
check("over the limit", over == ("Over your exposure limit: your limit is 3x your sleeve's value = $300,000 of gross "
                                 "exposure, and you hold $500,000 (5.00x), which is $200,000 over. Your orders at this look "
                                 "must bring your gross exposure to $300,000 or less: close or shrink positions by at least "
                                 "$200,000 more than you open or add. A stop or trailing order does not bring it down until "
                                 "it fills."), over)
check("value at or below zero", O.limit_line(0.0, 0.0) == O.limit_line(-5.0, -3.0) == (
    "Over your exposure limit: your sleeve's value is not positive, so you have no room. Place only orders that close or "
    "shrink positions."))

POS = [{"symbol": "QQQ", "quantity": -200, "close": 650.0}, {"symbol": "XLV", "quantity": 400, "close": 185.0},
       {"symbol": "AMD", "quantity": -100, "close": 250.0}, {"symbol": "ZZ", "quantity": 0, "close": 10.0}]
ps = O.positions_sentence(POS, 150000.0)
check("positions by size with the running total", ps == (
    "Your positions by size in dollars at the price shown, largest first, with the running total in brackets: QQQ short "
    "$130,000 [$130,000]; XLV long $74,000 [$204,000]. The running total first reaches the $150,000 you are over at the 2nd "
    "position, XLV. The rest: AMD short $25,000."), ps)
check("all of them less than the overage", O.positions_sentence(POS, 1e6).endswith(
    "AMD short $25,000 [$229,000]. All of them together are less than the $1,000,000 you are over."))
check("no positions, no sentence", O.positions_sentence([], 5.0) == "")
check("ordinals", [O.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 111)] ==
      ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "111th"])
st_over = {"value": 100000.0, "gross": 2.29, "positions": POS}
check("state_line: the over form carries the positions sentence",
      O.state_line(st_over, 2.0) == O.limit_line(100000.0, 2.29, 2.0) + " " + O.positions_sentence(POS, 29000.0))
check("state_line: under, no positions sentence", O.state_line(st_over, 3.0) == O.limit_line(100000.0, 2.29, 3.0, None))

# ------------------------------------------------------------------------------------------------ the system text
s_on = trader.live_system(90000, "2026-11-18", ml=True, ml_tool="ML4", max_gross=None, own_limit=3.0)
s_off = trader.live_system(90000, "2026-11-18", ml=True, ml_tool="ML4", max_gross=None)
check("system: the term in place of the no-limit sentence", SPEC_TERM in s_on and trader.NO_GROSS_RULE not in s_on
      and s_on.replace(SPEC_TERM, trader.NO_GROSS_RULE) == s_off)
check("system: no term, the text as before", trader.NO_GROSS_RULE in s_off and SPEC_TERM not in s_off)
check("system: a software limit keeps its own sentence", trader.live_system(90000, "2026-11-18", max_gross=2.0, own_limit=3.0)
      == trader.live_system(90000, "2026-11-18", max_gross=2.0))
check("system: text A's base too", SPEC_TERM in trader.live_system(90000, "2026-11-18", max_gross=None, own_limit=3.0))

# ------------------------------------------------------------------------------------------------ the term's setting
cases = [({"max_gross": None, "own_limit": 3.0}, 3.0, None), ({"max_gross": None, "own_limit": 3}, 3.0, None),
         ({"max_gross": None}, None, None), ({"max_gross": None, "own_limit": None}, None, None),
         ({"max_gross": None, "own_limit": 0}, None, "finite number above 0"),
         ({"max_gross": None, "own_limit": "3"}, None, "finite number above 0"),
         ({"max_gross": None, "own_limit": True}, None, "finite number above 0"),
         ({"max_gross": 2.0, "own_limit": 3.0}, None, "applies only with max_gross null"),
         ({"own_limit": 3.0}, None, "applies only with max_gross null")]
check("own_limit_of and own_limit_bad", all(trader.own_limit_of(m) == want and
                                            ((trader.own_limit_bad(m) or "") .find(bad) >= 0 if bad else trader.own_limit_bad(m) is None)
                                            for m, want, bad in cases),
      [(m, trader.own_limit_of(m), trader.own_limit_bad(m)) for m, _, _ in cases])

# ------------------------------------------------------------------------------------------------ the state
FIX = json.loads(Path("C:/Claude/TWS-Gateway-handoff/whatif_fixture_20261115.json").read_text(encoding="utf-8"))
ACCT = dict(FIX["account"])
DANGER = dict(ACCT, SMA="-50000", read_time="2026-11-18T15:00:00Z")    # a margin warning for a sleeve above the account's x
ST = {"value": 100000.0, "cash": 150000.0, "gross": 2.29, "net": -0.81, "beta_net": -0.9, "positions": [
    dict(p, avg_cost=p["close"], unrealized=0.0, days=1) for p in POS if p["quantity"]], "orders": [], "events": []}
INFO = {"account": DANGER, "sleeves": 10, "total": 900000.0}


def lines_of(**kw):
    return trader.render_account(dict(ST), {}, "HEADER", live=True, max_gross=None, cash_clause=True, **kw).split("\n")


b0, b1 = lines_of(account=INFO), lines_of(account=INFO, own_limit=3.0)
i = next(k for k, l in enumerate(b0) if l.startswith("Sleeve value"))
acct0 = b0[i + 1]
check("text B: the account line carries the margin warning here (the test's premise)", "IBKR margin warning" in acct0, acct0)
check("text B: the line straight after the sleeve-value line, the sentence after the margin warning",
      b1 == b0[:i + 1] + [O.state_line(ST, 3.0)] + [acct0 + O.ACCOUNT_SENTENCE] + b0[i + 2:], b1[:i + 3])
nr0 = lines_of(account={"account": None})
nr1 = lines_of(account={"account": None}, own_limit=3.0)
check("text B, IBKR's figures not read: the line, no sentence", nr1 == nr0[:i + 1] + [O.state_line(ST, 3.0)] + nr0[i + 1:]
      and O.ACCOUNT_SENTENCE not in "\n".join(nr1), nr1[:i + 3])
a0 = trader.render_account(dict(ST), {}, "HEADER", live=True, max_gross=None)
a1 = trader.render_account(dict(ST), {}, "HEADER", live=True, max_gross=None, own_limit=3.0)
check("text A: the line alone", a1.split("\n") == a0.split("\n")[:i + 1] + [O.state_line(ST, 3.0)] + a0.split("\n")[i + 1:]
      and O.ACCOUNT_SENTENCE not in a1)
check("no term: the state as before", "\n".join(lines_of(account=INFO, own_limit=None)) == "\n".join(b0)
      and "exposure limit" not in "\n".join(b0))
over_st = dict(ST, gross=5.0)
check("over: the positions sentence is in the state", O.positions_sentence(over_st["positions"], 200000.0) in
      trader.render_account(dict(over_st), {}, "H", live=True, max_gross=None, own_limit=3.0))

# ------------------------------------------------------------------------------------------------ the order check
NOW = datetime(2026, 11, 18, 10, 31, 7, tzinfo=checks.ET)
R3 = FIX["whatifs"]["R3"]["result"]


class FakeGW:
    def __init__(self, held, resting, fail_positions=False):
        self.held, self.resting, self.fail = held, resting, fail_positions

    def account(self):
        return dict(ACCT)

    def positions(self):
        if self.fail:
            raise TimeoutError("positions")
        return [{"symbol": s, "quantity": q} for s, q in self.held.items()]

    def all_open_orders(self):
        return [dict(r) for r in self.resting]

    def whatif_many(self, orders, timeout=10.0):
        return [dict(R3) for _ in orders]

    def place(self, *a, **k):
        raise AssertionError("place() called during a check")

    def close(self):
        pass


paper.close_only_refs = lambda sleeve: {"tt-cand2-ccc"} if sleeve == "cand2" else set()
FILLS = []
paper.fills = lambda: list(FILLS)
ST2 = {"value": 100000.0, "gross": 2.04, "positions": [{"symbol": "QQQ", "quantity": -200, "close": 650.0},
                                                     {"symbol": "XLV", "quantity": 400, "close": 185.0}]}
CTX = checks.context(ST2, {}, {"QQQ": 650.0, "XLV": 185.0, "SPY": 700.0, "NVDA": 150.0}, 10, 8, sleeve="cand2", own_limit=3.0)
check("context: the term's fields", CTX["own_limit"] == 3.0 and CTX["gross"] == 2.04 and CTX["sleeve"] == "cand2")
check("context without the term: as before", set(checks.context(ST2, {}, {}, 10, 8)) ==
      {"value", "positions", "prices", "sleeves", "looking"})
PLACE = [{"symbol": "QQQ", "action": "BUY", "quantity": 100, "type": "MKT", "tif": "DAY"},             # cuts $65,000
         {"symbol": "SPY", "action": "BUY", "quantity": 50, "type": "LMT", "limit": 700, "tif": "DAY"},  # adds $35,000
         {"symbol": "SPY", "action": "SELL", "quantity": 50, "type": "STP", "stop": 690, "tif": "DAY"},  # protects: 0
         {"symbol": "QQQ", "action": "BUY", "quantity": 100, "type": "STP", "stop": 660, "tif": "DAY"},  # closes: 0
         {"symbol": "XLV", "action": "SELL", "quantity": 500, "type": "STP", "stop": 180, "tif": "DAY"}, # 100 past: $18,000
         {"symbol": "XLV", "action": "BUY", "quantity": 100, "type": "LMT", "limit": 186, "tif": "DAY",  # close_only that
          "close_only": True}]                                                                          # would add: left out
RESTING = [{"ref": "tt-cand2-aaa", "symbol": "XLV", "action": "SELL", "quantity": 100, "order_type": "STP", "aux": 175.0,
            "status": "Submitted"},                                                                     # protects: 0
           {"ref": "tt-cand2-bbb", "symbol": "XLV", "action": "SELL", "quantity": 600, "order_type": "STP", "aux": 170.0,
            "status": "Submitted"},                                     # with the list's 500 and aaa's 100: short 800 at 170
           {"ref": "tt-cand2-ccc", "symbol": "QQQ", "action": "BUY", "quantity": 300, "order_type": "STP", "aux": 700.0,
            "status": "Submitted"},                                                                     # close_only: 0
           {"ref": "tt-base-ml-ddd", "symbol": "NVDA", "action": "BUY", "quantity": 1000, "order_type": "LMT",
            "limit": 150.0, "status": "Submitted"},                                                     # another sleeve's
           {"ref": "tt-cand2-eee", "symbol": "NVDA", "action": "BUY", "quantity": 10, "order_type": "LMT", "limit": 150.0,
            "status": "Submitted"},                                                                     # opens: $1,500
           {"ref": "tt-cand2-fff", "symbol": "NVDA", "action": "BUY", "quantity": 10, "order_type": "LMT", "limit": 150.0,
            "status": "Cancelled"}]                                                                     # done: not resting
WANT = ("Your sleeve, by your own exposure limit (3x its value = $300,000): before these orders its gross exposure is "
        "$204,000 (2.04x). These orders open or add $35,000 and close or shrink $65,000 (a stop or trailing order is not "
        "counted until it fills), so after them it is $174,000 (1.74x): inside your limit, with $126,000 of room left. Stops "
        "and trailing orders in this list would open or add another $18,000 if they fill, and your resting orders another "
        "$127,500 unless you cancel them; they count against your limit too.")    # XLV's three sell stops stack: short 800
text, counted, raw = checks.check(checks.Link(FakeGW({"QQQ": -200, "XLV": 400}, RESTING)), {"place": PLACE}, CTX, now=NOW)
check("check: the sleeve's line is the answer's last line", counted and text.split("\n")[-1] == WANT, text.split("\n")[-1])
check("check: the order lines are still there", text.count("Your sleeve holds") == 6, text)

text_nr, c_nr, _ = checks.check(checks.Link(FakeGW({}, RESTING, fail_positions=True)), {"place": PLACE}, CTX, now=NOW)
check("check: resting orders not read -> said so", c_nr and text_nr.split("\n")[-1] == WANT.replace(
    ", and your resting orders another $127,500 unless you cancel them; they count", "; they count") +
      " Your resting orders are not in these figures; one that would open or add counts against your limit too.",
      text_nr.split("\n")[-1])

ctx_over = dict(CTX, gross=5.0)
t_over, _, _ = checks.check(checks.Link(FakeGW({}, [])), {"place": [PLACE[1]]}, ctx_over, now=NOW)
check("check: over after the orders", t_over.split("\n")[-1] == (
    "Your sleeve, by your own exposure limit (3x its value = $300,000): before these orders its gross exposure is $500,000 "
    "(5.00x). These orders open or add $35,000 and close or shrink $0 (a stop or trailing order is not counted until it "
    "fills), so after them it is $535,000 (5.35x): $235,000 over your limit. Before you submit, change the list until this "
    "figure is at or below $300,000: take out orders that open or add, and put in orders that close or shrink positions."),
      t_over.split("\n")[-1])
t_rest, _, _ = checks.check(checks.Link(FakeGW({}, RESTING[4:5])), {"place": [PLACE[0]]}, CTX, now=NOW)
check("check: resting orders alone", t_rest.split("\n")[-1].endswith(
    "Your resting orders would open or add another $1,500 if they fill, unless you cancel them; they count against your "
    "limit too."), t_rest.split("\n")[-1])
t_neg, _, _ = checks.check(checks.Link(FakeGW({}, [])), {"place": [PLACE[0]]}, dict(CTX, value=-10.0, gross=0.0), now=NOW)
check("check: value not positive", t_neg.split("\n")[-1] == (
    "Your sleeve's value is not positive, so it has no room under its exposure limit. These orders open or add $0 and "
    "close or shrink $65,000. Submit only orders that close or shrink positions."), t_neg.split("\n")[-1])
t_mkt, _, _ = checks.check(checks.Link(FakeGW({}, [])), {"place": [{"symbol": "ZZZ", "action": "BUY", "quantity": 5,
                                                                    "type": "MKT", "tif": "DAY"}]}, CTX, now=NOW)
check("check: an order with no price is named, not counted", t_mkt.split("\n")[-1].endswith(
    "1 order could not be priced and is not counted.") and "open or add $0 " in t_mkt.split("\n")[-1], t_mkt.split("\n")[-1])

plain = {k: v for k, v in CTX.items() if k not in ("own_limit", "gross", "sleeve")}
t_plain, _, _ = checks.check(checks.Link(FakeGW({"QQQ": -200, "XLV": 400}, RESTING)), {"place": PLACE}, plain, now=NOW)
check("check without the term: the answer as before", t_plain == "\n".join(text.split("\n")[:-1]) and "exposure limit" not in t_plain)

real = O.check_block
O.check_block = lambda *a, **k: 1 / 0
said = []
t_fail, c_fail, raw_fail = checks.check(checks.Link(FakeGW({}, [])), {"place": PLACE}, CTX, now=NOW, log=said.append)
O.check_block = real
check("check: a failure of the line costs the line only, and is logged", c_fail and "exposure limit" not in t_fail
      and "ZeroDivisionError" in (raw_fail or {}).get("own_limit_error", "")
      and any("exposure-limit line left out" in m for m in said), (raw_fail and raw_fail.get("own_limit_error"), said))

many = [{"symbol": "QQQ", "action": "BUY", "quantity": 10, "type": "MKT", "tif": "DAY"} for _ in range(24)]
t_many, c_many, _ = checks.check(checks.Link(FakeGW({}, [])), {"place": many}, CTX, now=NOW)
check("check: the line counts the whole list, not only the 20 checked", c_many
      and "open or add $26,000 and close or shrink $130,000" in t_many.split("\n")[-1]      # 240 bought on a 200 short
      and "Orders after the 20th were not checked" in t_many,
      t_many.split("\n")[-1])

real_sl = O.state_line
O.state_line = lambda *a, **k: 1 / 0
said = []
g_fail = trader.render_account(dict(ST), {}, "HEADER", live=True, max_gross=None, cash_clause=True, account=INFO,
                               own_limit=3.0, log=said.append)
O.state_line = real_sl
check("state: a failure of the limit line costs the line only, and is logged",
      g_fail.split("\n") == b0      # no limit line, so no sentence pointing at it
      and any("exposure-limit line left out" in m for m in said), said)

# the walk (the review of 18 Nov): only orders that fill now move the after-figure, at the look's price
PX = {"SMH": [300.0, "delayed last"], "MU": [701.51, "delayed last"], "XYZ": [100.0, "delayed last"],
      "SPY": [700.0, "delayed last"], "XLV": [185.0, "delayed last"], "NVDA": [150.0, "delayed last"]}
BASE = {"value": 100000.0, "gross": 1.0, "own_limit": 3.0, "prices": PX, "positions": {}}


def block(orders, positions=None, resting=(), co=(), filled=None, **kw):
    ctx = dict(BASE, positions=positions or {}, **kw)
    cleaned = []
    for raw in orders:
        o, err = trader.clean_order(raw)
        assert not err, err
        cleaned.append(o)
    return O.check_block(ctx, cleaned, list(resting), set(co), filled)


def order(sym, act, q, typ="MKT", **kw):
    return {"symbol": sym, "action": act, "quantity": q, "type": typ, "tif": "DAY", **kw}


AWAY = "(a stop, a trailing order or a limit order away from the price is not counted until it fills)"
br = block([order("SMH", "SELL", 100), order("SMH", "BUY", 100, "LMT", limit=280), order("SMH", "BUY", 100, "STP", stop=320)])
check("a bracket: the entry adds, its take-profit and its stop add nothing", br == (
    "Your sleeve, by your own exposure limit (3x its value = $300,000): before these orders its gross exposure is $100,000 "
    f"(1.00x). These orders open or add $30,000 and close or shrink $0 {AWAY}, so after them it is $130,000 (1.30x): "
    "inside your limit, with $170,000 of room left."), br)
far = block([order("XYZ", "SELL", 100, "LMT", limit=150)], {"XYZ": 100})
check("a far closing limit is not a cut", "close or shrink $0 " + AWAY in far and "after them it is $100,000" in far, far)
mu = block([order("MU", "BUY", 22, "LMT", limit=660)])
check("a limit entry away from the price is named, at its limit", mu.endswith(
    "Limit orders away from the price in this list would open or add another $14,520 if they fill; they count against "
    "your limit too.") and "open or add $0 " in mu, mu)
mk = block([order("SPY", "BUY", 10, "LMT", limit=710), order("XYZ", "SELL", 50, "LMT", limit=99)], {"XYZ": 100})
check("marketable limits are walked at the look's price, with SPEC's parenthesis",
      "open or add $7,000 and close or shrink $5,000 (a stop or trailing order is not counted until it fills)" in mk, mk)
both = block([order("SMH", "BUY", 10, "STP", stop=310), order("MU", "BUY", 1, "LMT", limit=600)])
check("stops and away limits together", "Stops, trailing orders and limit orders away from the price in this list would "
      "open or add another $3,700 if they fill" in both, both)

moc_cut = block([order("XYZ", "SELL", 100, "MOC")], {"XYZ": 100})
check("an order at the close: what it closes is not counted, and the line says so",
      "close or shrink $0 (a stop or trailing order is not counted until it fills; what an order at the close closes is not "
      "counted before the close), so after them it is $100,000" in moc_cut, moc_cut)
moc_add = block([order("SPY", "BUY", 10, "MOC"), order("XYZ", "SELL", 100)], {"XYZ": 100})
check("an order at the close: what it adds counts, after the orders that fill now",
      "open or add $7,000 and close or shrink $10,000 (a stop or trailing order is not counted until it fills), so" in moc_add,
      moc_add)
flip = block([order("XYZ", "SELL", 150, "MOC")], {"XYZ": 100})
check("an order at the close that flips: the new side counts, the close does not",
      "open or add $5,000 and close or shrink $0 " in flip and "at the close closes" in flip, flip)
loc = block([order("SPY", "BUY", 10, "LOC", limit=690)])
check("a LOC away from the price is a limit away from the price", loc.endswith(
    "Limit orders away from the price in this list would open or add another $6,900 if they fill; they count against "
    "your limit too."), loc)
CO_REST = [{"ref": "tt-x-co1", "symbol": "XLV", "action": "SELL", "quantity": 400, "order_type": "STP", "aux": 170.0}]
co_cov = block([order("XLV", "SELL", 400, close_only=True)], {"XLV": 400}, CO_REST, {"tt-x-co1"})
co_free = block([order("XLV", "SELL", 400, close_only=True)], {"XLV": 400})
check("close_only: already covered by a resting close_only order, refused, not a cut", "close or shrink $0 " in co_cov
      and "close or shrink $74,000 " in co_free, (co_cov, co_free))
co_part = block([order("XLV", "SELL", 100, close_only=True)], {"XLV": 400}, CO_REST, {"tt-x-co1"},
                {"tt-x-co1": 300.0})
check("close_only: a partly filled resting close_only covers what is left of it", "close or shrink $18,500 " in co_part, co_part)
seq = block([order("NVDA", "BUY", 100), order("NVDA", "SELL", 100, "STP", stop=140, close_only=True),
             order("NVDA", "SELL", 50, close_only=True)])
check("close_only: an exit after the list's own close_only stop is refused", "open or add $15,000 and close or shrink $0 " in seq,
      seq)
check("close_only: an order on the wrong side is refused", "close or shrink $0 " in
      block([order("XLV", "BUY", 100, close_only=True)], {"XLV": 400}) and "open or add $0 " in
      block([order("XLV", "BUY", 100, close_only=True)], {"XLV": 400}))

PART = [{"ref": "tt-cand2-ggg", "symbol": "NVDA", "action": "BUY", "quantity": 1000, "order_type": "LMT", "limit": 150.0}]
p1 = block([], {"NVDA": 600}, PART, (), {"tt-cand2-ggg": 600.0})
p2 = block([], {"NVDA": 600}, [dict(PART[0], remaining=400)])
check("resting: a partly filled order counts what is left to fill", p1.endswith(
    "Your resting orders would open or add another $60,000 if they fill, unless you cancel them; they count against your "
    "limit too.") and p1 == p2, (p1, p2))
FILLS[:] = [{"exec_id": "e1", "time": "t", "ref": "tt-cand2-ggg", "shares": 600}]
t_part, _, _ = checks.check(checks.Link(FakeGW({}, [dict(PART[0], status="Submitted")])), {"place": [PLACE[0]]},
                            dict(CTX, positions={"NVDA": 600, "QQQ": -200}), now=NOW)
FILLS[:] = []
check("check: what is left of a resting order, from the ledger's fills under its ref",
      "your resting orders another" not in t_part and "Your resting orders would open or add another $60,000" in t_part,
      t_part.split("\n")[-1])
un = block([order("ZZZ", "SELL", 10, "TRAIL", trail_percent=5)], resting=[
    {"ref": "tt-x-r", "symbol": "QQQQ", "action": "BUY", "quantity": 5, "order_type": "TRAIL"}])
check("unpriced stops and resting orders are named", un.endswith(
    " 1 order could not be priced and is not counted. 1 of your resting orders could not be priced and is not counted."), un)

# orders that fill at a price, read along a move one way (the second review of 18 Nov)
PX.update({"AAA": [100.0, "delayed last"], "INTC": [55.0, "delayed last"]})
st2 = block([order("INTC", "BUY", 300, "STP", stop=56.5), order("INTC", "BUY", 200, "STP", stop=57.5)], {"INTC": -300})
st1 = block([order("INTC", "BUY", 500, "STP", stop=56.5)], {"INTC": -300})
check("stops stacked on one side are read together (two stops of 300 and 200 on a short 300, as one of 500)",
      "would open or add another $11,500 if they fill" in st2 and "would open or add another $11,300 if they fill" in st1,
      (st2, st1))
lr = block([order("INTC", "BUY", 300, "STP", stop=56.5)], {"INTC": -300},
           [{"ref": "tt-x-r1", "symbol": "INTC", "action": "BUY", "quantity": 300, "order_type": "STP", "aux": 57.0}])
check("a list stop beside a resting stop on the same short: the resting one is named", lr.endswith(
    "Your resting orders would open or add another $17,100 if they fill, unless you cancel them; they count against your "
    "limit too."), lr)
ent = block([order("AAA", "BUY", 100, "LMT", limit=95), order("AAA", "SELL", 100, "STP", stop=90),
             order("AAA", "SELL", 100, "LMT", limit=110)])
check("an entry away from the price with its stop and take-profit: counted once (the worst way the price can go)",
      "would open or add another $11,000 if they fill" in ent, ent)
mkt_br = block([order("AAA", "BUY", 100), order("AAA", "SELL", 100, "LMT", limit=110),
                order("AAA", "SELL", 100, "STP", stop=90)])
check("a market entry's take-profit and stop: a round trip is not read, so they add nothing",
      "open or add $10,000 and close or shrink $0" in mkt_br and "would open or add another" not in mkt_br, mkt_br)
co_br = block([order("AAA", "SELL", 100, "LMT", limit=110, close_only=True),
               order("AAA", "SELL", 100, "STP", stop=90, close_only=True)], {"AAA": 100})
check("close_only exits of a held position add nothing", "would open or add another" not in co_br, co_br)
PX.update({"NVDA": [150.0, "delayed last"], "XLE": [100.0, "delayed last"]})
R_PLAIN = {"ref": "tt-x-p", "symbol": "NVDA", "action": "BUY", "quantity": 100, "order_type": "STP", "aux": 160.0}
R_CO = {"ref": "tt-x-c", "symbol": "NVDA", "action": "BUY", "quantity": 100, "order_type": "STP", "aux": 160.0}
t1 = block([], {"NVDA": -100}, [R_PLAIN, R_CO], {"tt-x-c"})
t2 = block([], {"NVDA": -100}, [R_CO, R_PLAIN], {"tt-x-c"})
check("a close_only and a plain stop at one price: the same figure in either order (both fire: long 100)",
      t1 == t2 and "another $16,000 if they fill" in t1, (t1, t2))
eloc = block([order("XLE", "SELL", 100, "LOC", limit=105), order("XLE", "BUY", 100, "STP", stop=110)], {"XLE": 100})
check("an order at the close does not hide what a stop adds during the day", "would open or add another $11,000" in eloc, eloc)
moc_co = block([order("XLE", "SELL", 200, "MOC", close_only=True), order("XLE", "SELL", 100)], {"XLE": 200})
check("a close_only order at the close is never walked past zero", "open or add $0 " in moc_co, moc_co)
thr = block([order("XLV", "SELL", 100, "STP", stop=190)], {"XLV": 400})
check("a stop the price is already through fills now, and the parenthesis says so",
      "close or shrink $18,500 (a stop or trailing order is not counted until it fills, except a stop the price is already "
      "through), so" in thr, thr)

ST_R = dict(ST, orders=[{"ref": "tt-x-o1", "symbol": "INTC", "action": "BUY", "quantity": 500, "type": "STP", "stop": 57.0,
                         "limit": None, "trail_percent": None, "tif": "GTC", "cancel_pending": False}])
check("state, the switch off (the default): the room without the claim that resting orders are in it, resting orders or not",
      O.STATE_RESTING is False and O.state_line(ST_R, 3.0) == O.state_line(ST, 3.0) == O.limit_line(100000.0, 2.29, 3.0, None)
      == ("Your exposure limit: 3x your sleeve's value = $300,000 of gross exposure. You hold $229,000 (2.29x). Room left "
          "for orders that open or add to positions: $71,000.")
      and abs(O.state_resting(ST_R) - 28500.0) < 1e-6, O.state_line(ST_R, 3.0))
check("limit line, resting read and none: the Researcher's form of 18 Nov (no word on resting orders)",
      O.limit_line(100000.0, 2.29, 3.0, 0.0) == O.limit_line(100000.0, 2.29, 3.0, None))
check("state, the switch off, over: the over form as before (no resting sentence)",
      O.state_line(dict(ST_R, gross=5.0), 3.0).startswith(O.limit_line(100000.0, 5.0, 3.0, 0.0)))
O.STATE_RESTING = True
check("state, the switch on: resting orders that would open or add are taken off the room and named", O.state_line(ST_R, 3.0) == (
    "Your exposure limit: 3x your sleeve's value = $300,000 of gross exposure. You hold $229,000 (2.29x), and your resting "
    "orders would open or add another $28,500 if they fill, unless you cancel them (by name, largest first: INTC "
    "$28,500). Room left for orders that open or add to positions, after those resting orders: $42,500."),
      O.state_line(ST_R, 3.0))
big = dict(ST_R, orders=[dict(ST_R["orders"][0], quantity=2000)])
check("state: resting orders past the room", O.state_line(big, 3.0).endswith(
    "Room left for orders that open or add to positions, after those resting orders: none; with them you would be "
    "$43,000 over."), O.state_line(big, 3.0))
ovr = O.state_line(dict(ST_R, gross=5.0), 3.0)
check("state: over, the resting orders' sentence before the positions sentence", (
    "A stop or trailing order does not bring it down until it fills. Your resting orders would open or add another $28,500 "
    "if they fill, unless you cancel them (by name, largest first: INTC $28,500); they count against your limit too. Your "
    "positions by size") in ovr, ovr)
check("state: value not positive, the over form's sentence", O.state_line(dict(ST_R, value=-5.0, gross=0.0), 3.0) == (
    "Over your exposure limit: your sleeve's value is not positive, so you have no room. Place only orders that close or "
    "shrink positions. Your resting orders would open or add another $28,500 if they fill, unless you cancel them (by name, "
    "largest first: INTC $28,500); they count against your limit too."), O.state_line(dict(ST_R, value=-5.0, gross=0.0), 3.0))
prot = dict(ST, orders=[{"ref": "tt-x-o2", "symbol": "XLV", "action": "SELL", "quantity": 400, "type": "STP", "stop": 170.0,
                         "limit": None, "trail_percent": None, "tif": "GTC", "cancel_pending": True}])
check("state: a protective stop adds nothing, and the line has no word on resting orders",
      O.state_line(prot, 3.0) == O.state_line(ST, 3.0) == O.limit_line(100000.0, 2.29, 3.0, None))

# the Researcher's gap of 18 Nov: a trailing order on a name not held has no look price in the state's positions
TR = {"ref": "tt-x-t1", "symbol": "NVDA", "action": "SELL", "quantity": 200, "type": "TRAIL", "stop": None,
      "trail_percent": 1.5, "limit": None, "tif": "GTC", "cancel_pending": False}
check("worst_adds: an unheld trailing order with no price is unpriced; with a look price of 160, $31,520",
      O.worst_adds(0.0, [O._norm(TR, "TRAIL", None, None, 1.5, None, 200, False)], None) == (0.0, 0.0, 1)
      and abs(O.worst_adds(0.0, [O._norm(TR, "TRAIL", None, None, 1.5, None, 200, False)], 160.0)[1] - 31520.0) < 1e-6)
ST_T = dict(ST, orders=[TR])
rf = O.resting_figure(ST_T, {"NVDA": 160.0})
check("resting figure: an unheld trailing order is priced from the look's prices", abs(rf["usd"] - 31520.0) < 1e-6
      and [s for s, _ in rf["names"]] == ["NVDA"] and rf["unpriced"] == 0, rf)
rf0 = O.resting_figure(ST_T, None)
check("resting figure: without a price it is counted as unpriced, not dropped silently",
      rf0 == {"usd": 0.0, "names": [], "unpriced": 1}, rf0)
check("resting figure: IBKR's own trail stop is used where the row has it",
      abs(O.resting_figure(dict(ST, orders=[dict(TR, trail_stop=150.0)]), {"NVDA": 160.0})["usd"] - 30000.0) < 1e-6)
check("resting figure: a partly filled order counts what is left to fill",
      abs(O.resting_figure(dict(ST_R, orders=[dict(ST_R["orders"][0], remaining=200)]))["usd"] - 11400.0) < 1e-6)
held_px = O.resting_figure(dict(ST_R), {"INTC": 1.0})       # a held name keeps the state's own mark... here INTC is not held
check("resting figure: a stop on an unheld name is read at its stop price, whatever the look price",
      abs(held_px["usd"] - 28500.0) < 1e-6, held_px)
un_line = O.state_line(ST_T, 3.0)
check("state: an unpriced resting order alone is said, under the limit", un_line.endswith(
    "Room left for orders that open or add to positions: $71,000; 1 of your resting orders could not be priced and is not "
    "counted in it."), un_line)
check("state: an unpriced resting order alone is said, over the limit", O.state_line(dict(ST_T, gross=5.0), 3.0).split(
    " Your positions by size")[0].endswith("until it fills. 1 of your resting orders could not be priced; if it would open "
                                           "or add to a position, it counts against your limit too."),
      O.state_line(dict(ST_T, gross=5.0), 3.0))
both = dict(ST, orders=[TR, ST_R["orders"][0]])
check("state: with resting dollars, an unpriced one has its own sentence, 'other'", O.state_line(both, 3.0) == (
    "Your exposure limit: 3x your sleeve's value = $300,000 of gross exposure. You hold $229,000 (2.29x), and your resting "
    "orders would open or add another $28,500 if they fill, unless you cancel them (by name, largest first: INTC "
    "$28,500). 1 other resting order could not be priced. Room left for orders that open or add to positions, after those "
    "resting orders: $42,500."), O.state_line(both, 3.0))
check("state: over, with resting dollars and an unpriced one, 'they' keeps its plural", O.state_line(dict(both, gross=5.0),
      3.0).split(" Your positions by size")[0].endswith(
    "unless you cancel them (by name, largest first: INTC $28,500); they count against your limit too. 1 other resting "
    "order could not be priced."), O.state_line(dict(both, gross=5.0), 3.0))
check("limit line: plurals of the unpriced forms",
      O.limit_line(100000.0, 5.0, 3.0, 0.0, (), 2).endswith("2 of your resting orders could not be priced; any that would "
                                                           "open or add to a position count against your limit too.")
      and O.limit_line(100000.0, 2.0, 3.0, 0.0, (), 2).endswith("2 of your resting orders could not be priced and are not "
                                                               "counted in it.")
      and O.limit_line(100000.0, 2.0, 3.0, 9000.0, [("A", 9000.0)], 2).count(" 2 other resting orders could not be priced. ")
      == 1)
five = dict(ST, orders=[dict(ST_R["orders"][0], ref=f"tt-x-n{i}", symbol=s, quantity=q)
                        for i, (s, q) in enumerate([("AAA", 100), ("BBB", 300), ("CCC", 200), ("DDD", 50), ("EEE", 10)])])
nm = O.state_line(five, 3.0)
check("state: the three largest names, then 'and N more names'",
      "(by name, largest first: BBB $17,100, CCC $11,400, AAA $5,700, and 2 more names)." in nm
      and O._names([("A", 4.0), ("B", 3.0), ("C", 2.0), ("D", 1.0)]).endswith(", and 1 more name)"), nm)
nothing = dict(ST, orders=[dict(TR, symbol="ZZZ", remaining=0), dict(TR, symbol="YYY", close_only=True)])
check("resting figure: an order with nothing left, or a close_only one with nothing to close, is not 'could not be priced'",
      O.resting_figure(nothing, None) == {"usd": 0.0, "names": [], "unpriced": 0}, O.resting_figure(nothing, None))
_rf = O.resting_figure
O.resting_figure, said_r = (lambda st, prices=None: 1 / 0), []
try:
    check("state: a failure in the resting figure costs its words only, the line is the switch-off form",
          O.state_line(ST_R, 3.0, None, said_r.append) == O.limit_line(100000.0, 2.29, 3.0, None)
          and O.state_line(dict(ST_R, gross=5.0), 3.0, None, said_r.append).startswith(O.limit_line(100000.0, 5.0, 3.0, None))
          and len(said_r) == 2 and said_r[0].startswith("resting orders left out of the exposure-limit line: ZeroDivision"))
finally:
    O.resting_figure = _rf
JUNK = 1.7976931348623157e308
check("resting figure: IBKR's unset value as a price is no price (counted as unpriced, never $inf)",
      O.resting_figure(dict(ST, orders=[dict(TR, trail_stop=JUNK)]), {"NVDA": JUNK}) == {"usd": 0.0, "names": [], "unpriced": 1}
      and O.resting_figure(dict(ST, orders=[dict(ST_R["orders"][0], stop=JUNK)]), None)["unpriced"] == 1
      and "inf" not in O.state_line(dict(ST, orders=[dict(TR, trail_stop=JUNK)]), 3.0, {"NVDA": JUNK}))
check("resting figure: the look's prices never override a held name's own mark",
      abs(O.resting_figure(dict(ST, orders=[dict(TR, symbol="QQQ", action="BUY", quantity=400)]), {"QQQ": 1.0})["usd"]
          - 200 * 650.0 * 1.015) < 1e-6)
_sl, seen = O.state_line, []
O.state_line = lambda st, limit, prices=None, log=None: seen.append(prices) or _sl(st, limit, prices, log)
try:
    trader.render_account(dict(ST_T), {"NVDA": {"close": 160.0}}, "H", {"NVDA": 161.0}, None, own_limit=3.0)
    check("render: the limit line is given the look's prices (table close, delayed last over it)",
          seen == [{"NVDA": 161.0}], seen)
finally:
    O.state_line = _sl
check("prices_of: the table's close, the delayed last over it, junk left out",
      O.prices_of({"A": {"close": 10.0}, "B": {"close": float("nan")}, "C": {"close": 5.0}, "D": "x"},
                  {"C": 6.0, "E": 0.0}) == {"A": 10.0, "C": 6.0})
O.STATE_RESTING = False
check("account sentence: the form for a margin warning that names the sleeve's own gross",
      O.account_sentence("... Your gross exposure is $5,000 with its value at or below zero; ...") == O.ACCOUNT_SENTENCE_OWN
      and O.account_sentence(acct0) == O.ACCOUNT_SENTENCE)
ctx_m = checks.context(dict(ST2, positions=ST2["positions"] + [{"symbol": "ABC", "quantity": 5, "close": 12.5}]), {}, {},
                       10, 8, sleeve="cand2", own_limit=3.0)
check("context: held names get the state's mark when the table and quotes lack them",
      ctx_m["prices"].get("ABC") == [12.5, "position mark"] and "ABC" not in checks.context(
          dict(ST2, positions=ST2["positions"] + [{"symbol": "ABC", "quantity": 5, "close": 12.5}]), {}, {}, 10, 8)["prices"])
check("own_limit: infinity and NaN are not a limit", trader.own_limit_of({"max_gross": None, "own_limit": float("inf")}) is None
      and trader.own_limit_of({"max_gross": None, "own_limit": float("nan")}) is None
      and "finite" in trader.own_limit_bad({"max_gross": None, "own_limit": float("inf")}))

# a sleeve without the term reads exactly what the committed code (7fa16b14, before this change) gave
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
OLD = Path(tempfile.mkdtemp())


def old_module(name):
    src = subprocess.run(["git", "show", f"7fa16b14:{name}.py"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    (OLD / f"{name}_old.py").write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"{name}_old", OLD / f"{name}_old.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T0, C0 = old_module("trader"), old_module("checks")
combos = [dict(ml=ml, news=nw, kelly=k, max_gross=mg, ml_tool=t) for ml in (False, True) for nw in (False, True)
          for k in (False, True) for mg in (2.0, None, 1.5) for t in (None, "ML4")]
check("as before: the system text without the term, against 7fa16b14 (48 combinations)",
      all(trader.live_system(90000, "2026-11-18", **c) == T0.live_system(90000, "2026-11-18", **c) for c in combos))
recs = sorted(Path(ROOT / "runs" / "live" / "2026-11-17").glob("*-1530.json"))[:10]
same = []
for p in recs:
    r = json.loads(p.read_text(encoding="utf-8"))
    s0 = dict(r["state"], events=r.get("events") or [])
    for kw in (dict(live=True, max_gross=None, cash_clause=True, account=s0.get("account")), dict(live=True, max_gross=None)):
        same.append(trader.render_account(dict(s0), {}, "H", **kw) == T0.render_account(dict(s0), {}, "H", **kw))
check(f"as before: the state without the term, against 7fa16b14 ({len(same)} renders of Tue 15:30's records)",
      same and all(same), same)
t_old, _, _ = C0.check(C0.Link(FakeGW({"QQQ": -200, "XLV": 400}, RESTING)), {"place": PLACE}, plain, now=NOW)
check("as before: the check's answer without the term, against 7fa16b14", t_old == t_plain, (t_old[-300:], t_plain[-300:]))

# the look's context: a check's answer cut to fit keeps the sleeve's own line
import agent  # noqa: E402
own = WANT
body = "\n".join([f"{k}. BUY 10 QQQ MKT DAY: value $6,500 at the delayed last 650.00 = 0.07x your sleeve's value." * 3
                  for k in range(1, 30)] + [own])
counter = "(order check 1 of 3)"
room = len(agent.SUBMIT_OPEN()) + agent.SUBMIT_ROOM
toks, cut_text = agent._fit_text("check_orders", body, room + 600, counter)
check("context: a cut check keeps its exposure-limit line after the cut line", agent.CUT_LINE in cut_text
      and cut_text.endswith(agent.CUT_LINE + "\n" + own + "\n" + counter) and cut_text.startswith("1. BUY"), cut_text[-400:])
toks2, tiny = agent._fit_text("check_orders", body, room + 60, counter)
check("context: no room for the line: cut as before (as much of the first line as fits, the cut line, the counter)",
      own not in tiny and tiny.endswith(agent.CUT_LINE + "\n" + counter) and tiny.startswith("1. BUY"), tiny)
plain_body = "\n".join(body.split("\n")[:-1])
check("context: an answer without the line is cut as before", agent._fit_text("check_orders", plain_body, room + 600, counter)[1]
      .endswith(agent.CUT_LINE + "\n" + counter))
ck = lambda a: None
ck.own_limit = 3.0
check("context: the check's estimate allows for the line, only for a checker that carries it",
      agent._estimate("check_orders", {"place": [1, 2]}, ck) == 300 + 150 + 220
      and agent._estimate("check_orders", {"place": [1, 2]}, lambda a: None) == 300 + 220
      and agent._estimate("check_orders", {"place": [1, 2]}) == 300 + 220)

# ------------------------------------------------------------------------------------------------ the world and the wiring
paper.sleeves = lambda: {"cand2": {"model": "x", "start_cash": 90000.0, "max_gross": None, "own_limit": 3.0},
                         "fable": {"model": "claude-fable-5-2", "start_cash": 30000.0, "max_gross": None}}
check("PaperWorld: the sleeve's term", trader.PaperWorld(None, "cand2", None).OWN_LIMIT == 3.0
      and trader.PaperWorld(None, "fable", None).OWN_LIMIT is None)
src = inspect.getsource(live)


def seg(start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


check("live: the term reaches the system text, the state, the check, the record and research.jsonl",
      "own_limit=trader.own_limit_of(meta)" in seg("system = trader.live_system(", "\n            return world")
      and "world.OWN_LIMIT = trader.own_limit_of(meta)" in seg("world = trader.PaperWorld(gw, name, data)", "text, st, feats")
      and "own_limit=trader.look_limit(trader.own_limit_of(meta))" in seg("check_ctxs[name] = checks.context(", "except")
      and "marks=trader.own_limit_of(meta) is not None" in seg("check_ctxs[name] = checks.context(", "except")
      and '"own_limit": trader.own_limit_of(meta)' in seg("rec = {", "out_dir /")
      and '"own_limit": rec["own_limit"]' in seg('paper.append(LIVE / "research.jsonl"', "log(")
      and "trader.own_limit_bad(v)" in src and 'checker.own_limit = ctx.get("own_limit")' in src)

# ------------------------------------------------------------------------------------------------ the fill lines' position
held = {"MU": -22, "GLD": 3, "BRK B": 10}
check("fill line: no position now", trader.fill_position_note(
    "filled BUY 100 NVDA @ 155.05 (commission $1.00) [tt-base-90k-99acde49d6]", held) == " (you now hold no NVDA)")
check("fill line: short and long now, a split fill on each line, a symbol with a space",
      trader.fill_position_note("filled BUY 28 MU @ 702.80 (commission $1.00) [tt-x-1]", held) == " (you are now short 22 MU)"
      and trader.fill_position_note("filled SELL 47 GLD @ 352.41 (commission $0.31) [ibkr-liquidation-20261117-GLD]", held)
      == " (you are now long 3 GLD)"
      and trader.fill_position_note("filled BUY 10 BRK B @ 500.00 (commission $1.00) [tt-x-2]", held)
      == " (you are now long 10 BRK B)")
check("fill line: other lines get nothing", all(trader.fill_position_note(e, held) == "" for e in (
    "BUY 95 AMD TRAIL: about $46,479 at the last price", "framework note (Wed 18 Nov): ...",
    "cancel tt-x-3: GatewayError: Ref 'tt-x-3' is not a", "refused: tt-x BUY 5 XYZ MKT")))
EVS = dict(ST, orders=[], events=["filled BUY 100 QQQ @ 650.00 (commission $1.00) [tt-x-a]",
                                  "filled SELL 5 NVDA @ 155.00 (commission $1.00) [tt-x-b]", "BUY 5 XYZ MKT: refused"])
ra_off = trader.render_account(EVS, {}, "H", None, None)
ra_on = trader.render_account(EVS, {}, "H", None, None, fill_position=True)
check("render: fill_position off (the default) is the text as before; on, a symbol's fill line ends with the position now",
      ra_off == trader.render_account(EVS, {}, "H", None, None, fill_position=False)
      and "[tt-x-a] (you are now short 200 QQQ); filled SELL 5 NVDA @ 155.00 (commission $1.00) [tt-x-b] (you now hold no "
          "NVDA); BUY 5 XYZ MKT: refused" in ra_on
      and ra_on.replace(" (you are now short 200 QQQ)", "").replace(" (you now hold no NVDA)", "") == ra_off, ra_on)
SPLIT = dict(ST, orders=[], positions=ST["positions"] + [{"symbol": "AMD", "quantity": 30, "avg_cost": 500.0, "close": 503.0,
                                                          "unrealized": 0.0, "days": 0}],
             events=["filled BUY 30 AMD @ 503.48 (commission $1.00) [tt-x-1]", "filled SELL 100 QQQ @ 650.00 (commission $1.00) "
                     "[tt-x-2]", "filled BUY 30 AMD @ 503.48 (commission $1.00) [tt-x-3]", "filled BUY 40 AMD @ 503.45 "
                     "(commission $0.00) [tt-x-3]", "filled SELL 70 AMD @ 504.00 (commission $1.00) [tt-x-4]"])
sp = trader.render_account(SPLIT, {}, "H", None, None, fill_position=True).split(chr(10))[-1]
check("render: only the last fill line of each symbol carries the note (the Researcher, 18 Nov)",
      sp.count("(you are now long 30 AMD)") == 1 and sp.count("(you are now short 200 QQQ)") == 1
      and "[tt-x-4] (you are now long 30 AMD)" in sp and "[tt-x-2] (you are now short 200 QQQ)" in sp
      and "[tt-x-1];" in sp and "[tt-x-3];" in sp, sp)
check("fill line: the quantity as the positions table prints it (no 1e+06)",
      trader.fill_position_note("filled BUY 1000000 SIRI @ 4.00 (commission $1.00) [tt-x]", {"SIRI": 1500000})
      == " (you are now long 1500000 SIRI)")
_fp, said = trader.fill_position_note, []
trader.fill_position_note = lambda e, h: 1 / 0
try:
    ra_err = trader.render_account(EVS, {}, "H", None, None, fill_position=True, log=said.append)
    check("render: a failure in the fill notes costs the notes only, logged", ra_err == ra_off
          and said and said[0].startswith("fill-position notes left out: ZeroDivisionError"), (said, ra_err[-200:]))
finally:
    trader.fill_position_note = _fp
check("switches: the resting figure and the fill-line fact are on (from the 10:30 look of 18 Nov)",
      DEFAULTS == (True, True), DEFAULTS)
check("account_text: fill_position is the one switch, for every sleeve and text",
      "fill_position=FILL_POSITION" in inspect.getsource(trader.account_text))


# ------------------------------------------------------------------------------------------------ Thursday 19 Nov: the term alone
# Clayton, 18 Nov: "Don't decide for the trader model."; the Researcher's one change: keep the system term, take out every
# per-look piece (the state's limit line, the account sentence, the check's line), add nothing in their place.
check("19 Nov: the per-look pieces are off by default, and look_limit gives None", trader.OWN_LIMIT_LOOK is False
      and trader.look_limit(3.0) is None and trader.look_limit(None) is None)


class _TW:                                  # account_text's world: a text-B sleeve with the term, IBKR's figures read
    TEXT, MAX_GROSS, OWN_LIMIT, SHOCK_CAP, ACCOUNT, log = "B", None, 3.0, None, INFO, None
    state = staticmethod(lambda d, feats, live=None: dict(ST))


class _TD:
    iso = staticmethod(lambda d: "2026-11-19")


acct_on_world, _ = trader.account_text(_TW, _TD, 0, {}, "HEADER", live=True, cash_clause=True)
check("19 Nov: account_text with the sleeve's term reads the state as with no term: no limit line, no account sentence",
      acct_on_world == "\n".join(lines_of(account=INFO)) and "exposure limit" not in acct_on_world
      and O.ACCOUNT_SENTENCE not in acct_on_world and O.ACCOUNT_SENTENCE_OWN not in acct_on_world
      and "Gross exposure 2.29x" in acct_on_world, acct_on_world[:600])
_TW.TEXT = "A"
acct_a, _ = trader.account_text(_TW, _TD, 0, {}, "HEADER", live=True)
check("19 Nov: the text-A control too", acct_a == trader.render_account(dict(ST), {}, "HEADER", live=True, max_gross=None)
      and "exposure limit" not in acct_a, acct_a[:400])
_TW.TEXT = "B"
_st_over = dict(ST, gross=5.0)
check("19 Nov: over the limit, nothing is added either (the over form and the positions sentence are gone)",
      "Over your exposure limit" not in trader.render_account(dict(_st_over), {}, "H", live=True, max_gross=None,
                                                             own_limit=trader.look_limit(3.0))
      and "Your positions by size" not in trader.render_account(dict(_st_over), {}, "H", live=True, max_gross=None,
                                                               own_limit=trader.look_limit(3.0)))
ctx_thu = checks.context(ST2, {}, {"QQQ": 650.0, "XLV": 185.0, "SPY": 700.0, "NVDA": 150.0}, 10, 8, sleeve="cand2",
                         own_limit=trader.look_limit(3.0))
check("19 Nov: the check's context carries no limit, so the answer has no own-limit line and the estimate no allowance",
      "own_limit" not in ctx_thu and "gross" not in ctx_thu and "sleeve" not in ctx_thu)
ST_M = dict(ST2, positions=ST2["positions"] + [{"symbol": "ABC", "quantity": 5, "close": 12.5}])
PX = {"QQQ": 650.0, "XLV": 185.0, "SPY": 700.0, "NVDA": 150.0}
ctx_mk = checks.context(ST_M, {}, PX, 10, 8, sleeve="cand2", own_limit=trader.look_limit(3.0), marks=True)
ctx_wed = checks.context(ST_M, {}, PX, 10, 8, sleeve="cand2", own_limit=3.0)
check("19 Nov: a sleeve with the term keeps Wednesday's prices in the check (the position mark for a name the table and "
      "quotes lack) without the limit; a sleeve without the term (Fable) reads its prices as before",
      ctx_mk["prices"] == ctx_wed["prices"] and ctx_mk["prices"].get("ABC") == [12.5, "position mark"]
      and "own_limit" not in ctx_mk and "ABC" not in checks.context(ST_M, {}, PX, 10, 8, sleeve="fable")["prices"]
      and checks._value(trader.clean_order({"symbol": "ABC", "action": "SELL", "quantity": 5, "type": "MOO",
                                            "tif": "DAY"})[0], ctx_mk).startswith("value $62 at the position mark 12.50"),
      ctx_mk["prices"])
check("19 Nov: the system text keeps the term word for word, through live_system and not look_limit",
      trader.live_system(90000, "2026-11-19", ml=True, ml_tool="ML4", max_gross=None, own_limit=3.0).count(SPEC_TERM) == 1
      and "look_limit" not in inspect.getsource(trader.live_system))
trader.OWN_LIMIT_LOOK = True
try:
    acct_old, _ = trader.account_text(_TW, _TD, 0, {}, "HEADER", live=True, cash_clause=True)
    check("19 Nov: the switch on gives 18 Nov's state (for replays)", acct_old == "\n".join(lines_of(account=INFO, own_limit=3.0))
          and trader.look_limit(3.0) == 3.0, acct_old[:600])
finally:
    trader.OWN_LIMIT_LOOK = False
print(f"\n{sum(ok)} of {len(ok)} checks passed")
O.STATE_RESTING = DEFAULTS[0]
sys.exit(0 if all(ok) else 1)
