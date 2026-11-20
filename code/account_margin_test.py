"""The whole account's margin after a look (no software limit on a sleeve's exposure from Mon 16 Nov 2026, so IBKR's
account margin is the only one). Offline.
- live.account_margin: the log line, from IBKR's own SMA when the framework reports it, else from its floor (equity with
  loan minus half of gross); the warning thresholds; never raises.
- live.reconcile and live.liquidations: on a position mismatch IBKR's own liquidation fills are read from every client's
  executions and logged once each; with the books right nothing is asked; an older framework costs only a log line.

    python account_margin_test.py
"""
import sys

sys.dont_write_bytecode = True
import live  # noqa: E402
import paper  # noqa: E402

ok, lines = [], []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


class GW:
    def __init__(self, row=None, fail=False, positions=(), execs=(), old=False):
        self.row, self.fail, self._pos, self._execs, self.old, self.asked = row, fail, list(positions), list(execs), old, []

    def account(self):
        if self.fail:
            raise RuntimeError("No reply from IB Gateway")
        return self.row

    def positions(self):
        return self._pos

    def executions(self, days=None, **k):
        self.asked.append((days, k))
        if self.old and k:
            raise TypeError("executions() got an unexpected keyword argument 'all_clients'")
        return [dict(e) for e in self._execs]


def row(nl, gross, xl, af=500000.0, sma=None):
    r = {"NetLiquidation": nl, "GrossPositionValue": gross, "EquityWithLoanValue": nl - 1000, "ExcessLiquidity": xl, "AvailableFunds": af}
    return r if sma is None else dict(r, SMA=sma, RegTMargin=0.5 * gross, Cushion=xl / nl if nl else 0.0)


saved = [(live, "log", live.log), (paper, "sleeves", paper.sleeves), (paper, "book", paper.book)]
live.log = lines.append
try:
    # 1. the margin line
    w = live.account_margin(GW(row(714825.78, 775594.78, 510898.06, 509798.74)))      # Sat 14 Nov's account, before SMA was reported
    check("no SMA in the row: the floor is shown, no warning", w is False and lines[-1].startswith("account margin: ")
          and "gross $775,595 (1.09x)" in lines[-1] and "Reg T room at the close at least $326,028" in lines[-1], lines[-1])
    w = live.account_margin(GW(row(714825.78, 775594.78, 510898.06, 509798.74, sma=489150.32)))
    check("SMA in the row: IBKR's own figure is shown, with its floor beside it", w is False
          and "SMA $489,150 (Reg T room at the close; never below $326,028)" in lines[-1], lines[-1])
    w = live.account_margin(GW(row(700000.0, 1500000.0, 330000.0, sma=210000.0)))     # gross 2.14x, the floor negative, SMA carried
    check("a negative floor with SMA still well above zero is no warning: SMA is what IBKR checks", w is False and "never below $-51,000" in lines[-1],
          lines[-1])
    w = live.account_margin(GW(row(700000.0, 1500000.0, 330000.0, sma=40000.0)))
    check("SMA under 10% of net liquidation: a warning", w is True and lines[-1].startswith("ACCOUNT MARGIN WARNING: "), lines[-1])
    w = live.account_margin(GW(row(700000.0, 1500000.0, 330000.0, sma=-12000.0)))
    check("a negative SMA warns and shows the figure", w is True and "SMA $-12,000" in lines[-1], lines[-1])
    w = live.account_margin(GW(row(700000.0, 1300000.0, 380000.0)))                   # no SMA: floor 49,000, under 10% of 700,000
    check("no SMA and the floor under 10% of net liquidation: a warning", w is True and lines[-1].startswith("ACCOUNT MARGIN WARNING: "), lines[-1])
    w = live.account_margin(GW(row(700000.0, 600000.0, 150000.0, sma=400000.0)))      # a 21% cushion
    check("excess liquidity under 25% of net liquidation: a warning", w is True and lines[-1].startswith("ACCOUNT MARGIN WARNING: "))
    w = live.account_margin(GW(row(700000.0, 0.0, 700000.0, sma=699000.0)))
    check("no positions: a plain line", w is False and "gross $0 (0.00x)" in lines[-1], lines[-1])
    w = live.account_margin(GW(fail=True))
    check("a gateway error is logged and never raised", w is None and lines[-1].startswith("account margin not read"), lines[-1][:60])
    w = live.account_margin(GW({"NetLiquidation": 1.0}))
    check("a row without the fields is logged and never raised", w is None and lines[-1].startswith("account margin not read"))
    w = live.account_margin(GW(row(0.0, 0.0, 0.0)))
    check("a zero net liquidation neither divides by zero nor passes quietly", lines[-1].startswith("ACCOUNT MARGIN WARNING: "), lines[-1][:80])

    # 2. a position mismatch asks for IBKR's liquidation fills
    paper.sleeves = lambda: {"a": {}, "b": {}}
    paper.book = lambda name, marks=None: {"positions": {"NVDA": {"quantity": 100 if name == "a" else -40}, "XLP": {"quantity": 50 if name == "a" else 0}}}
    ex = [{"exec_id": "x1", "time": "20261117 15:52:10 US/Eastern", "symbol": "NVDA", "side": "SLD", "shares": 60, "price": 161.2, "ref": "",
           "client_id": 0, "liquidation": 1},
          {"exec_id": "x2", "time": "20261117 10:00:01 US/Eastern", "symbol": "XLP", "side": "BOT", "shares": 50, "price": 92.3, "ref": "tt-a-1",
           "client_id": 10, "liquidation": 0}]
    live._liquidations_seen.clear()
    lines.clear()
    g = GW(positions=[{"symbol": "NVDA", "quantity": 60}, {"symbol": "XLP", "quantity": 50}], execs=ex)
    check("books that match the account: no mismatch, and no executions are asked for", live.reconcile(g) == {} and not g.asked and not lines, lines)
    g = GW(positions=[{"symbol": "NVDA", "quantity": 0}, {"symbol": "XLP", "quantity": 50}], execs=ex)
    d = live.reconcile(g)
    check("a mismatch is logged with the account and sleeve figures", d == {"NVDA": (0, 60)} and lines[0].startswith("POSITION MISMATCH"), lines[:1])
    check("and every client's executions are read for IBKR's liquidation fills", g.asked == [(2, {"all_clients": True})]
          and len([m for m in lines if m.startswith("IBKR LIQUIDATION: ")]) == 1
          and "SLD 60 NVDA @ 161.2" in lines[1] and "client 0" in lines[1] and "x1" in lines[1], lines[1:])
    lines.clear()
    live.reconcile(g)
    check("the same liquidation fill is logged once, the mismatch every time",
          len([m for m in lines if m.startswith("POSITION MISMATCH")]) == 1 and not [m for m in lines if m.startswith("IBKR LIQUIDATION")], lines)
    lines.clear()
    g = GW(positions=[{"symbol": "XLP", "quantity": 50}], execs=ex[1:])
    live.reconcile(g)
    check("a mismatch with no liquidation fill (a reset, a lost fill) logs the mismatch only", len(lines) == 1 and lines[0].startswith("POSITION MISMATCH"), lines)
    lines.clear()
    g = GW(positions=[{"symbol": "XLP", "quantity": 50}], execs=ex, old=True)
    d = live.reconcile(g)
    check("an older framework without all_clients costs a log line, never the reconcile",
          d == {"NVDA": (0, 60)} and any(m.startswith("liquidation fills not read") for m in lines), [m[:40] for m in lines])
finally:
    for m, a, v in saved:
        setattr(m, a, v)
print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
