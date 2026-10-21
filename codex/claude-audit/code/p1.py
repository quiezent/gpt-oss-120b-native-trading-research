"""Codex's "P1" simulator interface, reimplemented to re-test NAV4 and NAV9 the way they were trained.

NAV1-NAV5 (NAV4 included) trained only on this interface, and 12 of NAV9's 16 updates came from it;
Codex's controlled base/NAV4/NAV9 comparison used it too. The audit (audit/interface.md) extracted
its exact prompt, tool and result formats; this module reproduces them byte for byte (checked
against Codex's saved prompt tokens and tool results by validate_p1.py) from my own code.

World: AAPL, MSFT, SPY daily bars (split-adjusted, not dividend-adjusted TRADES), $10,000 cash,
long-only DAY limit orders that fill only at the next session's open adjusted 7.5 bps against the
trader, commission max($1, $0.005/share), $2,500 per order, $5,000 per symbol, $9,000 gross,
$500 minimum cash, 4 orders per session. One deliberate fix (sell_cap_bug=False): a SELL is never
blocked by the per-order notional cap. Codex's host accepted such SELLs and then silently dropped
them at the open (audit SIM-3); sell_cap_bug=True reproduces that for validation.
"""

import json
import math
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYMBOLS = ("AAPL", "MSFT", "SPY")
CONIDS = {"AAPL": 265598, "MSFT": 272093, "SPY": 756733}
UP, DOWN = Decimal("1.00075"), Decimal("0.99925")
ONE, PER_SHARE = Decimal("1"), Decimal("0.005")
CASH0 = Decimal("10000")
MAX_ORDER, MAX_SYMBOL, MAX_GROSS, MIN_CASH = Decimal("2500"), Decimal("5000"), Decimal("9000"), Decimal("500")
MAX_ORDERS_PER_SESSION, MAX_POSITIONS = 4, 4
REF = re.compile(r"^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$")
FIELDS = {
    "account": {"command"},
    "historical-daily": {"command", "symbol", "count"},
    "submit-stock-order": {"command", "conid", "symbol", "side", "quantity", "order_type", "limit", "tif", "order_ref"},
    "cancel-order": {"command", "order_ref"},
    "advance": {"command", "sessions"},
    "finish": {"command"},
}
QUALIFICATION = {
    "blind_evaluation": False, "historical_point_in_time_proven": False,
    "kind": "DEVELOPER_EXPOSED_RETROSPECTIVE_PRICE_SIMULATION",
    "limitations": ["Current-listed universe has survivorship bias.",
                    "Historical correction/adjustment and original publication vintages are unknown.",
                    "Dividend cashflows and corporate actions are not reconstructed.",
                    "Next-open daily fills omit queues, intraday liquidity and partial fills.",
                    "Previously used historical evaluation windows are development exposed."],
    "price_basis": "retrieved_split_adjusted_non_dividend_TRADES", "real_broker_fills": False,
}
SYSTEM = ("You are one autonomous trading agent. Your objective is positive NAV returns after transaction costs over "
          "the specified horizon. Choose all tool calls and exact arguments yourself. You may research, trade, manage "
          "inventory or retain cash. No action is prescribed. This is a retrospective simulator using developer-exposed "
          "historical prices; economic outcomes are revealed only by advancing its clock. Act only on information "
          "already available in the observation. Finalize when your strategy is committed. No hidden role routes exist.")


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dec(x):
    return Decimal(str(x))


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def load_bars(source="ibkr"):
    """{symbol: [bar, ...]} with session 'YYYY-MM-DD', oldest first, in the units the models saw.

    source="codex": Codex's saved snapshot (through 2026-10-02). source="ibkr": data/trades/*.json
    fetched through the framework (TRADES, volume divided by 40 to match the training data's units;
    prices match the snapshot exactly on all 494 overlapping sessions). Incomplete sessions are dropped.
    """
    out = {}
    if source == "codex":
        snap = json.loads((ROOT / "data/codex/controlled_prepared/selected_history_snapshot.json").read_text())
        for s in SYMBOLS:
            out[s] = [{k: b[k] for k in ("open", "high", "low", "close", "volume", "session")} for b in snap[s]["bars"]]
        return out
    from bars import completed
    for s in SYMBOLS:
        rows = completed([{"date": str(r["date"])[:8], **r} for r in json.loads((ROOT / f"data/trades/{s}.json").read_text())])
        out[s] = [{"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
                   "volume": float(r["volume"]) / 40, "session": f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:8]}"}
                  for r in rows]
    return out


def tool_spec():
    """The pa_tws tool exactly as NAV4 trained on it (property order matters for the rendered prompt)."""
    cmd_desc = "Choose one supported command and supply exactly its declared fields. No additional fields or invented commands are accepted."
    sym = {"description": "Required for historical-daily and submit-stock-order. Exact declared symbol; conid must refer to the same contract.",
           "enum": list(SYMBOLS), "type": "string"}
    ref = {"description": "Required for submit-stock-order (new unique reference) or cancel-order (the queued order's existing exact reference). Must match native CLI syntax ^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$, 8..64 characters. Invalid or duplicate supplied model references are rejected, never rewritten.",
           "maxLength": 64, "minLength": 8, "pattern": "^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$", "type": "string"}
    props = {
        "command": {"description": cmd_desc, "enum": list(FIELDS), "type": "string"},
        "symbol": sym,
        "count": {"description": "Required only for historical-daily: unquoted whole JSON integer 1..252. A quoted string such as a number is invalid.",
                  "maximum": 252, "minimum": 1, "type": "integer"},
        "conid": {"description": "Required only for submit-stock-order: unquoted positive whole JSON integer from verified_contracts, matching symbol.",
                  "enum": [CONIDS[s] for s in SYMBOLS], "type": "integer"},
        "side": {"description": "Required only for submit-stock-order. BUY purchases shares; SELL disposes of owned shares. No short sale.",
                 "enum": ["BUY", "SELL"], "type": "string"},
        "quantity": {"description": "Required only for submit-stock-order: unquoted positive whole JSON integer number of shares. Fractional, quoted or zero quantities are invalid.",
                     "minimum": 1, "type": "integer"},
        "order_type": {"description": "Required only for submit-stock-order. Explicit LMT; no default is supplied.", "enum": ["LMT"], "type": "string"},
        "limit": {"description": "Required only for submit-stock-order: positive USD price per share, finite decimal string or JSON number. Model chooses the limit; it is never replaced or clipped.",
                  "type": ["string", "number"]},
        "tif": {"description": "Required only for submit-stock-order. Explicit DAY; expires at the next execution open if not filled. No default is supplied.",
                "enum": ["DAY"], "type": "string"},
        "order_ref": ref,
        "sessions": {"description": "Required only for advance: unquoted positive whole JSON integer <= account.sessions_remaining. Simulation clock movement, not an order.",
                     "minimum": 1, "type": "integer"},
    }
    one_of = []
    for c, fields in FIELDS.items():
        p = {k: (dict(v, enum=[c]) if k == "command" else v) for k, v in props.items() if k in fields}
        order = ["command", "symbol", "count", "conid", "side", "quantity", "order_type", "limit", "tif", "order_ref", "sessions"]
        req = {"submit-stock-order": ["command", "conid", "symbol", "side", "quantity", "order_type", "limit", "tif", "order_ref"],
               "historical-daily": ["command", "symbol", "count"], "cancel-order": ["command", "order_ref"],
               "advance": ["command", "sessions"]}.get(c, ["command"])
        one_of.append({"additionalProperties": False, "properties": {k: p[k] for k in order if k in p},
                       "required": req, "type": "object"})
    desc = ("Execute an explicitly declared retrospective portfolio simulator command. This tool never connects to IBKR. "
            "Call functions.pa_tws with a JSON object in the native tool channel. Do not emit shell strings. "
            + " ".join(f"{c}: {SEMANTICS[c]}" for c in FIELDS)
            + " An ordinary final response also commits the current strategy for terminal NAV marking; already queued orders remain effective, and no order is invented.")
    params = {
        "additionalProperties": False,
        "description": "Exact command field sets: account(command); historical-daily(command, symbol, count); submit-stock-order(command, conid, symbol, side, quantity, order_type, limit, tif, order_ref); cancel-order(command, order_ref); advance(command, sessions); finish(command). Omit every other field.",
        "properties": props, "required": ["command"], "type": "object", "oneOf": one_of,
    }
    return {"name": "pa_tws", "description": desc, "parameters": params}


SEMANTICS = {
    "account": "Read current cash, marked NAV, positions, queued orders, actual simulated fills and sessions_remaining. Does not advance time.",
    "historical-daily": "Read the requested symbol's last count completed daily bars through the current session, inclusive. count is an unquoted whole JSON integer 1..252. Does not advance time or expose future bars.",
    "submit-stock-order": "Queue exactly the model's complete stock order for the next session open. conid must match symbol in verified_contracts. quantity is an unquoted positive whole JSON integer. Model supplies order_type=LMT and tif=DAY explicitly. BUY requires cash and SELL requires owned, unreserved shares; shorting is unsupported. At the next open, a BUY fills only when adverse-adjusted open <= limit; a SELL fills only when adverse-adjusted open >= limit, within disclosed account limits. Otherwise the DAY order expires. No intraday range-touch or partial fills. Submission itself does not advance time.",
    "cancel-order": "Cancel one queued, unfilled simulation order by its original unique order_ref. Cannot cancel a past fill. Does not advance time.",
    "advance": "Simulation clock command only: advance sessions trading sessions, an unquoted positive whole JSON integer no greater than account.sessions_remaining. Execute already queued next-open orders and expose observations at the new current session. Creates no orders and does not select investments.",
    "finish": "Simulation terminal clock command only: advance all remaining sessions, settle already queued orders, mark terminal NAV and end the episode. Creates no entry/exit orders and does not liquidate remaining inventory.",
}


def declaration(horizon):
    spec = tool_spec()["parameters"]
    schema = {k: spec[k] for k in ("additionalProperties", "description", "oneOf", "properties", "required", "type")}
    return {
        "commands": {"account": {}, "advance": {"sessions": "positive integer within remaining horizon"},
                     "cancel-order": {"order_ref": "original model order ref"}, "finish": {},
                     "historical-daily": {"count": "1..252", "symbol": "declared symbol"},
                     "submit-stock-order": {"conid": "verified contract id", "limit": "positive USD decimal", "order_ref": "unique string",
                                            "order_type": "LMT", "quantity": "positive integer", "side": "BUY|SELL",
                                            "symbol": "declared symbol", "tif": "DAY"}},
        "environment": "RETROSPECTIVE_SIMULATION_ONLY",
        "fills": "DAY LIMIT at next-session adverse-adjusted open only; rejected, never clipped; no range-touch fills",
        "horizon_sessions": horizon, "interface": "pa_tws",
        "limits_and_costs": {"commission_per_share_usd": "0.005", "history_window": 60, "initial_cash_usd": "10000",
                             "max_gross_notional_usd": "9000", "max_order_notional_usd": "2500", "max_orders_per_session": 4,
                             "max_positions": 4, "max_symbol_notional_usd": "5000", "min_cash_usd": "500",
                             "minimum_commission_usd": "1", "slippage_bps": "5", "spread_bps": "5"},
        "observations": "completed past daily bars only; original publication vintage unverified",
        "public_api_contract": {
            "argument_encoding": "Exact JSON object. Integer fields are unquoted whole JSON integers; prices may be finite decimal strings or numbers. No coercion, defaults or extra fields.",
            "argument_schema": schema, "command_semantics": SEMANTICS,
            "native_cli_transfer": "submit-stock-order uses native CLI field names and order_ref syntax ^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$ is aligned with the native CLI; supplied model references are rejected when invalid, never rewritten. Native historical-daily requires conid and primary_exchange and uses duration/as_of instead of count. Native cancellation is cancel-pa-order with order_id/order_ref/conid. Simulation account bundles reads that the real CLI exposes separately. advance and finish are simulation clocks unavailable in the real broker CLI. No simulation call is translated into a live or paper order. Simulation competence is not evidence of a native broker lifecycle.",
            "termination": "Final response commits current strategy. Terminal valuation advances the remaining clock and settles already queued orders; it never generates liquidation or other orders.",
        },
        "qualification": QUALIFICATION, "symbols": list(SYMBOLS),
        "terminal": "marked NAV; no forced liquidation; residual inventory/orders reported separately",
        "verified_contracts": [{"conid": CONIDS[s], "symbol": s} for s in SYMBOLS],
    }


def system_content(horizon):
    return SYSTEM + "\nExecution constraints supplied by the host:\n" + canonical(declaration(horizon))


class Host:
    """One episode: start session index, horizon in sessions."""

    def __init__(self, bars, start, horizon, sell_cap_bug=False):
        self.bars = bars
        self.dates = [b["session"] for b in bars[SYMBOLS[0]]]
        self.by = {s: {b["session"]: b for b in bars[s]} for s in SYMBOLS}
        self.i = self.dates.index(start) if isinstance(start, str) else start
        self.end = self.i + horizon
        if self.end >= len(self.dates):
            raise ValueError("horizon runs past the data")
        self.horizon, self.sell_cap_bug = horizon, sell_cap_bug
        self.cash, self.pos, self.queue, self.fills, self.refs = CASH0, {}, [], [], set()
        self.submitted = {}          # session -> count
        self.dropped = []            # orders refused at execution (not visible to the model, as in Codex)
        self.expired = []
        self.model_advanced = 0      # sessions advanced by the model (advance/finish) vs the evaluator
        self.finished = False        # set only by finish, as in Codex (advance to the end leaves it False)

    # ----- views -----
    @property
    def session(self):
        return self.dates[self.i]

    def close(self, s):
        return dec(self.by[s][self.session]["close"])

    def nav(self):
        v = Decimal("0.00")
        for s in sorted(self.pos):
            v += self.pos[s] * self.close(s)
        return self.cash + v

    def reserved(self):
        r = Decimal("0")
        for o in self.queue:
            if o["side"] == "BUY":
                r += o["quantity"] * dec(o["limit"]) + commission(o["quantity"])
        return r

    def account(self):
        return {"cash_usd": str(self.cash), "environment": "SIMULATION", "episode_done": self.finished,
                "nav_usd": str(self.nav()), "open_orders": [dict(o) for o in self.queue],
                "owned_fills": list(self.fills),
                "positions": [{"mark_usd": str(self.close(s)), "quantity": q, "symbol": s} for s, q in sorted(self.pos.items())],
                "reserved_buy_cash_usd": str(self.reserved()), "session": self.session,
                "sessions_remaining": self.end - self.i}

    def history(self, symbol, count):
        rows = [self.by[symbol][d] for d in self.dates[max(0, self.i + 1 - count): self.i + 1]]
        return {"as_of_session": self.session,
                "bars": [{"close": b["close"], "high": b["high"], "low": b["low"], "open": b["open"],
                          "session": b["session"], "volume": b["volume"]} for b in rows],
                "ok": True, "qualification": QUALIFICATION, "symbol": symbol}

    def initial_observation(self):
        return {"account": self.account(), "history": {s: self.history(s, 20) for s in sorted(SYMBOLS)}}

    # ----- commands -----
    def execute(self, args):
        """The tool result the model sees, wrapped as Codex wrapped it."""
        result = self._execute(args)
        return {"arguments": args, "result": result, "session": self.session}

    def _execute(self, a):
        if not isinstance(a, dict) or a.get("command") not in FIELDS or set(a) != FIELDS[a["command"]]:
            return {"ok": False, "error": "unsupported command or fields"}
        c = a["command"]
        if c == "account":
            return {**self.account(), "ok": True}
        if c == "historical-daily":
            if a["symbol"] not in SYMBOLS or not is_int(a["count"]) or not 1 <= a["count"] <= 252:
                return {"ok": False, "error": "unsupported command or fields"}
            return self.history(a["symbol"], a["count"])
        if c == "submit-stock-order":
            return self._submit(a)
        if c == "cancel-order":
            for o in self.queue:
                if o["order_ref"] == a["order_ref"]:
                    self.queue.remove(o)
                    return {"ok": True, "order_ref": a["order_ref"], "status": "CANCELLED"}
            return {"ok": False, "error": "unknown queued order_ref"}
        if c == "advance":
            n = a["sessions"]
            if not is_int(n) or n < 1 or n > self.end - self.i:
                return {"ok": False, "error": "unsupported command or fields"}
            self._advance(n, model=True)
            return {**self.account(), "ok": True}
        if c == "finish":
            self._advance(self.end - self.i, model=True)
            self.finished = True
            return {**self.account(), "ok": True}

    def _submit(self, a):
        s, side, q, lim, ref = a["symbol"], a["side"], a["quantity"], a["limit"], a["order_ref"]
        if (s not in SYMBOLS or a["conid"] != CONIDS[s] or side not in ("BUY", "SELL") or not is_int(q) or q < 1
                or a["order_type"] != "LMT" or a["tif"] != "DAY"):
            return {"ok": False, "error": "unsupported command or fields"}
        try:
            price = dec(lim) if isinstance(lim, (str, int, float)) and not isinstance(lim, bool) else None
        except Exception:
            price = None
        if (price is None or not price.is_finite() or price <= 0 or not isinstance(ref, str) or not REF.match(ref)
                or not 8 <= len(ref) <= 64 or ref in self.refs):
            return {"ok": False, "error": "positive limit and new unique canonical native PA order_ref required"}
        if self.i >= self.end:
            return {"ok": False, "error": "NO_FUTURE_EXECUTION_SESSION"}
        if self.submitted.get(self.session, 0) >= MAX_ORDERS_PER_SESSION:
            return {"ok": False, "error": "MAX_ORDER_RATE"}
        err = self._capacity(s, side, q, price, admission=True)
        if err:
            return {"ok": False, "error": err}
        order = {**a, "submitted_session": self.session}
        self.queue.append(order)
        self.refs.add(ref)
        self.submitted[self.session] = self.submitted.get(self.session, 0) + 1
        return {"ok": True, "order": order, "status": "QUEUED_NEXT_OPEN"}

    def _capacity(self, s, side, q, price, admission, exclude=None):
        notional = q * price
        if side == "SELL":
            if self.sell_cap_bug and notional > MAX_ORDER:
                return "MAX_ORDER_NOTIONAL"
            pending = sum(o["quantity"] for o in self.queue if o["symbol"] == s and o["side"] == "SELL" and o is not exclude)
            if q + (pending if admission else 0) > self.pos.get(s, 0):
                return "INSUFFICIENT_UNRESERVED_OWNED_INVENTORY"
            return None
        if notional > MAX_ORDER:
            return "MAX_ORDER_NOTIONAL"
        buys = [o for o in self.queue if o["side"] == "BUY" and o is not exclude] if admission else []
        reserved = sum(o["quantity"] * dec(o["limit"]) + commission(o["quantity"]) for o in buys)
        if self.cash - reserved - notional - commission(q) < MIN_CASH:
            return "CASH_OR_MINIMUM_CASH"
        sym = self.pos.get(s, 0) * self.close(s) + sum(o["quantity"] * dec(o["limit"]) for o in buys if o["symbol"] == s)
        if sym + notional > MAX_SYMBOL:
            return "MAX_SYMBOL_NOTIONAL"
        gross = sum(qq * self.close(ss) for ss, qq in self.pos.items()) + sum(o["quantity"] * dec(o["limit"]) for o in buys)
        if gross + notional > MAX_GROSS:
            return "MAX_GROSS_NOTIONAL"
        held = set(self.pos) | {o["symbol"] for o in buys}
        if s not in held and len(held) >= MAX_POSITIONS:
            return "MAX_POSITIONS"
        return None

    def _advance(self, n, model=False):
        for _ in range(n):
            if self.i >= self.end:
                return
            self.i += 1
            if model:
                self.model_advanced += 1
            queue, self.queue = self.queue, []
            for o in queue:                       # fill or expire every queued DAY order at this open
                s, q = o["symbol"], o["quantity"]
                op = dec(self.by[s][self.session]["open"])
                px = op * (UP if o["side"] == "BUY" else DOWN)
                ok = px <= dec(o["limit"]) if o["side"] == "BUY" else px >= dec(o["limit"])
                if not ok:
                    self.expired.append(o["order_ref"])
                    continue
                err = self._capacity_at_fill(s, o["side"], q, px)
                if err:
                    self.dropped.append({"order_ref": o["order_ref"], "reason": err, "session": self.session})
                    continue
                fee = commission(q)
                if o["side"] == "BUY":
                    self.cash -= q * px + fee
                    self.pos[s] = self.pos.get(s, 0) + q
                else:
                    self.cash += q * px - fee
                    self.pos[s] -= q
                    if not self.pos[s]:
                        del self.pos[s]
                self.fills.append({"commission_usd": str(fee), "order_ref": o["order_ref"], "price_usd": str(px),
                                   "quantity": q, "session": self.session, "side": o["side"], "simulated": True, "symbol": s})

    def _capacity_at_fill(self, s, side, q, px):
        if side == "SELL":
            if self.sell_cap_bug and q * px > MAX_ORDER:
                return "MAX_ORDER_NOTIONAL"
            return None if q <= self.pos.get(s, 0) else "INSUFFICIENT_UNRESERVED_OWNED_INVENTORY"
        notional = q * px
        if notional > MAX_ORDER:
            return "MAX_ORDER_NOTIONAL"
        prev = self.dates[self.i - 1]
        mark = lambda ss: dec(self.by[ss][prev]["close"])
        if self.pos.get(s, 0) * mark(s) + notional > MAX_SYMBOL:
            return "MAX_SYMBOL_NOTIONAL"
        if sum(qq * mark(ss) for ss, qq in self.pos.items()) + notional > MAX_GROSS:
            return "MAX_GROSS_NOTIONAL"
        if self.cash - notional - commission(q) < MIN_CASH:
            return "CASH_OR_MINIMUM_CASH"
        return None

    def terminal(self):
        """Codex's terminal valuation: the evaluator advances the rest of the horizon, holding inventory."""
        evaluator = self.end - self.i
        self._advance(evaluator)
        self.finished = True
        return evaluator


def commission(q):
    return max(ONE, PER_SHARE * q)


# ----- outcome accounting (my additions: what Codex never reported) -----
def outcome(host):
    """Marked return (Codex's metric), realized vs unrealized P&L, and return after liquidating at the
    horizon close with the same costs (what an exit would have cost)."""
    nav = host.nav()
    realized, cost = Decimal("0"), {}
    qty = {}
    for f in host.fills:
        s, q, px, fee = f["symbol"], f["quantity"], dec(f["price_usd"]), dec(f["commission_usd"])
        realized -= fee
        if f["side"] == "BUY":
            c, n = cost.get(s, Decimal("0")), qty.get(s, 0)
            cost[s], qty[s] = c + q * px, n + q
        else:
            avg = cost[s] / qty[s]
            realized += q * (px - avg)
            cost[s] -= q * avg
            qty[s] -= q
    liq = host.cash
    for s, q in host.pos.items():
        px = host.close(s) * DOWN
        liq += q * px - commission(q)
    unrealized = nav - CASH0 - realized
    return {"marked_return_pct": float((nav - CASH0) / CASH0 * 100), "realized_usd": float(realized),
            "unrealized_usd": float(unrealized), "liquidated_return_pct": float((liq - CASH0) / CASH0 * 100),
            "fills_buy": sum(f["side"] == "BUY" for f in host.fills), "fills_sell": sum(f["side"] == "SELL" for f in host.fills),
            "residual_positions": {s: q for s, q in host.pos.items()}, "dropped": host.dropped,
            "closed_flat": not host.pos and bool(host.fills), "model_advanced_sessions": host.model_advanced}
