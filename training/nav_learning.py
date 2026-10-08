"""Outcome learning from an explicit retrospective market simulation.

No import performs network I/O. There are no trade targets, role routers,
technical signal rules or rewards for activity/abstention. Only model-generated
orders alter the book. Historical captures remain developer-exposed retrieval
vintage data; simulated NAV is not real broker PnL or a blind alpha claim.
"""

from __future__ import annotations

import argparse

from copy import deepcopy

from dataclasses import asdict, dataclass

from datetime import date, datetime, time, timedelta, timezone

from decimal import Decimal, InvalidOperation

import hashlib

import json

from pathlib import Path

import re

from typing import Any

from zoneinfo import ZoneInfo

HORIZONS = (1, 5, 20, 63)

PARENT_LORA_TARGETS = {"train_attn": True, "train_mlp": True, "train_unembed": False}

QUALIFICATION = {
    "kind": "DEVELOPER_EXPOSED_RETROSPECTIVE_PRICE_SIMULATION",
    "historical_point_in_time_proven": False,
    "blind_evaluation": False,
    "real_broker_fills": False,
    "price_basis": "retrieved_split_adjusted_non_dividend_TRADES",
    "limitations": ["Current-listed universe has survivorship bias.",
                    "Historical correction/adjustment and original publication vintages are unknown.",
                    "Dividend cashflows and corporate actions are not reconstructed.",
                    "Next-open daily fills omit queues, intraday liquidity and partial fills.",
                    "Previously used historical evaluation windows are development exposed."]}

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()

def reference(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def write_once(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")

def number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a monetary number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("finite decimal required") from exc
    if not result.is_finite():
        raise ValueError("finite decimal required")
    return result

def money(value: Decimal) -> str:
    return format(value, "f")

@dataclass(frozen=True)
class SimulationConfig:
    initial_cash_usd: str = "10000"
    max_order_notional_usd: str = "2500"
    max_symbol_notional_usd: str = "5000"
    max_gross_notional_usd: str = "9000"
    min_cash_usd: str = "500"
    max_positions: int = 4
    max_orders_per_session: int = 4
    commission_per_share_usd: str = "0.005"
    minimum_commission_usd: str = "1"
    spread_bps: str = "5"
    slippage_bps: str = "5"
    history_window: int = 60

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in {"max_positions", "max_orders_per_session", "history_window"}:
                if type(value) is not int or value < 1:
                    raise ValueError("positive integer configuration required: " + key)
            elif number(value) < 0:
                raise ValueError("nonnegative monetary configuration required: " + key)
        if number(self.initial_cash_usd) <= 0 or self.history_window > 252:
            raise ValueError("positive initial NAV and bounded history required")
        if number(self.spread_bps) / 2 + number(self.slippage_bps) >= 10000:
            raise ValueError("transaction friction exceeds price")

def load_histories(path: Path, *, expected_sha256: str | None = None,
                   symbols: list[str] | None = None) -> dict:
    """Read/hash/parse identical bytes; no references embedded in data are opened."""
    raw = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("immutable price source changed")
    histories = json.loads(raw)
    if not isinstance(histories, dict) or not histories:
        raise ValueError("symbol-keyed normalized daily source required")
    selected = sorted(symbols or histories)
    if len(selected) != len(set(selected)) or not set(selected) <= set(histories):
        raise ValueError("unique source symbols required")
    result = {}
    for symbol in selected:
        item = histories[symbol]
        if item.get("symbol") != symbol or item.get("currency") != "USD":
            raise ValueError("USD source identity differs")
        bars = item["bars"]
        dates = [bar["session"] for bar in bars]
        if dates != sorted(set(dates)):
            raise ValueError("duplicate or unordered source dates")
        for bar in bars:
            datetime.strptime(bar["session"], "%Y-%m-%d")
            prices = [number(bar[key]) for key in ("open", "high", "low", "close")]
            if min(prices) <= 0 or prices[1] < max(prices) or prices[2] > min(prices):
                raise ValueError("invalid source OHLC")
        result[symbol] = deepcopy(item)
    return result

def common_sessions(histories: dict) -> list[str]:
    dates = set.intersection(*({bar["session"] for bar in row["bars"]} for row in histories.values()))
    return sorted(dates)

def prepare_dataset(source: Path, output: Path, *, symbols: list[str] | None = None,
                    config: SimulationConfig | None = None, embargo_sessions: int = 5,
                    rights_attestation: Path | None = None) -> dict:
    """Freeze chronological development partitions, all horizons grouped by start.

    Entire outcome windows must lie inside a partition. The 63-session purge is
    applied to every horizon so related short/long episodes cannot cross splits.
    Future prices/rewards are never stored in the episode descriptor/prompt.
    """
    if type(embargo_sessions) is not int or embargo_sessions < 0:
        raise ValueError("nonnegative embargo required")
    config = config or SimulationConfig()
    source_ref = reference(source)
    histories = load_histories(source, expected_sha256=source_ref["sha256"], symbols=symbols)
    authority = None
    if rights_attestation is not None:
        raw = Path(rights_attestation).read_bytes()
        authority = json.loads(raw)
        if authority.get("remote_training_permitted") is not True or authority.get("source") != "DIRECT_PRINCIPAL_ATTESTATION":
            raise ValueError("direct principal remote-use attestation required")
    sessions = common_sessions(histories)
    if len(sessions) < config.history_window + 3 * max(HORIZONS) + 2 * embargo_sessions + 15:
        raise ValueError("insufficient chronology for purged train/validation/test")
    train_end, validation_end = int(len(sessions) * .55), int(len(sessions) * .78)
    episodes = []
    for index in range(config.history_window - 1, len(sessions) - max(HORIZONS)):
        if index + max(HORIZONS) <= train_end:
            split = "train"
        elif train_end + embargo_sessions < index and index + max(HORIZONS) <= validation_end:
            split = "validation"
        elif index > validation_end + embargo_sessions:
            split = "test"
        else:
            continue
        for horizon in HORIZONS:
            episodes.append({"episode_id": f"{sessions[index]}-h{horizon}",
                             "group_id": sessions[index], "start_session": sessions[index],
                             "horizon_sessions": horizon, "end_session": sessions[index + horizon],
                             "split": split})
    if any(not any(row["split"] == split for row in episodes) for split in ("train", "validation", "test")):
        raise ValueError("each chronological partition needs at least one episode")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "episodes.json", episodes)
    manifest = {"kind": "nav_learning_dataset_v1", "source": source_ref,
                "code": reference(Path(__file__)), "symbols": sorted(histories),
                "config": asdict(config), "sessions": sessions,
                "episode_reference": reference(output / "episodes.json"),
                "partition": {"train_end": sessions[train_end], "validation_end": sessions[validation_end],
                              "purge_sessions": max(HORIZONS), "embargo_sessions": embargo_sessions,
                              "group_all_horizons_by_start": True,
                              "counts": {s: sum(e["split"] == s for e in episodes)
                                         for s in ("train", "validation", "test")}},
                "qualification": QUALIFICATION, "teacher_trade_labels": False,
                "remote_use_authority": {"reference": reference(rights_attestation), "attestation": authority,
                                         "old_source_rights_annotations_preserved": True} if authority else None,
                "future_outcomes_in_episode_prompts": False, "provider_calls": 0,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_once(output / "manifest.json", manifest)
    return manifest

class PortfolioSimulationHost:
    """A declared pa_tws-shaped simulator; no account/broker connection exists.

    All actions use exact generated arguments. DAY LIMIT orders execute only at
    the next session's adverse-cost-adjusted open if the model's limit admits it.
    Admission uses visible close/limit and is rechecked at execution. Orders are
    rejected rather than clipped. Reducing sales remain available when a market
    move lifts existing inventory above the entry exposure limits.
    """

    def __init__(self, histories: dict, episode: dict, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.histories = deepcopy(histories)
        self.sessions = common_sessions(histories)
        self.start = self.sessions.index(episode["start_session"])
        self.horizon = episode["horizon_sessions"]
        if self.horizon not in HORIZONS or self.start + self.horizon >= len(self.sessions):
            raise ValueError("complete declared horizon required")
        if episode.get("end_session", self.sessions[self.start + self.horizon]) != self.sessions[self.start + self.horizon]:
            raise ValueError("episode terminal date differs from chronology")
        self.cursor = self.start
        self.end = self.start + self.horizon
        self.initial_nav = number(self.config.initial_cash_usd)
        self.cash = self.initial_nav
        self.inventory = {s: 0 for s in histories}
        self.contract_ids = {s: row.get("contract_id", row.get("conid")) for s, row in histories.items()}
        if any(type(value) is not int or value <= 0 for value in self.contract_ids.values()):
            raise ValueError("verified source contract ids required")
        self.bars = {s: {b["session"]: b for b in row["bars"]} for s, row in histories.items()}
        self.orders: list[dict] = []
        self.order_keys: set[str] = set()
        self.orders_this_session = 0
        self.fills: list[dict] = []
        self.receipts: list[dict] = []
        self.external_flows: list[dict] = []
        self.total_commission = Decimal(0)
        self.total_price_friction = Decimal(0)
        self.done = False

    def declaration(self) -> dict:
        from training.nav_tool_schema import public_api_contract
        declaration = {"interface": "pa_tws", "environment": "RETROSPECTIVE_SIMULATION_ONLY",
                "commands": {"account": {}, "historical-daily": {"symbol": "declared symbol", "count": "1..252"},
                             "submit-stock-order": {"conid": "verified contract id", "symbol": "declared symbol", "side": "BUY|SELL",
                                       "quantity": "positive integer", "order_type": "LMT", "limit": "positive USD decimal",
                                       "tif": "DAY", "order_ref": "unique string"},
                             "cancel-order": {"order_ref": "original model order ref"},
                             "advance": {"sessions": "positive integer within remaining horizon"}, "finish": {}},
                "symbols": sorted(self.histories), "horizon_sessions": self.horizon,
                "verified_contracts": [{"symbol": s, "conid": self.contract_ids[s]} for s in sorted(self.histories)],
                "limits_and_costs": asdict(self.config),
                "fills": "DAY LIMIT at next-session adverse-adjusted open only; rejected, never clipped; no range-touch fills",
                "terminal": "marked NAV; no forced liquidation; residual inventory/orders reported separately",
                "observations": "completed past daily bars only; original publication vintage unverified",
                "qualification": QUALIFICATION}
        declaration["public_api_contract"] = public_api_contract(declaration)
        return declaration

    def _mark(self, symbol: str) -> Decimal:
        return number(self.bars[symbol][self.sessions[self.cursor]]["close"])

    def _nav(self) -> Decimal:
        return self.cash + sum((number(q) * self._mark(s) for s, q in self.inventory.items()), Decimal(0))

    def account(self) -> dict:
        reserved_cash = sum((number(o["quantity"]) * number(o["limit"]) + self._fee(o["quantity"])
                             for o in self.orders if o["side"] == "BUY"), Decimal(0))
        return {"environment": "SIMULATION", "session": self.sessions[self.cursor],
                "cash_usd": money(self.cash), "nav_usd": money(self._nav()),
                "reserved_buy_cash_usd": money(reserved_cash),
                "positions": [{"symbol": s, "quantity": q, "mark_usd": money(self._mark(s))}
                              for s, q in self.inventory.items() if q],
                "open_orders": deepcopy(self.orders), "owned_fills": deepcopy(self.fills),
                "sessions_remaining": self.end - self.cursor, "episode_done": self.done}

    def _fee(self, quantity: int) -> Decimal:
        return max(number(self.config.minimum_commission_usd), number(quantity) * number(self.config.commission_per_share_usd))

    def _capacity_error(self, symbol: str, side: str, quantity: int, price: Decimal,
                        *, include_pending: bool, execution_open_prices: dict | None = None) -> str | None:
        if quantity * price > number(self.config.max_order_notional_usd):
            return "MAX_ORDER_NOTIONAL"
        orders = self.orders if include_pending else []
        sells = sum(o["quantity"] for o in orders if o["symbol"] == symbol and o["side"] == "SELL")
        if side == "SELL":
            return "INSUFFICIENT_UNRESERVED_OWNED_INVENTORY" if quantity + sells > self.inventory[symbol] else None
        reserved_cash = sum((number(o["quantity"]) * number(o["limit"]) + self._fee(o["quantity"])
                             for o in orders if o["side"] == "BUY"), Decimal(0))
        if self.cash - reserved_cash - quantity * price - self._fee(quantity) < number(self.config.min_cash_usd):
            return "CASH_OR_MINIMUM_CASH"
        pending_by_symbol = {s: sum(number(o["quantity"]) * number(o["limit"])
                                    for o in orders if o["symbol"] == s and o["side"] == "BUY") for s in self.inventory}
        exposures = {s: number(q) * (execution_open_prices[s] if execution_open_prices is not None else self._mark(s))
                     + pending_by_symbol[s] for s, q in self.inventory.items()}
        exposures[symbol] += quantity * price
        if exposures[symbol] > number(self.config.max_symbol_notional_usd):
            return "MAX_SYMBOL_NOTIONAL"
        if sum(exposures.values(), Decimal(0)) > number(self.config.max_gross_notional_usd):
            return "MAX_GROSS_NOTIONAL"
        if sum(x > 0 for x in exposures.values()) > self.config.max_positions:
            return "MAX_POSITIONS"
        return None

    def execute(self, arguments: dict) -> dict:
        """Visible local command failures are observations, never replacement trades."""
        try:
            result = self._execute(arguments)
        except (ValueError, KeyError, TypeError) as exc:
            result = {"ok": False, "error": str(exc)}
        receipt = {"arguments": deepcopy(arguments), "result": result, "session": self.sessions[self.cursor]}
        self.receipts.append(deepcopy(receipt))
        return receipt

    def _execute(self, args: dict) -> dict:
        if type(args) is not dict or type(args.get("command")) is not str:
            raise ValueError("one exact command object required")
        command = args["command"]
        fields = {"account": {"command"}, "historical-daily": {"command", "symbol", "count"},
                  "submit-stock-order": {"command", "conid", "symbol", "side", "quantity", "order_type", "limit", "tif", "order_ref"},
                  "cancel-order": {"command", "order_ref"}, "advance": {"command", "sessions"}, "finish": {"command"}}
        if command not in fields or set(args) != fields[command]:
            raise ValueError("unsupported command or fields")
        if command == "account":
            return {"ok": True, **self.account()}
        if self.done:
            raise ValueError("episode already finished")
        if command == "historical-daily":
            symbol, count = args["symbol"], args["count"]
            if symbol not in self.histories or type(count) is not int or not 1 <= count <= 252:
                raise ValueError("declared symbol and count1..252 required")
            date = self.sessions[self.cursor]
            # Raw source fields and future bars never enter this response.
            bars = [{key: deepcopy(bar[key]) for key in ("session", "open", "high", "low", "close", "volume") if key in bar}
                    for bar in self.histories[symbol]["bars"] if bar["session"] <= date][-count:]
            return {"ok": True, "symbol": symbol, "as_of_session": date, "bars": bars,
                    "qualification": QUALIFICATION}
        if command == "submit-stock-order":
            from training.nav_tool_schema import ORDER_REF_PATTERN
            s, side, q, ref = args["symbol"], args["side"], args["quantity"], args["order_ref"]
            if s not in self.inventory or side not in {"BUY", "SELL"} or type(q) is not int or q <= 0:
                raise ValueError("declared symbol, BUY/SELL and positive whole shares required")
            if (type(args["conid"]) is not int or args["conid"] != self.contract_ids[s]
                    or args["order_type"] != "LMT" or args["tif"] != "DAY"):
                raise ValueError("exact verified conid and explicit model LMT/DAY terms required")
            price = number(args["limit"])
            if price <= 0 or not isinstance(ref, str) or re.fullmatch(ORDER_REF_PATTERN, ref) is None or ref in self.order_keys:
                raise ValueError("positive limit and new unique canonical native PA order_ref required")
            if self.cursor == self.end:
                raise ValueError("no future execution session remains")
            if self.orders_this_session >= self.config.max_orders_per_session:
                raise ValueError("MAX_ORDER_RATE")
            error = self._capacity_error(s, side, q, price, include_pending=True)
            if error:
                raise ValueError(error)
            order = deepcopy(args)
            order.update(submitted_session=self.sessions[self.cursor])
            self.orders.append(order)
            self.order_keys.add(ref)
            self.orders_this_session += 1
            return {"ok": True, "status": "QUEUED_NEXT_OPEN", "order": deepcopy(order)}
        if command == "cancel-order":
            matches = [o for o in self.orders if o["order_ref"] == args["order_ref"]]
            if len(matches) != 1:
                raise ValueError("one queued owned order required")
            self.orders.remove(matches[0])
            return {"ok": True, "status": "CANCELLED", "order_ref": args["order_ref"]}
        if command == "advance":
            count = args["sessions"]
            if type(count) is not int or not 1 <= count <= self.end - self.cursor:
                raise ValueError("positive count within horizon required")
            self._advance(count)
            return {"ok": True, **self.account()}
        self._advance(self.end - self.cursor)
        self.done = True
        return {"ok": True, **self.account()}

    def _advance(self, count: int) -> None:
        for _ in range(count):
            self.cursor += 1
            self.orders_this_session = 0
            queued, self.orders = self.orders, []
            execution_open_prices = {s: number(self.bars[s][self.sessions[self.cursor]]["open"]) for s in self.inventory}
            for order in queued:
                s, side, q = order["symbol"], order["side"], order["quantity"]
                raw_open = number(self.bars[s][self.sessions[self.cursor]]["open"])
                friction = (number(self.config.spread_bps) / 2 + number(self.config.slippage_bps)) / 10000
                price = raw_open * (1 + friction if side == "BUY" else 1 - friction)
                crosses = price <= number(order["limit"]) if side == "BUY" else price >= number(order["limit"])
                error = self._capacity_error(s, side, q, price, include_pending=False,
                                             execution_open_prices=execution_open_prices) if crosses else "LIMIT_NOT_MARKETABLE_AT_OPEN"
                # Risk check above must value incumbent inventory at known open,
                # never the close later in this same execution session.
                if error:
                    self.receipts.append({"session": self.sessions[self.cursor], "order_ref": order["order_ref"],
                                          "status": "EXPIRED_DAY_UNFILLED", "reason": error})
                    continue
                fee = self._fee(q)
                self.cash += -(q * price + fee) if side == "BUY" else q * price - fee
                self.inventory[s] += q if side == "BUY" else -q
                self.total_commission += fee
                self.total_price_friction += q * abs(price - raw_open)
                self.fills.append({"order_ref": order["order_ref"], "symbol": s, "side": side, "quantity": q,
                                   "session": self.sessions[self.cursor], "price_usd": money(price), "commission_usd": money(fee),
                                   "simulated": True})

    def terminal_result(self) -> dict:
        """Sealed evaluator valuation after the model stops; no fabricated exit."""
        unfinished = self.end - self.cursor
        self._advance(unfinished)
        self.done = True
        nav = self._nav()
        flow = sum((number(row["amount_usd"]) for row in self.external_flows), Decimal(0))
        reward = (nav - self.initial_nav - flow) / self.initial_nav
        return {"kind": "simulated_costed_nav_reward_v1", "reward": float(reward),
                "reward_formula": "(terminal_marked_NAV - initial_NAV - net_external_cashflows) / initial_NAV",
                "horizon_sessions": self.horizon, "initial_nav_usd": money(self.initial_nav),
                "terminal_nav_usd": money(nav), "net_external_flows_usd": money(flow),
                "commission_usd": money(self.total_commission), "price_friction_usd": money(self.total_price_friction),
                "terminal_account": self.account(), "valuation_only_automatic_time_advance_sessions": unfinished,
                "forced_liquidations": 0, "closed_inventory": all(q == 0 for q in self.inventory.values()),
                "residual_orders": len(self.orders), "qualification": QUALIFICATION,
                "fees_included_in_reward": True, "reward_activity_or_HOLD_bonus": 0}

    def record_external_flow(self, amount_usd: str, *, reference_id: str) -> None:
        """Evaluator cashflow evidence, never an agent tool/reward shortcut."""
        if self.done or not isinstance(reference_id, str) or not reference_id or any(
                row["reference_id"] == reference_id for row in self.external_flows):
            raise ValueError("new evidence-backed flow reference required before terminal")
        amount = number(amount_usd)
        if self.cash + amount < 0:
            raise ValueError("flow exceeds cash")
        self.cash += amount
        self.external_flows.append({"reference_id": reference_id, "amount_usd": money(amount),
                                    "session": self.sessions[self.cursor]})

def group_advantages(rewards: list[float]) -> list[float]:
    """Same-start/horizon on-policy rewards only, with no teacher action label."""
    if len(rewards) < 2:
        raise ValueError("at least two trajectories per group required")
    values = [number(r) for r in rewards]
    mean = sum(values, Decimal(0)) / len(values)
    return [float(value - mean) for value in values]

def admissible_group_advantages(outcomes: list[dict]) -> list[float | None]:
    """Retain failures, but do not promote malformed cash outcomes as winners.

    A bounded STEP_LIMIT with exact accepted actions is still a committed partial
    policy for economic valuation; its operational completion remains false.
    Provider/raw/parse/unknown host failures never enter advantage centering.
    """
    eligible = [i for i, row in enumerate(outcomes) if row["runtime_status"] in {"FINAL", "STEP_LIMIT"}]
    result = [None] * len(outcomes)
    if len(eligible) >= 2:
        for i, value in zip(eligible, group_advantages([outcomes[i]["reward"] for i in eligible])):
            result[i] = value
    return result

def trajectory_datums(steps: list[dict], advantage: float) -> list:
    """Actual native generated tokens get gradient; observations get zero.

    Steps contain the actual generation prompt tokens, generated tokens and
    corresponding sampled logprobs. No decoded text is re-tokenized/repaired.
    Full output, including native analysis/tool/final markers, remains on-policy.
    """
    import tinker
    result = []
    for step in steps:
        prompt, generated, logprobs = step["prompt_tokens"], step["tokens"], step["logprobs"]
        if not prompt or not generated or len(generated) != len(logprobs) or len(prompt) + len(generated) > 32768:
            raise ValueError("complete finite actual generation tokens/logprobs required")
        if any(type(t) is not int for t in prompt + generated):
            raise ValueError("native integer tokens required")
        for lp in logprobs:
            number(lp)
        tokens = prompt + generated
        targets = tokens[1:]
        mask = [0.0] * (len(prompt) - 1) + [float(advantage)] * len(generated)
        old_lp = [0.0] * (len(prompt) - 1) + list(logprobs)
        result.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                                  loss_fn_inputs={"target_tokens": tinker.TensorData(data=targets, dtype="int64", shape=[len(targets)]),
                                                  "logprobs": tinker.TensorData(data=old_lp, dtype="float32", shape=[len(targets)]),
                                                  "advantages": tinker.TensorData(data=mask, dtype="float32", shape=[len(targets)])}))
    return result

def datum_wire(datum) -> dict:
    """Stable public SDK datum fields, preserving native tokens and float32 data."""
    return {"model_input_tokens": datum.model_input.to_ints(),
            "loss_fn_inputs": {key: {"data": tensor.data, "dtype": tensor.dtype, "shape": tensor.shape}
                               for key, tensor in datum.loss_fn_inputs.items()}}

def sdk_response_wire(result) -> dict:
    """Persist typed SDK responses without introspecting client credentials."""
    if hasattr(result, "loss_fn_outputs") and hasattr(result, "metrics"):
        return {"loss_fn_output_type": result.loss_fn_output_type, "metrics": result.metrics,
                "loss_fn_outputs": [{key: {"data": tensor.data, "dtype": tensor.dtype, "shape": tensor.shape}
                                      for key, tensor in row.items()} for row in result.loss_fn_outputs]}
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return {"type": type(result).__name__, "model_id": getattr(result, "model_id", None),
            "actual_sampling_session_id": getattr(result, "_sampling_session_id", None)}

def create_compatible_training_client(service, plan):
    """Pass every frozen trainability flag explicitly; SDK defaults differ."""
    if plan["lora_targets"] != PARENT_LORA_TARGETS or plan["rank"] != 8:
        raise ValueError("frozen rank8 attention/MLP trainability required")
    return service.create_lora_training_client(base_model=plan["base_model"], rank=plan["rank"], seed=plan["seed"],
                                               **plan["lora_targets"])

def episode_decision_at(episode: dict) -> str:
    """Declared reconstruction lag: midnight after the completed session."""
    next_day = date.fromisoformat(episode["start_session"]) + timedelta(days=1)
    return datetime.combine(next_day, time(), ZoneInfo("America/New_York")).isoformat()

def validation_directory_name(arm: str, episode: dict) -> str:
    """Each frozen arm, start session and horizon owns a distinct journal."""
    return f"validation-{arm}-{episode['start_session']}-h{episode['horizon_sessions']}"

def uniform_episode_selection(episodes: list[dict], split: str, start_count: int) -> list[dict]:
    """Uniform calendar membership selection; never read prices or outcomes."""
    starts = sorted({row["start_session"] for row in episodes if row["split"] == split})
    if type(start_count) is not int or not 2 <= start_count <= len(starts):
        raise ValueError("at least two distinct admissible chronological starts required")
    indices = [i * (len(starts) - 1) // (start_count - 1) for i in range(start_count)]
    chosen = [starts[index] for index in indices]
    return [next(row for row in episodes if row["split"] == split and row["start_session"] == start
                 and row["horizon_sessions"] == horizon) for start in chosen for horizon in HORIZONS]

class FinitePilotBudget:
    """Local append-only dispatch forecasts. Reservations are not invoices."""
    def __init__(self, directory: Path, plan: dict):
        self.path = directory / "dispatch_budget.jsonl"
        self.plan = plan
        self.calls = 0
        self.joint_tokens = 0
        self.training_tokens = 0
        self.reserved_usd = number(plan["checkpoint_storage_forecast_usd"])
        self.events = 0
        self.path.touch(exist_ok=False)

    def _persist(self, event: dict):
        import os
        self.events += 1
        row = {"event": self.events, **event, "cumulative_reserved_usd": money(self.reserved_usd),
               "sample_calls": self.calls, "joint_sample_tokens": self.joint_tokens,
               "training_input_tokens": self.training_tokens, "forecast_not_invoice": True}
        with self.path.open("a", encoding="utf8", newline="\n") as stream:
            stream.write(canonical(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def sample(self, request: dict):
        tokens = request["input_tokens"] + request["max_output_tokens"]
        charge = number(tokens) * Decimal("0.84") / 1000000
        if (self.calls + 1 > self.plan["max_sample_calls"]
                or self.joint_tokens + tokens > self.plan["max_joint_sample_tokens"]
                or self.reserved_usd + charge > number(self.plan["cost_cap_usd"])):
            raise ValueError("finite pilot sampling ceiling")
        self.calls += 1
        self.joint_tokens += tokens
        self.reserved_usd += charge
        self._persist({"kind": "SAMPLE_RESERVED_BEFORE_DISPATCH", "request": request, "reserved_usd": money(charge)})

    def train(self, datums: list):
        tokens = sum(d.model_input.length for d in datums)
        charge = number(tokens) * Decimal("0.737") / 1000000
        if (self.training_tokens + tokens > self.plan["max_training_input_tokens"]
                or self.reserved_usd + charge > number(self.plan["cost_cap_usd"])):
            raise ValueError("finite pilot training ceiling")
        self.training_tokens += tokens
        self.reserved_usd += charge
        self._persist({"kind": "RL_BACKWARD_RESERVED_BEFORE_DISPATCH", "input_tokens": tokens, "reserved_usd": money(charge)})
