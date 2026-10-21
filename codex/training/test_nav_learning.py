"""Economic accounting, chronology, exact orders and native gradient tests."""
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from training.nav_learning import (PortfolioSimulationHost, SimulationConfig, group_advantages,
                                  load_histories, prepare_dataset, trajectory_datums, admissible_group_advantages, datum_wire)
from training.nav_learning import create_compatible_training_client, PARENT_LORA_TARGETS
from training.nav_tool_schema import COMMAND_FIELDS, argument_schema, simulator_tool_definition


def histories(length=500, symbols=("A", "B")):
    dates = [(date(2020, 1, 1) + timedelta(days=n)).isoformat() for n in range(length)]
    return {symbol: {"symbol": symbol, "currency": "USD", "contract_id": n + 1, "bars": [
        {"session": day, "open": "100", "high": "20000", "low": "1", "close": "100", "volume": "10000"}
        for day in dates]} for n, symbol in enumerate(symbols)}


def host(horizon=5, config=None, data=None):
    data = data or histories()
    return PortfolioSimulationHost(data, {"start_session": data["A"]["bars"][60]["session"],
                                         "horizon_sessions": horizon}, config)


def order(book, side="BUY", quantity=2, limit="110", ref="pa:entry", symbol="A"):
    return book.execute({"command": "submit-stock-order", "conid": book.contract_ids[symbol], "symbol": symbol,
                         "side": side, "quantity": quantity, "order_type": "LMT", "limit": limit,
                         "tif": "DAY", "order_ref": ref})["result"]


class NAVLearningTests(unittest.TestCase):
    def test_costed_marked_nav_and_no_forced_liquidation(self):
        data = histories()
        data["A"]["bars"][61]["open"] = "101"
        data["A"]["bars"][65]["close"] = "105"
        book = host(data=data)
        self.assertTrue(order(book)["ok"])
        result = book.terminal_result()
        expected = Decimal("10000") - 2 * Decimal("101.07575") - 1 + 2 * 105
        self.assertEqual(Decimal(result["terminal_nav_usd"]), expected)
        self.assertAlmostEqual(result["reward"], float((expected - 10000) / 10000))
        self.assertEqual(result["forced_liquidations"], 0)
        self.assertFalse(result["closed_inventory"])
        self.assertEqual(result["terminal_account"]["positions"][0]["quantity"], 2)
        self.assertEqual(result["commission_usd"], "1")

    def test_cashflows_do_not_create_profit(self):
        book = host()
        book.record_external_flow("5000", reference_id="deposit-evidence")
        result = book.terminal_result()
        self.assertEqual(result["terminal_nav_usd"], "15000")
        self.assertEqual(result["reward"], 0)
        with self.assertRaises(ValueError):
            book.record_external_flow("5000", reference_id="late-deposit")

    def test_future_suffix_cannot_change_visible_history(self):
        data = histories()
        other = histories()
        for bar in other["A"]["bars"][61:]:
            bar["close"] = "12345"
        a, b = host(data=data), host(data=other)
        args = {"command": "historical-daily", "symbol": "A", "count": 252}
        self.assertEqual(a.execute(args), b.execute(args))
        self.assertEqual(a.account(), b.account())
        self.assertLessEqual(a.execute(args)["result"]["bars"][-1]["session"], a.sessions[a.start])

    def test_open_risk_never_reads_same_session_future_close(self):
        cfg = replace(SimulationConfig(), max_order_notional_usd="3000", max_symbol_notional_usd="6000",
                      max_gross_notional_usd="6000", min_cash_usd="0", commission_per_share_usd="0",
                      minimum_commission_usd="0", spread_bps="0", slippage_bps="0")
        data = histories()
        data["A"]["bars"][61]["close"] = "10000"
        book = host(horizon=1, config=cfg, data=data)
        self.assertTrue(order(book, quantity=25, limit="100", ref="pa:order-a")["ok"])
        self.assertTrue(order(book, quantity=25, limit="100", ref="pa:order-b", symbol="B")["ok"])
        book.execute({"command": "advance", "sessions": 1})
        self.assertEqual([f["symbol"] for f in book.fills], ["A", "B"])

    def test_no_range_touch_fill_and_exact_limit(self):
        book = host()
        self.assertTrue(order(book, limit="99")["ok"])
        book.execute({"command": "advance", "sessions": 1})
        self.assertEqual(book.fills, [])
        self.assertEqual(book.cash, 10000)
        self.assertIn("LIMIT_NOT_MARKETABLE", book.receipts[-2]["reason"])

    def test_buy_cash_and_owned_sell_reservations(self):
        cfg = replace(SimulationConfig(), initial_cash_usd="1000", min_cash_usd="0")
        book = host(config=cfg)
        self.assertTrue(order(book, quantity=5, limit="110", ref="pa:first")["ok"])
        self.assertFalse(order(book, quantity=5, limit="110", ref="pa:second")["ok"])
        book.execute({"command": "advance", "sessions": 1})
        self.assertTrue(order(book, side="SELL", quantity=4, limit="90", ref="pa:exit1")["ok"])
        self.assertFalse(order(book, side="SELL", quantity=2, limit="90", ref="pa:oversell")["ok"])
        self.assertFalse(order(book, side="SELL", quantity=1, limit="90", ref="pa:exit1")["ok"])
        book.execute({"command": "advance", "sessions": 1})
        self.assertEqual(book.inventory["A"], 1)

    def test_all_horizons_retain_terminal_and_advance_bound(self):
        for horizon in (1, 5, 20, 63):
            book = host(horizon=horizon)
            self.assertFalse(book.execute({"command": "advance", "sessions": horizon + 1})["result"]["ok"])
            result = book.terminal_result()
            self.assertEqual(book.cursor, book.start + horizon)
            self.assertEqual(result["horizon_sessions"], horizon)
            self.assertEqual(result["reward"], 0)

    def test_purge_all_horizons_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            source.write_text(json.dumps(histories()))
            manifest = prepare_dataset(source, root / "prepared")
            episodes = json.loads((root / "prepared/episodes.json").read_text())
            by_split = {s: [e for e in episodes if e["split"] == s] for s in ("train", "validation", "test")}
            self.assertLess(max(e["end_session"] for e in by_split["train"]), min(e["start_session"] for e in by_split["validation"]))
            self.assertLess(max(e["end_session"] for e in by_split["validation"]), min(e["start_session"] for e in by_split["test"]))
            self.assertFalse(manifest["qualification"]["blind_evaluation"])
            source.write_text(source.read_text() + " ")
            with self.assertRaises(ValueError):
                load_histories(source, expected_sha256=manifest["source"]["sha256"])

    def test_same_start_group_relative_reward_and_native_mask(self):
        advantages = group_advantages([.02, -.02, 0, 0])
        self.assertEqual(advantages, [.02, -.02, 0, 0])
        datums = trajectory_datums([{"prompt_tokens": [10, 20, 30], "tokens": [40, 50], "logprobs": [-1., -2.]}], -.02)
        datum = datums[0]
        self.assertEqual(datum.model_input.to_ints(), [10, 20, 30, 40])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [20, 30, 40, 50])
        for actual, expected in zip(datum.loss_fn_inputs["advantages"].data, [0., 0., -.02, -.02]):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            trajectory_datums([{"prompt_tokens": [10], "tokens": [40, 50], "logprobs": [-1.]}], .1)
        self.assertEqual(datum_wire(datum)["model_input_tokens"], [10, 20, 30, 40])

    def test_failed_cash_episode_is_not_an_economic_winner(self):
        results = [{"runtime_status": "MALFORMED_OUTPUT", "reward": 0},
                   {"runtime_status": "FINAL", "reward": -.01}]
        self.assertEqual(admissible_group_advantages(results), [None, None])
        results += [{"runtime_status": "STEP_LIMIT", "reward": .01}]
        self.assertEqual(admissible_group_advantages(results), [None, -.01, .01])
        results += [{"runtime_status": "LOCAL_MODEL_BUDGET_EXHAUSTED", "reward": .02}]
        self.assertEqual(admissible_group_advantages(results), [None, -.01, .01, None])

    def test_native_order_terms_are_supplied_by_model_and_unchanged(self):
        book = host()
        args = {"command": "submit-stock-order", "conid": 1, "symbol": "A", "side": "BUY", "quantity": 2,
                "order_type": "LMT", "limit": "110", "tif": "DAY", "order_ref": "pa:native"}
        missing = {k: v for k, v in args.items() if k != "tif"}
        self.assertFalse(book.execute(missing)["result"]["ok"])
        self.assertEqual(book.orders, [])
        wrong = {**args, "conid": 999}
        self.assertFalse(book.execute(wrong)["result"]["ok"])
        receipt = book.execute(args)
        self.assertEqual(receipt["arguments"], args)
        self.assertEqual({k: v for k, v in book.orders[0].items() if k != "submitted_session"}, args)

    def test_native_training_target_matches_actual_parent_not_sdk_defaults(self):
        class Service:
            def create_lora_training_client(self, **kwargs):
                self.kwargs = kwargs
                return "actual-client"
        service = Service()
        plan = {"base_model": "openai/gpt-oss-120b", "rank": 8, "seed": 123,
                "lora_targets": dict(PARENT_LORA_TARGETS)}
        self.assertEqual(create_compatible_training_client(service, plan), "actual-client")
        self.assertEqual(service.kwargs, {"base_model": "openai/gpt-oss-120b", "rank": 8, "seed": 123,
                                          "train_attn": True, "train_mlp": True, "train_unembed": False})
        with self.assertRaises(ValueError):
            create_compatible_training_client(service, {**plan, "lora_targets": {**PARENT_LORA_TARGETS, "train_unembed": True}})

    def test_exact_api_contract_is_visible_in_actual_harmony_tool_rendering(self):
        from tinker_cookbook.renderers.gpt_oss import _format_tool_definition
        book = host()
        declaration = book.declaration()
        schema = argument_schema(declaration)
        self.assertEqual(schema, declaration["public_api_contract"]["argument_schema"])
        by_command = {row["properties"]["command"]["enum"][0]: row for row in schema["oneOf"]}
        for command, fields in COMMAND_FIELDS.items():
            self.assertEqual(set(by_command[command]["required"]), set(fields))
            self.assertEqual(set(by_command[command]["properties"]), set(fields))
            self.assertFalse(by_command[command]["additionalProperties"])
        rendered = _format_tool_definition(simulator_tool_definition(declaration)[0]["function"])
        for command in COMMAND_FIELDS:
            self.assertIn('"' + command + '"', rendered)
        for text in ("unquoted whole JSON integer", "quantity", "conid", "order_type", "LMT", "DAY",
                     "next session open", "count", "sessions_remaining"):
            self.assertIn(text, rendered)
        self.assertNotIn("Keeping cash is an available choice", rendered)
        self.assertEqual(schema["properties"]["count"]["type"], "integer")
        self.assertEqual(schema["properties"]["count"]["maximum"], 252)
        self.assertIn("Simulation competence is not evidence", declaration["public_api_contract"]["native_cli_transfer"])




if __name__ == "__main__":
    unittest.main()
