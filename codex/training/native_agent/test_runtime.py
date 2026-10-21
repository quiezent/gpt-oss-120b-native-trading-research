"""Offline causal tests; mock transports are never broker/fill evidence."""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from .cli_host import BOUNDARY, NativeCliHost, PaperCliConfig
from .runtime import JsonCompletionModel, LocalGenerationBudgetExhausted, PureAgentRuntime, canonical, parse_json_action


class HostFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cli = self.root / "pa_tws" / "pa_tws.py"
        cli.parent.mkdir()
        cli.write_text('HOST = "127.0.0.1"\nPORT = 4002\nACCOUNT = "PAPER_ACCOUNT"\nDEFAULT_CLIENT_ID = 9901\n')
        policy = self.root / "risk.json"
        policy.write_text(canonical({key: BOUNDARY[key] for key in ("host", "port", "account")}))
        self.config = PaperCliConfig(workspace=self.root, interpreter=Path(sys.executable), cli=cli,
            cli_sha256=hashlib.sha256(cli.read_bytes()).hexdigest(), risk_policy=policy,
            receipt_directory=self.root / "native", allow_mutations=True,
            max_quantity=20, max_order_notional_usd="1000")
        self.calls = []

    def transport(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return SimpleNamespace(stdout=b'{"ok":true,"result":{"status":"WORKING"}}', stderr=b"", returncode=0)

    def host(self, config=None, transport=None):
        return NativeCliHost(config or self.config, transport=transport or self.transport)

    @staticmethod
    def order(**changes):
        return {"command": "submit-stock-limit", "conid": 265598, "symbol": "AAPL",
                "primary_exchange": "NASDAQ", "side": "SELL", "quantity": 4,
                "limit": "12.340", "tif": "GTC", "order_ref": "pa:raw-terms-001", **changes}


class NativeTransportTests(HostFixture):
    def test_exact_model_terms_reach_only_canonical_cli(self):
        request = self.order()
        result = self.host().execute(request)
        self.assertEqual(result["arguments"], request)
        argv, kwargs = self.calls[0]
        self.assertEqual(argv[4], str(self.config.cli.resolve()))
        self.assertEqual(argv[7], "submit-stock-limit")
        for key, value in request.items():
            if key != "command":
                self.assertEqual(argv[argv.index("--" + key.replace("_", "-")) + 1], str(value))
        self.assertEqual(result["root_fields"], {"risk_policy": str(self.config.risk_policy.resolve()),
                                               "confirm": "SUBMIT_PAPER_ORDER"})
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["env"]["IBKR_PA_ALLOW_PAPER_ORDER"], "YES")
        raw = json.loads(Path(result["raw_receipt_ref"]["path"]).read_text())
        self.assertEqual(base64.b64decode(raw["stdout_base64"]), b'{"ok":true,"result":{"status":"WORKING"}}')

    def test_limits_refuse_without_clamping_or_side_replacement(self):
        host = self.host()
        for changes in ({"quantity": 21}, {"limit": "10000"}, {"side": "HOLD"}, {"quantity": 0}):
            request = self.order(**changes)
            result = host.execute(request)
            self.assertFalse(result["cli_invoked"])
            self.assertEqual(result["arguments"], request)
        self.assertEqual(self.calls, [])

    def test_low_reference_price_cannot_hide_excessive_limit(self):
        result = self.host().execute(self.order(limit="10000", reference_price="0.01"))
        self.assertFalse(result["cli_invoked"])
        self.assertEqual(self.calls, [])

    def test_host_owned_fields_and_abbreviations_cannot_redirect_boundary(self):
        host = self.host()
        for key in ("account", "host", "client_id", "conf", "risk_pol", "env"):
            result = host.execute({"command": "status", key: "changed"})
            self.assertEqual(result["status"], "HOST_REFUSAL")
        self.assertEqual(self.calls, [])

    def test_default_mutation_disabled_and_reads_still_exact(self):
        host = self.host(replace(self.config, allow_mutations=False))
        self.assertFalse(host.execute(self.order())["cli_invoked"])
        read = {"command": "quote", "conid": 1, "market_data_type": 3}
        self.assertEqual(host.execute(read)["arguments"], read)
        self.assertEqual(len(self.calls), 1)

    def test_native_unknown_argument_error_is_visible_without_repair(self):
        def failing(argv, **kwargs):
            self.calls.append((argv, kwargs))
            return SimpleNamespace(stdout=b"", stderr=b"unrecognized arguments: --extra-ordinary value\n", returncode=2)
        request = {"command": "status", "extra_ordinary": "value"}
        result = self.host(transport=failing).execute(request)
        self.assertEqual(result["arguments"], request)
        self.assertIn("--extra-ordinary", self.calls[0][0])
        self.assertIn("unrecognized arguments", result["stderr"])
        self.assertFalse(result["ok"])

    def test_uncertain_mutation_halts_all_mutations_but_allows_reconciliation_reads(self):
        def timeout(argv, **kwargs):
            self.calls.append((argv, kwargs))
            raise subprocess.TimeoutExpired(argv, 1, output=b"partial", stderr=b"uncertain")
        host = self.host(transport=timeout)
        first = host.execute(self.order())
        self.assertEqual(first["status"], "OUTCOME_UNKNOWN")
        self.assertEqual(host.execute(self.order(order_ref="pa:different-ref"))["status"],
                         "MUTATIONS_HALTED_RECONCILIATION_REQUIRED")
        host.transport = self.transport
        self.assertTrue(host.execute({"command": "open-orders"})["cli_invoked"])
        self.assertEqual(len(self.calls), 2)
        # Restart keeps the uncertainty latch; changing an order reference is
        # never an automatic recovery mechanism.
        restarted = self.host()
        self.assertFalse(restarted.execute(self.order(order_ref="pa:third-ref"))["cli_invoked"])

    def test_attempted_same_order_ref_never_resends_with_changed_price(self):
        host = self.host()
        host.execute(self.order())
        result = host.execute(self.order(limit="13.000"))
        self.assertEqual(result["status"], "DUPLICATE_OR_UNKNOWN_MUTATION_REFUSED")
        self.assertEqual(len(self.calls), 1)

    def test_parseable_pinned_native_uncertainty_latches_new_reference_mutations(self):
        def native_uncertain(argv, **kwargs):
            self.calls.append((argv, kwargs))
            payload = {"ok": False, "error": "BrokerTimeout", "message":
                "post-submission failure; order is uncertain and must not be retried: callback missing"}
            return SimpleNamespace(stdout=canonical(payload).encode(), stderr=b"", returncode=2)
        host = self.host(transport=native_uncertain)
        first = host.execute(self.order())
        self.assertEqual(first["status"], "OUTCOME_UNKNOWN")
        self.assertEqual(host.execute(self.order(order_ref="pa:new-ref"))["status"],
                         "MUTATIONS_HALTED_RECONCILIATION_REQUIRED")
        self.assertEqual(len(self.calls), 1)

    def test_native_structured_uncertain_success_latches_without_interpreting_other_prose(self):
        from .cli_host import native_uncertain_result
        self.assertTrue(native_uncertain_result({"ok": True, "result": {
            "outcome": "SUBMISSION_UNCERTAIN_DO_NOT_RETRY"}}))
        self.assertFalse(native_uncertain_result({"ok": False, "error": "PolicyError",
            "message": "uncertain market outlook; buy cheaper"}))
        self.assertFalse(native_uncertain_result({"ok": True, "result": {
            "news": "post-submission failure; order is uncertain and must not be retried: X"}}))

    def test_source_pin_change_refuses_before_broker(self):
        host = self.host()
        self.config.cli.write_text(self.config.cli.read_text() + "PORT = 4001\n")
        result = host.execute({"command": "account"})
        self.assertFalse(result["cli_invoked"])
        self.assertEqual(self.calls, [])

    def test_credentials_are_removed_from_child_environment(self):
        with patch.dict("os.environ", {"TINKER_API_KEY": "private", "OPENAI_API_KEY": "private",
                                      "SERVICE_ACCESS_TOKEN": "private", "IBKR_PA_ALLOW_PAPER_ORDER": "YES"}):
            self.host().execute({"command": "status"})
        env = self.calls[0][1]["env"]
        for key in ("TINKER_API_KEY", "OPENAI_API_KEY", "SERVICE_ACCESS_TOKEN", "IBKR_PA_ALLOW_PAPER_ORDER"):
            self.assertNotIn(key, env)


class RuntimeTests(HostFixture):
    def runtime(self, outputs, host=None, max_steps=4):
        values = iter(outputs)
        model = JsonCompletionModel(lambda conversation: next(values))
        return PureAgentRuntime(model=model, host=host or self.host(),
            journal_directory=self.root / "episode", system_prompt="Choose your own economic actions.",
            user_prompt="Run the task.", max_steps=max_steps)

    def test_malformed_output_never_causes_implicit_tool_action(self):
        raw = b'{"kind":"tool","name":"pa_tws","arguments":{"command":"status","command":"account"}}'
        runtime = self.runtime([raw])
        report = runtime.run()
        self.assertEqual(report["status"], "MALFORMED_OUTPUT")
        self.assertFalse(report["trace"][0]["tool_invoked"])
        self.assertEqual((runtime.directory / "step-0001-model.raw").read_bytes(), raw)
        self.assertEqual(self.calls, [])

    def test_raw_and_exact_action_are_durable_before_transport(self):
        action = {"kind": "tool", "name": "pa_tws", "arguments": self.order()}
        raw = canonical(action).encode()
        def verify_persisted(argv, **kwargs):
            self.assertEqual((self.root / "episode/step-0001-model.raw").read_bytes(), raw)
            retained = json.loads((self.root / "episode/step-0001-action.json").read_text())
            self.assertEqual(retained, action)
            return self.transport(argv, **kwargs)
        runtime = self.runtime([raw, b'{"kind":"final","text":"done"}'], self.host(transport=verify_persisted))
        report = runtime.run()
        self.assertEqual(report["status"], "FINAL")
        self.assertEqual(len(self.calls), 1)

    def test_refusal_is_visible_to_model_in_same_conversation(self):
        seen = []
        action = {"kind": "tool", "name": "pa_tws", "arguments": self.order(quantity=30)}
        def generate(conversation):
            seen.append(conversation)
            return canonical(action if len(seen) == 1 else {"kind": "final", "text": "limit refused"}).encode()
        runtime = PureAgentRuntime(model=JsonCompletionModel(generate), host=self.host(),
            journal_directory=self.root / "episode", system_prompt="Trade", user_prompt="Task", max_steps=2)
        runtime.run()
        observation = json.loads(seen[1][-1]["content"])
        self.assertEqual(observation["status"], "HOST_REFUSAL")
        self.assertEqual(observation["arguments"], action["arguments"])
        self.assertEqual(seen[1][-2]["tool_call"], action)
        self.assertEqual(self.calls, [])

    def test_final_hold_and_step_ceiling_never_force_entry_or_exit(self):
        runtime = self.runtime([b'{"kind":"final","text":"HOLD"}'])
        self.assertEqual(runtime.run()["final"], "HOLD")
        self.assertEqual(self.calls, [])
        # A nonfinal observation at the ceiling causes no cleanup trade.
        action = {"kind": "tool", "name": "pa_tws", "arguments": {"command": "positions"}}
        model = JsonCompletionModel(lambda conversation: canonical(action).encode())
        runtime = PureAgentRuntime(model=model, host=self.host(), journal_directory=self.root / "second-episode",
                                  system_prompt="Trade", user_prompt="Task", max_steps=1)
        self.assertEqual(runtime.run()["status"], "STEP_LIMIT")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0][7], "positions")

    def test_wrong_tool_and_prose_cannot_be_routed_to_action(self):
        for raw in (b"BUY four AAPL", b'{"kind":"tool","name":"submit_order","arguments":{}}',
                    b'{"kind":"tool","name":"pa_tws","arguments":{"quantity":NaN}}'):
            with self.assertRaises(ValueError):
                parse_json_action(raw)

    def test_local_budget_status_retains_prior_real_raw_and_tool_observations(self):
        seen = []
        action = {"kind": "tool", "name": "pa_tws", "arguments": {"command": "positions"}}
        first_raw = canonical(action).encode()
        def generate(conversation):
            seen.append(conversation)
            if len(seen) == 1:
                return first_raw
            raise LocalGenerationBudgetExhausted("context exhausted before dispatch", details={"remaining_context_tokens": 0})
        runtime = PureAgentRuntime(model=JsonCompletionModel(generate), host=self.host(),
            journal_directory=self.root / "episode", system_prompt="Choose actions", user_prompt="Task", max_steps=3)
        report = runtime.run()
        self.assertEqual(report["status"], "LOCAL_MODEL_BUDGET_EXHAUSTED")
        self.assertEqual((runtime.directory / "step-0001-model.raw").read_bytes(), first_raw)
        self.assertFalse((runtime.directory / "step-0002-model.raw").exists())
        self.assertEqual(report["trace"][0]["action"], action)
        self.assertEqual(report["trace"][1]["provider_dispatched"], False)
        self.assertEqual(runtime.conversation[-1]["role"], "tool")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(runtime.run()["steps"], 2)
        self.assertEqual(len(seen), 2)


class WorkerSerializationTests(HostFixture):
    def test_parent_lock_only_for_basic_reads_same_cli_for_every_command(self):
        from . import canonical_worker as worker
        events = []
        class Lock:
            def __enter__(self):
                events.append("locked")
            def __exit__(self, *ignored):
                events.append("unlocked")
        native = ModuleType("pa_tws.pa_tws")
        native.__file__ = str(self.config.cli)
        native.HOST, native.PORT, native.ACCOUNT, native.DEFAULT_CLIENT_ID = "127.0.0.1", 4002, "PAPER_ACCOUNT", 9901
        native.MutationLock = Lock
        package = ModuleType("pa_tws")
        package.pa_tws = native
        def exact(argv, **kwargs):
            events.append(("dispatch", argv[7], str(Path(argv[4]).resolve())))
            return SimpleNamespace(stdout=b'{"ok":true}', stderr=b"", returncode=0)
        with patch.object(worker, "ROOT", self.root), patch.object(worker.subprocess, "run", exact), \
                patch.dict(sys.modules, {"pa_tws": package, "pa_tws.pa_tws": native}):
            for command in ("status", "historical-daily", "submit-stock-limit"):
                argv = [sys.executable, "-B", "-X", "utf8", str(self.config.cli), "--timeout", "15", command]
                result = worker.execute({"argv": argv, "cli_sha256": self.config.cli_sha256,
                                         "child_deadline_seconds": 60})
                self.assertFalse(result["uncertain"])
        self.assertEqual(events, ["locked", ("dispatch", "status", str(self.config.cli)), "unlocked",
                                  ("dispatch", "historical-daily", str(self.config.cli)),
                                  ("dispatch", "submit-stock-limit", str(self.config.cli))])


if __name__ == "__main__":
    unittest.main()
