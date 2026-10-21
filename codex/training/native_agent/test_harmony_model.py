"""Offline native tokenizer/continuation checks with an authored sampler."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from training import v7_protocol
from .harmony_model import HarmonySamplerModel
from .runtime import LocalGenerationBudgetExhausted, PureAgentRuntime, strict_json


class HarmonyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = v7_protocol.local_renderer("2026-10-08")

    def model_fixture(self, *, max_output_tokens=8192, max_total_generated_tokens=32768,
                      stop_reason="stop"):
        tokens = self.renderer.tokenizer.encode('<|channel|>final<|message|>HOLD<|return|>', add_special_tokens=False)
        calls, admitted = [], []
        class Sampler:
            def sample(self, **kwargs):
                calls.append(kwargs)
                response = SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens,
                    logprobs=[-0.1] * len(tokens), stop_reason=stop_reason)])
                return SimpleNamespace(result=lambda timeout: response)
        model = HarmonySamplerModel(sampler=Sampler(), model_path="offline-authored-sampler",
            decision_at="2026-10-08T05:00:00Z", renderer=self.renderer,
            max_output_tokens=max_output_tokens, max_total_generated_tokens=max_total_generated_tokens,
            before_sample=admitted.append,
            tool_definitions=[{"name": "pa_tws", "description": "Exact test native commands",
                               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}])
        return model, calls, admitted

    def test_configured_8192_cap_reaches_native_sampler_and_admission_metadata(self):
        model, calls, admitted = self.model_fixture()
        raw = model.generate([{"role": "system", "content": "Choose actions."},
                              {"role": "user", "content": "Run the task."}])
        request = strict_json(raw)["native_request"]
        self.assertEqual(calls[0]["sampling_params"].max_tokens, 8192)
        self.assertEqual(request["requested_max_output_tokens"], 8192)
        self.assertEqual(request["effective_max_output_tokens"], 8192)
        self.assertEqual(admitted, [request])
        self.assertEqual(model.parse(raw), {"kind": "final", "text": "HOLD"})

    def test_initial_native_prefix_is_byte_equivalent_to_previous_builder_when_it_fits(self):
        from training import direct_native_episode as native
        model, calls, admitted = self.model_fixture()
        conversation = [{"role": "system", "content": "Choose actions yourself."},
                        {"role": "user", "content": "Run the exact native task."}]
        old_prompt, old_renderer, old_names = native.build_direct_prompt({
            "decision_at": model.decision_at, "tool_definitions": model.tool_definitions,
            "system_prompt": conversation[0]["content"], "user_prompt": conversation[1]["content"]}, self.renderer)
        raw = model.generate(conversation)
        self.assertEqual(strict_json(raw)["prompt_tokens"], old_prompt.to_ints())
        self.assertEqual(model.names, old_names)

    def test_actual_long_initial_context_reduces_only_output_cap(self):
        model, calls, admitted = self.model_fixture()
        text = " a" * 32000
        raw = model.generate([{"role": "system", "content": "Choose actions."},
                              {"role": "user", "content": text}])
        wire = strict_json(raw)
        request = wire["native_request"]
        self.assertGreater(request["input_tokens"], 32768 - 4096)
        self.assertEqual(request["effective_max_output_tokens"], 32768 - request["input_tokens"])
        self.assertEqual(calls[0]["sampling_params"].max_tokens, request["effective_max_output_tokens"])
        self.assertEqual(request["requested_max_output_tokens"], 8192)
        self.assertEqual(admitted, [request])
        self.assertIn(text, self.renderer.tokenizer.decode(wire["prompt_tokens"]))

    def test_remaining_total_generation_budget_reduces_only_output_cap(self):
        model, calls, admitted = self.model_fixture(max_total_generated_tokens=64)
        model.generated_tokens = 48
        raw = model.generate([{"role": "system", "content": "Choose actions."},
                              {"role": "user", "content": "Run the task."}])
        request = strict_json(raw)["native_request"]
        self.assertEqual(request["requested_max_output_tokens"], 8192)
        self.assertEqual(request["effective_max_output_tokens"], 16)
        self.assertEqual(calls[0]["sampling_params"].max_tokens, 16)
        self.assertEqual(admitted, [request])

    def test_exhausted_context_does_not_trim_conversation_or_sample(self):
        import tinker
        model, calls, admitted = self.model_fixture()
        model.prompt = tinker.ModelInput.from_ints([1] * 32768)
        model._last_observation = {"role": "tool", "name": "pa_tws", "content": "{}"}
        with self.assertRaisesRegex(LocalGenerationBudgetExhausted, "context ceiling"):
            model.generate([model._last_observation])
        self.assertEqual(len(model.prompt.to_ints()), 32768)
        self.assertEqual(calls, [])
        self.assertEqual(admitted, [])

    def test_local_generation_limit_classified_without_provider_dispatch_or_raw_fabrication(self):
        model, calls, admitted = self.model_fixture(max_total_generated_tokens=64)
        model.generated_tokens = 64
        executed = []
        class Host:
            def declaration(self):
                return {"environment": "AUTHORED_TEST_SIMULATION"}
            def execute(self, arguments):
                executed.append(arguments)
                return {"ok": True}
        with tempfile.TemporaryDirectory() as directory:
            runtime = PureAgentRuntime(model=model, host=Host(), journal_directory=Path(directory)/"episode",
                system_prompt="Choose actions.", user_prompt="Run task.", max_steps=3)
            report = runtime.run()
            receipt = strict_json((runtime.directory / "step-0001-receipt.json").read_bytes())
            self.assertFalse((runtime.directory / "step-0001-model.raw").exists())
            self.assertEqual(receipt["status"], "LOCAL_MODEL_BUDGET_EXHAUSTED")
            self.assertFalse(receipt["provider_dispatched"])
            self.assertFalse(receipt["tool_invoked"])
            self.assertEqual(receipt["details"]["remaining_total_generated_tokens"], 0)
            self.assertEqual(runtime.run()["steps"], 1)
        self.assertEqual(report["status"], "LOCAL_MODEL_BUDGET_EXHAUSTED")
        self.assertEqual(calls, [])
        self.assertEqual(admitted, [])
        self.assertEqual(executed, [])

    def test_incomplete_completion_is_not_repaired_or_replayed_at_higher_cap(self):
        model, calls, admitted = self.model_fixture(stop_reason="length")
        executed = []
        class Host:
            def declaration(self):
                return {"environment": "AUTHORED_TEST_SIMULATION"}
            def execute(self, arguments):
                executed.append(arguments)
                return {"ok": True}
        with tempfile.TemporaryDirectory() as directory:
            runtime = PureAgentRuntime(model=model, host=Host(), journal_directory=Path(directory)/"episode",
                system_prompt="Choose actions.", user_prompt="Run task.", max_steps=3)
            report = runtime.run()
            retained = strict_json((runtime.directory / "step-0001-model.raw").read_bytes())
            self.assertEqual(retained["sequences"][0]["stop_reason"], "length")
            self.assertEqual(retained["native_request"]["requested_max_output_tokens"], 8192)
            self.assertEqual(report["trace"][0]["raw_model_ref"]["path"], str(runtime.directory / "step-0001-model.raw"))
            self.assertFalse(report["trace"][0]["tool_invoked"])
            self.assertEqual(runtime.run()["steps"], 1)
        self.assertEqual(report["status"], "MALFORMED_OUTPUT")
        self.assertEqual(report["steps"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(executed, [])

    def test_whole_native_tokens_and_logprobs_retained_exact_terms_and_same_continuation(self):
        tokenizer = self.renderer.tokenizer
        action_tokens = tokenizer.encode(
            '<|channel|>analysis<|message|>Authored private test reasoning.<|end|>'
            '<|start|>assistant to=functions.pa_tws<|channel|>commentary<|constrain|>json<|message|>'
            '{"command":"order","side":"SELL","quantity":7,"limit_price":"13.900"}<|call|>',
            add_special_tokens=False)
        final_tokens = tokenizer.encode('<|channel|>final<|message|>HOLD<|return|>', add_special_tokens=False)
        sampled = iter((action_tokens, final_tokens))
        calls = []
        class Sampler:
            def sample(self, **kwargs):
                calls.append(kwargs)
                tokens = next(sampled)
                response = SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens,
                    logprobs=[-0.1] * len(tokens), stop_reason="stop")])
                return SimpleNamespace(result=lambda timeout: response)
        executed = []
        class SimulationHost:
            def declaration(self):
                return {"environment": "AUTHORED_TEST_SIMULATION"}
            def execute(self, arguments):
                executed.append(arguments)
                return {"ok": False, "status": "VISIBLE_TEST_REFUSAL", "arguments": arguments}
        model = HarmonySamplerModel(sampler=Sampler(), model_path="offline-authored-sampler",
            decision_at="2026-10-08T05:00:00Z", renderer=self.renderer, temperature=0.7, seed=7,
            tool_definitions=[{"name": "pa_tws", "description": "Exact test native commands",
                               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}])
        with tempfile.TemporaryDirectory() as directory:
            runtime = PureAgentRuntime(model=model, host=SimulationHost(), journal_directory=Path(directory)/"episode",
                system_prompt="Select all actions yourself.", user_prompt="Test native use.", max_steps=3)
            report = runtime.run()
            raw = strict_json((runtime.directory / "step-0001-model.raw").read_bytes())
        self.assertEqual(report["status"], "FINAL")
        self.assertEqual(executed, [{"command": "order", "side": "SELL", "quantity": 7, "limit_price": "13.900"}])
        self.assertEqual(raw["sequences"][0]["tokens"], action_tokens)
        self.assertEqual(raw["sequences"][0]["logprobs"], [-0.1] * len(action_tokens))
        self.assertEqual(model.training_steps[0], raw)
        self.assertEqual(model.training_steps[0]["temperature"], 0.7)
        self.assertEqual(calls[0]["sampling_params"].temperature, 0.7)
        first_prompt = model.training_steps[0]["prompt_tokens"]
        second_prompt = model.training_steps[1]["prompt_tokens"]
        self.assertEqual(second_prompt[:len(first_prompt)], first_prompt)
        self.assertEqual(second_prompt[len(first_prompt):len(first_prompt)+len(action_tokens)], action_tokens)
        decoded = tokenizer.decode(second_prompt)
        self.assertIn("VISIBLE_TEST_REFUSAL", decoded)
        self.assertEqual(len(model.training_steps), 2)


if __name__ == "__main__":
    unittest.main()
