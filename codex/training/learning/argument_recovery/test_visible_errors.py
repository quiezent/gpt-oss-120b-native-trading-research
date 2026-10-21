"""Authored fixtures for refusal continuation and exact provenance; no model run."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from training.native_agent.runtime import canonical, strict_json
from training.learning.argument_recovery import interface_adapter as adapter
from training.learning.argument_recovery.visible_errors import (
    VisibleDeclaredArgumentModel, VisibleDeclaredArgumentRuntime, argument_error, refused_result)
from training.learning.argument_recovery.failure_classification import known_local_argument_refusal
from training.learning.argument_recovery.evidence_eligibility import (
    captured_trajectory_integrity, evidence_group_advantages)


class FixtureModel:
    def __init__(self, actions):
        self.actions = deepcopy(actions)
        self.observations = []

    def generate(self, conversation):
        return canonical(self.actions.pop(0)).encode()

    def parse(self, raw):
        return strict_json(raw)

    def observe(self, action, result):
        self.observations.append((deepcopy(action), deepcopy(result)))


class FixtureHost:
    def __init__(self):
        self.calls = []

    def declaration(self):
        return {'synthetic_fixture': True, 'broker_connected': False}

    def execute(self, arguments):
        self.calls.append(deepcopy(arguments))
        return {'ok': True, 'synthetic_fixture': True}


class VisibleErrorChecks(unittest.TestCase):
    def setUp(self):
        self.tools = adapter.named_tool_definitions()

    def test_nonnumeric_refusal_is_visible_then_model_selects_new_valid_call(self):
        invalid = {'kind': 'tool', 'name': 'clock', 'arguments': {'command': 'clock', 'extra': 'unchanged'}}
        valid = {'kind': 'tool', 'name': 'clock', 'arguments': {'command': 'clock'}}
        model, host = FixtureModel([invalid, valid]), FixtureHost()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = VisibleDeclaredArgumentRuntime(model=model, host=host,
                journal_directory=Path(tmp) / 'run', system_prompt='Synthetic fixture',
                user_prompt='No provider or broker', tool_definitions=self.tools, max_steps=2)
            first = runtime.step()
            self.assertEqual(first['status'], 'RUNNING')
            self.assertFalse(first['tool_invoked'])
            self.assertEqual(first['action'], invalid)
            self.assertFalse(first['tool_result']['arguments_repaired'])
            self.assertFalse(first['tool_result']['automatic_retry'])
            self.assertEqual(first['tool_result']['arguments'], invalid['arguments'])
            self.assertEqual(host.calls, [])
            self.assertEqual(model.observations, [(invalid, first['tool_result'])])
            self.assertTrue(Path(first['raw_model_ref']['path']).exists())
            self.assertTrue(Path(first['action_ref']['path']).exists())
            second = runtime.step()
            self.assertTrue(second['tool_invoked'])
            self.assertEqual(host.calls, [valid['arguments']])
            self.assertEqual(runtime.steps, 2)
            self.assertEqual(runtime.step()['status'], 'STEP_LIMIT')

    def test_recipient_body_mismatch_remains_fatal_without_dispatch(self):
        invalid = {'kind': 'tool', 'name': 'clock', 'arguments': {'command': 'framework_status'}}
        model, host = FixtureModel([invalid]), FixtureHost()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = VisibleDeclaredArgumentRuntime(model=model, host=host,
                journal_directory=Path(tmp) / 'run', system_prompt='Fixture', user_prompt='Fixture',
                tool_definitions=self.tools, max_steps=2)
            result = runtime.step()
            self.assertEqual(result['status'], 'MALFORMED_OUTPUT')
            self.assertEqual(host.calls, [])
            self.assertEqual(model.observations, [])

    def test_changed_refusal_result_fails_exact_saved_provenance(self):
        action = {'kind': 'tool', 'name': 'clock', 'arguments': {'command': 'clock', 'extra': 'unchanged'}}
        result = refused_result(action, argument_error(action, self.tools))
        public_frame = {'channel': 'commentary', 'recipient': 'functions.clock', 'content_type': 'json',
            'terminal': '<|call|>', 'commentary_action_condition_passes': True,
            'public_body': canonical(action['arguments'])}
        reader = lambda *args, **kwargs: ([public_frame], {})
        native = {'sequences': [{'stop_reason': 'stop', 'tokens': [1]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'action.json'
            path.write_bytes(canonical(action).encode())
            receipt = {'status': 'RUNNING', 'tool_invoked': False, 'local_argument_validation_refused': True,
                'action': action, 'tool_result': result,
                'action_ref': {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
            self.assertTrue(known_local_argument_refusal(receipt, native, {}, self.tools, reader)[0])
            receipt['tool_result'] = {**result, 'arguments_repaired': True}
            self.assertFalse(known_local_argument_refusal(receipt, native, {}, self.tools, reader)[0])

    def test_verified_refusal_retains_signed_nav_and_unknown_excludes_group(self):
        action = {'kind': 'tool', 'name': 'clock', 'arguments': {'command': 'clock', 'extra': 'unchanged'}}
        receipt = {'step': 1, 'status': 'RUNNING', 'tool_invoked': False,
            'local_argument_validation_refused': True,
            'tool_result': refused_result(action, argument_error(action, self.tools))}
        record = {'receipt': receipt, 'known_local_argument_refusal_verified': True,
            'raw_verified': True, 'exact_sample_join_verified': True,
            'request_digest_verified': True, 'accounted_verified': True,
            'native': {'model_path': 'synthetic', 'temperature': 1.0, 'prompt_tokens': [1],
                'sequences': [{'tokens': [2], 'logprobs': [-1.0], 'stop_reason': 'stop'}],
                'native_request': {'sample_index': 0, 'model_path': 'synthetic', 'input_tokens': 1,
                    'max_sequence_tokens': 32768, 'max_output_tokens': 8192, 'effective_max_output_tokens': 8192,
                    'requested_max_output_tokens': 8192, 'remaining_total_generated_tokens': 8192,
                    'remaining_context_tokens': 32767}}}
        def outcome(reward):
            return {'runtime_status': 'STEP_LIMIT', 'reward': reward, 'initial_nav_usd': '100',
                'terminal_nav_usd': str(Decimal(100) * (1 + Decimal(str(reward)))),
                'net_external_flows_usd': '0', 'kind': 'simulated_costed_nav_reward_v1',
                'fees_included_in_reward': True, 'forced_liquidations': 0}
        def proof(value, capture):
            return captured_trajectory_integrity(value, [capture], expected_sampler='synthetic',
                expected_temperature=1.0, known_token_ids={1, 2}, max_steps=1, max_generated_tokens=8192)
        outcomes = [outcome(-0.02), outcome(0.01)]
        proofs = [proof(value, record) for value in outcomes]
        self.assertTrue(all(row.eligible for row in proofs))
        self.assertEqual(evidence_group_advantages(outcomes, proofs), [-0.015, 0.015])
        unknown = deepcopy(record)
        unknown['known_local_argument_refusal_verified'] = False
        unknown_proof = proof(outcomes[1], unknown)
        self.assertFalse(unknown_proof.eligible)
        self.assertEqual(evidence_group_advantages(outcomes, [proofs[0], unknown_proof]), [None, None])
        self.assertIs(adapter.model_class(adapter.NAMED), VisibleDeclaredArgumentModel)
        self.assertIs(adapter.runtime_class(adapter.NAMED), VisibleDeclaredArgumentRuntime)


if __name__ == '__main__':
    unittest.main()
