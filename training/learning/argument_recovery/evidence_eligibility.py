"""STAGED future-only capture eligibility; no SDK, broker, or Budget imports.

This module does not rewrite actions, manufacture rewards, or install itself.
Caller supplies exact captured records and a separately verified provider
accounting join. The completed V6 experiment must retain its original rule.
"""
from dataclasses import dataclass
from decimal import Decimal
import math
import hashlib
import json


@dataclass(frozen=True)
class TrajectoryIntegrity:
    eligible: bool
    reason: str
    captured_steps: int = 0
    generated_tokens: int = 0
    outcome_sha256: str | None = None
    captured_records_sha256: str | None = None


def value_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def captured_trajectory_integrity(outcome, records, *, expected_sampler,
        expected_temperature, known_token_ids, max_sequence_tokens=32768,
        max_output_tokens=8192, max_generated_tokens=98304, max_steps=12):
    """Admit only complete durable known-outcome samples, independent of reward.

    `records` contains: receipt, original native JSON, durable raw_verified,
    exact_sample_join_verified, request_digest_verified and accounted_verified.
    Those joins must be built from immutable bytes/claim/usage references by the
    future runner rather than asserted from `training_steps` alone.

    Exact source-identified public framing, strict JSON body and declared-schema
    refusals can be admitted with the unchanged signed costed NAV. Unclassified
    failures and incomplete length terminations remain excluded; this is not a
    blanket ValueError admission or schema-success reward.
    """
    status = outcome.get("runtime_status")
    if status not in {"FINAL", "STEP_LIMIT", "MALFORMED_OUTPUT"}:
        return TrajectoryIntegrity(False, "UNKNOWN_OR_LOCAL_RUNTIME_OUTCOME")
    if not finite_number(outcome.get("reward")):
        return TrajectoryIntegrity(False, "NONFINITE_OR_MISSING_ECONOMIC_REWARD")
    try:
        initial = Decimal(outcome["initial_nav_usd"])
        terminal = Decimal(outcome["terminal_nav_usd"])
        flows = Decimal(outcome["net_external_flows_usd"])
        expected_reward = (terminal - initial - flows) / initial
        if (not all(value.is_finite() for value in (initial, terminal, flows)) or initial <= 0
                or abs(expected_reward - Decimal(str(outcome["reward"]))) > Decimal("1e-12")
                or outcome.get("kind") != "simulated_costed_nav_reward_v1"
                or outcome.get("fees_included_in_reward") is not True
                or outcome.get("forced_liquidations") != 0):
            return TrajectoryIntegrity(False, "UNKNOWN_OR_INCONSISTENT_COSTED_NAV_OUTCOME")
    except (KeyError, ArithmeticError, ValueError, TypeError):
        return TrajectoryIntegrity(False, "UNKNOWN_OR_INCONSISTENT_COSTED_NAV_OUTCOME")
    if not records:
        return TrajectoryIntegrity(False, "NO_DURABLE_CAPTURE")
    if len(records) > max_steps or (status == "STEP_LIMIT" and len(records) != max_steps):
        return TrajectoryIntegrity(False, "RUNTIME_DECISION_LIMIT_MISMATCH")
    generated = 0
    for index, record in enumerate(records):
        if not all(record.get(flag) is True for flag in
                ("raw_verified", "exact_sample_join_verified", "request_digest_verified", "accounted_verified")):
            return TrajectoryIntegrity(False, "CAPTURE_IDENTITY_OR_ACCOUNTING_UNVERIFIED")
        receipt, wire = record["receipt"], record["native"]
        if receipt.get("step") != index + 1:
            return TrajectoryIntegrity(False, "NONCONTIGUOUS_CAPTURE_SEQUENCE")
        if wire.get("model_path") != expected_sampler or wire.get("temperature") != expected_temperature:
            return TrajectoryIntegrity(False, "WRONG_ON_POLICY_SAMPLER_OR_TEMPERATURE")
        prompt, sequences = wire.get("prompt_tokens"), wire.get("sequences")
        if type(prompt) is not list or not prompt or type(sequences) is not list or len(sequences) != 1:
            return TrajectoryIntegrity(False, "INVALID_NATIVE_CAPTURE_SHAPE")
        sequence = sequences[0]
        tokens, logprobs = sequence.get("tokens"), sequence.get("logprobs")
        if type(tokens) is not list or not tokens or type(logprobs) is not list or len(tokens) != len(logprobs):
            return TrajectoryIntegrity(False, "MISSING_OR_UNALIGNED_SAMPLED_LOGPROBS")
        if any(type(token) is not int or token not in known_token_ids for token in prompt + tokens):
            return TrajectoryIntegrity(False, "INVALID_PROVIDER_TOKEN_VOCABULARY")
        if any(not finite_number(lp) for lp in logprobs):
            return TrajectoryIntegrity(False, "NONFINITE_SAMPLED_LOGPROBS")
        request = wire.get("native_request", {})
        cap = request.get("effective_max_output_tokens")
        if (request.get("sample_index") != index or request.get("model_path") != expected_sampler
                or request.get("input_tokens") != len(prompt)
                or request.get("max_sequence_tokens") != max_sequence_tokens
                or type(cap) is not int or not 0 < cap <= max_output_tokens
                or request.get("max_output_tokens") != cap
                or request.get("requested_max_output_tokens") != max_output_tokens
                or cap != min(max_output_tokens, max_generated_tokens - generated, max_sequence_tokens - len(prompt))
                or len(tokens) > cap or len(prompt) + len(tokens) > max_sequence_tokens
                or request.get("remaining_total_generated_tokens") != max_generated_tokens - generated
                or request.get("remaining_context_tokens") != max_sequence_tokens - len(prompt)):
            return TrajectoryIntegrity(False, "DECLARED_SAMPLE_QUOTA_OR_CONTEXT_MISMATCH")
        generated += len(tokens)
        if generated > max_generated_tokens:
            return TrajectoryIntegrity(False, "GENERATED_QUOTA_OVERFLOW")
        if sequence.get("stop_reason") != "stop":
            return TrajectoryIntegrity(False, "CAPTURED_BUT_INCOMPLETE_PUBLIC_OUTPUT")
        if receipt.get("tool_result", {}).get("status") == "HOST_EXCEPTION_OUTCOME_UNKNOWN":
            return TrajectoryIntegrity(False, "UNKNOWN_HOST_EXECUTION_OUTCOME")
        local_refusal = receipt.get("local_argument_validation_refused") is True
        if local_refusal and record.get("known_local_argument_refusal_verified") is not True:
            return TrajectoryIntegrity(False, "UNVERIFIED_LOCAL_ARGUMENT_REFUSAL")
        if index < len(records) - 1:
            if receipt.get("status") != "RUNNING" or not (receipt.get("tool_invoked") is True or (receipt.get("tool_invoked") is False and local_refusal)):
                return TrajectoryIntegrity(False, "UNKNOWN_OR_INCONSISTENT_PRIOR_RUNTIME_STEP")
    terminal = records[-1]["receipt"]
    if status == "MALFORMED_OUTPUT":
        if (terminal.get("status") != status or terminal.get("tool_invoked") is not False
                or terminal.get("error") != "ValueError"
                or records[-1].get("known_pre_dispatch_failure_verified") is not True):
            return TrajectoryIntegrity(False, "UNCLASSIFIED_PARSE_FAILURE")
        reason = "COMPLETE_SOURCE_CLASSIFIED_PRE_DISPATCH_FAILURE_WITH_KNOWN_SIGNED_NAV"
    elif status == "FINAL":
        if terminal.get("status") != "FINAL" or terminal.get("tool_invoked") is not False:
            return TrajectoryIntegrity(False, "FINAL_RECEIPT_MISMATCH")
        reason = "COMPLETE_FINAL_WITH_KNOWN_NAV"
    else:
        if (terminal.get("status") not in {"RUNNING", "STEP_LIMIT"}
                or not (terminal.get("tool_invoked") is True or (terminal.get("tool_invoked") is False and terminal.get("local_argument_validation_refused") is True and records[-1].get("known_local_argument_refusal_verified") is True))
                or type(terminal.get("tool_result")) is not dict):
            return TrajectoryIntegrity(False, "STEP_LIMIT_RECEIPT_MISMATCH")
        reason = "KNOWN_BOUNDED_PARTIAL_POLICY_NAV"
    return TrajectoryIntegrity(True, reason, len(records), generated,
        value_digest(outcome), value_digest(records))


def evidence_group_advantages(outcomes, integrities):
    """Center every eligible outcome symmetrically; no new penalty or bonus."""
    if len(outcomes) != len(integrities):
        raise ValueError("EXACT_OUTCOME_CAPTURE_PROOF_ALIGNMENT_REQUIRED")
    eligible = [i for i, proof in enumerate(integrities) if proof.eligible]
    if any(integrities[i].outcome_sha256 != value_digest(outcomes[i]) for i in eligible):
        raise ValueError("INTEGRITY_PROOF_OUTCOME_IDENTITY_MISMATCH")
    result = [None] * len(outcomes)
    if len(eligible) != len(outcomes):
        return result  # Unknown member makes the whole group inadmissible; no validity selection.
    if len(eligible) >= 2:
        rewards = [Decimal(str(outcomes[i]["reward"])) for i in eligible]
        if not all(value.is_finite() for value in rewards):
            raise ValueError("FINITE_UNMODIFIED_NET_NAV_REWARDS_REQUIRED")
        mean = sum(rewards, Decimal(0)) / len(rewards)
        for i, reward in zip(eligible, rewards):
            result[i] = float(reward - mean)
    return result
