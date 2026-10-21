"""Prospective full-output admission for the existing generic Harmony models.

This mixin supplies no action, summary, memory operation, sampler or accounting
hook. Place it before the existing named/wrapper model in the MRO. Every
admitted inherited request then has the unchanged8192 output allowance.
"""
from training.native_agent.runtime import LocalGenerationBudgetExhausted

FULL_REQUESTED_OUTPUT_TOKENS = 8192
JOINT_SEQUENCE_TOKENS = 32768
MAX_FULL_OUTPUT_INPUT_TOKENS = JOINT_SEQUENCE_TOKENS - FULL_REQUESTED_OUTPUT_TOKENS


class FullOutputReserveMixin:
    def generate(self, conversation):
        if type(self.max_output_tokens) is not int or self.max_output_tokens != FULL_REQUESTED_OUTPUT_TOKENS:
            raise ValueError("explicit full8192 requested output required")
        if self.prompt is not None:
            if self._last_observation is None or conversation[-1] != self._last_observation:
                raise ValueError("exact native continuation observation required")
            self._require_full_output_reserve()
        # The unchanged inherited generate validates the original initial
        # system/user join and invokes _initial_prompt once. Its MRO hook below
        # checks that exact new prefix before any sample/admission hook.
        return super().generate(conversation)

    def _initial_prompt(self, conversation, protocol):
        super()._initial_prompt(conversation, protocol)
        self._require_full_output_reserve()

    def _require_full_output_reserve(self):
        inputs = len(self.prompt.to_ints())
        remaining = self.max_total_generated_tokens - self.generated_tokens
        context_remaining = JOINT_SEQUENCE_TOKENS - inputs
        if inputs > MAX_FULL_OUTPUT_INPUT_TOKENS or remaining < FULL_REQUESTED_OUTPUT_TOKENS:
            raise LocalGenerationBudgetExhausted(
                "full8192 native output reserve unavailable before dispatch; no substituted action",
                details={"requested_max_output_tokens": FULL_REQUESTED_OUTPUT_TOKENS,
                         "input_tokens": inputs,
                         "remaining_total_generated_tokens": remaining,
                         "remaining_context_tokens": context_remaining,
                         "max_sequence_tokens": JOINT_SEQUENCE_TOKENS,
                         "full_requested_output_reserve_required": True,
                         "provider_dispatched": False})
