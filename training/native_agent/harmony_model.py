"""Caller-owned GPT-OSS native Harmony sampler for PureAgentRuntime.

This adapter creates no service, credential, paid reservation, or sampler.
The caller supplies its already-authorized sampler and optional pre-call hook.
Raw sampled tokens/logprobs are returned before any public action validation.
"""
from __future__ import annotations

from copy import deepcopy

from .runtime import LocalGenerationBudgetExhausted, canonical, strict_json, validate_action


class HarmonySamplerModel:
    def __init__(self, *, sampler, model_path: str, decision_at: str,
                 tool_definitions: list[dict], renderer=None, temperature: float = 0,
                 seed: int = 0, max_output_tokens: int = 2048,
                 max_total_generated_tokens: int = 32768,
                 future_timeout_seconds: float = 180, before_sample=None):
        if (type(model_path) is not str or not model_path
                or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 8192
                or type(max_total_generated_tokens) is not int or max_total_generated_tokens <= 0
                or type(temperature) not in {int, float} or not 0 <= temperature <= 2
                or not 0 < future_timeout_seconds <= 600):
            raise ValueError("explicit finite native sampling parameters required")
        specs = [deepcopy(item.get("function", item)) for item in tool_definitions]
        if len(specs) != 1 or specs[0].get("name") != "pa_tws":
            raise ValueError("one native pa_tws tool definition required")
        self.sampler, self.model_path, self.decision_at = sampler, model_path, decision_at
        self.tool_definitions, self.renderer = deepcopy(tool_definitions), renderer
        self.temperature, self.seed = temperature, seed
        self.max_output_tokens = max_output_tokens
        self.max_total_generated_tokens = max_total_generated_tokens
        self.future_timeout_seconds, self.before_sample = future_timeout_seconds, before_sample
        self.training_steps = []
        self.prompt, self.names, self.segment = None, None, None
        self.generated_tokens = 0
        self._last_observation = None

    def _initial_prompt(self, conversation, protocol):
        # Same native prefix as the frozen direct prompt builder, without its
        # fixed 4096-token reservation. The actual effective cap below reserves
        # only available context; no conversation or sampled token is trimmed.
        self.renderer = (protocol.local_renderer(self.decision_at[:10]) if self.renderer is None
                         else protocol._configured_renderer(self.renderer, self.decision_at[:10]))
        for row in conversation:
            protocol._safe_content(row["content"], self.renderer)
        specs = [deepcopy(item.get("function", item)) for item in self.tool_definitions]
        messages = self.renderer.create_conversation_prefix_with_tools(specs, conversation[0]["content"])
        messages.append({"role": "user", "content": conversation[1]["content"]})
        self.prompt = self.renderer.build_generation_prompt(messages)
        self.names = {"functions.pa_tws"}

    def generate(self, conversation):
        import tinker
        from training import v7_protocol as protocol
        from training import direct_native_episode as native
        if self.prompt is None:
            if len(conversation) != 2 or [row["role"] for row in conversation] != ["system", "user"]:
                raise ValueError("initial same-conversation system/user required")
            self._initial_prompt(conversation, protocol)
        elif (self._last_observation is None or conversation[-1] != self._last_observation):
            raise ValueError("exact native continuation observation required")
        remaining = self.max_total_generated_tokens - self.generated_tokens
        prompt_tokens = list(self.prompt.to_ints())
        context_remaining = native.MAX_SEQUENCE - len(prompt_tokens)
        cap = min(self.max_output_tokens, remaining, context_remaining)
        if cap <= 0:
            raise LocalGenerationBudgetExhausted("native token/context ceiling reached; no substituted action",
                details={"requested_max_output_tokens": self.max_output_tokens, "input_tokens": len(prompt_tokens),
                         "remaining_total_generated_tokens": remaining, "remaining_context_tokens": context_remaining})
        request = {"model_path": self.model_path, "sample_index": len(self.training_steps),
                   "input_tokens": len(prompt_tokens), "max_output_tokens": cap,
                   "requested_max_output_tokens": self.max_output_tokens,
                   "effective_max_output_tokens": cap, "remaining_total_generated_tokens": remaining,
                   "remaining_context_tokens": context_remaining, "max_sequence_tokens": native.MAX_SEQUENCE}
        if self.before_sample is not None:
            self.before_sample(deepcopy(request))
        response = self.sampler.sample(prompt=self.prompt, num_samples=1,
            sampling_params=tinker.SamplingParams(temperature=self.temperature, seed=self.seed,
                max_tokens=cap, stop=list(protocol.stop_token_ids(self.renderer)))).result(
                    timeout=self.future_timeout_seconds)
        # Retain the entire actual sampled native sequence, including analysis.
        sequences = []
        for sequence in response.sequences:
            stop = getattr(sequence.stop_reason, "value", sequence.stop_reason)
            logprobs = getattr(sequence, "logprobs", None)
            sequences.append({"tokens": list(sequence.tokens), "stop_reason": stop,
                              "logprobs": None if logprobs is None else list(logprobs)})
        wire = {"kind": "pure_gpt_oss_native_sample_v1", "model_path": self.model_path,
                "prompt_tokens": prompt_tokens, "sequences": sequences,
                "temperature": self.temperature, "seed": self.seed, "native_request": request}
        raw = canonical(wire).encode("utf-8")
        self.training_steps.append(deepcopy(wire))
        self.generated_tokens += sum(len(row["tokens"]) for row in sequences)
        return raw

    def parse(self, raw):
        from training import direct_native_episode as native
        wire = strict_json(raw)
        if wire != self.training_steps[-1] or len(wire["sequences"]) != 1:
            raise ValueError("one unchanged actual native sampled sequence required")
        sequence = wire["sequences"][0]
        if sequence["stop_reason"] != "stop" or self.generated_tokens > self.max_total_generated_tokens:
            raise ValueError("native incomplete output; no inferred action")
        self.segment = native.parse_direct_segment(sequence["tokens"], self.renderer, self.names)
        if self.segment["kind"] == "final":
            return validate_action({"kind": "final", "text": self.segment["final_text"]})
        if self.segment["tool_name"] != "functions.pa_tws":
            raise ValueError("one pa_tws native tool required")
        return validate_action({"kind": "tool", "name": "pa_tws",
                                "arguments": deepcopy(self.segment["arguments"])})

    def observe(self, action, result):
        from training import direct_native_episode as native
        if (self.segment is None or self.segment["kind"] != "tool"
                or action["arguments"] != self.segment["arguments"]):
            raise ValueError("exact native model tool arguments required")
        self.prompt = native.append_direct_tool_result(self.prompt, self.segment, result,
            self.renderer, self.names, "pure-native:" + str(len(self.training_steps)))
        self._last_observation = {"role": "tool", "name": "pa_tws", "content": canonical(result)}
        self.segment = None
