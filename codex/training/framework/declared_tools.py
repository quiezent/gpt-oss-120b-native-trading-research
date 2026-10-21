"""Prospective declared framework/clock tools; no service or transport setup.

The sampler prefix declares each operation before sampling. A public recipient
must match the complete body command and its declared schema. No alias, missing
argument inference, output repair or economic decision is supplied here.
Existing mandate, framework, native and clock hosts execute unchanged arguments.
"""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path

from training.native_agent.runtime import (
    PureAgentRuntime, LocalGenerationBudgetExhausted, canonical, strict_json, utc_now, write_new)
from training.native_agent.harmony_model import HarmonySamplerModel
from training.native_agent.compact_presentation import DiagnosticRuntime, exact_json_equal

OPERATION_FIELDS = {
    "framework_configure_mandate": {"mandate"},
    "framework_status": set(), "framework_snapshot": {"contracts"},
    "framework_preview": {"intent"}, "framework_submit": {"intent"},
    "framework_reconcile": {"intent"},
    "framework_cancel": {"order_id", "order_ref", "conid"},
    "clock": set(), "wait_until": {"until_utc"}}


def tool_definitions(component, adapter):
    original = adapter.tool_definition(component)[0]["function"]
    props = original["parameters"]["properties"]
    if set(props["command"]["enum"]) != set(OPERATION_FIELDS):
        raise ValueError("exact existing nine framework/clock operations required")
    full = original["description"]
    configure = full[full.index(" framework_configure_mandate is a LOCAL"):]
    framework = full[full.index("An intent has exactly"):full.index(" Local host operations")]
    # The schema and original exact field values are unchanged. Descriptions
    # explain the operation, paper boundary and existing refusal semantics.
    descriptions = {
        "framework_configure_mandate": configure,
        "framework_status": "Obtain genuine framework broker status; this is a read operation.",
        "framework_snapshot": "Obtain current non-atomic framework account/positions/orders/executions/contracts/quotes for your exact contracts. This does not establish complete history.",
        "framework_preview": "Preview your exact complete eleven-field intent. " + framework,
        "framework_submit": "Submit your exact complete eleven-field intent. " + framework,
        "framework_reconcile": "Reconcile your exact complete eleven-field intent without resubmission. " + framework,
        "framework_cancel": "Cancel only your exact owned order_id, order_ref and conid. Refusals and underlying receipts are returned unchanged; uncertain attempts retain their mutation halt.",
        "clock": "Local system UTC observation only; no broker or provider action.",
        "wait_until": "Wait until your exact UTC ISO8601 target ending Z or +00:00 in this same conversation. No model sampling or broker polling during waiting. Past/present targets return immediately. Targets beyond the disclosed session deadline are refused unchanged. No timing is supplied or clipped; FINAL terminates and schedules nothing."}
    tools = []
    for name, fields in sorted(OPERATION_FIELDS.items()):
        schema = {"type": "object", "additionalProperties": False,
            "required": sorted(fields | {"command"}), "properties": {
                "command": {"type": "string", "enum": [name]},
                **{field: deepcopy(props[field]) for field in sorted(fields)}}}
        tools.append({"type": "function", "function": {"name": name,
            "description": "Use this declared function with command exactly " + name
                + ". Choose every argument yourself; exact arguments are never replaced. " + descriptions[name],
            "parameters": schema}})
    validate_manifest(tools)
    return tools


def validate_manifest(tools):
    if type(tools) is not list or len(tools) != len(OPERATION_FIELDS):
        raise ValueError("all nine explicit declared operations required before sampling")
    specs = {}
    for item in tools:
        if type(item) is not dict or set(item) != {"type", "function"} or item["type"] != "function":
            raise ValueError("exact standard function tool declaration required")
        spec = item["function"]
        if type(spec) is not dict or set(spec) != {"name", "description", "parameters"}:
            raise ValueError("exact function declaration fields required")
        name = spec["name"]
        if name not in OPERATION_FIELDS or name in specs or type(spec["description"]) is not str:
            raise ValueError("unique existing operation tool names required; no alias")
        schema = spec["parameters"]
        keys = OPERATION_FIELDS[name] | {"command"}
        if (schema.get("type") != "object" or schema.get("additionalProperties") is not False
                or set(schema.get("properties", {})) != keys
                or set(schema.get("required", [])) != keys
                or schema["properties"]["command"] != {"type": "string", "enum": [name]}):
            raise ValueError("recipient must declare its exact singleton command and operation fields")
        specs[name] = deepcopy(spec)
    canonical(specs)
    return specs


def validate_schema(value, schema, path="arguments"):
    """Strict standard schema subset used by the authenticated declarations.

    JSON values retain their exact types; bool is never an integer or number.
    Unsupported schema keywords fail closed rather than being ignored.
    This validates presentation only; native policy/refusal checks stay in host.
    """
    supported = {"type", "description", "enum", "properties", "required",
        "additionalProperties", "items", "minimum", "maximum"}
    if type(schema) is not dict or set(schema) - supported:
        raise ValueError("unsupported declared schema keyword")
    types = schema.get("type")
    types = types if type(types) is list else [types]
    matches = {"object": type(value) is dict, "array": type(value) is list,
        "string": type(value) is str, "integer": type(value) is int,
        "number": type(value) in {int, float} and (type(value) is int or math.isfinite(value)),
        "boolean": type(value) is bool}
    if any(kind not in matches for kind in types) or not any(matches[kind] for kind in types):
        raise ValueError("declared schema type mismatch: " + path)
    if "enum" in schema and not any(exact_json_equal(value, item) for item in schema["enum"]):
        raise ValueError("declared schema enum mismatch: " + path)
    if type(value) is dict:
        props = schema.get("properties", {})
        if not set(schema.get("required", [])) <= set(value):
            raise ValueError("missing required declared fields: " + path)
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError("extra undeclared fields: " + path)
        for key, item in value.items():
            if key in props:
                validate_schema(item, props[key], path + "." + key)
    elif type(value) is list:
        for index, item in enumerate(value):
            validate_schema(item, schema["items"], path + "[" + str(index) + "]")
    if type(value) in {int, float}:
        if ("minimum" in schema and value < schema["minimum"]) or ("maximum" in schema and value > schema["maximum"]):
            raise ValueError("declared schema numeric bound mismatch: " + path)


def validate_declared_action(action, tools):
    specs = validate_manifest(tools)
    if type(action) is not dict:
        raise ValueError("model action must be an object")
    if action.get("kind") == "tool":
        if set(action) != {"kind", "name", "arguments"} or action["name"] not in specs:
            raise ValueError("exact explicitly declared operation action required; no alias")
        # Explicit matching is in addition to the singleton enum schema.
        if type(action["arguments"]) is not dict or action["arguments"].get("command") != action["name"]:
            raise ValueError("model recipient and body command must match exactly")
        validate_schema(action["arguments"], specs[action["name"]]["parameters"])
    elif action.get("kind") == "final":
        if set(action) != {"kind", "text"} or type(action["text"]) is not str:
            raise ValueError("expected exact final action with text")
    else:
        raise ValueError("model must emit a declared tool or final; no inferred action")
    canonical(action)
    return deepcopy(action)


class DeclaredOperationHost:
    """Declare actual recipient addresses around unchanged operational host."""
    def __init__(self, host, tools):
        validate_manifest(tools)
        self.host, self.tools, self.directory = host, deepcopy(tools), host.directory

    def declaration(self):
        result = deepcopy(self.host.declaration())
        result["tool_address"] = sorted(OPERATION_FIELDS)
        result["local_temporal_operations"]["tool_address"] = ["clock", "wait_until"]
        result["declared_operation_protocol"] = {
            "recipients": sorted(OPERATION_FIELDS), "body_command_must_equal_recipient": True,
            "schema_sha256": __import__("hashlib").sha256(canonical(self.tools).encode()).hexdigest(),
            "generic_pa_tws_alias_declared": False, "argument_repair": False,
            "model_selected_order_and_mandate_values_preserved": True}
        return result

    def execute(self, arguments):
        return self.host.execute(deepcopy(arguments))


class DeclaredAgentRuntime(PureAgentRuntime):
    """One conversation, bounded model steps, append-only raw output receipts.

    ``host.declaration`` is explicitly an execution constraint, included in the
    initial prompt. It must not contain provider/account credentials. Each
    call receives exact parsed arguments. A refused valid call remains visible
    in the same conversation. Invalid completions stop without action/repair.
    Reaching the step ceiling does not submit, cancel, or flatten anything.
    The supplied model/host remain caller-owned and are never reopened here.
    """

    def __init__(self, *, model: Model, host: Host, journal_directory: str | Path,
                 system_prompt: str, user_prompt: str, tool_definitions: list[dict], max_steps: int = 32):
        if type(max_steps) is not int or not 1 <= max_steps <= 4096:
            raise ValueError("finite max_steps required")
        if type(system_prompt) is not str or type(user_prompt) is not str:
            raise ValueError("text prompts required")
        validate_manifest(tool_definitions)
        self.tool_definitions = deepcopy(tool_definitions)
        self.model, self.host = model, host
        self.directory = Path(journal_directory).resolve()
        self.max_steps, self.steps = max_steps, 0
        self.status, self.final = "READY", None
        self.trace = []
        declaration = deepcopy(host.declaration())
        self.conversation = [
            {"role": "system", "content": system_prompt + "\nExecution constraints supplied by the host:\n"
             + canonical(declaration)},
            {"role": "user", "content": user_prompt}]
        # Refuse reuse before any sample or tool action. Raw completions may
        # contain private analysis; retain locally without exposing it to tools.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.start_ref = write_new(self.directory / "start.json", canonical({
            "kind": "pure_native_agent_start_v1", "created_at_utc": utc_now(),
            "max_steps": max_steps, "host_declaration": declaration,
            "conversation": self.conversation, "decision_routing": False,
            "silent_output_repair": False}).encode("utf-8"))

    def _receipt(self, data):
        ref = write_new(self.directory / f"step-{self.steps:04d}-receipt.json",
                        canonical(data).encode("utf-8"))
        result = {**data, "receipt_ref": ref}
        self.trace.append(result)
        return result

    def step(self):
        if self.status not in {"READY", "RUNNING"}:
            return self.report()
        if self.steps >= self.max_steps:
            self.status = "STEP_LIMIT"
            return self.report()
        self.steps += 1
        self.status = "RUNNING"
        started = utc_now()
        try:
            raw = self.model.generate(deepcopy(self.conversation))
            if type(raw) is not bytes:
                raise TypeError("model backend must return original bytes")
            raw_ref = write_new(self.directory / f"step-{self.steps:04d}-model.raw", raw)
        except LocalGenerationBudgetExhausted as error:
            self.status = "LOCAL_MODEL_BUDGET_EXHAUSTED"
            return self._receipt({"step": self.steps, "status": self.status,
                                  "error": type(error).__name__, "message": str(error),
                                  "details": error.details, "tool_invoked": False,
                                  "provider_dispatched": False, "started_at_utc": started})
        except Exception as error:
            self.status = "MODEL_OR_RAW_CAPTURE_FAILED"
            return self._receipt({"step": self.steps, "status": self.status,
                                  "error": type(error).__name__, "message": str(error),
                                  "tool_invoked": False, "started_at_utc": started})
        try:
            action = validate_declared_action(self.model.parse(raw), self.tool_definitions)
        except Exception as error:
            self.status = "MALFORMED_OUTPUT"
            return self._receipt({"step": self.steps, "status": self.status,
                                  "error": type(error).__name__, "message": str(error),
                                  "raw_model_ref": raw_ref, "tool_invoked": False,
                                  "started_at_utc": started})
        if action["kind"] == "final":
            self.status, self.final = "FINAL", action["text"]
            self.conversation.append({"role": "assistant", "content": action["text"]})
            return self._receipt({"step": self.steps, "status": self.status,
                                  "action": action, "raw_model_ref": raw_ref,
                                  "tool_invoked": False, "started_at_utc": started})
        # Save exact public tool arguments before transport; failure to persist
        # halts before dispatch. This file is distinct from the raw completion.
        action_ref = write_new(self.directory / f"step-{self.steps:04d}-action.json",
                              canonical(action).encode("utf-8"))
        try:
            result = self.host.execute(deepcopy(action["arguments"]))
            if type(result) is not dict:
                raise TypeError("host must return an object receipt")
            canonical(result)
        except Exception as error:
            result = {"ok": False, "status": "HOST_EXCEPTION_OUTCOME_UNKNOWN",
                      "error": type(error).__name__, "message": str(error),
                      "arguments": deepcopy(action["arguments"]), "retry_authorized": False}
            self.status = "HOST_EXCEPTION_OUTCOME_UNKNOWN"
        data = {"step": self.steps, "status": self.status, "action": action,
                "raw_model_ref": raw_ref, "action_ref": action_ref,
                "tool_invoked": True, "tool_result": deepcopy(result),
                "started_at_utc": started, "finished_at_utc": utc_now()}
        receipt = self._receipt(data)
        self.conversation += [
            {"role": "assistant", "tool_call": deepcopy(action)},
            {"role": "tool", "name": action["name"], "content": canonical(result)}]
        self.model.observe(deepcopy(action), deepcopy(result))
        return receipt

    def run(self):
        while self.status in {"READY", "RUNNING"} and self.steps < self.max_steps:
            self.step()
        if self.status == "RUNNING" and self.steps == self.max_steps:
            self.status = "STEP_LIMIT"
        return self.report()

    def report(self):
        return {"kind": "pure_native_agent_report_v1", "status": self.status,
                "steps": self.steps, "max_steps": self.max_steps, "final": self.final,
                "journal_directory": str(self.directory), "trace": deepcopy(self.trace),
                "decision_routing": False, "silent_output_repair": False,
                "automatic_flattening": False, "performance_established": False}


class DeclaredDiagnosticRuntime(DiagnosticRuntime, DeclaredAgentRuntime):
    """Original diagnostic wrapper over exact declared-operation step semantics."""
    pass


class DeclaredHarmonySamplerModel(HarmonySamplerModel):
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
        validate_manifest(tool_definitions)
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
        self.names = {"functions." + spec["name"] for spec in specs}

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
            return validate_declared_action({"kind": "final", "text": self.segment["final_text"]}, self.tool_definitions)
        names = {"functions." + spec["name"]: spec["name"] for spec in validate_manifest(self.tool_definitions).values()}
        if self.segment["tool_name"] not in names:
            raise ValueError("exact declared operation recipient required")
        return validate_declared_action({"kind": "tool", "name": names[self.segment["tool_name"]],
            "arguments": deepcopy(self.segment["arguments"])}, self.tool_definitions)

    def observe(self, action, result):
        import tinker
        from tinker_cookbook.renderers.base import RenderContext
        from training import v7_protocol as protocol
        from training import direct_native_episode as native
        validate_declared_action(action, self.tool_definitions)
        if (self.segment is None or self.segment["kind"] != "tool"
                or self.segment["tool_name"] != "functions." + action["name"]
                or not exact_json_equal(action["arguments"], self.segment["arguments"])):
            raise ValueError("exact native model tool arguments required")
        native.require(native.parse_direct_segment(self.segment["raw_tokens"], self.renderer, self.names)
                       == self.segment, "DIRECT_TOOL_TRACE_CHANGED")
        content = canonical(result)
        protocol._safe_content(content, self.renderer)
        rendered = self.renderer.render_message({"role": "tool", "name": self.segment["tool_name"],
            "content": content, "tool_call_id": "pure-native:" + str(len(self.training_steps))},
            RenderContext(idx=0, is_last=True))
        result_tokens = rendered.header.tokens + [token for chunk in rendered.output for token in chunk.tokens]
        suffix = self.renderer.tokenizer.encode("<|start|>assistant", add_special_tokens=False)
        native.require(self.prompt.to_ints()[-len(suffix):] == suffix, "DIRECT_NATIVE_PREFIX_CHANGED")
        tokens = self.prompt.to_ints() + self.segment["raw_tokens"] + result_tokens + suffix
        native.require(len(tokens) < native.MAX_SEQUENCE, "DIRECT_CONTINUATION_CONTEXT_EXHAUSTED")
        self.prompt = tinker.ModelInput.from_ints(tokens)
        self._last_observation = {"role": "tool", "name": action["name"], "content": content}
        self.segment = None
