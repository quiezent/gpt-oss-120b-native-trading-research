"""Lossless public native-result presentation and terminal diagnostics.

Stage only; proposed canonical path training/native_agent/compact_presentation.py.
No trade selection, output/argument repair, truncation, provider or broker setup.
"""
from copy import deepcopy
import ast
import hashlib
import math
from pathlib import Path
import traceback

from training.native_agent.runtime import (
    PureAgentRuntime, canonical, strict_json, utc_now, write_new)
from training.native_agent.harmony_model import HarmonySamplerModel


def exact_json_equal(left, right):
    """Strict JSON types, including bool/int and the sign of floating zero."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(exact_json_equal(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left.hex() == right.hex()
    return left == right


def compact_native_result(result):
    """Replace only a proved redundant JSON stdout representation.

    Every other wrapper/result field is retained exactly. Structured native
    errors remain visible in result. Ambiguous/nonJSON stdout is unchanged.
    Original byte-for-byte stdout/stderr already reside in raw_receipt_ref.
    """
    presented = deepcopy(result)
    if (type(result) is not dict or type(result.get("stdout")) is not str
            or "result" not in result or result.get("parse_error") is not None
            or "stdout_presentation" in result):
        return presented
    try:
        parsed = strict_json(result["stdout"].encode("utf-8"))
    except (ValueError, UnicodeError):
        return presented
    if not exact_json_equal(parsed, result["result"]):
        return presented
    raw_ref = result.get("raw_receipt_ref")
    if not isinstance(raw_ref, dict) or not {"path", "sha256"} <= raw_ref.keys():
        return presented
    stdout = presented.pop("stdout")
    presented["stdout_presentation"] = {
        "representation": "IDENTICAL_STRICT_JSON_VALUE_ALREADY_PRESENT_IN_RESULT",
        "raw_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "raw_stdout_utf8_bytes": len(stdout.encode("utf-8")),
        "original_bytes_retained_in_raw_receipt_ref": True,
        "stderr_unchanged": True,
    }
    return presented


def native_constructor_defaults(cli_ref):
    path = Path(cli_ref["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cli_ref["sha256"]:
        raise ValueError("authenticated canonical order constructor required")
    tree = ast.parse(raw, filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_stock_order")
    values = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "order" and target.attr in {"transmit", "outsideRth"}):
                    values[target.attr] = ast.literal_eval(node.value)
    if set(values) != {"transmit", "outsideRth"} or any(type(value) is not bool for value in values.values()):
        raise ValueError("literal authentic native transmit/session construction fields required")
    return {"source_ref": deepcopy(cli_ref), "constructor": "make_stock_order",
        "transmit": values["transmit"], "outside_rth": values["outsideRth"],
        "construction_defaults_are_acknowledgements_or_fills": False,
        "transmit_is_a_risk_policy_field": False}


def policy_tool_definition(tool_definition, policy, *, host_max_quantity):
    tool = deepcopy(tool_definition)
    function = tool[0]["function"]
    old = "Native order_type: LMT,MKT,STP,'STP LMT',MIT,LIT,TRAIL,'TRAIL LIMIT'; native type-specific price requirements apply."
    if old not in function["description"]:
        raise ValueError("authenticated native signature text required")
    function["description"] = function["description"].replace(old,
        "Order types admitted by the actual risk policy are " + canonical(policy["order_types"]) + "; native type-specific price requirements apply.")
    function["description"] += (
        " The host supplies the actual native risk-policy fields and their source hash verbatim, separately from narrower host quantity/notional bounds."
        " Native order construction sets transmit=true; this is not a risk-policy field or proof of submission/fill."
        " Unambiguous strict JSON stdout, including structured errors, is represented once as the identical result value."
        " Original transport stdout/stderr bytes and full wrapper receipt are retained at disclosed references; nonJSON/ambiguous stdout and all stderr remain visible."
        " This presentation changes no native model argument, receipt field value or economic decision.")
    properties = function["parameters"]["properties"]
    properties["order_type"]["enum"] = deepcopy(policy["order_types"])
    properties["side"]["enum"] = deepcopy(policy["allowed_sides"])
    properties["tif"]["enum"] = deepcopy(policy["allowed_tif"])
    properties["quantity"].update(minimum=1, maximum=min(host_max_quantity, policy["max_quantity"]))
    properties["limit"]["type"] = "number"
    properties["limit"]["description"] = (
        "Native price limits may be fractional. Commands that use limit as a count require integer values."
        " The native CLI validates the command-specific meaning; exact model arguments are never converted or rounded.")
    return tool


class CompactNativeHost:
    """Declared policy + public encoding around the unchanged exact CLI host."""
    def __init__(self, native):
        self.native, self.directory, self.config = native, Path(native.directory), native.config
        self.ordinal = 0
        policy_path = Path(self.config.risk_policy).resolve()
        raw = policy_path.read_bytes()
        self.policy = strict_json(raw)
        self.policy_ref = {"path": str(policy_path), "sha256": hashlib.sha256(raw).hexdigest()}
        self.cli_ref = {"path": str(Path(self.config.cli).resolve()), "sha256": self.config.cli_sha256}
        self.defaults = native_constructor_defaults(self.cli_ref)

    def declaration(self):
        result = deepcopy(self.native.declaration())
        result["native_risk_policy"] = {"source_ref": deepcopy(self.policy_ref), "fields": deepcopy(self.policy)}
        result["native_order_constructor_defaults"] = deepcopy(self.defaults)
        result["public_result_presentation"] = {
            "strict_identical_stdout_JSON_represented_once": True,
            "original_transport_bytes_and_full_wrapper_retained": True,
            "all_stderr_and_nonJSON_stdout_visible": True,
            "structured_error_fields_visible": True,
            "no_summaries_or_history_trimming": True,
            "model_arguments_preserved": True}
        return result

    def execute(self, arguments):
        result = self.native.execute(deepcopy(arguments))
        self.ordinal += 1
        original = write_new(self.directory / f"presentation-{self.ordinal:04d}-original.json", canonical(result).encode())
        if "original_host_result_ref" in result:
            raise ValueError("reserved presentation reference collision; original host result retained")
        presented = compact_native_result(result)
        presented["original_host_result_ref"] = original
        return presented


class CompactHarmonySamplerModel(HarmonySamplerModel):
    """Same parser/prefix/history/32K boundary; compact tool JSON whitespace."""
    def observe(self, action, result):
        import tinker
        from tinker_cookbook.renderers.base import RenderContext
        from training import v7_protocol as protocol
        from training import direct_native_episode as native
        if (self.segment is None or self.segment["kind"] != "tool"
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
        self._last_observation = {"role": "tool", "name": "pa_tws", "content": content}
        self.segment = None


def exception_diagnostic(error, *, phase):
    # Frame metadata only: no locals, source-line text, tensors or model analysis.
    frames = [{"filename": frame.filename, "line": frame.lineno, "function": frame.name}
              for frame in traceback.extract_tb(error.__traceback__)]
    return {"kind": "local_native_exception_diagnostic_v1", "phase": phase,
        "error_type": type(error).__name__, "message": str(error), "traceback_frames": frames,
        "locals_or_private_reasoning_included": False, "recorded_at_utc": utc_now()}


class DiagnosticRuntime(PureAgentRuntime):
    """Preserve a successful tool receipt before diagnosing continuation failure."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.terminal_diagnostic = None

    def step(self):
        try:
            return super().step()
        except Exception as error:
            last = self.trace[-1] if self.trace and self.trace[-1].get("step") == self.steps else None
            after = last is not None and last.get("tool_invoked") is True
            known = after and type(error) is ValueError and str(error) == "DIRECT_CONTINUATION_CONTEXT_EXHAUSTED"
            # A local append failure cannot erase a prior uncertain tool outcome.
            self.status = ("HOST_EXCEPTION_OUTCOME_UNKNOWN" if self.status == "HOST_EXCEPTION_OUTCOME_UNKNOWN"
                else "LOCAL_MODEL_BUDGET_EXHAUSTED" if known else "STEP_EXCEPTION_NO_AUTOMATIC_RETRY")
            self.terminal_diagnostic = {**exception_diagnostic(error,
                phase="AFTER_DURABLE_TOOL_RECEIPT" if after else "STEP_EXCEPTION"),
                "status": self.status, "step": self.steps,
                "already_recorded_receipt_ref": None if last is None else last.get("receipt_ref"),
                "current_tool_receipt_preserved": after, "next_provider_dispatch": False,
                "automatic_retry": False, "automatic_resume": False}
            # Separate exclusive sidecar: never overwrite step-N receipt/raw.
            ref = write_new(self.directory / f"step-{self.steps:04d}-terminal-exception.json",
                canonical(self.terminal_diagnostic).encode())
            self.terminal_diagnostic["diagnostic_ref"] = ref
            return self.report()

    def report(self):
        result = super().report()
        result["terminal_diagnostic"] = deepcopy(self.terminal_diagnostic)
        return result
