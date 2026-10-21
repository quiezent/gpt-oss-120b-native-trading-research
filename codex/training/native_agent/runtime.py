"""A model chooses every action; the runtime records and executes its output.

There are no roles, regime classifiers, trade templates, action replacements,
forced entries/exits, or response repair. Hosts are mechanical tool transports.
Importing this module opens no provider, broker, account or session.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Protocol


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def strict_json(raw: bytes) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=nonfinite)


def validate_action(action: Any) -> dict:
    if type(action) is not dict:
        raise ValueError("model action must be an object")
    if action.get("kind") == "tool":
        if (set(action) != {"kind", "name", "arguments"}
                or action["name"] != "pa_tws" or type(action["arguments"]) is not dict):
            raise ValueError("expected exact tool action with name pa_tws and object arguments")
    elif action.get("kind") == "final":
        if set(action) != {"kind", "text"} or type(action["text"]) is not str:
            raise ValueError("expected exact final action with text")
    else:
        raise ValueError("model must emit tool or final; no inferred action")
    canonical(action)
    return deepcopy(action)


def parse_json_action(raw: bytes) -> dict:
    return validate_action(strict_json(raw))


class Model(Protocol):
    def generate(self, conversation: list[dict]) -> bytes: ...
    def parse(self, raw: bytes) -> dict: ...
    def observe(self, action: dict, result: dict) -> None: ...


class Host(Protocol):
    def declaration(self) -> dict: ...
    def execute(self, arguments: dict) -> dict: ...


class LocalGenerationBudgetExhausted(RuntimeError):
    """Known local token/context ceiling before provider dispatch, no action."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = deepcopy(details or {})


class JsonCompletionModel:
    """Adapter for a caller-owned sampler returning exact UTF-8 model bytes."""

    def __init__(self, generate: Callable[[list[dict]], bytes]):
        self._generate = generate

    def generate(self, conversation):
        return self._generate(deepcopy(conversation))

    def parse(self, raw):
        return parse_json_action(raw)

    def observe(self, action, result):
        pass


class PureAgentRuntime:
    """One conversation, bounded model steps, append-only raw output receipts.

    ``host.declaration`` is explicitly an execution constraint, included in the
    initial prompt. It must not contain provider/account credentials. Each
    call receives exact parsed arguments. A refused valid call remains visible
    in the same conversation. Invalid completions stop without action/repair.
    Reaching the step ceiling does not submit, cancel, or flatten anything.
    The supplied model/host remain caller-owned and are never reopened here.
    """

    def __init__(self, *, model: Model, host: Host, journal_directory: str | Path,
                 system_prompt: str, user_prompt: str, max_steps: int = 32):
        if type(max_steps) is not int or not 1 <= max_steps <= 4096:
            raise ValueError("finite max_steps required")
        if type(system_prompt) is not str or type(user_prompt) is not str:
            raise ValueError("text prompts required")
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
            action = validate_action(self.model.parse(raw))
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
            {"role": "tool", "name": "pa_tws", "content": canonical(result)}]
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
