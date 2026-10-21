"""Staged single-conversation framework API candidate; no provider lifecycle.

This file is not installed or funded. Importing creates no broker or provider.
The caller owns the model, financial adapter and broker callbacks. There is no
role sequence, teacher answer, trade template, forced close, or FINAL restart.
The existing Harmony/runtime tool address pa_tws is retained only as a protocol
address; framework_* commands select the declared Python framework API.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

from training.native_agent.clock_host import ClockWaitHost, temporal_tool_definition
from training.native_agent.runtime import PureAgentRuntime, canonical, strict_json, utc_now, write_new

BOUNDARY = {"host": "127.0.0.1", "port": 4002, "account": "PAPER_ACCOUNT", "client_id": 9901}
INTENT_FIELDS = {"intent_id", "account", "conid", "symbol", "primary_exchange", "side",
                 "quantity", "limit_price", "order_type", "tif", "outside_rth"}
KEYS = {"framework_status": set(), "framework_snapshot": {"contracts"},
        "framework_preview": {"intent"}, "framework_submit": {"intent"},
        "framework_reconcile": {"intent"}, "framework_cancel": {"order_id", "order_ref", "conid"}}
MUTATIONS = {"framework_submit", "framework_cancel"}


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value):
    if type(value) not in {str, int, float}:
        raise ValueError("finite monetary value required; booleans are not prices")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("finite monetary value required") from None
    if not result.is_finite():
        raise ValueError("finite monetary value required")
    return result


def framework_tool_definition():
    intent = {"type": "object", "additionalProperties": False,
        "required": sorted(INTENT_FIELDS), "properties": {
            "intent_id": {"type": "string"}, "account": {"type": "string", "enum": [BOUNDARY["account"]]},
            "conid": {"type": "integer", "minimum": 1}, "symbol": {"type": "string"},
            "primary_exchange": {"type": "string"}, "side": {"type": "string", "enum": ["BUY", "SELL"]},
            "quantity": {"type": "integer", "minimum": 1},
            "limit_price": {"type": ["number", "string"]}, "order_type": {"type": "string", "enum": ["LMT"]},
            "tif": {"type": "string", "enum": ["DAY"]}, "outside_rth": {"type": "boolean", "enum": [False]}}}
    tool = [{"type": "function", "function": {"name": "pa_tws",
        "description": (
            "This address selects the declared Python framework API, not direct CLI argument generation. "
            "Choose every operation and complete intent yourself. framework_status takes command only; "
            "framework_snapshot takes command and contracts; framework_preview, framework_submit and framework_reconcile "
            "take command and intent; framework_cancel takes command, order_id, order_ref, conid. "
            "An intent has exactly the declared eleven fields. Account is fixed paper PAPER_ACCOUNT; the framework supports "
            "whole-share USD US-stock LMT DAY outside_rth=false. Quantity and price are never supplied, changed or rounded for you. "
            "Order reference is the existing mechanical PaperTWSBroker hash of your intent_id; choose a stable immutable identity. "
            "A submit call delegates unchanged intent to the framework broker; canonical native connected preflight validates "
            "the caller's disclosed risk mandate. Refusals and raw underlying receipts are returned in the same conversation. "
            "Snapshot obtains current non-atomic framework account/positions/orders/executions/contracts/quotes; it does not "
            "establish complete account history. Reconcile never resubmits. Cancellation requires exact PA-owned identity. "
            "Mutations are durable once-only; an uncertain attempt blocks further mutation without manufacturing an action. "
            "No role sequence, host trade price/deadline, entry quota, forced exit or FINAL restart exists."),
        "parameters": {"type": "object", "additionalProperties": False, "required": ["command"],
            "properties": {"command": {"type": "string", "enum": sorted(KEYS)}, "intent": intent,
                "contracts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "required": ["conid", "symbol", "primary_exchange"], "properties": {
                        "conid": {"type": "integer", "minimum": 1}, "symbol": {"type": "string"},
                        "primary_exchange": {"type": "string"}}}},
                "order_id": {"type": "integer", "minimum": 1}, "order_ref": {"type": "string"},
                "conid": {"type": "integer", "minimum": 1}}}}}]
    return temporal_tool_definition(tool)


class FrameworkHost:
    """Exact generated API calls plus schema/paper policy and durable ownership.

    callbacks command receives the complete unchanged command object. It must
    return result + newly captured underlying process receipts. The actual
    canonical adapter below implements this contract; tests inject fakes.
    No callback name or economic content is inferred from current observations.
    """
    def __init__(self, *, directory, callbacks, broker_declaration, admission_check,
                 allow_mutations=False, max_quantity=3, max_notional_usd="2500"):
        if (set(callbacks) != set(KEYS) or not all(callable(value) for value in callbacks.values())
                or not callable(admission_check) or type(allow_mutations) is not bool
                or type(max_quantity) is not int or max_quantity < 1 or number(max_notional_usd) <= 0
                or broker_declaration.get("boundary") != BOUNDARY):
            raise ValueError("exact callbacks, paper boundary and finite explicit limits required")
        canonical(broker_declaration)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.callbacks, self.admission_check = dict(callbacks), admission_check
        self.broker_declaration = deepcopy(broker_declaration)
        self.allow_mutations, self.max_quantity = allow_mutations, max_quantity
        self.max_notional_usd = max_notional_usd
        self.ordinal, self.receipts = 0, []
        self.lock = threading.Lock()

    def declaration(self):
        return {"tool_address": "pa_tws", "interface": "PYTHON_FRAMEWORK_API", "boundary": deepcopy(BOUNDARY),
            "operations": sorted(KEYS), "allow_mutations": self.allow_mutations,
            "maximum_quantity": self.max_quantity, "modeled_absolute_limit_notional_usd": self.max_notional_usd,
            "actual_fill_cost_guaranteed_by_modeled_limit": False,
            "framework_broker": deepcopy(self.broker_declaration), "model_arguments_preserved": True,
            "economic_routing": False, "role_contexts": False, "teacher_trades": False,
            "forced_exit": False, "final_is_terminal": True,
            "lifecycle_evidence": "Actual model-selected entry and exit fills, attributable inventory and no residual owned orders are distinct from callback/service closure. Missing fees/PnL remain unknown."}

    def _validate(self, request):
        if type(request) is not dict or request.get("command") not in KEYS:
            raise ValueError("declared framework command required")
        canonical(request)
        command = request["command"]
        if set(request) != KEYS[command] | {"command"}:
            raise ValueError("exact command-specific fields required; no host fields or silent argument repair")
        if "intent" in request:
            value = request["intent"]
            if type(value) is not dict or set(value) != INTENT_FIELDS:
                raise ValueError("exact eleven-field framework intent required")
            if (value["account"] != BOUNDARY["account"] or value["side"] not in {"BUY", "SELL"}
                    or value["order_type"] != "LMT" or value["tif"] != "DAY" or value["outside_rth"] is not False
                    or type(value["quantity"]) is not int or not 1 <= value["quantity"] <= self.max_quantity
                    or type(value["conid"]) is not int or value["conid"] <= 0
                    or any(type(value[key]) is not str or not value[key] or len(value[key]) > 256
                           for key in ("intent_id", "symbol", "primary_exchange"))):
                raise ValueError("framework intent exceeds disclosed paper schema/quantity boundary")
            price = number(value["limit_price"])
            if price <= 0 or abs(price) * value["quantity"] > number(self.max_notional_usd):
                raise ValueError("framework intent exceeds disclosed modeled notional boundary")
        if command == "framework_snapshot":
            contracts = request["contracts"]
            if type(contracts) is not list or len(contracts) > 32:
                raise ValueError("at most 32 model-selected contract observations per snapshot")
            for contract in contracts:
                if (type(contract) is not dict or set(contract) != {"conid", "symbol", "primary_exchange"}
                        or type(contract["conid"]) is not int or contract["conid"] <= 0
                        or any(type(contract[key]) is not str or not contract[key] or len(contract[key]) > 256
                               for key in ("symbol", "primary_exchange"))):
                    raise ValueError("exact model-selected contract identity required")
        if command == "framework_cancel":
            if (any(type(request[key]) is not int or request[key] <= 0 for key in ("order_id", "conid"))
                    or type(request["order_ref"]) is not str or not request["order_ref"].startswith("pa:")
                    or len(request["order_ref"]) > 256):
                raise ValueError("exact PA-owned cancellation identity required")
        return command

    @staticmethod
    def mutation_identity(request):
        command = request["command"]
        if command == "framework_submit":
            from trader_runtime.tws_broker import PaperTWSBroker
            value = request["intent"]
            return {"kind": "submission", "order_ref": PaperTWSBroker.order_ref(value["intent_id"])}
        return {"kind": "cancel",
                **{key: request[key] for key in ("conid", "order_id", "order_ref")}}

    def seed_prior_mutation(self, identity, source_ref):
        """Root supplies authenticated prior identities; no automatic discovery."""
        path = Path(source_ref["path"])
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != source_ref["sha256"]:
            raise ValueError("prior mutation receipt source changed")
        if (identity != strict_json(source)["identity"]
                or path.name != "once-" + digest(identity) + ".json"
                or identity.get("kind") not in {"submission", "cancel"}):
            raise ValueError("exact canonical native once identity/source required")
        return write_new(self.directory / ("once-" + digest(identity) + ".json"),
            source)

    def _halt(self, reason, intent_ref):
        path = self.directory / "mutation-halt.json"
        if not path.exists():
            write_new(path, canonical({"reason": reason, "intent_ref": intent_ref,
                "reads_permitted": True, "automatic_retry": False, "economic_action_supplied": False}).encode())

    def execute(self, arguments):
        with self.lock:
            self.ordinal += 1
            stem = f"framework-{self.ordinal:04d}"
            request = deepcopy(arguments)
            intent_ref = write_new(self.directory / (stem + "-intent.json"),
                canonical({"arguments": request, "started_at_utc": utc_now(), "api_invoked": False}).encode())
            invoked = False
            mutating = type(request) is dict and request.get("command") in MUTATIONS
            try:
                command = self._validate(request)
                if mutating:
                    if not self.allow_mutations:
                        raise ValueError("framework paper writes disabled by caller scope")
                    if (self.directory / "mutation-halt.json").exists():
                        raise ValueError("uncertain mutation ownership retained; further mutations halted")
                self.admission_check()
                if mutating:
                    identity = self.mutation_identity(request)
                    write_new(self.directory / ("once-" + digest(identity) + ".json"),
                        canonical({"identity": identity, "intent_ref": intent_ref,
                                   "model_request_sha256": digest(request)}).encode())
            except (ValueError, FileExistsError) as error:
                result = {"ok": False, "status": "FRAMEWORK_REFUSAL", "message": str(error),
                          "result": None, "raw_transport_receipts": []}
            else:
                invoked = True
                try:
                    observation = self.callbacks[command](deepcopy(request))
                    if (type(observation) is not dict
                            or set(observation) != {"result", "raw_transport_receipts", "transport_attempted"}
                            or type(observation["transport_attempted"]) is not bool
                            or type(observation["raw_transport_receipts"]) is not list):
                        raise TypeError("callback must preserve result and underlying transport receipts")
                    canonical(observation)
                    unknown = (mutating and observation["transport_attempted"] and (not observation["raw_transport_receipts"]
                        or any(row.get("uncertain") for row in observation["raw_transport_receipts"])
                        or (type(observation["result"]) is dict and observation["result"].get("outcome")
                            in {"UNCERTAIN_DO_NOT_RETRY", "SUBMISSION_UNCERTAIN_DO_NOT_RETRY"})
                        or (type(observation["result"]) is dict and observation["result"].get("status")
                            in {"OUTCOME_UNKNOWN", "HOST_EXCEPTION_OUTCOME_UNKNOWN"})))
                    refused = type(observation["result"]) is dict and observation["result"].get("ok") is False
                    result = {"ok": not unknown and not refused,
                        "status": "FRAMEWORK_OUTCOME_UNKNOWN" if unknown else "FRAMEWORK_API_REFUSAL" if refused else "FRAMEWORK_API_RESULT", **observation}
                    if unknown:
                        self._halt("UNKNOWN_FRAMEWORK_MUTATION", intent_ref)
                except BaseException as error:
                    attempted = getattr(error, "transport_attempted", mutating)
                    if mutating and attempted:
                        self._halt("FRAMEWORK_CALLBACK_EXCEPTION", intent_ref)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    result = {"ok": False, "status": "FRAMEWORK_OUTCOME_UNKNOWN" if mutating and attempted else "FRAMEWORK_API_REFUSAL",
                        "error_type": type(error).__name__, "message": str(error), "result": None,
                        "transport_attempted": bool(attempted),
                        "raw_transport_receipts": deepcopy(getattr(error, "raw_transport_receipts", []))}
            raw = {"kind": "observed_framework_api_receipt_v1", **result, "arguments": request,
                "api_invoked": invoked, "interface": "PYTHON_FRAMEWORK_API", "intent_ref": intent_ref,
                "finished_at_utc": utc_now(), "retry_authorized": False, "broker_close_implied": False}
            try:
                raw_ref = write_new(self.directory / (stem + "-raw.json"), canonical(raw).encode())
            except BaseException:
                if mutating and invoked:
                    self._halt("FRAMEWORK_RECEIPT_PERSISTENCE_FAILED", intent_ref)
                raise
            retained = {**raw, "raw_receipt_ref": raw_ref}
            self.receipts.append(deepcopy(retained))
            return retained


class RecordingFrameworkRunner:
    """Capture canonical subprocess bytes before the framework interprets them.

    Uses an already source-pinned NativeCliHost canonical-worker transport.
    The broker's economic argv is retained; only interpreter -B/-X utf8 flags
    are added for the existing canonical worker protocol. No direct subprocess
    fallback exists. The worker owns canonical basic-read locking and leaves
    history/mutation commands' locks to native pa_tws.
    """
    def __init__(self, native_host, *, directory):
        self.native = native_host
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.receipts, self.ordinal, self.attempts = [], 0, 0

    def __call__(self, argv, **kwargs):
        config = self.native.config
        if (type(argv) is not list or len(argv) < 5
                or Path(argv[0]).resolve() != Path(config.interpreter).resolve()
                or Path(argv[1]).resolve() != Path(config.cli).resolve()
                or argv[2] != "--timeout" or kwargs.get("shell") is not False):
            raise ValueError("exact source-pinned framework broker argv required")
        self.native._assert_boundary()
        self.ordinal += 1
        stem = f"framework-process-{self.ordinal:04d}"
        adapted = [argv[0], "-B", "-X", "utf8", *argv[1:]]
        intent_ref = write_new(self.directory / (stem + "-intent.json"), canonical({
            "original_broker_argv": argv, "canonical_worker_argv": adapted,
            "source_ref": {"path": str(config.cli), "sha256": config.cli_sha256},
            "started_at_utc": utc_now(), "environment_omitted": True}).encode())
        error = None
        try:
            self.attempts += 1
            value = self.native.transport(adapted, cwd=str(config.workspace), env=kwargs["env"],
                shell=False, capture_output=True, timeout=config.child_deadline_seconds)
            stdout, stderr, returncode = value.stdout, value.stderr, value.returncode
            if type(stdout) is not bytes or type(stderr) is not bytes:
                raise TypeError("canonical worker transport must return exact bytes")
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, returncode, error = exc.stdout or b"", exc.stderr or b"", None, exc
        except BaseException as exc:
            stdout, stderr, returncode, error = b"", b"", None, exc
        raw = {"kind": "framework_canonical_process_receipt_v1", "original_broker_argv": argv,
            "canonical_worker_argv": adapted, "intent_ref": intent_ref,
            "stdout_base64": base64.b64encode(stdout).decode(), "stderr_base64": base64.b64encode(stderr).decode(),
            "returncode": returncode, "uncertain": error is not None,
            "error_type": None if error is None else type(error).__name__, "finished_at_utc": utc_now()}
        raw_ref = write_new(self.directory / (stem + "-raw.json"), canonical(raw).encode())
        public = {"argv": adapted, "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"), "returncode": returncode,
            "uncertain": error is not None, "raw_receipt_ref": raw_ref, "intent_ref": intent_ref,
            "cli_invoked": True, "native_command": argv[4]}
        self.receipts.append(public)
        if error is not None:
            error.raw_transport_receipts = [deepcopy(public)]
            error.transport_attempted = True
            raise error
        return SimpleNamespace(stdout=public["stdout"], stderr=public["stderr"], returncode=returncode)


def seed_native_prior_receipts(native_host, refs):
    """Copy all authentic prior native once bytes, including the historical 18.

    Root must freeze the exhaustive later set after native compact closes.
    This helper verifies supplied bytes/identity filenames, not broker outcomes.
    """
    if type(refs) is not list or len(refs) < 18:
        raise ValueError("at least all eighteen historical native identities required before framework wiring")
    seen = set()
    for ref in refs:
        path = Path(ref["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("prior native once source changed")
        record = strict_json(raw)
        identity = record["identity"]
        if type(identity) is not dict or set(identity) not in ({"kind", "order_ref"}, {"kind", "order_ref", "order_id", "conid"}):
            raise ValueError("canonical prior native mutation identity required")
        if identity["kind"] not in {"submission", "cancel"}:
            raise ValueError("canonical prior mutation family required")
        name = "once-" + digest(identity) + ".json"
        if path.name != name or name in seen:
            raise ValueError("prior native identity filename mismatch or duplicate")
        seen.add(name)
        write_new(native_host.directory / name, raw)
    return write_new(native_host.directory / "prior_framework_native_seed_manifest.json", canonical({
        "kind": "authentic_prior_native_identity_seeds_for_framework_v1", "source_refs": deepcopy(refs),
        "seed_count": len(refs), "old_actions_dispatched": False,
        "later_native_continuation_identities_must_also_be_frozen": True}).encode())


def canonical_framework_callbacks(*, broker, native_host, recording_runner, portfolio_mandate,
                                  quote_execution_profile, prior_native_once_refs):
    """Mechanical existing broker API binding. No V8 supervisors/trial planners.

    Mandate/profile are caller-owned execution constraints, disclosed by host.
    Submission connected preflight remains native. No close deadline/price,
    owned-exit ledger, candidate shortlist or replacement order is fabricated.
    """
    from trader_runtime.tws_broker import PaperTWSBroker
    from trading_desk.v8_tws_control import preview_exact_paper_profile, submit_exact_paper_profile
    if type(broker) is not PaperTWSBroker or broker._runner is not recording_runner:
        raise ValueError("canonical framework broker with the recorded source-pinned transport required")
    if Path(broker.risk_policy_path).resolve() != Path(native_host.config.risk_policy).resolve():
        raise ValueError("framework and source-pinned native transport risk policies must agree")
    seed_native_prior_receipts(native_host, prior_native_once_refs)
    canonical(portfolio_mandate)
    mandate = deepcopy(portfolio_mandate)
    def callback(request):
        before = len(recording_runner.receipts)
        before_attempts = recording_runner.attempts
        try:
            command = request["command"]
            if command == "framework_status":
                result = broker.status()
            elif command == "framework_snapshot":
                result = broker.snapshot(deepcopy(request["contracts"]))
            elif command == "framework_preview":
                result = preview_exact_paper_profile(broker, deepcopy(request["intent"]),
                    quote_execution_profile=quote_execution_profile, mandate=deepcopy(mandate))
            elif command == "framework_submit":
                if not native_host.config.allow_mutations:
                    raise ValueError("source-pinned native transport paper writes disabled")
                result = submit_exact_paper_profile(broker, deepcopy(request["intent"]), confirm="SUBMIT_PAPER_ORDER",
                    mandate=deepcopy(mandate), quote_execution_profile=quote_execution_profile)
            elif command == "framework_reconcile":
                result = broker.reconcile(deepcopy(request["intent"]))
            elif command == "framework_cancel":
                result = native_host.execute({"command": "cancel-pa-order",
                    **{key: deepcopy(request[key]) for key in ("order_id", "order_ref", "conid")}})
                return {"result": result, "raw_transport_receipts": [deepcopy(result)],
                        "transport_attempted": result.get("cli_invoked") is True}
            else:
                raise ValueError("exact declared framework operation required")
            return {"result": result, "raw_transport_receipts": deepcopy(recording_runner.receipts[before:]),
                    "transport_attempted": recording_runner.attempts > before_attempts}
        except BaseException as error:
            error.raw_transport_receipts = deepcopy(recording_runner.receipts[before:])
            error.transport_attempted = (request["command"] == "framework_cancel"
                                         or recording_runner.attempts > before_attempts)
            raise
    return {name: callback for name in KEYS}


@dataclass(frozen=True)
class FrameworkCandidate:
    output_directory: Path
    max_steps: int = 128
    max_wait_seconds: int = 86400
    allow_mutations: bool = False
    max_quantity: int = 3
    max_notional_usd: str = "2500"
    task: str = "Use the declared Python framework API to pursue positive NAV returns over days through three months. Choose every action, complete order intent and review/wait time yourself. Report actual evidence and remaining uncertainty."


def run_framework_candidate(config, *, model, callbacks, broker_declaration, admission_check,
                            runtime_class=PureAgentRuntime, prior_mutation_receipts=()):
    """One caller-owned model; no bootstrap/reserve/close/reopen or host exits.

    Production release still needs root's sampled-context cost claims, owned
    SDK lifecycle, checkpoint/source freeze and exclusive launch helper.
    This candidate exposes those gaps; it is not an independently live launcher.
    Supply reviewed CompactHarmonySamplerModel/DiagnosticRuntime after their
    real installation; this adapter never rewrites model history or outputs.
    """
    if type(config.max_steps) is not int or not 1 <= config.max_steps <= 128:
        raise ValueError("finite framework decision ceiling in 1..128 required")
    if hasattr(model, "tool_definitions") and model.tool_definitions != framework_tool_definition():
        raise ValueError("caller-owned native model must have the exact framework tool definition")
    output = Path(config.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    framework = FrameworkHost(directory=output / "framework_receipts", callbacks=callbacks,
        broker_declaration=broker_declaration, admission_check=admission_check,
        allow_mutations=config.allow_mutations, max_quantity=config.max_quantity,
        max_notional_usd=config.max_notional_usd)
    for row in prior_mutation_receipts:
        framework.seed_prior_mutation(row["identity"], row["source_ref"])
    host = ClockWaitHost(framework, max_wait_seconds=config.max_wait_seconds)
    class AdmittedModel:
        def generate(self, conversation):
            host.assert_within_session()
            admission_check()
            return model.generate(deepcopy(conversation))
        def parse(self, raw):
            return model.parse(raw)
        def observe(self, action, result):
            return model.observe(deepcopy(action), deepcopy(result))
    tool_ref = write_new(output / "public_framework_tool.json", canonical(framework_tool_definition()).encode())
    runtime = runtime_class(model=AdmittedModel(), host=host, journal_directory=output / "episode",
        max_steps=config.max_steps, system_prompt=(
            "You are the principal's autonomous GPT-OSS-120B paper model using the declared Python framework API. "
            "Your objective is positive NAV returns over days through three months. In one conversation choose every "
            "economic action, complete order intent and time. No role sequence or host-selected trades are supplied. "
            "Use only disclosed execution constraints. Interpret genuine receipts; distinguish orders from fills, "
            "account NAV from model-attributable returns, observed closure from complete history. Missing fees and "
            "funding/reset coverage remain unknown. FINAL is terminal and schedules nothing."), user_prompt=config.task)
    episode = runtime.run()
    rows = framework.receipts
    report = {"kind": "staged_single_conversation_framework_candidate_result_v1", "status": runtime.status,
        "interface": "PYTHON_FRAMEWORK_API", "episode": episode, "tool_definition_ref": tool_ref,
        "framework_api_invocations": sum(row["api_invoked"] for row in rows),
        "framework_mutation_invocations": sum(row["api_invoked"] and row["arguments"].get("command") in MUTATIONS for row in rows),
        "underlying_cli_invocations": sum(item.get("cli_invoked") is True for row in rows for item in row["raw_transport_receipts"]),
        "local_clock_invocations": sum(row["arguments"].get("command") == "clock" for row in host.local_receipts),
        "local_wait_invocations": sum(row["arguments"].get("command") == "wait_until" for row in host.local_receipts),
        "owned_model_service_closed_by_this_candidate": False, "broker_lifecycle_established": False,
        "profitability_established": False, "automatic_resume": False, "economic_routing": False}
    write_new(output / "report.json", canonical(report).encode())
    return report
