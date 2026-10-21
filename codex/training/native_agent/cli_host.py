"""One canonical pa_tws CLI transport for exact model tool arguments.

The only added argv fields are declared host-owned paper controls. They never
replace a supplied economic field. Commands/order side/price/size are selected
only by the model. Native parser errors are returned as actual child output.
"""
from __future__ import annotations

import ast
import base64
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from types import SimpleNamespace

from .runtime import canonical, strict_json, utc_now, write_new


BOUNDARY = {"host": "127.0.0.1", "port": 4002,
            "account": "PAPER_ACCOUNT", "client_id": 9901}
READ_COMMANDS = frozenset({"status", "account", "positions", "portfolio", "pnl",
    "open-orders", "completed-orders", "executions", "contract", "quote",
    "historical-daily", "news-providers", "historical-news", "news-article",
    "live-news", "broadtape-news"})
ORDER_COMMANDS = frozenset({"preview-stock-limit", "preview-stock-order",
                           "submit-stock-limit", "submit-stock-order"})
MUTATIONS = frozenset({"submit-stock-limit", "submit-stock-order", "cancel-pa-order"})
HOST_FIELDS = frozenset({"host", "port", "account", "client_id", "risk_policy",
    "risk_policy_path", "audit", "audit_path", "lock", "lock_path", "confirm",
    "paper_trial_profile", "portfolio_risk_policy_json", "max_quote_age_seconds",
    "require_live_preflight", "technical_trial_controls_json", "executable",
    "cwd", "env", "environment"})
NATIVE_UNCERTAINTY_PREFIXES = (
    "post-submission failure; order is uncertain and must not be retried: ",
    "post-cancel failure; cancellation is uncertain and must not be retried: ")


def native_uncertain_result(parsed):
    """Recognize pinned pa_tws failure contract, never arbitrary output prose."""
    if type(parsed) is not dict:
        return False
    result = parsed.get("result")
    if type(result) is dict and result.get("outcome") == "SUBMISSION_UNCERTAIN_DO_NOT_RETRY":
        return True
    return (parsed.get("ok") is False and parsed.get("error") in {"PATWSError", "BrokerTimeout"}
            and type(parsed.get("message")) is str
            and parsed["message"].startswith(NATIVE_UNCERTAINTY_PREFIXES))


@dataclass(frozen=True)
class PaperCliConfig:
    workspace: Path
    interpreter: Path
    cli: Path
    cli_sha256: str
    risk_policy: Path
    receipt_directory: Path
    allow_mutations: bool = False
    max_quantity: int = 1000
    max_order_notional_usd: str = "100000"
    allowed_contract_ids: frozenset[int] | None = None
    default_timeout_seconds: float = 15
    child_deadline_seconds: float = 60
    # For instance a caller may explicitly declare matching delayed profile
    # controls. No historical technical-trial policy/window is supplied here.
    root_order_fields: dict = field(default_factory=dict)


def child_environment(mutating: bool) -> dict:
    result = {key: value for key, value in os.environ.items()
              if key.upper() not in {"IBKR_PA_ALLOW_PAPER_ORDER", "IBKR_PA_CONFIRM"}
              and not re.search(r"(?i)(API_KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)$", key)}
    if mutating:
        result["IBKR_PA_ALLOW_PAPER_ORDER"] = "YES"
    return result


class NativeCliHost:
    """Source-pinned native paper CLI, no alternate dispatch path.

    The caller must exclusively own client 9901 while using this host; the
    shared lock serializes instances in this process. The canonical CLI keeps
    its existing cross-process mutation/history locks and broker checks.
    Host receipts preserve exact argv and stdout/stderr before interpretation.
    An attempted mutation key is durable before wire and cannot be resent by
    this host, including after uncertain subprocess or persistence failure.
    No position ownership, fill, net NAV or account-wide history is inferred.
    """
    _client_lock = threading.Lock()

    def __init__(self, config: PaperCliConfig, *, transport=None):
        self.config = config
        self.transport = self._run_canonical_worker if transport is None else transport
        self.directory = Path(config.receipt_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ordinal = max((int(match[1]) for path in self.directory.iterdir()
            if (match := re.fullmatch(r"native-(\d+)-(?:intent|raw|refusal)\.json", path.name))), default=0)
        if (type(config.allow_mutations) is not bool or type(config.max_quantity) is not int
                or config.max_quantity <= 0 or Decimal(config.max_order_notional_usd) <= 0
                or not Decimal(config.max_order_notional_usd).is_finite()
                or not 0 < config.default_timeout_seconds <= 60
                or not 0 < config.child_deadline_seconds <= 90):
            raise ValueError("finite authorized paper limits required")
        if set(config.root_order_fields) - {"paper_trial_profile", "portfolio_risk_policy_json",
                                          "max_quote_age_seconds", "require_live_preflight"}:
            raise ValueError("undeclared host-owned order control")
        self._assert_boundary()

    def _run_canonical_worker(self, argv, **kwargs):
        worker = Path(__file__).with_name("canonical_worker.py").resolve()
        response = subprocess.run([str(Path(self.config.interpreter).resolve()), "-B", "-X", "utf8", str(worker)],
            input=canonical({"argv": argv, "cli_sha256": self.config.cli_sha256,
                             "child_deadline_seconds": self.config.child_deadline_seconds}).encode(),
            cwd=kwargs["cwd"], env=kwargs["env"], shell=False, capture_output=True,
            timeout=self.config.child_deadline_seconds + 10)
        result = strict_json(response.stdout)
        if response.returncode != 0 or result.get("kind") != "canonical_native_worker_v1":
            raise RuntimeError("canonical native worker failed; reconcile before any further mutation")
        stdout = base64.b64decode(result["stdout_base64"], validate=True)
        stderr = base64.b64decode(result["stderr_base64"], validate=True)
        if result["uncertain"]:
            raise subprocess.TimeoutExpired(argv, self.config.child_deadline_seconds, output=stdout, stderr=stderr)
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=result["returncode"])

    def _halt_mutations(self, reason, intent_ref):
        path = self.directory / "mutation-halt.json"
        if not path.exists():
            write_new(path, canonical({"reason": reason, "intent_ref": intent_ref,
                "created_at_utc": utc_now(), "reads_permitted": True,
                "requirement": "reconcile actual order/account state; no implicit new-reference resend"}).encode())

    def _assert_boundary(self):
        config = self.config
        cli, root = Path(config.cli).resolve(), Path(config.workspace).resolve()
        if cli != root / "pa_tws" / "pa_tws.py":
            raise ValueError("canonical pa_tws path required")
        payload = cli.read_bytes()
        if hashlib.sha256(payload).hexdigest() != config.cli_sha256:
            raise ValueError("canonical CLI source changed")
        values = {}
        for node in ast.parse(payload).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"HOST", "PORT", "ACCOUNT", "DEFAULT_CLIENT_ID"}:
                        values[target.id] = ast.literal_eval(node.value)
        if values != {"HOST": BOUNDARY["host"], "PORT": BOUNDARY["port"],
                       "ACCOUNT": BOUNDARY["account"], "DEFAULT_CLIENT_ID": BOUNDARY["client_id"]}:
            raise ValueError("exact fixed paper boundary required")
        if not Path(config.interpreter).is_file() or not Path(config.risk_policy).is_file():
            raise ValueError("actual interpreter and risk policy required")
        policy = strict_json(Path(config.risk_policy).read_bytes())
        if any(policy.get(key) != BOUNDARY[key] for key in ("host", "port", "account")):
            raise ValueError("risk policy paper boundary disagrees")

    def declaration(self):
        return {"tool": "pa_tws", "environment": "IBKR_PAPER", "boundary": deepcopy(BOUNDARY),
                "allowed_commands": sorted(READ_COMMANDS | ORDER_COMMANDS | {"cancel-pa-order"}),
                "allow_mutations": self.config.allow_mutations,
                "max_quantity": self.config.max_quantity,
                "max_order_notional_usd": self.config.max_order_notional_usd,
                "notional_limit_absolute_model_fields": ["limit", "reference_price", "stop", "trail_stop_price"],
                "actual_fill_cost_guaranteed_by_modeled_notional_limit": False,
                "allowed_contract_ids": None if self.config.allowed_contract_ids is None else sorted(self.config.allowed_contract_ids),
                "host_owned_fields": sorted(HOST_FIELDS),
                "host_owned_order_controls": deepcopy(self.config.root_order_fields),
                "default_timeout_seconds": self.config.default_timeout_seconds,
                "native_risk_policy_path": str(Path(self.config.risk_policy).resolve()),
                "model_fields_preserved": True, "silent_term_repair": False,
                "forced_order_on_final_or_step_limit": False}

    @staticmethod
    def _money(value):
        try:
            number = Decimal(str(value))
        except (ValueError, InvalidOperation):
            raise ValueError("positive finite model price required") from None
        if not number.is_finite() or number <= 0:
            raise ValueError("positive finite model price required")
        return number

    def plan(self, arguments):
        self._assert_boundary()
        if type(arguments) is not dict or not 0 < len(arguments) <= 48:
            raise ValueError("flat argument object required")
        request = deepcopy(arguments)
        command = request.get("command")
        if command not in READ_COMMANDS | ORDER_COMMANDS | {"cancel-pa-order"}:
            raise ValueError("unsupported native command")
        for key, value in request.items():
            if type(key) is not str or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
                raise ValueError("exact underscore argument spelling required")
            # Refuse argparse's prefix abbreviations of any protected field.
            if key in HOST_FIELDS or any(protected.startswith(key) for protected in HOST_FIELDS):
                raise ValueError("host-owned field or abbreviation refused: " + key)
            if type(value) not in {str, int, float} or len(str(value)) > 65536:
                raise ValueError("bounded scalar native arguments required")
            canonical(value)
        if command in MUTATIONS and not self.config.allow_mutations:
            raise ValueError("paper mutations disabled by explicit host configuration")
        if (command in MUTATIONS and self.config.allowed_contract_ids is not None
                and request.get("conid") not in self.config.allowed_contract_ids):
            raise ValueError("contract outside explicitly authorized set")
        if command in ORDER_COMMANDS:
            quantity = request.get("quantity")
            if type(quantity) is not int or not 0 < quantity <= self.config.max_quantity:
                raise ValueError("model quantity outside authorized limit")
            if request.get("side") not in {"BUY", "SELL"}:
                raise ValueError("exact model BUY or SELL required")
            if self.config.allowed_contract_ids is not None and request.get("conid") not in self.config.allowed_contract_ids:
                raise ValueError("contract outside explicitly authorized set")
            # Native preflight remains responsible for whole-account holdings,
            # BBO, PnL and order-specific validity. Never invent missing prices.
            if not any(key in request for key in ("reference_price", "limit")):
                raise ValueError("model must supply limit or reference_price for the notional limit")
            # Check every absolute model price. Relative trail distances,
            # percentages and offsets are not prices; native type validation
            # remains authoritative. Slippage can exceed this modeled bound.
            prices = {key: self._money(request[key]) for key in
                      ("limit", "reference_price", "stop", "trail_stop_price") if key in request}
            maximum_price = max(prices.values())
            if maximum_price * quantity > Decimal(self.config.max_order_notional_usd):
                raise ValueError("model notional exceeds authorized limit")
        timeout = request.get("timeout", self.config.default_timeout_seconds)
        if type(timeout) not in {int, float} or not 0 < timeout <= 60:
            raise ValueError("native timeout outside (0,60]")
        argv = [str(Path(self.config.interpreter).resolve()), "-B", "-X", "utf8",
                str(Path(self.config.cli).resolve()), "--timeout", str(timeout), command]
        # Ordering is transport-only. Values and names retain exact spellings.
        for key, value in request.items():
            if key not in {"command", "timeout"}:
                argv += ["--" + key.replace("_", "-"), str(value)]
        root_fields = {}
        if command in ORDER_COMMANDS:
            root_fields["risk_policy"] = str(Path(self.config.risk_policy).resolve())
            root_fields.update(deepcopy(self.config.root_order_fields))
        if command in MUTATIONS:
            root_fields["confirm"] = "CANCEL_PAPER_ORDER" if command == "cancel-pa-order" else "SUBMIT_PAPER_ORDER"
        for key, value in root_fields.items():
            if type(value) is bool:
                if value:
                    argv += ["--" + key.replace("_", "-")]
            else:
                argv += ["--" + key.replace("_", "-"), str(value)]
        return {"request": request, "argv": argv, "root_fields": root_fields,
                "boundary": deepcopy(BOUNDARY), "mutating": command in MUTATIONS,
                "model_argument_digest": hashlib.sha256(canonical(request).encode()).hexdigest()}

    def execute(self, arguments):
        with self._client_lock:
            self.ordinal += 1
            stem = f"native-{self.ordinal:04d}"
            try:
                planned = self.plan(arguments)
            except (ValueError, OSError, TypeError) as error:
                result = {"ok": False, "status": "HOST_REFUSAL", "message": str(error),
                          "arguments": deepcopy(arguments), "cli_invoked": False}
                ref = write_new(self.directory / (stem + "-refusal.json"), canonical(result).encode())
                return {**result, "receipt_ref": ref}
            intent_ref = write_new(self.directory / (stem + "-intent.json"), canonical({
                **planned, "started_at_utc": utc_now(), "shell": False}).encode())
            if planned["mutating"]:
                if (self.directory / "mutation-halt.json").exists():
                    return {"ok": False, "status": "MUTATIONS_HALTED_RECONCILIATION_REQUIRED",
                            "arguments": deepcopy(arguments), "cli_invoked": False, "intent_ref": intent_ref,
                            "message": "retained uncertain attempt must be reconciled; reads remain available"}
                identity = {"kind": "submission", "order_ref": arguments.get("order_ref")}
                if arguments["command"] == "cancel-pa-order":
                    identity = {"kind": "cancel", **{key: arguments.get(key) for key in ("order_id", "order_ref", "conid")}}
                if not identity.get("order_ref"):
                    return {"ok": False, "status": "HOST_REFUSAL", "message": "model order_ref required for durable once-only dispatch",
                            "arguments": deepcopy(arguments), "cli_invoked": False, "intent_ref": intent_ref}
                once_name = "once-" + hashlib.sha256(canonical(identity).encode()).hexdigest() + ".json"
                try:
                    write_new(self.directory / once_name, canonical({"identity": identity, "intent_ref": intent_ref}).encode())
                except FileExistsError:
                    return {"ok": False, "status": "DUPLICATE_OR_UNKNOWN_MUTATION_REFUSED",
                            "message": "reconcile the retained attempt; no implicit resend",
                            "arguments": deepcopy(arguments), "cli_invoked": False, "intent_ref": intent_ref}
            try:
                response = self.transport(planned["argv"], cwd=str(Path(self.config.workspace).resolve()),
                    env=child_environment(planned["mutating"]), shell=False, capture_output=True,
                    timeout=self.config.child_deadline_seconds)
                stdout, stderr = response.stdout, response.stderr
                returncode, uncertain, error = response.returncode, False, None
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = exc.stdout or b"", exc.stderr or b""
                returncode, uncertain, error = None, True, "TimeoutExpired"
            except Exception as exc:
                stdout, stderr = b"", b""
                returncode, uncertain, error = None, True, type(exc).__name__
            if uncertain and planned["mutating"]:
                self._halt_mutations("UNCERTAIN_NATIVE_MUTATION", intent_ref)
            raw = {"kind": "canonical_pa_tws_raw_receipt_v1", **planned, "intent_ref": intent_ref,
                   "finished_at_utc": utc_now(), "stdout_base64": base64.b64encode(stdout).decode(),
                   "stderr_base64": base64.b64encode(stderr).decode(), "returncode": returncode,
                   "uncertain": uncertain, "process_error": error, "retry_authorized": False}
            try:
                raw_ref = write_new(self.directory / (stem + "-raw.json"), canonical(raw).encode())
            except Exception:
                if planned["mutating"]:
                    self._halt_mutations("NATIVE_RAW_PERSISTENCE_FAILED", intent_ref)
                raise
            try:
                parsed, parse_error = strict_json(stdout), None
            except (ValueError, UnicodeDecodeError) as exc:
                parsed, parse_error = None, str(exc)
            native_uncertain = native_uncertain_result(parsed)
            if planned["mutating"] and native_uncertain:
                self._halt_mutations("NATIVE_RECONCILIATION_UNCERTAIN", intent_ref)
            if planned["mutating"] and type(parsed) is not dict:
                self._halt_mutations("UNPARSEABLE_NATIVE_MUTATION_RESULT", intent_ref)
            return {"ok": returncode == 0 and isinstance(parsed, dict) and parsed.get("ok") is True
                           and not uncertain and not native_uncertain,
                    "status": "OUTCOME_UNKNOWN" if uncertain or native_uncertain else "NATIVE_CLI_RESULT",
                    "arguments": deepcopy(arguments), "root_fields": planned["root_fields"],
                    "argv": planned["argv"], "returncode": returncode, "result": parsed,
                    "parse_error": parse_error, "raw_receipt_ref": raw_ref, "cli_invoked": True,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "retry_authorized": False}
