"""Model-selected native portfolio mandate; no teacher contracts or order repair.

Additive staging outside the running native episode. Configuration is a local
tool operation and changes no broker state. Native connected validation still
resolves the submitted contract and checks the exact selected portfolio policy.
"""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from training.native_agent.runtime import canonical, strict_json, utc_now, write_new

MANDATE_FIELDS = {"mandate_id", "allowed_contracts", "max_order_notional_usd",
    "max_symbol_exposure_usd", "max_gross_exposure_usd", "max_daily_loss_usd",
    "max_account_age_seconds", "max_quote_age_seconds", "execution_profile"}
PROFILES = {"LIVE_TYPE1", "PAPER_DELAYED_TYPE3"}
CONFIGURE = "framework_configure_mandate"


def validate_mandate(value, *, maximum_order_notional_usd="2500"):
    if type(value) is not dict or set(value) != MANDATE_FIELDS:
        raise ValueError("exact complete model-selected native portfolio mandate required")
    canonical(value)
    if (type(value["mandate_id"]) is not str or not value["mandate_id"].strip()
            or len(value["mandate_id"]) > 256 or value["execution_profile"] not in PROFILES):
        raise ValueError("explicit mandate identity and supported execution profile required")
    contracts = value["allowed_contracts"]
    if type(contracts) is not list or not 1 <= len(contracts) <= 32:
        raise ValueError("one through32 exact model-selected contract identities required")
    seen = set()
    for row in contracts:
        if (type(row) is not dict or set(row) != {"conid", "symbol", "primary_exchange"}
                or type(row["conid"]) is not int or row["conid"] <= 0
                or any(type(row[key]) is not str or not row[key] or len(row[key]) > 256
                       for key in ("symbol", "primary_exchange"))):
            raise ValueError("exact positive contract identity required; fresh native qualification remains authoritative")
        key = canonical(row)
        if key in seen:
            raise ValueError("duplicate model-selected contract identity")
        seen.add(key)
    for key in ("max_account_age_seconds", "max_quote_age_seconds"):
        if type(value[key]) is not int or not 1 <= value[key] <= 300:
            raise ValueError("native evidence-age bound in1..300seconds required")
    for key in ("max_order_notional_usd", "max_symbol_exposure_usd", "max_gross_exposure_usd", "max_daily_loss_usd"):
        if type(value[key]) not in {str, int, float}:
            raise ValueError("explicit finite positive native monetary limit required")
        try:
            amount = Decimal(str(value[key]))
        except Exception:
            raise ValueError("explicit finite positive native monetary limit required") from None
        if not amount.is_finite() or amount <= 0:
            raise ValueError("explicit finite positive native monetary limit required")
        if key == "max_order_notional_usd" and amount > Decimal(maximum_order_notional_usd):
            raise ValueError("model-selected order ceiling exceeds disclosed USD2500 host authorization")
    return deepcopy(value)


def tool_definition(component):
    tool = component.framework_tool_definition()
    function = tool[0]["function"]
    properties = function["parameters"]["properties"]
    properties["command"]["enum"].append(CONFIGURE)
    properties["command"]["enum"].sort()
    properties["mandate"] = {"type": "object", "additionalProperties": False,
        "required": sorted(MANDATE_FIELDS), "properties": {
            "mandate_id": {"type": "string"},
            "allowed_contracts": deepcopy(properties["contracts"]),
            "execution_profile": {"type": "string", "enum": sorted(PROFILES)},
            **{key: {"type": ["number", "string"]} for key in MANDATE_FIELDS if key.endswith("_usd")},
            **{key: {"type": "integer", "minimum": 1, "maximum": 300}
               for key in ("max_account_age_seconds", "max_quote_age_seconds")}}}
    function["description"] += (
        " framework_configure_mandate is a LOCAL model operation accepting exactly command and mandate. "
        "Supply all nine native policy fields yourself, including your exact allowed_contracts and execution_profile. "
        "No shortlist, numeric loss/exposure policy, price or trade direction is supplied by the host. "
        "The per-order policy ceiling must be positive and at most USD2500; account/quote ages must be1..300seconds. "
        "Other monetary limits must be finite and positive. Configuration may be revised by a later exact call; "
        "old configurations and attempts remain durable. Submission requires a successfully configured mandate. "
        "LIVE_TYPE1 and PAPER_DELAYED_TYPE3 retain their canonical quote/portfolio validation requirements. "
        "A mismatch or missing policy is refused visibly without clipping, adding contracts, repairing values or sending an order. "
        "An allowed_contracts declaration is not broker qualification; native fresh contract resolution remains authoritative.")
    return tool


class ModelMandateHost:
    def __init__(self, framework, *, directory):
        self.framework = framework
        self.directory = Path(framework.directory)
        self.configuration_directory = Path(directory)
        self.configuration_directory.mkdir(parents=True, exist_ok=False)
        self.ordinal, self.current, self.current_reference, self.configuration_receipts = 0, None, None, []

    def declaration(self):
        result = self.framework.declaration()
        result["model_portfolio_mandate"] = {
            "configuration_operation": CONFIGURE, "all_fields_selected_by_model": True,
            "initial_policy": None, "current_policy": deepcopy(self.current),
            "current_policy_receipt": deepcopy(self.current_reference),
            "required_fields": sorted(MANDATE_FIELDS), "maximum_order_notional_usd": self.framework.max_notional_usd,
            "evidence_age_seconds": {"minimum": 1, "maximum": 300},
            "positive_finite_model_exposure_and_daily_loss_limits_required": True,
            "host_contract_shortlist": None, "native_connected_contract_qualification_required": True,
            "supported_profiles": sorted(PROFILES), "values_clipped_or_repaired": False,
            "configuration_is_broker_action": False}
        return result

    def policy(self):
        if self.current is None:
            raise ValueError("model-selected portfolio mandate is not configured; no submission or delayed preview sent")
        return deepcopy(self.current)

    def execute(self, arguments):
        if type(arguments) is not dict or arguments.get("command") != CONFIGURE:
            return self.framework.execute(deepcopy(arguments))
        self.framework.admission_check()
        self.ordinal += 1
        stem = f"configuration-{self.ordinal:04d}"
        intent = write_new(self.configuration_directory / (stem + "-intent.json"), canonical({
            "arguments": deepcopy(arguments), "started_at_utc": utc_now(), "broker_invoked": False}).encode())
        try:
            if set(arguments) != {"command", "mandate"}:
                raise ValueError("exact command and complete mandate required")
            policy = validate_mandate(arguments["mandate"], maximum_order_notional_usd=self.framework.max_notional_usd)
            result = {"ok": True, "status": "MODEL_MANDATE_CONFIGURED", "policy": deepcopy(policy)}
        except (ValueError, TypeError) as error:
            policy = None
            result = {"ok": False, "status": "MODEL_MANDATE_REFUSAL", "message": str(error),
                      "current_policy_unchanged": True}
        result.update(arguments=deepcopy(arguments), intent_ref=intent, broker_invoked=False,
                      finished_at_utc=utc_now(), economic_action_supplied=False)
        ref = write_new(self.configuration_directory / (stem + "-raw.json"), canonical(result).encode())
        if policy is not None:
            self.current, self.current_reference = policy, ref
        result["raw_receipt_ref"] = ref
        self.configuration_receipts.append(deepcopy(result))
        return result


def configured_callbacks(component, *, broker, native_host, recording_runner, policy_provider, prior_native_once_refs):
    """The same exact canonical APIs, with the policy chosen in this conversation.

    Policy is read once for each preview/submit; no argument or policy mutation
    occurs. Previous policies/once identities are retained. No V8 planner runs.
    """
    from trader_runtime.tws_broker import PaperTWSBroker
    from trading_desk.v8_tws_control import preview_exact_paper_profile, submit_exact_paper_profile
    if (type(broker) is not PaperTWSBroker or broker._runner is not recording_runner
            or Path(broker.risk_policy_path).resolve() != Path(native_host.config.risk_policy).resolve()):
        raise ValueError("same canonical broker, recorded transport and native risk source required")
    component.seed_native_prior_receipts(native_host, prior_native_once_refs)

    def callback(request):
        before, attempts = len(recording_runner.receipts), recording_runner.attempts
        try:
            command = request["command"]
            if command == "framework_status":
                result = broker.status()
            elif command == "framework_snapshot":
                result = broker.snapshot(deepcopy(request["contracts"]))
            elif command in {"framework_preview", "framework_submit"}:
                policy = policy_provider()
                if command == "framework_preview":
                    result = preview_exact_paper_profile(broker, deepcopy(request["intent"]),
                        quote_execution_profile=policy["execution_profile"], mandate=deepcopy(policy))
                else:
                    if not native_host.config.allow_mutations:
                        raise ValueError("source-pinned native transport paper writes disabled")
                    result = submit_exact_paper_profile(broker, deepcopy(request["intent"]),
                        confirm="SUBMIT_PAPER_ORDER", mandate=deepcopy(policy),
                        quote_execution_profile=policy["execution_profile"])
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
                    "transport_attempted": recording_runner.attempts > attempts}
        except BaseException as error:
            error.raw_transport_receipts = deepcopy(recording_runner.receipts[before:])
            error.transport_attempted = (request["command"] == "framework_cancel" or recording_runner.attempts > attempts)
            raise
    return {name: callback for name in component.KEYS}
