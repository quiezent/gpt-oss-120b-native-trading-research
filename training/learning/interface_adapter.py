"""Unfunded retrospective framework ABI views selected before model sampling.

The two conditions differ only in their declared recipient protocol. Existing
framework simulation owns all policy checks, timing, orders, fees and NAV.
No import or host construction opens a client, creates a directory, samples,
executes an economic action, repairs an output, or selects an investment.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from training.native_agent import framework_candidate
from training.native_agent.harmony_model import HarmonySamplerModel
from training.native_agent.runtime import PureAgentRuntime, validate_action as validate_generic_action
from training.framework import (
    declared_tools as declared_tools, model_mandate)
from training.framework.simulator import (
    FrameworkSimulationHost, TIME_SOURCE)

NAMED = "framework_named_tools"
WRAPPER = "framework_pa_tws_wrapper"
CONDITIONS = (NAMED, WRAPPER)

# Descriptions alone qualify the genuine production manifest. Exact operation
# names and every command/intent/mandate argument schema originate in e1ab.
SIMULATION_QUALIFICATION = (
    "RETROSPECTIVE SIMULATION ONLY; observations are historical ledger evidence, never TWS observations. "
    "Clock source is SIMULATED_HISTORICAL_UTC. No provider, broker connection, live/delayed BBO or wall-clock sleep exists. "
    "Completed daily TRADES bars become available at the next New York calendar midnight under the existing simulation assumption; "
    "original publication vintages are unverified. Account, fills, fees and marked NAV are modeled, with incomplete evidence still unknown. ")
OPERATION_SEMANTICS = {
    "framework_configure_mandate": (
        "Configure exactly the nine complete model-selected policy fields. Allowed contracts must exactly match dataset conid/symbol/primary_exchange metadata; "
        "dataset identity is not fresh broker qualification. Positive order policy is at most USD2500; evidence ages are integers1..300seconds. "
        "Other monetary policy limits must be finite and positive. Both original execution-profile labels retain the same disclosed historical ledger semantics; "
        "no profile flip or BBO invention occurs. Account receipt age is zero on fresh local observation; quote age runs from the completed-bar availability gate. "
        "Daily loss means simulated marked NAV change since the preceding observed session. Policy revisions preserve each earlier intent's exact policy snapshot."),
    "framework_status": "Read simulator readiness, fixed paper boundary and current simulated UTC. connected=false truthfully reports that no broker is connected.",
    "framework_snapshot": (
        "Read modeled account/positions/orders/executions and requested exact dataset contracts, completed daily history and close marks through the current session only. "
        "No future or intraday bar, actual BBO, complete broker account history or actual PnL is supplied."),
    "framework_preview": (
        "Validate your unchanged complete eleven-field intent and configured model mandate against the same simulation admission limits. "
        "Preview creates no order, once-only submission identity, fill or fee. Model-selected limits must be positive exact cents; quantity is at most3 "
        "and modeled absolute limit notional at most USD2500. No argument is rounded, clipped or replaced."),
    "framework_submit": (
        "Queue your unchanged complete eleven-field whole-share USD stock LMT DAY outside_rth=false intent in the existing PortfolioSimulationHost. "
        "At the next session open, the exact model limit must admit the adverse-cost-adjusted open, and the unchanged ledger rechecks its disclosed limits. "
        "Otherwise the DAY order expires unfilled. No intraday range-touch or partial fills. At most3 shares and USD2500 modeled absolute limit notional; "
        "positive prices must be exact cents. Order reference is the canonical PaperTWSBroker hash of your stable intent_id. Submission is once-only; "
        "reconciliation never retransmits and different terms cannot replace an earlier identity. No forced entry, cancellation or liquidation."),
    "framework_reconcile": (
        "Read recorded simulated attempt, queued order, fill, cancellation or expiry evidence for your unchanged complete eleven-field intent. "
        "Never retransmit. A missing identity or incomplete terminal/fee evidence stays unknown; working order acknowledgement is not a fill."),
    "framework_cancel": (
        "Cancel one queued unfilled simulation order only with its exact modeled PA-owned order_id, order_ref and conid. "
        "The canonical reference and original intent remain immutable. Cannot cancel a past fill; a refusal is returned unchanged."),
    "clock": "Read current SIMULATED_HISTORICAL_UTC only. This neither reads production system/broker UTC nor advances the ledger.",
    "wait_until": (
        "Supply your exact complete UTC ISO8601 target ending Z or +00:00. The local calculation immediately advances historical time and the unchanged ledger "
        "only through completed-session availability gates no later than your target. No actual waiting, sampling or broker polling occurs. Intra-session targets "
        "reveal no new price/fill. Past/present targets return current simulated UTC. Targets beyond the historical terminal gate are refused unchanged, never clipped. "
        "The same model conversation retains all generated history. FINAL terminates and schedules nothing."),
}


def named_tool_definitions():
    """Genuine nine named schemas, with explicitly retrospective descriptions."""
    tools = declared_tools.tool_definitions(framework_candidate, model_mandate)
    for item in tools:
        function = item["function"]
        name = function["name"]
        function["description"] = (
            "Use this declared function with command exactly " + name
            + "; recipient and body.command must match. Choose all complete arguments yourself; no alias or repair. "
            + SIMULATION_QUALIFICATION + OPERATION_SEMANTICS[name])
        if name == "wait_until":
            function["parameters"]["properties"]["until_utc"]["description"] = (
                "Exact model-selected complete UTC target for SIMULATED_HISTORICAL_UTC; never production wall-clock sleeping.")
    declared_tools.validate_manifest(tools)
    return tools


class _SimulationFacade:
    """Supply a pure directory attribute and protocol label around old host."""
    def __init__(self, simulation, condition, directory=None):
        self.simulation, self.condition = simulation, condition
        # DeclaredOperationHost requires this attribute but never creates it.
        # Runtime journals, if requested later, remain the caller's ownership.
        self.directory = None if directory is None else Path(directory).resolve()

    @property
    def ledger(self):
        return self.simulation.ledger

    def declaration(self):
        value = deepcopy(self.simulation.declaration())
        value["simulation_ledger_interface"] = value["interface"]
        value["interface"] = self.condition
        value["learning_interface_condition"] = self.condition
        value["recipient_protocol_selected_before_sampling"] = True
        return value

    def execute(self, arguments):
        return self.simulation.execute(deepcopy(arguments))

    def account(self):
        return self.simulation.account()

    def terminal_result(self):
        return self.simulation.terminal_result()


class NamedFrameworkSimulationHost(declared_tools.DeclaredOperationHost):
    """Authentic named-operation declaration over the exact old economics."""
    condition = NAMED

    def __init__(self, histories, descriptor, config=None, *, directory=None):
        simulation = FrameworkSimulationHost(histories, descriptor, config)
        facade = _SimulationFacade(simulation, self.condition, directory)
        super().__init__(facade, named_tool_definitions())

    @property
    def simulation(self):
        return self.host.simulation

    @property
    def ledger(self):
        return self.host.ledger

    def tool_definition(self):
        return deepcopy(self.tools)

    def account(self):
        return self.host.account()

    def terminal_result(self):
        return self.host.terminal_result()


class PaTwsFrameworkSimulationHost(_SimulationFacade):
    """Original single pa_tws wrapper condition over the exact old host."""
    condition = WRAPPER

    def __init__(self, histories, descriptor, config=None, *, directory=None):
        simulation = FrameworkSimulationHost(histories, descriptor, config)
        super().__init__(simulation, WRAPPER, directory)

    def tool_definition(self):
        return self.simulation.tool_definition()


def _condition(value):
    if type(value) is not str or value not in CONDITIONS:
        raise ValueError("exact predeclared framework interface condition required; no output-based protocol routing")
    return value


def create_host(histories, descriptor, config=None, *, directory=None):
    if type(descriptor) is not dict:
        raise ValueError("exact predeclared descriptor required")
    condition = _condition(descriptor.get("interface"))
    host_class = NamedFrameworkSimulationHost if condition == NAMED else PaTwsFrameworkSimulationHost
    return host_class(histories, descriptor, config, directory=directory)


def tool_definition(host):
    if type(host) not in {NamedFrameworkSimulationHost, PaTwsFrameworkSimulationHost}:
        raise ValueError("one exact prospective framework host view required")
    return host.tool_definition()


def model_class(condition):
    return declared_tools.DeclaredHarmonySamplerModel if _condition(condition) == NAMED else HarmonySamplerModel


def runtime_class(condition):
    return declared_tools.DeclaredAgentRuntime if _condition(condition) == NAMED else PureAgentRuntime


def runtime_kwargs(host):
    tools = tool_definition(host)
    return {"tool_definitions": tools} if host.condition == NAMED else {}


def validate_action(host, action):
    """Expose the genuine validator/errors for the owner's evidence classifier.

    Runtime/model classes above perform this check before host dispatch. This
    helper changes no output or ledger and supplies no failure/reward policy.
    """
    tools = tool_definition(host)
    return (declared_tools.validate_declared_action(action, tools) if host.condition == NAMED
            else validate_generic_action(action))


def common_initial_observation(host, histories, config):
    """Exact shared ledger economics and initial completed daily bar facts."""
    ledger = host.ledger
    return {"account": ledger.account(), "history": {symbol: ledger._execute({
        "command": "historical-daily", "symbol": symbol, "count": min(config.history_window, 20)})
        for symbol in sorted(histories)}}
