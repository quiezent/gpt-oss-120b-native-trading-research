"""One canonical owned-order cancellation transport for the separate V8 trial.

This additive surface leaves PaperTWSBroker and PA saved instructions unchanged.
It grants no ownership: the trial adapter must durably reserve and independently
verify the exact current trial order before calling this narrow transport.
"""

from __future__ import annotations

import json

import os

import re

from copy import deepcopy

from datetime import datetime

from trader_runtime.tws_broker import PaperTWSBroker, TWSBrokerError, PA_PYTHON, PA_SCRIPT, _COMMAND_LOCK

def _execution_profile(broker, profile):
    if type(broker) is not PaperTWSBroker or type(profile) is not str or profile not in {"LIVE_TYPE1", "PAPER_DELAYED_TYPE3"}:
        raise TWSBrokerError("exact canonical broker and explicit supported execution profile required")

def preview_exact_paper_profile(broker, intent, *, quote_execution_profile="LIVE_TYPE1", mandate=None):
    """Preserve the original live route; explicitly commission delayed preview."""
    _execution_profile(broker, quote_execution_profile)
    if quote_execution_profile == "LIVE_TYPE1":
        return broker.preview(intent)
    if type(mandate) is not dict or mandate.get("execution_profile") != quote_execution_profile:
        raise TWSBrokerError("canonical delayed preview requires the same explicit portfolio profile")
    try:
        profile = json.dumps(mandate, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise TWSBrokerError("finite canonical preview portfolio required") from None
    if len(profile.encode("utf-8")) > 65536:
        raise TWSBrokerError("bounded canonical preview portfolio required")
    native = broker._invoke("preview-stock-limit", broker._intent_arguments(intent) + [
        "--paper-trial-profile", quote_execution_profile, "--portfolio-risk-policy-json", profile])
    return {"allowed": True, "intent": deepcopy(intent), "source": "native_tws",
            "broker_connection": False, "native_preview": native,
            "quote_execution_profile": quote_execution_profile,
            "connected_preflight": "REQUIRED_NOT_PERFORMED", "native_pnl_guard_checked": False,
            "execution_eligible": False}

def submit_exact_paper_profile(broker, intent, *, confirm, mandate, quote_execution_profile="LIVE_TYPE1"):
    """Use canonical submit's existing dedup, env, receipt and uncertainty logic.

    The instance-local argv adapter is serialized by the existing command lock.
    It affects this explicit invocation only and is restored even on refusal.
    No risk observation is serialized as canonical authority: pa_tws performs
    its own real connected book, quote and native session-PnL preflight.
    """
    _execution_profile(broker, quote_execution_profile)
    if quote_execution_profile == "LIVE_TYPE1":
        if type(mandate) is dict and mandate.get("execution_profile", "LIVE_TYPE1") != "LIVE_TYPE1":
            raise TWSBrokerError("live route cannot consume a delayed portfolio profile")
        return broker.submit(intent, confirm=confirm, mandate=mandate)
    if type(mandate) is not dict or mandate.get("execution_profile") != quote_execution_profile:
        raise TWSBrokerError("canonical delayed argv and portfolio profile must agree")
    with _COMMAND_LOCK:
        original = broker._invoke
        def invoke(command, arguments=None, *, mutation=False):
            if command != "submit-stock-limit" or mutation is not True:
                raise TWSBrokerError("explicit delayed submit wrapper permits only the canonical submission")
            values = list(arguments or [])
            if "--paper-trial-profile" in values:
                raise TWSBrokerError("duplicate canonical execution profile is refused")
            return original(command, values + ["--paper-trial-profile", quote_execution_profile], mutation=True)
        broker._invoke = invoke
        try:
            return broker.submit(intent, confirm=confirm, mandate=mandate)
        finally:
            broker._invoke = original
