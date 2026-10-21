"""Disclosed local clock/wait commands with an unchanged native broker transport.

The existing runtime has one pa_tws tool address. clock and wait_until at that
address are local host operations, never native IBKR CLI commands. The model
chooses the command and complete UTC target; the host supplies no trade timing.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import re
import threading
import time

from training.native_agent.runtime import LocalGenerationBudgetExhausted, canonical, write_new

MAXIMUM_SESSION_SECONDS = 86400
LOCAL_COMMANDS = {"clock", "wait_until"}


def temporal_tool_definition(native_definition):
    tool = deepcopy(native_definition)
    function = tool[0]["function"]
    properties = function["parameters"]["properties"]
    properties["command"]["enum"] = sorted(properties["command"]["enum"] + list(LOCAL_COMMANDS))
    properties["until_utc"] = {"type": "string", "description": "Complete model-selected UTC timestamp for the local wait_until command."}
    function["description"] += (
        " Local host operations share this tool address but are not IBKR CLI commands: "
        "clock accepts exactly {command:'clock'} and returns local system UTC time. "
        "wait_until accepts exactly command and until_utc, an ISO8601 UTC timestamp ending Z or +00:00. "
        "It waits until your exact timestamp, then returns observed UTC time and elapsed seconds in this same conversation. "
        "A past or present target returns immediately. No sampling or broker polling occurs during the wait. "
        "Targets after the disclosed session deadline are refused unchanged; no target is clipped or supplied for you. "
        "Local commands do not accept timeout or any broker arguments. "
        "Choose your own timing and actions. FINAL ends the episode and schedules no future activity.")
    return tool


class ClockWaitHost:
    def __init__(self, native_host, *, max_wait_seconds=MAXIMUM_SESSION_SECONDS,
                 clock=None, monotonic=None, sleep=None, stop_event=None):
        if type(max_wait_seconds) is not int or not 1 <= max_wait_seconds <= MAXIMUM_SESSION_SECONDS:
            raise ValueError("integer local session bound in 1..86400 seconds required")
        self.native = native_host
        self.directory = Path(native_host.directory)
        self.local_directory = self.directory / "local_time"
        self.max_wait_seconds = max_wait_seconds
        self.clock = (lambda: datetime.now(timezone.utc)) if clock is None else clock
        self.monotonic = time.monotonic if monotonic is None else monotonic
        self.sleep = time.sleep if sleep is None else sleep
        self.stop_event = threading.Event() if stop_event is None else stop_event
        self.started_at = self._now()
        self.monotonic_started = self.monotonic()
        self.deadline = self.started_at + timedelta(seconds=max_wait_seconds)
        self.monotonic_deadline = self.monotonic_started + max_wait_seconds
        self.ordinal = 0
        self.local_lock = threading.Lock()
        self.local_receipts = []

    def _now(self):
        current = self.clock()
        if not isinstance(current, datetime) or current.tzinfo is None or current.utcoffset() != timedelta(0):
            raise ValueError("aware UTC system clock required")
        return current.astimezone(timezone.utc)

    def assert_within_session(self):
        if self.monotonic() >= self.monotonic_deadline or self._now() >= self.deadline:
            raise LocalGenerationBudgetExhausted("disclosed local session wall limit reached before sampling; no substituted action",
                details={"budget_kind": "SESSION_WALL_CLOCK", "maximum_session_seconds": self.max_wait_seconds,
                         "session_deadline_utc": self.deadline.isoformat(), "provider_dispatched": False})

    def declaration(self):
        declared = deepcopy(self.native.declaration())
        declared["local_temporal_operations"] = {
            "tool_address": "pa_tws", "native_ibkr_commands": False,
            "clock_source": "local_system_UTC", "current_utc": self._now().isoformat(),
            "session_deadline_utc": self.deadline.isoformat(), "maximum_session_seconds": self.max_wait_seconds,
            "clock_arguments": {"command": "clock"},
            "wait_until_arguments": {"command": "wait_until", "until_utc": "model-selected UTC ISO8601 timestamp"},
            "past_or_present_target": "Returns actual current UTC immediately; no broker or provider action.",
            "target_after_session_deadline": "Visible refusal retaining your exact target; never clipped.",
            "waiting": "Same in-memory model conversation; no broker polling or model sampling while waiting.",
            "sdk_session_heartbeat": "The existing SDK may maintain its owned session in its background thread; this is not a model or broker decision.",
            "final_is_terminal": True, "automatic_resume": False, "scheduler_change": False}
        return declared

    @staticmethod
    def _target(value):
        if (type(value) is not str or len(value) > 40
                or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|\+00:00)", value) is None):
            raise ValueError("until_utc must be a complete UTC ISO8601 timestamp ending Z or +00:00")
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    def execute(self, arguments):
        if type(arguments) is not dict or arguments.get("command") not in LOCAL_COMMANDS:
            try:
                self.assert_within_session()
            except LocalGenerationBudgetExhausted:
                with self.local_lock:
                    stem, request, intent, started, monotonic_started = self._begin(arguments)
                    return self._finish(stem, {"ok": False, "status": "LOCAL_SESSION_LIMIT_REFUSAL",
                        "message": "disclosed session wall limit expired before native dispatch"},
                        request, intent, started, monotonic_started)
            return self.native.execute(arguments)  # Exact original broker arguments and unchanged native host.
        with self.local_lock:
            stem, request, intent, started, monotonic_started = self._begin(arguments)
            target = None
            try:
                expected_keys = {"command"} if request["command"] == "clock" else {"command", "until_utc"}
                if set(request) != expected_keys:
                    raise ValueError("exact declared local temporal fields required; broker fields and timeout are refused")
                if request["command"] == "wait_until":
                    target = self._target(request["until_utc"])
                    if target > self.deadline or (target - started).total_seconds() > MAXIMUM_SESSION_SECONDS:
                        raise ValueError("model-selected UTC target exceeds the disclosed session deadline")
            except (ValueError, TypeError) as error:
                result = {"ok": False, "status": "LOCAL_TIME_REFUSAL", "message": str(error)}
            else:
                try:
                    while target is not None:
                        current = self._now()
                        if current >= target:
                            break
                        if self.stop_event.is_set():
                            raise KeyboardInterrupt("local session stop requested")
                        remaining_wall = self.monotonic_deadline - self.monotonic()
                        if remaining_wall <= 0:
                            break
                        self.sleep(min(1.0, (target - current).total_seconds(), remaining_wall))
                    reached = target is None or self._now() >= target
                    result = {"ok": reached, "status": "LOCAL_CLOCK" if target is None else (
                        "WAIT_TARGET_REACHED" if reached else "LOCAL_WAIT_WALL_LIMIT"),
                        "target_reached": reached if target is not None else None}
                except BaseException as error:
                    self._finish(stem, {"ok": False, "status": "LOCAL_WAIT_INTERRUPTED" if isinstance(error, KeyboardInterrupt)
                        else "LOCAL_WAIT_FAILED", "error_type": type(error).__name__}, request, intent, started, monotonic_started)
                    raise  # No continuation, retry, trade, or FINAL resume is manufactured after interruption.
            return self._finish(stem, result, request, intent, started, monotonic_started)

    def _begin(self, arguments):
        self.ordinal += 1
        stem = self.local_directory / ("time-" + format(self.ordinal, "04d"))
        started, monotonic_started = self._now(), self.monotonic()
        request = deepcopy(arguments)
        intent = write_new(Path(str(stem) + "-intent.json"), canonical({
            "kind": "model_selected_local_time_intent_v1", "arguments": request,
            "model_argument_sha256": hashlib.sha256(canonical(request).encode()).hexdigest(),
            "started_at_utc": started.isoformat(), "session_deadline_utc": self.deadline.isoformat(),
            "cli_invoked": False, "broker_calls": 0, "sampling_calls": 0}).encode())
        return stem, request, intent, started, monotonic_started

    def _finish(self, stem, result, request, intent, started, monotonic_started):
        current = self._now()
        raw = {"kind": "observed_local_time_receipt_v1", **result, "arguments": deepcopy(request),
            "started_at_utc": started.isoformat(), "current_utc": current.isoformat(),
            "elapsed_monotonic_seconds": self.monotonic() - monotonic_started,
            "clock_source": "local_system_UTC", "intent_ref": intent,
            "cli_invoked": False, "broker_calls": 0, "sampling_calls": 0,
            "automatic_resume": False, "retry_authorized": False}
        raw_ref = write_new(Path(str(stem) + "-raw.json"), canonical(raw).encode())
        retained = {**raw, "raw_receipt_ref": raw_ref}
        self.local_receipts.append(deepcopy(retained))
        return retained
