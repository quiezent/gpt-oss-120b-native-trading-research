"""Mechanical IBAPI 1051.1 fee evidence capture; no requests or order policy.

The per-instance decoder hooks retain the input before the vendor projects
missing numeric fields to zero. The vendor parser still runs exactly once.
"""
from __future__ import annotations

import base64
import copy
import math
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from ibapi.message import IN
from ibapi.protobuf.CommissionAndFeesReport_pb2 import CommissionAndFeesReport

FIELDS = ("execId", "commissionAndFees", "currency", "realizedPNL",
          "bondYield", "yieldRedemptionDate")
UNSET_DOUBLE = Decimal(str(sys.float_info.max))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _number(value, present):
    if not present or isinstance(value, bool):
        return None
    try:
        decimal = Decimal(str(value))
        if not decimal.is_finite() or abs(decimal) == UNSET_DOUBLE:
            return None
        number = float(decimal)
        return number if math.isfinite(number) else None
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None


class FeeCapture:
    def __init__(self, *, account=None, clock=time.monotonic):
        self.account = account
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._reports = []

    def _record(self, protocol, raw, fields, *, error_type=None):
        with self._condition:
            self._reports.append({
                "sequence": len(self._reports) + 1,
                "received_at_utc": _utc(),
                "protocol": protocol,
                "message_id": IN.COMMISSION_AND_FEES_REPORT,
                "raw": raw,
                "fields": fields,
                "capture_error_type": error_type,
            })
            self._condition.notify_all()

    def record_classic(self, fields):
        original = tuple(fields)
        raw = {"field_bytes_base64": [base64.b64encode(item).decode("ascii")
                                      for item in original]}
        decoded = {}
        error_type = None
        try:
            # One version token precedes the six report fields in IBAPI 1051.1.
            if len(original) < 7:
                raise ValueError("short commission report")
            for name, token in zip(FIELDS, original[1:7]):
                decoded[name] = {"present": bool(token),
                                 "value": token.decode("UTF-8") if token else None}
        except Exception as exc:
            error_type = type(exc).__name__
        self._record("classic", raw, decoded, error_type=error_type)

    def record_protobuf(self, payload):
        raw = {"payload_base64": base64.b64encode(payload).decode("ascii")}
        decoded = {}
        error_type = None
        try:
            message = CommissionAndFeesReport()
            message.ParseFromString(payload)
            for name in FIELDS:
                present = message.HasField(name)
                value = getattr(message, name) if present else None
                # Raw payload is authoritative; strings also preserve NaN/Inf
                # diagnostics while keeping the public envelope strict JSON.
                if isinstance(value, float):
                    value = repr(value)
                decoded[name] = {"present": present, "value": value}
        except Exception as exc:
            error_type = type(exc).__name__
        self._record("protobuf", raw, decoded, error_type=error_type)

    def snapshot(self):
        with self._condition:
            return copy.deepcopy(self._reports)

    def _evidence_locked(self, exec_id):
        matched = [report for report in self._reports
                   if report["fields"].get("execId", {}).get("present")
                   and report["fields"]["execId"].get("value") == exec_id]
        evidence = {"status": "UNKNOWN_MISSING_REPORT", "commission_and_fees": None,
                    "currency": None, "realized_pnl": None,
                    "realized_pnl_status": "UNKNOWN",
                    "report_sequences": [report["sequence"] for report in matched],
                    "reports": copy.deepcopy(matched)}
        if not exec_id:
            evidence["status"] = "UNKNOWN_EXECUTION_ID"
            return evidence
        if not matched:
            return evidence
        values = []
        realized = []
        invalid = False
        for report in matched:
            fields = report["fields"]
            fee = fields.get("commissionAndFees", {})
            currency = fields.get("currency", {})
            amount = _number(fee.get("value"), fee.get("present", False))
            currency_value = currency.get("value")
            if (report["capture_error_type"] or amount is None
                    or not currency.get("present") or not isinstance(currency_value, str)
                    or not currency_value.strip()):
                invalid = True
            else:
                values.append((amount, currency_value))
            pnl = fields.get("realizedPNL", {})
            realized.append(_number(pnl.get("value"), pnl.get("present", False)))
        if invalid:
            evidence["status"] = "UNKNOWN_INCOMPLETE_REPORT"
        elif len(set(values)) != 1:
            evidence["status"] = "UNKNOWN_CONFLICTING_REPORTS"
        else:
            evidence["status"] = "OBSERVED"
            evidence["commission_and_fees"], evidence["currency"] = values[0]
        if realized and None not in realized and len(set(realized)) == 1:
            evidence["realized_pnl_status"] = "OBSERVED"
            evidence["realized_pnl"] = realized[0]
        return evidence

    def enrich(self, executions):
        with self._condition:
            result = copy.deepcopy(executions)
            for row in result:
                row["fee_evidence"] = self._evidence_locked(row.get("exec_id"))
                if self.account is not None and row.get("account") != self.account:
                    row["fee_evidence"] = {
                        "status": "UNKNOWN_ACCOUNT_MISMATCH", "commission_and_fees": None,
                        "currency": None, "realized_pnl": None,
                        "realized_pnl_status": "UNKNOWN", "report_sequences": [], "reports": [],
                    }
            return result

    def report(self, executions):
        with self._condition:
            rows = self.enrich(executions)
            for row in rows:
                row["fee_evidence"].pop("reports")
            return {"executions": rows, "commission_reports": self.snapshot(),
                    "snapshot_at_utc": _utc()}

    def wait_for_ids(self, exec_ids, deadline):
        ids = set(exec_ids)
        with self._condition:
            while any(self._evidence_locked(exec_id)["status"] != "OBSERVED"
                      for exec_id in ids):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


def attach_decoder(decoder, capture):
    """Wrap two entrypoints on one decoder; never mutate vendor class tables."""
    if decoder is None:
        raise ValueError("fee capture requires the connected owned decoder")
    attached = getattr(decoder, "_pa_fee_capture", None)
    if attached is capture:
        return
    if attached is not None:
        raise ValueError("decoder already has another fee capture owner")
    original_classic = decoder.interpret
    original_protobuf = decoder.processProtoBuf

    def classic(fields, msg_id):
        if msg_id == IN.COMMISSION_AND_FEES_REPORT:
            original = tuple(fields)
            capture.record_classic(original)
            return original_classic(original, msg_id)
        return original_classic(fields, msg_id)

    def protobuf(payload, msg_id):
        if msg_id == IN.COMMISSION_AND_FEES_REPORT:
            capture.record_protobuf(payload)
        return original_protobuf(payload, msg_id)

    decoder.interpret = classic
    decoder.processProtoBuf = protobuf
    decoder._pa_fee_capture = capture
