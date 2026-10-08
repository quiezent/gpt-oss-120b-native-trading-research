"""Vendor callback/envelope compatibility, not a change to trading policy.

IB API 10.33+ supplies errorTime; 10.51 supports protobuf send intents.
Never spoof protocol versions or turn material errors into success.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCallback:
    req_id: int
    code: int
    message: str
    advanced_reject: str
    broker_time: int | None

    def arguments(self):
        if self.broker_time is None:
            return self.req_id, self.code, self.message, self.advanced_reject
        return self.req_id, self.broker_time, self.code, self.message, self.advanced_reject


def error_callback(req_id, *args, **kwargs) -> ErrorCallback:
    """Recognize old/new signatures by their types; never shift an error code."""
    if kwargs:
        if args or set(kwargs) - {"errorTime", "errorCode", "errorString", "advancedOrderRejectJson"}:
            raise TypeError("unsupported error callback arguments")
        code = kwargs["errorCode"]
        message = kwargs["errorString"]
        advanced = kwargs.get("advancedOrderRejectJson", "")
        broker_time = kwargs.get("errorTime")
    elif len(args) in (2, 3) and type(args[1]) is str:
        code, message = args[:2]
        advanced = args[2] if len(args) == 3 else ""
        broker_time = None
    elif len(args) in (3, 4) and type(args[1]) is int and type(args[2]) is str:
        broker_time, code, message = args[:3]
        advanced = args[3] if len(args) == 4 else ""
    else:
        raise TypeError("unrecognized error callback signature")
    if (type(req_id) is not int or type(code) is not int or type(message) is not str
            or type(advanced) is not str or (broker_time is not None and type(broker_time) is not int)):
        raise TypeError("invalid error callback field types")
    return ErrorCallback(req_id, code, message, advanced, broker_time)


def classic_message(*args) -> str:
    """Journal the original SDK id/payload without changing its transport call."""
    if len(args) == 1 and type(args[0]) is str:
        return args[0]
    if len(args) == 2 and type(args[0]) is int and type(args[1]) is str:
        return f"{args[0]}\0{args[1]}"
    raise ValueError("unrecognized classic SDK send signature")


# Only the existing observer read opcodes plus completed orders. No order,
# configuration update, global cancel, auto-binding or unreviewed opcode.
PROTO_READ_TYPES = {
    1: ("MarketDataRequest", "MarketDataRequest"),
    2: ("CancelMarketData", "CancelMarketData"),
    5: ("OpenOrdersRequest", "OpenOrdersRequest"),
    6: ("AccountDataRequest", "AccountDataRequest"),
    7: ("ExecutionRequest", "ExecutionRequest"),
    9: ("ContractDataRequest", "ContractDataRequest"),
    16: ("AllOpenOrdersRequest", "AllOpenOrdersRequest"),
    17: ("HistoricalDataRequest", "HistoricalDataRequest"),
    49: ("CurrentTimeRequest", "CurrentTimeRequest"),
    59: ("MarketDataTypeRequest", "MarketDataTypeRequest"),
    61: ("PositionsRequest", "PositionsRequest"),
    62: ("AccountSummaryRequest", "AccountSummaryRequest"),
    63: ("CancelAccountSummary", "CancelAccountSummary"),
    64: ("CancelPositions", "CancelPositions"),
    71: ("StartApiRequest", "StartApiRequest"),
    92: ("PnLRequest", "PnLRequest"),
    93: ("CancelPnL", "CancelPnL"),
    99: ("CompletedOrdersRequest", "CompletedOrdersRequest"),
}


def protobuf_read_intent(message_id, payload):
    """Decode only allowlisted native reads; return scoped cleanup metadata.

Scope strings below are validation representations, not invented wire bytes.
The caller retains/hashes the actual serialized SDK payload separately.
"""
    if type(message_id) is not int or type(payload) is not bytes or len(payload) > 1024 * 1024:
        raise ValueError("invalid protobuf send envelope")
    code = message_id - 200  # ibapi.common.PROTOBUF_MSG_ID in official 10.51
    if code not in PROTO_READ_TYPES:
        raise ValueError(f"non-read protobuf SDK message refused: {message_id}")
    module_name, type_name = PROTO_READ_TYPES[code]
    module = importlib.import_module(f"ibapi.protobuf.{module_name}_pb2")
    message = getattr(module, type_name)()
    message.ParseFromString(payload)
    known = type(message)()
    known.CopyFrom(message)
    known.DiscardUnknownFields()
    if known != message:
        raise ValueError("unreviewed protobuf fields refused")
    fields = None
    if code == 6:
        fields = ["6", "2", "1" if message.subscribe else "0", message.acctCode, ""]
    elif code in {2, 63, 93}:
        version = {2: "2", 63: "1", 93: None}[code]
        fields = [str(code)] + ([version] if version else []) + [str(message.reqId), ""]
    elif code == 64:
        fields = ["64", "1", ""]
    elif code == 99:
        fields = ["99", "1" if message.apiOnly else "0", ""]
    return code, fields, message
