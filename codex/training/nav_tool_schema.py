"""Public simulator API documentation; no market policy or action selection.

The installed Harmony renderer displays top-level properties and descriptions,
but does not display root ``oneOf``. Keep a visible command enum and field
descriptions as well as the complete machine-readable exact argument contract.
"""
from __future__ import annotations

from copy import deepcopy


ORDER_REF_PATTERN = r"^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$"


COMMAND_FIELDS = {
    "account": ("command",),
    "historical-daily": ("command", "symbol", "count"),
    "submit-stock-order": ("command", "conid", "symbol", "side", "quantity", "order_type", "limit", "tif", "order_ref"),
    "cancel-order": ("command", "order_ref"),
    "advance": ("command", "sessions"),
    "finish": ("command",),
}

COMMAND_SEMANTICS = {
    "account": "Read current cash, marked NAV, positions, queued orders, actual simulated fills and sessions_remaining. Does not advance time.",
    "historical-daily": "Read the requested symbol's last count completed daily bars through the current session, inclusive. count is an unquoted whole JSON integer 1..252. Does not advance time or expose future bars.",
    "submit-stock-order": "Queue exactly the model's complete stock order for the next session open. conid must match symbol in verified_contracts. quantity is an unquoted positive whole JSON integer. Model supplies order_type=LMT and tif=DAY explicitly. BUY requires cash and SELL requires owned, unreserved shares; shorting is unsupported. At the next open, a BUY fills only when adverse-adjusted open <= limit; a SELL fills only when adverse-adjusted open >= limit, within disclosed account limits. Otherwise the DAY order expires. No intraday range-touch or partial fills. Submission itself does not advance time.",
    "cancel-order": "Cancel one queued, unfilled simulation order by its original unique order_ref. Cannot cancel a past fill. Does not advance time.",
    "advance": "Simulation clock command only: advance sessions trading sessions, an unquoted positive whole JSON integer no greater than account.sessions_remaining. Execute already queued next-open orders and expose observations at the new current session. Creates no orders and does not select investments.",
    "finish": "Simulation terminal clock command only: advance all remaining sessions, settle already queued orders, mark terminal NAV and end the episode. Creates no entry/exit orders and does not liquidate remaining inventory.",
}


def argument_schema(declaration: dict) -> dict:
    """Every accepted argument set is declared, without defaults or coercion."""
    symbols = list(declaration["symbols"])
    conids = [row["conid"] for row in declaration["verified_contracts"]]
    fields = {
        "command": {"type": "string", "enum": list(COMMAND_FIELDS),
                    "description": "Choose one supported command and supply exactly its declared fields. No additional fields or invented commands are accepted."},
        "symbol": {"type": "string", "enum": symbols,
                   "description": "Required for historical-daily and submit-stock-order. Exact declared symbol; conid must refer to the same contract."},
        "count": {"type": "integer", "minimum": 1, "maximum": 252,
                  "description": "Required only for historical-daily: unquoted whole JSON integer 1..252. A quoted string such as a number is invalid."},
        "conid": {"type": "integer", "enum": conids,
                  "description": "Required only for submit-stock-order: unquoted positive whole JSON integer from verified_contracts, matching symbol."},
        "side": {"type": "string", "enum": ["BUY", "SELL"],
                 "description": "Required only for submit-stock-order. BUY purchases shares; SELL disposes of owned shares. No short sale."},
        "quantity": {"type": "integer", "minimum": 1,
                     "description": "Required only for submit-stock-order: unquoted positive whole JSON integer number of shares. Fractional, quoted or zero quantities are invalid."},
        "order_type": {"type": "string", "enum": ["LMT"],
                       "description": "Required only for submit-stock-order. Explicit LMT; no default is supplied."},
        "limit": {"type": ["string", "number"],
                  "description": "Required only for submit-stock-order: positive USD price per share, finite decimal string or JSON number. Model chooses the limit; it is never replaced or clipped."},
        "tif": {"type": "string", "enum": ["DAY"],
                "description": "Required only for submit-stock-order. Explicit DAY; expires at the next execution open if not filled. No default is supplied."},
        "order_ref": {"type": "string", "pattern": ORDER_REF_PATTERN, "minLength": 8, "maxLength": 64,
                      "description": "Required for submit-stock-order (new unique reference) or cancel-order (the queued order's existing exact reference). Must match native CLI syntax ^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$, 8..64 characters. Invalid or duplicate supplied model references are rejected, never rewritten."},
        "sessions": {"type": "integer", "minimum": 1,
                     "description": "Required only for advance: unquoted positive whole JSON integer <= account.sessions_remaining. Simulation clock movement, not an order."},
    }
    signatures = "; ".join(command + "(" + ", ".join(names) + ")" for command, names in COMMAND_FIELDS.items())
    return {"type": "object", "description": "Exact command field sets: " + signatures + ". Omit every other field.",
            "properties": fields, "required": ["command"], "additionalProperties": False,
            "oneOf": [{"type": "object", "properties": {
                name: ({**deepcopy(fields[name]), "enum": [command]} if name == "command" else deepcopy(fields[name]))
                for name in names}, "required": list(names), "additionalProperties": False}
                for command, names in COMMAND_FIELDS.items()]}


def simulator_tool_definition(declaration: dict) -> list[dict]:
    """One native tool with all API semantics visibly rendered in its help."""
    help_text = ("Execute an explicitly declared retrospective portfolio simulator command. "
                 "This tool never connects to IBKR. Call functions.pa_tws with a JSON object in the native tool channel. "
                 "Do not emit shell strings. " + " ".join(command + ": " + text for command, text in COMMAND_SEMANTICS.items())
                 + " An ordinary final response also commits the current strategy for terminal NAV marking; "
                   "already queued orders remain effective, and no order is invented.")
    return [{"type": "function", "function": {"name": "pa_tws", "description": help_text,
            "parameters": argument_schema(declaration)}}]


def public_api_contract(declaration: dict) -> dict:
    return {"argument_schema": argument_schema(declaration), "command_semantics": deepcopy(COMMAND_SEMANTICS),
            "argument_encoding": "Exact JSON object. Integer fields are unquoted whole JSON integers; prices may be finite decimal strings or numbers. No coercion, defaults or extra fields.",
            "termination": "Final response commits current strategy. Terminal valuation advances the remaining clock and settles already queued orders; it never generates liquidation or other orders.",
            "native_cli_transfer": "submit-stock-order uses native CLI field names and order_ref syntax ^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$ is aligned with the native CLI; supplied model references are rejected when invalid, never rewritten. Native historical-daily requires conid and primary_exchange and uses duration/as_of instead of count. Native cancellation is cancel-pa-order with order_id/order_ref/conid. Simulation account bundles reads that the real CLI exposes separately. advance and finish are simulation clocks unavailable in the real broker CLI. No simulation call is translated into a live or paper order. Simulation competence is not evidence of a native broker lifecycle."}
