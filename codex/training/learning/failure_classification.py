"""Future source-classified public failures; no action or reward replacement.

All raw/accounting/ledger admission remains in capture_provenance. This helper
only proves an exact known pre-dispatch parser/schema refusal, decoding public
call bodies and headers; private analysis bodies are never decoded here.
"""
from training.v7_contract import parse_response
from training.framework import declared_tools as declared


def known_pre_dispatch_failure(receipt, native, tokenizer, tools, header_reader):
    if (receipt.get("status") != "MALFORMED_OUTPUT" or receipt.get("tool_invoked") is not False
            or receipt.get("error") != "ValueError" or len(native.get("sequences", [])) != 1):
        return False, "UNCLASSIFIED_FAILURE"
    names = {"functions." + item["function"]["name"] for item in tools}
    try:
        frames, _ = header_reader(native["sequences"][0]["tokens"], tokenizer,
            allowed_tool_names=names, include_public_bodies=True)
        if (receipt.get("message") == "INVALID_DIRECT_TOOL_ACTION"
                and any(row["commentary_action_condition_passes"] is False for row in frames)):
            return True, "KNOWN_PUBLIC_ACTION_FRAMING_REJECTION"
        last = frames[-1]
        if (last["channel"] != "commentary"
                or last["recipient"] not in names or last["content_type"] != "json"
                or last["terminal"] != "<|call|>"):
            return False, "UNCLASSIFIED_FAILURE"
        # The real native sampler invokes this exact parser on the public STR.
        # Syntax, duplicate keys, nonfinite constants and nonobject bodies are
        # known only when the saved terminal error exactly matches its refusal.
        try:
            arguments = parse_response(last["public_body"])
        except ValueError as error:
            if str(error) == receipt.get("message"):
                return True, "KNOWN_SOURCE_PUBLIC_JSON_BODY_REJECTION"
            return False, "UNCLASSIFIED_FAILURE"
        if names == {"functions.pa_tws"}:
            return False, "UNCLASSIFIED_FAILURE"
        try:
            declared.validate_declared_action({"kind": "tool",
                "name": last["recipient"].removeprefix("functions."), "arguments": arguments}, tools)
        except ValueError as error:
            if str(error) == receipt.get("message"):
                return True, "KNOWN_SOURCE_DECLARED_SCHEMA_REJECTION"
    except (KeyError, ValueError, AssertionError, UnicodeError, StopIteration, IndexError):
        pass
    return False, "UNCLASSIFIED_FAILURE"
