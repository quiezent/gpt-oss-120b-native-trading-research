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


def known_local_argument_refusal(receipt, native, tokenizer, tools, header_reader):
    """Prove the exact visible local refusal without decoding private bodies."""
    from pathlib import Path
    import hashlib
    import json
    from training.learning.argument_recovery.visible_errors import argument_error, refused_result
    def same(a, b):
        return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (receipt.get('status') != 'RUNNING' or receipt.get('tool_invoked') is not False
            or receipt.get('local_argument_validation_refused') is not True
            or len(native.get('sequences', [])) != 1 or native['sequences'][0].get('stop_reason') != 'stop'):
        return False, 'NOT_COMPLETE_LOCAL_ARGUMENT_REFUSAL'
    names = {'functions.' + item['function']['name'] for item in tools}
    if names == {'functions.pa_tws'}:
        return False, 'WRAPPER_HAS_NO_DECLARED_ARGUMENT_REFUSAL_PATH'
    try:
        frames, _ = header_reader(native['sequences'][0]['tokens'], tokenizer,
            allowed_tool_names=names, include_public_bodies=True)
        last = frames[-1]
        phase = 'analysis'
        for row in frames[:-1]:
            if row['channel'] == 'analysis':
                if phase != 'analysis' or row['recipient'] is not None or row['content_type'] is not None or row['terminal'] != '<|end|>':
                    return False, 'LOCAL_ARGUMENT_REFUSAL_PUBLIC_PHASE_UNVERIFIED'
            elif row['channel'] == 'commentary':
                phase = 'commentary'
                if row['recipient'] is not None or row['content_type'] is not None or row['terminal'] != '<|end|>':
                    return False, 'LOCAL_ARGUMENT_REFUSAL_PUBLIC_PHASE_UNVERIFIED'
            else:
                return False, 'LOCAL_ARGUMENT_REFUSAL_PUBLIC_PHASE_UNVERIFIED'
        if (last['channel'] != 'commentary' or last['recipient'] not in names
                or last['content_type'] != 'json' or last['terminal'] != '<|call|>'
                or last['commentary_action_condition_passes'] is not True):
            return False, 'LOCAL_ARGUMENT_REFUSAL_PUBLIC_FRAMING_UNVERIFIED'
        arguments = parse_response(last['public_body'])
        action = {'kind': 'tool', 'name': last['recipient'].removeprefix('functions.'), 'arguments': arguments}
        message = argument_error(action, tools)
        if message is None or not same(receipt.get('action'), action) or not same(receipt.get('tool_result'), refused_result(action, message)):
            return False, 'LOCAL_ARGUMENT_REFUSAL_EXACT_ERROR_OR_ACTION_CHANGED'
        ref = receipt['action_ref']
        raw = Path(ref['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256'] or not same(json.loads(raw), action):
            return False, 'LOCAL_ARGUMENT_REFUSAL_DURABLE_ACTION_CHANGED'
        return True, 'EXACT_SOURCE_KNOWN_ARGUMENT_ERROR_RETURNED_UNCHANGED_TO_MODEL'
    except (KeyError, ValueError, TypeError, AssertionError, UnicodeError, StopIteration, IndexError, OSError):
        return False, 'LOCAL_ARGUMENT_REFUSAL_PROVENANCE_UNKNOWN'
