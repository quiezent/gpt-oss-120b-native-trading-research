"""Exact durable sampling/claim/usage proof; no clients, decoding of bodies.

Model-chosen action framing failures can be eligible without repaired actions.
Other missing/incomplete/unknown outcomes remain explicitly excluded.
"""
from pathlib import Path
import hashlib
import json
import re

from training.learning.evidence_eligibility import captured_trajectory_integrity, TrajectoryIntegrity, value_digest


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def read_verified_bytes(ref):
    if reference(ref["path"]) != ref:
        raise ValueError("DURABLE_PROVENANCE_REFERENCE_CHANGED")
    return Path(ref["path"]).read_bytes()


def strict_json(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("DUPLICATE_NATIVE_CAPTURE_KEY")
            out[key] = value
        return out
    def bad(_):
        raise ValueError("NONFINITE_NATIVE_CAPTURE")
    return json.loads(data, object_pairs_hook=unique, parse_constant=bad)


def read_verified(ref):
    return strict_json(read_verified_bytes(ref))


def journal_records(data):
    if not data or not data.endswith(b"\n"):
        raise ValueError("COMPLETE_DURABLE_ACCOUNTING_JOURNAL_REQUIRED")
    records = [strict_json(line) for line in data.splitlines()]
    previous = None
    for index, record in enumerate(records):
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (record.get("record_sha256") != value_digest(body)
                or type(body.get("sequence")) is not int or body["sequence"] != index + 1
                or body.get("previous_sha256") != previous):
            raise ValueError("DURABLE_ACCOUNTING_JOURNAL_HASH_CHAIN_BROKEN")
        previous = record["record_sha256"]
    if records[0]["kind"] != "CARRYFORWARD":
        raise ValueError("ORIGINAL_ACCOUNTING_JOURNAL_PREFIX_REQUIRED")
    return records


def persist_journal_snapshot(journal_path, destination, *, anchor_ref=None):
    """Read current completed journal bytes once; never construct or mutate Budget.

    Hash chaining detects byte changes; it is not a provider signature or invoice.
    The frozen runner reads the actual Budget path after completed accounting.
    """
    data = Path(journal_path).read_bytes()
    records = journal_records(data)
    if anchor_ref is not None and not data.startswith(read_verified_bytes(anchor_ref)):
        raise ValueError("ACCOUNTING_JOURNAL_CHANGED_FROM_RUN_START_PREFIX")
    with Path(destination).open("xb") as stream:
        stream.write(data)
    return reference(destination), {"record_count": len(records),
        "last_record_sha256": records[-1]["record_sha256"],
        "source_path": str(Path(journal_path).resolve())}


def verified_accounting_rows(snapshot_ref, anchor_ref, captures):
    data, anchor = read_verified_bytes(snapshot_ref), read_verified_bytes(anchor_ref)
    if not data.startswith(anchor):
        raise ValueError("ACCOUNTING_JOURNAL_CHANGED_FROM_RUN_START_PREFIX")
    anchor_count = len(journal_records(anchor))
    records = journal_records(data)
    joined = {}
    for capture in captures:
        claim, completed = capture["claim"], capture["completed_request"]
        request_id = claim.get("request_id")
        matches = [row for row in records if row.get("payload", {}).get("request_id") == request_id]
        claims = [row for row in matches if row["kind"] == "DISPATCH_CLAIMED"]
        usages = [row for row in matches if row["kind"] == "USAGE_ESTIMATED"]
        if (len(claims) != 1 or len(usages) != 1 or len(matches) != 2
                or claims[0]["sequence"] >= usages[0]["sequence"]
                or claims[0]["sequence"] <= anchor_count):
            raise ValueError("ONE_ACTUAL_POST_ANCHOR_CLAIM_AND_COMPLETED_USAGE_REQUIRED")
        expected_claim = {**claims[0]["payload"], "claim_record_sha256": claims[0]["record_sha256"],
            "logical_retry_allowed": False}
        usage = usages[0]["payload"]
        expected_completed = {**claims[0]["payload"], "status": "completed",
            "estimated_actual_nanos": usage["estimated_actual_nanos"], "usage": usage["usage"]}
        if value_digest(claim) != value_digest(expected_claim) or value_digest(completed) != value_digest(expected_completed):
            raise ValueError("EXACT_DURABLE_JOURNAL_CLAIM_USAGE_JOIN_REQUIRED")
        joined[request_id] = {"claim_record_sha256": claims[0]["record_sha256"],
            "usage_record_sha256": usages[0]["record_sha256"]}
    return joined


def trajectory_proof(outcome, directory, captures, *, expected_sampler,
        expected_temperature, expected_operation, request_namespace, tokenizer_ref,
        journal_snapshot_ref, journal_anchor_ref, max_steps=12,
        max_sequence_tokens=32768, max_output_tokens=8192, tool_definitions=None):
    sealed_outcome_path = Path(directory) / "economic_outcome.json"
    if value_digest(read_verified(reference(sealed_outcome_path))) != value_digest(outcome):
        raise ValueError("UNCHANGED_SAVED_SEALED_ECONOMIC_OUTCOME_REQUIRED")
    tokenizer = read_verified(tokenizer_ref)
    known_token_ids = set(tokenizer["model"]["vocab"].values()) | {row["id"] for row in tokenizer["added_tokens"]}
    paths = sorted(Path(directory).glob("step-*-receipt.json"))
    if len(paths) != len(captures) or not captures:
        return TrajectoryIntegrity(False, "NO_EXACT_DURABLE_CAPTURE_FOR_ALL_RUNTIME_STEPS"), []
    accounting_rows = verified_accounting_rows(journal_snapshot_ref, journal_anchor_ref, captures)
    records, request_ids = [], set()
    for path, capture in zip(paths, captures):
        receipt = read_verified(reference(path))
        persisted_capture = read_verified(capture["provenance_ref"])
        if value_digest(persisted_capture) != value_digest({key: value for key, value in capture.items() if key != "provenance_ref"}):
            raise ValueError("PERSISTED_SAMPLING_ACCOUNTING_PROVENANCE_CHANGED")
        raw_ref = receipt.get("raw_model_ref")
        if raw_ref is None:
            return TrajectoryIntegrity(False, "RUNTIME_STEP_HAS_NO_DURABLE_RAW_REF"), []
        native = read_verified(raw_ref)
        sample = read_verified(capture["native_capture"])
        claim, completed = capture["claim"], capture["completed_request"]
        request_id = claim.get("request_id")
        if request_id in request_ids or not isinstance(request_id, str):
            raise ValueError("UNIQUE_ACTUAL_SAMPLING_REQUESTS_REQUIRED")
        request_ids.add(request_id)
        native_name = Path(capture["native_capture"]["path"]).stem
        if not re.fullmatch(r"native-sample-[1-9][0-9]*", native_name):
            raise ValueError("EXACT_NATIVE_SAMPLING_CAPTURE_IDENTITY_REQUIRED")
        serial = native_name.rsplit("-", 1)[1]
        if request_id != request_namespace + "-sample-" + serial or claim.get("operation_id") != expected_operation:
            raise ValueError("EXACT_OWNED_OPERATION_AND_REQUEST_IDENTITY_REQUIRED")
        sequence = native.get("sequences", [])
        output_count = sum(len(row.get("tokens", [])) for row in sequence)
        usage = {"input_tokens": len(native.get("prompt_tokens", [])), "output_tokens": output_count}
        request_digest = value_digest(native["native_request"])
        raw_verified = raw_ref["sha256"] == capture["native_capture"]["sha256"]
        request_verified = (claim.get("request_kind") == "sampling"
            and claim.get("request_sha256") == request_digest == capture["native_request_digest"]
            and claim["claim_record_sha256"] == accounting_rows[request_id]["claim_record_sha256"]
            and value_digest(claim.get("request_limits")) == value_digest({key: native["native_request"][key]
                for key in ("input_tokens", "max_output_tokens", "max_sequence_tokens")})
            and claim.get("maximum_cost_nanos") == usage["input_tokens"] * 330
                + native["native_request"]["max_output_tokens"] * 840)
        accounted = (completed.get("status") == "completed" and completed.get("request_id") == request_id
            and completed.get("operation_id") == claim.get("operation_id")
            and completed.get("request_kind") == "sampling"
            and completed.get("request_sha256") == request_digest
            and completed.get("maximum_cost_nanos") == claim.get("maximum_cost_nanos")
            and value_digest(completed.get("request_limits")) == value_digest(claim.get("request_limits"))
            and value_digest(completed.get("usage")) == value_digest(usage)
            and type(completed.get("estimated_actual_nanos")) is int
            and type(claim.get("maximum_cost_nanos")) is int
            and 0 <= completed["estimated_actual_nanos"] <= claim["maximum_cost_nanos"])
        from training.learning.failure_classification import known_pre_dispatch_failure
        failure_verified, failure_reason = known_pre_dispatch_failure(receipt, native, tokenizer,
            tool_definitions or [{"function": {"name": "pa_tws"}}], header_metadata)
        records.append({"receipt": receipt, "native": native,
            "raw_verified": raw_verified, "exact_sample_join_verified": value_digest(native) == value_digest(sample),
            "request_digest_verified": request_verified, "accounted_verified": accounted,
            "accounting_journal_snapshot": journal_snapshot_ref,
            "accounting_journal_anchor": journal_anchor_ref,
            "accounting_journal_records": accounting_rows[request_id],
            "known_pre_dispatch_failure_verified": failure_verified, "known_failure_source_classification": failure_reason})
    proof = captured_trajectory_integrity(outcome, records,
        expected_sampler=expected_sampler, expected_temperature=expected_temperature,
        known_token_ids=known_token_ids, max_steps=max_steps,
        max_sequence_tokens=max_sequence_tokens, max_output_tokens=max_output_tokens,
        max_generated_tokens=max_output_tokens * max_steps)
    return proof, records


def verify_datum_step_binding(proof, records, steps):
    if not proof.eligible or proof.captured_records_sha256 != value_digest(records) or len(records) != len(steps):
        raise ValueError("EXACT_ELIGIBLE_CAPTURE_RECORDS_TO_DATUM_BINDING_REQUIRED")
    for record, step in zip(records, steps):
        native = record["native"]
        if len(native["sequences"]) != 1:
            raise ValueError("EXACT_ONE_SAMPLED_SEQUENCE_REQUIRED")
        sequence = native["sequences"][0]
        expected = {"prompt_tokens": native["prompt_tokens"], "tokens": sequence["tokens"], "logprobs": sequence["logprobs"]}
        if value_digest(step) != value_digest(expected):
            raise ValueError("UNCHANGED_ACTUAL_NATIVE_DATUM_STEPS_REQUIRED")
    return True


def header_metadata(tokens, tokenizer, *, allowed_tool_names=None, include_public_bodies=False):
    allowed_tool_names = {"functions.pa_tws"} if allowed_tool_names is None else set(allowed_tool_names)
    vocab = {value: key for key, value in tokenizer["model"]["vocab"].items()}
    vocab.update({row["id"]: row["content"] for row in tokenizer["added_tokens"]})
    ids = {row["content"]: row["id"] for row in tokenizer["added_tokens"]}
    # Literal ByteLevel inverse. Decode control headers only; skip body spans.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    inverse = {chr(code): byte for byte, code in zip(bs, cs)}
    def decode_header(header_tokens):
        chunks, buf = [], bytearray()
        for token in header_tokens:
            word = vocab[token]
            if word in ids:
                chunks.extend((buf.decode("utf-8"), word))
                buf.clear()
            else:
                buf.extend(inverse[char] for char in word)
        chunks.append(buf.decode("utf-8"))
        return "".join(chunks)
    index, frames = 0, []
    boundaries = {ids["<|end|>"], ids["<|return|>"], ids["<|call|>"]}
    while index < len(tokens):
        implicit = index == 0
        if not implicit:
            assert tokens[index] == ids["<|start|>"]
            index += 1
        header_end = tokens.index(ids["<|message|>"], index)
        header = ("assistant" if implicit else "") + decode_header(tokens[index:header_end])
        match = re.fullmatch(r"assistant(?: to=(functions\.[a-z_]+))?<\|channel\|>(analysis|commentary|final)(?: to=(functions\.[a-z_]+))?(?: ?<\|constrain\|>((?ai:json)))?", header)
        assert match is not None
        beginning = header_end + 1
        boundary = next(i for i in range(beginning, len(tokens)) if tokens[i] in boundaries)
        recipient, channel, content_type = match[1] or match[3], match[2], match[4]
        terminal = vocab[tokens[boundary]]
        valid = ((recipient in allowed_tool_names and terminal == "<|call|>" and content_type == "json")
                 if recipient else content_type is None and terminal == "<|end|>")
        frames.append({"frame_index": len(frames), "channel": channel, "recipient": recipient,
            "content_type": content_type, "terminal": terminal, "body_token_count": boundary - beginning,
            "commentary_action_condition_passes": valid if channel == "commentary" else None})
        if include_public_bodies and channel == "commentary":
            frames[-1]["public_body"] = decode_header(tokens[beginning:boundary])
        index = boundary + 1
    return frames, set(vocab)
