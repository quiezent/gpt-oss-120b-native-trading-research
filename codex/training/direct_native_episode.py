"""Separate direct native episodes; no V7 role/output-schema requirement.

Pure protocol/episode helpers derived from the frozen V9 evaluator's source
c7e25d84e0e02f6986059d5ffeffe4f414cf481782b74a3cf6a1b19fb305daf0.
This new module binds the supplied candidate URI and per-case output cap, while
retaining the same strict public Harmony parser, three yields, four steps,
zero-temperature seeded native sampling and raw-before-validation capture.
No original artifact, financial owner or active Trainer source is modified.
"""

from copy import deepcopy

from datetime import datetime, timezone

import re

import time

MAX_OUTPUT = 4096

MAX_TOOL_CALLS = 3

MAX_EPISODE_SECONDS = 600

MAX_SEQUENCE = 32768

def now():
    return datetime.now(timezone.utc).isoformat()

def require(condition, code):
    if not condition:
        raise ValueError(code)

def build_direct_prompt(case, renderer=None):
    """Root-authored direct interface, with no V7 projection/schema validation."""
    from training import v7_protocol as protocol
    renderer = protocol.local_renderer(case["decision_at"][:10]) if renderer is None else protocol._configured_renderer(
        renderer, case["decision_at"][:10])
    definitions = case["tool_definitions"]
    require(type(definitions) is list and len(definitions) > 0, "DIRECT_TOOLS_REQUIRED")
    specs = []
    for definition in definitions:
        spec = deepcopy(definition.get("function", definition))
        require(re.fullmatch(r"[a-z][a-z_]*", spec["name"]) is not None,
                "FLAT_DIRECT_TOOL_NAME_REQUIRED")
        specs.append(spec)
    names = {"functions." + spec["name"] for spec in specs}
    require(len(names) == len(specs), "DUPLICATE_DIRECT_TOOL_DEFINITION")
    for value in (case["system_prompt"], case["user_prompt"]):
        protocol._safe_content(value, renderer)
    messages = renderer.create_conversation_prefix_with_tools(specs, case["system_prompt"])
    messages.append({"role": "user", "content": case["user_prompt"]})
    prompt = renderer.build_generation_prompt(messages)
    require(len(prompt.to_ints()) + MAX_OUTPUT <= MAX_SEQUENCE, "DIRECT_CONTEXT_EXHAUSTED")
    return prompt, renderer, names

def parse_direct_segment(tokens, renderer, tool_names):
    """Parse authoritative public Harmony actions for the separate tool host."""
    from training import v7_protocol as protocol
    from training.v7_contract import parse_response
    raw = list(tokens)
    require(raw and all(type(token) is int and token >= 0 for token in raw), "INVALID_DIRECT_NATIVE_TOKENS")
    require(all(token in protocol._known_token_ids(renderer.tokenizer) for token in raw), "INVALID_DIRECT_TOKEN_VOCABULARY")
    ret, call = protocol.stop_token_ids(renderer)
    require(sum(token in {ret, call} for token in raw) == 1 and raw[-1] in {ret, call},
            "INVALID_DIRECT_NATIVE_TERMINAL")
    controls = set(renderer.tokenizer.get_added_vocab().values())
    start, message, end, channel, constrain = [protocol._single_token(renderer, text) for text in
        ("<|start|>", "<|message|>", "<|end|>", "<|channel|>", "<|constrain|>")]
    index, phase, frames = 0, "analysis", []
    while index < len(raw):
        beginning, implicit = index, index == 0
        require(raw[index] != start if implicit else raw[index] == start, "INVALID_DIRECT_NATIVE_FRAME_START")
        if not implicit:
            index += 1
        require(message in raw[index:], "DIRECT_NATIVE_BODY_SEPARATOR_MISSING")
        header_end = raw.index(message, index)
        header_tokens = raw[index:header_end]
        require(not any(token in controls - {channel, constrain} for token in header_tokens)
                and header_tokens.count(channel) == 1 and header_tokens.count(constrain) <= 1,
                "INVALID_DIRECT_NATIVE_HEADER_CONTROLS")
        header = ("assistant" if implicit else "") + renderer.tokenizer.decode(header_tokens)
        match = re.fullmatch(r"assistant(?: to=(functions\.[a-z_]+))?<\|channel\|>(analysis|commentary|final)"
                             r"(?: to=(functions\.[a-z_]+))?(?: ?<\|constrain\|>((?ai:json)))?", header)
        require(match is not None and not (match[1] and match[3]), "INVALID_DIRECT_NATIVE_HEADER")
        recipient, ch, content_type = match[1] or match[3], match[2], match[4]
        body_start = header_end + 1
        boundary = next((i for i in range(body_start, len(raw)) if raw[i] in {end, ret, call}), None)
        require(boundary is not None and not any(token in controls for token in raw[body_start:boundary]),
                "INVALID_DIRECT_NATIVE_BODY_CONTROL")
        body, terminal = renderer.tokenizer.decode(raw[body_start:boundary]), raw[boundary]
        protocol._safe_content(body, renderer)
        if ch == "analysis":
            require(phase == "analysis" and recipient is None and content_type is None and terminal == end,
                    "DIRECT_PRIVATE_ANALYSIS_HAS_NO_ACTION_AUTHORITY")
        elif ch == "commentary":
            phase = "commentary"
            require((recipient in tool_names and terminal == call and content_type == "json") if recipient else
                    (content_type is None and terminal == end), "INVALID_DIRECT_TOOL_ACTION")
        else:
            phase = "final"
            require(recipient is None and (content_type is None or content_type.lower() == "json")
                    and terminal == ret and boundary == len(raw) - 1, "INVALID_DIRECT_PUBLIC_FINAL")
        frames.append({"channel": ch, "recipient": recipient, "body": body, "terminal": terminal})
        index = boundary + 1
    last = frames[-1]
    if last["terminal"] == call:
        args = parse_response(last["body"])
        require(type(args) is dict, "DIRECT_TOOL_ARGUMENTS_MUST_BE_OBJECT")
        return {"kind": "tool", "tool_name": last["recipient"], "arguments": args, "raw_tokens": raw}
    require(last["channel"] == "final" and last["terminal"] == ret, "DIRECT_PUBLIC_ACTION_MISSING")
    return {"kind": "final", "final_text": last["body"], "raw_tokens": raw}

def append_direct_tool_result(prompt, segment, result, renderer, names, call_id):
    import tinker
    from tinker_cookbook.renderers.base import RenderContext
    from training import v7_protocol as protocol
    require(parse_direct_segment(segment["raw_tokens"], renderer, names) == segment
            and segment["kind"] == "tool", "DIRECT_TOOL_TRACE_CHANGED")
    content = canonical(result)
    protocol._safe_content(content, renderer)
    rendered = renderer.render_message({"role": "tool", "name": segment["tool_name"],
        "content": content, "tool_call_id": call_id}, RenderContext(idx=0, is_last=True))
    result_tokens = rendered.header.tokens + [token for chunk in rendered.output for token in chunk.tokens]
    suffix = renderer.tokenizer.encode("<|start|>assistant", add_special_tokens=False)
    require(prompt.to_ints()[-len(suffix):] == suffix, "DIRECT_NATIVE_PREFIX_CHANGED")
    tokens = prompt.to_ints() + segment["raw_tokens"] + result_tokens + suffix
    require(len(tokens) < MAX_SEQUENCE, "DIRECT_CONTINUATION_CONTEXT_EXHAUSTED")
    return tinker.ModelInput.from_ints(tokens)

from training.native_agent.runtime import canonical
