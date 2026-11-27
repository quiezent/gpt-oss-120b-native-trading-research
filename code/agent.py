"""One look: a model reads its sleeve and the market, may research with tools.py, then submits orders.

gptoss_look  runs base / trained gpt-oss checkpoints on Tinker with native (Harmony) tool calls.
claude_look  runs a Claude model through the Claude CLI, headless, with every built-in tool disabled and
             only mcp_server.py's tools (the same research tools and submit).
Both return (call or None, trace), where trace records every research call for the research log.
"""

import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import tinker

import checks
import tools
import trader

ROOT = Path(__file__).resolve().parent
# openai/gpt-oss-120b (rl2's base) has a 32,768-token context on Tinker (server capabilities). At effort "high"
# rl2-step010 wrote turns up to 7,174 tokens and whole looks up to 19,313 (18 Oct, 12 simulated looks), so each turn
# may use up to MAX_OUT, less whatever the context has no room for.
CONTEXT, MAX_OUT, MIN_ROOM = 32768, 16384, 1024
# The context reserve for the decision (16 Nov 2026; the Trading Model Researcher, 15 Nov): for sleeves with the forecast
# tool the prompt at the final turn runs a median 23,400 tokens (90th percentile 27,700) and the final answer a median 3,000
# (90th percentile 6,000); one check_orders adds about 1,400 to 2,100. So the answer would not fit in 12% of looks after
# one check, 26% after two and 39% after three, and room would fall under MIN_ROOM (no decision at all) in about 1%, 6% and
# 13%. Under RESERVE the submit is opened instead of a free turn; a tool call whose result would take room under RESERVE
# is not run; a look ends with no decision only when even the opened submit has under MIN_SUBMIT tokens of room. Only
# where a look would otherwise be lost: a look that stays clear of the reserve runs exactly as before (all gpt-oss
# sleeves, text A too: trader.text_of's fourth agreed exception).
RESERVE, MIN_SUBMIT = 2500, 300
# What a tool-call turn must leave for the decision (16 Nov 2026; the Trading Model Researcher's checker, 15 Nov): a long
# turn ending in a tool call could use nearly all the room, and the look then had too little left for the decision (no
# decision in 12 of 164 looks with 3,000-token turns that only call check_orders, 0 of 164 with 1,500-token turns).
# A free turn is still sampled with all the room, so a long decision that fits is kept as written (the coordinator's count
# of 9-13 Nov: submit turns run a median 2,974 tokens, p90 6,336, max 13,518; capping every free turn would have cut 28
# of 682 decisions that fit, and only 2 tool-call turns). Instead:
# - a free turn that ends in a tool call (not submit) and leaves less than SUBMIT_ROOM is dropped: not added to the
#   prompt, its tool not run, nothing counted; the submit is opened at the prompt as it was before that turn;
# - a turn cut off at max_tokens is a malformed turn, as at fd95e2a: re-asked free at the same prompt, and the submit is
#   opened at the second (two reviewers, 16 Nov: at temperature 1 the re-ask often writes a reasoned decision);
# - a tool's own result is cut so that SUBMIT_ROOM is left after the submit's prefix, keeping its counter line last;
#   the framework's own one-line replies are added whole.
SUBMIT_ROOM = 1500
# Reasoning before an opened submit (15 Nov 2026; switch off until it is turned on): free submits never sent an empty
# order list (0 of 640 live looks in week 5, 0 of 711 in a held-out read), but opened ones did in about 1 in 7
# (3 of 25 live, 18 of 112 in the read), most with the orders meant named in "notes": the model closed "place" before it
# had reasoned. With FORCED_MODE "reasoning" the opened submit is written in two samples in one step: first the model's own
# reasoning on the analysis channel (until <|end|>), then the call with that reasoning in front of it, as a free submit
# reads. Nothing is chosen between samples and nothing is re-asked for what the decision says. MIN_REASON: the least
# reasoning room worth a phase; under it the plain opened submit is used. 600 tokens is room for a few sentences naming
# the orders and why (a cut reasoning is kept, so a small room costs a cut, not a turn); a submit the reserve opens
# (room under RESERVE) reasons only when its room is at least SUBMIT_ROOM + MIN_REASON + the three prefixes (2,117).
# The cost (review, 15 Nov): a reasoning run to its cap leaves the call SUBMIT_ROOM, less than the plain call would have,
# so a long JSON that fits plain can be cut; a cut call after reasoning is one malformed turn, and every later try in the
# look is then the plain opened submit.
# FORCED_MODE "notes" is the cheaper way to the same end (the Researcher's checkers, 15 Nov): the opened submit's prefix
# ends with NOTES_OPEN(), {"notes": ", so the model writes its plan before "place", in one sample as today. Its costs: a
# plan written inside a JSON string must escape its line breaks and quotes (a raw one is a malformed turn); "notes" is
# what the sleeve reads back at its next look; and an opened submit then no longer has a free one's form.
# One setting, three values; "plain" is the opened submit as through 2edf75f. Which one runs live is Clayton's decision on
# the Researcher's paired read (15 Nov: "the forced-submit fix ships Monday if the Researcher's paired test backs it").
FORCED_MODES = ("plain", "reasoning", "notes")
FORCED_MODE = "plain"
MIN_REASON = 600
# The line that opens the submit says so (15 Nov 2026; off until it is turned on). The Researcher's forced-submit test
# (models-w3 2609be3): every empty order list of base, in every opening, sat on states whose last line was "research limit
# reached (12 calls): call submit now" (plain 28 of 90 draws; 0 of 54 on the other forced states), and where the model
# reasoned first, each empty one ended by reaching for another tool call (check_orders in 7 of 16): it did not know that
# the opened submit is final and that no tool runs before it. With FINAL_LINE the occurrence of a limit line that opens
# the submit states those facts. The first occurrence stays as it was: a check can still run after it. "check_orders
# included" is said only to a look that has that tool.
FINAL_LINE = False


def final_words(has_check):
    """What a limit line says when it opens the submit (FINAL_LINE): facts about the loop, nothing about the decision."""
    return ("no more tool calls are run in this look" + (", check_orders included" if has_check else "")
            + ". Your next message is the submit call; it ends the look.")
EST_RESEARCH, EST_CHECK, EST_CHECK_ORDER = 1500, 300, 110    # estimated result tokens: a research call; a check, + per order
EST_OWN_LIMIT = 150     # + a check's exposure-limit line (own_limit.check_block, 18 Nov; about 120 tokens), only for a
                        # checker with own_limit set (live.py), so a look without the line runs as before
OWN_LIMIT_HEADS = ("Your sleeve, by your own exposure limit", "Your sleeve's value is not positive, so it has no room")
CUT_LINE = "(result cut: context nearly full)"
CLAUDE_TIMEOUT = 1200


# ---------------------------------------------------------------- gpt-oss (Tinker)
SEG = trader.SEG


def is_submit(val):
    """A decision in form: place a list, cancel a list or absent, and no literal "..." left from a template (28 Oct, base-ml's forced
    turn: {"symbol": "...", "quantity": "..."} and notes "...", with a real ref among its cancels, cancelled that order and wiped the notes)."""
    if not isinstance(val, dict) or not isinstance(val.get("place"), list) or not isinstance(val.get("cancel", []), list):
        return False
    return "..." not in val.get("cancel", []) + [val.get("notes")] + [v for o in val["place"] if isinstance(o, dict) for v in o.values()]


def submit_in(text):
    """The submit's JSON if it is the whole text, give or take a short label and a code fence, else None: with words around it or a
    second object, a draft or a choice can't be told from a decision, so those turns are re-asked."""
    found, dec, i = [], json.JSONDecoder(), text.find("{")
    while i >= 0:
        try:
            val, end = dec.raw_decode(text, i)
        except (ValueError, RecursionError):
            end = i + 1
        else:
            if isinstance(val, dict) and ("place" in val or "cancel" in val):
                found.append((val, i, end))
        i = text.find("{", end)
    if len(found) != 1:
        return None
    val, start, end = found[0]
    return val if len(text[:start].strip()) <= 20 and not text[end:].strip(" \t\r\n`") and is_submit(val) else None


def unwrap_args(name, args):
    """A call's arguments, out of an OpenAI-style envelope {"name": <this tool>, "arguments": {...}} if the model wrote one. Week 5
    (9-13 Nov): h-ml at 11:00 on 9 Nov wrote all 12 research calls and 3 submits this way (13 envelopes ran as is, so every tool saw
    no symbols, and the 3 submits were rejected as malformed: no decision); c1 on 11 Nov wrote one with no recipient. Only exactly
    these two keys, a dict inside, and the name of the tool called (with or without "functions."): no tool's parameters (tools.py,
    forecast, submit) are "name" and "arguments", and an envelope naming another tool is left for that tool's own error."""
    if (isinstance(args, dict) and set(args) == {"name", "arguments"} and isinstance(args["arguments"], dict)
            and str(args["name"]).strip().removeprefix("functions.") == name):
        return args["arguments"]
    return args


def parse_any(tokens, stop_reason, names):
    """('TOOL', name, args) | ('FINAL', None, text) | ('MALFORMED', None, reason). A closing turn that holds the model's decision is
    that decision, however it ended: of the 8 prose turns from 26 Oct to 2 Nov, 3 were whole calls ending in <|return|> and 4
    answers holding the submit's JSON (3 of them little else). Rewriting them with no reasoning turn changed 4 of those 7 decisions."""
    tok = trader.tokenizer()
    if str(stop_reason).split(".")[-1].lower() != "stop" or not tokens or tokens[-1] not in (trader.CALL, trader.RET):
        return "MALFORMED", None, "output ended without a tool call"
    segs = tok.decode(tokens).split("<|start|>assistant")
    for k, seg in enumerate(segs):
        m = SEG.match(seg)
        if not m or "<|" in m["body"]:
            return "MALFORMED", None, f"unparseable segment {k}"
        rc = m["r1"] or m["r2"]
        if k < len(segs) - 1:
            if m["term"] != "<|end|>" or rc:
                return "MALFORMED", None, "a call before the end"
            continue
        if m["term"] == "<|call|>" or rc:
            name = (rc or "").removeprefix("functions.")
            if name not in names:
                return "MALFORMED", None, f"call to undeclared tool {rc!r}"
            try:
                args = json.loads(m["body"])
            except (ValueError, RecursionError) as e:
                return "MALFORMED", None, f"bad JSON: {e}"
            args = unwrap_args(name, args)
            if name == "submit" and not is_submit(args):   # e.g. orders under another key (21 Oct, rl3-ml): told nothing, it would be an empty hold
                return "MALFORMED", None, "submit arguments malformed"
            return "TOOL", name, args
        decision = submit_in(m["body"]) if m["ch"] != "analysis" and "submit" in names else None
        return ("TOOL", "submit", decision) if decision else ("FINAL", None, m["body"])
    return "MALFORMED", None, "empty"


def SUBMIT_OPEN():
    return list(trader.tokenizer().encode("<|channel|>commentary to=functions.submit <|constrain|>json<|message|>",
                                          add_special_tokens=False))


def NOTES_OPEN():
    """What follows the opened submit's prefix in FORCED_MODE "notes": the call's JSON begun at its notes."""
    return list(trader.tokenizer().encode('{"notes": "', add_special_tokens=False))


def ANALYSIS_OPEN():
    return list(trader.tokenizer().encode("<|channel|>analysis<|message|>", add_special_tokens=False))


def REASON_GLUE():
    """What joins a reasoning phase to the submit's prefix: its <|end|> (the loop's own if the phase was cut at max_tokens,
    else the one sampled, which the phase's text leaves out) and the next segment's start."""
    return list(trader.tokenizer().encode("<|end|><|start|>assistant", add_special_tokens=False))


def END_TOKEN():
    """<|end|>'s id (200007), from the tokenizer: the reasoning phase's only stop."""
    (end,) = trader.tokenizer().encode("<|end|>", add_special_tokens=False)
    return end


def _reasoning(tokens):
    """(the reasoning's tokens, cut, why not usable or None) of a reasoning phase's sample: an ending <|end|> is not part of
    it (cut: there was none, so it ran to max_tokens); unusable if empty, or if it holds a special token or the text "<|"
    (parse_any would call the segment unparseable)."""
    tok = trader.tokenizer()
    cut = not (tokens and tokens[-1] == END_TOKEN())
    body = list(tokens) if cut else list(tokens[:-1])
    if not body:
        return body, cut, "empty"
    if any(t >= tok.vocab_size or t in tok.all_special_ids for t in body):
        return body, cut, "a special token"
    if "<|" in tok.decode(body):
        return body, cut, 'the text "<|"'
    return body, cut, None


def tool_result_tokens(name, text):
    s = f"<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{text}<|end|><|start|>assistant"
    return list(trader.tokenizer().encode(s, add_special_tokens=False))


def harmony_specs(specs):
    """Tool specs for the gpt-oss renderer, which writes a description as one "// " comment line: a multi-line
    description (forecast's) gets "// " on every line, as the Harmony format writes it."""
    return [dict(s, description=s["description"].replace("\n", "\n// ")) if "\n" in s.get("description", "") else s
            for s in specs]


SAMPLE_TIMEOUT = 600   # s a call; a 16,384-token answer takes about 5 minutes


def sample_once(sc, renew, model_input, params, trace, step):
    """One sample with a deadline. tinker-feedback #151 (open; Tinker Owner, 25 Oct): a dead pooled connection can hang a
    sample forever, which in live.py would stall every later sleeve's look. On a timeout, retry once on a new client from
    renew(); a second timeout raises, ending only this look."""
    try:
        return sc.sample(model_input, 1, params).result(timeout=SAMPLE_TIMEOUT), sc
    except TimeoutError:
        if renew is None:
            raise
        trace.append({"step": step, "error": f"sample timed out after {SAMPLE_TIMEOUT} s; retried on a new client"})
        sc = renew()
        return sc.sample(model_input, 1, params).result(timeout=SAMPLE_TIMEOUT), sc


def _estimate(name, args, checker=None):
    """The tokens a tool's result is expected to take, to decide before running it whether it leaves room for the decision:
    a check, 300 plus 110 an order (at most the 20 the tool takes), plus 150 when the checker carries the sleeve's
    exposure-limit line (its own_limit attribute, set by live.py; 18 Nov); a research call, 1,500."""
    if name == "check_orders":
        place = checks.unwrap(args).get("place") if isinstance(checks.unwrap(args), dict) else None
        return (EST_CHECK + (EST_OWN_LIMIT if getattr(checker, "own_limit", None) else 0)
                + EST_CHECK_ORDER * min(len(place) if isinstance(place, list) else 0, checks.MAX_ORDERS))
    return EST_RESEARCH


def _fit_text(name, body, left, counter=None):
    """(the result's tokens as the prompt takes them, the text) for a tool's own result: body, then its counter line if
    it has one ("(research call 7 of 12)"). As it is (body + "\\n" + counter) if it leaves at least SUBMIT_ROOM after the
    opened submit's prefix; else the body is cut to fit, then CUT_LINE, then the counter, so the model still reads its
    count (review, 16 Nov). The body keeps its most whole lines; if not even its first line fits, as much of that line as
    fits (cut at a character, by its token count); if nothing of it fits, CUT_LINE and the counter alone.
    A check's last line, the sleeve's own exposure-limit line (own_limit.check_block, 18 Nov), is kept after CUT_LINE
    when the body is cut and it fits there with nothing else kept (the review: cutting from the end lost it first)."""
    budget = left - SUBMIT_ROOM - len(SUBMIT_OPEN())  # the opened submit keeps SUBMIT_ROOM (16 Nov, the checker's case)
    tail = [CUT_LINE] + ([counter] if counter is not None else [])
    whole = body + ("\n" + counter if counter is not None else "")
    res = tool_result_tokens(name, whole)
    if len(res) <= budget:
        return res, whole
    lines = body.split("\n")
    if name == "check_orders" and len(lines) > 1 and lines[-1].startswith(OWN_LIMIT_HEADS):
        keep = [CUT_LINE, lines[-1]] + ([counter] if counter is not None else [])
        if len(tool_result_tokens(name, "\n".join(keep))) <= budget:
            tail, lines = keep, lines[:-1]
    fits = lambda kept: len(tool_result_tokens(name, "\n".join(kept + tail))) <= budget
    lo, hi = 0, len(lines) - 1        # the most whole lines that fit with the cut line and the counter after them
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(lines[:mid]) else (lo, mid - 1)
    kept = lines[:lo]
    if not kept:                      # not one whole line: as much of the first as fits, cut at a character
        first, a, b = lines[0], 0, len(lines[0])
        while a < b:
            mid = (a + b + 1) // 2
            a, b = (mid, b) if fits([first[:mid]]) else (a, mid - 1)
        kept = [first[:a]] if a else []
    text = "\n".join(kept + tail)
    return tool_result_tokens(name, text), text


def gptoss_look(sc, gw, system, state_text, current_date, research_specs, effort, seed=None, renew=None, ml_tool=None,
                lock=None, submit_tool=None, sleeve_value=None, look=None, checker=None, forced_reasoning=None,
                forced_mode=None, final_line=None, max_calls=None, context=None):
    """lock: held around each research call (live.py runs the gpt-oss sleeves side by side on one client-10
    connection, and the tool modules keep caches); None runs them as before. submit_tool: the sleeve's submit form (None:
    trader.TOOL_LIVE, text B's; live.py gives the text-A control trader.TOOL_LIVE_PROBS). sleeve_value and look go to
    tools.run for the forecast tool (its sleeve% base, and a key for this look so its legend prints once). checker: text B's
    check_orders, a callable(args) -> (text, counted, raw) (live.py: checks.check on the look's shared what-if connection);
    None (text A, or no connection) offers no such tool. Clayton, 14 Nov: up to checks.MAX_CHECKS a look, outside the
    research calls, so a check never uses up a research call. Every check call counts toward checks.MAX_TRIES, served or
    not; the first call past the limit is told it, and a further one is a call past a limit, as for research: the first is
    told to submit, the next opens the submit call (review, 15 Nov: a look always reaches a submit). No lock: the Gateway
    is thread-safe and a check touches no tool cache.
    Near the end of the context (16 Nov 2026), so that a look always reaches a decision:
    - with less than RESERVE tokens of room the submit is opened instead of a free turn;
    - every turn may use all the room, so a long decision that fits is kept as written; a turn cut off at max_tokens is
      a malformed turn as before (re-asked free at the same prompt; the submit is opened at the second);
    - a free turn that ends in a tool call and leaves less than SUBMIT_ROOM is dropped (not added, not run, not counted)
      and the submit is opened at the prompt as it was before it;
    - a tool call whose result would leave less than RESERVE is not run: one line says so, and the submit is opened;
    - a tool's own result is cut so SUBMIT_ROOM is left after the submit's prefix: its text at a line break (or inside
      its first line if no whole line fits), then CUT_LINE, then its counter line; the framework's own lines (that one,
      the research-limit and check-limit texts) are added whole; the trace holds what the model read;
    - a look ends with no decision ("context full") only when even the opened submit has under MIN_SUBMIT tokens.
    forced_mode (None: the module's FORCED_MODE, "plain" until Clayton turns another on) is how a submit the loop opens
    is written: "plain" as through 2edf75f, "reasoning" or "notes". forced_reasoning is the first build's keyword and
    is kept for its tests: True is "reasoning", False is "plain", and forced_mode wins when both are given. An unknown
    mode raises before anything is sampled.
    With "notes" the opened submit's prefix is followed by NOTES_OPEN(), so the model writes its plan in "notes" before
    "place": one sample, as plain, with that many tokens less room; the entry adds "opened": "notes". It is opened plain
    on the last try (two malformed turns already) and after a notes-first call was cut at max_tokens. A JSON with no
    "place" list is a malformed turn, as always, so a call that holds only notes is not read as an empty decision.
    With "reasoning" an opened submit is
    written in two samples in the same step, so its call has reasoning in front of it as a free submit has:
    - first the reasoning: the prompt then ANALYSIS_OPEN(), stopped on <|end|> (and on <|return|> and <|call|>, which
      make it unusable), seed seed * 100 + 50 + step, with max_tokens leaving SUBMIT_ROOM after REASON_GLUE() and the
      submit's prefix (at most MAX_OUT);
    - then the call: the prompt, ANALYSIS_OPEN(), the reasoning, REASON_GLUE() and the submit's prefix, sampled as the
      opened submit is without the switch (its seed, stops and max_tokens of all the room left); the whole turn is parsed
      and traced as a free one is, its entry with "reasoned": {"tokens", "cut"} and "tokens" the step's total;
    - a reasoning cut at max_tokens is kept as written and the loop writes its <|end|>; that is not a malformed turn;
    - the plain opened submit is used instead when the reasoning would have under MIN_REASON tokens, on the last try
      (two malformed turns already, so a look's final try is as without the switch), after a call written after
      reasoning was cut at max_tokens (the reasoning may leave the call only SUBMIT_ROOM, less than the plain call has),
      or when the reasoning sample is empty, holds a special token or "<|", or raises (a note in the trace says which;
      its tokens are counted in prefill and sampled and in the step's entry's "tokens", not in the note; a client
      sample_once renewed to is kept);
    - a malformed call counts as one malformed turn, as before; the next try reasons again, from the same prompt, unless
      the call was cut at max_tokens;
    - the model's decision, an empty "place" too, is taken as it comes; nothing is re-asked for what it says.
    Free turns are the same in every mode.
    final_line (None: the module's FINAL_LINE, False until it is turned on): the occurrence of a limit line that opens the
    submit states that no more tool calls are run in this look (check_orders included, when the look has that tool) and
    that the next message is the submit call, which ends the look. That is the second research call past the research
    limit, or a second check past the check limit once the model has been told to submit. The earlier occurrences, the
    "context nearly full" line and a dropped turn are unchanged. It is independent of forced_mode.
    The missing-price re-ask (Clayton, 16 Nov: "Yes, build it"): a submit, free or opened, holding an order whose type needs
    a price field it lacks (tools.missing_prices: LMT and LOC "limit", STP "stop", STP LMT both, TRAIL "trail_percent") is
    answered once a look with tools.missing_text, the trace entry "reask": "submit" (no "tool", so no research and no
    decision), and the next turn is as that one was (free, or the opened submit). The next submit is taken as written. It is
    skipped, and the submit taken as written, when the answer would leave less than SUBMIT_ROOM after the submit's prefix.
    The held-back submit is the look's decision if no later one comes (a third malformed turn, a failed sample, context
    full, the step range): a trace entry "tool": "submit", "fallback": "reask" says so (the review of 16 Nov evening).
    A submit with every price field present is taken at once, as before. It applies to both texts (a malformed call).
    max_calls (None: tools.MAX_CALLS, 12): the look's research-call limit, the sleeve's own (trader.max_calls_of; Clayton,
    15 Nov: 15 on the gpt-oss sleeves from Mon 16 Nov, the text-A control kept at 12). It is the limit the count stops at,
    the counter's "(research call k of N)", the "research limit reached (N calls)" line and the step range; nothing else
    changes, and None or 12 is the look as it was. Anything but a whole number from 1 to tools.MAX_CALLS_TOP raises
    before anything is sampled.
    context (None: CONTEXT, 32,768): the look's context, for a sleeve sampled on a longer-context variant of the base
    model (live.context_of: the sleeve's "base_model" and "context", 20 Nov 2026). Every room and reserve figure is
    counted against it; RESERVE, SUBMIT_ROOM, MIN_SUBMIT and MAX_OUT are unchanged, and None is the look as it was."""
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    mode = forced_mode if forced_mode is not None else (
        FORCED_MODE if forced_reasoning is None else "reasoning" if forced_reasoning else "plain")
    if mode not in FORCED_MODES:
        raise ValueError(f"forced_mode {mode!r}: one of {FORCED_MODES}")
    limit = tools.MAX_CALLS if max_calls is None else max_calls
    CTX = CONTEXT if context is None else context
    if not (isinstance(CTX, int) and not isinstance(CTX, bool) and MIN_ROOM < CTX <= 1048576):
        raise ValueError(f"context {context!r}: a whole number of tokens")
    if not (isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= tools.MAX_CALLS_TOP):
        raise ValueError(f"max_calls {max_calls!r}: a whole number from 1 to {tools.MAX_CALLS_TOP}")
    reason_first, notes_first = mode == "reasoning", mode == "notes"
    final = FINAL_LINE if final_line is None else bool(final_line)
    held = lock if lock is not None else contextlib.nullcontext()
    r = GptOssRenderer(trader.tokenizer(), use_system_prompt=True, reasoning_effort=effort, current_date=current_date)
    specs = research_specs + ([checks.CHECK_SPEC] if checker else []) + [submit_tool or trader.TOOL_LIVE]
    names = {s["name"] for s in specs}
    msgs = r.create_conversation_prefix_with_tools(harmony_specs(specs), system)
    msgs.append({"role": "user", "content": state_text})
    prompt = list(r.build_generation_prompt(msgs).to_ints())
    trace, calls, sampled, prefill, bad, force, told, checked, tries, past = [], 0, 0, 0, 0, [], False, 0, 0, 0
    cut_call = False                      # a call after reasoning, or begun at its notes, was cut at max_tokens: plain from then on
    reasked = None                        # the submit the one re-ask a look has held back (17 Nov): the decision if no later one comes
    def fallback(step, why):                  # no later decision after a re-ask: the held-back submit, as written (the review, 16 Nov)
        trace.append({"step": step, "tool": "submit", "fallback": "reask", "note": f"{why}: the submit re-asked earlier is taken as written"})
        return reasked, trace, prefill, sampled
    # The step range (limit: 12 unless the sleeve has its own max_calls; the arithmetic below is for 12, and only its
    # research calls change with the limit). Without a checker it is fd95e2a's: 12 research calls, one told the limit,
    # the next one forced, and room for malformed turns. A checker adds at most MAX_TRIES checks run and one answered with the limit; any further call past
    # a limit is told to submit (once, shared with research) and the next opens it: MAX_TRIES + 1 more steps, so the range
    # never ends before the forced submit (review, 15 Nov; check_orders_test drives it with every mix). The context reserve
    # adds no step: opening the submit under RESERVE happens inside a step; a call not run and a dropped turn each take
    # the one step that sets the force, in place of the call that step would have been, and each can happen once, since
    # every turn after the force is the submit. Without a checker (fd95e2a's arithmetic): 12 research calls, one told, one
    # malformed, then the step that sets the force, then at most 2 forced tries = 17; with one: 4 checks run and 1
    # answered with the limit more = 22. The missing-price re-ask (17 Nov) can take one more step, once a look: 18 and 23.
    for step in range(limit + 6 + (checks.MAX_TRIES + 1 if checker else 0)):
        room = CTX - len(prompt) - len(force)
        if not force and room < RESERVE:  # too little room for a free turn and then a decision: open the submit here
            trace.append({"step": step, "room": room, "note": "context nearly full: submit opened"})
            force = SUBMIT_OPEN()
            room = CTX - len(prompt) - len(force)
        if room < MIN_SUBMIT:             # not even the opened submit fits (fd95e2a: under MIN_ROOM, opened or not)
            trace.append({"step": step, "error": f"context full ({len(prompt)} tokens)"})
            return fallback(step, "context full") if reasked is not None else (None, trace, prefill, sampled)
        free = not force                  # a turn the model writes from its own reasoning, not into the opened submit
        head, reasoned, spent = force, None, 0   # what the sample continues: the opened submit's prefix, or reasoning and
        aside = len(ANALYSIS_OPEN()) + len(REASON_GLUE()) + len(force) + SUBMIT_ROOM   # then it; the call's share of the room
        if force and reason_first and bad < 2 and not cut_call and CTX - len(prompt) - aside >= MIN_REASON:
            # the reasoning phase (switch on): not on the last try, not after a call it left too little room (cut_call),
            # and only with MIN_REASON of room after the call's share; steps run under 50, so seed * 100 + 50 + step is no
            # other request's seed in the look. Stops: <|end|>, and <|return|> and <|call|> too, since a stop list given
            # replaces the sampler's own end-of-turn stops (tinker.SamplingParams), so a stray one ends the sample at once
            # (it is then discarded: a special token) rather than running on to max_tokens (review, 15 Nov).
            p1 = tinker.SamplingParams(temperature=1.0, max_tokens=min(MAX_OUT, CTX - len(prompt) - aside),
                                       stop=[END_TOKEN(), trader.RET, trader.CALL],
                                       **({"seed": seed * 100 + 50 + step} if seed is not None else {}))
            renewed = []                  # a client sample_once renewed to, kept even if its sample then raised
            renew1 = (lambda: renewed.append(renew()) or renewed[-1]) if renew is not None else None
            try:
                r1, sc = sample_once(sc, renew1, tinker.ModelInput.from_ints(prompt + ANALYSIS_OPEN()), p1, trace, step)
            except Exception as e:        # noqa: BLE001 - this try is then the plain opened submit
                sc = renewed[-1] if renewed else sc
                trace.append({"step": step, "note": f"reasoning phase discarded (the sample failed: {type(e).__name__}: "
                                                    f"{str(e)[:200]}): submit opened plain"})
            else:
                t1 = list(r1.sequences[0].tokens)
                prefill += len(prompt) + len(ANALYSIS_OPEN())
                sampled += len(t1)
                spent = len(t1)
                body, cut, why = _reasoning(t1)
                if why:                   # its tokens go in the step's entry below ("tokens", the step's total), not here
                    trace.append({"step": step, "discarded_tokens": len(t1), "said": trader.tokenizer().decode(ANALYSIS_OPEN() + t1),
                                  "note": f"reasoning phase discarded ({why}): submit opened plain"})
                else:
                    head, reasoned = ANALYSIS_OPEN() + body + REASON_GLUE() + force, {"tokens": len(t1), "cut": cut}
            room = CTX - len(prompt) - len(head)
        opened = None
        if force and notes_first and bad < 2 and not cut_call:    # "notes": the call begun at its notes, one sample as plain;
            head, opened = force + NOTES_OPEN(), "notes"          # not on the last try, nor after such a call was cut
            room = CTX - len(prompt) - len(head)
        params = tinker.SamplingParams(temperature=1.0, max_tokens=min(MAX_OUT, room), stop=[trader.RET, trader.CALL],
                                       **({"seed": seed * 100 + step} if seed is not None else {}))
        try:
            res, sc = sample_once(sc, renew, tinker.ModelInput.from_ints(prompt + head), params, trace, step)
        except Exception as e:            # noqa: BLE001 - after a re-ask a failed sample costs the correction, not the decision
            if reasked is None:
                raise
            trace.append({"step": step, "error": f"sample failed after the re-ask: {type(e).__name__}: {str(e)[:200]}"})
            return fallback(step, "the sample failed")
        seq = res.sequences[0]
        toks = head + list(seq.tokens)
        prefill += len(prompt) + len(head)
        sampled += len(seq.tokens)
        kind, name, val = parse_any(toks, seq.stop_reason, names)
        said = trader.tokenizer().decode(toks)            # the reasoning and call, in full: week 3 is the next replay set
        extra = {"reasoned": reasoned} if reasoned else {"opened": opened} if opened else {}
        n_toks = len(seq.tokens) + spent                  # the step's sampled tokens, both phases (a discarded one too)
        if (reasoned or opened) and len(seq.tokens) >= params.max_tokens and seq.tokens[-1] not in (trader.RET, trader.CALL):
            cut_call = True               # a call after reasoning cut at max_tokens (review, 15 Nov): its JSON may need
                                          # more than the reasoning left it, so every later try is the plain opened submit
        if kind != "TOOL":
            bad += 1
            trace.append({"step": step, "error": f"{kind}: {str(val)[:200]}", "tokens": n_toks, "said": said, **extra})
            if bad >= 3:
                return fallback(step, "three malformed turns") if reasked is not None else (None, trace, prefill, sampled)
            if bad == 2:                           # a second try with no decision in it: open the submit call; the model writes its orders
                force = SUBMIT_OPEN()
            continue                               # re-ask at the same point, with a reasoning turn (as the daily loop re-asks malformed output)
        if name == "submit":
            val.setdefault("cancel", [])
            lack = tools.missing_prices(val.get("place"))
            if lack and reasked is None:      # Clayton, 16 Nov: "Yes, build it": an order of a type that needs a price field
                out = tools.missing_text(lack)                    # without it is asked about once, naming the field
                res_toks = tool_result_tokens(name, out)
                after = CTX - len(prompt) - len(toks) - len(res_toks)
                if after - len(SUBMIT_OPEN()) >= SUBMIT_ROOM:     # only with room for the decision after it; else the
                    reasked = val                                 # submit is taken as written, as before; held as the fallback
                    trace.append({"step": step, "reask": "submit", "args": val, "result": out, "missing": [
                        {"order": i, "fields": f} for i, _, f, _ in lack], "tokens": n_toks, "said": said, **extra})
                    prompt = prompt + toks[:-1] + [trader.CALL] + res_toks   # "reask", not "tool": no research, no decision
                    continue                                      # the next turn as this one was: free, or the opened submit
            trace.append({"step": step, "tool": "submit", "tokens": n_toks, "said": said, **extra})
            return val, trace, prefill, sampled
        left = CTX - len(prompt) - len(toks)          # what the context has left after this turn
        if free and left < SUBMIT_ROOM:   # a tool-call turn that leaves too little for the decision (16 Nov): dropped, not
            trace.append({"step": step, "room": room, "left": left, "dropped": name, "args": val, "tokens": len(seq.tokens),
                          "said": said, "note": "turn left too little room for the decision: dropped, submit opened"})
            force = SUBMIT_OPEN()         # run or counted; the submit is opened at the prompt as it was before the turn
            continue                      # "dropped", not "tool": the call is no part of the look (live.research_summary, replays)
        runs = (name == "check_orders" and not (checked >= checks.MAX_CHECKS or tries + 1 > checks.MAX_TRIES)) or (
            name != "check_orders" and calls < limit)                        # a call that would run a tool
        if runs and left - _estimate(name, val, checker) < RESERVE:   # its result would leave no room for the decision: not run,
            out = (f"context nearly full ({left} tokens left); no "     # not counted, and the submit is opened next
                   + ("check is run" if name == "check_orders" else "research call is run"))
            if final:                     # the same fact as the limit lines' final words (the Researcher, 17 Nov: base-ml
                out += "; " + final_words(bool(checker))   # Mon 14:00 and Tue 11:30 read it as "try again" and sent an
            force = SUBMIT_OPEN()                           # empty or a "test" decision); with the switch on only
            trace.append({"step": step, "tool": name, "args": val, "result": out, "tokens": len(seq.tokens), "said": said})
            prompt = prompt + toks[:-1] + [trader.CALL] + tool_result_tokens(name, out)   # the framework's line: whole
            continue
        if name == "check_orders":        # outside the research count: `calls` is not touched
            raw, own = None, False        # own: the checker's result (cut to fit), else the framework's line (whole)
            tries += 1                    # every call, served or not
            if checked >= checks.MAX_CHECKS or tries > checks.MAX_TRIES:
                out, past = checks.cap_text(checked), past + 1
                if past > 1:              # a further call past the limit is a call past a limit, as for research: the first
                    out = out.rstrip(".") + "; call submit now"   # is told to submit, the next opens the submit call
                    if told:
                        force = SUBMIT_OPEN()
                        if final:         # the line that opens the submit says what is true from here on
                            out = checks.cap_text(checked).rstrip(".") + "; " + final_words(False)
                    told = True
            else:
                own = True
                try:
                    out, counted, raw = checker(val)
                except Exception as e:    # noqa: BLE001 - checks.check never raises; this is the last net
                    out, counted = f"error: the order check failed ({type(e).__name__}: {str(e)[:200]}); this check did not count.", False
                counter = None            # an error or an uncounted check has no counter
                if counted:
                    checked += 1
                    counter = f"(order check {checked} of {checks.MAX_CHECKS})"
            res_toks, out = _fit_text(name, out, left, counter) if own else (tool_result_tokens(name, out), out)
            trace.append({"step": step, "tool": name, "args": val, "result": out, "ibkr": raw, "tokens": len(seq.tokens), "said": said})
            prompt = prompt + toks[:-1] + [trader.CALL] + res_toks
            continue
        if calls >= limit:                # one free turn to reason and submit, as the Claude sleeves have (2 Nov: opening the
            out = f"research limit reached ({limit} calls): call submit now"   # submit at once left no reasoning turn);
            if told:                      # another call after that: open the submit call, as for prose
                force = SUBMIT_OPEN()
                if final:                 # the line that opens the submit says what is true from here on
                    out = f"research limit reached ({limit} calls): " + final_words(bool(checker))
            told = True
            res_toks = tool_result_tokens(name, out)             # the framework's line: whole
        else:                             # the count: week 2, 135 of 276 gpt-oss looks reached the cap, never told it
            calls += 1
            with held:
                out = tools.run(gw, name, val, ml_tool=ml_tool, sleeve_value=sleeve_value, look=look)
            res_toks, out = _fit_text(name, out, left, f"(research call {calls} of {limit})")   # cut to fit, count kept
        trace.append({"step": step, "tool": name, "args": val, "result": out, "tokens": len(seq.tokens), "said": said})
        prompt = prompt + toks[:-1] + [trader.CALL] + res_toks   # a call that ended in <|return|> reads as a call from here on
    trace.append({"error": "no submit within the step limit"})
    return fallback(step, "no submit within the step limit") if reasked is not None else (None, trace, prefill, sampled)


# ---------------------------------------------------------------- Claude (CLI + MCP)
def claude_exe():
    """The newest Claude CLI bundled with the desktop app. The app is a packaged Windows app, so its
    %APPDATA% is virtualized: processes inside the package see it under %APPDATA%, others only under
    %LOCALAPPDATA%\\Packages\\Claude_*\\LocalCache\\Roaming. Since the app update of 20 Oct 2026 it lives
    in %LOCALAPPDATA%\\Claude-Data\\claude_*\\claude-code."""
    local = Path(os.environ["LOCALAPPDATA"])
    bases = [Path(os.environ["APPDATA"]) / "Claude" / "claude-code"]
    bases += [p / "LocalCache" / "Roaming" / "Claude" / "claude-code" for p in (local / "Packages").glob("Claude_*")]
    bases += [p / "claude-code" for p in (local / "Claude-Data").glob("claude_*")]
    exes = sorted((e for b in bases for e in b.glob("*/*/claude.exe")), key=lambda p: p.stat().st_mtime)
    if not exes:
        raise FileNotFoundError("Claude CLI not found (desktop app bundle)")
    return str(exes[-1])


def _num_env(x):
    """A number for the MCP server's environment, "" if there is none (the forecast tool then falls back as before)."""
    try:
        return "" if x is None else repr(float(x))
    except (TypeError, ValueError):
        return ""


def claude_look(model, system, state_text, research_specs, workdir, effort, timeout=CLAUDE_TIMEOUT, client="trader-mcp-a",
                ml_tool=None, max_gross=2.0, sleeve_value=None, text="B", check_ctx=None, max_calls=None):
    """Run one look with the Claude CLI at reasoning effort low|medium|high|xhigh|max. The MCP server researches on
    the research client named by `client` (trader-mcp-a to -d = 12 to 15), so the caller may hold client 10. ml_tool: the
    sleeve's forecast tool version (ml_tools.py). max_gross: the sleeve's gross-exposure term (trader.max_gross_of), which
    the submit tool's reply names. sleeve_value: the sleeve's value in its state, the forecast tool's sleeve% base. text:
    the sleeve's text (trader.text_of), which picks the submit tool's description. check_ctx: text B's check_orders
    (checks.context(); the MCP process cannot see the sleeve's state, so it reads this from check.json, and runs its what-ifs
    on its own research client); None offers no such tool. max_calls: the sleeve's research-call limit (trader.max_calls_of;
    None: tools.MAX_CALLS, 12), named in the system prompt ("at most N calls") and counted to by the MCP server
    (TRADER_MAX_CALLS); anything but a whole number from 1 to tools.MAX_CALLS_TOP raises before the CLI starts. A submit
    made before the CLI times out still counts."""
    limit = tools.MAX_CALLS if max_calls is None else max_calls
    if not (isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= tools.MAX_CALLS_TOP):
        raise ValueError(f"max_calls {max_calls!r}: a whole number from 1 to {tools.MAX_CALLS_TOP}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    submit_file, log_file, check_file = workdir / "submit.json", workdir / "research.jsonl", workdir / "check.json"
    for f in (submit_file, log_file, check_file):
        if f.exists():
            f.unlink()
    if check_ctx is not None:
        check_file.write_text(json.dumps(check_ctx), encoding="utf-8")
    have = {s["name"] for s in research_specs}
    env = dict(os.environ, TRADER_SUBMIT_FILE=str(submit_file), TRADER_LOG_FILE=str(log_file),
               TRADER_HAS_NEWS="1" if "news" in have else "0", TRADER_HAS_SCANNER="1" if "scanner" in have else "0",
               TRADER_HAS_STATS="1" if "stats" in have else "0", TRADER_HAS_FORECAST="1" if "forecast" in have else "0",
               TRADER_MCP_CLIENT=client, TRADER_ML_TOOL=ml_tool or "", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
               TRADER_MAX_GROSS="none" if max_gross is None else f"{max_gross:g}",
               TRADER_SLEEVE_VALUE=_num_env(sleeve_value), TRADER_TEXT=str(text),
               TRADER_HAS_CHECK="1" if check_ctx is not None else "0", TRADER_CHECK_FILE=str(check_file) if check_ctx is not None else "",
               TRADER_MAX_CALLS=str(limit),
               TRADER_CLOSE_ONLY_V2="1" if tools.co.V2 else "0",   # the loop's own rule, for the tool server's words

               # the CLI cuts every MCP tool description to 4,096 characters by default; the forecast tool's runs to
               # about 13,000 (ML3), so Claude ML looks saw only its first 4,096 from week 2 to week 4 (review, 9 Nov)
               CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH="20000")
    cfg = {"mcpServers": {"trader": {"command": sys.executable,
                                     "args": [str(ROOT / "mcp_server.py")], "env": {k: env[k] for k in
                                     ("TRADER_SUBMIT_FILE", "TRADER_LOG_FILE", "TRADER_HAS_NEWS", "TRADER_HAS_SCANNER",
                                      "TRADER_HAS_STATS", "TRADER_HAS_FORECAST", "TRADER_MCP_CLIENT", "TRADER_ML_TOOL",
                                      "TRADER_MAX_GROSS", "TRADER_SLEEVE_VALUE", "TRADER_TEXT", "TRADER_HAS_CHECK",
                                      "TRADER_CHECK_FILE", "TRADER_MAX_CALLS", "TRADER_CLOSE_ONLY_V2")}}}}
    cfg_file = workdir / "mcp.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    allowed = ",".join(f"mcp__trader__{n}" for n in sorted(have | {"submit"} | ({"check_orders"} if check_ctx is not None else set())))
    instructions = (system + "\n\nUse the tools named " + ", ".join(sorted(have)) + f" for research (at most {limit} calls), "
                    "then call submit exactly once. Your decision is what you pass to submit; any other text is ignored."
                    + (" " + checks.CLAUDE_LINE if check_ctx is not None else ""))   # the checker's sentence (15 Nov), text B only
    # The look sees only its system prompt, state and tools: --setting-sources "" keeps out the user's and the repo's
    # CLAUDE.md, and the env var above the session memory index, which the CLI otherwise loads from the workdir (25 Oct).
    exe = claude_exe()
    cmd = [exe, "-p", state_text, "--model", model, "--effort", effort, "--system-prompt", instructions, "--tools", "",
           "--mcp-config", str(cfg_file), "--strict-mcp-config", "--allowedTools", allowed, "--setting-sources", "",
           "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                           timeout=timeout, cwd=str(workdir))
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        txt = lambda x: x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
        rc, stdout, stderr = f"timeout after {timeout:.0f}s", txt(e.stdout), txt(e.stderr)
    trace = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()] if log_file.exists() else []
    events = []
    for l in stdout.splitlines():
        try:
            events.append(json.loads(l))
        except ValueError:
            pass
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    out = next((e for e in reversed(events) if e.get("type") == "result"), None) or {"raw": stdout[-2000:]}
    blocks = [b for e in events if e.get("type") == "assistant" for b in (e.get("message") or {}).get("content") or []]
    # cli_result stays the final message (week_read's frozen M3 input); cli_text and cli_thinking keep every message, so
    # the opus looks can be read like gpt-oss's "said"; the CLI version and served model record what ran (25 Oct probe).
    trace.append({"cli_returncode": rc, "cli_result": str(out.get("result", out.get("raw", "")))[:2000],
                  "stderr": stderr[-1000:], "cost_usd": out.get("total_cost_usd"), "turns": out.get("num_turns"),
                  "usage": out.get("usage"),
                  "cli_version": Path(exe).parents[1].name, "cli_model": init.get("model"),
                  "cli_mcp": init.get("mcp_servers"),
                  "cli_args": ["<state>" if a == state_text else "<system>" if a == instructions else a for a in cmd[1:]],
                  "cli_text": [b.get("text") for b in blocks if b.get("type") == "text"],
                  "cli_thinking": [b["thinking"] for b in blocks if b.get("type") == "thinking" and b.get("thinking")],
                  # the plan's usage windows as the CLI reports them (the 7-day window was at 83% on 29 Oct)
                  "cli_rate_limit": [e for e in events if e.get("type") == "rate_limit_event"][-3:]})
    call = json.loads(submit_file.read_text(encoding="utf-8")) if submit_file.exists() else None
    return call, trace
