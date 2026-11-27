"""One look by Inkling-2-Small (Thinking Machines) through Tinker's OpenAI-compatible endpoint (20 Nov 2026).

Why this road: the native road (tinker_cookbook's tml_v0 renderer and the Inkling tokenizer) needs the tml_renderers
package, which ships wheels for Linux and macOS only (the Tinker Owner, 20 Nov: no supported way on Windows); this
machine is Windows with no WSL, and PyPI's 0.0.1 in the venv is an empty placeholder. The endpoint renders on the server,
so the look runs at the level of messages, not tokens.
The Trading Model Researcher's decisions (20 Nov): route A now; Clayton's model thinkingmachines/Inkling-2-Small (relayed
09:20 ET); the text-B words as base-90k-b and base-90k-ml read them; effort 0.99; the loop's context 131,072 for these
sleeves (a sleeve's "context" key; nothing is added to the prompt, so only the reserve moves); base weights.

The look follows agent.gptoss_look's rules, written for messages:
- the same tools, descriptions, research-call limit (max_calls), check_orders limits, limit lines and final words;
- the opened submit becomes tool_choice naming submit (TMLv0 cannot continue a partly written assistant message). The
  server cannot make the model obey; a reply that breaks tool_choice comes back HTTP 422, which is one malformed turn, as
  is a forced reply that calls another tool. A forced turn runs at the sleeve's effort except where gpt-oss writes the
  plain opened submit: on the last try (two malformed turns already), after a forced reply was cut at max_tokens, and
  with less room than SUBMIT_ROOM + MIN_REASON (the review of 20 Nov: at effort 0.99 a forced turn near the end of the
  context could reason to its cap three times and lose the look). There it runs at FORCED_PLAIN_EFFORT (0.0, "none").
  Each forced turn's entry records "forced": true, "effort" and "reasoned" (whether reasoning_content came first);
- a malformed turn (cut at max_tokens, no tool call and no bare submit JSON, a call to an undeclared tool, bad JSON,
  submit arguments that fail agent.is_submit, two submit calls in one reply, a 422, a 400 naming the context length) is
  re-asked at the same messages; the second opens the submit and the third ends the look, as for gpt-oss;
- the missing-price re-ask, once a look: answered on the submit's call id, or, for a decision written as text, in a
  user message with the same words;
- the context reserve keeps agent's numbers (RESERVE, SUBMIT_ROOM, MIN_SUBMIT, MAX_OUT). The room is counted from each
  reply's usage.prompt_tokens plus an estimate of what the loop appends (the assistant turn as usage.completion_tokens,
  each tool result as o200k tokens x SLACK, FRAME tokens a message: Inkling-2-Small's tokenizer is h200k, not o200k), and
  MARGIN tokens are kept below the context so that an estimate a little short never asks for more than fits;
- several tool calls in one reply are run in order, each counted as one call under the same limits. A limit line can
  open the submit only if the model had read "call submit now" in an earlier reply (gpt-oss's free turn after the
  warning); once a line in a reply opens the submit, the later calls in that reply are not run and get that line;
  a submit among other calls is the decision and the calls beside it are not run (the entry names them).
The look record keeps cost.prefill_tokens and cost.sampled_tokens from usage, and each reply's trace entry its
response_id, prompt_tokens, cached_tokens, completion tokens ("tokens") and the UTC time it came back ("utc"): every
endpoint request bills to one shared session with no project (the Tinker Owner, 20-21 Nov), so these entries are the only
per-sleeve cost record, priced at the rate card, and the way to check billing by the hour. An attempt that brought no
reply (a 422, a retried or failed request) is traced with its UTC time too. "said" holds reasoning_content (where
gpt-oss's decoded turn goes), "content" the reply's text and "calls" its raw tool calls; the last trace entry carries
"inkling": the 422s, the forced turns, how many ran at FORCED_PLAIN_EFFORT ("forced_plain"), how many reasoned first, the cached prompt tokens, the model, effort and
context. A look that raises carries its trace and token counts on the exception (inkling_trace, inkling_cost); the
trace's last entry is then "look ended: ..." with its UTC time and "inkling", for the attempt that ended it.
"""

import contextlib
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request

import agent
import checks
import tools
import trader

URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
MODEL = "thinkingmachines/Inkling-2-Small"
CONTEXT = 131072           # the Researcher's setting for the Inkling-2-Small sleeves (20 Nov); a sleeve's "context" key
CONTEXT_RANGE = (16384, 1048576)
EFFORT = 0.99              # the endpoint's highest ("xhigh"/"max"); Clayton's standing wish: every sleeve at max effort
FORCED_PLAIN_EFFORT = 0.0  # a forced turn where gpt-oss writes its plain opened submit (see above)
FRAME = 32                 # estimated tokens of framing a message the loop appends (role, ids), corrected by the next reply
SLACK = 1.25               # o200k counts of appended text scaled up: Inkling-2-Small's h200k tokenizer differs (Tinker Owner)
MARGIN = 1024              # kept between the estimated prompt plus max_completion_tokens and the context
FIRST_PAD = 600            # estimated framing of the first request: the effort system message, the tool declaration's form
HEADERS = {}               # project routing: what the Tinker Owner says to send; empty sends nothing more (20 Nov: the
                           # endpoint documents no project field and ignores unknown fields)
TIMEOUT = agent.SAMPLE_TIMEOUT
EFFORT_NAMES = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}


def is_inkling(model):
    return str(model or "").split(":", 1)[0].startswith("thinkingmachines/Inkling")


def effort_value(e):
    """The sleeve's effort as the endpoint takes it: a number from 0.0 to 0.99, or one of its names (mapped to that number)."""
    if isinstance(e, str) and e.strip().lower() in EFFORT_NAMES:
        return EFFORT_NAMES[e.strip().lower()]
    if isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) and 0.0 <= e <= 0.99:
        return float(e)
    raise ValueError(f"effort {e!r}: a number from 0.0 to 0.99 or one of {sorted(EFFORT_NAMES)}")


def context_of(meta):
    """A sleeve's context for the loop's reserve: sleeves.json "context", a whole number in CONTEXT_RANGE, else CONTEXT."""
    v = (meta or {}).get("context")
    return v if isinstance(v, int) and not isinstance(v, bool) and CONTEXT_RANGE[0] <= v <= CONTEXT_RANGE[1] else CONTEXT


class HttpError(Exception):
    def __init__(self, status, text, retry_after=None):
        super().__init__(f"HTTP {status}: {text[:300]}")
        self.status, self.text, self.retry_after = status, text, retry_after


class BadBody(Exception):
    """A 200 whose body is not JSON (a proxy's page, a cut transfer)."""


def post(body, timeout=TIMEOUT):
    """One chat-completions request; the reply's JSON, or HttpError / BadBody. The key is read from the environment and
    never logged."""
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise RuntimeError("TINKER_API_KEY is not set")
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": "python-client/1.0", **HEADERS})       # a bare urllib agent is refused by the CDN (403, code 1010)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        ra = e.headers.get("Retry-After") if e.headers else None
        raise HttpError(e.code, e.read().decode(errors="replace"), ra) from None
    try:
        return json.loads(raw)
    except ValueError:
        raise BadBody(f"reply is not JSON: {raw[:200]!r}") from None


RETRY = (408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)
WAITS = (2, 8)             # seconds before the second and third attempt (a Retry-After up to 30 s is honoured instead)


def utc():
    """The UTC time to the second, as billing's hours read it (2026-11-23T14:30:05Z)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def request(send, body, trace, step, stats=None):
    """send(body), retried on a 408/409/429/5xx (CDN 52x too), a body that is not JSON, a cut transfer or a dropped
    connection: three attempts at most, and at most one retry after a timeout (agent.sample_once's rule: a call that
    hangs is retried once). Other errors (a 422 is a malformed turn; a 400 a bad request) are raised at once."""
    timeouts = 0
    stats = stats if stats is not None else {}
    for k in range(len(WAITS) + 1):
        stats["attempts"] = stats.get("attempts", 0) + 1
        try:
            return send(body)
        except Exception:                 # every attempt that brought no reply (retried or raised): its cost may be
            stats["lost_attempts"] = stats.get("lost_attempts", 0) + 1   # billed and not counted in usage
            try:
                raise
            except HttpError as e:
                if e.status not in RETRY or k == len(WAITS):
                    raise
                why, wait = f"HTTP {e.status}", WAITS[k]
                try:
                    wait = min(30.0, max(wait, float(e.retry_after))) if e.retry_after else wait
                except ValueError:
                    pass
            except (TimeoutError, urllib.error.URLError) as e:
                if isinstance(e, urllib.error.URLError) and not isinstance(getattr(e, "reason", None), TimeoutError):
                    if k == len(WAITS):       # a dropped or refused connection: retried as below
                        raise
                    trace.append({"step": step, "error": f"request failed (URLError: {str(e)[:120]}); retried", "utc": utc()})
                    time.sleep(WAITS[k])
                    continue
                timeouts += 1                 # a timeout, or a connect timeout urllib wraps in URLError (review, 20 Nov)
                if timeouts > 1 or k == len(WAITS):
                    raise
                why, wait = f"timed out after {TIMEOUT} s", 0
            except (BadBody, http.client.IncompleteRead, ConnectionError, OSError) as e:
                if k == len(WAITS):
                    raise
                why, wait = f"{type(e).__name__}: {str(e)[:120]}", WAITS[k]
            trace.append({"step": step, "error": f"request failed ({why}); retried", "utc": utc()})
            time.sleep(wait)


def ntok(text):
    """o200k tokens of a plain text (trader.tokenizer(), gpt-oss's o200k_harmony, loaded once by the loop)."""
    return len(trader.tokenizer().encode(text or "", add_special_tokens=False))


def est(text):
    """The estimated tokens a message of this text adds: o200k x SLACK, plus FRAME."""
    return math.ceil(ntok(text) * SLACK) + FRAME


DESCRIPTIVE = ("description", "title", "examples", "default")


def _bare(schema):
    """A schema below a tool's top-level properties as gpt-oss's renderer shows it: types, enums, items and property names,
    with no description, title, examples or default (tinker_cookbook.renderers.gpt_oss writes only those of the tool, its
    parameters and its top-level properties; the Researcher's read of 20 Nov: Inkling would otherwise read 388 more tokens,
    the order fields' words inside place[] that gpt-oss never sees)."""
    if isinstance(schema, list):
        return [_bare(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    return {k: ({n: _bare(x) for n, x in v.items()} if k == "properties" and isinstance(v, dict) else _bare(v))
            for k, v in schema.items() if k not in DESCRIPTIVE}


def model_parameters(params):
    """A tool's parameters with the words gpt-oss reads: its own description and each top-level property's descriptive
    keys; everything nested bare (_bare)."""
    p = dict(params or {"type": "object", "properties": {}})
    props = p.get("properties")
    if isinstance(props, dict):
        p["properties"] = {n: ({k: (v if k in DESCRIPTIVE else
                                    {m: _bare(y) for m, y in v.items()} if k == "properties" and isinstance(v, dict) else _bare(v))
                                for k, v in x.items()} if isinstance(x, dict) else x)
                           for n, x in props.items()}
    return p


def oai_tools(specs):
    return [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""),
                                              "parameters": model_parameters(s.get("parameters"))}}
            for s in specs]


def harmony_words(spec):
    """The descriptive words gpt-oss's renderer writes for a tool spec, in order: its description, its parameters'
    description, and each top-level property's title, description, examples and default."""
    out = [spec.get("description")]
    params = spec.get("parameters") or {}
    out.append(params.get("description"))
    for x in (params.get("properties") or {}).values():
        if isinstance(x, dict):
            out += [x.get(k) for k in DESCRIPTIVE]
    return [json.dumps(w, sort_keys=True) for w in out if w not in (None, "", [])]


def json_words(tool):
    """Every descriptive value in an OpenAI tool's JSON, at any depth (property names excluded), in document order."""
    out = []

    def walk(x, in_props=False):
        if isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for k, v in x.items():
                if in_props:
                    walk(v)
                elif k in DESCRIPTIVE and v not in (None, "", []):
                    out.append(json.dumps(v, sort_keys=True))
                else:
                    walk(v, in_props=k == "properties")
    fn = tool["function"]
    if fn.get("description"):
        out.append(json.dumps(fn["description"], sort_keys=True))
    walk(fn.get("parameters") or {})
    return out


def words_sha(specs):
    """A hash of the tool words a model reads (the specs in the form both renderers show), for the look record."""
    import hashlib
    return hashlib.sha256(json.dumps(oai_tools(specs), sort_keys=True).encode()).hexdigest()[:16]


SUBMIT_CHOICE = {"type": "function", "function": {"name": "submit"}}
CONTEXT_WORDS = ("context length", "maximum context", "context window", "too many tokens", "prompt is too long", "token")


def parse_reply(finish, content, calls, names):
    """('TOOL', [(index in calls or None, call id or None, name, args), ...]) | ('MALFORMED', reason): agent.parse_any's
    rules for a reply. One well-formed submit is the decision whatever its siblings are (they are not run; review, 20 Nov);
    two calls naming submit are malformed (gpt-oss: a turn with two calls is; agent.submit_in: two decisions cannot be
    told apart); otherwise any malformed call makes the reply malformed."""
    if finish == "length":
        return "MALFORMED", "output cut at max_tokens"
    if calls:
        out, bad, subs = [], None, 0
        for k, c in enumerate(calls):
            fn = c.get("function") or {}
            name = str(fn.get("name") or "").removeprefix("functions.")
            subs += name == "submit"
            if name not in names:
                bad = bad or f"call to undeclared tool {fn.get('name')!r}"
                continue
            raw = fn.get("arguments")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            except (ValueError, RecursionError) as e:
                bad = bad or f"bad JSON: {e}"
                continue
            args = agent.unwrap_args(name, args)
            if name == "submit" and not agent.is_submit(args):
                bad = bad or "submit arguments malformed"
                continue
            out.append((k, c.get("id"), name, args))
        if subs > 1:
            return "MALFORMED", "two submit calls in one reply"
        if any(g[2] == "submit" for g in out):
            return "TOOL", out
        return ("MALFORMED", bad) if bad else ("TOOL", out)
    decision = agent.submit_in(content or "") if "submit" in names else None
    return ("TOOL", [(None, None, "submit", decision)]) if decision else ("MALFORMED", "output ended without a tool call")


def sent_calls(raw_calls, step):
    """The reply's tool calls as the history re-sends them: ids missing or repeated replaced by the loop's own
    (call_<step>_<k>), arguments as a JSON string, as the API takes them."""
    ids = [c.get("id") for c in raw_calls]
    if None in ids or len(set(ids)) != len(ids):
        ids = [f"call_{step}_{k}" for k in range(len(raw_calls))]
    out = []
    for c, i in zip(raw_calls, ids):
        a = (c.get("function") or {}).get("arguments")
        out.append(dict(c, id=i, function=dict(c.get("function") or {}, arguments=a if isinstance(a, str) else json.dumps(a))))
    return out


def fit_text(body, left, counter=None):
    """agent._fit_text's rule in estimated tokens: (text, its estimate) leaving SUBMIT_ROOM after it; else the body cut at
    a line (or inside its first line), then agent.CUT_LINE, then the counter; a check's exposure-limit line kept after the
    cut line when it fits."""
    budget = left - agent.SUBMIT_ROOM
    tail = [agent.CUT_LINE] + ([counter] if counter is not None else [])
    whole = body + ("\n" + counter if counter is not None else "")
    if est(whole) <= budget:
        return whole, est(whole)
    lines = body.split("\n")
    if len(lines) > 1 and lines[-1].startswith(agent.OWN_LIMIT_HEADS):
        keep = [agent.CUT_LINE, lines[-1]] + ([counter] if counter is not None else [])
        if est("\n".join(keep)) <= budget:
            tail, lines = keep, lines[:-1]
    fits = lambda kept: est("\n".join(kept + tail)) <= budget
    lo, hi = 0, len(lines) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(lines[:mid]) else (lo, mid - 1)
    kept = lines[:lo]
    if not kept:
        first, a, b = lines[0], 0, len(lines[0])
        while a < b:
            mid = (a + b + 1) // 2
            a, b = (mid, b) if fits([first[:mid]]) else (a, mid - 1)
        kept = [first[:a]] if a else []
    text = "\n".join(kept + tail)
    return text, est(text)


def look(gw, system, state_text, research_specs, effort=EFFORT, seed=None, ml_tool=None, lock=None, submit_tool=None,
         sleeve_value=None, look_key=None, checker=None, final_line=None, max_calls=None, model=MODEL, context=CONTEXT,
         send=None, run_tool=None):
    """(call or None, trace, prefill, sampled), as agent.gptoss_look returns. send: the request function (post, or a fake
    in tests); run_tool: tools.run, or a replay's stub. The arguments mean what they mean for gptoss_look; effort is a
    number to 0.99 or a name; context the sleeve's (context_of). Each reply's entry has "est_prompt" (the loop's estimate
    before the request) and "prompt_tokens" (the server's count), so the estimate can be checked. If the look raises,
    the exception carries inkling_trace and inkling_cost (prefill, sampled) for the record."""
    st = {"trace": [], "prefill": 0, "sampled": 0}
    try:
        return _look(st, gw, system, state_text, research_specs, effort, seed, ml_tool, lock, submit_tool, sleeve_value,
                     look_key, checker, final_line, max_calls, model, context, send or post, run_tool or tools.run)
    except Exception as e:                # the attempt that ended the look gets its UTC time and the look's counts too:
        end = {"error": f"look ended: {type(e).__name__}: {str(e)[:200]}", "utc": utc()}   # it may be billed (review, 21 Nov)
        if "stats" in st:
            end["inkling"] = dict(st["stats"], **st["meta"])
        st["trace"].append(end)
        e.inkling_trace, e.inkling_cost = st["trace"], (st["prefill"], st["sampled"])
        raise


def _look(st, gw, system, state_text, research_specs, effort, seed, ml_tool, lock, submit_tool, sleeve_value, look_key,
          checker, final_line, max_calls, model, context, send, run_tool):
    eff = effort_value(effort)
    if not (isinstance(context, int) and not isinstance(context, bool) and CONTEXT_RANGE[0] <= context <= CONTEXT_RANGE[1]):
        raise ValueError(f"context {context!r}: a whole number from {CONTEXT_RANGE[0]} to {CONTEXT_RANGE[1]}")
    limit = tools.MAX_CALLS if max_calls is None else max_calls
    if not (isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= tools.MAX_CALLS_TOP):
        raise ValueError(f"max_calls {max_calls!r}: a whole number from 1 to {tools.MAX_CALLS_TOP}")
    final = agent.FINAL_LINE if final_line is None else bool(final_line)
    held = lock if lock is not None else contextlib.nullcontext()
    specs = research_specs + ([checks.CHECK_SPEC] if checker else []) + [submit_tool or trader.TOOL_LIVE]
    names = {s["name"] for s in specs}
    tool_list = oai_tools(specs)
    ctx = context - MARGIN                # the room is counted against this
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": state_text}]
    used = est(system) + est(json.dumps(tool_list, separators=(",", ":"))) + est(state_text) + FIRST_PAD
    trace = st["trace"]
    calls, bad, force, told, checked, tries, past = 0, 0, False, False, 0, 0, 0
    cut_forced = False                    # a forced reply was cut at max_tokens: plain forced turns from then on
    reasked = None
    stats = {"http_422": 0, "forced_turns": 0, "forced_plain": 0, "forced_reasoned": 0, "cached_tokens": 0, "requests": 0,
             "attempts": 0, "lost_attempts": 0}
    st["stats"], st["meta"] = stats, {"model": model, "effort": eff, "context": context}   # for a look that raises

    def done(val):
        info = dict(stats, model=model, effort=eff, context=context)
        if trace:                         # on the last entry, so trace[-1] stays the decision (or the reason there is none)
            trace[-1]["inkling"] = info
        else:
            trace.append({"inkling": info})
        return val, trace, st["prefill"], st["sampled"]

    def fallback(step, why):
        trace.append({"step": step, "tool": "submit", "fallback": "reask", "note": f"{why}: the submit re-asked earlier is taken as written"})
        return done(reasked)

    def strike():                         # one malformed turn: (finished, what to return)
        nonlocal bad, force
        bad += 1
        if bad >= 3:
            return True
        if bad == 2:
            force = True
        return False

    for step in range(limit + 6 + (checks.MAX_TRIES + 1 if checker else 0)):
        room = ctx - used
        if not force and room < agent.RESERVE:
            trace.append({"step": step, "room": room, "note": "context nearly full: submit opened"})
            force = True
        if room < agent.MIN_SUBMIT:
            trace.append({"step": step, "error": f"context full ({used} tokens estimated)"})
            return fallback(step, "context full") if reasked is not None else done(None)
        free = not force
        turn_eff = eff
        if force and (bad >= 2 or cut_forced or room < agent.SUBMIT_ROOM + agent.MIN_REASON):
            turn_eff = FORCED_PLAIN_EFFORT
        body = {"model": model, "messages": msgs, "tools": tool_list, "tool_choice": SUBMIT_CHOICE if force else "auto",
                "reasoning_effort": turn_eff, "temperature": 1.0, "max_completion_tokens": min(agent.MAX_OUT, room)}
        if seed is not None:
            body["seed"] = seed * 100 + step
        stats["requests"] += 1
        estimate = used
        fx = {"forced": True, "effort": turn_eff} if force else {}
        if force:
            stats["forced_turns"] += 1
            stats["forced_plain"] += turn_eff == FORCED_PLAIN_EFFORT and eff != FORCED_PLAIN_EFFORT
        try:
            rep = request(send, body, trace, step, stats)
        except HttpError as e:
            overflow = e.status == 400 and any(w in e.text.lower() for w in CONTEXT_WORDS)
            if e.status != 422 and not overflow:
                if reasked is None:
                    raise
                trace.append({"step": step, "error": f"request failed after the re-ask: {e}", "utc": utc()})
                return fallback(step, "the request failed")
            if e.status == 422:           # the reply broke tool_choice (or parallel_tool_calls): one malformed turn
                stats["http_422"] += 1
            else:                         # the request would not fit the context: the estimate was short, so keep more back,
                import re as _re          # leaving room for a try (review, 20 Nov); a limit the server names is used as is
                lim = _re.search(r"(?:maximum context length|context length|context window)\D{0,40}?(\d{4,7})", e.text, _re.I)
                if lim and int(lim.group(1)) - MARGIN < ctx:
                    ctx = int(lim.group(1)) - MARGIN
                    used = max(used, estimate)
                else:
                    ctx -= max(256, min(4096, (room - agent.MIN_SUBMIT) // 2))
            trace.append({"step": step, "error": f"MALFORMED: {e}", "http": e.status, "est_prompt": estimate, **fx, "utc": utc()})
            if strike():
                return fallback(step, "three malformed turns") if reasked is not None else done(None)
            continue
        except Exception as e:            # noqa: BLE001 - after a re-ask a failed request costs the correction, not the decision
            if reasked is None:
                raise
            trace.append({"step": step, "error": f"request failed after the re-ask: {type(e).__name__}: {str(e)[:200]}",
                          "utc": utc()})
            return fallback(step, "the request failed")
        usage = rep.get("usage") or {}
        pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        stats["cached_tokens"] += cached
        st["prefill"] += pt
        st["sampled"] += ct
        if pt:
            used = pt                     # the messages sent, as the server counted them
        choice = (rep.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        fin = choice.get("finish_reason")
        reasoning, content, raw_calls = msg.get("reasoning_content") or "", msg.get("content") or "", msg.get("tool_calls") or []
        if force:
            fx["reasoned"] = bool(reasoning.strip())
            stats["forced_reasoned"] += fx["reasoned"]
            if fin == "length":
                cut_forced = True
        base = {"step": step, "tokens": ct, "est_prompt": estimate, "prompt_tokens": pt, "said": reasoning, "content": content,
                "calls": raw_calls, "finish": fin, **fx,
                "response_id": rep.get("id"), "cached_tokens": cached, "utc": utc()}   # the only per-sleeve cost record:
                                                # every request bills to one shared session (the Tinker Owner, 20-21 Nov)
        if not rep.get("choices"):
            base["reply"] = {k: v for k, v in rep.items() if k != "usage"}   # an error object in a 200, kept as it came
        kind, got = parse_reply(fin, content, raw_calls, names)
        after = used + ct + FRAME         # the messages with this reply in them (estimated until the next reply counts them)
        if kind != "TOOL":
            trace.append({**base, "error": f"{kind}: {str(got)[:200]}"})
            if strike():
                return fallback(step, "three malformed turns") if reasked is not None else done(None)
            continue
        sub = next((g for g in got if g[2] == "submit"), None)
        if force and sub is None:         # the opened submit could only be a submit; a forced reply calling another tool is
            trace.append({**base, "error": "MALFORMED: forced turn without a submit call"})   # one malformed turn, as a 422 is
            if strike():
                return fallback(step, "three malformed turns") if reasked is not None else done(None)
            continue
        if sub is not None:
            idx, _, _, val = sub
            others = [str((c.get("function") or {}).get("name") or "?") for k, c in enumerate(raw_calls) if k != idx]
            val.setdefault("cancel", [])
            lack = tools.missing_prices(val.get("place"))
            if lack and reasked is None:
                out = tools.missing_text(lack)
                if ctx - after - est(out) >= agent.SUBMIT_ROOM:
                    reasked = val
                    trace.append({**base, "reask": "submit", "args": val, "result": out, "missing": [
                        {"order": i, "fields": f} for i, _, f, _ in lack], **({"not_run": others} if others else {}),
                        **({} if idx is not None else {"answered_as": "user"})})
                    if idx is not None:   # answered on the submit's call (its id as the history sends it)
                        call_sent = sent_calls(raw_calls, step)[idx]
                        msgs = msgs + [{"role": "assistant", "content": content or None, "reasoning_content": reasoning,
                                        "tool_calls": [call_sent]},
                                       {"role": "tool", "tool_call_id": call_sent["id"], "content": out}]
                    else:                 # a decision written as text: the same words, in a user message
                        msgs = msgs + [{"role": "assistant", "content": content, "reasoning_content": reasoning},
                                       {"role": "user", "content": out}]
                    used = after + est(out)
                    continue
            trace.append({**base, "tool": "submit", **({"not_run": others} if others else {})})
            return done(val)
        left = ctx - after
        per = max(est(x) for x in (                       # the longest line the loop adds for a call (review, 20 Nov: the
            f"context nearly full ({ctx} tokens left); no research call is run; " + agent.final_words(bool(checker)),
            f"research limit reached ({limit} calls): " + agent.final_words(bool(checker)),   # rest of a reply needs room
            checks.cap_text(checks.MAX_CHECKS).rstrip(".") + "; " + agent.final_words(False)))   # for its lines too)
        if free and left - (len(raw_calls) - 1) * per < agent.SUBMIT_ROOM:
            trace.append({**base, "room": room, "left": left, "dropped": [g[2] for g in got], "args": [g[3] for g in got],
                          "note": "turn left too little room for the decision: dropped, submit opened"})
            force = True
            continue
        told_before, opened = told, None  # opened: the line that opened the submit in this reply; later calls get it, unrun
        results = {}                      # index in raw_calls -> the text that answers it
        for n_k, (k, cid, name, val) in enumerate(got):
            rest = (len(raw_calls) - 1 - n_k) * per       # kept back for the lines the later calls get
            if opened is not None:
                out = opened
                trace.append({**base, "tool": name, "args": val, "result": out,
                              "not_run": "after the line that opened the submit in this reply"})
            else:
                runs = (name == "check_orders" and not (checked >= checks.MAX_CHECKS or tries + 1 > checks.MAX_TRIES)) or (
                    name != "check_orders" and calls < limit)
                if runs and left - rest - math.ceil(agent._estimate(name, val, checker) * SLACK) - FRAME < agent.RESERVE:
                    out = (f"context nearly full ({left} tokens left); no "
                           + ("check is run" if name == "check_orders" else "research call is run"))
                    if final:
                        out += "; " + agent.final_words(bool(checker))
                    force, opened = True, out
                    trace.append({**base, "tool": name, "args": val, "result": out})
                elif name == "check_orders":
                    raw, own = None, False
                    tries += 1
                    if checked >= checks.MAX_CHECKS or tries > checks.MAX_TRIES:
                        out, past = checks.cap_text(checked), past + 1
                        if past > 1:
                            out = out.rstrip(".") + "; call submit now"
                            if told_before:       # read "call submit now" in an earlier reply: this opens the submit
                                force = True
                                if final:
                                    out = checks.cap_text(checked).rstrip(".") + "; " + agent.final_words(False)
                                opened = out
                            told = True
                    else:
                        own = True
                        try:
                            out, counted, raw = checker(val)
                        except Exception as e:    # noqa: BLE001 - checks.check never raises; this is the last net
                            out, counted = f"error: the order check failed ({type(e).__name__}: {str(e)[:200]}); this check did not count.", False
                        counter = None
                        if counted:
                            checked += 1
                            counter = f"(order check {checked} of {checks.MAX_CHECKS})"
                    if own:
                        out, _ = fit_text(out, left - rest, counter)
                    trace.append({**base, "tool": name, "args": val, "result": out, "ibkr": raw})
                elif calls >= limit:
                    out = f"research limit reached ({limit} calls): call submit now"
                    if told_before:               # read "call submit now" in an earlier reply: this opens the submit
                        force = True
                        if final:
                            out = f"research limit reached ({limit} calls): " + agent.final_words(bool(checker))
                        opened = out
                    told = True
                    trace.append({**base, "tool": name, "args": val, "result": out})
                else:
                    calls += 1
                    with held:
                        out = run_tool(gw, name, val, ml_tool=ml_tool, sleeve_value=sleeve_value, look=look_key)
                    out, _ = fit_text(out, left - rest, f"(research call {calls} of {limit})")
                    trace.append({**base, "tool": name, "args": val, "result": out})
            left -= est(out)
            results[k] = out
            base = {"step": step, "same_reply": True}   # the reply's text is in its first call's entry only
        for k in range(len(raw_calls)):               # a malformed call beside good ones and no submit makes the reply
            if k not in results:                      # malformed, so this does not run; kept as a net so that every
                results[k] = "not run: this call could not be read"   # call id the history re-sends gets an answer
                left -= est(results[k])
        sent = sent_calls(raw_calls, step)
        msgs = msgs + [{"role": "assistant", "content": content or None, "reasoning_content": reasoning, "tool_calls": sent}] + [
                       {"role": "tool", "tool_call_id": c["id"], "content": results[k]} for k, c in enumerate(sent)]
        used = ctx - left
    trace.append({"error": "no submit within the step limit"})
    return fallback(step, "no submit within the step limit") if reasked is not None else done(None)
