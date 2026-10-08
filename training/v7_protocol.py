"""Offline, strict GPT-OSS Harmony protocol for V7 read-only tool trajectories.

No client, authentication, provider, or broker is created here.  Sampling starts
after the native bare assistant header.  Private sampled analysis is retained
only in the current token continuation; it is never final or tool authority.
"""

from __future__ import annotations

import hashlib

import inspect

import json

import re

from dataclasses import dataclass

from datetime import date as _date

from functools import lru_cache

from pathlib import Path

from typing import Any

MAX_SEQUENCE_TOKENS = 32768

MAX_TOTAL_GENERATED_TOKENS = 8192

MAX_TOOL_YIELDS = 6

REASONING_EFFORT = "medium"

PINNED_RENDERER_SHA256 = "0f3975177f453399c83ea20ac213f0a178a2023e8314421dc801431dd46fd27d"

SUPERVISION_POLICY = "all_reviewed_assistant_native_outputs_v7_no_private_cot"

def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def protocol_binding() -> dict:
    """Bind the actual installed renderer source, not an online/main implementation."""
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    from training.tinker_adapter import verify_versions
    source = Path(inspect.getfile(GptOssRenderer))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != PINNED_RENDERER_SHA256:
        raise ValueError("V7 pinned GPT-OSS renderer source changed")
    return {"renderer": "gpt_oss", "renderer_source_sha256": actual,
            "packages": verify_versions(), "reasoning_effort": REASONING_EFFORT,
            "as_of_date": "decision_at UTC date; never wall clock",
            "generation_prefix": "<|start|>assistant", "terminal_return": 200002,
            "terminal_tool": "derived from pinned tokenizer <|call|>",
            "parser": "strict token spans; recipient-free final only",
            "supervision_policy": SUPERVISION_POLICY,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "max_total_generated_tokens": MAX_TOTAL_GENERATED_TOKENS,
            "max_tool_yields": MAX_TOOL_YIELDS,
            "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def local_renderer(current_date: str | None = None):
    """Load only the verified local tokenizer; the projection supplies the date."""
    from training.tinker_adapter import local_renderer as load
    base = load()
    protocol_binding()
    if current_date is None:
        return base
    return _configured_renderer(base, current_date)

def _configured_renderer(renderer, date: str):
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("V7 requires an explicit as-of date")
    _date.fromisoformat(date)
    if type(renderer) is not GptOssRenderer:
        raise ValueError("V7 requires the pinned GPT-OSS renderer")
    return GptOssRenderer(renderer.tokenizer, use_system_prompt=True,
                          reasoning_effort=REASONING_EFFORT, current_date=date)

def _safe_content(text: str, renderer, *, empty: bool = False) -> None:
    if not isinstance(text, str) or (not empty and not text.strip()):
        raise ValueError("V7 message content must be nonempty text")
    if (any(token in text for token in renderer.tokenizer.get_added_vocab())
            or re.search(r"<\|[^<>\r\n]*\|>", text)):
        raise ValueError("literal tokenizer control token in V7 message content")

def stop_token_ids(renderer) -> list[int]:
    tokens = [_single_token(renderer, token) for token in ("<|return|>", "<|call|>")]
    if tokens[0] != 200002 or tokens != renderer.get_stop_sequences():
        raise ValueError("V7 pinned terminal token contract changed")
    return tokens

def _single_token(renderer, text: str) -> int:
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError("Harmony control must be one token: " + text)
    return tokens[0]

@lru_cache(maxsize=2)
def _known_token_ids(tokenizer) -> frozenset[int]:
    return frozenset(tokenizer.get_vocab().values())
