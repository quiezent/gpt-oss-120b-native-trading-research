"""Offline Tinker datum preparation. No ServiceClient or provider calls."""

from __future__ import annotations

import hashlib

import inspect

import json

import math

import platform

import sys

from dataclasses import dataclass

from importlib.metadata import version

from pathlib import Path

from typing import Any

from training.environment.prepare_tokenizer import DIRECTORY, verify_tokenizer

RENDERER_NAME = "gpt_oss_no_sysprompt"

MAX_SEQUENCE_TOKENS = 32768

PINS = {
    "tinker": "0.31.0", "tinker-cookbook": "0.5.7", "torch": "2.10.0+cpu",
    "transformers": "5.5.4", "tiktoken": "0.12.0",
}

def verify_versions() -> dict[str, str]:
    actual = {name: version(name) for name in PINS}
    if actual != PINS:
        raise ValueError("training dependency versions do not match the frozen pins")
    return actual

def local_renderer(tokenizer_dir: Path = DIRECTORY):
    verify_versions()
    verify_tokenizer(tokenizer_dir)
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False,
    )
    return renderers.get_renderer(RENDERER_NAME, tokenizer, model_name=BASE_MODEL)

BASE_MODEL = "openai/gpt-oss-120b"
