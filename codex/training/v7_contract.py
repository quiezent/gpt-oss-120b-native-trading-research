"""Separately bound tool-capable V7 decisions with unchanged V6 hard gates.

Only genuine host observations enter the final receipt envelope. This offline
profile neither admits a checkpoint nor grants any broker execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping

from copy import deepcopy

import hashlib

import re

from typing import Any

from training import v6_contract as v6

def parse_response(text: str) -> dict:
    parsed = v6.parse_response(text)
    guard_content(parsed)
    return parsed

from trading_desk.v7_tools import guard_content
