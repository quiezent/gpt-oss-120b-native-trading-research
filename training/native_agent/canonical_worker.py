"""Broker-interpreter worker: exact canonical argv, genuine read serialization.

This process never selects an economic action. Basic reads hold the canonical
MutationLock; commands which lock inside pa_tws are invoked without nesting it.
All commands launch the same canonical CLI. No module load connects to IBKR.
"""
from __future__ import annotations

import base64
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASIC_READS = frozenset({"status", "account", "positions", "portfolio", "pnl",
                       "open-orders", "completed-orders", "executions", "contract", "quote"})


def execute(request):
    sys.path.insert(0, str(ROOT))
    from pa_tws import pa_tws as native
    argv = request["argv"]
    cli = ROOT / "pa_tws" / "pa_tws.py"
    if (type(argv) is not list or len(argv) < 8 or argv[:4] != [sys.executable, "-B", "-X", "utf8"]
            or Path(argv[4]).resolve() != cli or argv[5] != "--timeout"
            or Path(native.__file__).resolve() != cli
            or hashlib.sha256(cli.read_bytes()).hexdigest() != request["cli_sha256"]
            or (native.HOST, native.PORT, native.ACCOUNT, native.DEFAULT_CLIENT_ID) !=
                ("127.0.0.1", 4002, "PAPER_ACCOUNT", 9901)
            or not 0 < request["child_deadline_seconds"] <= 90):
        raise ValueError("exact canonical broker interpreter/CLI/boundary required")
    lock = native.MutationLock() if argv[7] in BASIC_READS else nullcontext()
    with lock:
        try:
            result = subprocess.run(argv, cwd=str(ROOT), env=None, shell=False,
                                    capture_output=True, timeout=request["child_deadline_seconds"])
            return {"stdout_base64": base64.b64encode(result.stdout).decode(),
                    "stderr_base64": base64.b64encode(result.stderr).decode(),
                    "returncode": result.returncode, "uncertain": False}
        except subprocess.TimeoutExpired as error:
            return {"stdout_base64": base64.b64encode(error.stdout or b"").decode(),
                    "stderr_base64": base64.b64encode(error.stderr or b"").decode(),
                    "returncode": None, "uncertain": True}


def main():
    try:
        request = json.loads(sys.stdin.buffer.read(262144))
        result = execute(request)
        print(json.dumps({"kind": "canonical_native_worker_v1", **result}))
        return 0
    except Exception as error:
        print(json.dumps({"kind": "canonical_native_worker_failure_v1",
                          "error": type(error).__name__, "message": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
