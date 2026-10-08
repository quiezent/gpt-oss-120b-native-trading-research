"""Fetch only a pinned public tokenizer; never model weights or Tinker state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os

REPO_ID = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
FILES = (
    "chat_template.jinja", "config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json",
)
DIRECTORY = Path(os.environ.get("GPT_OSS_TOKENIZER_DIR", str(Path(__file__).resolve().parent / "tokenizer")))
MANIFEST = Path(__file__).resolve().parent / "tokenizer_manifest.json"


def verify_tokenizer(directory: Path = DIRECTORY) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != REPO_ID or manifest.get("revision") != REVISION:
        raise ValueError("tokenizer revision mismatch")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("tokenizer manifest is incomplete")
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("tokenizer file hash mismatch: " + name)
    return manifest


def main() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID, revision=REVISION, allow_patterns=list(FILES),
        local_dir=str(DIRECTORY), token=False,
    )
    # First bootstrap prints evidence for review; it does not silently trust or
    # replace a frozen manifest. Normal setup requires that manifest to exist.
    if not MANIFEST.exists():
        print(json.dumps({
            "repo_id": REPO_ID, "revision": REVISION,
            "files": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                      for name in FILES},
        }, sort_keys=True, indent=2))
        raise SystemExit("Tokenizer fetched; freeze reviewed checksums before use.")
    verify_tokenizer()
    print("Pinned public tokenizer files verified; no model weights downloaded.")


if __name__ == "__main__":
    main()
