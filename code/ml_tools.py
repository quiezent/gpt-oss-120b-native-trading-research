"""The forecast tool's versions, one module each, so a sleeve keeps its version when a newer one arrives.

Clayton, 2 Nov 2026: ML2 stays available when there is an ML3 or ML4. A sleeve names its version in sleeves.json's
ml_tool. A bug fix goes into that version's module and reaches its sleeves at the next restart (ML2.2). An upgrade, such
as extra training or changed models, is a new version: a new module, with its own model files and cache, added to
VERSIONS. It leaves the older modules and their files as they are; ML2_FILES' hashes catch one changed by
mistake. A new module offers what forecast.py does: VERSION, SPEC, run(gw, args), compute(); its SPEC keeps the name
forecast and the parameters symbols, notional_usd and sleeve_usd, which mcp_server.py declares. History, the sleeves on
each version and what a new version must not touch: runs/ml/research/ml_tool_versions.md.
"""

import hashlib
import importlib
import json
from pathlib import Path

VERSIONS = {"ML2": "forecast"}        # version -> module
DEFAULT = "ML2"                       # a sleeve with the tool and no ml_tool

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "runs" / "ml" / "research" / "ml2_manifest.json"
ML2_FILES = ([f"runs/ml/research/{n}" for n in ("vol2.json", "cost_model.json", "events_model.pkl", "bondfx.json",
                                                "intraday_model.pkl", "regime.json")]
             + ["data/edgar/companies.json", "data/edgar/filings.csv", "data/edgar/earnings_dates.json"]    # not in git:
             + ["forecast.py", "ml.py"]                                                # a copy is in data/edgar/frozen_ml2_20261102
             + [f"ml_research/{n}.py" for n in ("vol2", "risk", "cost", "events", "edgar_fetch", "bondfx", "intraday", "regime",
                                                "macro_feats")])


def module(version=None):
    """The module of a version. An unknown one raises: a sleeve never gets another version's tool silently."""
    name = VERSIONS.get(version or DEFAULT)
    if name is None:
        raise KeyError(f"no ML tool {version!r}; have {sorted(VERSIONS)}")
    return importlib.import_module(name)


def in_use(sleeves):
    """The versions the sleeves with the tool name, the default first."""
    return sorted({m.get("ml_tool") or DEFAULT for m in sleeves.values() if m.get("ml")}, key=lambda v: (v != DEFAULT, v))


def unknown(sleeves):
    """Sleeves with the tool whose version isn't in VERSIONS: they get no look until it is."""
    return sorted(k for k, m in sleeves.items() if m.get("ml") and (m.get("ml_tool") or DEFAULT) not in VERSIONS)


def version_of(meta):
    """The exact revision a sleeve's look ran on (forecast.VERSION), None without the tool, the name itself if unknown."""
    if not meta.get("ml"):
        return None
    v = meta.get("ml_tool") or DEFAULT
    return module(v).VERSION if v in VERSIONS else v


def _sha(p):                           # line endings aside: git may rewrite them on a text file
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _texts():
    """What the sleeves on ML2 read about the tool."""
    import trader
    tool = module("ML2")
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:16] for k, v in
            {"forecast.DESCRIPTION": tool.DESCRIPTION, "forecast.SPEC": json.dumps(tool.SPEC, sort_keys=True),
             "trader.S3_ML": trader.S3_ML, "trader.live_steps": trader.live_steps(True)}.items()}


def changed_files():
    """What differs from ML2's manifest: its source, model and data files, what it tells the model, or a new revision
    with no new manifest. A fix that means to change one of these writes a new manifest with its revision
    (write_manifest); anything else here is an upgrade reaching ML2, or ML2 being overwritten."""
    if not MANIFEST.exists():
        return ["(no manifest)"]
    doc, tool = json.loads(MANIFEST.read_text(encoding="utf-8")), module("ML2")
    out = sorted(f for f, h in doc["files"].items() if not (ROOT / f).exists() or _sha(ROOT / f) != h)
    out += sorted(k for k, h in _texts().items() if doc.get("texts", {}).get(k) != h)
    if doc.get("version") != tool.VERSION:
        out.append(f"forecast.VERSION {tool.VERSION}, manifest {doc.get('version')}")
    return out


def write_manifest():
    MANIFEST.write_text(json.dumps({"version": module("ML2").VERSION, "texts": _texts(),
                                    "files": {f: _sha(ROOT / f) for f in ML2_FILES}}, indent=1), encoding="utf-8")
