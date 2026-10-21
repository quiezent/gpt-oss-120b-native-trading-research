# Authored code export

This is a sanitized, portable subset of my project code. The release manifest maps every adapted file to its repository-relative original and SHA-256, with exact extraction, relocation and configuration changes. The private campaign receipt graph, market history, real broker payloads, model reasoning, token/logprob arrays, vendor code and credentials are excluded.

The actual NAV8 checkpoint came from the completed corrective named-framework campaign using NAV7 weights. The earlier NAV7 campaign used NAV6 weights. The named-tool declaration, shared simulator and whole-group failure-classification interfaces were genuinely used by that learner. The portable training wrapper remains a derivative of the genuine outcome-update path, not a replay of the private campaign. The optional numeric_recovery module is a separately staged generic repair; it was not used in NAV7 training or established by a paper run.

The separate `training/learning/argument_recovery` package is an import-relocated
export of authored components frozen for the actually completed corrective learner.
Its generic visible error path differs from NAV7's terminal argument refusal.
The completed run and separately verified sampler publication establish actual
NAV8; `CURRENT_WORK.md` and the report record its limited results. The exact
private financial controller and receipt graph remain excluded. The prior numeric_recovery module is preserved.

## Local setup

Use Python 3.11. The simulation accounting example uses the standard library. For native Harmony and gradient code, install `requirements-training.txt` in your own environment. No SDK, model weights or tokenizer bytes are vendored.

```sh
python -m venv .venv
# Activate your environment, then:
python -m pip install -r requirements-training.txt
python -m pip install -e .
python -m examples.synthetic_accounting
```

Fetch the pinned public tokenizer yourself with `python -m training.environment.prepare_tokenizer`. Its reviewed checksum manifest is included. `GPT_OSS_TOKENIZER_DIR` can instead point to your own already verified tokenizer directory. Downloading tokenizer assets does not create a Tinker model/session.

For IBKR, install the official Python TWS API separately under its applicable license, then `requirements-paper.txt`. The compatibility code was exercised against API 1051.1. `PA_PYTHON` selects that interpreter. The public code contains a `PAPER_ACCOUNT` placeholder. Configure an ignored local copy using:

```sh
python -m examples.configure_paper --paper-account YOUR_PAPER_ACCOUNT
```

This only creates `.local/paper_workspace`; it does not connect, submit or cancel. Use your own paper TWS/Gateway settings, account, risk-policy file and exclusive client ownership. Native mutations remain explicitly disabled by default in `PaperCliConfig`. The exact CLI/source/boundary checks and unchanged model arguments are retained.

## Components and training

- `training/nav_learning.py`: chronological daily-bar simulation, next-open limit-order execution, modeled spread/slippage/commission, cash-flow-adjusted marked NAV, native-token datums and zero observation-gradient mask.
- `training/native_agent`: one-conversation runtime, raw-byte-before-parse persistence, strict Harmony output, exact CLI arguments, durable once identities, visible refusals, clock/wait support and framework callbacks.
- `training/framework`: exact model-supplied mandate, the original wrapper ABI
  and the declared nine-function ABI used by the named learners. A recipient/body
  mismatch is refused, never aliased or repaired. The separate corrective
  package returns exact declared argument-schema errors visibly to the model.
- `training/learning`: shared simulator interface, source-known failure classification, capture/accounting proof and symmetric whole-group centering. A known late malformed output retains its actual signed terminal NAV; an unknown member excludes the whole group. Zero variation produces no learning signal.
- `training/learning/outcome_training.py`: `train_verified_group` uses the genuine `trajectory_datums`/SDK importance-sampling update path. Supply fresh same-start/horizon/interface on-policy episodes, their unmodified outcomes and `TrajectoryIntegrity` proofs from `capture_provenance`. The caller owns the SDK session/sampler, explicit finite dispatch budget, data rights and unconditional owned close. The wrapper makes at most one optimizer update and skips a zero-signal group. It does not choose trades, fabricate proofs, score schema compliance or manufacture a checkpoint.
- `training/learning/argument_recovery`: the actual corrective model/runtime,
  interface, failure-classification, capture and eligibility bodies with listed
  import relocations. Use this package's `interface_adapter`, `capture_provenance`
  and `evidence_eligibility` together. Its `outcome_training` entry uses the same
  public one-update wrapper with the corrective eligibility import. This is a
  portable derivative; the private financial/run controller is excluded.

`create_compatible_training_client` retains rank/attention/MLP/unembed flags needed by the published parent. Native sampler identity can differ from checkpoint owner: bind the actual SDK saved URI and actual current sampling session separately. `datum_wire` records the SDK's real float32 representation; exact reconstruction must round before equality. No raw real tensor values are distributed here. The previous backend loss-scalar reconstruction uncertainty remains unresolved; this code does not assert it was explained.

Using the provider is an explicit caller action. `config/checkpoint.example.json` contains the actual NAV8 sampler URI whose Tinker metadata was confirmed public with no scheduled expiry. Select your own Tinker project/authentication. No API key is included. This metadata is distinct from an anonymously downloadable file or independently tested access by a second principal.

## Focused checks

```sh
python -m unittest -v training.test_nav_learning training.native_agent.test_runtime training.native_agent.test_harmony_model
```

These use authored synthetic/fake transport fixtures. They check accounting, chronology, native mask alignment, exact terms, refusals, raw persistence and context bounds. They are not a new economic model evaluation. The original full campaign and private data are not required or distributed. Check receipts report the actual environment used; tokenizer assets were consumer supplied.

Daily bars hide queues, intraday liquidity, partial fills and original publication vintages. Terminal evaluator advance marks remaining inventory; it does not prove a model-chosen exit. Current-universe and previously exposed windows limit the historical evidence. There is no profitability or production claim.

Real future raw completions, private reasoning, native tokens/logprobs, datums, broker payloads and credentials belong in ignored local run directories. Aggregate published digests identify original retained private records; they are not released files or download links.

The additional numeric-recovery check imports the sanitized modules under a no-network/no-provider/broker-constructor fence and exercises an authored invalid-argument fixture. It is a component check only, with no model generation or economic evaluation.

The corrective package adds focused fixtures for a nonnumeric refusal followed
by a new model-selected valid call, fatal recipient/body mismatch, changed
refusal provenance, signed-NAV retention and whole-group unknown exclusion:

```sh
python -m unittest -v training.learning.argument_recovery.test_visible_errors
```

These fixtures use fake outputs and a fake host. They are not actual model
generations, provider/broker calls or economic evaluations.
