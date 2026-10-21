"""Portable batch API derived from the executed expanded-compute learner.

This is an authored portability wrapper, not the executed campaign driver.
Caller owns authentic capture/accounting proofs, data rights, current on-policy
weights, finite admission, SDK dispatch, raw-return persistence and owned close.
All generated tokens receive the unchanged signed centered NAV advantage;
observation tokens receive zero. No investment, action repair or schema reward.
"""
from decimal import Decimal
from pathlib import Path

from training.nav_learning import datum_wire, digest, trajectory_datums, write_once
from training.learning.argument_recovery.evidence_eligibility import evidence_group_advantages
from training.learning.argument_recovery.capture_provenance import verify_datum_step_binding


def assemble_verified_batch(groups):
    """Accumulate every eligible varied group into one batch before any update.

    A group is a sequence of three episodes. Each episode supplies outcome,
    native_steps, integrity and records from the original trajectory_proof.
    Caller must authenticate those proofs against the actual on-policy sampler,
    exact original captures, requests and completed accounting. Passing values
    to this API alone does not authenticate provider or account history.
    """
    datums, descriptive_groups = [], []
    for episodes in groups:
        if len(episodes) != 3:
            raise ValueError("EXACT_THREE_ORIGINAL_ROLLOUTS_PER_GROUP_REQUIRED")
        outcomes = [row['outcome'] for row in episodes]
        integrities = [row['integrity'] for row in episodes]
        advantages = evidence_group_advantages(outcomes, integrities)
        for row, advantage in zip(episodes, advantages):
            if advantage is not None and advantage != 0:
                verify_datum_step_binding(row['integrity'], row['records'], row['native_steps'])
                datums.extend(trajectory_datums(row['native_steps'], advantage))
        descriptive_groups.append({'outcomes': outcomes, 'advantages': advantages,
            'training_integrity': [{'eligible': proof.eligible, 'reason': proof.reason,
                'outcome_sha256': proof.outcome_sha256,
                'captured_records_sha256': proof.captured_records_sha256}
                for proof in integrities], 'all_descriptive_outcomes_retained': True})
    return datums, descriptive_groups


def apply_verified_batch(training_client, groups, *, output_directory, batch_index,
                         admit_training, dispatch, learning_rate=1e-5):
    """Apply at most one outcome update to the whole accumulated batch.

    admit_training(datums) must enforce the caller's already-funded local quota.
    dispatch has the executed runner's sdk_call signature and must claim before
    its once-only function, preserve original raw return before accounting,
    then return that genuine result. The wrapper creates no session, sampler,
    reservation, financial journal or checkpoint. The historical USD0.737/M
    training-input estimate is the executed conditional schedule, not an invoice
    or a promise about future pricing. Output files can contain private tokens.
    """
    if not callable(admit_training) or not callable(dispatch):
        raise ValueError("CALLER_OWNED_FINITE_ADMISSION_AND_RAW_FIRST_DISPATCH_REQUIRED")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("EXPLICIT_NONNEGATIVE_BATCH_INDEX_REQUIRED")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    datums, descriptive_groups = assemble_verified_batch(groups)
    batch = {'batch': batch_index, 'groups': descriptive_groups, 'optimizer_steps': 0,
        'status': 'NO_ADMISSIBLE_WITHIN_GROUP_VARIATION_NO_UPDATE'}
    if datums:
        admit_training(datums)
        native = [datum_wire(datum) for datum in datums]
        write_once(output / f"batch-{batch_index}-actual-native-rl-datums.json", native)
        tokens = sum(datum.model_input.length for datum in datums)
        charge = Decimal(tokens) * Decimal(".737") / 1000000
        backward = dispatch(f"batch-{batch_index}-outcome-backward",
            lambda: training_client.forward_backward(datums, loss_fn="importance_sampling"),
            maximum=charge, request={"input_tokens": tokens, "native_datums_hash": digest(native)},
            limits={"training_input_tokens": tokens})
        import tinker
        optimizer = dispatch(f"batch-{batch_index}-optimizer",
            lambda: training_client.optim_step(tinker.AdamParams(learning_rate=learning_rate,
                beta1=.9, beta2=.95, eps=1e-8)), kind="optimizer")
        batch.update(status="ACTUAL_OUTCOME_UPDATE", optimizer_steps=1,
            backward_metrics=backward.metrics, optimizer_metrics=optimizer.metrics)
    write_once(output / f"batch-{batch_index}-report.json", batch)
    return batch
