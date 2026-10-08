"""Costed-NAV one-update operation over caller-verified on-policy episodes.

The caller owns its SDK session/sampler, finite dispatch accounting, source proof
and data rights. This function never chooses or repairs a trade. It returns no
successor on zero variation. Local logs from real generations must stay private.
"""
from pathlib import Path
import json
from training.nav_learning import trajectory_datums, datum_wire, sdk_response_wire
from training.learning.evidence_eligibility import evidence_group_advantages

def train_verified_group(training_client, episodes, *, output_directory, learning_rate=1e-5):
    """Each episode supplies original signed outcome, verified proof and samples.

    Fresh on-policy samples must belong to this training client's current sampler.
    Proofs are the source-classified TrajectoryIntegrity objects; any unknown
    member excludes the whole group. Whole generated output receives gradient;
    observation tokens receive zero. No trade, activity or schema reward is added.
    """
    output=Path(output_directory)
    output.mkdir(parents=True,exist_ok=False)
    outcomes=[row['outcome'] for row in episodes]
    proofs=[row['integrity'] for row in episodes]
    advantages=evidence_group_advantages(outcomes,proofs)
    datums=[]
    for episode,advantage in zip(episodes,advantages):
        if advantage is not None and advantage != 0:
            datums.extend(trajectory_datums(episode['native_steps'],advantage))
    summary={'new_optimizer_steps':0,'group_advantages':advantages,
             'datum_count':len(datums),'profitability_proven':False}
    # Exact SDK wire has float32 conversion, including advantages/logprobs.
    with (output/'datums.private.json').open('x') as f:
        json.dump([datum_wire(d) for d in datums],f,allow_nan=False)
    if datums:
        backward=training_client.forward_backward(datums,loss_fn='importance_sampling').result(timeout=600)
        with (output/'backward.private.json').open('x') as f:
            json.dump(sdk_response_wire(backward),f,allow_nan=False)
        import tinker
        update=training_client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result(timeout=600)
        with (output/'optimizer.json').open('x') as f:
            json.dump(sdk_response_wire(update),f,allow_nan=False)
        summary['new_optimizer_steps']=1
    with (output/'summary.json').open('x') as f:json.dump(summary,f,allow_nan=False)
    return summary
