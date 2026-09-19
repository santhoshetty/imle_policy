

import torch
import wandb
import pdb

def rs_imle_loss(real_samples, fake_samples, epsilon=0.03):
    B, T, D = real_samples.shape
    n_samples = fake_samples.shape[1]

    real_flat = real_samples.reshape(B, 1, -1)
    fake_flat = fake_samples.reshape(B, n_samples, -1)

    distances = torch.cdist(real_flat, fake_flat).squeeze(1)

    valid_samples = (distances > epsilon).float()
    # wandb.log({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon})
    min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
    valid_real_samples = (min_distances < distances.max()).float()
    if valid_real_samples.sum() > 0:
        loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
    else:
        loss = torch.tensor(0.0, device=real_samples.device)

    wandb_log = ({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon, "loss": loss.item()})
    return loss, wandb_log


def rs_imle_loss_batch_global(real_samples, fake_samples, epsilon=0.03):
    """
    Batch-global rejection variant of the rs-IMLE loss above, adapted from PRISM
    (Bhaskar, Tokekar, Di Cairano, Schperberg -- "PRISM: Performer RS-IMLE for Single-pass
    Multisensory Imitation Learning", arXiv:2602.02396). See IMPROVEMENTS.md for the full
    write-up of why this was tried and what happened.

    THE PROBLEM THIS TARGETS
    -------------------------
    `rs_imle_loss` above only ever compares a real target trajectory against the
    `n_samples_per_condition` candidates generated *for that same conditioning context*
    (real_flat is (B,1,D), fake_flat is (B,K,D) -- torch.cdist batches over B, so target i
    only ever sees candidates i's own generator call produced, never any other target's
    candidates). On this repo's pusht_vanilla_imle run, training loss and the rs-IMLE
    diagnostics (mean_distance, max_distance) plateaued hard around epoch ~270-290 while
    min_distance had been sitting below epsilon since roughly epoch 90 -- consistent with
    candidates cheaply satisfying their own target without the generator ever being pushed to
    cover genuinely different/harder trajectories, because there was never any pressure to
    produce candidates that are useful outside their own conditioning context.

    THE FIX
    -------
    Pool every candidate generated across the whole batch (B * n_samples_per_condition of
    them) into one shared set, and let every real target search that *entire* shared pool for
    its nearest not-yet-covered ("epsilon-far") candidate, instead of only its own K. A
    candidate is "not yet covered" the same way as before (distance > epsilon means it hasn't
    already matched something), but now a target can be satisfied by -- or need to pull
    closer -- a candidate that was generated for a *different* context entirely. This is
    strictly more information per gradient step (up to B times more candidates considered per
    target) at negligible extra cost, since the distance matrix here is over flattened
    (pred_horizon * action_dim)-dim trajectory vectors (32-d for this task's default config),
    not full images or activations -- a (B, B*n_samples_per_condition) cdist is cheap next to
    the network forward/backward pass that produced the candidates.

    v2 CORRECTION (see IMPROVEMENTS.md, "batch-global rejection v1 regression")
    -----------------------------------------------------------------------------
    The first version of this function only rejected a candidate *for the target it already
    covers* (a per-(target, candidate) pair check, `distances[i, k] > epsilon`) -- it pooled
    the candidates but never stopped a second, third, ... target from ALSO trying to pull that
    same candidate toward itself. A live run showed this made things worse than the original
    per-sample loss (min_distance regressed despite searching a strict superset of candidates),
    which only makes sense if multiple targets fighting over one candidate was diluting its
    gradient.

    PRISM's actual rejection mask (re-checked against the paper) is per-CANDIDATE and computed
    GLOBALLY across every target, not per-pair: candidate k is unavailable to *every* target,
    not just the one it already covers, the moment ANY target in the batch is within epsilon of
    it (paper: reject k iff min_j D(j,k) < epsilon). That is what actually prevents the
    many-targets-one-candidate contention -- once a candidate is "claimed" by covering some
    target, it drops out of consideration for everyone else too, rather than staying available
    for every other unsatisfied target to also pull on. The paper also specifies a fallback: if
    this global filtering would leave a target with zero available candidates (everything left
    in the pool is already claimed by someone else), that target reverts to its unrestricted
    per-pair-far-enough set rather than being left with nothing to pull toward.

    Usage: opt-in via train.py's --use_batch_global_rejection flag; the original
    `rs_imle_loss` is left untouched above so the two can be compared directly (see
    IMPROVEMENTS.md for the recorded before/after).
    """
    B, T, D = real_samples.shape
    n_samples = fake_samples.shape[1]

    real_flat = real_samples.reshape(B, -1)                    # (B, T*D)
    fake_flat = fake_samples.reshape(B * n_samples, -1)        # (B*K, T*D) -- pooled across the whole batch

    # (B, B*K): distance from every real target to every candidate generated anywhere in this
    # batch, not just the ones generated for that target's own conditioning context.
    distances = torch.cdist(real_flat, fake_flat)

    # Per-pair check (same rule as rs_imle_loss): candidate k is "good enough" *for target i*
    # once it's within epsilon of i specifically. This alone is what the first (buggy) version
    # used as its only filter.
    per_pair_far_enough = distances > epsilon                                    # (B, B*K)

    # The missing piece: a candidate is "claimed" -- and should drop out for EVERY target, not
    # just the one it covers -- the moment it's within epsilon of *any* target in the batch.
    # Without this, target j can still drag a candidate that already covers target i toward
    # itself too, and the resulting gradient update fights between i and j instead of letting
    # i's coverage stand and j look elsewhere.
    candidate_claimed_by_someone = distances.min(dim=0).values < epsilon         # (B*K,)
    globally_available = per_pair_far_enough & (~candidate_claimed_by_someone).unsqueeze(0)

    # Paper's fallback: if global filtering leaves a target with no available candidates at all
    # (everything remaining in the pool already covers someone else), fall back to that target's
    # raw per-pair-far-enough set rather than giving it nothing to pull toward.
    row_has_available_candidate = globally_available.any(dim=1)
    valid_samples = torch.where(
        row_has_available_candidate.unsqueeze(1), globally_available, per_pair_far_enough
    ).float()

    min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
    # A target only contributes to the loss if it actually found a not-yet-covered candidate to
    # pull closer (min_distances < distances.max() means the min() above hit a real, non-masked
    # entry rather than every candidate for that target being masked out).
    valid_real_samples = (min_distances < distances.max()).float()
    if valid_real_samples.sum() > 0:
        loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
    else:
        loss = torch.tensor(0.0, device=real_samples.device)

    wandb_log = ({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon, "loss": loss.item()})
    return loss, wandb_log
