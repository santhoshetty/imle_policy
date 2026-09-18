

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

    # Same rejection rule as rs_imle_loss: a candidate already within epsilon of a target is
    # "good enough" and excluded (set to distances.max() so it can't win the min()) -- the loss
    # should only pull the closest candidate that ISN'T already a good match, not fight over
    # ones that already are.
    valid_samples = (distances > epsilon).float()
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
