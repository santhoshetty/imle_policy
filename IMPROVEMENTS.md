# Improvements Log

This file tracks *modeling* changes tried on top of the working baseline: the reasoning for
trying each one, what we expected to happen, and what actually happened once it was run. It's
separate from `CHANGES.md`, which tracks engineering/infra changes (GPU setup, checkpointing,
bug fixes, benchmarks) rather than modeling decisions.

**Format for each entry**: what it is, why we're trying it (the specific evidence that
motivated it), what we expect, and — filled in once actually run — what happened. Newest at
the top. An entry with an unfilled "What actually happened" is still running or not yet run.

---

## 2026-09-19 — Batch-global rejection rs-IMLE loss (PRISM-inspired)

**What it is**: an alternative to the existing `rs_imle_loss` (in
`imle_policy/utils/losses.py`), added as `rs_imle_loss_batch_global` and gated behind
`train.py --use_batch_global_rejection` (default off — the original loss is untouched and
remains the default). Implements the training-objective half of *PRISM: Performer RS-IMLE for
Single-pass Multisensory Imitation Learning* (Bhaskar, Tokekar, Di Cairano, Schperberg,
arXiv:2602.02396) — not the paper's Performer/linear-attention architecture or multisensory
encoder, which don't fit this task (see "What we didn't implement, and why" below).

The change itself: `rs_imle_loss` computes each real action's nearest-not-yet-covered candidate
only among the `n_samples_per_condition` candidates generated *for that same conditioning
context* (a (B,1,D) vs (B,K,D) batched `cdist`, so target *i* only ever sees candidates *i*'s
own generator call produced). `rs_imle_loss_batch_global` instead pools every candidate
generated across the whole batch (B×K of them) into one shared set and computes a full
(B, B·K) distance matrix, so a target's nearest-not-yet-covered candidate can come from *any*
conditioning context in the batch, not just its own.

**Why we're trying it**: the `pusht_vanilla_imle` baseline run (see `CHANGES.md`) plateaued at
~45-50% success rate from roughly epoch 270 onward, with three independent signals pointing at
a genuine ceiling rather than "just needs more training":
- Training loss flattened at 0.043-0.045 for ~65 epochs (steep drop before that, then flat).
- The cosine LR schedule still had ~62% of its decay budget left at the point of plateau —
  ruling out "the LR just needs to finish annealing."
- The rs-IMLE `min_distance` diagnostic had been below `epsilon` (0.03) since ~epoch 90, while
  `max_distance` stopped shrinking and got noisier — consistent with candidates cheaply
  satisfying their own target's epsilon-ball without the generator ever being pressured to
  produce genuinely diverse/novel trajectories, since a candidate only ever needs to look good
  against its own K siblings.

Batch-global rejection targets that specific mechanism: it removes the "only needs to beat its
own K siblings" escape hatch by making every candidate compete against, and be available to,
every target in the batch.

**What we expect**: if the diagnosis is right, this should show up as continued (rather than
flattened) improvement in `mean_distance`/`max_distance` (the rs-IMLE diagnostics logged to
wandb) past the point where the baseline plateaued, and ideally a success rate that climbs
past ~48-50% rather than sitting there. If the diagnosis is wrong (e.g. the ceiling is really
just raw model capacity — the 7.5M-param `down_dims=[64,128,256]` network — and not candidate
diversity), we'd expect this change to make little difference, in which case model capacity
(reverting toward a larger `down_dims`, at whatever batch-size/VRAM trade-off that requires) is
the next thing to try instead.

**What actually happened**: _(not yet run — filled in after training with
`--use_batch_global_rejection` completes far enough to compare against the epoch-289
baseline (48% success, 0.776 mean reward) and the pre-change checkpoint saved at epoch 313:
`saved_weights/comfy-pine-1_..._pusht/{net,ema_net}_weights_baseline_pre_prism_epoch313.pth`)._

**What we didn't implement, and why**: PRISM's Performer/linear-attention generator and
multisensory fusion encoder target real-time (30-50Hz) multi-task control with multiple sensor
modalities on real hardware (the paper evaluates on MetaWorld/CALVIN/Robomimic and two real
robots). This repo's PushT setup is single-camera + 2D lowdim state, already runs well under
the real-time budget, and has no multi-sensor-fusion problem to solve — adopting that
architecture would be a large rewrite reopening the 2GB-VRAM capacity fight already solved for
the conv-net policy (see `CHANGES.md`), for a problem this task doesn't have. Also not
implemented yet: PRISM's EMA-calibrated adaptive epsilon (current `epsilon=0.03` stays fixed
for the whole run) and the Charbonnier robust distance in place of L2 — both plausible small
follow-ups, listed here rather than in a fresh entry since they haven't been tried.

---

## Template for future entries

```
## YYYY-MM-DD — <name of the change>

**What it is**: <the change, and where it lives in the code>

**Why we're trying it**: <the specific evidence/reasoning that motivated it, not just "seemed
like a good idea">

**What we expect**: <a falsifiable prediction — what would confirm this was worth doing, what
would suggest it wasn't>

**What actually happened**: <filled in once run — the actual numbers, and whether the
prediction above held>
```
