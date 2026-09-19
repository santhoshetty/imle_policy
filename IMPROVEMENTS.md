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

**What actually happened (v1 — killed, regressed)**: ran as `pusht_prism_batch_global`
(wandb run `fk21wwb0`). Result was the opposite of expected: loss converged *slower* than the
baseline, and the gap widened monotonically rather than closing — +22% worse at step 4000,
+53% worse at step 8000 (mean loss in 200-step bins). `min_distance` (best candidate found per
target) was also consistently *worse* than the baseline from step ~400 onward, which shouldn't
be possible given the new loss searches a strict superset of candidates (B·K=1280 pooled vs.
K=20 own-only) — a superset search can never do worse than a subset search on the same network
unless the search/selection mechanism itself is broken. No other red flags (no NaN/inf,
`zero_loss` not spiking, per-step wall-clock ~0.94s vs. baseline's ~0.91s — not a speed
regression). Killed at epoch 20/500 (step ~8000) rather than let it keep burning GPU time on a
clearly-diverging trend.

**Root cause (found by re-reading the paper's algorithm box, not just its prose summary)**: v1
only rejected a candidate *for the target it already covers* — a per-(target, candidate) pair
check (`distances[i,k] > epsilon`). It pooled the candidates but implemented none of the
bookkeeping that makes pooling safe: nothing stopped a second, third, ... target from also
trying to pull that same already-claimed candidate toward itself, so gradient updates for
"popular" candidates fought between multiple targets simultaneously. PRISM's actual rejection
mask is per-CANDIDATE and computed GLOBALLY: candidate k is unavailable to *every* target the
moment ANY target in the batch is within epsilon of it (`reject k iff min_j D(j,k) < epsilon`),
with a fallback to the unrestricted set if that filtering would leave a target with zero
candidates. v1 implemented the pooling but omitted exactly the piece the paper calls out as
what prevents this contention.

**v2 fix applied**: `rs_imle_loss_batch_global` corrected to do the global per-candidate
rejection (`candidate_claimed_by_someone = distances.min(dim=0).values < epsilon`, applied to
every target's row) with the paper's empty-pool fallback. Unit-checked on synthetic data before
rerunning: v2's `min_distance` is now properly *better* than the per-sample loss's (as it should
be, searching a superset), unlike v1 which regressed on that exact metric.

**What actually happened (v2 — killed, same failure shape as v1, just delayed)**: ran as
`pusht_prism_batch_global_v2` (wandb run `tli5zovm`,
https://wandb.ai/santhoshetty-norican-digital/pusht_prism_batch_global_v2). Started with a
genuine early advantage over the baseline (loss -29% better at step 500, -19% at step 1000) —
something v1 never showed — but the baseline caught up by step ~2000 and pulled steadily ahead
from there: +15% worse at step 3000, +38% at step 4000, +52% at step 8000, **+56% at step
8800**. That final number is essentially identical to v1's own +53% at the same step count
(8000) — meaning the rejection-scope fix changed the shape of the failure (a real early win
before it flips) but not the eventual outcome. Killed at epoch 22/500 (step ~9100).

### VERDICT

**Batch-global rejection, in either implementation, does not help on this task and should not
be pursued further without first implementing PRISM's adaptive epsilon (or understanding why it
matters here) — see the deeper investigation below for why.** Two runs, two independent
implementations of the mechanism (one admittedly buggy, one verified correct against the
paper's algorithm box), both converged slower than the unmodified `rs_imle_loss` by a similar
final margin (~53-56% worse loss by step ~8000-8800). The corrected version's early advantage
(steps 500-1500) shows the *idea* has some validity — batch-pooling clearly helps early when
few candidates are "claimed" yet — but something about running it against this task/pool-size
combination for longer actively hurts rather than plateaus-and-holds. That's a stronger, more
specific failure than "didn't help"; it's actively worse than doing nothing, at least in its
current form.

Dispatched a deeper investigation (see below once filled in) into two live hypotheses:
(a) PushT's action space may be too narrow/low-dimensional a manifold for batch-pooling to make
semantic sense — pooling helps when a batch's targets are genuinely diverse (PRISM's own
multi-task benchmarks), and may actively hurt when most targets are drawn from one task's
smooth, narrow action manifold and end up spuriously "satisfying" each other; (b) a fixed
`epsilon=0.03` may be miscalibrated for a candidate pool 64x larger (B·K=1280 vs K=20) — order-
statistics alone predicts the expected nearest-neighbor distance shrinks as the pool grows,
independent of whether the network learned anything, which could be starving `valid_real_samples`
of genuine gradient signal. PRISM's own paper pairs batch-global rejection with an EMA-adaptive
epsilon rather than a fixed one — we did not implement that piece, so hypothesis (b) would mean
our fixed-epsilon version was structurally never going to work regardless of the rejection-scope
fix, independent of hypothesis (a).

**_(deeper investigation results — filled in once the dispatched subagent returns)_**

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
