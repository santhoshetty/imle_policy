# Changes Log

Running log of changes made to get `pusht` training running at a reasonable speed on this
machine's hardware (NVIDIA GeForce MX450 laptop GPU, 2GB VRAM), plus how to use the
checkpointing/resume system that came out of it. **Newest entries at the top.** When making
further changes (e.g. trying a different policy architecture), add a new dated section above
the previous one: what changed, why, and any benchmark numbers — this file is meant to keep
tracking the model's progress over time, not just the state it's in today.

This file is the engineering changelog (infra, config, bugs, benchmarks). For the reasoning
behind and outcomes of actual modeling changes (new losses, architecture tweaks, hyperparameter
experiments), see `IMPROVEMENTS.md` instead.

---

## 2026-09-19 — stopped the vanilla run to pursue a PRISM-inspired loss change

The `pusht_vanilla_imle` run (`comfy-pine-1_...` in `saved_weights/`) was killed at **epoch 313**
(32.84h elapsed) to free the GPU for a modeling change rather than let the remaining ~187 epochs
(~19-20h) finish, because a subagent-run diagnosis (see the conversation, not repeated here)
concluded the ~45-50% success-rate plateau since epoch ~270 is a genuine capacity/diversity
ceiling of the current config, not something more epochs would fix: training loss flattened at
0.043-0.045 for ~65 epochs, the cosine LR schedule still had ~62% of its budget left (ruling out
"just needs to finish annealing"), and the rs-IMLE `min_distance` diagnostic had been below
`epsilon` since ~epoch 90 while `max_distance` stopped shrinking — consistent with candidates
cheaply satisfying their own conditioning target without being pushed toward genuinely diverse
coverage.

**Preserved as the pre-change baseline** (epoch 313, best recorded eval was epoch 289 at 48%
success / 0.776 mean reward, from `progress_log.csv`):
- `saved_weights/comfy-pine-1_..._pusht/net_weights_baseline_pre_prism_epoch313.pth`
- `saved_weights/comfy-pine-1_..._pusht/ema_net_weights_baseline_pre_prism_epoch313.pth`
- (the full resumable `most_recent_checkpoint.pth`/`checkpoint_epoch313.pth` from that run are
  also still there if this exact run ever needs to be resumed instead of restarted)

Next step, tracked in `IMPROVEMENTS.md`: implemented the batch-global rejection rs-IMLE loss
from PRISM (arXiv:2602.02396) as an opt-in alternative (`--use_batch_global_rejection`,
`imle_policy/utils/losses.py:rs_imle_loss_batch_global`), aimed directly at that diversity
ceiling. Original `rs_imle_loss` untouched so the two remain comparable.

---

## 2026-09-18 (follow-up) — repeated interruptions, a likely power-delivery issue, and a lighter checkpoint interval

Ran into a string of unplanned interruptions on this same run: two more hard reboots (10:38,
10:49) beyond the one already logged below, each recovered the same way (verify GPU back via
`nvidia-smi`, `--resume_from latest_checkpoint.pth`). Investigated the cause via forked
subagents rather than blocking the main session on it:

- Ruled out every software cause across all the reboot logs: unattended-upgrades auto-reboot is
  disabled, no OOM-killer entries, no thermal throttle/shutdown entries, no crash traceback (the
  training log just cuts off mid-batch each time — the signature of a hard power cut, not a
  graceful shutdown or software crash).
- `upower`/sysfs checks showed AC genuinely connected but the battery stuck oscillating between
  `discharging`/`pending-charge` rather than charging steadily, *while the GPU was at 99%
  utilization drawing 34-48W*. Swapping the power adapter didn't fix it. Most likely explanation:
  the adapter can't sustain the combined draw of the running GPU/CPU load plus charging at the
  same time (this ProBook's GPU configs often want a 65W adapter) — or, less likely given the
  swap didn't help, a DC-jack/charging-circuit fault on the laptop itself. Not something fixable
  in software; flagged to the user as a physical/hardware issue to investigate separately.
- Net effect on training: each interruption cost only the time to notice + resume (checkpoint
  system worked exactly as designed), but repeated interruptions meant repeated small manual
  resumes.

### Added a lighter, more frequent checkpoint tier

The 6-hour full checkpoint (which also runs eval) was the only thing protecting progress —
pausing training to move the laptop between full checkpoints would have lost up to ~5 hours of
training with no way to save on demand. Added `--light_checkpoint_interval_minutes` (default 20)
to `imle_policy/train.py`: between full checkpoints, just the resumable state gets saved (no
eval, no new snapshot file — only overwrites `most_recent_checkpoint.pth`), bounding potential
loss from an interruption to minutes instead of hours. Checked at the same epoch-boundary point
as the full checkpoint (epochs are ~6 minutes at the current config, so this is already fine
granularity without adding mid-epoch checkpointing complexity); if a full checkpoint fires in the
same epoch, the lightweight save is skipped for that epoch since the full one already covers it.
Smoke-tested: lightweight saves fired at epochs 0/1/2 (no eval), full checkpoint+eval correctly
fired on the final epoch.

Training was paused (clean `kill`, not another hard interruption) at epoch 278 so the laptop
could be moved; resume with `--resume_from latest_checkpoint.pth` when ready.

## 2026-09-18 — first real run: progress, an unplanned reboot, and a resume-logging fix

Launched the actual training run (`pusht_vanilla_imle`, full dataset, all adopted defaults) on
2026-09-17 at 09:31. Progress so far, straight from `progress_log.csv`:

| Epoch | Elapsed | Success rate | Mean reward | Notes |
|---|---|---|---|---|
| 57 | 6.05h | 48% | 0.788 | New best |
| 115 | 12.12h | 40% | 0.744 | Regression — `latest_checkpoint.pth` correctly stayed at epoch 57 |
| 173 | 18.18h | 48% (tie) | 0.786 | Tie refreshed `latest_checkpoint.pth` to epoch 173 |
| 231 | 24.25h | 48% (tie) | 0.728 | Tie refreshed again, to epoch 231 |

48% success after ~24h (11-46% of the 500-epoch budget) with a 7.5M-param net is a solid signal
the GPU/config changes didn't just make training faster, they still produce a working policy.
Plateauing around 48% rather than climbing further is worth watching over the next few
checkpoints.

### The machine rebooted mid-run — exactly the scenario checkpointing was built for

At 2026-09-18 10:04, the machine rebooted (unrelated to training — confirmed via `journalctl`
showing a fresh NVIDIA module load and gnome-shell restart, not an OOM kill or a crash in our
code) while training was mid-epoch-234, killing the `nohup`'d process. Recovered in under a
minute: verified the GPU came back (`nvidia-smi`, `torch.cuda.is_available()`), then
`python train.py --resume_from .../latest_checkpoint.pth`, which picked back up at epoch 232
(the checkpoint saved at epoch 231) with no manual state surgery needed.

### Found (and fixed) a second wandb resume gap

The earlier `train_step + 1` fix (see below) only handles a *clean* shutdown, where the last
thing logged was exactly the saved checkpoint's step. An abrupt kill is different: training kept
logging every batch between the last on-disk checkpoint (epoch 231, step 93033) and the actual
moment it died (mid-epoch 234, step ~94025) — none of that reached a checkpoint file, but all of
it already reached wandb. Resuming with `checkpoint_step + 1` therefore collided with wandb's
already-advanced step counter by ~993, not 1, and would have spammed a "step less than current
step" warning (silently dropping the log call) for roughly 993 batches until `train_step` caught
up naturally.

Fixed by querying the actual server-side last-committed step on resume
(`wandb.Api().run(...).summary['_step']`) and starting from `max(checkpoint_step, that) + 1`,
falling back to the old `+1` behavior if the API call fails (e.g. offline). Verified: restarted
the resume with the fix, `train_step` correctly picked up at 94026 (last committed 94025 + 1),
zero warnings.

---

## 2026-09-17 (follow-up) — best-as-latest checkpoint semantics, eval/GUI smoketest

Refinement of the checkpoint/resume system from earlier today, plus an end-to-end verification
pass before kicking off the first real multi-hour run.

### Checkpoint naming changed: `latest_checkpoint.pth` now means "best", not "most recent"

Original design (see below) made `latest_checkpoint.pth` an unconditional overwrite of whatever
the model looked like at that checkpoint tick. Reconsidered: since `rs_imle` training can be
noisy (small model, small `n_samples_per_condition`), a checkpoint at hour 12 could plausibly
score worse on eval than the one at hour 6. Defaulting `--resume_from` to "whatever happened most
recently" risks resuming right after a bad patch. Changed to two separate files per run:

- **`latest_checkpoint.pth`** — the full resumable state as of the best-scoring eval seen so far
  (ties refresh it too, via `mean_success >= best_resume_success`, so it isn't stuck on a stale
  checkpoint once performance plateaus at the same score for a while — common early in training
  when success rate sits at 0.0). This is what `--resume_from` should normally point at.
- **`most_recent_checkpoint.pth`** — unconditionally overwritten every interval, the strict
  "continue with zero rollback" option, for when you don't want the possible redo.

Trade-off worth knowing: resuming from `latest_checkpoint.pth` can redo up to one checkpoint
interval's worth of training (its epoch/step counters roll back to whenever that best checkpoint
was actually written) if the most recent interval had regressed. Considered acceptable — one
interval's compute vs. protection against building on top of a regression — but if that's ever
not the trade-off wanted, `most_recent_checkpoint.pth` remains available.

`checkpoint_epoch<N>.pth` (kept forever, one per interval) is unaffected by this — still written
unconditionally.

### Eval-time GUI + reduced eval count, for smoketesting

Added `--render_eval` (off by default) and `--num_trails` (overrides the config's 50) to
`train.py`, threaded through to `imle_policy/evaluation/eval_policy_pusht.py`'s existing (but
previously always-commented-out) `env.render(mode="human")` call. PushT's environment
(`imle_policy/envs/pusht_env.py`) is pymunk/pygame-based, not mujoco/gazebo — no extra
simulator install was needed; pygame/pymunk were already present in the `imle_policy` conda env.

### Smoketest performed before the first real run

Ran `train.py` end-to-end online (real wandb login, not `WANDB_MODE=offline`) with
`--dataset_percentage 0.05 --num_epochs 2 --checkpoint_interval_hours 0.0003 --num_trails 2
--render_eval`:
- Confirmed via the wandb API (`run.history(keys=['loss'])`) that every training step's loss
  actually lands in the run's history, not just the local tqdm bar.
- `env.render(mode="human")` ran without error under `--render_eval` (this machine has an active
  Wayland session at `DISPLAY=:0`; verified pygame can open a window here before the full test).
- Found and fixed a real bug in the resume path along the way: `EMAModel`/`Optimizer.load_state_dict`
  don't move restored tensors back onto `device` after a `map_location='cpu'` checkpoint load,
  which crashed `ema.step()` with a cuda/cpu tensor mismatch on the first resumed step. Now
  explicitly moved after loading.
- Found and fixed a second bug: resuming a `resume='allow'` wandb run reused the exact step
  number the previous process had already logged at (the eval-metrics log right before the
  checkpoint was written), causing wandb to silently drop the first few post-resume log calls
  with a "step less than current step" warning. Fixed by starting the resumed `train_step` one
  past the checkpoint's saved value.
- Verified twice more that resume continues the same wandb run and the loss trajectory stays
  continuous across the process restart (no reset, no warnings) after each fix.
- Deleted the smoketest's throwaway wandb run and local `saved_weights/` output afterward.

## 2026-09-17 — GPU enablement, right-sized policy net, and time-based checkpoint/resume

### The problem

`python train.py --task pusht` was taking **~2.7–3.5 hours per epoch** (500 epochs configured).
Profiled the actual training step against the real `datasets/pusht.pkl` data to find out why:

| Stage | Time/step (CPU, original config) |
|---|---|
| Data loading | ~5ms — non-issue, 11 dataloader workers kept up easily |
| Vision encoder (ResNet18) | ~1.4-1.6s |
| Policy net forward | ~10-12s |
| Backward pass | ~36-49s |
| Optimizer step | ~0.5-0.8s |
| **Total** | **~48-63s/step** |

Two root causes:
1. **GPU was unusable.** `nvidia-smi` failed — the installed NVIDIA kernel module
   (`linux-modules-nvidia-595-open-6.8.0-124-generic`) was built for a different kernel than the
   one actually running (`6.8.0-138-generic`), a leftover from a kernel upgrade. `torch.cuda.is_available()`
   was `False`, so everything ran on CPU.
2. **The `rs_imle` training step is inherently heavy.** `train_rs_imle_step` runs
   `batch_size × n_samples_per_condition` = 128×20 = **2,560 samples** through the policy network
   every step, to estimate the IMLE loss over 20 candidate trajectories per condition. With the
   original 75M-parameter U-Net (`down_dims=[256,512,1024]`), that's the entire cost — data
   loading and the vision encoder are negligible by comparison.

### Fix 1: GPU driver (system-level, not part of this repo)

```
sudo apt install linux-modules-nvidia-595-open-6.8.0-138-generic
sudo reboot
```
Confirmed after reboot: `nvidia-smi` works, `torch.cuda.is_available() == True` (NVIDIA GeForce MX450).

### Discovery: the original config OOMs on this GPU regardless of batch size

The MX450 only exposes ~1.76GB usable VRAM to PyTorch. The original 75M-param policy net's
*static* footprint — parameters + gradients + AdamW's two momentum buffers per parameter — is
already **~1.2-1.5GB before any activations are computed**. Verified with an isolated benchmark
that this is a model-size problem, not a batch-size problem: `batch_size=8, n_samples_per_condition=1`
still OOMs with `down_dims=[256,512,1024]`. Shrinking the network is the only way to fit a normal
forward+backward pass on this card.

### Fix 2: shrink the policy network + reduce batch size (adopted as new defaults)

Benchmarked candidates (full pipeline: vision encoder + policy net + backward + optimizer step,
on the real dataset):

| Config | down_dims | batch_size | n_samples_per_condition | Result | Time/step | Peak VRAM |
|---|---|---|---|---|---|---|
| Original | `[256,512,1024]` (75M params) | 128 | 20 | **OOM on GPU** | ~48-63s (CPU) | n/a |
| **Adopted** | **`[64,128,256]` (7.5M params)** | **64** | **20 (unchanged)** | OK | **~0.91s** | 1.31GB |
| Alternative | `[128,256,512]` (22.6M params) | 64 | 8 | OK | ~0.86s | 1.41GB |

Went with the "Adopted" row: it keeps `n_samples_per_condition=20` (the original IMLE loss's
sample diversity) and only shrinks the network's channel width. PushT is a low-dimensional task
(2D agent position, 2D action) so a 75M-parameter U-Net was arguably oversized for it anyway.
`batch_size=128` doesn't fit on this GPU with any `down_dims` we tried — 64 is the practical
ceiling.

**Net effect:** ~55-70x faster per step than the CPU baseline. Epoch time drops from
~2.7-3.5 hours to **~6 minutes** (400 steps/epoch at batch_size=64). 500 epochs goes from
~62 days to roughly 2 days of wall-clock time.

Code changes (originals kept, commented out, right next to the replacement):
- `imle_policy/models/rs_imle_network.py`: `GeneratorConditionalUnet1D`'s default `down_dims`
  changed from `[256, 512, 1024]` to `[64, 128, 256]`.
- `imle_policy/configs/pusht_config.json`: `batch_size` changed from 128 to 64 (128 kept
  alongside as `batch_size_original`).
- `imle_policy/train.py`: added `--down_dims` and `--batch_size` CLI overrides so either can be
  changed per-run without editing code (useful once we start experimenting with the policy
  architecture itself).

### Fix 3: eval crashed when CUDA wasn't available

`imle_policy/evaluation/eval_policy_pusht.py` built its device from `args['device']`, which is
hardcoded to `"cuda"` in `pusht_config.json`. The training loop's device already had a CPU
fallback (`train.py`), but eval — triggered periodically via `save_checkpoint` — did not, and
would have crashed the first time it ran on a machine without CUDA. Now mirrors the same
fallback: `torch.device(args['device'] if torch.cuda.is_available() else 'cpu')`.

### Feature: time-based checkpointing + resume

The old cadence was `if epoch_idx % 50 == 0`, which is unpredictable in wall-clock terms once
epoch speed changes (as it just did, by ~150x). Replaced with a wall-clock interval so long runs
get predictable checkpoints regardless of epoch speed. What changed in `imle_policy/train.py`:

- **`--checkpoint_interval_hours`** (default `6.0`): at least once per interval (and always on
  the last epoch), training now saves:
  - `saved_weights/<run>/net_weights_epoch<N>.pth` / `ema_net_weights_epoch<N>.pth` — a
    permanent, individually-testable snapshot for that checkpoint (previously only the
    best-eval weights were kept for the `pusht` task; every checkpoint is now kept).
  - `saved_weights/<run>/best_net_weights.pth` / `best_ema_net_weights.pth` — unchanged
    behavior, updated only when eval success rate improves.
  - `saved_weights/<run>/checkpoint_epoch<N>.pth` and `.../latest_checkpoint.pth` — full
    resumable state: model, EMA, optimizer, LR scheduler, epoch/step counters, RNG state,
    wandb run id, and the exact args used for the run.
  - `saved_weights/<run>/progress_log.csv` — one row per checkpoint (epoch, elapsed hours,
    mean_max_reward, mean_success_rate, whether it was a new best) — for tracking model
    progress over a multi-day run without digging through wandb.
- **`--resume_from <path>`**: resumes training from a `checkpoint_epoch<N>.pth` or
  `latest_checkpoint.pth` file. All model/data-defining settings (task, method, batch_size,
  down_dims, epsilon, n_samples_per_condition, ...) are restored **from the checkpoint**, not
  from the command line, so the reloaded optimizer/EMA/model state stays valid. Only
  `--num_epochs` and `--checkpoint_interval_hours` can be changed at resume time (e.g. to extend
  a run past its original epoch budget).
- wandb logging continues in the same run (same charts) across a resume, via the `wandb_run_id`
  stored in the checkpoint.
- Smoke-tested: ran 3 epochs, resumed from `latest_checkpoint.pth` for 2 more, confirmed the
  loss trajectory was continuous across the restart (no reset) and `progress_log.csv` appended
  rather than overwrote.

**How to resume a run:**
```
python train.py --resume_from saved_weights/<run_name>/latest_checkpoint.pth
```
**How to test how a specific checkpoint performs** (e.g. inspect the 6-hour mark specifically):
weights are at `saved_weights/<run_name>/ema_net_weights_epoch<N>.pth` — load them the same way
`evaluate()` in `imle_policy/evaluation/eval_policy_pusht.py` already does, or check
`progress_log.csv` for the eval numbers that were already computed for that checkpoint.

### Known follow-ups (not yet applied)

- `num_workers=11` in the dataloader is more than this machine's 8 CPU cores. It wasn't a
  bottleneck while the policy net dominated step time, but worth revisiting now that GPU steps
  are sub-second and data loading could start to matter.
- Only `GeneratorConditionalUnet1D` (the `rs_imle` policy) was resized. `ConditionalUnet1D`
  (used by `diffusion`/`flow_matching`) wasn't profiled or touched.
