# Colab workflow: parallel GPU experiments + bringing results back locally

Goal: run experiments on Colab's larger-VRAM GPU (T4: 15GB, A100: 40GB) *in parallel* with the
local MX450 (2GB) runs, using the exact same `train.py` code, so results are directly
comparable -- not a separate training pipeline, just a different place to run the same one.

## Notebooks

- `pusht_colab_train.ipynb` -- PushT, the task all local benchmarking/experiments have used.
- `lift_colab_train.ipynb` -- the Lift task (single-arm pick-and-lift, robosuite-based), used
  to check whether PushT's ~50-58% success-rate ceiling is task-specific or a broader property
  of this method. Same workflow shape, but: different dependencies (`robosuite`/`mujoco`
  instead of `pymunk`/`pygame`), and a different dataset-acquisition step (see below) since
  Lift has no standalone dataset zip the way PushT does. No per-epoch timing has been measured
  for this task on any GPU yet -- the notebook runs a short calibration before committing to
  the full 500 epochs, rather than guessing.
- To add another task (`PickPlaceCan`, `ToolHang`, `TwoArmTransport`, `NutAssemblySquare`,
  `kitchen`, `ur3_blockpush`): copy `lift_colab_train.ipynb` and change the task name, the
  dataset filename search pattern in the dataset cell, and the dependency list in the install
  cell (`kitchen`/`ur3_blockpush` additionally need `d4rl`/`dm_control`, per `pyproject.toml`
  -- not yet set up in any notebook here).

## Dataset availability (checked via the HuggingFace API, not assumed)

The project's dataset repo (`krishanrana/imle_policy` on HuggingFace) has exactly two files:
`pusht_dataset/datasets.zip` (PushT only, small) and `datasets.zip` (**25.8GB, every task
bundled together** -- confirmed via `huggingface.co/api/datasets/krishanrana/imle_policy`,
there is no per-task zip for anything other than PushT). For any non-PushT task, the dataset
cell downloads the full 25.8GB zip to the Colab VM's *local* disk (not Drive -- avoids eating a
free Google account's Drive quota on a temporary file), extracts only the matching task's
`.pkl` via `zipfile` (without fully unpacking the other tasks), saves that one file to Drive,
then deletes the local zip. This is a one-time cost per task (~10-25 min depending on Colab's
network that session) -- once a task's `.pkl` is on Drive, every future session reuses it
exactly like `pusht.pkl` already does.

## What to upload / where things live

| Thing | Where it lives | Why |
|---|---|---|
| Code (`train.py`, losses, models, configs) | GitHub (your fork: `santhoshetty/imle_policy`) | Same code path locally and on Colab -- `git clone` in the notebook, never hand-copy files |
| `datasets/pusht.pkl` (~710MB) | Google Drive (`MyDrive/imle_policy_colab/datasets/`) | Gitignored, too large for git; upload once, every future Colab session reuses it |
| Checkpoints (`saved_weights/`) | Google Drive (`MyDrive/imle_policy_colab/saved_weights/`), symlinked into the Colab VM | Colab VMs are ephemeral -- anything only on the VM disk is lost when the runtime disconnects/recycles, which *will* happen on a multi-hour run |
| wandb logs | wandb.ai, same account/project family as local runs | Colab and local runs show up side-by-side automatically, no extra plumbing |

## Steps

1. Open `colab/pusht_colab_train.ipynb` in Colab: either upload it directly (colab.research.google.com -> File -> Upload notebook), or File -> Open notebook -> GitHub -> paste your fork's URL and pick the file from the `colab/` folder.
2. Runtime -> Change runtime type -> GPU (T4 is fine to start).
3. Run cells 1-3 (mount Drive, clone your fork, install dependencies).
4. **One-time**: upload `pusht.pkl` to `MyDrive/imle_policy_colab/datasets/pusht.pkl` via the Drive web UI (drag-and-drop works fine for a single 710MB file), then run cell 4. Every future session just reuses this — you never need to re-upload it.
5. Run cell 5 (`wandb.login()`), paste your API key from wandb.ai/authorize.
6. Run cell 6's symlink cell, then the training cell. **This is the parameterized part**: edit the `--down_dims`/`--batch_size`/`--use_batch_global_rejection`/etc. flags to run whatever experiment you want next — it's the same CLI as local `train.py`, so anything documented in `IMPROVEMENTS.md`/`CHANGES.md` for local runs applies here too. The notebook's default runs the **original full-size architecture** (`down_dims=256,512,1024`, `batch_size=128`) that doesn't fit on the local GPU — that's the point of running it here.
7. If the runtime disconnects (free-tier Colab caps sessions, and this repo's own runs have taken 30+ hours locally — expect the same or longer here depending on GPU tier and idle limits): reopen the notebook, re-run cells 1-4 and the symlink cell in cell 6, then resume with `--resume_from saved_weights/<run_name>/latest_checkpoint.pth` — same resume mechanism as local, because it's the same code path. Consider Colab Pro/Pro+ if disconnects are frequent; free-tier idle/session limits are the main practical constraint on a run this long, not the GPU itself.

### Gotcha: resuming after "Interrupt execution" (not a full runtime restart) can fail with `run ID ... is in use`

Interrupting a cell (Runtime → Interrupt execution) kills the foreground `python train.py`
process but leaves wandb's background `wandb-core` service daemon running in the same Colab VM,
still holding the interrupted run as "active". The next `train.py --resume_from ...` reuses that
lingering service (you'll see `wandb: Using an existing wandb-core service via WANDB_SERVICE` in
the log) and collides with the run ID it's trying to reattach to:
`wandb.sdk.mailbox.mailbox_handle.ServerResponseError: run ID <id> is in use`.

Fix: in a cell, before retrying the resume command:
```python
!pkill -9 -f wandb-core || true
import os
os.environ.pop("WANDB_SERVICE", None)
```
This forces a fresh wandb service instead of reattaching to the dead one. If it still fails
(rare — would mean wandb's server hasn't yet timed out the old run's heartbeat), wait 1-2
minutes and retry, or as a last resort fully restart the runtime (Runtime → Restart session,
not just Interrupt) and re-run the setup cells — a full restart kills every background process,
not just wandb's, guaranteeing a clean slate. A plain Colab disconnect/reconnect (step 7 above)
doesn't hit this, since the whole VM and its background processes go away together.

## Bringing a trained checkpoint back to the local machine

Nothing to export specially — the symlink in step 6 means Drive already has every checkpoint
the moment it's written. From the local machine, either:
- Install the Google Drive desktop app and let it sync `MyDrive/imle_policy_colab/saved_weights/` down, or
- `rclone copy 'gdrive:imle_policy_colab/saved_weights/<run_name>' ./imle_policy/saved_weights/<run_name>` (set up an rclone Google Drive remote once), or
- Download just the files you need from the Drive web UI: for local eval you only need `stats.pth` and one of `ema_net_weights_epoch<N>.pth` / `net_weights_epoch<N>.pth` / `best_ema_net_weights.pth` — not the full resumable `checkpoint_epoch<N>.pth` (that one also bundles optimizer state, ~4x larger, and you only need it if you intend to keep training that exact run, not just eval it).

## Running a Colab-trained (bigger) model's eval/PushT GUI locally

**The local GPU's 2GB limit is a *training*-time constraint, not an eval-time one, and this
matters a lot here.** Training needs VRAM for parameters + gradients + AdamW's two momentum
buffers per parameter, multiplied by a batch of `batch_size * n_samples_per_condition` samples
being pushed through the network at once (that's what forced `down_dims` and `batch_size` down
for local training in the first place — see `CHANGES.md`). Eval (`eval_policy_pusht.py`) does
none of that: no gradients, no optimizer state, and it runs the policy net on a **single**
sample at a time (`noise = torch.randn((1, pred_horizon, action_dim))` — see
`eval_policy_pusht.py`, the rs-IMLE inference branch). Even the *original* 75M-parameter
`down_dims=[256,512,1024]` network only needs ~300MB just to hold its weights in fp32, plus
small per-step activations for a batch of 1 — well within the local MX450's budget.

**Caveat proven in practice**: if a local training run is *also* active on the same GPU at the
time (a realistic scenario — that's the whole point of running Colab experiments in parallel
with local ones), the local GPU's 2GB can genuinely be saturated by that other process even
though eval alone would easily fit. `eval_colab_checkpoint.py` already falls back to CPU
automatically in that case (same `torch.device('cuda' if torch.cuda.is_available() else
'cpu')` pattern used everywhere else in this repo) — confirmed working (100% success on a
quick 2-trial check) while a local capacity-experiment training run was actively using 1.53GB
of the 2GB budget. No flag needed for this, it just happens.

**So: try loading the Colab-trained checkpoint locally as-is first, no quantization.** Use
`imle_policy/evaluation/eval_policy_pusht.py::evaluate()` (or `train.py`'s own periodic eval
path — the same function) exactly as done earlier in this project's local smoketests, just
construct the network with whatever `down_dims` the Colab run actually used, e.g.:

```python
policy_net = GeneratorConditionalUnet1D(
    input_dim=args['action_dim'],
    global_cond_dim=args['obs_dim'] * args['obs_horizon'],
    down_dims=[256, 512, 1024],  # match whatever the Colab run trained with
)
```

then `nets.load_state_dict(...)`, `nets.to(device)` (`device = torch.device('cuda' if
torch.cuda.is_available() else 'cpu')` — CPU eval already proven to work fine in this project,
with `--render_eval`/GUI rendering, at roughly real-time speed), and run `evaluate()` /
`train.py --render_eval --num_trails <small number>` as usual. Expect this to just work.

A ready-to-use script for this is included: `imle_policy/evaluation/eval_colab_checkpoint.py`.

```
python -m imle_policy.evaluation.eval_colab_checkpoint \
    --run_dir imle_policy/saved_weights/pusht_colab_original_capacity_.../ \
    --down_dims 256,512,1024 \
    --weights best_ema_net_weights.pth \
    --num_trails 10 --render_eval
```

**If it somehow doesn't fit** (unlikely for eval, but in case VRAM is tighter than expected due
to other GPU usage at the time): fp16 (`nets.half()`) would halve the weight memory (~150MB for
the original architecture) at no real accuracy cost for inference-only use — but note
`eval_policy_pusht.py` currently hardcodes its input tensors to `float32` regardless of model
dtype (`nimages = ....to(device, dtype=torch.float32)`), so a `.half()` model would crash on a
dtype mismatch on the first forward pass as-is; that file's dtype casts would need updating to
match the model first. `eval_colab_checkpoint.py`'s `--half` flag raises a clear error rather
than attempting this half-implemented path. Given fp32 already fits comfortably (see the
napkin math above), this almost certainly isn't worth doing — flagged here so it's not a
surprise if VRAM ever is tight, not as a recommendation to build it.
