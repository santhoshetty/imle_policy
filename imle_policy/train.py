from torch import nn
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import wandb
import copy
import csv
import os
import sys
import time
import argparse
import json
import logging

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imle_policy.models.rs_imle_network import GeneratorConditionalUnet1D
from imle_policy.models.diffusion_network import ConditionalUnet1D
from imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from imle_policy.utils.losses import rs_imle_loss, rs_imle_loss_batch_global

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Args that are safe to change when resuming a run: they don't affect model shape or optimizer
# state, so overriding them can't break a loaded checkpoint. Everything else in a resumed run's
# config is taken verbatim from the checkpoint (see main()).
RESUME_SAFE_OVERRIDE_KEYS = ['num_epochs', 'checkpoint_interval_hours', 'light_checkpoint_interval_minutes', 'num_trails', 'render_eval']
# Same idea for a fresh (non-resume) run: these must win over the task config file if passed on
# the command line, rather than being silently clobbered by a plain dict.update(task_config).
FRESH_RUN_CLI_OVERRIDE_KEYS = ['num_epochs', 'batch_size', 'num_trails', 'render_eval']

def parse_args():
    parser = argparse.ArgumentParser(description='Train Policy')
    parser.add_argument('--method', type=str, default='rs_imle',
                      help='Training method (rs_imle, diffusion, flow_matching)')
    parser.add_argument('--epsilon', type=float, default=0.03,
                      help='IMLE epsilon parameter')
    parser.add_argument('--n_samples_per_condition', type=int, default=20,
                      help='Number of samples per condition')
    parser.add_argument('--use_batch_global_rejection', action='store_true', default=False,
                      help='Use the batch-global rejection rs-IMLE loss (PRISM-inspired, see '
                           'IMPROVEMENTS.md and imle_policy/utils/losses.py:rs_imle_loss_batch_global) '
                           'instead of the original per-sample rs_imle_loss. Off by default so '
                           'existing runs/behavior are unaffected.')
    parser.add_argument('--dataset_percentage', type=float, default=1.0,
                      help='Percentage of dataset to use')
    parser.add_argument('--task', type=str, default='pusht',
                      help='Task to train on (pusht, Lift, PickPlaceCan, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--use_traj_consistency', type=bool, default=False,
                      help='Use trajectory consistency')
    parser.add_argument('--wandb_run_name', type=str, default='testing',
                      help='Wandb run name')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Override the batch_size from the task config')
    parser.add_argument('--down_dims', type=str, default=None,
                      help='Comma-separated U-Net channel widths, e.g. "64,128,256". '
                           'Overrides the policy network default (ignored on --resume_from).')
    parser.add_argument('--num_epochs', type=int, default=None,
                      help='Override num_epochs from the task config (also usable to extend '
                           'training length when resuming)')
    parser.add_argument('--checkpoint_interval_hours', type=float, default=6.0,
                      help='Save a resumable checkpoint + run eval at least this often, in hours')
    parser.add_argument('--light_checkpoint_interval_minutes', type=float, default=20.0,
                      help='Between the (expensive, eval-running) checkpoints above, also save '
                           'just the resumable state (no eval, no new snapshot files -- only '
                           'overwrites most_recent_checkpoint.pth) at least this often, in '
                           'minutes. Bounds how much gets lost if training is interrupted '
                           '(power loss, closing the laptop, etc.) between full checkpoints.')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to a checkpoint (e.g. saved_weights/<run>/latest_checkpoint.pth) '
                           'to resume training from. All model/data-defining settings are restored '
                           'from the checkpoint; only RESUME_SAFE_OVERRIDE_KEYS can be changed.')
    parser.add_argument('--num_trails', type=int, default=None,
                      help='Override num_trails (eval episode count) from the task config, e.g. '
                           'to run a quick 2-episode eval smoketest instead of the full 50.')
    parser.add_argument('--render_eval', action='store_true', default=None,
                      help='Pop up the environment\'s live GUI window during eval (PushT uses '
                           'pygame/pymunk, not mujoco/gazebo). Off by default since it needs a '
                           'display and slows eval down; meant for manual smoketests, not the '
                           'unattended long run.')
    return parser.parse_args()

def load_config(task):
    # Get the directory where train_policy.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, 'configs', f'{task}_config.json')

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}. Please make sure the config file exists in rs_imle_policy/configs/ directory.")

    with open(config_file_path, 'r') as file:
        return json.load(file)

def setup_wandb(args_dict):
    wandb.init(project=args_dict['wandb_run_name'])

    if wandb.run.name is None: # if user does not login to wandb
        wandb.run.name = args_dict['wandb_run_name']

    if args_dict['method'] == 'diffusion':
        run_name = wandb.run.name + "_" + args_dict['method'] + f"_dataset_percentage_{args_dict['dataset_percentage']}_{args_dict['task']}"
    elif args_dict['method'] == 'rs_imle':
        run_name = wandb.run.name + "_" + args_dict['method'] + f"_eps_{args_dict['epsilon']}_;_n_samples_{args_dict['n_samples_per_condition']}__dataset_percentage_{args_dict['dataset_percentage']}_{args_dict['task']}"
    elif args_dict['method'] == 'flow_matching':
        run_name = wandb.run.name + "_" + args_dict['method'] + f"_num_flow_iters_{args_dict['num_flow_iters']}_dataset_percentage_{args_dict['dataset_percentage']}_{args_dict['task']}"

    wandb.run.name = run_name
    os.makedirs(f'saved_weights/{run_name}', exist_ok=True)
    wandb.config.update(args_dict)
    return run_name

def get_dataset_class(task):
    if task == 'pusht':
        from imle_policy.dataloaders.dataset_pusht import PolicyDataset
        from imle_policy.evaluation.eval_policy_pusht import evaluate
    elif task == 'ToolHang':
        from imle_policy.dataloaders.dataset_robomimic_hdf5 import PolicyDataset
        from imle_policy.evaluation.eval_policy_robomimic import evaluate
    elif task == 'TwoArmTransport':
        from imle_policy.dataloaders.dataset_robomimic_hdf5 import PolicyDataset
        from imle_policy.evaluation.eval_policy_robomimic_two_arm import evaluate
    elif task == 'ur3_blockpush':
        from imle_policy.dataloaders.dataset_ur3_blockpush import PolicyDataset
        from imle_policy.evaluation.eval_policy_ur3_blockpush import evaluate
    elif task == 'kitchen':
        from imle_policy.dataloaders.dataset_kitchen import PolicyDataset
        from imle_policy.evaluation.eval_policy_kitchen import evaluate
    else:
        from imle_policy.dataloaders.dataset_robomimic import PolicyDataset
        from imle_policy.evaluation.eval_policy_robomimic import evaluate
    return PolicyDataset, evaluate

def create_networks(args_dict):
    nets = nn.ModuleDict()

    if args_dict['task'] != "ur3_blockpush" or args_dict['task'] != "kitchen":
        for i in range(args_dict['num_cameras']):
            vision_encoder = get_resnet('resnet18')
            vision_encoder = replace_bn_with_gn(vision_encoder)
            nets[f'vision_encoder_{i}'] = vision_encoder

    if args_dict['method'] == 'diffusion':
        policy_net = ConditionalUnet1D(
            input_dim=args_dict['action_dim'],
            global_cond_dim=args_dict['obs_dim']*args_dict['obs_horizon'])

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args_dict['num_diffusion_iters'],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon')

    elif args_dict['method'] == 'rs_imle':
        # GeneratorConditionalUnet1D's own default down_dims is already reduced to fit small
        # GPUs (see rs_imle_network.py); --down_dims lets it be overridden per-run without
        # touching code, e.g. to try a bigger network on a machine with more VRAM.
        down_dims_kwargs = {}
        if args_dict.get('down_dims'):
            down_dims_kwargs['down_dims'] = [int(x) for x in args_dict['down_dims'].split(',')]
        policy_net = GeneratorConditionalUnet1D(
            input_dim=args_dict['action_dim'],
            global_cond_dim=args_dict['obs_dim']*args_dict['obs_horizon'],
            **down_dims_kwargs)

    elif args_dict['method'] == 'flow_matching':
        policy_net = ConditionalUnet1D(
            input_dim=args_dict['action_dim'],
            global_cond_dim=args_dict['obs_dim']*args_dict['obs_horizon'])

    nets['policy_net'] = policy_net
    return nets, noise_scheduler if args_dict['method'] == 'diffusion' else None

def process_batch(nbatch, nets, device, args_dict):
    if args_dict['task'] == 'pusht':
        # data normalized in dataset
        nimage = nbatch['image'][:,:args_dict['obs_horizon']].to(device)
        nagent_pos = nbatch['agent_pos'][:,:args_dict['obs_horizon']].to(device)

        # encoder vision features
        image_features = nets['vision_encoder_0'](
            nimage.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

    elif args_dict['task'] == 'ur3_blockpush':
        nobs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)
        obs_cond = nobs.flatten(start_dim=1)

    elif args_dict['task'] == 'kitchen':
        nobs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)
        obs_cond = nobs.flatten(start_dim=1)

    elif args_dict['task'] == 'TwoArmTransport':
        nimage_front_0 = nbatch['front_images_0'][:,:args_dict['obs_horizon']].to(device)
        nimage_hand_0 = nbatch['hand_images_0'][:,:args_dict['obs_horizon']].to(device)
        nimage_front_1 = nbatch['front_images_1'][:,:args_dict['obs_horizon']].to(device)
        nimage_hand_1 = nbatch['hand_images_1'][:,:args_dict['obs_horizon']].to(device)

        nagent_obs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)

        # encoder vision features front
        image_features_front_0 = nets['vision_encoder_0'](nimage_front_0.flatten(end_dim=1))
        image_features_front_0 = image_features_front_0.reshape(*nimage_front_0.shape[:2],-1)
        # encoder vision features hand
        image_features_hand_0 = nets['vision_encoder_1'](nimage_hand_0.flatten(end_dim=1))
        image_features_hand_0 = image_features_hand_0.reshape(*nimage_hand_0.shape[:2],-1)

        image_features_front_1 = nets['vision_encoder_2'](nimage_front_1.flatten(end_dim=1))
        image_features_front_1 = image_features_front_1.reshape(*nimage_front_1.shape[:2],-1)

        image_features_hand_1 = nets['vision_encoder_3'](nimage_hand_1.flatten(end_dim=1))
        image_features_hand_1 = image_features_hand_1.reshape(*nimage_hand_1.shape[:2],-1)

        image_features = torch.cat([image_features_front_0, image_features_hand_0, image_features_front_1, image_features_hand_1], dim=-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_obs], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
    else:
        nimage_front = nbatch['front_images'][:,:args_dict['obs_horizon']].to(device)
        nimage_hand = nbatch['hand_images'][:,:args_dict['obs_horizon']].to(device)
        nagent_obs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)

        # encoder vision features front
        image_features_front = nets['vision_encoder_0'](nimage_front.flatten(end_dim=1))
        image_features_front = image_features_front.reshape(*nimage_front.shape[:2],-1)

        # encoder vision features hand
        image_features_hand = nets['vision_encoder_1'](nimage_hand.flatten(end_dim=1))
        image_features_hand = image_features_hand.reshape(*nimage_hand.shape[:2],-1)

        image_features = torch.cat([image_features_front, image_features_hand], dim=-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_obs], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

    return obs_cond

def train_diffusion_step(nets, noise_scheduler, obs_cond, naction, B, device):
    # sample noise to add to actions
    noise = torch.randn(naction.shape, device=device)

    # sample a diffusion iteration for each data point
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B,), device=device
    ).long()

    noisy_actions = noise_scheduler.add_noise(
        naction, noise, timesteps)

    # predict the noise residual
    noise_pred = nets['policy_net'](
        noisy_actions, timesteps, global_cond=obs_cond)

    return nn.functional.mse_loss(noise_pred, noise)

def train_rs_imle_step(nets, obs_cond, naction, B, args_dict, device):
    noise = torch.randn(B * args_dict['n_samples_per_condition'], *naction.shape[1:], device=device)
    repeated_obs_cond = obs_cond.repeat_interleave(args_dict['n_samples_per_condition'], dim=0)

    pred_actions = nets['policy_net'](repeated_obs_cond, noise)
    pred_actions = pred_actions.reshape(B, args_dict['n_samples_per_condition'], *naction.shape[1:])

    # Compute IMLE loss. See IMPROVEMENTS.md and imle_policy/utils/losses.py for why the
    # batch-global variant exists and what it changes; --use_batch_global_rejection opts in
    # without touching the original per-sample behavior.
    if args_dict.get('use_batch_global_rejection'):
        return rs_imle_loss_batch_global(naction, pred_actions, args_dict['epsilon'])
    return rs_imle_loss(naction, pred_actions, args_dict['epsilon'])

def train_flow_matching_step(nets, obs_cond, naction, B, args_dict, device):
    noise = torch.randn(naction.shape, device=device)
    t = torch.rand(B, device=device)
    t_shaped = t.reshape(-1, *([1] * (noise.dim() - 1)))
    xt = t_shaped * naction + (1 - t_shaped) * noise
    vector = naction - noise
    timesteps = (t * args_dict['timestep_integer_scaler']).long()
    pred = nets['policy_net'](
        xt, timesteps, global_cond=obs_cond)

    return nn.functional.mse_loss(pred, vector)

def log_progress(run_name, epoch_idx, train_step, elapsed_seconds, mean_cov, mean_success, is_best):
    """Append one row per checkpoint to a per-run CSV, so model progress over a long run can be
    inspected without digging through wandb."""
    path = f'saved_weights/{run_name}/progress_log.csv'
    is_new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['epoch', 'train_step', 'elapsed_hours', 'mean_max_reward', 'mean_success_rate', 'is_best'])
        writer.writerow([epoch_idx, train_step, f"{elapsed_seconds/3600:.3f}", mean_cov, mean_success, is_best])

def save_checkpoint(args_dict, nets, ema, epoch_idx, best_mean_success, stats, run_name, train_step, evaluate_fn, elapsed_seconds):
    ema_nets = copy.deepcopy(nets)
    ema.copy_to(ema_nets.parameters())

    # Always keep a snapshot tagged with the epoch, not just the best-so-far one, so each
    # checkpoint interval leaves something that can be individually tested later.
    torch.save(nets.state_dict(), f'saved_weights/{run_name}/net_weights_epoch{epoch_idx}.pth')
    torch.save(ema_nets.state_dict(), f'saved_weights/{run_name}/ema_net_weights_epoch{epoch_idx}.pth')

    mean_success = None  # no eval signal at all for pusht_real/shoe_rack_real (no simulator to eval in)
    if (args_dict['task'] != 'pusht_real') and (args_dict['task'] != 'shoe_rack_real'):
        mean_cov, mean_success = evaluate_fn(args_dict, ema_nets, stats, method=args_dict['method'])

        is_best = mean_success > best_mean_success
        if is_best:
            best_mean_success = mean_success
            torch.save(nets.state_dict(), f'saved_weights/{run_name}/best_net_weights.pth')
            torch.save(ema_nets.state_dict(), f'saved_weights/{run_name}/best_ema_net_weights.pth')

        wandb.log({'mean_max_reward': mean_cov, 'mean_success_rate': mean_success}, step=train_step)
        log_progress(run_name, epoch_idx, train_step, elapsed_seconds, mean_cov, mean_success, is_best)

    return best_mean_success, mean_success

def build_resume_checkpoint(args_dict, nets, ema, optimizer, lr_scheduler, epoch_idx, train_step,
                             best_mean_success, best_resume_success, elapsed_seconds, run_name):
    """Full training state needed to continue training exactly where it left off."""
    return {
        'epoch_idx': epoch_idx,
        'train_step': train_step,
        'best_mean_success': best_mean_success,
        'best_resume_success': best_resume_success,
        'elapsed_seconds': elapsed_seconds,
        'nets_state_dict': nets.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'args_dict': args_dict,
        'run_name': run_name,
        'wandb_run_id': wandb.run.id if wandb.run is not None else None,
    }

def train(args_dict, nets, dataloader, device, noise_scheduler=None, stats=None, run_name=None,
          evaluate_fn=None, resume_checkpoint=None):
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * args_dict['num_epochs']
    )

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    train_step = 0
    start_epoch = 0
    best_mean_success = 0
    # Tracks the eval score of whatever is currently saved as latest_checkpoint.pth (the resumable
    # "best" pointer, see below). -inf so the very first checkpoint always seeds it, even if that
    # first eval's success rate happens to be 0.0 (common early in training).
    best_resume_success = -float('inf')
    elapsed_seconds_at_resume = 0.0

    if resume_checkpoint is not None:
        # Checkpoints are loaded with map_location='cpu' (see main()) so they can be inspected
        # on a machine without a GPU; load_state_dict on the optimizer/EMA does not itself move
        # those restored tensors back onto `device`, so do it explicitly or ema.step()/optimizer
        # .step() will crash mixing cuda and cpu tensors.
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler_state_dict'])
        ema.load_state_dict(resume_checkpoint['ema_state_dict'])
        ema.to(device)
        # +1: the saved train_step value was itself already used as a wandb log step (by the
        # eval-metrics log in save_checkpoint, right before this checkpoint was written), so
        # resuming at that same value collides with wandb's server-side step counter on a
        # resumed run ("Tried to log to step N that is less than the current step" and the log
        # call is dropped). Skipping the one already-used integer avoids it; nothing else depends
        # on train_step being contiguous, it's purely a wandb x-axis counter.
        train_step = resume_checkpoint['train_step'] + 1
        if wandb.run is not None:
            # The +1 above only covers a clean shutdown. An unclean one (crash, reboot, kill -9)
            # can leave wandb with steps logged well past the last on-disk checkpoint -- training
            # keeps logging every batch between checkpoints, so whatever ran after the last save
            # but before the process died is still sitting in wandb's history. Query the server
            # for the true last committed step so logging resumes immediately instead of getting
            # silently dropped (with a warning per call) until train_step catches back up.
            try:
                last_committed = wandb.Api().run(
                    f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
                ).summary.get('_step')
                if last_committed is not None and last_committed >= train_step:
                    train_step = last_committed + 1
            except Exception:
                pass  # best-effort; falls back to the +1 above
        start_epoch = resume_checkpoint['epoch_idx'] + 1
        best_mean_success = resume_checkpoint['best_mean_success']
        best_resume_success = resume_checkpoint.get('best_resume_success', -float('inf'))
        elapsed_seconds_at_resume = resume_checkpoint.get('elapsed_seconds', 0.0)
        torch.set_rng_state(resume_checkpoint['torch_rng_state'])
        np.random.set_state(resume_checkpoint['numpy_rng_state'])
        if torch.cuda.is_available() and resume_checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(resume_checkpoint['cuda_rng_state'])
        logger.info(f"Resuming from epoch {start_epoch}, train_step {train_step}, "
                    f"{elapsed_seconds_at_resume/3600:.1f}h of prior training")

    # Time-based checkpointing (replaces the old "every 50 epochs" cadence): saves a resumable
    # checkpoint + runs eval at least every `checkpoint_interval_hours`. The clock restarts on
    # each resume rather than trying to preserve exact cross-process cadence, which is fine since
    # this only needs to be "roughly every N hours", not exact.
    training_start_time = time.time() - elapsed_seconds_at_resume
    last_checkpoint_time = time.time()
    last_light_checkpoint_time = time.time()
    checkpoint_interval_seconds = args_dict['checkpoint_interval_hours'] * 3600
    light_checkpoint_interval_seconds = args_dict['light_checkpoint_interval_minutes'] * 60

    with tqdm(range(start_epoch, args_dict['num_epochs']), desc='Epoch',
              initial=start_epoch, total=args_dict['num_epochs']) as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            wandb.log({'epoch': epoch_idx}, step=train_step)

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    obs_cond = process_batch(nbatch, nets, device, args_dict)
                    naction = nbatch['action'].to(device)
                    B = naction.shape[0]

                    if args_dict['method'] == 'diffusion':
                        loss = train_diffusion_step(nets, noise_scheduler, obs_cond, naction, B, device)
                    elif args_dict['method'] == 'rs_imle':
                        loss, loss_logs = train_rs_imle_step(nets, obs_cond, naction, B, args_dict, device)
                        wandb.log(loss_logs, step=train_step)
                    elif args_dict['method'] == 'flow_matching':
                        loss = train_flow_matching_step(nets, obs_cond, naction, B, args_dict, device)

                    if loss == 0:
                        wandb.log({"zero_loss": 1}, step=train_step)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        ema.step(nets.parameters())
                        wandb.log({"zero_loss": 0}, step=train_step)

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    wandb.log({'loss': loss_cpu}, step=train_step)
                    train_step += 1

            # was: if (epoch_idx % 50 == 0):  -- replaced with a wall-clock interval so long runs
            # get checkpoints at a predictable real-time cadence regardless of epoch speed.
            is_last_epoch = (epoch_idx == args_dict['num_epochs'] - 1)
            if (time.time() - last_checkpoint_time) >= checkpoint_interval_seconds or is_last_epoch:
                elapsed_seconds = time.time() - training_start_time
                best_mean_success, mean_success = save_checkpoint(
                    args_dict, nets, ema, epoch_idx, best_mean_success,
                    stats, run_name, train_step, evaluate_fn, elapsed_seconds
                )

                # Decide (and finalize best_resume_success) *before* building the checkpoint dict,
                # so every file written this interval embeds the same, up-to-date value -- rather
                # than checkpoint_epoch<N>.pth/most_recent_checkpoint.pth embedding a stale value
                # that's one checkpoint behind latest_checkpoint.pth.
                #
                # latest_checkpoint.pth is what `--resume_from` should normally point at: it's
                # only updated when this checkpoint's eval is at least as good as the best one
                # seen so far, so resuming from it can't accidentally pick up training right after
                # a bad patch. Falls back to "always update" when there's no eval signal at all
                # (pusht_real/shoe_rack_real) or on the very first checkpoint. Note this means a
                # resume can redo up to one checkpoint interval's worth of training if the most
                # recent interval regressed -- see CHANGES.md.
                is_best_resume = (mean_success is None) or (mean_success >= best_resume_success)
                if is_best_resume and mean_success is not None:
                    best_resume_success = mean_success

                checkpoint = build_resume_checkpoint(
                    args_dict, nets, ema, optimizer, lr_scheduler, epoch_idx, train_step,
                    best_mean_success, best_resume_success, elapsed_seconds, run_name
                )
                # checkpoint_epoch<N>.pth: kept forever, one per interval, for full history.
                torch.save(checkpoint, f'saved_weights/{run_name}/checkpoint_epoch{epoch_idx}.pth')
                # most_recent_checkpoint.pth: the strict safety net -- always this interval's
                # actual state, so training can be continued with zero lost progress if needed.
                torch.save(checkpoint, f'saved_weights/{run_name}/most_recent_checkpoint.pth')
                if is_best_resume:
                    torch.save(checkpoint, f'saved_weights/{run_name}/latest_checkpoint.pth')

                logger.info(f"Saved checkpoint at epoch {epoch_idx} (train_step {train_step}, "
                            f"{elapsed_seconds/3600:.1f}h elapsed, "
                            f"{'updated' if is_best_resume else 'kept prior'} latest_checkpoint.pth)")
                last_checkpoint_time = time.time()
                last_light_checkpoint_time = time.time()  # the save above already covers this
            elif (time.time() - last_light_checkpoint_time) >= light_checkpoint_interval_seconds:
                # Between full checkpoints (which also run a ~50-episode eval and can be hours
                # apart), just save resumable state -- no eval, no new snapshot file, only
                # overwrites most_recent_checkpoint.pth -- so an interruption (closed laptop,
                # power loss) loses at most `light_checkpoint_interval_minutes`, not a whole
                # `checkpoint_interval_hours`.
                elapsed_seconds = time.time() - training_start_time
                checkpoint = build_resume_checkpoint(
                    args_dict, nets, ema, optimizer, lr_scheduler, epoch_idx, train_step,
                    best_mean_success, best_resume_success, elapsed_seconds, run_name
                )
                torch.save(checkpoint, f'saved_weights/{run_name}/most_recent_checkpoint.pth')
                logger.info(f"Saved lightweight resume checkpoint at epoch {epoch_idx} "
                            f"(train_step {train_step}, {elapsed_seconds/3600:.2f}h elapsed)")
                last_light_checkpoint_time = time.time()

            tglobal.set_postfix(loss=np.mean(epoch_loss))

def main():
    args = parse_args()
    args_dict = vars(args)

    resume_checkpoint = None
    if args_dict.get('resume_from'):
        logger.info(f"Loading checkpoint to resume: {args_dict['resume_from']}")
        resume_checkpoint = torch.load(args_dict['resume_from'], map_location='cpu', weights_only=False)
        run_name = resume_checkpoint['run_name']
        saved_args_dict = resume_checkpoint['args_dict']

        # Everything about the run (task, method, batch_size, down_dims, epsilon, ...) comes from
        # the checkpoint, so the reloaded model/optimizer/EMA state stays valid. Only a small,
        # shape-agnostic allowlist can be changed from the command line on a resume.
        cli_overrides = {k: args_dict[k] for k in RESUME_SAFE_OVERRIDE_KEYS
                          if args_dict.get(k) is not None}
        args_dict = dict(saved_args_dict)
        args_dict['resume_from'] = args.resume_from
        args_dict.update(cli_overrides)

        wandb.init(project=args_dict['wandb_run_name'], id=resume_checkpoint.get('wandb_run_id'),
                   resume='allow')
        wandb.run.name = run_name
        os.makedirs(f'saved_weights/{run_name}', exist_ok=True)
        wandb.config.update(args_dict, allow_val_change=True)
        logger.info(f"Resuming run '{run_name}' from epoch {resume_checkpoint['epoch_idx'] + 1}")
    else:
        # Load task-specific config, letting an explicit CLI flag win over the config file's value
        # for these keys (a plain dict.update would otherwise let the config silently overwrite
        # what was passed on the command line).
        cli_overrides = {k: args_dict[k] for k in FRESH_RUN_CLI_OVERRIDE_KEYS if args_dict.get(k) is not None}
        task_config = load_config(args.task)
        args_dict.update(task_config)
        args_dict.update(cli_overrides)

        # Setup wandb
        run_name = setup_wandb(args_dict)

    # Set random seeds
    np.random.seed(args_dict['seed'])
    torch.manual_seed(args_dict['seed'])

    # Get dataset class and evaluation function
    PolicyDataset, evaluate_fn = get_dataset_class(args_dict['task'])

    # Create dataset and dataloader
    dataset = PolicyDataset(
        dataset_path=args_dict['dataset_path'],
        pred_horizon=args_dict['pred_horizon'],
        obs_horizon=args_dict['obs_horizon'],
        action_horizon=args_dict['action_horizon'],
        dataset_percentage=args_dict['dataset_percentage']
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_dict['batch_size'],
        num_workers=11,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    # Save dataset stats
    stats = dataset.stats
    torch.save(stats, f'saved_weights/{run_name}/stats.pth')

    # Create networks
    # device = torch.device('cuda')  # original: crashed instead of falling back when CUDA wasn't available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets, noise_scheduler = create_networks(args_dict)
    nets = nets.to(device)

    if resume_checkpoint is not None:
        nets.load_state_dict(resume_checkpoint['nets_state_dict'])

    # Train
    train(args_dict, nets, dataloader, device, noise_scheduler, stats, run_name, evaluate_fn,
          resume_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    main()
