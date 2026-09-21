"""
Run local eval (optionally with the PushT GUI) on a checkpoint trained elsewhere (e.g. on
Colab with a larger down_dims than fits locally for training). See colab/COLAB_WORKFLOW.md for
why this works locally without quantization even when the checkpoint is the full-size
architecture: eval is a single-sample forward pass with no gradients/optimizer state, unlike
training, so the local GPU's 2GB training-time limit doesn't apply here.

Usage:
    python -m imle_policy.evaluation.eval_colab_checkpoint \\
        --run_dir saved_weights/pusht_colab_original_capacity_.../ \\
        --down_dims 256,512,1024 \\
        --weights ema_net_weights_epoch57.pth \\
        --num_trails 10 --render_eval --half
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from imle_policy.models.rs_imle_network import GeneratorConditionalUnet1D
from imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from imle_policy.evaluation.eval_policy_pusht import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Eval a checkpoint (e.g. from Colab) locally")
    parser.add_argument("--run_dir", type=str, required=True,
                         help="saved_weights/<run_name>/ directory containing the weights + stats.pth")
    parser.add_argument("--weights", type=str, default="best_ema_net_weights.pth",
                         help="Weights file inside --run_dir (e.g. ema_net_weights_epoch57.pth)")
    parser.add_argument("--down_dims", type=str, required=True,
                         help="Comma-separated U-Net channel widths the checkpoint was trained "
                              "with, e.g. '256,512,1024'. Must match exactly or load_state_dict fails.")
    parser.add_argument("--num_trails", type=int, default=10)
    parser.add_argument("--render_eval", action="store_true", default=False,
                         help="Show the live PushT GUI window (needs a display)")
    parser.add_argument("--half", action="store_true", default=False,
                         help="NOT CURRENTLY SUPPORTED: eval_policy_pusht.py hardcodes its "
                              "input tensors to float32, so a .half() model would crash on a "
                              "dtype mismatch on the first forward pass. Left as a flag (rather "
                              "than silently ignored) so this fails loudly with a clear message "
                              "instead of a confusing crash mid-eval. See colab/COLAB_WORKFLOW.md "
                              "-- eval fits comfortably in fp32 already, so this shouldn't be "
                              "needed; if it ever is, eval_policy_pusht.py's dtype casts need "
                              "updating to match the model's dtype first.")
    parser.add_argument("--use_traj_consistency", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "pusht_config.json")
    cfg = json.load(open(config_path))
    cfg.update({
        "method": "rs_imle",
        "task": "pusht",
        "epsilon": 0.03,
        "n_samples_per_condition": 20,
        "use_traj_consistency": args.use_traj_consistency,
        "render_eval": args.render_eval,
        "num_trails": args.num_trails,
    })

    down_dims = [int(x) for x in args.down_dims.split(",")]

    nets = nn.ModuleDict()
    nets["vision_encoder_0"] = replace_bn_with_gn(get_resnet("resnet18"))
    nets["policy_net"] = GeneratorConditionalUnet1D(
        input_dim=cfg["action_dim"],
        global_cond_dim=cfg["obs_dim"] * cfg["obs_horizon"],
        down_dims=down_dims,
    )

    weights_path = os.path.join(args.run_dir, args.weights)
    state_dict = torch.load(weights_path, map_location="cpu")
    nets.load_state_dict(state_dict)

    if args.half:
        raise NotImplementedError(
            "--half isn't wired up: eval_policy_pusht.py casts its inputs to float32 "
            "unconditionally, so a .half() model would crash with a dtype mismatch on the "
            "first forward pass. Eval fits comfortably in fp32 (see colab/COLAB_WORKFLOW.md), "
            "so just drop --half; if fp16 is ever genuinely needed, update "
            "eval_policy_pusht.py's nimages/nagent_poses .to(dtype=...) calls to match the "
            "model's dtype first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = nets.to(device)
    nets.eval()

    stats = torch.load(os.path.join(args.run_dir, "stats.pth"), map_location="cpu", weights_only=False)

    print(f"Loaded {weights_path} (down_dims={down_dims}) onto {device}"
          f"{' [fp16]' if args.half else ''}")
    evaluate(cfg, nets, stats, method="rs_imle")


if __name__ == "__main__":
    main()
