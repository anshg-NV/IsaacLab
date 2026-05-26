# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize action chunks of a chunking policy (e.g. diffusion policy) live during rollout.

Each rollout proceeds chunk-by-chunk:
    1. Predict the next ``prediction_horizon`` actions.
    2. Render the predicted trajectory as a cubic-spline curve drawn with
       ``isaacsim.util.debug_draw`` line segments. Active = green (gripper-open
       command) / red (gripper-close command). Previous chunks fade to gray.
    3. Execute the first ``action_horizon`` actions in the env.
    4. Repeat.

Requires a checkpoint from an action-chunking algorithm — the policy's underlying
``Algo`` must expose ``_get_action_trajectory``.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: Step horizon of each rollout.
    seed: Random seed used once at the start of the script.
    norm_factor_min: Optional dataset-action normalization minimum.
    norm_factor_max: Optional dataset-action normalization maximum.
    delta_pos_scale: Per-axis multiplier on action delta-pos for viz cumsum only
        (match the env's DifferentialInverseKinematicsActionCfg.scale).
    hide_viz_during_step: If set, clear the debug-draw lines immediately before each
        env.step and re-emit them after, so the visuomotor cameras render a clean
        scene. Causes visible flicker.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize chunk-by-chunk rollouts of a robomimic policy in Isaac Lab.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=600, help="Step horizon of each rollout.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument(
    "--delta_pos_scale",
    type=float,
    default=0.1,
    help=(
        "Multiplier applied to the action's delta-pos dims when cumsum-integrating the"
        " predicted EEF path for visualization only. The env's IK controller scale"
        " (0.5 for Franka stack IK-rel) gives the *commanded* delta, but the robot"
        " achieves only a fraction of that per control step due to joint-velocity"
        " limits, so a smaller scale produces a visualization that better matches the"
        " actual EEF trajectory. Does not affect actions sent to env.step."
    ),
)
parser.add_argument(
    "--hide_viz_during_step",
    action="store_true",
    default=False,
    help=(
        "Show the chunk curves briefly between chunks but clear them for the entire"
        " action_horizon block of env.steps so the visuomotor cameras render a clean"
        " scene. Causes per-chunk flashes rather than per-step flicker."
    ),
)
parser.add_argument(
    "--save_obs_images",
    type=str,
    default=None,
    help=(
        "Diagnostic: directory to dump table_cam and wrist_cam observations into during"
        " the rollout. Saves a frame every --save_obs_every steps. Use to verify what"
        " the visuomotor policy is actually seeing."
    ),
)
parser.add_argument(
    "--save_obs_every",
    type=int,
    default=20,
    help="Save an obs frame every N env.steps when --save_obs_images is set.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import select
import sys
import time
from collections import deque

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
from scipy.interpolate import CubicSpline

from isaacsim.util.debug_draw import _debug_draw

from isaaclab_tasks.utils import parse_env_cfg


# Cubic-spline samples placed between each pair of consecutive predicted action steps.
# Higher = smoother visual but more line segments per chunk.
SAMPLES_PER_STEP = 8
# A predicted gripper-action value above this counts as a "close" command (red); below
# is "open" (green). For IK-rel policies on Franka the gripper action is typically signed
# with 0 as the natural midpoint; tune if your policy uses a different convention.
GRIPPER_ACTION_CLOSED_THRESHOLD = 0.0
# Line widths (pixels) used by the debug-draw overlay.
LINE_THICKNESS = 4.0
# RGBA colors. Active uses fully saturated red/green; stale is a dark uniform gray so it
# does not wash out against bright scene lighting.
ACTIVE_OPEN_COLOR = (0.0, 1.0, 0.0, 1.0)
ACTIVE_CLOSED_COLOR = (1.0, 0.0, 0.0, 1.0)
STALE_COLOR = (0.25, 0.25, 0.25, 1.0)


def upsample_trajectory(positions, gripper, samples_per_step=SAMPLES_PER_STEP):
    """Fit a cubic spline through a predicted chunk's EEF positions and resample densely.

    Each interpolated sample inherits the gripper action of its nearest control point so
    color transitions land at the right places along the curve.

    Args:
        positions: ``(N, 3)`` array of predicted EEF positions [m].
        gripper: ``(N,)`` array of predicted gripper action values.
        samples_per_step: cubic-spline samples placed between consecutive control points.

    Returns:
        positions_fine: ``((N - 1) * samples_per_step + 1, 3)`` densely sampled positions
            (or the input as-is if ``N < 2``).
        gripper_fine: ``((N - 1) * samples_per_step + 1,)`` gripper values inherited from
            the nearest control point (or the input as-is if ``N < 2``).
    """
    n = len(positions)
    if n < 2:
        return positions, gripper
    t_orig = np.arange(n)
    t_fine = np.linspace(0, n - 1, (n - 1) * samples_per_step + 1)
    spline = CubicSpline(t_orig, positions, axis=0)
    positions_fine = spline(t_fine)
    nearest = np.round(t_fine).astype(int).clip(0, n - 1)
    gripper_fine = gripper[nearest]
    return positions_fine, gripper_fine


def save_camera_obs(obs_dict, save_dir, step):
    """Dump table_cam and wrist_cam from the current obs to PNGs for inspection.

    Args:
        obs_dict: the env's latest observation dict (contains a ``"policy"`` sub-dict).
        save_dir: directory to write PNGs into; created if missing.
        step: integer step index used in the filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    policy_obs = obs_dict.get("policy", {})
    for cam_key in ("table_cam", "wrist_cam"):
        if cam_key not in policy_obs:
            continue
        tensor = policy_obs[cam_key]
        if not isinstance(tensor, torch.Tensor):
            continue
        img = tensor.squeeze(0).detach().cpu().numpy()
        # IsaacLab cameras emit uint8 HWC. If we ever get float in [0, 1] (e.g. some
        # custom pipeline), rescale defensively.
        if img.dtype != np.uint8:
            if img.max() <= 1.0 + 1e-6:
                img = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(save_dir, f"{cam_key}_step{step:04d}.png"), img)


def wait_for_enter(prompt, sim_app):
    """Block until the user presses Enter, keeping the viewport responsive."""
    print(prompt, end="", flush=True)
    while sim_app.is_running():
        sim_app.update()
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            return True
    return False


class TrajectoryVisualizer:
    """Draws active and stale chunk trajectories as debug-draw line curves.

    The *active* curve is the most recently predicted chunk (red/green per gripper
    command). The *stale* buffer is every previously-predicted chunk in this rollout,
    drawn in dark gray. Calling :meth:`push_chunk` demotes the previous active curve to
    stale before rendering the new active.

    There is no native visibility flag for debug_draw, so :meth:`set_visibility` is
    implemented by clearing the overlay (hide) and re-emitting all stored segments
    (show).
    """

    def __init__(self):
        self.draw = _debug_draw.acquire_debug_draw_interface()
        # Stale segments accumulated across all previous chunks; each entry is one
        # segment expressed as start_xyz and end_xyz in parallel lists.
        self._stale_starts: list = []
        self._stale_ends: list = []
        # Active segments are the latest chunk only, kept separate so they can be
        # promoted to stale on the next push without a second pass.
        self._active_starts: list = []
        self._active_ends: list = []
        self._active_colors: list = []
        self._visible = True

    def push_chunk(self, positions, gripper_actions):
        """Demote the previous active curve to stale, then render the new active curve."""
        # Promote previous active to stale.
        if self._active_starts:
            self._stale_starts.extend(self._active_starts)
            self._stale_ends.extend(self._active_ends)

        positions = np.asarray(positions, dtype=np.float32)
        gripper_actions = np.asarray(gripper_actions, dtype=np.float32)
        starts = positions[:-1].tolist()
        ends = positions[1:].tolist()
        seg_grippers = gripper_actions[:-1]
        colors = [
            list(ACTIVE_CLOSED_COLOR) if g > GRIPPER_ACTION_CLOSED_THRESHOLD else list(ACTIVE_OPEN_COLOR)
            for g in seg_grippers
        ]
        self._active_starts = starts
        self._active_ends = ends
        self._active_colors = colors

        if self._visible:
            self._redraw()

    def set_visibility(self, visible):
        """Toggle whether the trajectory lines are emitted to the overlay.

        ``False`` clears the overlay. ``True`` re-emits all stored stale + active
        segments. Idempotent.
        """
        if visible == self._visible:
            return
        self._visible = visible
        if visible:
            self._redraw()
        else:
            self.draw.clear_lines()

    def clear(self):
        self.draw.clear_lines()
        self._stale_starts = []
        self._stale_ends = []
        self._active_starts = []
        self._active_ends = []
        self._active_colors = []

    def _redraw(self):
        self.draw.clear_lines()
        n_stale = len(self._stale_starts)
        n_active = len(self._active_starts)
        if n_stale + n_active == 0:
            return
        all_starts = self._stale_starts + self._active_starts
        all_ends = self._stale_ends + self._active_ends
        all_colors = [list(STALE_COLOR)] * n_stale + self._active_colors
        thicknesses = [LINE_THICKNESS] * (n_stale + n_active)
        self.draw.draw_lines(all_starts, all_ends, all_colors, thicknesses)


def rollout_with_chunk_viz(policy, env, success_term, horizon, device, viz):
    """One rollout that visualises each predicted action chunk before executing it."""
    policy.start_episode()
    obs_dict, _ = env.reset()

    algo = policy.policy
    if not hasattr(algo, "_get_action_trajectory"):
        raise RuntimeError(
            "play_traj.py expects an action-chunking algorithm (e.g. diffusion_policy)."
            f" The loaded algo ({type(algo).__name__}) has no _get_action_trajectory method."
        )

    horizons = algo.algo_config.horizon
    action_horizon = int(horizons.action_horizon)
    prediction_horizon = int(horizons.prediction_horizon)
    try:
        frame_stack = max(1, int(algo.global_config.train.frame_stack))
    except AttributeError:
        frame_stack = max(1, int(horizons.observation_horizon))

    # Initialise the frame-stack history from the reset observation. Clone tensors so
    # deque entries don't alias env-internal buffers that may be reused in-place.
    initial_raw = {
        k: torch.squeeze(v, dim=0).clone() if isinstance(v, torch.Tensor) else v
        for k, v in obs_dict["policy"].items()
    }
    obs_history = {
        k: deque([v.unsqueeze(0)] * frame_stack, maxlen=frame_stack)
        for k, v in initial_raw.items() if isinstance(v, torch.Tensor)
    }

    success = False
    terminated = False
    truncated = False
    step = 0
    first_chunk = True

    while step < horizon and not (success or terminated or truncated):
        # Build the framestacked obs for the policy.
        obs = {k: torch.cat(list(q), dim=0) for k, q in obs_history.items()}

        # Run the policy's standard obs preprocessing (batching, RGB HWC->CHW + scaling).
        prepared = policy._prepare_observation(obs, batched_ob=False)

        # Predict the full chunk directly, skipping the internal action queue.
        chunk = algo._get_action_trajectory(obs_dict=prepared)  # (1, Tp, A)
        chunk_np = chunk[0].detach().cpu().numpy().astype(np.float32)

        # Optional CLI rescaling — needed when the dataset was normalised by
        # --normalize_training_actions at train time.
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            chunk_np = (
                (chunk_np + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        if first_chunk:
            print(
                f"  [DBG] first chunk action range: full[{chunk_np.min():+.3f},{chunk_np.max():+.3f}]"
                f" delta_pos[{chunk_np[:, :3].min():+.3f},{chunk_np[:, :3].max():+.3f}]"
                f" gripper[{chunk_np[:, -1].min():+.3f},{chunk_np[:, -1].max():+.3f}]"
            )
            first_chunk = False

        # Integrate predicted delta-pos actions (first 3 dims) from the current EEF
        # position. Multiply by --delta_pos_scale to match the env IK controller scale.
        cur_eef = obs_history["eef_pos"][-1].squeeze(0).detach().cpu().numpy().reshape(-1)
        delta_pos = chunk_np[:, :3] * args_cli.delta_pos_scale
        future_positions = cur_eef[None, :] + np.cumsum(delta_pos, axis=0)
        predicted_positions = np.concatenate([cur_eef[None, :], future_positions], axis=0)

        gripper_actions = chunk_np[:, -1]
        gripper_for_curve = np.concatenate([gripper_actions[:1], gripper_actions], axis=0)

        positions_fine, gripper_fine = upsample_trajectory(predicted_positions, gripper_for_curve)
        viz.push_chunk(positions_fine, gripper_fine)

        # Execute the first action_horizon actions of the chunk. Only hide the markers
        # for the *last* frame_stack env.steps of the chunk — those are the obs that
        # survive in the deque the next chunk's prediction will read. Earlier obs get
        # evicted, so the cameras can pick up the markers there without consequence.
        # This minimises visible flicker (one brief wink at each chunk boundary
        # instead of a wink per env.step).
        chunk_t = torch.from_numpy(chunk_np).to(device).float()
        action_dim = env.action_space.shape[1]
        n_actions = min(action_horizon, prediction_horizon)
        for ai in range(n_actions):
            if step >= horizon:
                break
            action_t = chunk_t[ai].view(1, action_dim)
            needs_clean = args_cli.hide_viz_during_step and (n_actions - 1 - ai) < frame_stack
            if needs_clean:
                viz.set_visibility(False)
            obs_dict, _, terminated, truncated, _ = env.step(action_t)
            if needs_clean:
                viz.set_visibility(True)
                simulation_app.update()
            step += 1

            # Diagnostic: dump what the cameras actually saw on this step.
            if args_cli.save_obs_images is not None and step % args_cli.save_obs_every == 0:
                save_camera_obs(obs_dict, args_cli.save_obs_images, step)

            # Update the framestack history.
            new_raw = {
                k: torch.squeeze(v, dim=0).clone() if isinstance(v, torch.Tensor) else v
                for k, v in obs_dict["policy"].items()
            }
            for k in obs_history:
                obs_history[k].append(new_raw[k].unsqueeze(0))

            if bool(success_term.func(env, **success_term.params)[0]):
                success = True
                break
            if terminated or truncated:
                break

    return success


def main():
    """Run a single chunk-visualised rollout per Enter press."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
        print(
            f"[INFO] Unscaling actions from [-1, 1] back to"
            f" [{args_cli.norm_factor_min}, {args_cli.norm_factor_max}]."
        )
    else:
        print(
            "[INFO] No action unscaling (--norm_factor_min/max not provided)."
            " If you trained with --normalize_training_actions, pass those values"
            " from logs/.../normalization_params.txt."
        )

    viz = TrajectoryVisualizer()

    with torch.inference_mode():
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

        sequence = 0
        try:
            while simulation_app.is_running():
                sequence += 1
                print(f"\n[INFO] Sequence {sequence}: chunked rollout")
                viz.clear()

                start = time.perf_counter()
                success = rollout_with_chunk_viz(
                    policy, env, success_term, args_cli.horizon, device, viz
                )
                elapsed = time.perf_counter() - start
                print(f"[INFO] Rollout done: success={success}, time={elapsed:.1f}s")

                if not wait_for_enter(
                    "\n>>> Press Enter to start the next sequence (Ctrl+C to exit)... ",
                    simulation_app,
                ):
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user; exiting.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
