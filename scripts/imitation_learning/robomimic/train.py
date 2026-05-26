# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict

import gymnasium as gym
import h5py
import numpy as np
import psutil
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger
from torch.utils.data import DataLoader

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def _set_dataset_path(config: Config, new_path: str) -> None:
    """Set the HDF5 dataset path on a v0.5-form robomimic config.

    Args:
        config: Robomimic ``Config``.
        new_path: New filesystem path to assign.
    """
    config.train.data[0]["path"] = new_path


def normalize_hdf5_actions(config: Config, log_dir: str) -> None:
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Each entry in ``config.train.data`` gets its own ``_normalized`` copy on disk,
    with a per-dataset min/max scaling. Each entry's ``path`` field is updated
    in-place to point at the normalized copy. Per-dataset min/max values are
    written to ``<log_dir>/normalization_params.txt``.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.
    """
    params_lines = []
    for ds_idx, dataset_cfg in enumerate(config.train.data):
        original_path = dataset_cfg["path"]
        base, ext = os.path.splitext(original_path)
        normalized_path = base + "_normalized" + ext

        print(f"Creating normalized dataset at {normalized_path}")
        shutil.copyfile(original_path, normalized_path)

        with h5py.File(normalized_path, "r+") as hf:
            n_demos = len(hf["data"].keys())
            dataset_paths = [f"/data/demo_{i}/actions" for i in range(n_demos)]

            # Streaming min/max — avoids materialising all actions at once.
            ds_min = float("inf")
            ds_max = float("-inf")
            for path in dataset_paths:
                arr = hf[path][...]
                ds_min = min(ds_min, float(arr.min()))
                ds_max = max(ds_max, float(arr.max()))

            # In-place rescale — preserves the existing chunking/compression and
            # avoids the h5py del+reassign pattern, which accumulates internal
            # dataset metadata and grows the file on disk.
            scale = ds_max - ds_min
            for path in dataset_paths:
                ds = hf[path]
                ds[...] = 2.0 * ((ds[...] - ds_min) / scale) - 1.0

        dataset_cfg["path"] = normalized_path
        params_lines.append(f"dataset {ds_idx}: {original_path}\n  min: {ds_min}\n  max: {ds_max}\n")

    with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
        f.writelines(params_lines)


def train(config: Config, device: str, log_dir: str, ckpt_dir: str, video_dir: str):
    """Train a model using the algorithm specified in config.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dir: Directory to save logs.
        ckpt_dir: Directory to save checkpoints.
        video_dir: Directory to save videos.
    """
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # action_keys is shared across all datasets — v0.5's get_shape_metadata_from_dataset
    # requires it explicitly. Default to ["actions"] for v0.4-style configs.
    action_keys = list(config.train.get("action_keys") or ["actions"])

    # extract metadata (env + shape) for every configured dataset
    print("\n============= Loaded Environment Metadata =============")
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

        # v0.5 FileUtils.get_env_metadata_from_dataset dereferences env_meta["env_kwargs"] before
        # returning; IsaacLab Mimic datasets omit that key. Inline the helper so the default
        # can be injected before the lookup.
        with h5py.File(dataset_path, "r") as _f:
            env_meta = json.loads(_f["data"].attrs["env_args"])
        env_meta.setdefault("env_kwargs", {})
        env_meta["env_kwargs"].pop("env_lang", None)
        EnvUtils.set_env_specific_obs_processing(env_meta=env_meta)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True,
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        # mirror upstream: apply the override only to the first dataset's env_meta
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environments (kept structurally aligned with upstream v0.5; unused at train time
    # since the IsaacLab fork strips in-loop rollouts)
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        for env_i in range(len(env_meta_list)):
            dataset_cfg = config.train.data[env_i]
            if not dataset_cfg.get("eval", True):
                continue
            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]

            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            for env_name in env_names:
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                )
                envs[env.name] = env
                print(envs[env.name])

    print("")

    # setup for a new training run
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)

    # v0.5 LR schedulers (e.g. cosine for diffusion_policy) read num_train_batches and
    # num_epochs from optim_params before training starts. Upstream sources num_train_batches
    # from len(trainset); this fork constructs trainset after algo_factory, so we use
    # config.experiment.epoch_every_n_steps directly — which is what the train loop will
    # actually pass to TrainUtils.run_epoch anyway.
    train_num_steps = config.experiment.epoch_every_n_steps
    assert train_num_steps is not None, (
        "config.experiment.epoch_every_n_steps must be set; this fork builds the dataset"
        " after algo_factory, so len(trainset) is not available for the LR scheduler."
    )
    with config.values_unlocked():
        if "optim_params" in config.algo:
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta_list[0]["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            last_epoch_check = epoch == config.train.num_epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check or last_epoch_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # terminate logging
    data_logger.close()


def main(args: argparse.Namespace):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
    """
    # Determine the JSON config file: --config takes precedence over the gym-registry
    # entry-point lookup keyed by --task / --algo.
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found at: {args.config}")
        config_file = args.config
        print(f"Loading configuration from --config: {config_file}")
    elif args.task is not None:
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        if ":" in cfg_entry_point_file:
            mod_name, file_name = cfg_entry_point_file.split(":")
            mod = importlib.import_module(mod_name)
            if mod.__file__ is None:
                raise ValueError(f"Could not find module file for: '{mod_name}'")
            mod_path = os.path.dirname(mod.__file__)
            config_file = os.path.join(mod_path, file_name)
        else:
            config_file = cfg_entry_point_file
    else:
        raise ValueError("Please provide either --config or --task on the CLI.")

    with open(config_file) as f:
        ext_cfg = json.load(f)
        config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
        # v0.5 dataset_factory iterates config.train.data as a list of {"path": ...} dicts.
        # IsaacLab task configs ship `data` as a string (or omit it, leaving v0.5's None
        # default in place). Coerce once so the rest of this script and robomimic internals
        # can assume the v0.5 form.
        if not isinstance(config.train.data, list):
            default_path = config.train.data if isinstance(config.train.data, str) else ""
            config.train.data = [{"path": default_path}]

    if args.dataset is not None:
        _set_dataset_path(config, args.dataset)

    if args.name is not None:
        config.experiment.name = args.name

    if args.epochs is not None:
        config.train.num_epochs = args.epochs

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    # v0.5 get_exp_dir returns a 4th value (time_dir) used for resume; not needed here.
    log_dir, ckpt_dir, video_dir, _ = TrainUtils.get_exp_dir(config)

    if args.normalize_training_actions:
        normalize_hdf5_actions(config, log_dir)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device, log_dir, ckpt_dir, video_dir)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "(optional) path to a JSON config file. If provided, bypasses the"
            " gym-registry entry-point lookup and loads the algo config from this"
            " file directly."
        ),
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument("--log_dir", type=str, default="robomimic", help="Path to log directory")
    parser.add_argument("--normalize_training_actions", action="store_true", default=False, help="Normalize actions")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional: Number of training epochs. If specified, overrides the number of epochs from the JSON training"
            " config."
        ),
    )

    args = parser.parse_args()

    # run training
    main(args)
    # close sim app
    simulation_app.close()
