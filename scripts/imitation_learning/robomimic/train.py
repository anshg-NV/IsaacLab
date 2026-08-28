"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

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
from robomimic.utils.log_utils import DataLogger, PrintLogger, flush_warnings
from torch.utils.data import DataLoader

from isaaclab_arena.utils.experiment_paths import ExperimentPaths


def make_run_dirs(run_dir: str, save_enabled: bool, resume: bool, overwrite: bool) -> tuple[str, str | None, str]:
    """Create the flat output tree for a single run and return its (logs, models, run) dirs.

    Replaces robomimic's ``TrainUtils.get_exp_dir``, which nests outputs under an extra
    ``<experiment name>/<timestamp>`` pair. Here everything for one run lives directly in
    ``run_dir``: ``logs/``, ``models/``, ``config.json`` and ``last.pth``.

    Args:
        run_dir: Directory holding every output of this run.
        save_enabled: Whether model checkpointing is enabled; the ``models`` dir is only created if so.
        resume: Whether training resumes in an existing run directory.
        overwrite: Whether to delete an existing run directory instead of refusing to write into it.
    """
    if resume:
        assert os.path.isdir(run_dir), f"Resuming training run, but run directory {run_dir} does not exist"
    elif os.path.isdir(run_dir) and os.listdir(run_dir):
        assert overwrite, (
            f"Run directory {run_dir} already exists and is not empty. Pass --overwrite to replace it, --resume to"
            " continue training in it, or pick a different --run."
        )
        print(f"Removing existing run directory {run_dir}")
        shutil.rmtree(run_dir)

    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ckpt_dir = None
    if save_enabled:
        ckpt_dir = os.path.join(run_dir, "models")
        os.makedirs(ckpt_dir, exist_ok=True)

    return log_dir, ckpt_dir, run_dir


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


def train(config, device, run_dir, resume=False, overwrite=False, normalize_training_actions=False):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, time_dir = make_run_dirs(run_dir, save_enabled=config.experiment.save.enabled, resume=resume, overwrite=overwrite)

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if normalize_training_actions:
        normalize_hdf5_actions(config, log_dir)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # action_keys is shared across all datasets — v0.5's get_shape_metadata_from_dataset
    # requires it explicitly. Default to ["actions"] for v0.4-style configs.
    action_keys = list(config.train.get("action_keys") or ["actions"])

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
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
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        # if an environment name is specified, just use this env using the first dataset's metadata
        # and ignore envs from all datasets
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        for env_i in range(len(env_meta_list)):
            # check if this env should be evaluated
            dataset_cfg = config.train.data[env_i]
            if not dataset_cfg.get("eval", True):
                continue

            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]

            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                # if additional environments are specified, add them to the list
                # all additional environments use env_meta from the first dataset
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            # create environment for each env_name
            for env_name in env_names:
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
                )
                # handle environment wrappers
                envs[env.name] = env
                print(env)

    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta_list[0]["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
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
            drop_last=True
        )
    else:
        valid_loader = None

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # add info to optim_params
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs
        # handling for "hbc" and "iris" algorithms
        if config.algo_name == "hbc":
            for sub_algo in ["planner", "actor"]:
                # add info to optim_params of each net
                for k in config.algo[sub_algo].optim_params:
                    config.algo[sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo[sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs
        if config.algo_name == "iris":
            for sub_algo in ["planner", "value"]:
                # add info to optim_params of each net
                for k in config.algo["value_planner"][sub_algo].optim_params:
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device
    )

    if resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)

    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1 # start at next epoch, since this recorded the last epoch of training completed
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
        )

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        # always save latest model for resume functionality
        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
            shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )

        # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
        shutil.copyfile(latest_model_path, latest_model_backup_path)
        print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


def main(args):
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
        # default: the run's config inside the experiment, experiments/<experiment>/train/configs/dp_<run>.json
        config_file = str(ExperimentPaths(args.experiment).configs_dir / f"dp_{args.run}.json")
        assert os.path.exists(config_file), (
            f"No config for run {args.run} of experiment {args.experiment} at {config_file}. Add it there, or pass"
            " --config explicitly."
        )
        print(f"Loading configuration for run {args.run}: {config_file}")

    with open(config_file) as f:
        ext_cfg = json.load(f)
        config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
        if not isinstance(config.train.data, list):
            default_path = config.train.data if isinstance(config.train.data, str) else ""
            config.train.data = [{"path": default_path}]

    if args.dataset is not None:
        config.train.data[0]["path"] = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.epochs is not None:
        config.train.num_epochs = args.epochs

    # every output of this run lands in experiments/<experiment>/eval/<run>
    run_dir = str(ExperimentPaths(args.experiment).run_dir(args.run))
    config.train.output_dir = run_dir

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        run_dir = os.path.join("/tmp/tmp_trained_models", args.run)
        config.train.output_dir = run_dir

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(
            config,
            device,
            run_dir,
            resume=args.resume,
            overwrite=args.overwrite,
            normalize_training_actions=args.normalize_training_actions,
        )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            Defaults to the run's config in the experiment, experiments/<experiment>/train/configs/dp_<run>.json.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        default="diffusion_policy",
        help="name of algorithm to run. Only used when the config is resolved from --task",
    )

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

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name; outputs go to experiments/<experiment>/eval/<run>.",
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help=(
            "Run index within the experiment, e.g. '0'. Names the checkpoint directory and selects the training"
            " config dp_<run>.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="set this flag to delete an existing run directory instead of erroring out",
    )
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
    main(args)
