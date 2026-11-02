# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# Isaac Sim 5.1 (omni.physx 107.3) parses the cloned scene in parallel carb.tasking tasks when the simulation starts
# (PhysX replicator). With the extra collider meshes of the ball tasks' robot (forearms, split rubber hands; not
# instanced, unlike the G1's own colliders) a data race in that parse corrupts the heap: random aborts
# ("malloc(): invalid size (unsorted)", "free(): invalid next size") or a hang with one core busy while the scene is
# built. Measured on workstation 2 (2026-11-02): 6 of 39 starts failed with these colliders, 0 of 28 with the G1's own
# and 0 of 29 with one tasking thread (24 by default), which removes the race; the scene build takes ~2 s longer, the
# simulation steps are as fast. NVIDIA fixed the same kind of race in the parallel USD physics parsing in Isaac Sim 6.1
# (OpenUSD PR #4002): drop this after an upgrade.
SINGLE_TASKING_THREAD = "--/plugins/carb.tasking.plugin/threadCount=1"


def add_setup_workaround(args_cli: argparse.Namespace):
    """Append ``SINGLE_TASKING_THREAD`` to ``args_cli.kit_args`` (call before ``AppLauncher(args_cli)``).

    Skipped for a single robot: without clones the replicator does not run, and camera rendering (videos) also runs
    on the tasking pool and becomes several times slower with one thread (2026-11-02)."""
    if getattr(args_cli, "num_envs", None) == 1:
        return
    kit_args = getattr(args_cli, "kit_args", None) or ""
    if SINGLE_TASKING_THREAD not in kit_args:
        args_cli.kit_args = f"{kit_args} {SINGLE_TASKING_THREAD}".strip()


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--resume_experiment",
        type=str,
        default=None,
        help="Experiment folder to load the resume checkpoint from. Defaults to the experiment name. Use it to"
        " start a run from another task's checkpoint without logging into that task's folder.",
    )
    arg_group.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="With --resume: load only the actor weights (fresh critic, optimizer and iteration counter).",
    )
    arg_group.add_argument(
        "--finetune_std",
        type=float,
        default=None,
        help="With --resume: overwrite the loaded action noise std with this value.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if rslrl_cfg.experiment_name == "":
        rslrl_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    return agent_cfg
