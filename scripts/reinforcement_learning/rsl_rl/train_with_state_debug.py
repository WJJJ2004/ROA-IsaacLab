# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--imu_debug", action="store_true", default=False, help="Enable periodic IMU angular velocity debug print.")
parser.add_argument("--imu_debug_interval", type=int, default=200, help="Print IMU angular velocity stats every N environment steps.")
parser.add_argument("--imu_debug_threshold", type=float, default=20.0, help="Warning threshold for max abs IMU angular velocity.")
parser.add_argument("--imu_debug_diff_threshold", type=float, default=10.0, help="Warning threshold for max abs difference between IMU and base angular velocity.")
parser.add_argument("--imu_debug_max_envs", type=int, default=8, help="Maximum number of bad env indices to print each debug event.")
parser.add_argument("--joint_vel_debug_threshold", type=float, default=20.0, help="Warning threshold for max abs joint velocity.")
parser.add_argument("--action_debug_threshold", type=float, default=2.0, help="Warning threshold for max abs action value.")
parser.add_argument("--action_delta_debug_threshold", type=float, default=1.0, help="Warning threshold for max abs action delta.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def _safe_tensor_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def _extract_imu_ang_vel_from_env(env) -> torch.Tensor | None:
    """Return IMU angular velocity tensor with shape [num_envs, 3] if available."""
    try:
        base_env = env.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            return None

        imu_sensor = None
        if hasattr(scene, "__contains__") and "imu" in scene:
            imu_sensor = scene["imu"]
        elif hasattr(scene, "imu"):
            imu_sensor = scene.imu

        if imu_sensor is None:
            return None

        data = getattr(imu_sensor, "data", None)
        if data is None:
            return None

        for attr_name in ("ang_vel_b", "ang_vel", "angular_velocity"):
            if hasattr(data, attr_name):
                value = getattr(data, attr_name)
                if isinstance(value, torch.Tensor):
                    if value.ndim == 3:
                        return value[:, 0, :]
                    return value

        return None
    except Exception as e:
        print(f"[IMU DEBUG] Failed to extract imu angular velocity: {e}")
        return None


def _extract_base_ang_vel_from_env(env) -> torch.Tensor | None:
    """Return robot base angular velocity tensor with shape [num_envs, 3] if available."""
    try:
        base_env = env.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            return None

        robot = None
        if hasattr(scene, "__contains__") and "robot" in scene:
            robot = scene["robot"]
        elif hasattr(scene, "robot"):
            robot = scene.robot
        elif hasattr(scene, "__contains__") and "Robot" in scene:
            robot = scene["Robot"]
        elif hasattr(scene, "Robot"):
            robot = scene.Robot

        if robot is None:
            return None

        data = getattr(robot, "data", None)
        if data is None:
            return None

        for attr_name in ("root_ang_vel_b", "root_ang_vel_w", "body_ang_vel_w"):
            if hasattr(data, attr_name):
                value = getattr(data, attr_name)
                if isinstance(value, torch.Tensor):
                    if value.ndim == 3:
                        return value[:, 0, :]
                    return value

        return None
    except Exception as e:
        print(f"[IMU DEBUG] Failed to extract base angular velocity: {e}")
        return None




def _extract_joint_vel_from_env(env) -> torch.Tensor | None:
    """Return joint velocity tensor with shape [num_envs, num_joints] if available."""
    try:
        base_env = env.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            return None

        robot = None
        if hasattr(scene, "__contains__") and "robot" in scene:
            robot = scene["robot"]
        elif hasattr(scene, "robot"):
            robot = scene.robot
        elif hasattr(scene, "__contains__") and "Robot" in scene:
            robot = scene["Robot"]
        elif hasattr(scene, "Robot"):
            robot = scene.Robot

        if robot is None:
            return None

        data = getattr(robot, "data", None)
        if data is None:
            return None

        for attr_name in ("joint_vel", "joint_velocities"):
            if hasattr(data, attr_name):
                value = getattr(data, attr_name)
                if isinstance(value, torch.Tensor):
                    return value

        return None
    except Exception as e:
        print(f"[STATE DEBUG] Failed to extract joint velocity: {e}")
        return None


def attach_imu_debug_hook(
    env,
    interval: int = 200,
    imu_threshold: float = 20.0,
    diff_threshold: float = 10.0,
    max_envs_to_print: int = 8,
    joint_vel_threshold: float = 20.0,
    action_threshold: float = 2.0,
    action_delta_threshold: float = 1.0,
):
    """Monkey-patch env.step() to periodically inspect IMU/joint velocity/action statistics."""
    original_step = env.step
    step_counter = {"count": 0}
    debug_state = {"prev_actions": None}

    def wrapped_step(actions):
        result = original_step(actions)
        step_counter["count"] += 1
        step_idx = step_counter["count"]

        if interval <= 0 or (step_idx % interval != 0):
            if actions is not None:
                debug_state["prev_actions"] = actions.detach().clone()
            return result

        imu_ang = _extract_imu_ang_vel_from_env(env)
        base_ang = _extract_base_ang_vel_from_env(env)
        joint_vel = _extract_joint_vel_from_env(env)

        action_tensor = actions.detach()
        prev_actions = debug_state["prev_actions"]
        if prev_actions is None or prev_actions.shape != action_tensor.shape:
            action_delta = torch.zeros_like(action_tensor)
        else:
            action_delta = action_tensor - prev_actions

        debug_state["prev_actions"] = action_tensor.clone()

        # ---------- IMU ----------
        if imu_ang is None:
            print(
                f"[IMU DEBUG] step={step_idx} | IMU angular velocity not found. "
                "Check scene sensor name or sensor data field names."
            )
        elif imu_ang.ndim != 2 or imu_ang.shape[-1] != 3:
            print(f"[IMU DEBUG] step={step_idx} | Unexpected imu_ang_vel shape: {tuple(imu_ang.shape)}")
        else:
            imu_nan = torch.isnan(imu_ang).any(dim=1)
            imu_inf = torch.isinf(imu_ang).any(dim=1)
            imu_abs_max = imu_ang.abs().max(dim=1).values

            diff = None
            diff_abs_max = None
            if base_ang is not None and base_ang.ndim == 2 and base_ang.shape[-1] == 3:
                diff = imu_ang - base_ang
                diff_abs_max = diff.abs().max(dim=1).values
                imu_bad_mask = imu_nan | imu_inf | (imu_abs_max > imu_threshold) | (diff_abs_max > diff_threshold)
            else:
                imu_bad_mask = imu_nan | imu_inf | (imu_abs_max > imu_threshold)

            print(f"[IMU DEBUG] step={step_idx}")
            print(f"  imu_abs_max_all      : {imu_abs_max.max().item():.6f}")
            print(f"  imu_abs_mean_all     : {imu_ang.abs().mean().item():.6f}")
            print(f"  imu_nan_env_count    : {imu_nan.sum().item()}")
            print(f"  imu_inf_env_count    : {imu_inf.sum().item()}")

            if base_ang is not None and diff_abs_max is not None:
                print(f"  base_abs_max_all     : {base_ang.abs().max().item():.6f}")
                print(f"  imu_base_diff_max_all: {diff_abs_max.max().item():.6f}")

            imu_bad_ids = torch.nonzero(imu_bad_mask).squeeze(-1)
            if imu_bad_ids.numel() == 0:
                print("  status               : OK (no env exceeded thresholds)")
            else:
                imu_bad_ids = imu_bad_ids[:max_envs_to_print]
                print(f"  status               : WARNING ({imu_bad_ids.numel()} envs shown)")
                print(f"  bad_env_ids          : {imu_bad_ids.tolist()}")
                print(f"  imu_ang_vel[bad]     : {_safe_tensor_to_cpu(imu_ang[imu_bad_ids])}")
                if base_ang is not None and diff is not None:
                    print(f"  base_ang_vel[bad]    : {_safe_tensor_to_cpu(base_ang[imu_bad_ids])}")
                    print(f"  imu-base diff[bad]   : {_safe_tensor_to_cpu(diff[imu_bad_ids])}")

        # ---------- JOINT / ACTION / DELTA ----------
                # ---------- PER-DIMENSION ANALYSIS ----------
        print("\n[PER-DIM DEBUG]")

        # action per-dim max
        per_dim_action_max = action_tensor.abs().max(dim=0).values
        print(f"  per_dim_action_max   : {_safe_tensor_to_cpu(per_dim_action_max)}")

        # action per-dim mean
        per_dim_action_mean = action_tensor.abs().mean(dim=0)
        print(f"  per_dim_action_mean  : {_safe_tensor_to_cpu(per_dim_action_mean)}")

        # delta per-dim max
        per_dim_delta_max = action_delta.abs().max(dim=0).values
        print(f"  per_dim_delta_max    : {_safe_tensor_to_cpu(per_dim_delta_max)}")

        # delta per-dim mean
        per_dim_delta_mean = action_delta.abs().mean(dim=0)
        print(f"  per_dim_delta_mean   : {_safe_tensor_to_cpu(per_dim_delta_mean)}")

        # 어떤 index가 제일 문제인지
        worst_action_dim = torch.argmax(per_dim_action_max).item()
        worst_delta_dim = torch.argmax(per_dim_delta_max).item()

        print(f"  worst_action_dim     : {worst_action_dim}")
        print(f"  worst_delta_dim      : {worst_delta_dim}")
        
        print(f"[STATE DEBUG] step={step_idx}")

        # joint velocity stats
        if joint_vel is None:
            print("  joint_vel            : not found")
            joint_bad_mask = None
            joint_bad_ids = torch.empty(0, dtype=torch.long, device=action_tensor.device)
        else:
            joint_nan = torch.isnan(joint_vel).any(dim=1)
            joint_inf = torch.isinf(joint_vel).any(dim=1)
            joint_abs_max = joint_vel.abs().max(dim=1).values
            joint_bad_mask = joint_nan | joint_inf | (joint_abs_max > joint_vel_threshold)

            print(f"  joint_vel_abs_max_all: {joint_abs_max.max().item():.6f}")
            print(f"  joint_vel_abs_mean_all: {joint_vel.abs().mean().item():.6f}")
            print(f"  joint_vel_nan_count  : {joint_nan.sum().item()}")
            print(f"  joint_vel_inf_count  : {joint_inf.sum().item()}")

            joint_bad_ids = torch.nonzero(joint_bad_mask).squeeze(-1)

        # action stats
        action_nan = torch.isnan(action_tensor).any(dim=1)
        action_inf = torch.isinf(action_tensor).any(dim=1)
        action_abs_max = action_tensor.abs().max(dim=1).values
        action_bad_mask = action_nan | action_inf | (action_abs_max > action_threshold)

        print(f"  action_abs_max_all   : {action_abs_max.max().item():.6f}")
        print(f"  action_abs_mean_all  : {action_tensor.abs().mean().item():.6f}")
        print(f"  action_nan_count     : {action_nan.sum().item()}")
        print(f"  action_inf_count     : {action_inf.sum().item()}")

        # action delta stats
        action_delta_nan = torch.isnan(action_delta).any(dim=1)
        action_delta_inf = torch.isinf(action_delta).any(dim=1)
        action_delta_abs_max = action_delta.abs().max(dim=1).values
        delta_bad_mask = action_delta_nan | action_delta_inf | (action_delta_abs_max > action_delta_threshold)

        print(f"  delta_abs_max_all    : {action_delta_abs_max.max().item():.6f}")
        print(f"  delta_abs_mean_all   : {action_delta.abs().mean().item():.6f}")
        print(f"  delta_nan_count      : {action_delta_nan.sum().item()}")
        print(f"  delta_inf_count      : {action_delta_inf.sum().item()}")

        # combined bad ids
        combined_mask = action_bad_mask | delta_bad_mask
        if joint_vel is not None and joint_bad_mask is not None:
            combined_mask = combined_mask | joint_bad_mask

        bad_ids = torch.nonzero(combined_mask).squeeze(-1)
        if bad_ids.numel() == 0:
            print("  state_status         : OK (no env exceeded thresholds)")
        else:
            bad_ids = bad_ids[:max_envs_to_print]
            print(f"  state_status         : WARNING ({bad_ids.numel()} envs shown)")
            print(f"  bad_env_ids          : {bad_ids.tolist()}")

            if joint_vel is not None:
                print(f"  joint_vel[bad]       : {_safe_tensor_to_cpu(joint_vel[bad_ids])}")
            print(f"  action[bad]          : {_safe_tensor_to_cpu(action_tensor[bad_ids])}")
            if prev_actions is not None and prev_actions.shape == action_tensor.shape:
                print(f"  last_action[bad]     : {_safe_tensor_to_cpu(prev_actions[bad_ids])}")
            else:
                print("  last_action[bad]     : None (first debug step)")
            print(f"  action_delta[bad]    : {_safe_tensor_to_cpu(action_delta[bad_ids])}")
            
            per_env_bad_dim = action_tensor.abs().argmax(dim=1)

            print(f"  bad_env_worst_dims   : {_safe_tensor_to_cpu(per_env_bad_dim[bad_ids])}")

        return result

    env.step = wrapped_step
    return env



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.imu_debug:
        print("[INFO] IMU angular velocity debug hook is enabled.")
        env = attach_imu_debug_hook(
            env,
            interval=args_cli.imu_debug_interval,
            imu_threshold=args_cli.imu_debug_threshold,
            diff_threshold=args_cli.imu_debug_diff_threshold,
            max_envs_to_print=args_cli.imu_debug_max_envs,
            joint_vel_threshold=args_cli.joint_vel_debug_threshold,
            action_threshold=args_cli.action_debug_threshold,
            action_delta_threshold=args_cli.action_delta_debug_threshold,
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()