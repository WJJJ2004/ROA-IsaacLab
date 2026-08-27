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

parser.add_argument("--imu_debug", action="store_true", default=False, help="Enable IMU angular velocity debug print.")
parser.add_argument("--imu_debug_interval", type=int, default=200, help="Print IMU debug every N env steps.")
parser.add_argument("--imu_debug_threshold", type=float, default=20.0, help="Threshold for |imu_ang_vel| max.")
parser.add_argument(
    "--imu_debug_diff_threshold",
    type=float,
    default=10.0,
    help="Threshold for |imu_ang_vel - base_ang_vel| max.",
)
parser.add_argument("--imu_debug_max_envs", type=int, default=8, help="Max bad envs to print per debug event.")

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
def _print_scene_debug_once(env):
    try:
        base_env = env.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            print("[IMU DEBUG] scene is None")
            return

        print("\n[IMU DEBUG] ===== Scene Introspection =====")

        # 가능한 scene key들 출력
        scene_keys = []
        if hasattr(scene, "keys"):
            try:
                scene_keys = list(scene.keys())
            except Exception:
                scene_keys = []

        if scene_keys:
            print(f"[IMU DEBUG] scene keys: {scene_keys}")
        else:
            print("[IMU DEBUG] scene.keys() not available or empty")

        # scene의 주요 속성들도 출력
        attrs = [a for a in dir(scene) if not a.startswith("_")]
        print(f"[IMU DEBUG] scene attrs (partial): {attrs[:50]}")

        # key 기반으로 각 엔티티의 data field 확인
        for key in scene_keys:
            try:
                obj = scene[key]
                print(f"\n[IMU DEBUG] scene['{key}'] -> type: {type(obj)}")

                data = getattr(obj, "data", None)
                if data is not None:
                    data_attrs = [a for a in dir(data) if not a.startswith("_")]
                    print(f"[IMU DEBUG]   data attrs: {data_attrs}")

                    tensor_fields = []
                    for name in data_attrs:
                        try:
                            value = getattr(data, name)
                            if isinstance(value, torch.Tensor):
                                tensor_fields.append((name, tuple(value.shape)))
                        except Exception:
                            pass

                    if tensor_fields:
                        print(f"[IMU DEBUG]   tensor fields: {tensor_fields}")
            except Exception as e:
                print(f"[IMU DEBUG] failed to inspect scene['{key}']: {e}")

        print("[IMU DEBUG] ===== End Scene Introspection =====\n")

    except Exception as e:
        print(f"[IMU DEBUG] scene introspection failed: {e}")

def _extract_imu_ang_vel_from_env(env) -> torch.Tensor | None:
    """More robust IMU angular velocity extractor."""
    try:
        base_env = env.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            return None

        candidate_objects = []

        # 1) keys() 기반으로 후보 수집
        if hasattr(scene, "keys"):
            try:
                for key in scene.keys():
                    try:
                        obj = scene[key]
                        candidate_objects.append((str(key), obj))
                    except Exception:
                        pass
            except Exception:
                pass

        # 2) attr 기반 후보도 수집
        for attr_name in dir(scene):
            if attr_name.startswith("_"):
                continue
            try:
                obj = getattr(scene, attr_name)
                # data 속성이 있는 것만 후보
                if hasattr(obj, "data"):
                    candidate_objects.append((attr_name, obj))
            except Exception:
                pass

        # 중복 제거
        seen = set()
        unique_candidates = []
        for name, obj in candidate_objects:
            obj_id = id(obj)
            if obj_id not in seen:
                seen.add(obj_id)
                unique_candidates.append((name, obj))

        # 이름상 imu/inertial 관련 후보 우선
        prioritized = []
        fallback = []
        for name, obj in unique_candidates:
            lname = name.lower()
            if "imu" in lname or "inertial" in lname:
                prioritized.append((name, obj))
            else:
                fallback.append((name, obj))

        candidates = prioritized + fallback

        # data tensor field 탐색
        preferred_field_names = [
            "ang_vel_b",
            "ang_vel",
            "angular_velocity",
            "gyro",
            "gyro_b",
            "ang_vel_w",
        ]

        for name, obj in candidates:
            data = getattr(obj, "data", None)
            if data is None:
                continue

            # 1) 선호 필드명 우선 탐색
            for field_name in preferred_field_names:
                if hasattr(data, field_name):
                    value = getattr(data, field_name)
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 3:
                            return value[:, 0, :]
                        if value.ndim == 2 and value.shape[-1] == 3:
                            return value

            # 2) shape가 [..., 3] 인 tensor를 fallback으로 탐색
            for field_name in dir(data):
                if field_name.startswith("_"):
                    continue
                try:
                    value = getattr(data, field_name)
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 2 and value.shape[-1] == 3:
                            lname = field_name.lower()
                            if "ang" in lname or "gyro" in lname:
                                return value
                        elif value.ndim == 3 and value.shape[-1] == 3:
                            lname = field_name.lower()
                            if "ang" in lname or "gyro" in lname:
                                return value[:, 0, :]
                except Exception:
                    pass

        return None

    except Exception as e:
        print(f"[IMU DEBUG] Failed to extract imu angular velocity: {e}")
        return None


def _extract_base_ang_vel_from_env(env) -> torch.Tensor | None:
    """Try to extract base/root angular velocity tensor [num_envs, 3] from Isaac Lab env."""
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

        # Common Isaac Lab root angular velocity names
        for attr_name in ("root_ang_vel_b", "root_ang_vel_w", "body_ang_vel_w"):
            if hasattr(data, attr_name):
                value = getattr(data, attr_name)
                if isinstance(value, torch.Tensor):
                    # body_ang_vel_w may be [num_envs, num_bodies, 3]
                    if value.ndim == 3:
                        return value[:, 0, :]
                    return value

        return None
    except Exception as e:
        print(f"[IMU DEBUG] Failed to extract base angular velocity: {e}")
        return None

def attach_imu_debug_hook(
    env,
    interval: int = 200,
    imu_threshold: float = 20.0,
    diff_threshold: float = 10.0,
    max_envs_to_print: int = 8,
):
    original_step = env.step
    step_counter = {"count": 0}
    printed_scene_debug = {"done": False}

    def wrapped_step(actions):
        result = original_step(actions)
        step_counter["count"] += 1
        step_idx = step_counter["count"]

        # 처음 한 번 scene 내부 구조 출력
        if not printed_scene_debug["done"]:
            _print_scene_debug_once(env)
            printed_scene_debug["done"] = True

        if interval <= 0 or (step_idx % interval != 0):
            return result

        imu_ang = _extract_imu_ang_vel_from_env(env)
        base_ang = _extract_base_ang_vel_from_env(env)

        if imu_ang is None:
            print(
                f"[IMU DEBUG] step={step_idx} | IMU angular velocity not found. "
                "Check printed scene keys / data attrs above."
            )
            return result

        if imu_ang.ndim != 2 or imu_ang.shape[-1] != 3:
            print(f"[IMU DEBUG] step={step_idx} | Unexpected imu_ang_vel shape: {tuple(imu_ang.shape)}")
            return result

        imu_nan = torch.isnan(imu_ang).any(dim=1)
        imu_inf = torch.isinf(imu_ang).any(dim=1)
        imu_abs_max = imu_ang.abs().max(dim=1).values

        diff = None
        diff_abs_max = None
        if base_ang is not None and base_ang.ndim == 2 and base_ang.shape[-1] == 3:
            diff = imu_ang - base_ang
            diff_abs_max = diff.abs().max(dim=1).values
            bad_mask = imu_nan | imu_inf | (imu_abs_max > imu_threshold) | (diff_abs_max > diff_threshold)
        else:
            bad_mask = imu_nan | imu_inf | (imu_abs_max > imu_threshold)

        print(f"\n[IMU DEBUG] step={step_idx}")
        print(f"  imu_abs_max_all      : {imu_abs_max.max().item():.6f}")
        print(f"  imu_abs_mean_all     : {imu_ang.abs().mean().item():.6f}")
        print(f"  imu_nan_env_count    : {imu_nan.sum().item()}")
        print(f"  imu_inf_env_count    : {imu_inf.sum().item()}")

        if base_ang is not None and diff_abs_max is not None:
            print(f"  base_abs_max_all     : {base_ang.abs().max().item():.6f}")
            print(f"  imu_base_diff_max_all: {diff_abs_max.max().item():.6f}")

        bad_ids = torch.nonzero(bad_mask).squeeze(-1)
        if bad_ids.numel() == 0:
            print("  status               : OK (no env exceeded thresholds)")
            return result

        bad_ids = bad_ids[:max_envs_to_print]
        print(f"  status               : WARNING ({bad_ids.numel()} envs shown)")
        print(f"  bad_env_ids          : {bad_ids.tolist()}")
        print(f"  imu_ang_vel[bad]     : {_safe_tensor_to_cpu(imu_ang[bad_ids])}")

        if base_ang is not None and diff is not None:
            print(f"  base_ang_vel[bad]    : {_safe_tensor_to_cpu(base_ang[bad_ids])}")
            print(f"  imu-base diff[bad]   : {_safe_tensor_to_cpu(diff[bad_ids])}")

        return result

    env.step = wrapped_step
    return env

    def wrapped_step(actions):
        result = original_step(actions)
        step_counter["count"] += 1
        step_idx = step_counter["count"]

        # 주기적으로만 검사
        if interval <= 0 or (step_idx % interval != 0):
            return result

        imu_ang = _extract_imu_ang_vel_from_env(env)
        base_ang = _extract_base_ang_vel_from_env(env)

        if imu_ang is None:
            print(f"[IMU DEBUG] step={step_idx} | IMU angular velocity not found. Check scene sensor name / data field.")
            return result

        # [num_envs, 3] 가정
        if imu_ang.ndim != 2 or imu_ang.shape[-1] != 3:
            print(f"[IMU DEBUG] step={step_idx} | Unexpected imu_ang_vel shape: {tuple(imu_ang.shape)}")
            return result

        imu_nan = torch.isnan(imu_ang).any(dim=1)
        imu_inf = torch.isinf(imu_ang).any(dim=1)
        imu_abs_max = imu_ang.abs().max(dim=1).values

        if base_ang is not None and base_ang.ndim == 2 and base_ang.shape[-1] == 3:
            diff = imu_ang - base_ang
            diff_abs_max = diff.abs().max(dim=1).values
            bad_mask = imu_nan | imu_inf | (imu_abs_max > imu_threshold) | (diff_abs_max > diff_threshold)
        else:
            diff = None
            diff_abs_max = None
            bad_mask = imu_nan | imu_inf | (imu_abs_max > imu_threshold)

        print(f"\n[IMU DEBUG] step={step_idx}")
        print(f"  imu_abs_max_all      : {imu_abs_max.max().item():.6f}")
        print(f"  imu_abs_mean_all     : {imu_ang.abs().mean().item():.6f}")
        print(f"  imu_nan_env_count    : {imu_nan.sum().item()}")
        print(f"  imu_inf_env_count    : {imu_inf.sum().item()}")

        if base_ang is not None and diff_abs_max is not None:
            print(f"  base_abs_max_all     : {base_ang.abs().max().item():.6f}")
            print(f"  imu_base_diff_max_all: {diff_abs_max.max().item():.6f}")

        bad_ids = torch.nonzero(bad_mask).squeeze(-1)
        if bad_ids.numel() == 0:
            print("  status               : OK (no env exceeded thresholds)")
            return result

        bad_ids = bad_ids[:max_envs_to_print]
        print(f"  status               : WARNING ({bad_ids.numel()} envs shown)")
        print(f"  bad_env_ids          : {bad_ids.tolist()}")
        print(f"  imu_ang_vel[bad]     : {_safe_tensor_to_cpu(imu_ang[bad_ids])}")

        if base_ang is not None:
            print(f"  base_ang_vel[bad]    : {_safe_tensor_to_cpu(base_ang[bad_ids])}")
            print(f"  imu-base diff[bad]   : {_safe_tensor_to_cpu(diff[bad_ids])}")

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
        
    #         # create isaac environment
    # env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.imu_debug:
        print("[INFO] IMU angular velocity debug hook is enabled.")
        env = attach_imu_debug_hook(
            env,
            interval=args_cli.imu_debug_interval,
            imu_threshold=args_cli.imu_debug_threshold,
            diff_threshold=args_cli.imu_debug_diff_threshold,
            max_envs_to_print=args_cli.imu_debug_max_envs,
        )

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
