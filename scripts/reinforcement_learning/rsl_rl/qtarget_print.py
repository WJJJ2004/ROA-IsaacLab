# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ===== 추가한 디버그 옵션 =====
parser.add_argument("--debug_action_mapping", action="store_true", default=False,
                    help="Print action scale / default joint pos / q_target mapping.")
parser.add_argument("--debug_print_every", type=int, default=20,
                    help="Print debug info every N simulation steps.")
parser.add_argument("--debug_env_index", type=int, default=0,
                    help="Environment index to print.")
parser.add_argument("--debug_joint_limit", type=int, default=12,
                    help="How many joints to print.")
# ============================

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from typing import Any

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _to_tensor(x: Any, device=None):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x, device=device)
    except Exception:
        return None


def _safe_getattr_chain(obj: Any, chains: list[list[str]]):
    """여러 후보 경로를 순서대로 시도해서 첫 값을 반환."""
    for chain in chains:
        cur = obj
        ok = True
        try:
            for name in chain:
                cur = getattr(cur, name)
        except Exception:
            ok = False
        if ok:
            return cur, ".".join(chain)
    return None, None


def _get_robot_from_env(env_unwrapped):
    """
    IsaacLab 태스크마다 robot/articulation 이름이 달라서
    자주 쓰이는 후보를 순서대로 탐색.
    """
    candidates = [
        ["scene", "robot"],
        ["scene", "articulation"],
        ["scene", "robots"],
        ["robot"],
        ["articulation"],
        ["agent"],
    ]
    obj, path = _safe_getattr_chain(env_unwrapped, candidates)
    return obj, path


def _extract_default_joint_pos(robot, device):
    candidates = [
        ["data", "default_joint_pos"],
        ["data", "default_joint_positions"],
        ["default_joint_pos"],
        ["default_joint_positions"],
    ]
    value, path = _safe_getattr_chain(robot, candidates)
    return _to_tensor(value, device=device), path


def _extract_current_joint_pos(robot, device):
    candidates = [
        ["data", "joint_pos"],
        ["data", "joint_positions"],
        ["joint_pos"],
        ["joint_positions"],
    ]
    value, path = _safe_getattr_chain(robot, candidates)
    return _to_tensor(value, device=device), path


def _extract_joint_names(robot):
    candidates = [
        ["data", "joint_names"],
        ["joint_names"],
    ]
    value, path = _safe_getattr_chain(robot, candidates)
    return value, path


def _extract_action_scale(env_unwrapped, robot, agent_cfg, device):
    """
    action_scale는 태스크 구현마다 저장 위치가 다를 수 있어서
    env cfg / env / robot 순서로 폭넓게 탐색.
    """
    candidates = [
        ["cfg", "action_scale"],
        ["cfg", "actions", "joint_pos", "scale"],
        ["cfg", "actions", "joint_position", "scale"],
        ["action_scale"],
        ["_action_scale"],
    ]

    # env에서 찾기
    value, path = _safe_getattr_chain(env_unwrapped, candidates)
    if value is not None:
        return _to_tensor(value, device=device), f"env_unwrapped.{path}"

    # robot에서 찾기
    if robot is not None:
        value, path = _safe_getattr_chain(robot, candidates)
        if value is not None:
            return _to_tensor(value, device=device), f"robot.{path}"

    # agent_cfg에서 찾기
    try:
        if hasattr(agent_cfg, "action_scale"):
            return _to_tensor(agent_cfg.action_scale, device=device), "agent_cfg.action_scale"
    except Exception:
        pass

    return None, None


def _extract_applied_joint_target(robot, device):
    """
    실제 actuator target 버퍼가 있으면 출력해보기 위한 시도.
    태스크/버전에 따라 없을 수 있다.
    """
    candidates = [
        ["data", "joint_pos_target"],
        ["data", "joint_position_target"],
        ["data", "joint_targets"],
        ["joint_pos_target"],
        ["joint_position_target"],
        ["joint_targets"],
        ["_joint_pos_target"],
        ["_joint_targets"],
    ]
    value, path = _safe_getattr_chain(robot, candidates)
    return _to_tensor(value, device=device), path


def _format_tensor_slice(x: torch.Tensor, env_id: int, count: int):
    if x is None:
        return "None"
    if x.ndim == 0:
        return str(x.item())
    if x.ndim == 1:
        return str(x[:count].detach().cpu().tolist())
    return str(x[env_id, :count].detach().cpu().tolist())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    timestep = 0

    # ===== 디버그용 초기 탐색 =====
    debug_env_id = args_cli.debug_env_index
    debug_joint_count = args_cli.debug_joint_limit

    robot, robot_path = _get_robot_from_env(env.unwrapped)
    if robot is None:
        print("[DEBUG] robot/articulation object를 찾지 못했습니다.")
    else:
        print(f"[DEBUG] robot object found at: env.unwrapped.{robot_path}")

    joint_names, joint_names_path = _extract_joint_names(robot) if robot is not None else (None, None)
    if joint_names is not None:
        print(f"[DEBUG] joint names source: robot.{joint_names_path}")
        print(f"[DEBUG] num joints: {len(joint_names)}")
        print(f"[DEBUG] joint names (head): {joint_names[:debug_joint_count]}")

    action_scale, action_scale_path = _extract_action_scale(env.unwrapped, robot, agent_cfg, env.unwrapped.device)
    if action_scale is not None:
        print(f"[DEBUG] action_scale source: {action_scale_path}")
        print(f"[DEBUG] action_scale shape: {tuple(action_scale.shape)}")
        print(f"[DEBUG] action_scale sample: {_format_tensor_slice(action_scale, debug_env_id, debug_joint_count)}")
    else:
        print("[DEBUG] action_scale를 자동 탐색하지 못했습니다.")

    default_joint_pos, default_joint_pos_path = (
        _extract_default_joint_pos(robot, env.unwrapped.device) if robot is not None else (None, None)
    )
    if default_joint_pos is not None:
        print(f"[DEBUG] default_joint_pos source: robot.{default_joint_pos_path}")
        print(f"[DEBUG] default_joint_pos shape: {tuple(default_joint_pos.shape)}")
        print(f"[DEBUG] default_joint_pos sample: {_format_tensor_slice(default_joint_pos, debug_env_id, debug_joint_count)}")
    else:
        print("[DEBUG] default_joint_pos를 자동 탐색하지 못했습니다.")

    current_joint_pos, current_joint_pos_path = (
        _extract_current_joint_pos(robot, env.unwrapped.device) if robot is not None else (None, None)
    )
    if current_joint_pos is not None:
        print(f"[DEBUG] current_joint_pos source: robot.{current_joint_pos_path}")
        print(f"[DEBUG] current_joint_pos sample: {_format_tensor_slice(current_joint_pos, debug_env_id, debug_joint_count)}")

    applied_target, applied_target_path = (
        _extract_applied_joint_target(robot, env.unwrapped.device) if robot is not None else (None, None)
    )
    if applied_target is not None:
        print(f"[DEBUG] applied joint target source: robot.{applied_target_path}")
        print(f"[DEBUG] applied joint target sample: {_format_tensor_slice(applied_target, debug_env_id, debug_joint_count)}")
    else:
        print("[DEBUG] 내부 applied joint target buffer는 찾지 못했습니다.")
    # ===========================

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)

            # ===== 디버그 출력 =====
            if args_cli.debug_action_mapping and (timestep % args_cli.debug_print_every == 0):
                print("\n" + "=" * 90)
                print(f"[DEBUG STEP {timestep}]")

                current_joint_pos, _ = (
                    _extract_current_joint_pos(robot, env.unwrapped.device) if robot is not None else (None, None)
                )
                applied_target, _ = (
                    _extract_applied_joint_target(robot, env.unwrapped.device) if robot is not None else (None, None)
                )

                print(f"[DEBUG] raw action shape: {tuple(actions.shape)}")
                print(f"[DEBUG] raw action[{debug_env_id}][: {debug_joint_count}] = "
                      f"{_format_tensor_slice(actions, debug_env_id, debug_joint_count)}")

                if action_scale is not None:
                    scaled_action = actions * action_scale
                    print(f"[DEBUG] scaled_action = action * action_scale")
                    print(f"[DEBUG] scaled_action[{debug_env_id}][: {debug_joint_count}] = "
                          f"{_format_tensor_slice(scaled_action, debug_env_id, debug_joint_count)}")
                else:
                    scaled_action = None
                    print("[DEBUG] action_scale를 모름 -> scaled_action 계산 생략")

                if default_joint_pos is not None and scaled_action is not None:
                    q_target_est = default_joint_pos + scaled_action
                    print("[DEBUG] q_target_est = default_joint_pos + action_scale * action")
                    print(f"[DEBUG] q_target_est[{debug_env_id}][: {debug_joint_count}] = "
                          f"{_format_tensor_slice(q_target_est, debug_env_id, debug_joint_count)}")
                else:
                    q_target_est = None
                    print("[DEBUG] q_target_est 계산 생략")

                if current_joint_pos is not None and default_joint_pos is not None:
                    q_rel = current_joint_pos - default_joint_pos
                    print("[DEBUG] q_rel = current_joint_pos - default_joint_pos")
                    print(f"[DEBUG] current_joint_pos[{debug_env_id}][: {debug_joint_count}] = "
                          f"{_format_tensor_slice(current_joint_pos, debug_env_id, debug_joint_count)}")
                    print(f"[DEBUG] q_rel[{debug_env_id}][: {debug_joint_count}] = "
                          f"{_format_tensor_slice(q_rel, debug_env_id, debug_joint_count)}")

                if applied_target is not None:
                    print(f"[DEBUG] applied_target[{debug_env_id}][: {debug_joint_count}] = "
                          f"{_format_tensor_slice(applied_target, debug_env_id, debug_joint_count)}")

                print("=" * 90 + "\n")
            # ======================

            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()