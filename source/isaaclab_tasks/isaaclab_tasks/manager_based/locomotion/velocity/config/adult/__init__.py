"""Register Gym environments for Robit Adult Humanoid."""

"""
How to use:
1. Training from checkpoint (without rendering):

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-adult-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 10000 \
  --resume True\
  --load_run 2026-01-16_11-26-22 \
  --checkpoint model_10149.pt

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000 \
  --resume True\
  --load_run 2026-02-02_01-40-59 \
  --checkpoint model_134000.pt

2026-01-30_10-40-43

2. Play trained model:

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/print_name.py \
  --task Isaac-Velocity-Flat-adult-v0-Play \
  --headless \
  --num_envs 1 \
  --resume True \
  --load_run 2026-01-13_12-16-59 \
  --checkpoint /workspace/isaaclab/logs/rsl_rl/adult_flat/2026-01-13_12-16-59/model_7000.pt \
  --video \
  --video_length 1000


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/print_imu.py \
  --task Isaac-Velocity-Flat-adult-v0-Play \
  --headless \
  --num_envs 1 \
  --resume True \
  --load_run 2026-01-13_12-16-59 \
  --checkpoint /workspace/isaaclab/logs/rsl_rl/adult_flat/2026-01-13_12-16-59/model_7000.pt \
  --video \
  --video_length 1000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-adult-v0-Play \
  --headless \
  --num_envs 100 \
  --resume True \
  --load_run 2026-02-02_02-43-20 \
  --checkpoint /workspace/isaaclab/logs/rsl_rl/adult_rough/2026-02-02_02-43-20/model_157350.pt \
  --video \
  --video_length 1000


3. fine-tuning from a trained model:

# create a symbolic link to the previous training run
cd /workspace/isaaclab/logs/rsl_rl/adult_rough
ln -s ../adult_flat/2026-01-14_11-20-11 2026-01-14_11-20-11
cd /workspace/isaaclab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 10000 \
  --resume False\
  --load_run 2026-01-14_11-20-11 \
  --checkpoint model_11400.pt \
  --video
"""


import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-adult-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:adultFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-adult-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:adultFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-adult-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:adultRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-adult-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:adultRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
# gym.register(
#     id="Isaac-Velocity-Rough-adult-RNN-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:adultRoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughRNNPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-adult-RNN-v0-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:adultRoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughRNNPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-adult-LSTM-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:adultRoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughLSTMPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-adult-LSTM-v0-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:adultRoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughLSTMPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )


# gym.register(
#     id="Isaac-Velocity-Rough-adult-v0-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:adultRoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adultRoughPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )