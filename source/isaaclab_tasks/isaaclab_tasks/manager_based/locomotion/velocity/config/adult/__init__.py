"""Register Gym environments for Robit Adult Humanoid."""

"""
How to use:
1. Training from checkpoint (without rendering):

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000 \
  --resume \
  --load_run 2026-06-24_21-16-54 \
  --checkpoint model_9000.pt

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --video \
  --video_interval 1000 \
  --video_length 1000 


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --video \
  --video_interval 1000 \
  --video_length 1000 


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-adult-v0-Play \
  --headless \
  --num_envs 100 \
  --resume \
  --load_run 2026-06-25_10-13-14 \
  --checkpoint /home/robit/isaaclab_230/logs/rsl_rl/adult_rough/2026-06-25_10-13-14/model_11500.pt \
  --video \
  --video_length 100

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py   --task Isaac-Velocity-Rough-adult-v0-Play   --headless   --num_envs 100   --resume   --load_run 2026-05-22_22-06-28   --checkpoint /home/robit/isaaclab_230/logs/rsl_rl/adult_rough/2026-05-22_22-06-28/model_25000.pt   --video   --video_length 1000


  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-adult-v0-Play \
  --num_envs 100 \
  --resume \
  --load_run 2026-06-07_16-49-03 \
  --checkpoint /home/robit/isaaclab_230/logs/rsl_rl/adult_rough/2026-06-07_16-49-03/model_27000.pt
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