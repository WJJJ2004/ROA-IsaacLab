"""Register Gym environments for Robit Adult Humanoid."""

"""
How to use:
1. Training from checkpoint (without rendering):


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v2 \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000 \
  --resume \
  --load_run 2026-06-22_15-13-25 \
  --checkpoint model_4000.pt \

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v2 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --video \
  --video_interval 1000 \
  --video_length 1000 


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-adult-v2 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --video \
  --video_interval 1000 \
  --video_length 1000 


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-adult-v2-Play \
  --headless \
  --num_envs 100 \
  --resume \
  --load_run 2026-06-25_15-45-39 \
  --checkpoint /home/robit/isaaclab_230/logs/rsl_rl/adult_v2_rough/2026-06-25_15-45-39/model_30000.pt \
  --video \
  --video_length 1000



  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-adult-v2-Play \
  --num_envs 100 \
  --resume \
  --load_run 2026-04-28_21-19-30 \
  --checkpoint /home/robit/isaaclab_230/logs/rsl_rl/adult_v2_rough/2026-04-28_21-19-30/model_34000.pt
"""


import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-adult-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:adult_v2_FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adult_v2_FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-adult-v2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:adult_v2_FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adult_v2_FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-adult-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:adult_v2_RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adult_v2_RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-adult-v2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:adult_v2_RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:adult_v2_RoughPPORunnerCfg",
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