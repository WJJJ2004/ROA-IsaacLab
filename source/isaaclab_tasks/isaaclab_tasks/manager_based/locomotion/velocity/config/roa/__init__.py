"""Register Gym environments for Robit ROA Humanoid."""

"""
2026-08-20_11-02-47/model_33000.pt

How to use:
1. Training from checkpoint (without rendering):

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-ROA-v0 \
  --headless \
  --num_envs 1024

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-ROA-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000 \
  --resume \
  --load_run 2026-08-27_02-11-07 \
  --checkpoint model_6700.pt \
  --video \
  --video_interval 10000 \
  --video_length 250   



./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-ROA-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --video \
  --video_interval 10000 \
  --video_length 500   


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-ROA-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 300000 \
  --video \
  --video_interval 20000 \
  --video_length 250   


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ROA-v0-Play \
  --headless \
  --num_envs 100 \
  --resume \
  --load_run 2026-08-26_16-27-43 \
  --checkpoint model_33000.pt \
  --video \
  --video_length 100


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ROA-v0-Play \
  --headless \
  --num_envs 1048 \
  --resume \
  --load_run 2026-08-20_11-02-47 \
  --checkpoint /home/robit/isaaclab_230/logs/rsl_rl/ROA_rough/2026-08-20_11-02-47/model_33000.pt \
  --video \
  --video_length 1000
"""


import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-ROA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ROAFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROAFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-ROA-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ROAFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROAFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-ROA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ROARoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-ROA-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ROARoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
# gym.register(
#     id="Isaac-Velocity-Rough-ROA-RNN-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:ROARoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughRNNPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-ROA-RNN-v0-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:ROARoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughRNNPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-ROA-LSTM-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:ROARoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughLSTMPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-ROA-LSTM-v0-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_rnn_env_cfg:ROARoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughLSTMPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )


# gym.register(
#     id="Isaac-Velocity-Rough-ROA-v0-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ROARoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ROARoughPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )