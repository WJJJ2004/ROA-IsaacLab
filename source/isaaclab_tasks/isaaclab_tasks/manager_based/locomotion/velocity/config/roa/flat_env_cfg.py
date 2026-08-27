# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat environment config for ROA."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import ROARoughEnvCfg


@configclass
class ROAFlatEnvCfg(ROARoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.enable_randomization = False
        self.curriculum.terrain_levels = None

        # Rewards

        # NOTE : 
        # ALL REWARD TERMS ARE MODIFIED IN ROUGHT ENV CFG,
        # SO NO NEED TO MODIFY HERE. JUST KEEPING THIS COMMENT FOR REFERENCE.

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


class ROAFlatEnvCfg_PLAY(ROAFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.curriculum = None

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None