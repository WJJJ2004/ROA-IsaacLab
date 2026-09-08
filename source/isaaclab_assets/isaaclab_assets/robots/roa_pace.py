# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROA asset configuration with PACE actuators for Isaac Lab training.

The base robot configuration is imported from the PACE project.
Only the actuator dictionary is replaced for the training environment.
"""

import math

from isaaclab.assets.articulation import ArticulationCfg
from pace_sim2real.tasks.manager_based.pace.assets.roa.roa import ROA_CFG

from .roa_delayed_dc_motor import ROADelayedDCMotorCfg

# RVIZ TUNED VALUE (08/12/2026 UPDATE)
_INIT_JOINT_POS = {
    # "torso_yaw": 0.0,

    "left_hip_pitch": -0.42616745829582214,
    "left_hip_roll": -0.04507143050432205,
    "left_hip_yaw": -0.13962633907794952,
    "left_knee_pitch": 0.767944872379303,
    "left_ankle_pitch": -0.36651915311813354,
    "left_ankle_roll": 0.0,

    "right_hip_pitch": 0.42616745829582214,
    "right_hip_roll": 0.04507143050432205,
    "right_hip_yaw": 0.13962633907794952,
    "right_knee_pitch": -0.767944872379303,
    "right_ankle_pitch": 0.36651915311813354,
    "right_ankle_roll": 0.0,
}


# ******************************** ROA PACE CONFIGURATION ********************************

ROA_JOINT_ORDER = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
]

# Policy/ONNX order used by the controller and MuJoCo sim-to-sim.  Keep this
# distinct from ROA_JOINT_ORDER above, which is the PACE identification dataset
# order (left leg followed by right leg).
ROA_POLICY_JOINT_ORDER = [
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_knee_pitch",
    "right_knee_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
]


RSU_STATIC_FRICTION = 0.0
RSU_DYNAMIC_FRICTION = 0.0
RSU_VISCOUS_FRICTION = 0.0

GLOBAL_ARMATURE_SCALE = 1.0

# Delay is sampled per environment on reset.  At the 200 Hz physics rate,
# 2--5 steps correspond to 10--25 ms.  The direct-joint chirp responses showed
# approximately 21 ms of command-to-feedback dead time.
DIRECT_MIN_DELAY_STEPS = 2
DIRECT_MAX_DELAY_STEPS = 5

# The RSU motor path contributes approximately 10--12 ms; target selection and
# solver ZOH bring target-to-feedback latency to roughly 18 ms before virtual
# state reconstruction.  The observation-side delay is modeled in the task cfg.
RSU_MIN_DELAY_STEPS = 2
RSU_MAX_DELAY_STEPS = 4

# -----------------------------------------------------------------------------
# RobStride RS03
# 적용 관절:
#   - hip_roll
#   - hip_yaw
# -----------------------------------------------------------------------------
ROA_RS03_PACE_ACTUATOR_CFG = ROADelayedDCMotorCfg(
    joint_names_expr=[
        ".*_hip_roll",
        ".*_hip_yaw",
    ],
    saturation_effort=60.0,
    effort_limit=42.0,
    velocity_limit=18.849,

    stiffness={
        ".*_hip_roll": 200.0,
        ".*_hip_yaw": 100.0,
    },

    damping={
        ".*_hip_roll": 26.387,
        ".*_hip_yaw": 3.419,
    },
    armature={
        ".*": 0.02,
    },
    
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},

    min_delay=DIRECT_MIN_DELAY_STEPS,
    max_delay=DIRECT_MAX_DELAY_STEPS,
)


# -----------------------------------------------------------------------------
# RobStride RS04
# 적용 관절:
#   - hip_pitch: 기존 하드코딩 값 유지
#   - knee_pitch: 최신 PACE 식별값 적용
# -----------------------------------------------------------------------------
ROA_RS04_PACE_ACTUATOR_CFG = ROADelayedDCMotorCfg(
    joint_names_expr=[
        ".*_hip_pitch",
        ".*_knee_pitch",
    ],
    saturation_effort=120.0,
    effort_limit=84.0,
    velocity_limit=17.488,

    stiffness={
        ".*_hip_pitch": 150.0,
        ".*_knee_pitch": 150.0,
    },

    damping={
        ".*_hip_pitch": 24.722,
        ".*_knee_pitch": 8.654,
    },
    armature={
        ".*": 0.02,
    },
    
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},

    min_delay=DIRECT_MIN_DELAY_STEPS,
    max_delay=DIRECT_MAX_DELAY_STEPS,
)


# -----------------------------------------------------------------------------
# RSU
# ankle_pitch / ankle_roll에 최신 PACE 식별값을 적용한다.
# -----------------------------------------------------------------------------
RSU_KVALUE = 1.37  # RSU K-value ratio (ankle_roll / ankle_pitch)

ROA_ROLL_RSU_PACE_ACTUATOR_CFG = ROADelayedDCMotorCfg(
    joint_names_expr=[
        ".*_ankle_roll",
    ],
    saturation_effort=26.0,
    effort_limit=26.0,
    velocity_limit=5.0,

    stiffness={
        ".*_ankle_roll": 25.0 * RSU_KVALUE,
    },

    damping={
        ".*_ankle_roll": 1.2 * RSU_KVALUE,
    },
    
    armature={
        ".*_ankle_roll": 0.05847,
    },
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    min_delay=RSU_MIN_DELAY_STEPS,
    max_delay=RSU_MAX_DELAY_STEPS,
)

ROA_PITCH_RSU_PACE_ACTUATOR_CFG = ROADelayedDCMotorCfg(
    joint_names_expr=[
        ".*_ankle_pitch",
    ],
    saturation_effort=20.5,
    effort_limit=20.5,
    velocity_limit=5.0,

    stiffness={
        ".*_ankle_pitch": 25.0,
    },

    damping={
        ".*_ankle_pitch": 1.2,
    },
    
    armature={
        ".*_ankle_pitch": 0.04255,
    },
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    min_delay=RSU_MIN_DELAY_STEPS,
    max_delay=RSU_MAX_DELAY_STEPS,
)

# *****************************************************************************


_ROA_PACE_ACTUATORS = {
    "robstride_03": ROA_RS03_PACE_ACTUATOR_CFG,
    "robstride_04": ROA_RS04_PACE_ACTUATOR_CFG,
    "rsu_roll": ROA_ROLL_RSU_PACE_ACTUATOR_CFG,
    "rsu_pitch": ROA_PITCH_RSU_PACE_ACTUATOR_CFG,
}


# Reuse every robot/physics setting from the PACE ROA asset and replace only
# the actuator dictionary.
ROA_CFG = ROA_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos=_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators=_ROA_PACE_ACTUATORS,
)
