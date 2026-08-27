# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """ROA asset configuration with PACE actuators for Isaac Lab training.

# The base robot configuration is imported from the PACE project.
# Only the actuator dictionary is replaced for the training environment.
# """

# import math

# from isaaclab.assets.articulation import ArticulationCfg
# from pace_sim2real.utils import PaceDCMotorCfg
# from pace_sim2real.tasks.manager_based.pace.assets.roa.roa import ROA_CFG


# # -----------------------------------------------------------------------------
# # RobStride RS03
# #   - hip_roll
# #   - hip_yaw
# #
# # Current HIP PACE identification values are hard-coded directly here.
# # -----------------------------------------------------------------------------

# # RVIZ TUNED VALUE (08/12/2026 UPDATE)
# _INIT_JOINT_POS = {
#     # "torso_yaw": 0.0,

#     "left_hip_pitch": -0.42616745829582214,
#     "left_hip_roll": -0.04507143050432205,
#     "left_hip_yaw": -0.13962633907794952,
#     "left_knee_pitch": 0.767944872379303,
#     "left_ankle_pitch": -0.36651915311813354,
#     "left_ankle_roll": 0.0,

#     "right_hip_pitch": 0.42616745829582214,
#     "right_hip_roll": 0.04507143050432205,
#     "right_hip_yaw": 0.13962633907794952,
#     "right_knee_pitch": -0.767944872379303,
#     "right_ankle_pitch": 0.36651915311813354,
#     "right_ankle_roll": 0.0,
# }

# # # RVIZ TUNED VALUE (08/11/2026 UPDATE)
# # _INIT_JOINT_POS = {
# #     # "torso_yaw": 0.0,

# #     "left_hip_pitch": -0.31359851360321045,
# #     "left_hip_roll": -0.024696864187717438,
# #     "left_hip_yaw": -0.0872664600610733,
# #     "left_knee_pitch": 0.5235987901687622,
# #     "left_ankle_pitch": -0.22689279913902283,
# #     "left_ankle_roll": 0.0,

# #     "right_hip_pitch": 0.31359851360321045,
# #     "right_hip_roll": 0.024696864187717438,
# #     "right_hip_yaw": 0.0872664600610733,
# #     "right_knee_pitch": -0.5235987901687622,
# #     "right_ankle_pitch": 0.22689279913902283,
# #     "right_ankle_roll": 0.0,
# # }

# # Original hard-coded values for ROA PACE
# # _INIT_JOINT_POS = {
# #    # "torso_yaw": 0.0,

# #     "left_hip_pitch": math.radians(-20.0), #12
# #     "left_hip_roll": 0.0,
# #     "left_hip_yaw": 0.0,
# #     "left_knee_pitch": math.radians(50.0), #16
# #     "left_ankle_pitch": math.radians(-30.0),
# #     "left_ankle_roll": 0.0,

# #     "right_hip_pitch": math.radians(20.0), #11
# #     "right_hip_roll": 0.0,
# #     "right_hip_yaw": 0.0,
# #     "right_knee_pitch": math.radians(-50.0), #17
# #     "right_ankle_pitch": math.radians(30.0),
# #     "right_ankle_roll": 0.0,
# # }

# # ******************************** ROA PACE CONFIGURATION ********************************

# ROA_JOINT_ORDER = [
#     "left_hip_pitch",
#     "left_hip_roll",
#     "left_hip_yaw",
#     "left_knee_pitch",
#     "left_ankle_pitch",
#     "left_ankle_roll",
#     "right_hip_pitch",
#     "right_hip_roll",
#     "right_hip_yaw",
#     "right_knee_pitch",
#     "right_ankle_pitch",
#     "right_ankle_roll",
# ]


# # -----------------------------------------------------------------------------
# # 기본 파라미터
# # -----------------------------------------------------------------------------
# # HIP 3축은 기존 하드코딩 값을 유지하고,
# # knee/ankle 3자유도만 최신 식별값으로 덮어쓴다.
# RSU_STATIC_FRICTION = 0.0
# RSU_DYNAMIC_FRICTION = 0.0
# RSU_VISCOUS_FRICTION = 0.0

# GLOBAL_ARMATURE_SCALE = 1.0

# GLOBAL_MAX_DELAY = 1


# # -----------------------------------------------------------------------------
# # PACE 식별 결과: HIP 관절
# #
# # ROA_JOINT_ORDER 기준:
# #   0: left_hip_pitch
# #   1: left_hip_roll
# #   2: left_hip_yaw
# #   6: right_hip_pitch
# #   7: right_hip_roll
# #   8: right_hip_yaw
# # -----------------------------------------------------------------------------
# # NOTE: 업데이트 08/03/2026
# HIP_ARMATURE = {
#     "left_hip_pitch": 0.002385251224040985,
#     "left_hip_roll": 0.4547957479953766,
#     "left_hip_yaw": 0.0012947353534400463,
#     "right_hip_pitch": 0.013781693764030933,
#     "right_hip_roll": 0.4199865758419037,
#     "right_hip_yaw": 0.00041497970232740045,
# }

# HIP_VISCOUS_FRICTION = {
#     "left_hip_pitch": 2.6,
#     "left_hip_roll": 4.519671440124512,
#     "left_hip_yaw": 0.039075493812561035,
#     "right_hip_pitch": 2.6,
#     "right_hip_roll": 5.7763872146606445,
#     "right_hip_yaw": 0.011155962944030762,
# }

# # HIP_VISCOUS_FRICTION = {
# #     "left_hip_pitch": 4.33404541015625,
# #     "left_hip_roll": 4.519671440124512,
# #     "left_hip_yaw": 0.039075493812561035,
# #     "right_hip_pitch": 0.9965003728866577,
# #     "right_hip_roll": 5.7763872146606445,
# #     "right_hip_yaw": 0.011155962944030762,
# # }

# HIP_STATIC_DYNAMIC_FRICTION = {
#     "left_hip_pitch": 0.11539211869239807,
#     "left_hip_roll": 0.2831701338291168,
#     "left_hip_yaw": 0.0678943544626236,
#     "right_hip_pitch": 0.23442913591861725,
#     "right_hip_roll": 0.23406176269054413,
#     "right_hip_yaw": 0.21062113344669342,
# }

# HIP_ENCODER_BIAS = {
#     "left_hip_pitch": 0.0,
#     "left_hip_roll": 0.0,
#     "left_hip_yaw": 0.0,
#     "right_hip_pitch": 0.0,
#     "right_hip_roll": 0.0,
#     "right_hip_yaw": 0.0,
# }

# # -----------------------------------------------------------------------------
# # PACE 식별 결과: KNEE / ANKLE 관절
# #
# # HIP 3축은 위의 기존 하드코딩 값을 그대로 유지하고,
# # 아래 3자유도(knee_pitch, ankle_pitch, ankle_roll)만 최신 결과로 덮어쓴다.
# # -----------------------------------------------------------------------------
# # NOTE: 업데이트 08/03/2026
# KNEE_ARMATURE = {
#     "left_knee_pitch": 5.1842042012140155e-05,
#     "right_knee_pitch": 2.8864680643891916e-05,
# }

# KNEE_VISCOUS_FRICTION = {
#     "left_knee_pitch": 0.000673830509185791,
#     "right_knee_pitch": 0.00018525123596191406,
# }

# KNEE_STATIC_DYNAMIC_FRICTION = {
#     "left_knee_pitch": 0.0029055774211883545,
#     "right_knee_pitch": 0.0015920400619506836,
# }

# KNEE_ENCODER_BIAS = {
#     "left_knee_pitch": 0.0,
#     "right_knee_pitch": 0.0,
# }


# # -----------------------------------------------------------------------------
# # RobStride RS03
# # 적용 관절:
# #   - hip_roll
# #   - hip_yaw
# # -----------------------------------------------------------------------------
# ROA_RS03_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
#     joint_names_expr=[
#         ".*_hip_roll",
#         ".*_hip_yaw",
#     ],
#     saturation_effort=60.0,
#     effort_limit=42.0,
#     velocity_limit=18.849,

#     stiffness={
#         ".*_hip_roll": 200.0,
#         ".*_hip_yaw": 100.0,
#     },

#     damping={
#         ".*_hip_roll": 26.387,
#         ".*_hip_yaw": 3.419,
#     },

#     armature={
#         "left_hip_roll": HIP_ARMATURE["left_hip_roll"],
#         "left_hip_yaw": HIP_ARMATURE["left_hip_yaw"],
#         "right_hip_roll": HIP_ARMATURE["right_hip_roll"],
#         "right_hip_yaw": HIP_ARMATURE["right_hip_yaw"],
#     },

#     encoder_bias={
#         ".*": 0.0,
#     },
    
#     friction={
#         "left_hip_roll": HIP_STATIC_DYNAMIC_FRICTION["left_hip_roll"],
#         "left_hip_yaw": HIP_STATIC_DYNAMIC_FRICTION["left_hip_yaw"],
#         "right_hip_roll": HIP_STATIC_DYNAMIC_FRICTION["right_hip_roll"],
#         "right_hip_yaw": HIP_STATIC_DYNAMIC_FRICTION["right_hip_yaw"],
#     },

#     dynamic_friction={
#         "left_hip_roll": HIP_STATIC_DYNAMIC_FRICTION["left_hip_roll"],
#         "left_hip_yaw": HIP_STATIC_DYNAMIC_FRICTION["left_hip_yaw"],
#         "right_hip_roll": HIP_STATIC_DYNAMIC_FRICTION["right_hip_roll"],
#         "right_hip_yaw": HIP_STATIC_DYNAMIC_FRICTION["right_hip_yaw"],
#     },

#     viscous_friction={
#         "left_hip_roll": HIP_VISCOUS_FRICTION["left_hip_roll"],
#         "left_hip_yaw": HIP_VISCOUS_FRICTION["left_hip_yaw"],
#         "right_hip_roll": HIP_VISCOUS_FRICTION["right_hip_roll"],
#         "right_hip_yaw": HIP_VISCOUS_FRICTION["right_hip_yaw"],
#     },

#     max_delay=GLOBAL_MAX_DELAY,
# )


# # -----------------------------------------------------------------------------
# # RobStride RS04
# # 적용 관절:
# #   - hip_pitch: 기존 하드코딩 값 유지
# #   - knee_pitch: 최신 PACE 식별값 적용
# # -----------------------------------------------------------------------------
# ROA_RS04_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
#     joint_names_expr=[
#         ".*_hip_pitch",
#         ".*_knee_pitch",
#     ],
#     saturation_effort=120.0,
#     effort_limit=84.0,
#     velocity_limit=17.488,

#     stiffness={
#         ".*_hip_pitch": 150.0,
#         ".*_knee_pitch": 150.0,
#     },

#     damping={
#         ".*_hip_pitch": 24.722,
#         ".*_knee_pitch": 8.654,
#     },

#     armature={
#         # HIP: 식별값
#         "left_hip_pitch": HIP_ARMATURE["left_hip_pitch"],
#         "right_hip_pitch": HIP_ARMATURE["right_hip_pitch"],

#         # KNEE: 최신 식별값
#         "left_knee_pitch": KNEE_ARMATURE["left_knee_pitch"],
#         "right_knee_pitch": KNEE_ARMATURE["right_knee_pitch"],
#     },

#     encoder_bias={
#         ".*": 0.0,
#     },

#     friction={
#         # HIP: 식별값
#         "left_hip_pitch": HIP_STATIC_DYNAMIC_FRICTION["left_hip_pitch"],
#         "right_hip_pitch": HIP_STATIC_DYNAMIC_FRICTION["right_hip_pitch"],

#         # KNEE: 최신 식별값
#         "left_knee_pitch": KNEE_STATIC_DYNAMIC_FRICTION["left_knee_pitch"],
#         "right_knee_pitch": KNEE_STATIC_DYNAMIC_FRICTION["right_knee_pitch"],
#     },

#     dynamic_friction={
#         # HIP: 식별값
#         "left_hip_pitch": HIP_STATIC_DYNAMIC_FRICTION["left_hip_pitch"],
#         "right_hip_pitch": HIP_STATIC_DYNAMIC_FRICTION["right_hip_pitch"],

#         # KNEE: 최신 식별값
#         "left_knee_pitch": KNEE_STATIC_DYNAMIC_FRICTION["left_knee_pitch"],
#         "right_knee_pitch": KNEE_STATIC_DYNAMIC_FRICTION["right_knee_pitch"],
#     },

#     viscous_friction={
#         # HIP: 식별값
#         "left_hip_pitch": HIP_VISCOUS_FRICTION["left_hip_pitch"],
#         "right_hip_pitch": HIP_VISCOUS_FRICTION["right_hip_pitch"],

#         # KNEE: 최신 식별값
#         "left_knee_pitch": KNEE_VISCOUS_FRICTION["left_knee_pitch"],
#         "right_knee_pitch": KNEE_VISCOUS_FRICTION["right_knee_pitch"],
#     },

#     max_delay=GLOBAL_MAX_DELAY,
# )


# # -----------------------------------------------------------------------------
# # RSU
# # ankle_pitch / ankle_roll에 최신 PACE 식별값을 적용한다.
# # -----------------------------------------------------------------------------
# RSU_KVALUE = 1.37  # RSU K-value ratio (ankle_roll / ankle_pitch)

# ROA_RSU_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
#     joint_names_expr=[
#         ".*_ankle_pitch",
#         ".*_ankle_roll",
#     ],
#     saturation_effort=11.9,
#     effort_limit=11.9,
#     velocity_limit=5.0,

#     stiffness={
#         ".*_ankle_pitch": 25.0,
#         ".*_ankle_roll": 25.0 * RSU_KVALUE,
#     },

#     damping={
#         ".*_ankle_pitch": 1.2,
#         ".*_ankle_roll": 1.2 * RSU_KVALUE,
#     },
    
#     armature={
#         ".*": 0.02,
#     },
#     friction={
#         ".*": RSU_STATIC_FRICTION,
#     },
#     dynamic_friction={
#         ".*": RSU_DYNAMIC_FRICTION,
#     },
#     viscous_friction={
#         ".*": RSU_VISCOUS_FRICTION,
#     },

#     encoder_bias={
#         ".*": 0.0,
#     },
#     max_delay=GLOBAL_MAX_DELAY,
# )

# # *****************************************************************************


# _ROA_PACE_ACTUATORS = {
#     "robstride_03": ROA_RS03_PACE_ACTUATOR_CFG,
#     "robstride_04": ROA_RS04_PACE_ACTUATOR_CFG,
#     "rsu": ROA_RSU_PACE_ACTUATOR_CFG,
# }


# # Reuse every robot/physics setting from the PACE ROA asset and replace only
# # the actuator dictionary.
# ROA_CFG = ROA_CFG.replace(
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.75),
#         joint_pos=_INIT_JOINT_POS,
#         joint_vel={".*": 0.0},
#     ),
#     actuators=_ROA_PACE_ACTUATORS,
# )