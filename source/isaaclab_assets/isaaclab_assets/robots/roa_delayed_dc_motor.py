"""ROA DC-motor model with per-environment target-command delay.

The upstream :class:`DelayedPDActuator` delays position/velocity/effort set-points
but does not retain the DC-motor torque-speed saturation model.  ROA needs both:
the rosbag shows a command-path delay, while the motor limits are still relevant
for locomotion training.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class ROADelayedDCMotor(DCMotor):
    """DC motor whose set-points are delayed before PD and motor saturation."""

    cfg: "ROADelayedDCMotorCfg"

    def __init__(self, cfg: "ROADelayedDCMotorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._position_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._velocity_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._effort_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            env_ids = torch.arange(self._num_envs, device=self._device)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(len(env_ids),),
            dtype=torch.long,
            device=self._device,
        )
        self._position_delay.set_time_lag(time_lags, env_ids)
        self._velocity_delay.set_time_lag(time_lags, env_ids)
        self._effort_delay.set_time_lag(time_lags, env_ids)
        self._position_delay.reset(env_ids)
        self._velocity_delay.reset(env_ids)
        self._effort_delay.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action.joint_positions = self._position_delay.compute(control_action.joint_positions)
        control_action.joint_velocities = self._velocity_delay.compute(control_action.joint_velocities)
        control_action.joint_efforts = self._effort_delay.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class ROADelayedDCMotorCfg(DCMotorCfg):
    """Configuration for :class:`ROADelayedDCMotor`."""

    class_type: type = ROADelayedDCMotor
    min_delay: int = 0
    max_delay: int = 0

