"""Shared ROA V4-2 USD path and rigid-body name definitions."""

from pathlib import Path


ROA_V4_USD_RELATIVE_PATH = Path("roa_v4-2/roa_v4-2.usd")


def resolve_roa_v4_usd(start: Path) -> str:
    """Resolve the workspace-level ROA V4 USD from an asset module path."""
    candidates = (parent / ROA_V4_USD_RELATIVE_PATH for parent in start.resolve().parents)
    if path := next((candidate for candidate in candidates if candidate.is_file()), None):
        return str(path)
    raise FileNotFoundError(
        f"Could not find {ROA_V4_USD_RELATIVE_PATH} in any parent of {start.resolve()}"
    )


ROA_V4_BODY_NAMES = [
    "base_link",
    "left_hip_1",
    "left_hip_thigh_1",
    "left_thigh_1",
    "left_shin_1",
    "left_ankle_1",
    "left_foot_1",
    "left_TPU_pad_1",
    "right_hip_1",
    "right_hip_thigh_1",
    "right_thigh_1",
    "right_shin_1",
    "right_ankle_1",
    "right_foot_1",
    "right_TPU_pad_1",
    "torso_dummy_1",
    "imu_sensor_1",
    "left_shoulder_1",
    "left_upper_arm_1",
    "left_lower_arm_1",
    "left_hand_1",
    "right_shoulder_1",
    "right_upper_arm_1",
    "right_lower_arm_1",
    "right_hand_1",
]

ROA_V4_FOOT_CONTACT_BODY_NAMES = ["left_TPU_pad_1", "right_TPU_pad_1"]

# Ground contact on any body except the intended TPU pads ends the episode.
ROA_V4_NON_FOOT_CONTACT_BODY_NAMES = [
    body_name for body_name in ROA_V4_BODY_NAMES if body_name not in ROA_V4_FOOT_CONTACT_BODY_NAMES
]

# V4 moves the complete leg chain down 75.211 mm relative to the V3 root.
# Preserve the V3 configured-pose ground clearance until it is measured in Isaac.
ROA_V4_INITIAL_ROOT_HEIGHT = 0.75 + 0.075211
