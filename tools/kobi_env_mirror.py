"""NumPy / MuJoCo-C mirror of the Kobi training environment (our modified copy of the Open_Duck_Playground joystick env).

Source of truth: the training environment package (joystick.py + base.py of our Kobi fork of Open_Duck_Playground). It is NOT part of
this repository because the upstream repository has no license; the "l.NNN" numbers below refer to that private fork.

Order of operations of one control step (copied from Joystick.step, with action delay 0 and no push):
    1. imitation_i = (imitation_i + 1) % nb_steps_in_period, imitation_phase = (cos, sin)(2 pi i / nb)      [step l.516-533]
    2. motor targets from the action and the CURRENT command slots 3..6, rate-limited around the previous targets [l.597-618]
    3. 10 physics substeps with the targets as ctrl                                                          [l.620]
    4. feet contact (sole geom vs floor geom), termination test (upvector z < cos 60 deg = 0.5, or NaN)     [l.624-637, 726]
    5. command slots 3..6 are replaced by the gesture target of the NEXT step                                [l.657-672]
    6. observation = _get_obs with the OLD action history (last_act is the action of the step BEFORE this one) [l.674]
    7. action history shift, last_act = this action                                                          [l.679-681]
So the observation seen by the policy at control step k holds a(k-2), a(k-3), a(k-4) in its three action slots, never the
latest action a(k-1) (upstream mujoco_infer.py feeds a(k-1), a(k-2), a(k-3)). Mirrored on purpose.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKLASH_XML = REPO_ROOT / "printed_face" / "mujoco" / "kobi_scene_backlash.xml"
REFERENCE_PKL = REPO_ROOT / "policy" / "polynomial_coefficients_kobi.pkl"      # reference gait v2
HOME_KEY = "home"
FREE_JOINT_TYPE = mujoco.mjtJoint.mjJNT_FREE

LEG_JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
UPPER_JOINTS = ("neck_yaw", "neck_pitch", "shoulder_left", "shoulder_right")   # command slots 3..6
ACTUATOR_ORDER = LEG_JOINTS + UPPER_JOINTS
NUM_LEGS, NUM_UPPER, NUM_ACT = len(LEG_JOINTS), len(UPPER_JOINTS), len(ACTUATOR_ORDER)
NUM_MOVE = 3
FEET_GEOMS = ("left_foot_bottom_tpu", "right_foot_bottom_tpu")
FLOOR_GEOM = "floor"
GYRO_SENSOR, ACCEL_SENSOR, UP_SENSOR = "gyro", "accelerometer", "upvector"
# The env writes `accelerometer.at[0].set(accelerometer[0] + 1.3)` and drops the result (JAX arrays are immutable), so the
# training observation has NO 1.3 offset. Upstream mujoco_infer.py adds it. Mirror the env.
ACCEL_X_OFFSET = 0.0
# Termination (joystick.py FALL_TILT_DEG): the trunk tilts more than 60 deg, i.e. the upvector sensor z < cos(60 deg).
FALL_TILT_DEG = 60.0
FALL_UPVECTOR_Z = float(np.cos(np.deg2rad(FALL_TILT_DEG)))


@dataclass(frozen=True)
class EnvConfig:
    """Values of joystick.default_config() that matter for playback (re-checked against the source by check_env_mirror)."""
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    action_scale: float = 0.25
    dof_vel_scale: float = 0.05
    max_motor_velocity: float = 5.24
    torque_limit_leg_neck: float = 1.91
    torque_limit_shoulder: float = 0.20
    residual_neck_yaw: float = 0.10
    residual_neck_pitch: float = 0.05
    residual_shoulder: float = 0.15
    noise_gyro: float = 0.1
    noise_accelerometer: float = 0.05
    noise_hip: float = 0.03
    noise_knee: float = 0.05
    noise_ankle: float = 0.08
    noise_upper: float = 0.05
    noise_joint_vel: float = 2.5

    @property
    def n_substeps(self) -> int:
        return int(round(self.ctrl_dt / self.sim_dt))

    @property
    def residual_scale(self) -> np.ndarray:
        return np.array([self.residual_neck_yaw, self.residual_neck_pitch, self.residual_shoulder, self.residual_shoulder])


# (name, size, expression in joystick._get_obs that produces it) in observation order
OBS_LAYOUT: tuple[tuple[str, int, str], ...] = (
    ("gyro", 3, "noisy_gyro"),
    ("accelerometer", 3, "noisy_accelerometer"),
    ("command", 7, 'info["command"]'),
    ("joint_angles_minus_home", NUM_ACT, "noisy_joint_angles - self._default_actuator"),
    ("joint_velocity_x0.05", NUM_ACT, "noisy_joint_vel * self._config.dof_vel_scale"),
    ("last_action", NUM_ACT, 'info["last_act"]'),
    ("last_last_action", NUM_ACT, 'info["last_last_act"]'),
    ("last_last_last_action", NUM_ACT, 'info["last_last_last_act"]'),
    ("motor_targets", NUM_ACT, 'info["motor_targets"]'),
    ("foot_contacts", 2, "contact"),
    ("imitation_phase", 2, 'info["imitation_phase"]'),
)
OBS_SIZE = sum(size for _, size, _ in OBS_LAYOUT)


def obs_slices() -> dict[str, slice]:
    out, start = {}, 0
    for name, size, _ in OBS_LAYOUT:
        out[name] = slice(start, start + size)
        start += size
    return out


def layout_table() -> str:
    lines = ["idx range   size  name"]
    for (name, size, expr), sl in zip(OBS_LAYOUT, obs_slices().values()):
        lines.append(f"[{sl.start:3d}:{sl.stop:3d}]  {size:3d}  {name:24s} <- {expr}")
    lines.append(f"total {OBS_SIZE}")
    return "\n".join(lines)


def reference_period_steps(pkl: Path = REFERENCE_PKL) -> int:
    """nb_steps_in_period of the Kobi reference (PolyReferenceMotion: int(period * fps)); the local file is our own output."""
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    first = data[next(iter(data))]
    return int(first["period"] * first["fps"])


@dataclass
class StepResult:
    obs: np.ndarray
    done: bool
    motor_targets: np.ndarray
    action_used: np.ndarray


@dataclass
class KobiPlaybackEnv:
    xml: Path = BACKLASH_XML
    cfg: EnvConfig = field(default_factory=EnvConfig)
    noise_level: float = 0.0          # 1.0 reproduces the training observation noise, 0.0 is the clean playback default
    action_delay_steps: int = 0       # training draws 0..2; playback uses 0
    seed: int = 0
    substep_hook: Callable[[mujoco.MjData], None] | None = None   # called after every physics substep (evaluation probes)

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.model.opt.timestep = self.cfg.sim_dt
        m = self.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
        if tuple(names) != ACTUATOR_ORDER:
            raise ValueError(f"actuator order {names} differs from the Kobi order {list(ACTUATOR_ORDER)}")
        self.apply_torque_limits()
        self.data = mujoco.MjData(m)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.act_joint_ids = np.array([jid(n) for n in names])
        self.act_qpos = m.jnt_qposadr[self.act_joint_ids].astype(int)
        self.act_dof = m.jnt_dofadr[self.act_joint_ids].astype(int)
        joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        free = [j for j in range(m.njnt) if m.jnt_type[j] == FREE_JOINT_TYPE]
        self.free_qpos = int(m.jnt_qposadr[free[0]])
        self.backlash_names = [n for j, n in enumerate(joint_names) if n not in names and j != free[0]]
        self.backlash_qpos = np.array([m.jnt_qposadr[jid(n)] for n in self.backlash_names], dtype=int)
        # env: actuators without a "<name>_backlash" joint get a zero inserted at their index (base.py l.157-161)
        self.backlash_idx_to_add = [i for i, n in enumerate(names) if n + "_backlash" not in self.backlash_names]
        self.lowers, self.uppers = m.jnt_range[self.act_joint_ids].T
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
        self.key_id = key
        self.home_qpos = np.array(m.key_qpos[key])
        self.default_actuator = np.array(m.key_ctrl[key])
        sid = lambda n: int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)])
        self.gyro_adr, self.accel_adr, self.up_adr = sid(GYRO_SENSOR), sid(ACCEL_SENSOR), sid(UP_SENSOR)
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.feet_geoms = [gid(n) for n in FEET_GEOMS]
        self.floor_geom = gid(FLOOR_GEOM)
        self.nb_steps_in_period = reference_period_steps()
        self.noise_scale = self._qpos_noise_scale()
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    # ---- model preparation
    def apply_torque_limits(self) -> None:
        """base.apply_torque_limits: forcerange +-1.91 for legs and neck, +-0.20 for the shoulders, forcelimited on."""
        for name in ACTUATOR_ORDER:
            limit = self.cfg.torque_limit_shoulder if name.startswith("shoulder") else self.cfg.torque_limit_leg_neck
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.model.actuator_forcerange[a] = [-limit, limit]
            self.model.actuator_forcelimited[a] = 1

    def _qpos_noise_scale(self) -> np.ndarray:
        scale = np.zeros(NUM_ACT)
        for i, n in enumerate(ACTUATOR_ORDER):
            scale[i] = (self.cfg.noise_hip if "_hip" in n else self.cfg.noise_knee if "_knee" in n
                        else self.cfg.noise_ankle if "_ankle" in n else self.cfg.noise_upper)
        return scale

    # ---- readings
    def feet_contacts(self) -> np.ndarray:
        """geoms_colliding(sole geom, floor geom) for both feet: a MuJoCo contact between exactly these geoms with dist < 0."""
        out = np.zeros(2)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= 0.0:
                continue
            pair = {int(c.geom1), int(c.geom2)}
            for f, g in enumerate(self.feet_geoms):
                if pair == {g, self.floor_geom}:
                    out[f] = 1.0
        return out

    def joint_angles_with_backlash(self) -> np.ndarray:
        angles = self.data.qpos[self.act_qpos]
        backlash = self.data.qpos[self.backlash_qpos]
        for i in self.backlash_idx_to_add:
            backlash = np.insert(backlash, i, 0.0)
        return angles + backlash

    def _noise(self, shape: tuple[int, ...], scale: np.ndarray | float) -> np.ndarray:
        if self.noise_level == 0.0:
            return np.zeros(shape)
        return (2.0 * self.rng.random(shape) - 1.0) * self.noise_level * scale

    def get_obs(self, contact: np.ndarray) -> np.ndarray:
        d = self.data
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        accel = d.sensordata[self.accel_adr:self.accel_adr + 3].copy()
        accel[0] += ACCEL_X_OFFSET
        joint_angles = self.joint_angles_with_backlash()
        joint_vel = d.qvel[self.act_dof]      # env: actuator joints only, NO backlash velocity (joystick.py l.783)
        obs = np.concatenate([
            gyro + self._noise((3,), self.cfg.noise_gyro),
            accel + self._noise((3,), self.cfg.noise_accelerometer),
            self.command,
            joint_angles + self._noise((NUM_ACT,), self.noise_scale) - self.default_actuator,
            (joint_vel + self._noise((NUM_ACT,), self.cfg.noise_joint_vel)) * self.cfg.dof_vel_scale,
            self.last_act, self.last_last_act, self.last_last_last_act,
            self.motor_targets,
            contact,
            self.imitation_phase,
        ])
        if obs.shape != (OBS_SIZE,):
            raise RuntimeError(f"observation has shape {obs.shape}, expected ({OBS_SIZE},)")
        return obs

    def upright_z(self) -> float:
        return float(self.data.sensordata[self.up_adr + 2])

    # ---- episode
    def reset(self, walk_command: np.ndarray | None = None, upper_target: np.ndarray | None = None) -> np.ndarray:
        """Deterministic reset: home keyframe, no yaw/xy/joint randomisation (training randomises them).

        The env now starts every gesture at its first frame (home pose, targets 0) at reset, so pass the first row of the
        schedule's upper targets (0 for a gesture that starts at t = 0)."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.data.qpos[self.act_qpos]
        mujoco.mj_forward(self.model, self.data)
        walk = np.zeros(NUM_MOVE) if walk_command is None else np.asarray(walk_command, float)
        upper = np.zeros(NUM_UPPER) if upper_target is None else np.asarray(upper_target, float)
        self.command = np.concatenate([walk, upper])
        zero = np.zeros(NUM_ACT)
        self.last_act, self.last_last_act, self.last_last_last_act = zero.copy(), zero.copy(), zero.copy()
        self.motor_targets = self.default_actuator.copy()
        self.imitation_i = 0
        self.imitation_phase = np.zeros(2)                       # env.reset: zeros, not (1, 0)
        self.action_history = [zero.copy() for _ in range(max(1, self.action_delay_steps + 1))]
        self.obs = self.get_obs(self.feet_contacts())
        return self.obs

    def motor_targets_from_action(self, action: np.ndarray) -> np.ndarray:
        """Legs home + 0.25 a; upper clip(gesture target + residual a, joint range); then the 5.24 rad/s rate limit."""
        legs = self.default_actuator[:NUM_LEGS] + action[:NUM_LEGS] * self.cfg.action_scale
        upper = np.clip(self.command[NUM_MOVE:] + self.cfg.residual_scale * action[NUM_LEGS:],
                        self.lowers[NUM_LEGS:], self.uppers[NUM_LEGS:])
        targets = np.concatenate([legs, upper])
        step = self.cfg.max_motor_velocity * self.cfg.ctrl_dt
        return np.clip(targets, self.motor_targets - step, self.motor_targets + step)

    def step(self, action: np.ndarray, next_walk_command: np.ndarray, next_upper_target: np.ndarray) -> StepResult:
        """Apply `action` (14, the network output) for one control step and return the next observation."""
        action = np.asarray(action, dtype=float)
        self.imitation_i = (self.imitation_i + 1) % self.nb_steps_in_period
        angle = 2.0 * np.pi * self.imitation_i / self.nb_steps_in_period
        self.imitation_phase = np.array([np.cos(angle), np.sin(angle)])

        self.action_history = [action.copy()] + self.action_history[:self.action_delay_steps]
        action_used = self.action_history[self.action_delay_steps]
        targets = self.motor_targets_from_action(action_used)
        self.data.ctrl[:] = targets
        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)
            if self.substep_hook is not None:
                self.substep_hook(self.data)
        self.motor_targets = targets
        contact = self.feet_contacts()
        done = self.upright_z() < FALL_UPVECTOR_Z or bool(np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any())

        self.command = np.concatenate([np.asarray(next_walk_command, float), np.asarray(next_upper_target, float)])
        self.obs = self.get_obs(contact)
        self.last_last_last_act = self.last_last_act
        self.last_last_act = self.last_act
        self.last_act = action
        return StepResult(self.obs, done, targets, action_used)
