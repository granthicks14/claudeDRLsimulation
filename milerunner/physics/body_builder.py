"""Generate a MuJoCo MJCF humanoid from :class:`BodyParams`.

The model is a 3D bipedal humanoid whose segment lengths and masses are derived
from anthropometric fractions, so a 60 kg or 90 kg "body type" produces a
physically consistent skeleton. Joints are named after the muscle groups that
drive them (ankle=calf, knee=quad/hamstring, hip=glute, abdomen=core,
shoulder, elbow=arm, neck) so the biomechanical fatigue model can clip each
actuator to a human-strength limit.

Human joint ranges (in degrees, converted to radians) follow clinical
range-of-motion norms, guaranteeing the agent can never adopt impossible poses.
"""
from __future__ import annotations

import math
from typing import Dict, List

from ..biomech.params import BodyParams


def _deg(d: float) -> float:
    return d * math.pi / 180.0


# Actuated joints and their human ranges (radians) + which muscle-group joint
# key governs their torque budget.
JOINT_SPEC = [
    # name,            axis,   range(deg lo, hi),   muscle joint key
    ("abdomen_z",     "0 0 1", (-45, 45),           "trunk"),
    ("abdomen_y",     "0 1 0", (-30, 45),           "trunk"),
    ("abdomen_x",     "1 0 0", (-35, 35),           "trunk"),
    ("hip_x_right",   "1 0 0", (-25, 45),           "hip"),
    ("hip_z_right",   "0 0 1", (-35, 35),           "hip"),
    ("hip_y_right",   "0 1 0", (-120, 30),          "hip"),
    ("knee_right",    "0 1 0", (-160, 2),           "knee"),
    ("ankle_y_right", "0 1 0", (-50, 50),           "ankle"),
    ("ankle_x_right", "1 0 0", (-25, 25),           "ankle"),
    ("hip_x_left",    "1 0 0", (-25, 45),           "hip"),
    ("hip_z_left",    "0 0 1", (-35, 35),           "hip"),
    ("hip_y_left",    "0 1 0", (-120, 30),          "hip"),
    ("knee_left",     "0 1 0", (-160, 2),           "knee"),
    ("ankle_y_left",  "0 1 0", (-50, 50),           "ankle"),
    ("ankle_x_left",  "1 0 0", (-25, 25),           "ankle"),
    ("shoulder1_right","2 1 1",(-85, 60),           "shoulder"),
    ("shoulder2_right","0 -1 1",(-85, 60),          "shoulder"),
    ("elbow_right",   "0 -1 1",(-90, 50),           "elbow"),
    ("shoulder1_left","2 -1 1",(-85, 60),           "shoulder"),
    ("shoulder2_left","0 1 1", (-85, 60),           "shoulder"),
    ("elbow_left",    "0 -1 1",(-90, 50),           "elbow"),
    ("neck_y",        "0 1 0", (-40, 40),           "neck"),
]


def actuated_joint_names() -> List[str]:
    return [j[0] for j in JOINT_SPEC]


def joint_muscle_map() -> Dict[str, str]:
    return {j[0]: j[3] for j in JOINT_SPEC}


def build_humanoid_mjcf(params: BodyParams, friction: float = 1.0,
                        timestep: float = 0.001) -> str:
    """Return an MJCF XML string for the given body.

    ``timestep`` defaults to 1 ms, giving the required 1000+ physics steps
    per simulated second.
    """
    p = params
    # Segment lengths (m).
    L_foot = p.segment_length("foot")
    L_shank = p.segment_length("shank")
    L_thigh = p.segment_length("thigh")
    L_trunk = p.segment_length("trunk")
    L_head = p.segment_length("head_neck")
    L_uarm = p.segment_length("upper_arm")
    L_farm = p.segment_length("forearm")

    # Segment masses (kg).
    m_trunk = p.segment_mass("trunk")
    m_head = p.segment_mass("head_neck")
    m_uarm = p.segment_mass("upper_arm")
    m_farm = p.segment_mass("forearm") + p.segment_mass("hand")
    m_thigh = p.segment_mass("thigh")
    m_shank = p.segment_mass("shank")
    m_foot = p.segment_mass("foot")

    hip_z = L_foot * 0.0 + L_shank + L_thigh  # hip height when standing
    root_z = hip_z + 0.05
    hw = 0.09  # half hip width
    sw = L_trunk * 0.5  # shoulder half width scaled to trunk

    # radii for capsules (rough, mass set explicitly so these are visual/inertial shape)
    r_leg = 0.055
    r_arm = 0.038
    r_trunk = 0.10

    def joint_xml(name: str) -> str:
        spec = next(j for j in JOINT_SPEC if j[0] == name)
        _, axis, (lo, hi), _ = spec
        return (f'<joint name="{name}" type="hinge" axis="{axis}" '
                f'range="{_deg(lo):.4f} {_deg(hi):.4f}" armature="0.02" '
                f'damping="5" stiffness="4" limited="true"/>')

    xml = f"""<mujoco model="milerunner_humanoid">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{timestep}" iterations="30" solver="Newton" gravity="0 0 -9.81"
          integrator="implicitfast" cone="elliptic"/>
  <default>
    <joint limited="true" armature="0.02" damping="5"/>
    <geom conaffinity="1" condim="3" contype="1" friction="{friction} 0.1 0.1"
          material="body" rgba="0.8 0.6 0.4 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.1 0.2 0.4" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"
             width="512" height="512" mark="cross" markrgb="0.8 0.8 0.8"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="body" rgba="0.8 0.6 0.4 1"/>
  </asset>
  <worldbody>
    <light diffuse="0.9 0.9 0.9" pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="200 5 0.1" pos="0 0 0" material="grid"
          friction="{friction} 0.1 0.1" condim="3"/>
    <body name="pelvis" pos="0 0 {root_z:.4f}">
      <freejoint name="root"/>
      <geom name="pelvis_g" type="capsule" fromto="0 -{hw} 0 0 {hw} 0" size="{r_trunk}" mass="{m_trunk*0.35:.3f}"/>
      <body name="torso" pos="0 0 {L_trunk*0.55:.4f}">
        {joint_xml('abdomen_z')}
        {joint_xml('abdomen_y')}
        {joint_xml('abdomen_x')}
        <geom name="torso_g" type="capsule" fromto="0 -{sw*0.7:.3f} 0 0 {sw*0.7:.3f} 0" size="{r_trunk}" mass="{m_trunk*0.65:.3f}"/>
        <body name="head" pos="0 0 {L_trunk*0.45+L_head*0.5:.4f}">
          {joint_xml('neck_y')}
          <geom name="head_g" type="sphere" size="{L_head*0.5:.3f}" pos="0 0 0" mass="{m_head:.3f}"/>
        </body>
        <body name="upper_arm_right" pos="0 -{sw*0.75:.3f} {L_trunk*0.35:.3f}">
          {joint_xml('shoulder1_right')}
          {joint_xml('shoulder2_right')}
          <geom name="uarm_r_g" type="capsule" fromto="0 0 0 0 -0.02 -{L_uarm:.3f}" size="{r_arm}" mass="{m_uarm:.3f}"/>
          <body name="lower_arm_right" pos="0 -0.02 -{L_uarm:.3f}">
            {joint_xml('elbow_right')}
            <geom name="larm_r_g" type="capsule" fromto="0 0 0 0 0 -{L_farm:.3f}" size="{r_arm*0.85:.3f}" mass="{m_farm:.3f}"/>
          </body>
        </body>
        <body name="upper_arm_left" pos="0 {sw*0.75:.3f} {L_trunk*0.35:.3f}">
          {joint_xml('shoulder1_left')}
          {joint_xml('shoulder2_left')}
          <geom name="uarm_l_g" type="capsule" fromto="0 0 0 0 0.02 -{L_uarm:.3f}" size="{r_arm}" mass="{m_uarm:.3f}"/>
          <body name="lower_arm_left" pos="0 0.02 -{L_uarm:.3f}">
            {joint_xml('elbow_left')}
            <geom name="larm_l_g" type="capsule" fromto="0 0 0 0 0 -{L_farm:.3f}" size="{r_arm*0.85:.3f}" mass="{m_farm:.3f}"/>
          </body>
        </body>
      </body>
      <body name="thigh_right" pos="0 -{hw} 0">
        {joint_xml('hip_x_right')}
        {joint_xml('hip_z_right')}
        {joint_xml('hip_y_right')}
        <geom name="thigh_r_g" type="capsule" fromto="0 0 0 0 0 -{L_thigh:.3f}" size="{r_leg}" mass="{m_thigh:.3f}"/>
        <body name="shank_right" pos="0 0 -{L_thigh:.3f}">
          {joint_xml('knee_right')}
          <geom name="shank_r_g" type="capsule" fromto="0 0 0 0 0 -{L_shank:.3f}" size="{r_leg*0.85:.3f}" mass="{m_shank:.3f}"/>
          <body name="foot_right" pos="0 0 -{L_shank:.3f}">
            {joint_xml('ankle_y_right')}
            {joint_xml('ankle_x_right')}
            <geom name="foot_r_g" type="capsule" fromto="-{L_foot*0.3:.3f} 0 -0.02 {L_foot*0.7:.3f} 0 -0.02" size="0.028" mass="{m_foot:.3f}"/>
          </body>
        </body>
      </body>
      <body name="thigh_left" pos="0 {hw} 0">
        {joint_xml('hip_x_left')}
        {joint_xml('hip_z_left')}
        {joint_xml('hip_y_left')}
        <geom name="thigh_l_g" type="capsule" fromto="0 0 0 0 0 -{L_thigh:.3f}" size="{r_leg}" mass="{m_thigh:.3f}"/>
        <body name="shank_left" pos="0 0 -{L_thigh:.3f}">
          {joint_xml('knee_left')}
          <geom name="shank_l_g" type="capsule" fromto="0 0 0 0 0 -{L_shank:.3f}" size="{r_leg*0.85:.3f}" mass="{m_shank:.3f}"/>
          <body name="foot_left" pos="0 0 -{L_shank:.3f}">
            {joint_xml('ankle_y_left')}
            {joint_xml('ankle_x_left')}
            <geom name="foot_l_g" type="capsule" fromto="-{L_foot*0.3:.3f} 0 -0.02 {L_foot*0.7:.3f} 0 -0.02" size="0.028" mass="{m_foot:.3f}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
{_actuator_block(p)}
  </actuator>
  <sensor>
    <accelerometer name="pelvis_acc" site="pelvis_site"/>
    <gyro name="pelvis_gyro" site="pelvis_site"/>
  </sensor>
</mujoco>"""
    # add a site for sensors inside pelvis
    xml = xml.replace('<geom name="pelvis_g"',
                      '<site name="pelvis_site" pos="0 0 0" size="0.02"/>\n        <geom name="pelvis_g"')
    return xml


def _actuator_block(p: BodyParams) -> str:
    """One position-less torque motor per actuated joint.

    ``gear`` sets the peak torque (N*m) to the muscle-group budget; the env
    further scales the control range down as the group fatigues.
    """
    jm = joint_muscle_map()
    lines = []
    for name in actuated_joint_names():
        joint_key = jm[name]
        gear = p.muscle_peak_torque[joint_key]
        # knee shared by two groups -> allow the full budget through the motor
        lines.append(f'    <motor name="act_{name}" joint="{name}" gear="{gear:.1f}"/>')
    return "\n".join(lines)
