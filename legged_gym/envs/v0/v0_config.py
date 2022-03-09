import torch

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from legged_gym.envs import LeggedRobot


class V0Robot(LeggedRobot):
    def _reward_heading_deviation(self):
        deviation = self.compute_heading_deviation().reshape(-1)
        return torch.abs(deviation)

    def _reward_speed_norm(self):
        base_xy_velocity = self.base_lin_vel[:, :3]

        return base_xy_velocity[:, 0]

    def _reward_half_legs_on_ground(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        leg_contacts = torch.abs(
            torch.sum(1.0 * contacts, dim=1) - (len(self.feet_indices) / 2)
        )

        return 1.0 * leg_contacts

    def _reward_rotation_left(self):
        return self.base_ang_vel[:, 2]

    def _reward_rotation_right(self):
        return -self.base_ang_vel[:, 2]


class V0RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.22]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "l1_joint_1_to_2": -0.4,
            "l1_joint_2_to_3": 0.15,
            "l1_joint_3_to_4_FOOT": 0,
            "l2_joint_1_to_2": 0,
            "l2_joint_2_to_3": 0.15,
            "l2_joint_3_to_4_FOOT": 0,
            "l3_joint_1_to_2": 0.5,
            "l3_joint_2_to_3": 0.15,
            "l3_joint_3_to_4_FOOT": 0,
            "r1_joint_1_to_2": 0.4,
            "r1_joint_2_to_3": -0.15,
            "r1_joint_3_to_4_FOOT": 0,
            "r2_joint_1_to_2": 0,
            "r2_joint_2_to_3": -0.15,
            "r2_joint_3_to_4_FOOT": 0,
            "r3_joint_1_to_2": -0.5,
            "r3_joint_2_to_3": -0.15,
            "r3_joint_3_to_4_FOOT": 0,
        }

    class env(LeggedRobotCfg.env):
        # num_observations = 236
        num_observations = 49

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"  # none, plane, heightfield or trimesh
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # compute ang vel command from heading error
        num_commands = 4 if heading_command else 3

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.5, 2.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            heading = [0, 0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.05, 4.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 70.0}  # [N*m/rad]
        damping = {"joint": 0.2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/v0/urdf/v0.urdf"
        name = "v0"
        foot_name = "FOOT"
        penalize_contacts_on = [
            "CALF",
            "THIGH",
            "HIP",
        ]
        terminate_after_contacts_on = ["body"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 300.0
        only_positive_rewards = False
        base_height_target = -0.2

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.1
            lin_vel_z = -0.001
            ang_vel_xy = -0.0
            # orientation = -0.0
            # torques = -0.00002
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = 0.0
            feet_air_time = 0.0
            # collision = -0.00001
            # feet_stumble = -0.0
            action_rate = -0.0
            # stand_still = -0.000005
            heading_deviation = -1
            speed_norm = 0.1
            half_legs_on_ground = -0.25
            rotation_left = 0
            rotation_right = 0

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z


class V0SixRoughCfg(V0RoughCfg):
    class env(V0RoughCfg.env):
        num_observations = 254
        num_actions = 3 * 6

    class asset(V0RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/v0six/urdf/v0.urdf"


class V0RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_v0"
