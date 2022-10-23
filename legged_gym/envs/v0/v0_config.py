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

    def _reward_velwork(self):
        diff_joint_pos = self.actions - self.last_actions
        # torque_transpose = torch.transpose(self.torques, 0, 1)
        e = torch.abs(torch.sum(torch.inner(self.torques, diff_joint_pos), dim=1))     # this is the L1 norm
        # energy = -(e- 300000)**2 + 10

        # energy = -(e)**2 + 1
        # energy = -(e/1e5)**2 + 1 # for 4x body mass
        # energy = -(e/2e5)**2 + 1 # for 8x body mass
        # energy = -(e/4e5)**2 + 1 # for 16x body mass
        energy = -(e/4e5)**2  # for 16x body mass
        # energy = -(e/1e5)**4 + 1
        # energy = -torch.exp(e/1e5) + 15
        # energy[e < 300000] = 10
        # velocity = self.base_lin_vel[:, 0]
        velocity = torch.linalg.vector_norm(self.base_lin_vel, dim=-1)
        # print(e, energy, velocity)
        # reward = (velocity + 1) * energy
        # reward = velocity * energy
        # # avoid situation with negative speed and negative energy
        # # iow prevent robot to learn to move fast backwards and wasting a lot of energy
        # reward[fwd_speed < 0.1] = -1e10
        return energy

    def _reward_torques2(self):
        velocity = torch.linalg.vector_norm(self.base_lin_vel, dim=-1)
        return torch.sum(torch.abs(self.torques)) * velocity

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
        num_observations = 54

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"  # none, plane, heightfield or trimesh
        measure_heights = True
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # select a unique terrain type and pass all arguments
        # curriculum = False
        # selected = True
        # # # Dict of arguments for selected terrain
        # terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_stairs_terrain",
        #     "step_width": 0.4,
        #     "step_height": -0.2,
        #     "platform_size": 3.,
        #     }
        # terrain_kwargs = {
        #     "type": "terrain_utils.discrete_obstacles_terrain",
        #     "min_size": 1,
        #     "max_size": 2,
        #     "num_rects": 20,
        #     "platform_size": 3.,
        #     "max_height": 0.35,
        # }
        terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.2,
            "max_height": 0.3,
            "step": 0.025,
            "downsampled_scale": 0.2,
        }

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
            push_robots = True
            push_interval_s = 10
            max_push_vel_xy = 100.


    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.6, 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_base_mass = True
        added_mass_range = [-20., 400.]
        randomize_com = True

    class control(LeggedRobotCfg.control):
        # PD Drive parameters: must be in sync with asset.default_dof_drive_mode
        # control_type = "P"
        # control_type = "R"
        # control_type = "V" # set velocity of the joint
        control_type = "D" # set position of the joint
        # 2x
        # stiffness = {"joint": 120.0}  # [N*m/rad]
        # damping = {"joint": 0.6}  # [N*m*s/rad]
        # 4x
        # urdf effort 155
        # stiffness = {"joint": 1470.0}  # [N*m/rad]
        # damping = {"joint": 2.3}  # [N*m*s/rad]
        # 8x
        # urdf effort 255
        stiffness = {
                "joint_1": 4070.0,
                "joint_2": 4070.0,
                "joint_3": 4070.0,
            }  # [N*m/rad]
        damping = {
                "joint_1": 0.2,
                "joint_2": 0.2,
                "joint_3": 0.2,}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 3
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/v0/urdf/v0.urdf"
        name = "v0"
        foot_name = "FOOT"
        penalize_contacts_on = [
            "CALF",
            "THIGH",
            "HIP",
            "body",
        ]
        terminate_after_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # must be in sync with control.control_type
        default_dof_drive_mode = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 300.0
        only_positive_rewards = False
        base_height_target = -0.2

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.0
            tracking_lin_vel = 2
            tracking_ang_vel = 0
            lin_vel_z = -0.00
            ang_vel_xy = -0.0
            # orientation = -0.0
            torques = -0
            torques2 = 0
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = -0.9
            feet_air_time = 0.0
            # collision = -0.00001
            # feet_stumble = -0.01
            action_rate = -0.01
            # stand_still = -0.000005
            heading_deviation = -0.5
            speed_norm = 0.
            half_legs_on_ground = -0.01
            rotation_left = 0
            rotation_right = 0
            velwork = 0 #1e-2
            dof_pos_limits = -0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 0.5
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 1.0
            body_mass = 0.004
            actions = 0.125
            friction = 0.25
        clip_observations = 100.
        clip_actions = 1.

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class sim(LeggedRobotCfg.sim):
        dt = 1 / 60  # 60 Hz
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z


class V0SixRoughCfg(V0RoughCfg):
    class env(V0RoughCfg.env):
        num_observations = 72
        num_actions = 3 * 6

    class asset(V0RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/v0six/urdf/v0.urdf"


class V0RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 101

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 2.e-5 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_v0"
        policy_class_name = 'ActorCriticEnvEncoder'
        algorithm_class_name = 'PPO'

    class policy(LeggedRobotCfgPPO.policy):
        # keep in sync with legged_robot.py: compute_observation()
        # used for environment encoder training in rsl_rl: ActorCriticEnvEncoder
        num_env_params = 11 # last #n params of obs_buf represent environment inputs
        xt_index = 7
        action_index = 43
        orientation_index = 64
