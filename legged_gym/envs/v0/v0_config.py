from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


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

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"  # none, plane, heightfield or trimesh

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # compute ang vel command from heading error
        num_commands = 4 if heading_command else 3

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.2, 1.2]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 30.0}  # [N*m/rad]
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
        ]
        terminate_after_contacts_on = ["HIP", "body"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 300.0
        only_positive_rewards = False
        base_height_target = -0.2

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -20.0
            # tracking_lin_vel = 2.0
            tracking_ang_vel = 0.1
            # lin_vel_z = -2.001
            # ang_vel_xy = -2.0
            orientation = -0.1
            # torques = -0.00002
            # dof_vel = -0.0
            # dof_acc = -0.0
            # base_height = -0.0
            # feet_air_time = 0.0
            # collision = -0.00001
            # feet_stumble = -0.0
            action_rate = -0.0
            # stand_still = -0.000005

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z


class V0SixRoughCfg(V0RoughCfg):
    class env(V0RoughCfg.env):
        num_observations = 253
        num_actions = 3 * 6

    class asset(V0RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/v0six/urdf/v0.urdf"


class V0RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_v0"
