import torch
from gpugym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotCfgPPO

class Azure_Loong_config(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 173
        num_actions = 12
        episode_length_s = 70
        num_history_short = 3

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type =  'plane' # 'trimesh'
        measure_heights = False
        static_friction = 0.1
        dynamic_friction = 0.1

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        robot_height_command = True
        ang_vel_command = True
        #-------------------------------------------
        lin_vel_clip = 0.3
        ang_vel_yaw_clip = 0.2
        static_delay = 50.
        resampling_range = [3., 15.] # for random resample time

        class ranges:
            # TRAINING COMMAND RANGES #
            lin_vel_x = [-1.5, 1.5]        # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]     # min max [rad/s]
            robot_height = [-1., 1.]     # min max [scale]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'
        penetration_check = False
        pos = [0., 0., 1.122]        # x,y,z [m]  
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [1.122, 1.122],      
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5]
        ]

        default_joint_angles = {
            'Joint-hip-r-roll': 0.,
            'Joint-hip-r-yaw': 0.,
            'Joint-hip-r-pitch': 0.305913,
            'Joint-knee-r-pitch': -0.670418,
            'Joint-ankel-r-pitch': 0.371265,
            'Joint-ankel-r-roll': 0.,

            'Joint-hip-l-roll': 0.,
            'Joint-hip-l-yaw': 0.,
            'Joint-hip-l-pitch': 0.305913,
            'Joint-knee-l-pitch': -0.670418,
            'Joint-ankel-l-pitch': 0.371265,
            'Joint-ankel-l-roll': 0.,
        }

        dof_pos_range = {
            'Joint-hip-r-roll': [-0.2, 0.2],
            'Joint-hip-r-yaw': [-0.1, 0.1],
            'Joint-hip-r-pitch': [0.12, 0.52],
            'Joint-knee-r-pitch': [-0.72, -0.62],
            'Joint-ankel-r-pitch': [0.07, 0.67],
            'Joint-ankel-r-roll': [-0.1, 0.1],

            'Joint-hip-l-roll': [-0.2, 0.2],
            'Joint-hip-l-yaw': [-0.1, 0.1],
            'Joint-hip-l-pitch': [0.12, 0.52],
            'Joint-knee-l-pitch': [-0.72, -0.62],
            'Joint-ankel-l-pitch': [0.07, 0.67],
            'Joint-ankel-l-roll': [-0.1, 0.1],
        }

        dof_vel_range = {
            'Joint-hip-r-roll':[-0.1, 0.1],
            'Joint-hip-r-yaw': [-0.1, 0.1],
            'Joint-hip-r-pitch': [-0.1, 0.1],
            'Joint-knee-r-pitch': [-0.1, 0.1],
            'Joint-ankel-r-pitch': [-0.1, 0.1],
            'Joint-ankel-r-roll': [-0.1, 0.1],

            'Joint-hip-l-roll': [-0.1, 0.1],
            'Joint-hip-l-yaw': [-0.1, 0.1],
            'Joint-hip-l-pitch': [-0.1, 0.1],
            'Joint-knee-l-pitch': [-0.1, 0.1],
            'Joint-ankel-l-pitch': [-0.1, 0.1],
            'Joint-ankel-l-roll': [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        # stiffness and damping for joints
        stiffness = {
            'Joint-hip-r-roll': 300.,
            'Joint-hip-r-yaw': 200.,
            'Joint-hip-r-pitch': 200.,
            'Joint-knee-r-pitch': 400.,
            'Joint-ankel-r-pitch': 120.,
            'Joint-ankel-r-roll': 120.,

            'Joint-hip-l-roll': 300.,
            'Joint-hip-l-yaw': 200.,
            'Joint-hip-l-pitch': 200.,
            'Joint-knee-l-pitch': 400.,
            'Joint-ankel-l-pitch': 120.,
            'Joint-ankel-l-roll': 120.,
        }
        damping = {
            'Joint-hip-r-roll': 1.,
            'Joint-hip-r-yaw': 0.5,
            'Joint-hip-r-pitch': 1.,
            'Joint-knee-r-pitch': 4.,
            'Joint-ankel-r-pitch': 1.,
            'Joint-ankel-r-roll': 1.,

            'Joint-hip-l-roll': 1.,
            'Joint-hip-l-yaw': 0.5,
            'Joint-hip-l-pitch': 1.,
            'Joint-knee-l-pitch': 4.,
            'Joint-ankel-l-pitch': 1.,
            'Joint-ankel-l-roll': 1.,
        }

        action_scale = 1.0
        exp_avg_decay = 0.05
        decimation = 20

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        # friction_range = [0.5, 1.25]
        friction_range = [0.3, 3.]

        #old mass randomize
        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        randomize_all_mass = True
        rd_mass_range = [0.5, 1.5]

        randomize_com = True
        rd_com_range = [-0.05, 0.05]

        randomize_base_com = True
        rd_base_com_range = [-0.1, 0.1]
        
        
        push_robots = True
        push_interval_s = 2
        push_ratio= 0.4
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.4

        random_pd = True
        p_range = [0.7, 1.3]
        d_range = [0.7, 1.3]

        random_damping = True
        damping_range = [0.3, 4.0]

        random_inertia = True
        inertia_range = [0.7, 1.3]

        comm_delay = False
        comm_delay_range = [0, 11] # will exclude the upper limit

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}'\
            '/resources/robots/OGHR/urdf/OGHR_wholeBody_Simplified(12dof).urdf'
        keypoints = ["base_link"]
        end_effectors = ['Link-ankel-r-roll', 'Link-ankel-l-roll']
        foot_name = ['Link-ankel-l-roll', 'Link-ankel-r-roll']
        terminate_after_contacts_on = [
            "base_link",
        ]

        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

    class rewards(LeggedRobotCfg.rewards):
        # ! "Incorrect" specification of height
        # base_height_target = 0.7
        base_height_target = 1.05
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 1500.

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.5

        class scales(LeggedRobotCfg.rewards.scales):
            # * "True" rewards * #
            # reward for task
            tracking_lin_vel = 10.
            tracking_ang_vel = 5.

            no_fly = 1.0
            no_jump = 1.0
            stand_still = 1.0
            feet_air_time = 1.0

            foot_slip = 0.5

            # reward for smooth
            action_rate = -1.e-6
            action_rate2 = -1.e-6
            torques = -1e-6
            base_lin_acc = -1e-3
            base_ang_acc = -2e-5
            ##############################
            dof_acc = -4e-4
            dof_vel = -1e-4

            #reward for safety
            dof_pos_limits = -10
            torque_limits = -1e-2
            feet_contact_forces = -5e-3
            termination = -100
            
            # ang_vel_xy = -5.
            # lin_vel_z = -5.
            
            #reward for beauty
            # * Shaping rewards * #
            # Sweep values: [0.5, 2.5, 10, 25., 50.]
            # Default: 5.0
            # orientation = 5.0

            # Sweep values: [0.2, 1.0, 4.0, 10., 20.]
            # Default: 2.0
            # base_height = 2.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # joint_regularization = 1.0

            # * PBRS rewards * #
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            ori_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            baseHeight_pb = 0.3

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            jointReg_pb = 1.0

            ankleReg_pb = 0.1


    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            base_z = 1./0.6565

        clip_observations = 100.
        clip_actions = 10.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_z = 0.05
            dof_pos = 0.002 # 0.005
            dof_vel = 0.001 # 0.01
            lin_vel = 0.04 # 0.1
            ang_vel = 0.05
            gravity = 0.002 # 0.05
            in_contact = 0.1
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]

        class physx:
            max_depenetration_velocity = 10.0


class Azure_Loong_configPPO(LeggedRobotCfgPPO):
    do_wandb = True
    seed = -1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        weight_decay = 0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 30000
        run_name = 'AzureLoong_2024'
        experiment_name = 'AzureLoong_Locomotion'
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'elu'
        conv_dims = [(42, 32, 6, 5), (32, 16, 4, 2)]
        period_length = 100
        
        
        
        
        
        
"""
Configuration file for cassie
"""




