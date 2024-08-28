import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs import LeggedRobot, LeggedRobotCfg
from isaacgym import gymtorch, gymapi, gymutil
import os
from gpugym import LEGGED_GYM_ROOT_DIR, envs
from collections import deque
import random


class Azure_Loong_env(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if hasattr(self, "_custom_init"):
            self._custom_init(cfg)

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.p5 = 0.5*torch.ones(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.
        if self.cfg.env.num_privileged_obs:
            self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_history_short = self.cfg.env.num_history_short
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.static_delay = self.cfg.commands.static_delay
        self.resampling_time = self.cfg.commands.resampling_time * torch.ones((self.num_envs, ), device=self.device)
        self.min_sample_time = self.cfg.commands.resampling_range[0]
        self.max_sample_time = self.cfg.commands.resampling_range[1]
        # for random pd
        self.low_p_ratio = self.cfg.domain_rand.p_range[0]
        self.high_p_ratio = self.cfg.domain_rand.p_range[1]

        
        self.low_d_ratio = self.cfg.domain_rand.d_range[0]
        self.high_d_ratio = self.cfg.domain_rand.d_range[1]
        
        if self.cfg.domain_rand.random_pd:
            self.random_p = torch_rand_float(self.low_p_ratio, self.high_p_ratio, 
                                    shape=(self.num_envs, self.num_dof), device=self.device)
            self.random_d = torch_rand_float(self.low_d_ratio, self.high_d_ratio, 
                                    shape=(self.num_envs, self.num_dof), device=self.device)
        else:
            self.random_p = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
            self.random_d = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
            self.low_p_ratio = 1.
            self.high_p_ratio = 1.
            self.low_d_ratio = 1.
            self.high_d_ratio = 1.

        self.average_p_ratio = (self.low_p_ratio + self.high_p_ratio) / 2.
        self.p_ratio_diff = self.high_p_ratio - self.average_p_ratio + 1e-8
        self.average_d_ratio = (self.low_d_ratio + self.high_d_ratio) / 2.
        self.d_ratio_diff = self.high_d_ratio - self.average_d_ratio + 1e-8


        #----------------------
        self.lin_vel1 = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.lin_vel2 = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.ang_vel1 = torch.zeros_like(self.base_ang_vel, device=self.device)
        self.ang_vel2 = torch.zeros_like(self.base_ang_vel, device=self.device)
        self.contact_force1 = torch.zeros_like(self.contact_forces[:, self.feet_indices, 2])
        self.contact_force2 = torch.zeros_like(self.contact_forces[:, self.feet_indices, 2])
        self.max_vel =torch.full([6], float("-inf"),device=self.device)

        # short history deques
        self.ctrl_hist_deque_short = deque(maxlen=self.num_history_short)
        self.dof_pos_hist_deque_short = deque(maxlen=self.num_history_short)
        self.dof_vel_hist_deque_short = deque(maxlen=self.num_history_short)
        self.base_ang_vel_hist_deque_short = deque(maxlen=self.num_history_short)
        self.proj_gravity_hist_deque_short = deque(maxlen=self.num_history_short)
        for _ in range(self.num_history_short):
            self.ctrl_hist_deque_short.append(torch.zeros(self.num_envs, self.num_actions,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.dof_pos_hist_deque_short.append(torch.zeros(self.num_envs, self.num_actions,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.dof_vel_hist_deque_short.append(torch.zeros(self.num_envs, self.num_actions,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.base_ang_vel_hist_deque_short.append(torch.zeros(self.num_envs, 3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.proj_gravity_hist_deque_short.append(torch.zeros(self.num_envs, 3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.comm_delay = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        max_delay = self.cfg.domain_rand.comm_delay_range[1]
        # for i in range(self.num_envs):
        #     if self.cfg.domain_rand.comm_delay:
        #         rng = self.cfg.domain_rand.comm_delay_range
        #         rd_num = np.random.randint(rng[0], rng[1])
        #         self.delay_deque_lst.append(deque(maxlen=rd_num+1))
        #         self.comm_delay[i] = rd_num
        #     else:
        #         self.delay_deque_lst.append(deque(maxlen=1))
        self.actions_record = torch.zeros((self.num_envs, max_delay, self.num_actions), device=self.device, requires_grad=False)
        if self.cfg.domain_rand.comm_delay:
            rng = self.cfg.domain_rand.comm_delay_range
            self.comm_delay = torch.randint(rng[0], rng[1], (self.num_envs, 1), device=self.device)





    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if self.cfg.asset.disable_actions:
            self.actions[:] = 0.
        else:
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.pre_physics_step()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.comm_delay:
                self.actions_record = torch.cat((self.actions_record[:,1:, :], self.actions.unsqueeze(1)), dim=1)
                delayed_action = self.actions_record[torch.arange(self.num_envs), (-1 - self.comm_delay).squeeze(1)]
               

                if self.cfg.control.exp_avg_decay:
                    self.action_avg = exp_avg_filter(delayed_action, self.action_avg,
                                                    self.cfg.control.exp_avg_decay)
                    self.torques = self._compute_torques(self.action_avg).view(self.torques.shape)
                else:
                    self.torques = self._compute_torques(delayed_action).view(self.torques.shape)
            else:
                if self.cfg.control.exp_avg_decay:
                    self.action_avg = exp_avg_filter(self.actions, self.action_avg,
                                                    self.cfg.control.exp_avg_decay)
                    self.torques = self._compute_torques(self.action_avg).view(self.torques.shape)
                else:
                    self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            if self.cfg.asset.disable_motors:
                self.torques[:] = 0.

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, \
            self.reset_buf, self.extras


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.num_rigid_shape = len(rigid_shape_props_asset)

        self.friction_buf = torch.ones((self.num_envs, 1), device=self.device, requires_grad=False, dtype=torch.float32)
        self.damping_buf = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
        self.mass_mask = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        
        self.inertia_mask_xx = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_xy = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_xz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_yy = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_yz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_zz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        self.com_diff_x = torch.zeros((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.com_diff_y = torch.zeros((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.com_diff_z = torch.zeros((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        # print("+++++++++++++++++++++++++++++++++++")
        # print("rigid_shape:", self.num_rigid_shape)
        # print("num body:", self.num_bodies)
        # print("num dof:", self.num_dof)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        # self.num_dofs = len(self.dof_names)  # ! replaced with num_dof
        feet_names = self.cfg.asset.foot_name
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            legged_robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "legged_robot", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, legged_robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, legged_robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, legged_robot_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(legged_robot_handle)

        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])


        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        #foot sensors
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False # for example gravity
            sensor_options.enable_constraint_solver_forces = True # for example contacts
            sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    
    def compute_observations(self):
        ##################            check           ########################
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        ######################################################################
        in_contact = torch.gt(
            self.contact_forces[:, self.end_eff_ids, 2], 0).int()
        in_contact = torch.cat(
            (in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)),
            dim=1)
        self.commands[:, 0:2] = torch.where(
            torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.3,
            0., self.commands[:, 0:2].double()).float()
        self.commands[:, 2:3] = torch.where(
            torch.abs(self.commands[:, 2:3]) < 0.3,
            0., self.commands[:, 2:3].double()).float()
        square_wave = torch.where(self.time_to_stand_still.unsqueeze(1) > self.static_delay, self.p5, 0.* self.smooth_sqr_wave(self.phase))
        self.obs_buf = torch.cat((
            # base_z,                                 # [1] Base height *
            # self.base_lin_vel,                      # [3] Base linear velocity *
            self.commands[:, 0:4],                  # [4] Velocity commands
            square_wave,                            # [1] Contact schedule [;5]
            ####################################################################
            self.base_ang_vel,                      # [3] Base angular velocity [5:47]
            self.projected_gravity,                 # [3] Projected gravity
            # torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
            # torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
            self.actions*self.cfg.control.action_scale * 0., # [12] Joint actions
            self.dof_pos,                           # [12] Joint states
            self.dof_vel,                           # [12] Joint velocities
            # in_contact,                             # [2] Contact states
            ####################################################################
            self.base_ang_vel_hist,                 # [9] Base angular velocity history
            self.proj_gravity_hist,                 # [9] Projected gravity
            self.ctrl_hist,                         # [36] action history
            self.dof_pos_hist,                      # [36] dof position history
            self.dof_vel_hist,                      # [36] dof velocity history history
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                * self.noise_scale_vec


    def _get_noise_scale_vec(self, cfg):
        #######################  change later  ############################
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[0] = noise_scales.base_z * self.obs_scales.base_z
        noise_vec[5:8] = noise_scales.ang_vel
        noise_vec[8:11] = noise_scales.gravity
        noise_vec[11:23] = 0. # actions
        noise_vec[23:35] = noise_scales.dof_pos
        noise_vec[35:47] = noise_scales.dof_vel
        # noise_vec[47:83] = 0 # ctrl hist
        # noise_vec[83:119] = noise_scales.dof_pos
        # noise_vec[119:155] = noise_scales.dof_vel
        # noise_vec[155:164] = noise_scales.ang_vel
        # noise_vec[164:173] = noise_scales.gravity
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements \
        #         * noise_level \
        #         * self.obs_scales.height_measurements
        
        noise_vec = noise_vec * noise_level
        return noise_vec

    def _custom_reset(self, env_ids):
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device)
        self.max_feet_air_time[env_ids, :] =torch.zeros((len(env_ids), 2), device=self.device)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        self.lin_vel2[:] = self.base_lin_vel
        self.ang_vel2[:] = self.base_ang_vel
        self.contact_force2[:] = self.contact_forces[:, self.feet_indices, 2]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

       
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.ctrl_hist_deque_short.append(self.actions.clone()*self.cfg.control.action_scale)
        self.ctrl_hist = torch.cat([t for t in self.ctrl_hist_deque_short],dim=1)

        self.dof_pos_hist_deque_short.append(self.obs_buf[:, 23:35])
        self.dof_pos_hist = torch.cat([t for t in self.dof_pos_hist_deque_short],dim=1)

        self.dof_vel_hist_deque_short.append(self.obs_buf[:, 35:47])
        self.dof_vel_hist = torch.cat([t for t in self.dof_vel_hist_deque_short],dim=1)

        self.base_ang_vel_hist_deque_short.append(self.obs_buf[:, 5:8])
        self.base_ang_vel_hist = torch.cat([t for t in self.base_ang_vel_hist_deque_short],dim=1)
        
        self.proj_gravity_hist_deque_short.append(self.obs_buf[:, 8:11])
        self.proj_gravity_hist = torch.cat([t for t in self.proj_gravity_hist_deque_short],dim=1)

        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        static = (self.commands[:, 0] == 0) & (self.commands[:, 1] == 0) & (self.commands[:, 2] == 0)
        low_speed = (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.3)
        self.time_to_stand_still += 1. * static
        self.time_to_stand_still *= low_speed
        self.phase = torch.fmod(self.phase + self.dt, 1.0)
        env_ids = (
            self.episode_length_buf
            % (self.resampling_time / self.dt).to(torch.int32) == 0) \
            .nonzero(as_tuple=False).flatten()
        # if len(env_ids):
        #     print("set")
        if self.cfg.commands.resampling_time == -1 :
            # print(self.commands)
            pass  # when the joystick is used, the self.commands variables are overridden
        else:
            self._resample_commands(env_ids)

            if ( self.cfg.domain_rand.push_robots and
                (self.common_step_counter
                % self.cfg.domain_rand.push_interval == 0)):
                # self._push_robots()
                random_number = round(random.uniform(0, 1), 2)
                if random_number > 1-self.cfg.domain_rand.push_ratio:
                    # print('push time=',self.common_step_counter)
                    self._push_robots()

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
  
      
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_system(env_ids)
        if hasattr(self, "_custom_reset"):
            self._custom_reset(env_ids)
        self._resample_commands(env_ids)
        # reset buffers
        self.ctrl_hist[env_ids] = 0.
        self.dof_pos_hist[env_ids] = 0.
        self.dof_vel_hist[env_ids] = 0.
        self.base_ang_vel_hist[env_ids] = 0.
        self.proj_gravity_hist[env_ids] = 0.

        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        #reset deques
        for i in range(self.num_history_short):
            self.ctrl_hist_deque_short[i][env_ids] *= 0.
            self.dof_pos_hist_deque_short[i][env_ids] *= 0.
            self.dof_vel_hist_deque_short[i][env_ids] *= 0.
            self.base_ang_vel_hist_deque_short[i][env_ids] *= 0.
            self.proj_gravity_hist_deque_short[i][env_ids] *= 0.
       

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    
    def _push_robots(self):
        # Randomly pushes the robots.
        # Emulates an impulse by setting a randomized base velocity.

        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        # self.root_states[:, 7:8] = torch_rand_float(
        #     -max_vel, max_vel, (self.num_envs, 1), device=self.device)
        # self.gym.set_actor_root_state_tensor(
        #     self.sim, gymtorch.unwrap_tensor(self.root_states))
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        print("pushed")

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Termination for contact
        term_contact = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1)
        self.reset_buf = torch.any((term_contact > 1.), dim=1)


        # Termination for velocities, orientation, and low height
        self.reset_buf |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)

        self.reset_buf |= torch.any(
          torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)

        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)

        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)


        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)


        # # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.resampling_time[env_ids] = torch_rand_float(self.min_sample_time, self.max_sample_time, (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.robot_height_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["robot_height"][0], self.command_ranges["robot_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        elif self.cfg.commands.ang_vel_command:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.time_to_stand_still[env_ids] = 0.

        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.lin_vel_clip).unsqueeze(1)
        # self.commands[env_ids, 2] *= (self.commands[env_ids, 2] > self.cfg.commands.ang_vel_yaw_clip)|(self.commands[env_ids, 2] < -self.cfg.commands.ang_vel_yaw_clip)
        # if len(env_ids):
        #     print(self.commands[:, 2])
        command_mode = torch.rand((len(env_ids), 1), device=self.device).squeeze(1)

        # 0-0.1 stand still (all = 0)
        # 0.1-0.2 turn (ang vel != 0)
        # 0.2-0.4 walk along y axis (y vel != 0)
        # 0.4-0.6 walk along x axis (x vel != 0)
        # 0.6-1 hybrid (all != 0)

        self.commands[env_ids, 0] *= command_mode >= 0.4 # x vel
        self.commands[env_ids, 1] *= ((command_mode >= 0.2) & (command_mode < 0.4)) | (command_mode >= 0.6) # y vel
        self.commands[env_ids, 2] *= ((command_mode >0.1) & (command_mode < 0.2)) | (command_mode >= 0.6) # ang vel

        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        double_contact = torch.sum(1.*contacts, dim=-1) == 2
        self.time_to_stand_still[env_ids] += self.static_delay * double_contact[env_ids] * \
            (torch.norm(self.commands[env_ids, :3], dim=-1) == 0) * \
            (torch.norm(self.base_lin_vel[env_ids, :2], dim=-1) < 0.3)
        
        
            
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller

        # if self.cfg.control.exp_avg_decay:
        #     self.action_avg = exp_avg_filter(self.actions, self.action_avg,
        #                                     self.cfg.control.exp_avg_decay)
        #     actions = self.action_avg

        if self.cfg.control.control_type=="P":
            torques = self.random_p * self.p_gains*(actions * self.cfg.control.action_scale \
                                    + self.default_dof_pos \
                                    - self.dof_pos) \
                    - self.random_d * self.d_gains*self.dof_vel

        elif self.cfg.control.control_type=="T":
            torques = actions * self.cfg.control.action_scale

        elif self.cfg.control.control_type=="Td":
            torques = actions * self.cfg.control.action_scale \
                        - self.random_d * self.d_gains*self.dof_vel

        else:
            raise NameError(f"Unknown controller type: {self.cfg.control.control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self.friction_buf[env_id, :] = self.friction_coeffs[env_id]
        # default value of frictions are 1.0
        return props
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r \
                                           *self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

                
        rng = self.cfg.domain_rand.damping_range
        for i in range(len(props)):
            if self.cfg.domain_rand.random_damping:
                rd_num = np.random.uniform(rng[0], rng[1])
                self.damping_buf[env_id, i] = rd_num
                props["damping"][i] = rd_num
            else:
                self.damping_buf[env_id, i] = props["damping"][i].item()

        
        return props
    
    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            m = 0
            for i, p in enumerate(props):
                m += p.mass
            #     print(f"Mass of body {i}: {p.mass} (before randomization)")
            # print(f"Total mass {m} (before randomization)")
            self.mass_total = m

        # randomize mass of all link
        if self.cfg.domain_rand.randomize_all_mass:
            for s in range(len(props)):
                rng = self.cfg.domain_rand.rd_mass_range
                rd_num = np.random.uniform(rng[0], rng[1])
                self.mass_mask[env_id, s] = rd_num
                props[s].mass *= rd_num
        
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[-1].mass += np.random.uniform(rng[0], rng[1])
        
        # randomize com of all link other than base link
        if self.cfg.domain_rand.randomize_com:
            for s in range(len(props)-1):
                
                rng = self.cfg.domain_rand.rd_com_range
                rd_num = np.random.uniform(rng[0], rng[1])
                self.com_diff_x[env_id, s] = rd_num
                props[s].com.x += rd_num
                rd_num = np.random.uniform(rng[0], rng[1])
                self.com_diff_y[env_id, s] = rd_num
                props[s].com.y += rd_num
                rd_num = np.random.uniform(rng[0], rng[1])
                self.com_diff_z[env_id, s] = rd_num
                props[s].com.z += rd_num

        # randomize com of base link
        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.rd_base_com_range
            rd_num = np.random.uniform(rng[0], rng[1])
            self.com_diff_x[env_id, -1] = rd_num
            props[-1].com.x += rd_num
            rd_num = np.random.uniform(rng[0], rng[1])
            self.com_diff_y[env_id, -1] = rd_num
            props[-1].com.y += rd_num
            rd_num = np.random.uniform(rng[0], rng[1])
            self.com_diff_z[env_id, -1] = rd_num
            props[-1].com.z += rd_num

        # randomize inertia of all body
        if self.cfg.domain_rand.random_inertia:
            rng = self.cfg.domain_rand.inertia_range
            for s in range(len(props)):
                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xx[env_id, s] = rd_num
                props[s].inertia.x.x *= rd_num
                
                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xy[env_id, s] = rd_num
                props[s].inertia.x.y *= rd_num
                props[s].inertia.y.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xz[env_id, s] = rd_num
                props[s].inertia.x.z *= rd_num
                props[s].inertia.z.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_yy[env_id, s] = rd_num
                props[s].inertia.y.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_yz[env_id, s] = rd_num
                props[s].inertia.y.z *= rd_num
                props[s].inertia.z.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_zz[env_id, s] = rd_num
                props[s].inertia.z.z *= rd_num
        
        
        return props

    def _reset_system(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids

        # todo: make separate methods for each reset type, cycle through `reset_mode` and call appropriate method. That way the base ones can be implemented once in legged_robot.
        """

        if hasattr(self, self.cfg.init_state.reset_mode):
            eval(f"self.{self.cfg.init_state.reset_mode}(env_ids)")
        else:
            raise NameError(f"Unknown default setup: {self.cfg.init_state.reset_mode}")

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # start base position shifted in X-Y plane
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.root_states[env_ids]
            self.root_states[env_ids, :3] += self.env_origins[env_ids] 

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    # * implement reset methods
    def reset_to_basic(self, env_ids):
        """
        Reset to a single initial state
        """
        #dof 
        self.dof_pos[env_ids] = self.default_dof_pos  #torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0 
        self.root_states[env_ids] = self.base_init_state

    def reset_to_range(self, env_ids):
        """
        Reset to a uniformly random distribution of states, sampled from a
        range for each state
        """
        # dof states
        self.dof_pos[env_ids] = random_sample(env_ids,
                                    self.dof_pos_range[:, 0],
                                    self.dof_pos_range[:, 1],
                                    device=self.device)
        self.dof_vel[env_ids] = random_sample(env_ids,
                        self.dof_vel_range[:, 0],
                        self.dof_vel_range[:, 1],
                        device=self.device)

        # base states
        random_com_pos = random_sample(env_ids,
                                    self.root_pos_range[:, 0],
                                    self.root_pos_range[:, 1],
                                    device=self.device)

        quat = quat_from_euler_xyz(random_com_pos[:, 3],
                                        random_com_pos[:, 4],
                                        random_com_pos[:, 5]) 

        self.root_states[env_ids, 0:7] = torch.cat((random_com_pos[:, 0:3],
                                    quat_from_euler_xyz(random_com_pos[:, 3],
                                                        random_com_pos[:, 4],
                                                        random_com_pos[:, 5])),
                                                    1)
        self.root_states[env_ids, 7:13] = random_sample(env_ids,
                                    self.root_vel_range[:, 0],
                                    self.root_vel_range[:, 1],
                                    device=self.device)


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # find ids for end effectors defined in env. specific config files
        ee_ids = []
        kp_ids = []
        for body_name in self.cfg.asset.end_effectors:
            ee_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            ee_ids.append(ee_id)
        for keypoint in self.cfg.asset.keypoints:
            kp_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], keypoint)
            kp_ids.append(kp_id)
        self.end_eff_ids = to_torch(ee_ids, device=self.device, dtype=torch.long)
        self.keypoint_ids = to_torch(kp_ids, device=self.device, dtype=torch.long)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "legged_robot")
        mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "legged_robot")

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.jacobians = gymtorch.wrap_tensor(jacobian_tensor)
        self.mass_matrices = gymtorch.wrap_tensor(mass_matrix_tensor)
        
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        #foot sensors
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        # self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # move to custom init
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx),
                                device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.],
                                device=self.device).repeat((self.num_envs, 1))

        # self.torques = torch.zeros(self.num_envs, self.num_actions,
        #                            dtype=torch.float, device=self.device,
        #                            requires_grad=False)
        # SE HWAN CRIME (?): why are the torques the same dimension as the output of the neural network?
        # They shouldn't need to be...

        self.torques = torch.zeros(self.num_envs, self.num_dof,
                                   dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device,
                                   requires_grad=False)
        # * additional buffer for last ctrl: whatever is actually used for PD control (which can be shifted compared to action)
        self.ctrl_hist = torch.zeros(self.num_envs, self.num_actions*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.dof_pos_hist = torch.zeros(self.num_envs, self.num_dof*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.dof_vel_hist = torch.zeros(self.num_envs, self.num_dof*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.base_ang_vel_hist = torch.zeros(self.num_envs, 9,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.proj_gravity_hist = torch.zeros(self.num_envs, 9,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.commands = torch.zeros(self.num_envs,
                                    self.cfg.commands.num_commands,
                                    dtype=torch.float, device=self.device,
                                    requires_grad=False) # x vel, y vel, yaw vel, height
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel,
                                            self.obs_scales.lin_vel,
                                            self.obs_scales.ang_vel],
                                           device=self.device,
                                           requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.max_feet_air_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs,
                                         len(self.feet_indices),
                                         dtype=torch.bool,
                                         device=self.device,
                                         requires_grad=False)
        self.time_to_stand_still = torch.zeros(self.num_envs,
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        if self.cfg.control.exp_avg_decay:
            self.action_avg = torch.zeros(self.num_envs, self.num_actions,
                                            dtype=torch.float,
                                            device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # * check that init range highs and lows are consistent
        # * and repopulate to match 
        if self.cfg.init_state.reset_mode == "reset_to_range":
            self.dof_pos_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
            self.dof_vel_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)

            for joint, vals in self.cfg.init_state.dof_pos_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_pos_range[i, :] = to_torch(vals)

            for joint, vals in self.cfg.init_state.dof_vel_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_vel_range[i, :] = to_torch(vals)

            self.root_pos_range = torch.tensor(self.cfg.init_state.root_pos_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            self.root_vel_range = torch.tensor(self.cfg.init_state.root_vel_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            # todo check for consistency (low first, high second)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.cfg.terrain.border_size 
        hf_params.transform.p.y = -self.cfg.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        
# ########################## REWARDS ######################## #

    # * "True" rewards * #

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return 0.5 + 0.5 * torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Reward tracking yaw angular velocity command
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2]))*2
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        nact = self.num_actions
        dt2 = (self.dt)**2
        error = torch.square(self.actions*self.cfg.control.action_scale \
                             - self.ctrl_hist[:, :nact] \
                            )/dt2
        return torch.sum(error, dim=1)
    
    def _reward_action_rate2(self):
        # Penalize changes in actions
        nact = self.num_actions
        dt2 = (self.dt)**2
        error = torch.square(self.actions*self.cfg.control.action_scale \
                            - 2.*self.ctrl_hist[:, :nact]  \
                            + self.ctrl_hist[:, nact:2*nact]  \
                            )/dt2
        return torch.sum(error, dim=1)
    
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    # * Shaping rewards * #

    def _reward_base_height(self):
        # Reward tracking specified base height
        ############### change later ##################
        base_height = self.root_states[:, 2].unsqueeze(1)
        # print(base_height)
        error = (base_height - (0.05 * self.commands[:,3].unsqueeze(1) + 1.03))
        error = error.flatten()
        return torch.exp(-1000. * torch.square(error))

    def _reward_orientation(self):
        # Reward tracking upright orientation
        error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-5.*error/self.cfg.rewards.tracking_sigma)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            10.*(self.dof_pos[:, 1]) / self.cfg.normalization.obs_scales.dof_pos)
        # print("reward")
        # print(self.sqrdexp(
        #     (self.dof_pos[:, 1]) / self.cfg.normalization.obs_scales.dof_pos))
        # print("dof pos 2")
        # print(self.dof_pos[:, 3])
        error += self.sqrdexp(
            10.*(self.dof_pos[:, 7]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint regularization around 0
        error += self.sqrdexp(
            5.*(self.dof_pos[:, 0])
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            5.*(self.dof_pos[:, 6])
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        # error += self.sqrdexp(
        #     ((self.dof_pos[:, 2] + self.dof_pos[:, 8]) / 2. - self.cfg.init_state.default_joint_angles['Joint-hip-r-pitch'])
        #     / self.cfg.normalization.obs_scales.dof_pos)
        return error/4

    def _reward_ankle_regularization(self):
        # Ankle joint regularization around 0
        error = 0
        error += self.sqrdexp(
            (2.5*self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (2.5*self.dof_pos[:, 11]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 4]-self.cfg.init_state.default_joint_angles['Joint-ankel-l-pitch']) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 10]-self.cfg.init_state.default_joint_angles['Joint-ankel-r-pitch']) / self.cfg.normalization.obs_scales.dof_pos)
        return error / 4.

    # Added Reward ---------------------------------------------
    def _reward_no_fly(self):
        # reward one-foot contact when moving
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*((self.time_to_stand_still <= self.static_delay) & single_contact)
    
    def _reward_no_jump(self):
        # Penalize feet in the air at zero commands(static)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        double_contact = torch.sum(1.*contacts, dim=1)==2
        return 1.* (double_contact) * (self.time_to_stand_still > self.static_delay)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt  # todo: pull this out into post-physics
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= ((self.time_to_stand_still <= self.static_delay))
        self.feet_air_time *= ~contact_filt
        self.max_feet_air_time = torch.where(self.max_feet_air_time > self.feet_air_time, self.max_feet_air_time, self.feet_air_time)
        error = self.max_feet_air_time[:, 0] - self.max_feet_air_time[:, 1]
        error = torch.square(error)
        return rew_airTime - error
    
    def _reward_stand_still(self):
        # penalty to keep COM in the middle of two feet
        # root = self.root_states[:, :2]
        # left_foot_pos = self._rigid_body_pos[:, self.feet_indices[0], :2]
        # right_foot_pos = self._rigid_body_pos[:, self.feet_indices[1], :2]
        # avg_foot_pos = (left_foot_pos + right_foot_pos) / 2.

        # static = (self.commands[:, 0] == 0) & (self.commands[:, 1] == 0) & (self.commands[:, 2] == 0) & (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.3)

        # return torch.norm((root - avg_foot_pos), dim=1) * static

        # reward for same force of two feet
        left_foot_force = self.contact_forces[:, self.feet_indices[0], 2]
        right_foot_force = self.contact_forces[:, self.feet_indices[1], 2]
        foot_force_acc = self.contact_force2 - self.contact_force1

        rew = torch.exp(-torch.square(0.01*(left_foot_force -right_foot_force)))
        rew += torch.sum(self.sqrdexp(0.01*foot_force_acc), dim=-1)/2.
        return rew * (self.time_to_stand_still > self.static_delay)


    
    def _reward_foot_slip(self):
        """
        penalize foot slip, including x,y linear velocity and yaw angular velocity, when contacting ground
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self._rigid_body_vel[:, self.feet_indices, :2], dim=2) * contact
        foot_ang_vel = torch.norm(self._rigid_body_ang[:, self.feet_indices, :], dim=2) * contact
        rew = torch.exp(-torch.sum(torch.sqrt(foot_speed_norm), dim=1)) \
                + torch.exp(-torch.sum(torch.sqrt(foot_ang_vel), dim=1))

        return rew/2.
    
    def _reward_base_lin_acc(self):
        # penalize large linear acc of root base
        base_lin_acc = (self.lin_vel2 - self.lin_vel1) / self.dt
        return torch.sum(torch.square(base_lin_acc[:, :]), dim=1)
    
    def _reward_base_ang_acc(self):
        # penalize large angular acc of root base
        base_ang_acc = (self.ang_vel2 - self.ang_vel1) / self.dt
        return torch.sum(torch.square(base_ang_acc[:, :]), dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.dof_vel_hist[:, :self.num_dof] - self.dof_vel)), dim=1)   

    def _reward_dof_vel(self):
        # Reward zero dof velocities
        dof_vel_scaled = self.dof_vel
        return torch.sum(torch.square(dof_vel_scaled), dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    
    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_standStillPrev = self._reward_stand_still()
        self.rwd_ankleRegPrev = self._reward_ankle_regularization()
        self.lin_vel1[:] = self.base_lin_vel
        self.ang_vel1[:] = self.base_ang_vel
        self.contact_force1[:] = self.contact_forces[:, self.feet_indices, 2]

    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step
    
    def _reward_ankleReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_ankle_regularization() - self.rwd_ankleRegPrev)
        return delta_phi / self.dt_step

# ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.

    def analyze_max_vel(self):
        x = torch.max(self.base_lin_vel[:, 0]).unsqueeze(0)
        y = torch.max(self.base_lin_vel[:, 1]).unsqueeze(0)
        z = torch.max(self.base_lin_vel[:, 2]).unsqueeze(0)
        roll = torch.max(self.base_ang_vel[:, 0]).unsqueeze(0)
        pitch = torch.max(self.base_ang_vel[:, 1]).unsqueeze(0)
        yaw = torch.max(self.base_ang_vel[:, 2]).unsqueeze(0)
        cur_max = torch.cat((x,y,z,roll,pitch,yaw))
        true_max = torch.where(self.max_vel > cur_max, self.max_vel, cur_max)
        self.max_vel[:] = true_max
        print(true_max)


    def print_rs(self):
        r_s = 0.
        nact = self.num_actions
        r_s -= 4000. * torch.square(self.actions*self.cfg.control.action_scale \
                            - self.ctrl_hist[:, :nact] \
                        )
        r_s -= 2000. * torch.square(self.actions*self.cfg.control.action_scale \
                        - 2.*self.ctrl_hist[:, :nact]  \
                        + self.ctrl_hist[:, nact:2*nact]  \
                        )
        r_s -= torch.square(self.dof_vel)
        r_s -= 4. * torch.square((self.dof_vel_hist[:, :self.num_dof] - self.dof_vel))
        return torch.exp(0.00012*torch.sum(r_s, dim=-1))
    

    # def random_env(self, env_ids):
    #     self.shape_frictions = torch_rand_float(0.3, 3, (self.num_envs, self.num_rigid_shape),device=self.device)
    #     mass_mask = torch_rand_float(0.5, 1.0, (self.num_envs, self.num_rigid_body),device=self.device)

    #     self.body_mass = mass_mask * self.default_body_mass
            
    #     for i in env_ids:
    #         props = self.gym.get_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i])
           
    #         for s in range(len(props)):
    #             props[s].friction = self.shape_frictions[i, s]
    #             props[s].friction = 0.01

    #         self.gym.set_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i], props)
 
            
    #         body_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.actor_handles[i])
                
    #         inertia_mask = torch_rand_float(0.7, 1.3, (len(body_props), 9),device=self.device)
    #         self.body_inertia[i] = inertia_mask * self.default_body_inertia[i]
                
    #         com_dif = torch_rand_float(-0.05, 0.05, (len(body_props), 3),device=self.device)
    #         # set base inertia com diff
    #         com_dif[-1] = (2 * torch.rand((3,), device=self.device) - 1) * 0.1
    #         self.body_com[i] = self.default_body_com[i] + com_dif
    #         for s in range(len(body_props)):
    #             pass
    #             # body_props[s].mass = self.body_mass[i, s]

                
                
    #             # body_props[s].inertia.x.x = self.body_inertia[i, s, 0]
    #             # body_props[s].inertia.x.y = self.body_inertia[i, s, 1]
    #             # body_props[s].inertia.x.z = self.body_inertia[i, s, 2]
    #             # body_props[s].inertia.y.x = self.body_inertia[i, s, 3]
    #             # body_props[s].inertia.y.y = self.body_inertia[i, s, 4]
    #             # body_props[s].inertia.y.z = self.body_inertia[i, s, 5]
    #             # body_props[s].inertia.z.x = self.body_inertia[i, s, 6]
    #             # body_props[s].inertia.z.y = self.body_inertia[i, s, 7]
    #             # body_props[s].inertia.z.z = self.body_inertia[i, s, 8]

    #             # body_props[s].com.x = self.body_com[i, s, 0]
    #             # body_props[s].com.y = self.body_com[i, s, 1]
    #             # body_props[s].com.z = self.body_com[i, s, 2]
    #         self.gym.set_actor_rigid_body_properties(self.envs[i], self.actor_handles[i],body_props)