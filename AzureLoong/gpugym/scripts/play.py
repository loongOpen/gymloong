# policy_class_name = 'ActorCritic'
# algorithm_class_name = 'PPO'
# num_steps_per_env = 24 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
# max_iterations = 1500 # number of policy updates
# # logging
# save_interval = 50 # check for potential saves every this many iterations
# run_name = ''
# experiment_name = 'legged_robot'# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger
from gpugym.utils.ploter import Plotter, initCanvas

import numpy as np
import torch
import matplotlib.pyplot as plt



def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = True
    # utils terrain.py curiculum modified
    env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.terrain_proportions = [1, 0, 0, 0, 0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_all_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.random_pd = False
    env_cfg.domain_rand.random_damping = False
    env_cfg.domain_rand.random_inertia = False
    env_cfg.domain_rand.comm_delay = False
    # env_cfg.domain_rand.comm_delay_range = [0, 1]
    # env_cfg.domain_rand.randomize_friction = True
    # env_cfg.domain_rand.friction_range = [0.05, 0.1]
    env_cfg.init_state.reset_ratio = 0.8
    # env_cfg.domain_rand.comm_delay_range = [10, 11]


    # custom settings
    print("------------------------")
    print("Using play custom settings")
    print("------------------------")
    env_cfg.commands.ranges.lin_vel_x = [-1.5, 1.5]
    env_cfg.commands.ranges.lin_vel_y = [-0.75, 0.75]
    env_cfg.commands.ranges.ang_vel_yaw = [-1., 1.]

    env_cfg.init_state.pos = [0., 0., 1.122]        # x,y,z [m]  
    env_cfg.init_state.rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
    env_cfg.init_state.lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
    env_cfg.init_state.ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

    env_cfg.init_state.root_pos_range = [
                [0., 0.],
                [0., 0.],
                [1.122, 1.122],      
                [-0, 0],
                [-0, 0],
                [-0, 0]
        ]
    env_cfg.init_state.root_vel_range = [
                [-0., 0.],
                [-0., 0.],
                [-0., 0.],
                [-0., 0.],
                [-0., 0.],
                [-0., 0.]
        ] 

    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.resampling_time = 5.

    #different commmand for testing
    #stand_still
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_y = [0., 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [0, 0.0]

    #walking without turning
    # env_cfg.commands.ranges.lin_vel_x = [-1.5, 1.5]
    # env_cfg.commands.ranges.lin_vel_y = [-0.75, 0.75]
    # env_cfg.commands.ranges.ang_vel_yaw = [0, 0]

    #turning without walking
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_y = [0., 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [1., 3.]

    #turning without walking at low speed
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_y = [0., 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [.5, 1.]


    #high speed running
    # env_cfg.commands.ranges.lin_vel_x = [4., 4.]
    # env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [0, 0]

    # env_cfg.commands.ranges.lin_vel_x = [1., 1.]
    # env_cfg.commands.ranges.lin_vel_y = [0.7, 0.7]
    # env_cfg.commands.ranges.ang_vel_yaw = [1, 1]

    # train_cfg.runner.checkpoint = -1

    # env_cfg.init_state.reset_mode = 'reset_to_basic'

    # env_cfg.init_state.pos = [0., 0., 1.5]
    # env_cfg.asset.disable_gravity = True
    # env_cfg.asset.disable_motors = False

    # the_actions = torch.tensor(np.array([0.,0.,0,-0.,0,0.4,
    #                     0.,0,0,0.,0,0]),dtype=torch.float32).unsqueeze(0)




    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy(ppo_runner.alg.actor_critic, path)
        print('Exported policy model to: ', path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')
        export_critic(ppo_runner.alg.actor_critic, path)
        print('Exported critic model to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    #------------------------
    plot_interval = 5

    play_log = []
    flag = False # if plot shoule appear
    plt.ion()

    initCanvas(3, 2, 200)
    plotter0 = Plotter(0, 'plot 0')
    plotter1 = Plotter(1, 'plot 1')
    plotter2 = Plotter(2, 'plot 2')
    plotter3 = Plotter(3, 'plot 3')
    plotter4 = Plotter(4, 'plot 4')
    plotter5 = Plotter(5, 'plot 5')

    
    env.max_episode_length = 1000./env.dt
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        # actions = the_actions
        obs, _, rews, dones, infos = env.step(actions.detach())
        # obs, _, rews, dones, infos = env.step(the_actions)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if flag:
            # actions avg
            plotter0.plotLine(env.dof_pos[0, 0].item(), env.action_avg[0, 0].item() * env_cfg.control.action_scale - 0.,
                              labels=['actual', 'action'])
            plotter1.plotLine(env.dof_pos[0, 1].item(), env.action_avg[0, 1].item() * env_cfg.control.action_scale + 0.,
                              labels=['actual', 'action'])
            plotter2.plotLine(env.dof_pos[0, 2].item(),
                              env.action_avg[0, 2].item() * env_cfg.control.action_scale + 0.305913,
                              labels=['actual', 'action'])
            plotter3.plotLine(env.dof_pos[0, 3].item(),
                              env.action_avg[0, 3].item() * env_cfg.control.action_scale - 0.670418,
                              labels=['actual', 'action'])
            plotter4.plotLine(env.dof_pos[0, 4].item(),
                              env.action_avg[0, 4].item() * env_cfg.control.action_scale + 0.371265,
                              labels=['actual', 'action'])
            plotter5.plotLine(env.dof_pos[0, 5].item(), env.action_avg[0, 5].item() * env_cfg.control.action_scale + 0.,
                              labels=['actual', 'action'])

        pass

        env._reward_stand_still()
        if i < stop_state_log:
            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            play_log.append(
                [i*env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[0, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
            )
        elif i==stop_state_log:
            np.savetxt('../analysis/data/play_log.csv', play_log, delimiter=',')
            # logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes>0:
                    # logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    


if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_CRITIC = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)