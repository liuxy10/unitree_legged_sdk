#!/usr/bin/python

import sys
import time
import math
import numpy as np

import os
source_directory = "/home/tomtang/Documents/droplab/legged_gym" # have both legged_gym and unitree_sdk
print("Getting Policy Cofig from: ", os.path.dirname(os.path.realpath(__file__)))

# adding the parent directory to the sys.path.
sys.path.append(source_directory)

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from motion_stage import init_standup, cmd_angles_change, get_angle_state_flatten, get_ang_vel_flatten, adjust_standing, quit_tau

sys.path.append('../lib/python/amd64')
import robot_interface as sdk
from logger import Logger



stop_state_log = 100


"""
Policy Getting and Env Config
"""
def init_play(args):
    print("-"* 20, args.task, args, type(args))
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.draw_goals = args.draw_goals 

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    return policy, env, obs
    

"""
Constrains on angles
[0] Hip, [1] Thigh, [2] Calf
# """

go1_Hip_max = 1.047;     # unit:radian ( = 60   degree)
go1_Hip_min = -1.047;    # unit:radian ( = -60  degree)
go1_Thigh_max = 2.966;   # unit:radian ( = 170  degree)
go1_Thigh_min = -0.663;  # unit:radian ( = -38  degree)
go1_Calf_max = -0.837;   # unit:radian ( = -48  degree)
go1_Calf_min = -2.721;   # unit:radian ( = -156 degree)

default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip': 0.1,  # [rad]
        'RL_hip': 0.1,  # [rad]
        'FR_hip': -0.1,  # [rad]
        'RR_hip': -0.1,  # [rad]

        'FL_thigh': 0.8,  # [rad]
        'RL_thigh': 1.,  # [rad]
        'FR_thigh': 0.8,  # [rad]
        'RR_thigh': 1.,  # [rad]

        'FL_calf': -1.5,  # [rad]
        'RL_calf': -1.5,  # [rad]
        'FR_calf': -1.5,  # [rad]
        'RR_calf': -1.5  # [rad]
}

# action tensor formart: [hip, thigh, calf] [FL, FR, RL, RR]
default_angles = np.array([[0.1, 0.8, -1.5], [-0.1, 0.8, -1.5], [0.1, 1., -1.5], [-0.1, 1., -1.5]])
action_scale = 0.25

def checkSafe(act):
    return ((go1_Hip_min < act[0] < go1_Hip_max) and 
           (go1_Thigh_min < act[1] < go1_Thigh_max) and
           (go1_Calf_min < act[2] < go1_Calf_max))


def reindex(vec):
    return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
def construct_from_state(state, command):
    # wrapper from state to observation

    #  49 = 3 + 3 + 3 + 3 + 12 + 12 + 12 +1


    # self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
    # imu_obs,    #[1,2]
    # self.commands[:, :3] * self.commands_scale, #[1,3]
    # self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos), #[1,12]
    # self.reindex(self.dof_vel * self.obs_scales.dof_vel), #[1,12]
    # self.reindex(self.action_history_buf[:, -1]), #[1,12]
    # self.reindex_feet(self.contact_filt.float()-0.5), #[1,4]

    

    # init test:
    # return np.zeros(1, 49)

    # dummy var
    base_ang_vel = torch.tensor([state.imu.gyroscope[0], state.imu.gyroscope[0], state.imu.gyroscope[0]]).cuda()
    base_ang_vel = base_ang_vel[None, :]
    imu_obs = torch.tensor([state.imu.rpy[0], state.imu.rpy[1]]).cuda()
    imu_obs = imu_obs[None,:]
    comm =  command #command lin_vel_x, lin_vel_y, ang_vel_yaw



    dof_pos = torch.as_tensor(get_angle_state_flatten(state)).cuda()
    dof_pos = dof_pos[None, :]
    dof_vel = torch.as_tensor(get_ang_vel_flatten(state)).cuda()
    dof_vel = dof_vel[None, :]

    # height = torch.ones((1,1)).cuda() * 0.25
    dof_torque = torch.as_tensor([state.motorState[i].tauEst for i in range(12)])
    dof_torque = dof_torque[None,:]
    

    contact = torch.as_tensor([[state.footForce[0], state.footForce[1], state.footForce[2], state.footForce[3]]]).cuda()
    return base_ang_vel, imu_obs, comm, dof_pos, dof_vel, contact, dof_torque #torch.cat((),axis = -1)

def print_test_msg(info_dict):
    print("-"*25 + " step = "+str(motiontime) + " " + "-"*25)
    for key, value in info_dict.items():
        if isinstance(value, list):
            print(key, ["{:.4f}".format(val) for val in value])
        else:
            print(key, "{:.4f}".format(value))

if __name__ == '__main__':

    args = get_args()
    args.run_name = "test"
    args.task = 'go1_flat_walk'
    rn_policy, env, obs = init_play(args)
    
    
    stop_rew_log = 1000
    joint_index = 1 # which joint is used for logging
    logger = Logger(env.dt)

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff
    num_steps = 10000

    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    udp_h = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    highCmd = sdk.HighCmd()
    highState = sdk.HighState()
    udp_h.InitCmdData(highCmd)

    

    Tpi = 0
    motiontime = 0

    steps = 250
    motiontime, cur_angles = init_standup(cmd, udp, safe, state, steps = steps)

    # Kp = [[20] * 3 for _ in range(4)]
    # Kd = [[0.5] * 3 for _ in range(4)]
    # # [FL, FR, RL, RR] [hip, thigh, calf]
    # tau = [[15] * 3 for _ in range(4)] # scale 


    Kp = [[0] * 3 for _ in range(4)]
    Kd = [[0] * 3 for _ in range(4)]
    # [FL, FR, RL, RR] [hip, thigh, calf]
    tau = [[0] * 3 for _ in range(4)] # scale 


    # action torque tau
    motiontime = adjust_standing(cmd, udp, safe, state, motiontime, quit_tau, Kp, Kd, default_angles, steps = 100)

    qDes0, qDes1, qDes2, qDes3 = default_angles[0], default_angles[1], default_angles[2], default_angles[3]
    last_actions = torch.zeros((1, 12)).cuda()


    highCmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
    highCmd.gaitType = 0
    highCmd.speedLevel = 1
    highCmd.footRaiseHeight = 0.03
    highCmd.bodyHeight = 0.25
    highCmd.euler = [0, 0, 0]
    highCmd.velocity = [0, 0]
    highCmd.yawSpeed = 0.0
    highCmd.reserve = 0

    # for i in range(num_steps):
    while True:
        time.sleep(0.002)
        motiontime += 1
        # print(state.imu.rpy[0])
        
        udp.Recv()
        udp.GetRecv(state)  
        udp_h.Recv() 
        udp_h.GetRecv(highState)  

        command = env.commands[:, :3]
        
        

        base_ang_vel, imu_obs, comm, dof_pos, dof_vel, contact, dof_torque = construct_from_state(state, command)

        height = torch.ones((1, 1)).cuda() * 0.25
        oo = torch.cat((base_ang_vel, imu_obs, comm, dof_pos, dof_vel, last_actions, contact, height),axis = -1)

        actions = rn_policy(oo.cuda())

        actions_arr = actions.cpu().detach().numpy().reshape(-1, 3)

        target_angle = actions_arr * action_scale + default_angles

        cmd = cmd_angles_change(cmd, state, safe, tau = tau, Kp=Kp, Kd=Kd, qDes=target_angle)

        info_dict = {

                        # 'dof_pos_target': actions[0, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos_target': [actions[0, i].item() * env.cfg.control.action_scale for i in range(len(actions[0]))],
                        'dof_pos': [dof_pos[0, i].item() for i in range(len(dof_pos[0]))],
                        'dof_vel': [dof_vel[0, i].item() for i in range(len(dof_vel[0]))],
                        'base_acc_z': state.imu.accelerometer[2],
                        'base_acc_x': state.imu.accelerometer[0],
                        'base_acc_y': state.imu.accelerometer[1],
                        'base_pos_z': highState.position[2],
                        'base_pos_x': highState.position[0],
                        'base_pos_y': highState.position[1],
                        
                        'dof_torque': [dof_torque[0, i].item() for i in range(len(dof_torque[0]))],
                        'command_x_vel': env.commands[0, 0].item(),
                        'command_y_vel': env.commands[0, 1].item(),
                        'command_yaw': env.commands[0, 2].item(),
                        'base_vel_x': highState.velocity[0],
                        'base_vel_y': highState.velocity[1],
                        'base_vel_z': highState.velocity[2],
                        'base_vel_roll': state.imu.rpy[0],
                        'base_vel_pitch': state.imu.rpy[1],
                        'base_vel_yaw': state.imu.rpy[2],
                        'contact_forces_z': [contact[0,i].item() for i in range(len(contact[0]))],
                        
            }
        print_test_msg(info_dict)

        # if i < stop_state_log:
        #     logger.log_states(info_dict
        # elif i==stop_state_log:
        #     logger.plot_states()
        
        last_actions = actions

        udp.SetSend(cmd)
        udp.Send()
        udp_h.SetSend(highCmd)
        udp_h.Send()
    
    print("Done")