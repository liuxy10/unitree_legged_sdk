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

def jointLinearInterpolation(initPos, targetPos, rate):

    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p


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
    base_ang_vel = torch.zeros((1, 3)).cuda()
    imu_obs = torch.tensor([state.imu.rpy[0], state.imu.rpy[1]]).cuda()
    imu_obs = imu_obs[None,:]
    comm =  command #command lin_vel_x, lin_vel_y, ang_vel_yaw



    dof_pos = torch.as_tensor(get_angle_state_flatten(state)).cuda()
    dof_pos = dof_pos[None, :]
    dof_vel = torch.as_tensor(get_angle_state_flatten(state)).cuda()
    dof_vel = dof_vel[None, :]

    print("dof_pos, dof_vel = ",dof_pos, dof_vel)

    action = torch.zeros((1, 12)).cuda()
    contact = torch.zeros((1, 4)).cuda()
    height = torch.ones((1,1)).cuda() * 0.25
    return base_ang_vel, imu_obs, comm, dof_pos, dof_vel, action, contact,height #torch.cat((),axis = -1)



if __name__ == '__main__':

    args = get_args()
    args.run_name = "test"
    args.task = 'go1_flat_walk'
    rn_policy, env, obs = init_play(args)
    
    
    stop_rew_log = 100
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
    sin_mid_q = [0.0, 1.2, -2.0]
    dt = 0.002
    qInit = [0, 0, 0]
    qDes = [0, 0, 0]
    sin_count = 0
    rate_count = 0
    Kp = [0, 0, 0]
    Kd = [0, 0, 0]

    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime += 1

        # print(motiontime)
        # print(state.imu.rpy[0])
        
        
        udp.Recv()
        udp.GetRecv(state)
        
        if( motiontime >= 0):

            # first, get record initial position
            if( motiontime >= 0 and motiontime < 10):
                qInit[0] = state.motorState[d['FR_0']].q
                qInit[1] = state.motorState[d['FR_1']].q
                qInit[2] = state.motorState[d['FR_2']].q
            # print(state.motorState[d['FR_0']].q)
            print(state.imu.rpy)
            
            # second, move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 10 and motiontime < 400):
                rate_count += 1
                rate = rate_count/200.0                       # needs count to 200
                Kp = [5, 5, 5]
                Kd = [1, 1, 1]
                # Kp = [20, 20, 20]
                # Kd = [2, 2, 2]
                
                qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate)
                qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate)
                qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate)
            
            # last, do sine wave
            freq_Hz = 1
            # freq_Hz = 5
            freq_rad = freq_Hz * 2* math.pi
            t = dt*sin_count
            if( motiontime >= 400):
                sin_count += 1
                # sin_joint1 = 0.6 * sin(3*M_PI*sin_count/1000.0)
                # sin_joint2 = -0.9 * sin(3*M_PI*sin_count/1000.0)
                sin_joint1 = 0.6 * math.sin(t*freq_rad)
                sin_joint2 = -0.9 * math.sin(t*freq_rad)
                qDes[0] = sin_mid_q[0]
                qDes[1] = sin_mid_q[1] + sin_joint1
                qDes[2] = sin_mid_q[2] + sin_joint2
            

            cmd.motorCmd[d['FR_0']].q = qDes[0]
            cmd.motorCmd[d['FR_0']].dq = 0
            cmd.motorCmd[d['FR_0']].Kp = Kp[0]
            cmd.motorCmd[d['FR_0']].Kd = Kd[0]
            cmd.motorCmd[d['FR_0']].tau = -0.65

            cmd.motorCmd[d['FR_1']].q = qDes[1]
            cmd.motorCmd[d['FR_1']].dq = 0
            cmd.motorCmd[d['FR_1']].Kp = Kp[1]
            cmd.motorCmd[d['FR_1']].Kd = Kd[1]
            cmd.motorCmd[d['FR_1']].tau = 0.0

            cmd.motorCmd[d['FR_2']].q =  qDes[2]
            cmd.motorCmd[d['FR_2']].dq = 0
            cmd.motorCmd[d['FR_2']].Kp = Kp[2]
            cmd.motorCmd[d['FR_2']].Kd = Kd[2]
            cmd.motorCmd[d['FR_2']].tau = 0.0
            # cmd.motorCmd[d['FR_2']].tau = 2 * sin(t*freq_rad)


        if(motiontime > 10):
            safe.PowerProtect(cmd, state, 1)

        udp.SetSend(cmd)
        udp.Send()
