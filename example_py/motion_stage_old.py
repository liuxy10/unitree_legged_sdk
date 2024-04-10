#!/usr/bin/python

import sys
import time
import math
import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

"""
--------------
defualt values
--------------
"""
go1_Hip_max = 1.047;     # unit:radian ( = 60   degree)
go1_Hip_min = -1.047;    # unit:radian ( = -60  degree)
go1_Thigh_max = 2.966;   # unit:radian ( = 170  degree)
go1_Thigh_min = -0.663;  # unit:radian ( = -38  degree)
go1_Calf_max = -0.837;   # unit:radian ( = -48  degree)
go1_Calf_min = -2.721;   # unit:radian ( = -156 degree)

d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5, 
     'RR_0':6, 'RR_1':7, 'RR_2':8, 
     'RL_0':9, 'RL_1':10, 'RL_2':11 }

# defualt stiffness, damping, and tau [FL, FR, RL, RR] [hip, thigh, calf]
deKp = [[15, 15, 15], 
        [15, 15, 15], 
        [15, 15, 15], 
        [15, 15, 15]]
deKd = [[1, 1, 1], 
        [1, 1, 1], 
        [1, 1, 1], 
        [1, 1, 1]]
detau = [[2, 2, 2], 
         [2, 2, 2], 
         [2, 3, 10], 
         [2, 3, 10]]

default_standing_angles = np.array([[0.0, 0.4, -1.6], 
                                    [-0.2, 0.4, -1.6], 
                                    [-0.1, 0.8, -1.65], 
                                    [-0.2, 0.8, -1.65]], dtype=np.float128)

quit_threshold = 10000 # quit after this amount of motion time
quit_tau = [[0] * 3 for _ in range(4)]
quit_kp = [[5] * 3 for _ in range(4)]
quit_kd = [[1] * 3 for _ in range(4)]

lg_default_angles = np.array([[0.1, 0.8, -1.5], [-0.1, 0.8, -1.5], [0.1, 1., -1.5], [-0.1, 1., -1.5]])
lg_Kp = [[20] * 3 for _ in range(4)]
lg_Kd = [[0.5] * 3 for _ in range(4)]
action_scale = 0.25

"""
--------------
helper function for single/several steps operation(s)
--------------
"""
def jointLinearInterpolation(initPos, targetPos, rate):
    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

def checkSafe(act):
    return ((go1_Hip_min < act[0] < go1_Hip_max) and 
           (go1_Thigh_min < act[1] < go1_Thigh_max) and
           (go1_Calf_min < act[2] < go1_Calf_max))

def get_angle_state(state):
    FL = [state.motorState[d['FL_0']].q, state.motorState[d['FL_1']].q, state.motorState[d['FL_2']].q]
    FR = [state.motorState[d['FR_0']].q, state.motorState[d['FR_1']].q, state.motorState[d['FR_2']].q]
    RL = [state.motorState[d['RL_0']].q, state.motorState[d['RL_1']].q, state.motorState[d['RL_2']].q]
    RR = [state.motorState[d['RR_0']].q, state.motorState[d['RR_1']].q, state.motorState[d['RR_2']].q]
    return [FL, FR, RL, RR]

def cmd_angles_change(cmd, state, safe, tau=detau, Kp=deKp, Kd=deKd, qDes=default_standing_angles):
    Kd0, Kd1, Kd2, Kd3 = Kd
    Kp0, Kp1, Kp2, Kp3 = Kp
    tau0, tau1, tau2, tau3 = tau
    qDes0, qDes1, qDes2, qDes3 = qDes
    if checkSafe(qDes0):
        cmd.motorCmd[d['FL_0']].q = qDes0[0]
        cmd.motorCmd[d['FL_0']].dq = 0
        cmd.motorCmd[d['FL_0']].Kp = Kp0[0]
        cmd.motorCmd[d['FL_0']].Kd = Kd0[0]
        cmd.motorCmd[d['FL_0']].tau = tau0[0]
        
        cmd.motorCmd[d['FL_1']].q = qDes0[1]
        cmd.motorCmd[d['FL_1']].dq = 0
        cmd.motorCmd[d['FL_1']].Kp = Kp0[1]
        cmd.motorCmd[d['FL_1']].Kd = Kd0[1]
        cmd.motorCmd[d['FL_1']].tau = tau0[1]

        cmd.motorCmd[d['FL_2']].q =  qDes0[2]
        cmd.motorCmd[d['FL_2']].dq = 0
        cmd.motorCmd[d['FL_2']].Kp = Kp0[2]
        cmd.motorCmd[d['FL_2']].Kd = Kd0[2]
        cmd.motorCmd[d['FL_2']].tau = tau0[2]
                 
    if checkSafe(qDes1):
        cmd.motorCmd[d['FR_0']].q = qDes1[0]
        cmd.motorCmd[d['FR_0']].dq = 0
        cmd.motorCmd[d['FR_0']].Kp = Kp1[0]
        cmd.motorCmd[d['FR_0']].Kd = Kd1[0]
        cmd.motorCmd[d['FR_0']].tau = tau1[0]

        cmd.motorCmd[d['FR_1']].q = qDes1[1]
        cmd.motorCmd[d['FR_1']].dq = 0
        cmd.motorCmd[d['FR_1']].Kp = Kp1[1]
        cmd.motorCmd[d['FR_1']].Kd = Kd1[1]
        cmd.motorCmd[d['FR_1']].tau = tau1[1]

        cmd.motorCmd[d['FR_2']].q =  qDes1[2]
        cmd.motorCmd[d['FR_2']].dq = 0
        cmd.motorCmd[d['FR_2']].Kp = Kp1[2]
        cmd.motorCmd[d['FR_2']].Kd = Kd1[2]
        cmd.motorCmd[d['FR_2']].tau = tau1[2]

    if checkSafe(qDes2):
        cmd.motorCmd[d['RL_0']].q = qDes2[0]
        cmd.motorCmd[d['RL_0']].dq = 0
        cmd.motorCmd[d['RL_0']].Kp = Kp2[0]
        cmd.motorCmd[d['RL_0']].Kd = Kd2[0]
        cmd.motorCmd[d['RL_0']].tau = tau2[0]
        
        cmd.motorCmd[d['RL_1']].q = qDes2[1]
        cmd.motorCmd[d['RL_1']].dq = 0
        cmd.motorCmd[d['RL_1']].Kp = Kp2[1]
        cmd.motorCmd[d['RL_1']].Kd = Kd2[1]
        cmd.motorCmd[d['RL_1']].tau = tau2[1]

        cmd.motorCmd[d['RL_2']].q =  qDes2[2]
        cmd.motorCmd[d['RL_2']].dq = 0
        cmd.motorCmd[d['RL_2']].Kp = Kp2[2]
        cmd.motorCmd[d['RL_2']].Kd = Kd2[2]
        cmd.motorCmd[d['RL_2']].tau = tau2[2]

    if checkSafe(qDes3):
        cmd.motorCmd[d['RR_0']].q = qDes3[0]
        cmd.motorCmd[d['RR_0']].dq = 0
        cmd.motorCmd[d['RR_0']].Kp = Kp3[0]
        cmd.motorCmd[d['RR_0']].Kd = Kd3[0]
        cmd.motorCmd[d['RR_0']].tau = tau3[0]

        cmd.motorCmd[d['RR_1']].q = qDes3[1]
        cmd.motorCmd[d['RR_1']].dq = 0
        cmd.motorCmd[d['RR_1']].Kp = Kp3[1]
        cmd.motorCmd[d['RR_1']].Kd = Kd3[1]
        cmd.motorCmd[d['RR_1']].tau = tau3[1]

        cmd.motorCmd[d['RR_2']].q =  qDes3[2]
        cmd.motorCmd[d['RR_2']].dq = 0
        cmd.motorCmd[d['RR_2']].Kp = Kp3[2]
        cmd.motorCmd[d['RR_2']].Kd = Kd3[2]
        cmd.motorCmd[d['RR_2']].tau = tau3[2]     
    safe.PowerProtect(cmd, state, 1)

"""
--------------
operation function for operation(s) in different stages
--------------
"""
def init_standup(cmd, udp, safe, state, steps = 150):
    # action tensor formart: [hip, thigh, calf] [FL, FR, RL, RR]
    
    udp.Recv()
    udp.GetRecv(state)
    # udp.SetSend(cmd)
    # udp.Send()
    motiontime = 0
    time.sleep(0.002)
    
    udp.Recv()
    udp.GetRecv(state)
    
    motiontime += 1
        
    print(state.motorState[d['FL_0']].q, state.motorState[d['FL_1']].q, state.motorState[d['FL_2']].q)
    for i in range(10):
        time.sleep(0.01)
        L = np.array([
            [state.motorState[d['FL_0']].q, state.motorState[d['FL_1']].q, state.motorState[d['FL_2']].q],
            [state.motorState[d['FR_0']].q, state.motorState[d['FR_1']].q, state.motorState[d['FR_2']].q],
            [state.motorState[d['RL_0']].q, state.motorState[d['RL_1']].q, state.motorState[d['RL_2']].q],
            [state.motorState[d['RR_0']].q, state.motorState[d['RR_1']].q, state.motorState[d['RR_2']].q]
        ], dtype=np.float128)

    dangle = ((default_standing_angles - L) / steps)
    i = 0
    for j in range(steps + 10):
        time.sleep(0.01)
        print(f"{motiontime}: standing up {i} / {steps} done\n")
        udp.Recv()
        udp.GetRecv(state)
        motiontime += 1
        if i < steps:
            i += 1
        tangle = L + dangle * i

        cmd_angles_change(cmd, state, safe, qDes=tangle)
        udp.SetSend(cmd)
        udp.Send()

    return motiontime, get_angle_state(state)

def maintain_standing(cmd, udp, safe, state, motiontime, steps = None):
    if steps == None:
        steps = 1e20
    udp.Recv()
    udp.GetRecv(state)
    for i in range(int(steps)):
        motiontime += 1
        udp.Recv()
        udp.GetRecv(state)
        # udp.SetSend(cmd)
        # udp.Send()
        motiontime = 0
        time.sleep(0.002)
        if i > quit_threshold:
            cmd_angles_change(cmd, state, safe, tau = quit_tau, Kp=quit_kp, Kd=quit_kd)
        else:
            cmd_angles_change(cmd, state, safe)
        udp.SetSend(cmd)
        udp.Send()
    return motiontime

def adjust_standing(cmd, udp, safe, state, motiontime, tau=detau, Kp=deKp, Kd=deKd, qDes=default_standing_angles, steps=50):
    udp.Recv()
    udp.GetRecv(state)
    print("Adjusting to policy standing gesture")
    for i in range(int(steps)):
        time.sleep(0.002)
        motiontime += 1
        udp.Recv()
        udp.GetRecv(state)
        # udp.SetSend(cmd)
        # udp.Send()
        motiontime = 0
        time.sleep(0.002)
        cmd_angles_change(cmd, state, safe, tau, Kp, Kd, qDes)
        
        udp.SetSend(cmd)
        udp.Send()
    return motiontime































if __name__ == '__main__':

    
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff
    sin_mid_q = [0.0, 1.2, -2.0]

    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    

    steps = 150
    motiontime, q_states = init_standup(cmd, udp, safe, state, steps = steps)

    adjust_standing(cmd, udp, safe, state, motiontime, quit_tau, lg_Kp, lg_Kd, lg_default_angles, steps = 10000)
    maintain_standing(cmd, udp, safe, state, motiontime)
    print("Done", motiontime)