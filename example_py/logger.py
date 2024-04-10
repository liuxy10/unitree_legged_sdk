# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import os
from motion_stage import *

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()
        


        

    def _plot(self): 
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log



        # plot base xy 
        a = axs[0, 0]
        if log["base_pos_x"]: a.scatter(log["base_pos_x"], log["base_pos_y"], label='measured',c = time ,cmap="plasma", s = 5)
        if log["base_yaw"]: 
            draw_arrows(a, log["base_yaw"],log["base_pos_x"], log["base_pos_y"], density=30)
        if log["command_x_pos"]: 
            
            unique_xs, unique_ys = select_unique_points(log["command_x_pos"], log["command_y_pos"])
            points = a.scatter(unique_xs, unique_ys, label='commanded', s = 10)
            draw_circle(a, [unique_xs, unique_ys], radius = .15) # need to correspond to threshold distance of the goal reach function
            
        if log["origin_x"]:
            
            draw_ellipse(a, [log["origin_x"][0],log["origin_y"][0]], [unique_xs, unique_ys], 0.8)
        # a.set(xlabel='position x [m]', ylabel='position y [m]', title='xy trajectory')
        if log["command_x_pos"]: fig.colorbar(points)
        a.set_xlim([-1,1])
        a.set_ylim([-1,1])
        a.legend()


        # # plot base vel x
        a = axs[0,1]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x_vel"]: a.plot(time, log["command_x_vel"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.set_ylim([-1,1])
        a.legend()

         # plot base vel y
        a = axs[0,2]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y_vel"]: a.plot(time, log["command_y_vel"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.set_ylim([-1,1])
        a.legend()

        
        # # plot base z
        # a = axs[1, 0]
        # if log["base_pos_z"]: a.scatter(time, log["base_pos_z"], label='measured')
        # if log["command_z_pos"]: a.plot(time, log["command_z_pos"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base z [m]', title='Base z position')
        # a.set_ylim([-1,1])
        # a.legend()



        # plot base vel yaw
        a = axs[1, 1]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured_yaw')
        if log["base_vel_roll"]: a.plot(time, log["base_vel_roll"], label='measured_roll')
        if log["base_vel_pitch"]: a.plot(time, log["base_vel_pitch"], label='measured_pitch')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.set_ylim([-1,1])
        a.legend()




        # plot contact forces
        a = axs[1, 2]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[2, 0]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        a = axs[2, 1]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()


        # # plot base vel z
        # a = axs[2, 2]
        # if log["base_vel_z"]: a.scatter(time, log["base_vel_z"], label='measured')
        # if log["command_z"]: a.scatter(time, log["command_z"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        # a.legend()

        # plot joint targets and measured positions
        # a = axs[1, 0]
        # if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        # if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        # a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        # a.legend()
        # # plot joint velocity
        # a = axs[1, 1]
        # if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        # if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        # a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        # a.legend()


        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()


def draw_arrows(ax, headings, xs, ys, density=30):
    """
    Plot trajectories with headings as arrows.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to draw on.
    headings (list): List of headings in degrees.
    xs (list or numpy.ndarray): List or array of x-coordinates for trajectories.
    ys (list or numpy.ndarray): List or array of y-coordinates for trajectories.
    density (int): Density of arrows to plot along trajectories.

    Returns:
    None
    """


    # Plot trajectories
    ax.plot(xs, ys)

    # Plot arrows along trajectories
    # for x, y, heading_rad in zip(xs, ys, headings_rad):
    for i in range(len(xs)):
        if i % density == 0:
        # # Calculate arrow points
        # arrow_x = np.linspace(x, x + np.cos(heading_rad), density)
        # arrow_y = np.linspace(y, y + np.sin(heading_rad), density)

        # Plot arrows
            ax.arrow(xs[i], ys[i], np.cos(headings[i]), np.sin(headings[i]),
                    head_width=0.01, head_length=0.001, fc='red', ec='red')

    # ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectories with Headings')




    
def draw_ellipse(ax, focus1, focus2s, diff = 0.8):
    """
    Draw an ellipse based on the positions of the foci and the length of the major axis.

    Parameters:
    focus1 (tuple): Position of the first focus as a tuple (x, y).
    focus2 (tuple): Position of the second focus as a tuple (x, y).
    diff (float): |a-c|.

    Returns:
    None
    """
    f2xs, f2ys = focus2s[0], focus2s[1]
    for (f2x,f2y) in zip(f2xs, f2ys):
        distance_foci = np.linalg.norm([focus1[0] - f2x, focus1[1] - f2y])
        major_axis_length = distance_foci + diff*2
        
        # Calculate the center of the ellipse
        center = ((focus1[0] + f2x) / 2, (focus1[1] + f2y) / 2)

        # Calculate the semi-major axis length
        semi_major_axis = major_axis_length / 2

        # Calculate the semi-minor axis length
        semi_minor_axis = np.sqrt((major_axis_length / 2) ** 2 - (distance_foci / 2) ** 2)

        # Create an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, 100)

        # Parametric equations for an ellipse
        x = center[0] + semi_major_axis * np.cos(theta)
        y = center[1] + semi_minor_axis * np.sin(theta)

        # Plot the ellipse
        ax.plot(x, y, c = 'b')


def draw_circle(ax, centers, radius = .15):
    # Create an array of angles from 0 to 2*pi
    cxs, cys = centers[0], centers[1]
    
    for cx, cy in zip(cxs, cys):
        theta = np.linspace(0, 2 * np.pi, 100)

        # Parametric equations for an ellipse
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        ax.set_aspect('equal', adjustable='box')

        ax.plot(x,y, c = 'g')

def select_unique_points(xs, ys):
    unique_points = {}  # Dictionary to store unique points

    # Iterate through each point
    for x, y in zip(xs, ys):
        # Check if the point already exists in the dictionary
        if (x, y) not in unique_points:
            # If not, add it to the dictionary
            unique_points[(x, y)] = True
    
    # Extract the unique x and y coordinates
    unique_xs = [point[0] for point in unique_points.keys()]
    unique_ys = [point[1] for point in unique_points.keys()]

    return unique_xs, unique_ys

if __name__ == "__main__":

    # test if the logger is able to receive signals whenever there are sensor measurements
        
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff
    sin_mid_q = [0.0, 1.2, -2.0]

    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    highState = sdk.HighState()
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    Kp = [[0] * 3 for _ in range(4)]
    Kd = [[0] * 3 for _ in range(4)]
    # [FL, FR, RL, RR] [hip, thigh, calf]
    tau = [[0] * 3 for _ in range(4)] # scale 
    target_angle = [[0] * 3 for _ in range(4)]
    stop_state_log =100
    

    steps = 150
    while True:
        time.sleep(0.002)
        motiontime += 1
        # print(state.imu.rpy[0])
        
        udp.Recv()
        udp.GetRecv(state)   
        udp.GetRecv(highState)  
        
        cmd = cmd_angles_change(cmd, state, safe, 
                                tau = tau, 
                                Kp=Kp, 
                                Kd=Kd, 
                                qDes=target_angle)
        
       