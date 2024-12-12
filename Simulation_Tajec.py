import roboticstoolbox as rtb
import numpy as np

from spatialmath import SE3
from spatialmath.base import *
from math import pi
from spatialmath.base import tr2rpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Head_to_Waist = 10 #m
L1 = 15
L2 = 22

# ขาที่ 1
Waist_to_J1_1 = 6

# ขาที่ 2
Waist_to_J1_2 = 6

# ขาที่ 3
Waist_to_J1_3 = 6

Leg1 = rtb.DHRobot(
    [
        rtb.RevoluteMDH(alpha = pi/2, offset = pi/2),
        rtb.RevoluteMDH(a = L2),
        rtb.RevoluteMDH(a = L1, offset = pi/2),
        rtb.RevoluteMDH(a = Waist_to_J1_1, alpha = pi/2, offset = pi),
        rtb.RevoluteMDH(),
        rtb.RevoluteMDH(d = Head_to_Waist),
        rtb.RevoluteMDH(alpha = pi/2),
        rtb.RevoluteMDH(a = 5 , alpha = -pi/2),
        rtb.RevoluteMDH(a = -5),
    ],base = SE3(Waist_to_J1_1, 0, 0),
    name = "Leg1"
)

Leg2 = rtb.DHRobot(
    [
        rtb.RevoluteMDH(alpha = pi/2, offset = pi/2),
        rtb.RevoluteMDH(a = L2),
        rtb.RevoluteMDH(a = L1),
        rtb.RevoluteMDH(alpha = -pi/3, offset = -pi/2),
        rtb.RevoluteMDH(a = Waist_to_J1_2, alpha = -pi/2, offset = pi/3),
        rtb.RevoluteMDH(d = Head_to_Waist),
        rtb.RevoluteMDH(alpha = pi/2),
        rtb.RevoluteMDH(a = 5 , alpha = -pi/2),
        rtb.RevoluteMDH(a = -5),
    ],base =  SE3(-np.sin(np.deg2rad(30))*Waist_to_J1_2, np.cos(np.deg2rad(30))*Waist_to_J1_2, 0),
    name = "Leg2"
)

Leg3 = rtb.DHRobot(
    [
        rtb.RevoluteMDH(alpha = pi/2, offset = pi/2),
        rtb.RevoluteMDH(a = L2),
        rtb.RevoluteMDH(a = L1),
        rtb.RevoluteMDH(alpha = pi/3, offset = -pi/2),
        rtb.RevoluteMDH(a = Waist_to_J1_3, alpha = -pi/2, offset = -pi/3),
        rtb.RevoluteMDH(d = Head_to_Waist),
        rtb.RevoluteMDH(alpha = pi/2),
        rtb.RevoluteMDH(a = 5 , alpha = -pi/2),
        rtb.RevoluteMDH(a = -5),
    ],base =  SE3(-np.sin(np.deg2rad(30))*Waist_to_J1_3, -np.cos(np.deg2rad(30))*Waist_to_J1_3, 0),
    
    name = "Leg3"
)


Height = 22
h = Height - Head_to_Waist

max = L1 + L2 + Head_to_Waist
min = abs(L1 - L2)+Head_to_Waist
for i in range(min, max):
    beta_test = np.arccos((L1**2 + L2**2 - (i-Head_to_Waist)**2) / (2 * L1 * L2))
    q2_test = np.deg2rad(180) - beta_test
    x_h_test = L2 + L1*np.cos(q2_test)
    y_h_test = L1*np.sin(q2_test)
    out_theta_test = np.arctan(x_h_test/y_h_test)
    alpha_test = np.deg2rad(90)- out_theta_test
    
    if (np.rad2deg(alpha_test) < 90):
        min = i
        break

beta = np.arccos((L1**2 + L2**2 - h**2) / (2 * L1 * L2))
q2 = np.deg2rad(180) - beta

x_h = L2 + L1*np.cos(q2)
y_h = L1*np.sin(q2)
out_theta = np.arctan(x_h/y_h)
alpha = np.deg2rad(90)- out_theta
q1 = -(np.deg2rad(90) - out_theta)



gamma = np.deg2rad(180) - alpha - beta
q3 = -(gamma)

Degree_of_head_tiit = np.deg2rad(-30)
q7 = Degree_of_head_tiit

print("range height : ",min," < h < ", max)
print("q1 :", np.rad2deg(q1))
print("q2 :", np.rad2deg(q2))
print("q3:", np.rad2deg(q3))

q_Leg1 = [q1, q2, q3, 0, 0, 0, q7, 0, 0]
q_Leg2 = [-q1, -q2, -q3, 0, 0, 0, q7, 0, 0]
q_Leg3 = [-q1, -q2, -q3, 0, 0, 0, q7, 0, 0]
FK_Leg1 = Leg1.fkine(q_Leg1)
FK_Leg2 = Leg1.fkine(q_Leg2)
FK_Leg3 = Leg1.fkine(q_Leg3)

print(FK_Leg1)
print(FK_Leg2)
print(FK_Leg3)

J_sol_Leg1 = Leg1.jacobe(q_Leg1)
J_sol_Leg2 = Leg1.jacobe(q_Leg2)
J_sol_Leg3 = Leg1.jacobe(q_Leg3)


q_dot_Leg1 = np.diff(q_Leg1)
q_dot_Leg2 = np.diff(q_Leg2)
q_dot_Leg3 = np.diff(q_Leg3)

# ปรับ q_dot_Leg1 ให้มีขนาดตรงกับ Jacobian (9 มิติ)
q_dot_Leg1 = np.append(q_dot_Leg1, 0)
q_dot_Leg2 = np.append(q_dot_Leg2, 0)
q_dot_Leg3 = np.append(q_dot_Leg3, 0)


# คำนวณความเร็วที่จุดปลาย
v_Leg1 = np.dot(J_sol_Leg1, q_dot_Leg1)
v_Leg2 = np.dot(J_sol_Leg2, q_dot_Leg2)
v_Leg3 = np.dot(J_sol_Leg3, q_dot_Leg3)

print("Velocity at the end-effector for Leg1:", v_Leg1)
print("Velocity at the end-effector for Leg2:", v_Leg2)
print("Velocity at the end-effector for Leg3:", v_Leg3)


# คำนวณตำแหน่งข้อต่อของแต่ละขา
def get_joint_positions(robot, q):
    T = robot.fkine_all(q)  # Forward kinematics for all joints
    positions = np.array([T[i].t for i in range(len(T))])  # Extract positions
    return positions

# สร้างตำแหน่งของข้อต่อสำหรับแต่ละขา
positions_leg1 = get_joint_positions(Leg1, q_Leg1)
positions_leg2 = get_joint_positions(Leg2, q_Leg2)
positions_leg3 = get_joint_positions(Leg3, q_Leg3)


q_start = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Starting position (all zeros)
q_end_Leg1 = q_Leg1
q_end_Leg2 = q_Leg2
q_end_Leg3 = q_Leg3

qt_Leg1 = rtb.tools.trajectory.jtraj(q_start, q_end_Leg1, 50)
qt_Leg2 = rtb.tools.trajectory.jtraj(q_start, q_end_Leg2, 50)
qt_Leg3 = rtb.tools.trajectory.jtraj(q_start, q_end_Leg3, 50)
Leg1.plot(qt_Leg1.q)
Leg2.plot(qt_Leg2.q)
Leg3.plot(qt_Leg3.q)

# Create a single 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Function to plot robot kinematics in 3D
def plot_robot_kinematics(robot, trajectory, color, label):
    # Compute joint positions for each step of the trajectory
    for q in trajectory.q:
        joint_positions = robot.fkine_all(q)
        positions = np.array([T.t for T in joint_positions])
        
        # Plot joint positions and connecting lines
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, alpha=0.3)
    
    # Plot the first and last positions more distinctly
    first_pos = robot.fkine_all(trajectory.q[0])
    last_pos = robot.fkine_all(trajectory.q[-1])
    
    first_positions = np.array([T.t for T in first_pos])
    last_positions = np.array([T.t for T in last_pos])
    
    ax.plot(first_positions[:, 0], first_positions[:, 1], first_positions[:, 2], 
            color=color, linestyle='--', label=f'{label} Start')
    ax.plot(last_positions[:, 0], last_positions[:, 1], last_positions[:, 2], 
            color=color, linestyle='-', label=f'{label} End')

# Plot kinematics for each leg
plot_robot_kinematics(Leg1, qt_Leg1, 'red', 'Leg1')
plot_robot_kinematics(Leg2, qt_Leg2, 'green', 'Leg2')
plot_robot_kinematics(Leg3, qt_Leg3, 'blue', 'Leg3')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Robot Leg Kinematics')
ax.legend()

#plt.tight_layout()
plt.show(block=True)  # เพิ่ม block=True เพื่อให้โปรแกรมรอจนกว่าจะปิดหน้าต่าง