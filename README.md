# Fundamentals of Robotics: Mini Project 1

## Robot Frame Figure



## DH Table

$$
\left[ \begin{array}{cccc}
\theta_1  & L_1  & 0 & -90\\
\theta_2 -90 & 0 & L_2  & 180\\
\theta_3  & 0 & L_3  & 180\\
\theta_4 +90 & 0 & 0 & 90\\
\theta_5  & L_4 +L_5  & 0 & 0
\end{array} \right]
$$

## Forward Kinematics Equations

The forward kinematics of the 5-DOF robot arm is computed using the homogeneous transformation matrices derived from the DH parameters. The transformation from one frame to the next is given by:

$$
T_i^{i+1} = 
\begin{bmatrix} 
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\ 
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\ 
0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$$

The full transformation from the base to the end-effector is obtained by multiplying the individual transformations:

$$
T_0^5 = T_0^1 T_1^2 T_2^3 T_3^4 T_4^5
$$

where each  $T_i^{i+1}$ is derived using the corresponding DH parameters.

Using matrix multiplication, the final position and orientation of the end-effector relative to the base frame can be computed from $T_0^5$:

$$
T_0^5 =
\begin{bmatrix} 
R & d \\ 
0 & 1 
\end{bmatrix}
$$

where $R$ represents the rotation matrix and $P$ represents the position vector of the end-effector.

## Jacobian Matrix Derivation

To determine the linear velocity Jacobian $J_v$ for the 5-DOF robot arm, we use the standard Jacobian formulation:

$$
J_v = \begin{bmatrix} \frac{\partial x}{\partial \theta_1} & \frac{\partial x}{\partial \theta_2} & \dots & \frac{\partial x}{\partial \theta_5} \\ 
                      \frac{\partial y}{\partial \theta_1} & \frac{\partial y}{\partial \theta_2} & \dots & \frac{\partial y}{\partial \theta_5} \\ 
                      \frac{\partial z}{\partial \theta_1} & \frac{\partial z}{\partial \theta_2} & \dots & \frac{\partial z}{\partial \theta_5} 
       \end{bmatrix}
$$

where each column corresponds to a joint and describes how the end-effector’s position $(x, y, z)$ changes with respect to the joint angles.

### Derivation

For the joints, the linear velocity contribution is given by:

$$
J_{v_i} = Z_{0}^{i} \times (P_5 - P_i)
$$

where:
- $Z_{0}^{i}$ is the axis of rotation, obtained from the transformation matrix $T_0^i$ by multiplying with:

  $$
  Z_{0}^{i} = T_0^i \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}
  $$

- $P_5$ is the end-effector position, extracted from $T_0^5$.  
- $P_i$ is the position of joint $i$, extracted from $T_0^i$.  
- $J_v$ is built by crossing $Z_{0}^{i}$ with $P_5 - P_i$ for each joint.

The final Jacobian matrix for the linear velocity of the end-effector is:

$$
J_v =
\begin{bmatrix} 
Z_0^{1} \times (P_5 - P_1) \\ Z_0^{2} \times (P_5 - P_2) & \\ \dots & \\ Z_0^{5} \times (P_5 - P_5)
\end{bmatrix}
$$

where each column represents the contribution of a joint to the linear velocity.


## Code

## Simulation Verification

### Forward Position Kinematics

### Forward Velocity Kinematics

#### Seperate Motion

#### Combined Motion

## Gamepad Control

### Forward Velocity Kinematics

