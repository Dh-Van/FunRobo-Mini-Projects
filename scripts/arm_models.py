import numpy as np
import utils as ut


class FiveDOFRobot():
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """
    
    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        # self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12
        self.l1 = 15.5 * 0.01
        self.l2 = 9.9 * 0.01
        self.l3 = 9.5 * 0.01
        self.l4 = 5.5 * 0.01
        self.l5 = 10.5 * 0.01
        
        # Joint angles (initialized to zero)
        self.theta = [0, 0, np.pi/2, -np.pi/3, 0]  # degrees
        
        # Joint limits (in radians)
        self.theta_limits = [
            [-2 * np.pi / 3, 2 * np.pi / 3], 
            [-np.pi/2, np.pi/2], 
            [-2 * np.pi / 3, 2 * np.pi / 3],
            [-5 * np.pi / 9, 5 * np.pi / 9], 
            [-np.pi / 2, np.pi / 2], 
        ]
        
        # End-effector object
        self.ee = ut.EndEffector()
        
        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = np.zeros((5, 4))
        self.T = np.zeros((self.num_dof, 4, 4));
    
        self.J = np.zeros((5, 3))
            
    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """

        # Want to use radians to control the simulation since all the functions are written to work with radian values
        if not radians:
            theta = np.radians(theta)

        # DH Table, derivation shown above
        self.DH = [
            [theta[0], self.l1, 0, -np.pi/2],
            [theta[1] - np.pi/2, 0, self.l2, np.pi],
            [theta[2], 0, self.l3, np.pi],
            [theta[3] + np.pi/2, 0, 0, np.pi/2],
            [theta[4], self.l4 + self.l5, 0, 0],
        ]

        # This vertically stacks the transformation matricies from the DH table
        # self.T represents the transformation required to go from joint i-1 to joint i
        self.T = np.stack(
            [
                ut.dh_to_matrix(self.DH[0]),
                ut.dh_to_matrix(self.DH[1]),
                ut.dh_to_matrix(self.DH[2]),
                ut.dh_to_matrix(self.DH[3]),
                ut.dh_to_matrix(self.DH[4]),
            ],
            axis=0,
        )

        self.theta = theta
        
        # Calculate robot points (positions of joints)
        self.calc_robot_points()

        return self.theta


    def check_limits(self, angles):
        for i, angle in enumerate(angles):
            if(angle < self.theta_limits[i][0] and angle > self.theta_limits[i][1]):
                return False
        return True

    def calc_inverse_kinematics(self, EE: ut.EndEffector, soln=0):
        """
        Calculate inverse kinematics to determine the joint angles based on end-effector position.
        
        Args:
            EE: EndEffector object containing desired position and orientation.
            soln: Optional parameter for multiple solutions (not implemented).
        """
        solutions = np.zeros((8, 5))
        R_0_6 = ut.euler_to_rotm((EE.rotx, EE.roty, EE.rotz))
        z_rot = R_0_6[:, 2]
        d5 = self.l4 + self.l5
        pos_j4 = np.array([EE.x, EE.y, EE.z]) - (d5 * z_rot)
        j4_x, j4_y, j4_z = pos_j4[0], pos_j4[1], pos_j4[2]

        R_w = np.sqrt(j4_x ** 2 + j4_y ** 2)
        N = np.sqrt((j4_z - self.l1) ** 2 + R_w ** 2)

        # Could add pi for another sol
        solutions[0:3, 0] = ut.wraptopi(np.arctan2(j4_y, j4_x))
        solutions[4:7, 0] = ut.wraptopi(np.pi + ut.wraptopi(np.arctan2(j4_y, j4_x)))
        
        law_cos = (N ** 2 + self.l2 ** 2 - self.l3 ** 2) / (2 * self.l2 * N)
        mu = np.arccos(np.clip(law_cos, -1, 1))
        lmbda = np.arctan2((j4_z - self.l1), R_w)

        # Could subtract mu for another sol
        solutions[[0, 2, 4, 6], 1]  = ut.wraptopi((np.pi / 2) - lmbda) + mu
        solutions[[1, 3, 5, 7], 1]  = ut.wraptopi((np.pi / 2) - lmbda) - mu
        
        law_cos = (N ** 2 - self.l2 ** 2 - self.l3 ** 2) / (2 * self.l2 * self.l3)
        solutions[[0, 1, 4, 5], 2] = ut.wraptopi(np.arccos(np.clip(law_cos, -1, 1)))
        solutions[[2, 3, 6, 7], 2] = -ut.wraptopi(np.arccos(np.clip(law_cos, -1, 1)))

        for i, solution in enumerate(solutions):
            DH = [
                [solution[0], self.l1, 0, -np.pi/2],
                [solution[1], 0, self.l2, np.pi],
                [solution[2], 0, self.l3, np.pi]
            ]
            
            T_0_1 = ut.dh_to_matrix(DH[0])
            T_1_2 = ut.dh_to_matrix(DH[1])
            T_2_3 = ut.dh_to_matrix(DH[2])
            
            T_0_3 = T_0_1 @ T_1_2 @ T_2_3
            R_0_3 = T_0_3[:3, :3]
            
            R_3_6 = np.transpose(R_0_3) @ R_0_6
            
            solutions[i, 3] = ut.wraptopi((np.pi / 2) + np.arctan2(R_3_6[1, 2], R_3_6[0, 2]))
            solutions[i, 4] = np.arctan2(R_3_6[2, 0], R_3_6[2, 1])

        valid_solutions = []

        for i, solution in enumerate(solutions):
            if not self.check_limits(solution):
                continue
                
            target_pos = [EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz]
            self.calc_forward_kinematics(solution, radians=True)
            achieved_pos = [self.ee.x, self.ee.y, self.ee.z, self.ee.rotx, self.ee.roty, self.ee.rotz]
            
            error = np.linalg.norm(np.array(target_pos) - np.array(achieved_pos))
            valid_solutions.append((error, i, solution))

        if not valid_solutions:
            raise ValueError("No valid solutions found within joint limits")
        
        valid_solutions.sort()
        best_error, best_index, best_solution = valid_solutions[0]

        return self.calc_forward_kinematics(best_solution, radians=True)
         
    def check_limits(self, theta_list):
        for i, theta in enumerate(theta_list):
            if(theta < self.theta_limits[i][0] or theta > self.theta_limits[i][1]):
                return False
        return True

    def calc_jacobian(self):
        self.DH = [
            [self.theta[0], self.l1, 0, -np.pi/2],
            [self.theta[1] - np.pi/2, 0, self.l2, np.pi],
            [self.theta[2], 0, self.l3, np.pi],
            [self.theta[3] + np.pi/2, 0, 0, np.pi/2],
            [self.theta[4], self.l4 + self.l5, 0, 0],
        ]

        # This vertically stacks the transformation matricies from the DH table
        # self.T represents the transformation required to go from joint i-1 to joint i
        self.T = np.stack(
            [
                ut.dh_to_matrix(self.DH[0]),
                ut.dh_to_matrix(self.DH[1]),
                ut.dh_to_matrix(self.DH[2]),
                ut.dh_to_matrix(self.DH[3]),
                ut.dh_to_matrix(self.DH[4]),
            ],
            axis=0,
        )

        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 5):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            self.J[i] = np.cross(z[:3].flatten(), r.flatten())
            

    def calc_numerical_ik(self, EE: ut.EndEffector, tol=0.01, ilimit=50):
        """ Calculate numerical inverse kinematics based on input coordinates. """
        
        # Generates random theta valyes between each limit
        for i, limit in enumerate(self.theta_limits):
            min_value = limit[0]
            max_value = limit[1]
            random_value = np.random.uniform(min_value, max_value)
            self.theta[i] = random_value

        for i in range(0, ilimit):
            self.calc_forward_kinematics(self.theta, radians=True)
            error = [EE.x - self.ee.x, EE.y - self.ee.y, EE.z - self.ee.z]
            if(np.linalg.norm(error) <= tol):
                break
            self.calc_jacobian()
            J_inv = np.linalg.pinv(self.J)
            self.theta += error @ J_inv
                

        self.calc_forward_kinematics(self.theta, radians=True)

    
    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.
        
        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        
        # print(vel)
        # Computes the total transformation matricies to get from frame 0 to frame i
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 5):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            self.J[i] = np.cross(z[:3].flatten(), r.flatten())
            
        # Uses psuedoinverse to calculate inverse of jacobian
        # This is done since the jacobian is not square
        J_inv = np.linalg.pinv(self.J)
        # Multiplies the velocity vector by the inverse jacobian to get angular velocities of each joint
        theta_dot = np.dot(np.array(vel), J_inv)
        
        # Control cycle time step
        dt = 0.5
        # Calculates next theta values by multiplying angular velocities by time step
        self.theta = self.theta + (theta_dot * dt) * 0.25
        # Calls forward kinematics with new theta values
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_robot_points(self):
        """ Calculates the main arm points using the current joint angles """
        self.calc_jacobian()
        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        self.EE_axes = np.array([self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])

        print(self.ee.x, self.ee.y, self.ee.z, self.ee.rotx, self.ee.roty, self.ee.rotz)