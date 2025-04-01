# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
import numpy as np
from board_controller import BoardController
from servo_bus_controller import ServoBusController
import utils as ut
from arm_models import FiveDOFRobot

# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters

L1 = 15.5 * 0.01
L2 = 9.9 * 0.01
L3 = 9.5 * 0.01
L4 = 5.5 * 0.01
L5 = 10.5 * 0.01

class HiwonderRobot:
    def __init__(self):
        """Initialize motor controllers, servo bus, and default robot states."""
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        self.joint_values = [0, 0, 90, -30, 0, 0]  # degrees
        self.home_position = [0, 0, 90, -30, 0, 0]  # degrees
        self.joint_limits = [
            [-120, 120], [-90, 90], [-120, 120],
            [-100, 100], [-90, 90], [-120, 30]
        ]
        self.joint_control_delay = 0.2 # secs
        self.speed_control_delay = 0.2

        self.robot = FiveDOFRobot()

        self.move_to_home_position()

    # -------------------------------------------------------------
    # Methods for interfacing with the mobile base
    # -------------------------------------------------------------

    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands.

        Args:
            cmd (GamepadCmds): Command data class with velocities and joint commands.
        """

        if cmd.arm_home:
            self.move_to_home_position()

        print(f'---------------------------------------------------------------------')
        
        # self.set_base_velocity(cmd)
        self.set_arm_velocity(cmd)
        if(cmd.P1):
            pos = [-0.22673811495968096, -0.03824394198806677, 0.2683866227346938, -2.974495318957604, -1.4906135179820275, 3.141592653589793]
            self.go_to_pos(pos)


        ######################################################################

        position = [0]*3
        
        ######################################################################

        print(f'[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n')

    def go_to_pos(self, pos):
        EE = ut.EndEffector()
        EE.x = pos[0]
        EE.y = pos[1]
        EE.z = pos[2]
        EE.rotx = pos[3]
        EE.roty = pos[4]
        EE.rotz = pos[5]
        theta = self.robot.calc_inverse_kinematics(EE)

        theta = np.append(theta, 0)
        # print(theta)

        self.set_joint_values(theta, radians=True)

    def set_base_velocity(self, cmd: ut.GamepadCmds):
        """ Computes wheel speeds based on joystick input and sends them to the board """
        """
        motor3 w0|  ↑  |w1 motor1
                 |     |
        motor4 w2|     |w3 motor2
        
        """
        ######################################################################
        # insert your code for finding "speed"

        speed = [0]*4
        
        ######################################################################

        # Send speeds to motors
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Methods for interfacing with the 5-DOF robotic arm
    # -------------------------------------------------------------

    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """Calculates and sets new joint angles from linear velocities.

        Args:
            cmd (GamepadCmds): Contains linear velocities for the arm.
        """
        J = np.zeros((5, 3))

        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]

        # Sets initial theta values to be the current joint angles
        # This is needed to accurately calculate next theta values
        theta = self.joint_values[:5]

        # DH Table derived above
        DH = [
            [theta[0], L1, 0, -90],
            [theta[1] - 90, 0, L2, 180],
            [theta[2], 0, L3, 180],
            [theta[3] + 90, 0, 0, 90],
            [theta[4], L4 + L5, 0, 0],
        ]

        # This vertically stacks the transformation matricies from the DH table
        # T represents the transformation required to go from joint i-1 to joint i
        T = np.stack(
            [
                ut.dh_to_matrix(DH[0], radians=False),
                ut.dh_to_matrix(DH[1], radians=False),
                ut.dh_to_matrix(DH[2], radians=False),
                ut.dh_to_matrix(DH[3], radians=False),
                ut.dh_to_matrix(DH[4], radians=False),
            ],
            axis=0,
        )

        # Computes the total transformation matricies to get from frame 0 to frame i
        T_cumulative = [np.eye(4)]
        for i in range(5):
            T_cumulative.append(T_cumulative[-1] @ T[i])

        # Extracts the translational component to get from frame 0 to the end-effector frame
        d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        for i in range(0, 5):
            T_i = T_cumulative[i]
            z = T_i @ np.vstack([0, 0, 1, 0])
            d1 = T_i @ np.vstack([0, 0, 0, 1])
            r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
            J[i] = np.cross(z[:3].flatten(), r.flatten())
            
        # Uses psuedoinverse to calculate inverse of jacobian
        # This is done since the jacobian is not square
        J_inv = np.linalg.pinv(J)
        # Multiplies the velocity vector by the inverse jacobian to get angular velocities of each joine
        thetalist_dot = np.dot(np.array(vel), J_inv)
        # Add an extra entry at the end of the array to account for the end-effector opening and closing
        thetalist_dot = np.append(thetalist_dot, 0.0)

        # Update joint angles
        dt = 0.5 # Fixed time step
        K_vel = 0.25 # mapping gain for velocity control
        new_thetalist = [0.0]*6

        # linear velocity control
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * K_vel * thetalist_dot[i]     
        
        # set new joint angles
        self.set_joint_values(new_thetalist, radians=False)

        self.robot.calc_forward_kinematics(new_thetalist[0:5], radians=False)


    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """ Moves a single joint to a specified angle """
        if not (1 <= joint_id <= 6):
            raise ValueError("Joint ID must be between 1 and 6.")

        if radians:
            theta = np.rad2deg(theta)

        theta = self.enforce_joint_limits(theta, joint_id=joint_id)
        self.joint_values[joint_id] = theta

        pulse = self.angle_to_pulse(theta)
        self.servo_bus.move_servo(joint_id, pulse, duration)
        
        print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
        time.sleep(self.joint_control_delay)


    def set_joint_values(self, thetalist: list, duration=250, radians=False):
        """Moves all arm joints to the given angles.

        Args:
            thetalist (list): Target joint angles in degrees.
            duration (int): Movement duration in milliseconds.
        """
        if len(thetalist) != 6:
            raise ValueError("Provide 6 joint angles.")

        if radians:
            thetalist = [np.degrees(theta) for theta in thetalist]
            # print("converting")

        print(f"theta list {thetalist}" )
        thetalist = self.enforce_joint_limits(thetalist)
        self.joint_values = thetalist # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(thetalist) # remap the joint values from software to hardware

        for joint_id, theta in enumerate(thetalist, start=1):
            pulse = self.angle_to_pulse(theta)
            self.servo_bus.move_servo(joint_id, pulse, duration)


    def enforce_joint_limits(self, thetalist: list) -> list:
        """Clamps joint angles within their hardware limits.

        Args:
            thetalist (list): List of target angles.

        Returns:
            list: Joint angles within allowable ranges.
        """
        return [np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)]


    def move_to_home_position(self):
        print(f'Moving to home position...')
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f'Arrived at home position: {self.joint_values} \n')
        time.sleep(1.0)
        print(f'------------------- System is now ready!------------------- \n')


    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def angle_to_pulse(self, x: float):
        """ Converts degrees to servo pulse value """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return int((x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min)


    def pulse_to_angle(self, x: float):
        """ Converts servo pulse value to degrees """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return round((x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2)


    def stop_motors(self):
        """ Stops all motors safely """
        self.board.set_motor_speed([0]*4)
        print("[INFO] Motors stopped.")


    def remap_joints(self, thetalist: list):
        """Reorders angles to match hardware configuration.

        Args:
            thetalist (list): Software joint order.

        Returns:
            list: Hardware-mapped joint angles.

        Note: Joint mapping for hardware
            HARDWARE - SOFTWARE
            joint[0] = gripper/EE
            joint[1] = joint[5] 
            joint[2] = joint[4] 
            joint[3] = joint[3] 
            joint[4] = joint[2] 
            joint[5] = joint[1] 
        """
        return thetalist[::-1]