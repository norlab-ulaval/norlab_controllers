import numpy as np
from ..util.transform_algebra import *

class IdealArticulatedDrive:
    def __init__(self, r, L_front, L_back, dt):
        self.dt = dt
        self.r = r
        self.L_front = L_front
        self.L_back = L_back
        self.rotation_body_to_world = np.eye(2)
        self.body_vel_world_3d = np.zeros(6)
        self.body_vel_world_2d = np.zeros(3)
        self.state_2d = np.zeros(3)

    # TODO: Adapt to fit 2x2 jacobian matrix
    # def predict(self, init_state, input):
    #     """
    #     :param init_state: initial state array [x, y, z, roll, pitch, yaw]
    #     :param input: input array [omega_l, omega_r]
    #     :return: next_state
    #     """
    #     self.state_2d[:2] = init_state[:2]
    #     self.state_2d[2] = init_state[-1]
    #     yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
    #     body_vel = self.jacobian @ input
    #     self.body_vel_world_2d[:2] = self.rotation_body_to_world @ body_vel[:2]
    #     self.body_vel_world_2d[2] = body_vel[2]
    #     self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
    #     self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]
    #
    #     return init_state + self.body_vel_world_3d * self.dt

    def compute_body_vel(self, input):
        v= self.r*input[0]
        phi =input[1]

        omega = v*tan(phi)/self.L_front
        return np.array([v, omega])

    def compute_body_vel_horizon(self, horizon_input):

        body_vels = np.zeros((2, horizon_input.shape[1]))

        for i in range(horizon_input.shape[1]):
            body_vels[:, i] = self.compute_body_vel(horizon_input[:, i])

        return body_vels

    def predict(self, init_state, input):
        """
        :param init_state: initial state array [x, y, z, roll, pitch, yaw]
        :param input: input array [omega_l, omega_r]
        :return: next_state
        """
        self.state_2d[:2] = init_state[:2]
        self.state_2d[2] = init_state[-1]
        yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
        body_vel = self.compute_body_vel(input)
        self.body_vel_world_2d[:2] = self.rotation_body_to_world @ body_vel[:2]
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]

        # print(self.body_vel_world_3d)

        return init_state + self.body_vel_world_3d * self.dt

    def adjust_motion_params(self, params):
        return None