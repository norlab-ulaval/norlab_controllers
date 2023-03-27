from norlabcontrollib.controllers.controller import Controller
import numpy as np
import math


class DifferentialRotationP(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.p_gain = parameter_map["p_gain"]
        self.angular_distance_to_goal = 100000

    def compute_angular_error(self, current_angle, goal_angle, going_forward):
        if not going_forward:
            if current_angle >= 0:
                current_angle -= math.pi
            else:
                current_angle += math.pi

        angular_error = goal_angle - current_angle
        if angular_error > math.pi:
            angular_error -= 2 * math.pi
        elif angular_error < -math.pi:
            angular_error += 2 * math.pi
        return angular_error

    def compute_command_vector(self, state):
        error = self.compute_angular_error(state[5], self.path.poses[0, 5], self.path.going_forward)
        self.angular_distance_to_goal = abs(error)
        return np.array([0, min(self.p_gain * error, self.maximum_angular_velocity)])

    def update_path(self, new_path):
        super().update_path(new_path)
        self.angular_distance_to_goal = 100000
