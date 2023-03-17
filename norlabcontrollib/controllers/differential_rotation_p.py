from norlabcontrollib.controllers.controller import Controller
import numpy as np


class DifferentialRotationP(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.p_gain = parameter_map["p_gain"]
        self.current_path_pose_id = 0

    def compute_command_vector(self, state):
        error = self.path.poses[self.current_path_pose_id, 2] - state[5]
        if error < self.goal_tolerance and self.current_path_pose_id < len(self.path.poses) - 1:
            self.current_path_pose_id += 1
            error = self.path.poses[self.current_path_pose_id, 2] - state[5]
        return np.array([0, min(self.p_gain * error, self.maximum_angular_velocity)])

    def update_path(self, new_path):
        super().update_path(new_path)
        self.current_path_pose_id = 0
