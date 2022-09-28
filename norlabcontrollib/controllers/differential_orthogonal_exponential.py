from norlabcontrollib.controllers.controller import Controller
import numpy as np

class DifferentialOrthogonalExponential(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.gain_path_convergence = parameter_map['gain_path_convergence']
        self.gain_proportional_angular = parameter_map['gain_proportional_angular']
        self.gain_distance_to_goal_linear = parameter_map['gain_distance_to_goal_linear']
        self.gain_path_curvature_linear = parameter_map['gain_path_curvature_linear']
        self.path_look_ahead_distance = parameter_map['path_look_ahead_distance']

    def update_path(self, new_path):
        self.path = new_path

    def compute_linear_velocity(self, orthogonal_projection_id):

        path_curvature = self.path.look_ahead_curvatures[orthogonal_projection_id]


        distance_to_goal = self.path.distances_to_goal[orthogonal_projection_id]
        command_linear_velocity = self.maximum_linear_velocity * np.exp(-self.gain_path_curvature_linear * path_curvature -
                                                                        -self.gain_distance_to_goal_linear / distance_to_goal)
        command_linear_velocity = np.clip(command_linear_velocity, self.minimum_linear_velocity, self.maximum_linear_velocity)
        return command_linear_velocity

    def compute_angular_velocity(self, state, orthogonal_projection_dist, orthogonal_projection_id):
        target_exponential_tangent_angle = np.arctan(-self.gain_path_convergence * orthogonal_projection_dist)

        error_angle = target_exponential_tangent_angle - state[2]

        command_angular_velocity = self.gain_proportional_angular * error_angle
        command_angular_velocity = np.clip(command_angular_velocity, -self.maximum_angular_velocity, self.maximum_angular_velocity)
        return command_angular_velocity
    #TODO: Possibly need for wrap2pi here

    def compute_command_vector(self, state):
        orthogonal_projection_dist, orthogonal_projection_id = self.path.compute_orthogonal_projection(state[:2])
        command_linear_velocity = self.compute_linear_velocity(orthogonal_projection_id)
        command_angular_velocity = self.compute_angular_velocity(state, orthogonal_projection_dist, orthogonal_projection_id)
        return np.array([command_linear_velocity, command_angular_velocity])

        # TODO: Currently set up as a proportional controller for angular velocity control, investigate the need to use derivative component