from norlabcontrollib.controllers.controller import Controller
from norlabcontrollib.models.blr_slip import FullBodySlipBayesianLinearRegression

import numpy as np
import casadi as cs

class StochasticLinearMPC(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.gain_distance_to_goal_linear = parameter_map['gain_distance_to_goal_linear']
        self.gain_path_curvature_linear = parameter_map['gain_path_curvature_linear']
        self.path_look_ahead_distance = parameter_map['path_look_ahead_distance']
        self.query_radius = parameter_map['query_radius']
        self.query_knn = parameter_map['query_knn']
        self.id_window_size = parameter_map['id_window_size']

        self.number_states = 3
        self.number_inputs = 2

        self.horizon_length = parameter_map['horizon_length']
        self.state_cost_translational = parameter_map['state_cost_translational']
        self.state_cost_rotational = parameter_map['state_cost_rotational']
        self.state_cost_matrix = np.eye(3)
        self.state_cost_matrix[0,0] = self.state_cost_translational
        self.state_cost_matrix[1,1] = self.state_cost_translational
        self.state_cost_matrix[2,2] = self.state_cost_rotational
        self.input_cost_linear = parameter_map['input_cost_linear']
        self.input_cost_angular = parameter_map['input_cost_angular']
        self.input_cost_matrix = np.eye(2)
        self.input_cost_matrix[0,0] = self.input_cost_linear
        self.input_cost_matrix[1,1] = self.input_cost_angular
        self.constraint_tolerance = parameter_map['constraint_tolerance']
        self.prob_safety_level = parameter_map['prob_safety_level']
        self.initial_state_stdev = parameter_map['initial_state_stdev']

        self.wheel_radius = parameter_map['wheel_radius']
        self.baseline = parameter_map['baseline']
        self.a_param_init = parameter_map['a_param_init']
        self.b_param_init = parameter_map['b_param_init']
        self.param_variance_init = parameter_map['param_variance_init']
        self.variance_init = parameter_map['variance_init']
        self.kappa_param = parameter_map['kappa_param']

        self.distance_to_goal = 100000
        self.euclidean_distance_to_goal = 100000

        self.full_body_blr_model = FullBodySlipBayesianLinearRegression(1, 1, 3, self.a_param_init, self.b_param_init,
                                                                        self.param_variance_init, self.variance_init,
                                                                        self.baseline, self.wheel_radius, 1/self.rate, self.kappa_param)
        self.trained_model_path = parameter_map['trained_model_path']
        self.full_body_blr_model.load_params(self.trained_model_path)

    def update_path(self, new_path):
        self.path = new_path
        return None

    # def precompute_probalistic_limits(self):


    def compute_distance_to_goal(self, state, orthogonal_projection_id):
        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        self.distance_to_goal = self.path.distances_to_goal[orthogonal_projection_id]
        return None

    def compute_linear_velocity(self, orthogonal_projection_id):

        self.path_curvature = self.path.look_ahead_curvatures[orthogonal_projection_id]
        self.look_ahead_distance = self.path.look_ahead_distance_counter_array[orthogonal_projection_id]

        # print("Distance to goal :" + str(self.distance_to_goal))
        # print("Path curvature :" + str(path_curvature))

        command_linear_velocity = self.maximum_linear_velocity * np.exp(-self.gain_path_curvature_linear * self.path_curvature -
                                                                        self.gain_distance_to_goal_linear / self.distance_to_goal)
        command_linear_velocity = np.clip(command_linear_velocity, self.minimum_linear_velocity, self.maximum_linear_velocity)

        if not self.path.going_forward:
            command_linear_velocity = -command_linear_velocity
        return command_linear_velocity

    def wrap2pi(self, angle):
        if angle <= np.pi and angle >= -np.pi:
            return (angle)
        elif angle < -np.pi:
            return (self.wrap2pi(angle + 2 * np.pi))
        else:
            return (self.wrap2pi(angle - 2 * np.pi))

    def apply_tf(self, state, tf):
        homo_state = np.ones(3)
        homo_state[0] = state[0]
        homo_state[1] = state[1]
        homo_state_transformed =  tf @ homo_state
        return homo_state_transformed[:2]
    def compute_angular_velocity(self, state, orthogonal_projection_id):
        if not self.path.going_forward:
            if state[5] >= 0:
                self.current_yaw = -np.pi + state[5]
            else:
                self.current_yaw = np.pi + state[5]
        else:
            self.current_yaw = state[5]
        self.projection_pose_path_frame = self.apply_tf(state[:2],
                                                        self.path.world_to_path_tfs_array[orthogonal_projection_id])

        self.target_exponential_tangent_angle = np.arctan(-self.gain_path_convergence * self.projection_pose_path_frame[1])

        self.robot_yaw_path_frame = self.wrap2pi(self.current_yaw - self.path.angles[orthogonal_projection_id])

        self.error_angle = self.target_exponential_tangent_angle - self.robot_yaw_path_frame

        # print("yaw :" + str(state[5]))
        # print("error_angle :" + str(error_angle))

        command_angular_velocity = self.gain_proportional_angular * self.error_angle
        command_angular_velocity = np.clip(command_angular_velocity, -self.maximum_angular_velocity, self.maximum_angular_velocity)
        return command_angular_velocity
    #TODO: Possibly need for wrap2pi here

    def compute_command_vector(self, state):
        # print("state :", state)
        self.orthogonal_projection_dist, self.orthogonal_projection_id = self.path.compute_orthogonal_projection(state[:2])
        self.compute_distance_to_goal(state, self.orthogonal_projection_id)
        command_linear_velocity = self.compute_linear_velocity(self.orthogonal_projection_id)
        command_angular_velocity = self.compute_angular_velocity(state, self.orthogonal_projection_id)
        return np.array([command_linear_velocity, command_angular_velocity])

        # TODO: Currently set up as a proportional controller for angular velocity control, investigate the need to use derivative component