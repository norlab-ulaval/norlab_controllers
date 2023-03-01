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
        self.initial_state_covariance = np.eye(3) * self.initial_state_stdev**2

        self.previous_input_array = np.zeros((self.horizon_length, 3))
        self.previous_input_array[:, 0] = self.maximum_linear_velocity/2

        self.wheel_radius = parameter_map['wheel_radius']
        self.baseline = parameter_map['baseline']
        self.a_param_init = parameter_map['a_param_init']
        self.b_param_init = parameter_map['b_param_init']
        self.param_variance_init = parameter_map['param_variance_init']
        self.variance_init = parameter_map['variance_init']
        self.kappa_param = parameter_map['kappa_param']

        self.distance_to_goal = 100000
        self.euclidean_distance_to_goal = 100000
        self.last_path_pose_id = 0
        self.orthogonal_projection_ids_horizon = np.zeros(self.horizon_length)
        self.orthogonal_projection_dists_horizon = np.zeros(self.horizon_length)

        self.full_body_blr_model = FullBodySlipBayesianLinearRegression(1, 1, 3, self.a_param_init, self.b_param_init,
                                                                        self.param_variance_init, self.variance_init,
                                                                        self.baseline, self.wheel_radius, 1/self.rate, self.kappa_param)
        self.trained_model_path = parameter_map['trained_model_path']
        self.full_body_blr_model.load_params(self.trained_model_path)

    def update_path(self, new_path):
        self.path = new_path
        return None



    def predict_horizon(self, init_state, input_array):
        prediction_means, prediction_covariances = self.full_body_blr_model.predict_horizon_from_body_idd_vels(input_array, init_state, self.initial_state_covariance)
        return prediction_means, prediction_covariances

    def compute_orthogonal_projections(self, prediction_means):
        orthogonal_projection_dists, orthogonal_projection_ids = self.path.pose_kdtree.query(prediction_means[:2, :].T, k=self.query_knn, distance_upper_bound=self.query_radius)
        for i in range(0, self.horizon_length):
            for j in range(0, orthogonal_projection_ids.shape[1]):
                if np.abs(orthogonal_projection_ids[i, j] - self.last_path_pose_id) <= self.id_window_size:
                    self.orthogonal_projection_ids_horizon[i] = orthogonal_projection_ids[i, j]
                    self.orthogonal_projection_dists_horizon[i] = orthogonal_projection_dists[i, j]
                    break
        # self.orthogonal_projection_dists, self.orthogonal_projection_ids = self.path.compute_orthogonal_projection(
        #     state[:2], self.last_path_pose_id, self.query_knn, self.query_radius)
        # for i in range(0, self.orthogonal_projection_ids.shape[0]):
        #     if np.abs(self.orthogonal_projection_ids[i] - self.last_path_pose_id) <= self.id_window_size:
        #         self.orthogonal_projection_id = self.orthogonal_projection_ids[i]
        #         self.orthogonal_projection_dist = self.orthogonal_projection_dists[i]
        #         break

    def compute_prediction_errors(self, prediction_means):

    def compute_distance_to_goal(self, state, orthogonal_projection_id):
        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        self.distance_to_goal = self.path.distances_to_goal[orthogonal_projection_id]
        return None

    def compute_command_vector(self, state):
        prediction_means, prediction_covariances = self.predict_horizon(state, self.previous_input_array)
        self.compute_orthogonal_projections(prediction_means)

        # SMPC pipeline
        # compute prediction mean and uncertainty
        # compute prediction orthogonal prediction
        # compute prediction cost
        # define optimization problem

        # return np.array([command_linear_velocity, command_angular_velocity])

        # TODO: Currently set up as a proportional controller for angular velocity control, investigate the need to use derivative component