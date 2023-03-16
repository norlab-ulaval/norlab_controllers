from norlabcontrollib.controllers.controller import Controller
from norlabcontrollib.models.blr_slip import FullBodySlipBayesianLinearRegression
from norlabcontrollib.models.ideal_diff_drive import Ideal_diff_drive

import numpy as np
from scipy.optimize import minimize
import casadi as cas

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
        self.input_cost_wheel = parameter_map['input_cost_wheel']
        self.input_cost_matrix = np.eye(2) * self.input_cost_wheel
        self.ancillary_gain_linear = parameter_map['ancillary_gain_linear']
        self.ancillary_gain_angular = parameter_map['ancillary_gain_angular']
        self.ancillary_gain_array = np.array([self.ancillary_gain_linear, self.ancillary_gain_angular]).reshape(2, 1)
        self.constraint_tolerance = parameter_map['constraint_tolerance']
        self.prob_safety_level = parameter_map['prob_safety_level']
        self.initial_state_stdev = parameter_map['initial_state_stdev']
        self.initial_state_covariance = np.eye(3) * self.initial_state_stdev**2

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
        self.orthogonal_projection_ids_horizon = np.zeros(self.horizon_length).astype('int32')
        self.orthogonal_projection_dists_horizon = np.zeros(self.horizon_length)
        self.prediction_input_covariances = np.zeros((2, 2, self.horizon_length))

        self.ideal_diff_drive = Ideal_diff_drive(self.wheel_radius, self.baseline, 1/self.rate)
        # self.full_body_blr_model = FullBodySlipBayesianLinearRegression(1, 1, 3, self.a_param_init, self.b_param_init,
        #                                                                 self.param_variance_init, self.variance_init,
        #                                                                 self.baseline, self.wheel_radius, 1/self.rate, self.kappa_param)
        # self.trained_model_path = parameter_map['trained_model_path']
        # self.full_body_blr_model.load_params(self.trained_model_path)

        previous_body_vel_input_array = np.zeros((2, self.horizon_length))
        previous_body_vel_input_array[0, :] = self.maximum_linear_velocity / 2
        # self.previous_input_array = self.ideal_diff_drive.compute_wheel_vels(previous_body_vel_input_array)
        self.previous_input_array = np.zeros((2, self.horizon_length))
        self.nd_input_array = np.zeros((2, self.horizon_length))
        self.target_trajectory = np.zeros((3, self.horizon_length))

        ############################ CASADI optimal control test
        self.R = cas.SX.eye(3)
        self.J = cas.SX(3, 2)
        self.J[0, :] = self.wheel_radius / 2
        self.J[1, :] = 0
        self.J[2, 0] = -self.wheel_radius / self.baseline
        self.J[2, 1] = self.wheel_radius / self.baseline

        self.casadi_x = cas.SX.sym('x', 3)
        self.casadi_u = cas.SX.sym('u', 2)

        self.R[0, 0] = cas.cos(self.casadi_x[2])
        self.R[1, 1] = cas.cos(self.casadi_x[2])
        self.R[0, 1] = -cas.sin((self.casadi_x[2]))
        self.R[1, 0] = cas.sin((self.casadi_x[2]))
        self.R_inv = self.R.T

        x_k = self.casadi_x + cas.mtimes(self.R_inv, cas.mtimes(self.J, self.casadi_u)) / self.rate
        self.single_step_pred = cas.Function('single_step_pred', [self.casadi_x, self.casadi_u], [x_k])

        self.x_0 = cas.SX.sym('x_0', 3, 1)
        self.x_horizon_list = [self.x_0]
        self.u_horizon_flat = cas.SX.sym('u_horizon_flat', 2 * self.horizon_length)
        self.u_horizon = cas.SX(self.horizon_length, 2)
        self.u_horizon[:, 0] = self.u_horizon_flat[:self.horizon_length]
        self.u_horizon[:, 1] = self.u_horizon_flat[self.horizon_length:]

        # self.x_ref = cas.DM.zeros(3, self.horizon_length)
        self.x_ref_flat = cas.SX.sym('x_ref', 3 * self.horizon_length)
        self.x_ref = cas.SX.zeros(3, self.horizon_length)
        self.x_ref[0, :] = self.x_ref_flat[:self.horizon_length]
        self.x_ref[1, :] = self.x_ref_flat[self.horizon_length:2 * self.horizon_length]
        self.x_ref[2, :] = self.x_ref_flat[2 * self.horizon_length:3 * self.horizon_length]
        # self.x_ref[:, :] = self.target_trajectory
        self.cas_state_cost_matrix = cas.DM.zeros(3, 3)
        self.cas_state_cost_matrix[:, :] = self.state_cost_matrix
        self.u_ref = cas.DM.zeros(2, self.horizon_length)
        self.u_ref[:, :] = self.previous_input_array
        self.u_ref = self.u_ref.T
        self.cas_input_cost_matrix = cas.DM.zeros(2, 2)
        self.cas_input_cost_matrix[:, :] = self.input_cost_matrix
        self.prediction_cost = cas.SX(0)

        for i in range(1, self.horizon_length):
            self.x_horizon_list.append(self.single_step_pred(self.x_horizon_list[i - 1], self.u_horizon[i - 1, :]))
            x_error = self.x_ref[i] - self.x_horizon_list[i]
            state_cost = cas.mtimes(cas.mtimes(x_error.T, self.cas_state_cost_matrix), x_error)
            u_error = self.u_ref[i - 1, :] - self.u_horizon[i - 1, :]
            input_cost = cas.mtimes(cas.mtimes(u_error, self.cas_input_cost_matrix), u_error.T)
            self.prediction_cost = self.prediction_cost + state_cost + input_cost

        self.x_horizon = cas.hcat(self.x_horizon_list)
        self.horizon_pred = cas.Function('horizon_pred', [self.x_0, self.u_horizon_flat], [self.x_horizon])
        self.pred_cost = cas.Function('pred_cost', [self.u_horizon_flat], [self.prediction_cost])

        self.nlp_params = cas.vertcat(self.x_0, self.x_ref_flat)

        self.optim_problem = {"f": self.prediction_cost, "x": self.u_horizon_flat, "p": self.nlp_params}
        self.nlpsol_opts = {'verbose_init': False, 'print_in': False, 'print_out': False, 'print_time': False, 'verbose': False, 'ipopt':{'print_level': 0}}
        self.optim_problem_solver = cas.nlpsol("optim_problem_solver", "ipopt", self.optim_problem, self.nlpsol_opts)

    def update_path(self, new_path):
        self.path = new_path
        return None

    def input_to_body_vel(self, input_array):
        return self.full_body_blr_model.compute_body_vel(input_array)

    def predict_horizon(self, init_state, input_array):
        body_vels_horizon = self.input_to_body_vel(input_array).T
        prediction_means, prediction_covariances = self.full_body_blr_model.predict_horizon_from_body_idd_vels(body_vels_horizon, init_state, self.initial_state_covariance)
        # for i in range(0, self.horizon_length):
        #     self.prediction_input_covariances[:, :, i] = self.ancillary_gain_array @ prediction_covariances[:, :, i] @ self.ancillary_gain_array.T
        return prediction_means, prediction_covariances

    def compute_orthogonal_projection(self, state):
        self.orthogonal_projection_dists, self.orthogonal_projection_ids = self.path.compute_orthogonal_projection(state[:2], self.last_path_pose_id, self.query_knn, self.query_radius)
        for i in range(0, self.orthogonal_projection_ids.shape[0]):
            if np.abs(self.orthogonal_projection_ids[i] - self.last_path_pose_id) <= self.id_window_size:
                self.orthogonal_projection_id = self.orthogonal_projection_ids[i]
                self.orthogonal_projection_dist = self.orthogonal_projection_dists[i]
                self.last_path_pose_id = self.orthogonal_projection_id
                break
        return None

    def compute_desired_trajectory(self, state):
        self.compute_orthogonal_projection(state)
        target_displacement = 0
        target_displacement_rate = self.maximum_linear_velocity / self.rate
        target_trajectory_path_id = self.orthogonal_projection_id
        for i in range(0, self.horizon_length):
            if target_trajectory_path_id + 1 == self.path.poses.shape[0]:
                for j in range(i, self.horizon_length):
                    self.target_trajectory[:2, j] = self.path.poses[target_trajectory_path_id][:2]
                    self.target_trajectory[2, j] = self.path.angles[target_trajectory_path_id]
                break
            target_displacement += target_displacement_rate
            path_displacement = self.path.distances_to_goal[target_trajectory_path_id] - self.path.distances_to_goal[target_trajectory_path_id + 1]
            if target_displacement >= path_displacement / 2:
                target_trajectory_path_id += 1
                target_displacement = -path_displacement / 2
            self.target_trajectory[:2, i] = self.path.poses[target_trajectory_path_id][:2]
            self.target_trajectory[2, i] = self.path.angles[target_trajectory_path_id]
        return None
    def compute_orthogonal_projections(self, prediction_means):
        orthogonal_projection_dists, orthogonal_projection_ids = self.path.pose_kdtree.query(prediction_means[:2, :].T, k=self.query_knn, distance_upper_bound=self.query_radius)
        for i in range(0, self.horizon_length):
            for j in range(0, orthogonal_projection_ids.shape[1]):
                if np.abs(orthogonal_projection_ids[i, j] - self.last_path_pose_id) <= self.id_window_size:
                    self.orthogonal_projection_ids_horizon[i] = int(orthogonal_projection_ids[i, j])
                    self.orthogonal_projection_dists_horizon[i] = orthogonal_projection_dists[i, j]
                    break
        # self.orthogonal_projection_dists, self.orthogonal_projection_ids = self.path.compute_orthogonal_projection(
        #     state[:2], self.last_path_pose_id, self.query_knn, self.query_radius)
        # for i in range(0, self.orthogonal_projection_ids.shape[0]):
        #     if np.abs(self.orthogonal_projection_ids[i] - self.last_path_pose_id) <= self.id_window_size:
        #         self.orthogonal_projection_id = self.orthogonal_projection_ids[i]
        #         self.orthogonal_projection_dist = self.orthogonal_projection_dists[i]
        #         break


    def compute_distance_to_goal(self, state, orthogonal_projection_id):
        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        self.distance_to_goal = self.path.distances_to_goal[orthogonal_projection_id]
        return None

    def compute_horizon_cost(self, prediction_means, prediction_covariances, input_array):
        prediction_cost = 0
        for i in range(0, self.horizon_length):
            state_error = prediction_means[:, i] - self.path.planar_poses[self.orthogonal_projection_ids_horizon[i]]
            state_cost_timestep = state_error @ self.state_cost_matrix @ state_error.T + np.trace(self.state_cost_matrix @ prediction_covariances[:, :, i])

            input_error = input_array[:, i] - self.previous_input_array[:, i]
            input_cost_timestep = input_error @ self.input_cost_matrix @ input_error.T
            # TODO: Possibly implement ancillary input coriances to add to cost
            prediction_cost += state_cost_timestep + input_cost_timestep
        return prediction_cost

    def predict_then_compute_cost(self, input):
        # body_vel_array = self.full_body_blr_model.compute_body_vel(input)
        self.prediction_means, self.prediction_covariances = self.predict_horizon(self.init_state, input)
        self.compute_orthogonal_projections(self.prediction_means)
        return self.compute_horizon_cost(self.prediction_means, self.prediction_covariances, input)

    # def compute_prediction_lagrangian(self):
    # def compution_prediction_gradient(self, prediction_cost):
    def compute_objective(self, input):
        # body_vel_array = self.full_body_blr_model.compute_body_vel(input)
        self.nd_input_array[0, :] = input[:self.horizon_length]
        self.nd_input_array[1, :] = input[self.horizon_length:]
        prediction_means, prediction_covariances = self.predict_horizon(self.init_state, self.nd_input_array)
        self.compute_orthogonal_projections(prediction_means)
        cost = self.compute_horizon_cost(prediction_means, prediction_covariances, self.nd_input_array)
        return cost

    # def compute_objective_gradient(self, input):


    def compute_command_vector(self, state):
        planar_state = np.array([state[0], state[1], state[5]])
        self.compute_desired_trajectory(state)
        nlp_params = np.concatenate((planar_state, self.target_trajectory.flatten('C')))
        optim_control_solution = self.optim_problem_solver(x0=self.previous_input_array.flatten(), p=nlp_params)['x']
        self.previous_input_array[0, :] = np.array(optim_control_solution[:self.horizon_length]).reshape(self.horizon_length)
        self.previous_input_array[1, :] = np.array(optim_control_solution[:self.horizon_length]).reshape(self.horizon_length)
        wheel_input_array = np.array([optim_control_solution[0], optim_control_solution[self.horizon_length]]).reshape(2,1)
        body_input_array = self.ideal_diff_drive.compute_body_vel(wheel_input_array).astype('float64')
        return body_input_array.reshape(2)

        # fun = lambda x: self.compute_objective(x)
        # optimization_result = minimize(fun, self.previous_input_array.flatten(), method='SLSQP')
        # print(optimization_result.x)
        # return optimization_result.x
