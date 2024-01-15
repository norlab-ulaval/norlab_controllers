from norlabcontrollib.controllers.controller import Controller
from norlabcontrollib.models.ideal_diff_drive import Ideal_diff_drive

import numpy as np
from scipy.optimize import minimize
import casadi as cas

class IdealDiffDriveMPC(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        # self.gain_distance_to_goal_linear = parameter_map['gain_distance_to_goal_linear']     # CLEANUP
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
        self.angular_velocity_gain = parameter_map['angular_velocity_gain']

        self.wheel_radius = parameter_map['wheel_radius']
        self.baseline = parameter_map['baseline']

        self.distance_to_goal = 100000
        self.euclidean_distance_to_goal = 100000
        self.last_path_pose_id = 0
        self.orthogonal_projection_ids_horizon = np.zeros(self.horizon_length).astype('int32')
        self.orthogonal_projection_dists_horizon = np.zeros(self.horizon_length)
        self.prediction_input_covariances = np.zeros((2, 2, self.horizon_length))

        self.ideal_diff_drive = Ideal_diff_drive(self.wheel_radius, self.baseline, 1/self.rate)

        previous_body_vel_input_array = np.zeros((2, self.horizon_length))
        previous_body_vel_input_array[0, :] = self.maximum_linear_velocity / 2
        self.max_wheel_vel = self.ideal_diff_drive.compute_wheel_vels(np.array([self.maximum_linear_velocity, 0]))[0]
        self.previous_input_array = np.zeros((2, self.horizon_length))
        self.nd_input_array = np.zeros((2, self.horizon_length))
        self.target_trajectory = np.zeros((3, self.horizon_length))
        self.optim_trajectory_array = np.zeros((3, self.horizon_length))
        self.straight_line_input = np.full(2*self.horizon_length, 1.0)
        self.straight_line_input[self.horizon_length:] = np.full(self.horizon_length, 2.0)

        self.init_casadi_model()

    def init_casadi_model(self):
        self.R = cas.SX.eye(3)
        self.J = cas.SX(3, 2)
        self.J[0, :] = self.wheel_radius / 2
        self.J[1, :] = 0
        self.J[2, 0] = -self.angular_velocity_gain * self.wheel_radius / self.baseline
        self.J[2, 1] = self.angular_velocity_gain * self.wheel_radius / self.baseline

        self.casadi_x = cas.SX.sym('x', 3)
        self.casadi_u = cas.SX.sym('u', 2)

        self.R[0, 0] = cas.cos(self.casadi_x[2])
        self.R[1, 1] = cas.cos(self.casadi_x[2])
        self.R[0, 1] = -cas.sin((self.casadi_x[2]))
        self.R[1, 0] = cas.sin((self.casadi_x[2]))
        self.R_inv = self.R.T

        self.x_k = self.casadi_x + cas.mtimes(self.R, cas.mtimes(self.J, self.casadi_u)) / self.rate
        self.single_step_pred = cas.Function('single_step_pred', [self.casadi_x, self.casadi_u], [self.x_k])

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
            x_error = self.x_ref[:, i] - self.x_horizon_list[i]
            state_cost = cas.mtimes(cas.mtimes(x_error.T, self.cas_state_cost_matrix), x_error)
            u_error = self.u_ref[i - 1, :] - self.u_horizon[i - 1, :]
            input_cost = cas.mtimes(cas.mtimes(u_error, self.cas_input_cost_matrix), u_error.T)
            self.prediction_cost = self.prediction_cost + state_cost + input_cost

        self.x_horizon = cas.hcat(self.x_horizon_list)
        self.horizon_pred = cas.Function('horizon_pred', [self.x_0, self.u_horizon_flat], [self.x_horizon])
        self.pred_cost = cas.Function('pred_cost', [self.u_horizon_flat], [self.prediction_cost])

        self.nlp_params = cas.vertcat(self.x_0, self.x_ref_flat)
        self.lower_bound_input = np.full(2 * self.horizon_length, -self.max_wheel_vel)
        self.upper_bound_input = np.full(2 * self.horizon_length, self.max_wheel_vel)
        self.optim_problem = {"f": self.prediction_cost, "x": self.u_horizon_flat, "p": self.nlp_params}
        self.nlpsol_opts = {'verbose_init': False, 'print_in': False, 'print_out': False, 'print_time': False,
                            'verbose': False, 'ipopt': {'print_level': 0}}
        self.optim_problem_solver = cas.nlpsol("optim_problem_solver", "ipopt", self.optim_problem, self.nlpsol_opts)

    def update_path(self, new_path):
        self.path = new_path
        self.distance_to_goal = 100000
        self.euclidean_distance_to_goal = 100000

    def compute_distance_to_goal(self, state, orthogonal_projection_id):
        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        self.distance_to_goal = self.path.distances_to_goal[orthogonal_projection_id]

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

    def compute_command_vector(self, state):
        self.planar_state = np.array([state[0], state[1], state[5]])
        self.compute_desired_trajectory(self.planar_state)
        nlp_params = np.concatenate((self.planar_state, self.target_trajectory.flatten('C')))
        self.optim_control_solution = self.optim_problem_solver(x0=self.previous_input_array.flatten(),
                                                           p=nlp_params,
                                                           lbx= self.lower_bound_input,
                                                           ubx= self.upper_bound_input)['x']
        self.previous_input_array[0, :] = np.array(self.optim_control_solution[:self.horizon_length]).reshape(self.horizon_length)
        self.previous_input_array[1, :] = np.array(self.optim_control_solution[self.horizon_length:]).reshape(self.horizon_length)
        self.optimal_left = self.optim_control_solution[0]
        self.optimal_right = self.optim_control_solution[self.horizon_length]
        wheel_input_array = np.array([self.optim_control_solution[0], self.optim_control_solution[self.horizon_length]]).reshape(2,1)
        body_input_array = self.ideal_diff_drive.compute_body_vel(wheel_input_array).astype('float64')

        optim_trajectory = self.horizon_pred(self.planar_state, self.optim_control_solution)
        self.optim_solution_array = np.array(self.optim_control_solution)
        self.optim_trajectory_array[0, :] = optim_trajectory[0, :]
        self.optim_trajectory_array[1, :] = optim_trajectory[1, :]
        self.optim_trajectory_array[2, :] = optim_trajectory[2, :]
        self.previous_input_array[0, :] = self.optim_solution_array[:self.horizon_length].reshape(self.horizon_length)
        self.previous_input_array[1, :] = self.optim_solution_array[self.horizon_length:].reshape(self.horizon_length)
        self.compute_distance_to_goal(state, self.orthogonal_projection_id)
        self.last_path_pose_id = self.orthogonal_projection_id
        return body_input_array.reshape(2)
