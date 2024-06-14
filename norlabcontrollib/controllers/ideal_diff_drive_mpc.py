from norlabcontrollib.controllers.controller import Controller
from norlabcontrollib.models.ideal_diff_drive import Ideal_diff_drive

import numpy as np
from scipy.optimize import minimize
import casadi as cas
import math

class IdealDiffDriveMPC(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.path_look_ahead_distance = parameter_map['path_look_ahead_distance']
        self.query_radius = parameter_map['query_radius']
        self.query_knn = parameter_map['query_knn']
        self.id_window_size = parameter_map['id_window_size']

        self.number_states = 3
        self.number_inputs = 2

        self.function_to_re_init = False
        self.param_that_start_init = ['maximum_linear_velocity','horizon_length','angular_velocity_gain']

        self.horizon_length = parameter_map['horizon_length']
        self.state_cost_translational = parameter_map['state_cost_translational']
        self.state_cost_rotational = parameter_map['state_cost_rotational']
        

        #self.state_cost_matrix = np.eye(3)
        #self.state_cost_matrix[0,0] = self.state_cost_translational
        #self.state_cost_matrix[1,1] = self.state_cost_translational
        #self.state_cost_matrix[2,2] = self.state_cost_rotational

        self.input_cost_wheel = parameter_map['input_cost_wheel']
        self.input_cost_matrix_i = np.eye(2) * self.input_cost_wheel
        
        
        self.angular_velocity_gain = parameter_map['angular_velocity_gain']

        self.wheel_radius = parameter_map['wheel_radius']
        self.baseline = parameter_map['baseline']

        self.distance_to_goal = 100000
        self.euclidean_distance_to_goal = 100000
        self.next_path_idx = 0
        
        self.init_casadi_model()

    def init_casadi_model(self):
        ### Init value moved to be able to reset casadi init 
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
        ########
        # Add the self. casadi angular_velocity_gain as a value 
        #self.cas_angular_velocity_gain = cas.SX.sym('angular_velocity_gain', 1)
        # 
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
        
        self.u_ref = cas.DM.zeros(2, self.horizon_length)
        self.u_ref[:, :] = self.previous_input_array
        self.u_ref = self.u_ref.T

        # Change self.cas_input_cost_matrix as SX
        self.cas_input_cost_param = cas.SX.sym('input_cost_param', 1)
        self.cas_input_cost_matrix = cas.SX.eye(2)
        self.cas_input_cost_matrix[0,0] = self.cas_input_cost_param
        self.cas_input_cost_matrix[1,1] = self.cas_input_cost_param

        # Change state cost matrix (the next commented line are for information)
        # self.state_cost_matrix = np.eye(3)
        # self.state_cost_matrix[0,0] = self.state_cost_translational
        # self.state_cost_matrix[1,1] = self.state_cost_translational
        # self.state_cost_matrix[2,2] = self.state_cost_rotational
        self.cas_state_cost_translational = cas.SX.sym('state_cost_translationnal', 1)
        self.cas_state_cost_rotationnal = cas.SX.sym('state_cost_rotationnal', 1)
        
        self.cas_state_cost_matrix = cas.SX.eye(3)
        self.cas_state_cost_matrix[0,0] = self.cas_state_cost_translational
        self.cas_state_cost_matrix[1,1] = self.cas_state_cost_translational
        self.cas_state_cost_matrix[2,2] = self.cas_state_cost_rotationnal
        
        #self.cas_input_cost_matrix = cas.DM.zeros(2, 2)
        #self.cas_input_cost_matrix[:, :] = self.input_cost_matrix
        self.prediction_cost = cas.SX(0)

        for i in range(1, self.horizon_length):
            self.x_horizon_list.append(self.single_step_pred(self.x_horizon_list[i - 1], self.u_horizon[i - 1, :]))
            x_error = self.x_ref[:, i] - self.x_horizon_list[i]
            x_error[2] = cas.atan2(cas.sin(x_error[2]), cas.cos(x_error[2]))
            state_cost = cas.mtimes(cas.mtimes(x_error.T, self.cas_state_cost_matrix), x_error)
            u_error = self.u_ref[i - 1, :] - self.u_horizon[i - 1, :]
            input_cost = cas.mtimes(cas.mtimes(u_error, self.cas_input_cost_matrix), u_error.T)
            self.prediction_cost = self.prediction_cost + state_cost + input_cost

        self.x_horizon = cas.hcat(self.x_horizon_list)
        self.horizon_pred = cas.Function('horizon_pred', [self.x_0, self.u_horizon_flat], [self.x_horizon])
        # self.pred_cost = cas.Function('pred_cost', [self.u_horizon_flat], [self.prediction_cost])

        
        self.nlp_params = cas.vertcat(self.x_0, self.x_ref_flat,self.cas_input_cost_param,
        self.cas_state_cost_translational,self.cas_state_cost_rotationnal) # self.cas_angular_velocity_gain
        self.lower_bound_input = np.full(2 * self.horizon_length, -self.max_wheel_vel)
        self.upper_bound_input = np.full(2 * self.horizon_length, self.max_wheel_vel)
        self.optim_problem = {"f": self.prediction_cost, "x": self.u_horizon_flat, "p": self.nlp_params}
        self.nlpsol_opts = {'verbose_init': False, 'print_in': False, 'print_out': False, 'print_time': False,
                            'verbose': False, 'ipopt': {'print_level': 0}}
        self.optim_problem_solver = cas.nlpsol("optim_problem_solver", "ipopt", self.optim_problem, self.nlpsol_opts)
        # Casadi has been re_init. 
        self.function_to_re_init = False

    def update_path(self, new_path):
        self.path = new_path
        return None
    
    def compute_distance_to_goal(self, state, orthogonal_projection_id):
        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        self.distance_to_goal = self.path.distances_to_goal[orthogonal_projection_id]

    def compute_desired_trajectory(self, state):
        # Find closest point on path
        closest_pose, self.next_path_idx = self.path.compute_orthogonal_projection(state[:2], self.next_path_idx, self.id_window_size)

        # Find the final index of the horizon
        horizon_duration = self.horizon_length / self.rate
        horizon_poses, cumul_duration = self.path.compute_horizon(closest_pose, self.next_path_idx, horizon_duration, self.maximum_linear_velocity, self.maximum_angular_velocity)
        horizon_length = min(self.horizon_length, cumul_duration[-1])
        
        interp_distances = np.linspace(0, horizon_length, self.horizon_length)
        interp_x = np.interp(interp_distances, cumul_duration, horizon_poses[:, 0])
        interp_y = np.interp(interp_distances, cumul_duration, horizon_poses[:, 1])
        interp_yaw = np.interp(interp_distances, cumul_duration, horizon_poses[:, 2])        
        interp_poses = list(zip(interp_x, interp_y, interp_yaw))

        self.target_trajectory = np.array(interp_poses).T


    def compute_command_vector(self, state):
        self.planar_state = np.array([state[0], state[1], state[5]])
        self.compute_desired_trajectory(self.planar_state)
        nlp_params = np.concatenate((self.planar_state, self.target_trajectory.flatten('C'),
            np.array([self.input_cost_wheel]), np.array([self.state_cost_translational]),
            np.array([self.state_cost_rotational])
        )) #,np.array([self.angular_velocity_gain]
        self.optim_control_solution = self.optim_problem_solver(x0=self.previous_input_array.flatten(),
                                                           p=nlp_params,
                                                           lbx= self.lower_bound_input,
                                                           ubx= self.upper_bound_input)['x']

        self.optimal_left = self.optim_control_solution[0]
        self.optimal_right = self.optim_control_solution[self.horizon_length]
        wheel_input_array = np.array([self.optimal_left, self.optimal_right]).reshape(2,1)
        body_input_array = self.ideal_diff_drive.compute_body_vel(wheel_input_array).astype('float64')

        optim_trajectory = self.horizon_pred(self.planar_state, self.optim_control_solution)
        self.optim_solution_array = np.array(self.optim_control_solution)
        self.optim_trajectory_array[0:2, :] = optim_trajectory[0:2, :]
        self.previous_input_array[0, :] = self.optim_solution_array[:self.horizon_length].reshape(self.horizon_length)
        self.previous_input_array[1, :] = self.optim_solution_array[self.horizon_length:].reshape(self.horizon_length)
        self.compute_distance_to_goal(state, self.next_path_idx)
        return body_input_array.reshape(2)


if __name__ == "__main__":

    from norlabcontrollib.path.path import Path
    
    parameter_map = {
        'maximum_linear_velocity': 2.0,
        'minimum_linear_velocity': 0.1,
        'maximum_angular_velocity': 0.5,
        'goal_tolerance': 0.5,
        'path_look_ahead_distance': 2.0,
        'query_radius': 50,
        'query_knn': 10,
        'id_window_size': 50,
        'horizon_length': 80,
        'state_cost_translational': 1.0,
        'state_cost_rotational': 0.2,
        'input_cost_wheel': 0.0001,
        'angular_velocity_gain': 1.0,
        'wheel_radius': 0.3,
        'baseline': 1.2,
        'rate': 20.0,
    }

    robot_pose = np.array([-0.1, 1.7, 0.0, 0.0, 0.0, 1.6])

    dummy_path = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 0.3, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 0.6, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 0.9, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 1.2, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 1.5, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 1.8, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 2.1, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 2.4, 0.0, 0.0, 0.0, 1.6],
                           [0.0, 2.4, 0.0, 0.0, 0.0, 1.2],
                           [0.0, 2.4, 0.0, 0.0, 0.0, 0.8],
                           [0.0, 2.4, 0.0, 0.0, 0.0, 0.4],
                           [0.0, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [0.3, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [0.6, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [0.9, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [1.2, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [1.5, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [1.8, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [2.1, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [2.4, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [2.7, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [3.0, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [3.3, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [3.6, 2.4, 0.0, 0.0, 0.0, 0.0],
                           [3.9, 2.4, 0.0, 0.0, 0.0, 0.0],
    ])

    controller = IdealDiffDriveMPC(parameter_map)
    controller.update_path(Path(dummy_path))
    controller.compute_desired_trajectory(robot_pose)
    controller.compute_command_vector(robot_pose)

    print('Target trajectory:', controller.target_trajectory)

    # Plot trajectory
    import matplotlib.pyplot as plt
    plt.plot(dummy_path[:, 0], dummy_path[:, 1], 'ro-', label='Reference Path')
    plt.scatter(controller.target_trajectory[0, 0], controller.target_trajectory[1, 0], c='g', label='Orthogonal Projection')
    plt.plot(controller.target_trajectory[0, 1:], controller.target_trajectory[1, 1:], 'bo', label='Horizon Trajectory')
    plt.plot(controller.optim_trajectory_array[0, 1:], controller.optim_trajectory_array[1, 1:], 'yo', label='Optimal Trajectory')
    plt.scatter(robot_pose[0], robot_pose[1], c='k', label='Robot Pose')
    
    # Display robot orientation (x-forward)
    plt.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[5]), np.sin(robot_pose[5]), color='k')
    
    plt.axis('equal')
    plt.legend()
    plt.show()


