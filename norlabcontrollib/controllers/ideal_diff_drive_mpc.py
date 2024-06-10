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
        horizon_distance = self.maximum_linear_velocity / self.rate * self.horizon_length
        horizon_poses, _ = self.path.compute_horizon(closest_pose, self.next_path_idx, horizon_distance)

        # Compute the desired trajectory by interpolating the horizon path
        cumulative_distances = np.zeros(len(horizon_poses))
        for i in range(1, len(horizon_poses)):
            cumulative_distances[i] = cumulative_distances[i - 1] + np.linalg.norm(horizon_poses[i, :2] - horizon_poses[i - 1, :2])
        
        interp_distances = np.linspace(0, horizon_distance, self.horizon_length+1)
        interp_x = np.interp(interp_distances, cumulative_distances, horizon_poses[:, 0])
        interp_y = np.interp(interp_distances, cumulative_distances, horizon_poses[:, 1])
        interp_yaw = np.interp(interp_distances, cumulative_distances, horizon_poses[:, 2])        
        interp_poses = list(zip(interp_x, interp_y, interp_yaw))

        self.target_trajectory = np.array(interp_poses).T


    def compute_command_vector(self, state):
        self.planar_state = np.array([state[0], state[1], state[5]])
        self.compute_desired_trajectory(self.planar_state)
        nlp_params = np.concatenate((self.planar_state, self.target_trajectory.flatten('C')))
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



# if __name__ == "__main__":
    
#     parameter_map = {
#         'maximum_linear_velocity': 2.0,
#         'minimum_linear_velocity': 0.0,
#         'maximum_angular_velocity': 1.0,
#         'goal_tolerance': 0.1,
#         'path_look_ahead_distance': 1.0,
#         'query_radius': 0.5,
#         'query_knn': 5,
#         'id_window_size': 2000,
#         'horizon_length': 20,
#         'state_cost_translational': 1.0,
#         'state_cost_rotational': 1.0,
#         'input_cost_wheel': 0.01,
#         'angular_velocity_gain': 1.0,
#         'wheel_radius': 0.1,
#         'baseline': 0.2,
#         'rate': 10.0,
#     }

#     robot_pose = np.array([5.0, 3.0, -1.6])

#     dummy_path = np.array([[4.9192, 0.9774, -0.7686, -0.0343, -0.0452, -1.6334],
#                           [4.8898, 0.9345, -0.7768, -0.0343, -0.0509, -1.6334],
#                           [4.8872, 0.8612, -0.7752, -0.0329, -0.0626, -1.6299],
#                           [4.8821, 0.8017, -0.7790, -0.0312, -0.0654, -1.6307],
#                           [4.8729, 0.7305, -0.7780, -0.0319, -0.0633, -1.6302],
#                           [4.8683, 0.6348, -0.7808, -0.0291, -0.0601, -1.6283],
#                           [4.8412, 0.2877, -0.7658, -0.0289, -0.0512, -1.6284],
#                           [4.8167, -0.1704, -0.7454, -0.0277, -0.0525, -1.6265],
#                           [4.8168, -0.2741, -0.7444, -0.0254, -0.0540, -1.6272],
#                           [4.8076, -0.3741, -0.7480, -0.0250, -0.0505, -1.6271],
#                           [4.8026, -0.4651, -0.7424, -0.0263, -0.0479, -1.6280],
#                           [4.7931, -0.5668, -0.7332, -0.0241, -0.0499, -1.6281],
#                           [4.7861, -0.6897, -0.7369, -0.0238, -0.0534, -1.6270],
#                           [4.7762, -0.7906, -0.7357, -0.0234, -0.0510, -1.6279],
#                           [4.7766, -0.9113, -0.7310, -0.0215, -0.0544, -1.6289],
#                           [4.7591, -1.0305, -0.7354, -0.0234, -0.0528, -1.6294],
#                           [4.7525, -1.1428, -0.7378, -0.0211, -0.0474, -1.6283],
#                           [4.7445, -1.2558, -0.7289, -0.0190, -0.0458, -1.6296],
#                           [4.7380, -1.3876, -0.7197, -0.0197, -0.0470, -1.6297],
#                           [4.7249, -1.4855, -0.7181, -0.0209, -0.0443, -1.6309],
#                           [4.7139, -1.5937, -0.7173, -0.0185, -0.0432, -1.6292],
#                           [4.7055, -1.6969, -0.7105, -0.0171, -0.0431, -1.6301],
#                           [4.7042, -1.8112, -0.7040, -0.0132, -0.0439, -1.6296],
#                           [4.6992, -1.9425, -0.7093, -0.0122, -0.0475, -1.6273],
#                           [4.6947, -2.0598, -0.7004, -0.0129, -0.0486, -1.6298],
#                           [4.6921, -2.1696, -0.6949, -0.0184, -0.0460, -1.6327],
#                           [4.6721, -2.2915, -0.6964, -0.0205, -0.0503, -1.6318],
#                           [4.6643, -2.4106, -0.6962, -0.0177, -0.0499, -1.6307],
#                           [4.6603, -2.5276, -0.6903, -0.0155, -0.0493, -1.6312],
#                           [4.6412, -2.6400, -0.6909, -0.0147, -0.0470, -1.6308],
#                           [4.6279, -2.7431, -0.6802, -0.0101, -0.0492, -1.6290],
#                           [4.6337, -2.8586, -0.6709, -0.0096, -0.0554, -1.6307],
#                           [4.6205, -2.9988, -0.6688, -0.0148, -0.0644, -1.6360],
#                           [4.6021, -3.1046, -0.6723, -0.0194, -0.0564, -1.6383],
#                           [4.5850, -3.2175, -0.6639, -0.0194, -0.0549, -1.6383],
#                           [4.5799, -3.3280, -0.6615, -0.0173, -0.0574, -1.6382],
#                           [4.5708, -3.4621, -0.6493, -0.0180, -0.0707, -1.6374],
#                           [4.5623, -3.5712, -0.6459, -0.0176, -0.0741, -1.6370],
#                           [4.5453, -3.6822, -0.6382, -0.0152, -0.0742, -1.6376],
#                           [4.5340, -3.7850, -0.6350, -0.0128, -0.0723, -1.6379],
#                           [4.5402, -3.9054, -0.6218, -0.0082, -0.0797, -1.6365],
#                           [4.5325, -4.0432, -0.6137, -0.0127, -0.0884, -1.6386],
#                           [4.5159, -4.1452, -0.6223, -0.0194, -0.0750, -1.6447],
#                           [4.4880, -4.2659, -0.6138, -0.0168, -0.0710, -1.6448],
#                           [4.4847, -4.3928, -0.6129, -0.0126, -0.0674, -1.6424],
#                           [4.4800, -4.4833, -0.6002, -0.0114, -0.0642, -1.6442],
#                           [4.4693, -4.5887, -0.5899, -0.0161, -0.0736, -1.6457],
#                           [4.4577, -4.7374, -0.5815, -0.0124, -0.0880, -1.6458],
#                           [4.4421, -4.8826, -0.5756, -0.0082, -0.0856, -1.6433],
#                           [4.4288, -5.0177, -0.5711, -0.0099, -0.0751, -1.6471],
#                           [4.4179, -5.1591, -0.5664, -0.0116, -0.0802, -1.6477],
#                           [4.3987, -5.3096, -0.5526, -0.0112, -0.0862, -1.6459],
#                           [4.3846, -5.4634, -0.5453, -0.0087, -0.0862, -1.6477],
#                           [4.3735, -5.6095, -0.5295, -0.0096, -0.0764, -1.6534],
#                           [4.3599, -5.7257, -0.5330, -0.0156, -0.0661, -1.6591],
#                           [4.2718, -6.3195, -0.4964, -0.0128, -0.0867, -1.6928],
#                           [4.0932, -6.9990, -0.4635, -0.0143, -0.0936, -1.7814],
#                           [4.0652, -7.1368, -0.4495, -0.0115, -0.0911, -1.8032],
#                           [3.8552, -7.6624, -0.4522, -0.0196, -0.0899, -1.8984],
#                           [3.6968, -8.0394, -0.4370, -0.0261, -0.1004, -1.9736],
#                           [3.6096, -8.1629, -0.4479, -0.0185, -0.0888, -1.9952],
#                           [3.2113, -8.8702, -0.4166, -0.0251, -0.0919, -2.1498],
#                           [2.7641, -9.4077, -0.4530, -0.0375, -0.0957, -2.2955],
#                           [2.6695, -9.5035, -0.4240, -0.0293, -0.0991, -2.3268],
#                           [2.5695, -9.5986, -0.4322, -0.0280, -0.1081, -2.3516],
#                           [2.4514, -9.6904, -0.4275, -0.0206, -0.1023, -2.3753],
#                           [2.3468, -9.7721, -0.4165, -0.0180, -0.0982, -2.3991],
#                           [2.2407, -9.8512, -0.4038, -0.0165, -0.0996, -2.4265],
#                           [2.1274, -9.9307, -0.3776, -0.0185, -0.1133, -2.4556],
#                           [2.0088, -10.0127, -0.3796, -0.0221, -0.1099, -2.4835],
#                           [1.9055, -10.0749, -0.3709, -0.0232, -0.1015, -2.5114],
#                           [1.7837, -10.1433, -0.3716, -0.0222, -0.0985, -2.5326],
#                           [1.6552, -10.2209, -0.3577, -0.0123, -0.1048, -2.5504],
#                           [1.5357, -10.2840, -0.3554, -0.0147, -0.1061, -2.5662],
#                           [1.4171, -10.3562, -0.3285, -0.0143, -0.1014, -2.5840],
#                           [1.2917, -10.4276, -0.3244, -0.0155, -0.1053, -2.5968],
#                           [1.1559, -10.4962, -0.3021, -0.0146, -0.1077, -2.6091],
#                           [1.0367, -10.5529, -0.3018, -0.0194, -0.1014, -2.6217],
#                           [0.9052, -10.6356, -0.2893, -0.0134, -0.0997, -2.6310],
#                           [0.7946, -10.6928, -0.2670, -0.0152, -0.1027, -2.6359],
#                           [0.5373, -10.8147, -0.2491, -0.0260, -0.1034, -2.6520],
#                           [0.1294, -11.0108, -0.2062, -0.0151, -0.1042, -2.6686],
#                           [-0.1191, -11.1272, -0.1846, -0.0200, -0.1058, -2.6814],
#                           [-0.5141, -11.2972, -0.1458, -0.0214, -0.1153, -2.6938],
#                           [-0.6491, -11.3633, -0.1182, -0.0200, -0.1162, -2.6993],
#                           [-0.7923, -11.4254, -0.1300, -0.0193, -0.1055, -2.7067],
#                           [-0.8958, -11.4700, -0.1161, -0.0169, -0.0995, -2.7081],
#                           [-1.0091, -11.5167, -0.1050, -0.0200, -0.1115, -2.7089],
#                           [-1.0922, -11.5545, -0.0987, -0.0243, -0.1108, -2.7150],
#                           [-1.1783, -11.5891, -0.0856, -0.0250, -0.1181, -2.7186],
#                           [-1.2474, -11.6286, -0.0816, -0.0269, -0.1218, -2.7100],
#                           [-1.3232, -11.6719, -0.0738, -0.0298, -0.1280, -2.6901],
#                           [-1.3954, -11.7119, -0.0653, -0.0336, -0.1304, -2.6653], 
#                           [-1.4607, -11.7805, -0.0791, -0.0283, -0.1207, -2.6341],
#                           [-1.5181, -11.8458, -0.0680, -0.0264, -0.1229, -2.5912],
#                           [-1.5742, -11.8963, -0.0685, -0.0336, -0.1169, -2.5493],
#                           [-1.6359, -11.9625, -0.0592, -0.0377, -0.1124, -2.5075],
#                           [-1.6741, -12.0449, -0.0542, -0.0395, -0.1069, -2.4597],
#                           [-1.7162, -12.1248, -0.0468, -0.0415, -0.1007, -2.4058],
#                           [-1.7486, -12.1898, -0.0699, -0.0457, -0.0907, -2.3518],
#                           [-1.7997, -12.2866, -0.0656, -0.0466, -0.0969, -2.3009],
#                           [-1.8481, -12.3844, -0.0416, -0.0499, -0.1040, -2.2499],
#                           [-1.8870, -12.4720, -0.0323, -0.0468, -0.1003, -2.2040],
#                           [-1.9816, -12.6829, -0.0096, -0.0528, -0.0963, -2.1174],
#                           [-2.0993, -12.9243, 0.0692, -0.0541, -0.1018, -2.0364],
#                           [-2.1825, -13.1455, 0.0761, -0.0661, -0.0898, -1.9654],
#                           [-2.2441, -13.3891, 0.0840, -0.0636, -0.0843, -1.8812],
#                           [-2.2251, -13.5169, -0.0014, -0.0671, -0.0754, -1.8399],
#                           [-2.2779, -14.1285, -0.0656, -0.0654, -0.0544, -1.5892],
#                           [-2.2240, -14.3585, -0.0275, -0.0654, -0.0537, -1.4742],
#                           [-2.1953, -14.4747, -0.0392, -0.0559, -0.0515, -1.4126],
#                           [-2.1580, -14.5879, -0.0432, -0.0602, -0.0481, -1.3522],
#                           [-2.0993, -14.7067, -0.0609, -0.0578, -0.0423, -1.2902],
#                           [-2.0675, -14.8109, -0.0492, -0.0538, -0.0339, -1.2206],
#                           [-2.0103, -14.9256, -0.0765, -0.0526, -0.0362, -1.1594],
#                           [-1.9276, -15.0259, -0.0871, -0.0532, -0.0317, -1.0949],
#                           [-1.8225, -15.1368, -0.0711, -0.0446, -0.0372, -1.0359],
#                           [-1.7722, -15.2288, -0.0913, -0.0468, -0.0280, -0.9756],
#                           [-1.7031, -15.3069, -0.0791, -0.0401, -0.0194, -0.9099],
#                           [-1.2494, -15.6829, -0.0978, -0.0363, -0.0254, -0.7236],
#                           [-0.3865, -16.2965, -0.0910, -0.0184, -0.0265, -0.5642],
#                           [0.0017, -16.5036, -0.1014, -0.0056, -0.0249, -0.5126],
#                           [0.0857, -16.5636, -0.1142, -0.0054, -0.0097, -0.5007],
#                           [0.1647, -16.6111, -0.1115, -0.0057, 0.0016, -0.4923],
#                           [0.2025, -16.6467, -0.0925, -0.0041, -0.0015, -0.4881],
#                           [0.2707, -16.6813, -0.0813, -0.0052, -0.0095, -0.4826],
#                           [0.3383, -16.7108, -0.0823, -0.0045, -0.0118, -0.4843],
#                           [0.4017, -16.7478, -0.0726, -0.0023, -0.0179, -0.4837],
#                           [0.4528, -16.7831, -0.0754, -0.0017, -0.0193, -0.4773],
#                           [0.5085, -16.8160, -0.0681, -0.0014, -0.0151, -0.4732],
#                           [0.5812, -16.8451, -0.0648, 0.0008, -0.0117, -0.4695],
#                           [0.6726, -16.8995, -0.0720, 0.0036, -0.0066, -0.4635],
#                           [0.7143, -16.9318, -0.0758, 0.0045, -0.0051, -0.4606],
#                           [0.7624, -16.9568, -0.0818, 0.0053, -0.0020, -0.4587],
#                           [0.8260, -16.9774, -0.0739, 0.0043, -0.0004, -0.4575],
#                           [0.8755, -17.0059, -0.0790, 0.0022, 0.0012, -0.4557],
#                           [0.9222, -17.0264, -0.0707, 0.0003, 0.0104, -0.4482],
#                           [0.9712, -17.0814, -0.0694, -0.0046, -0.0115, -0.4557],
#                           [1.0394, -17.1180, -0.0805, -0.0040, -0.0154, -0.4561],
#                           [1.1441, -17.1679, -0.0931, -0.0016, -0.0028, -0.4562],
#                           [1.2480, -17.2177, -0.0945, -0.0008, 0.0050, -0.4560],
#                           [1.7108, -17.4750, -0.0980, -0.0053, -0.0043, -0.4593],
#                           [1.9060, -17.5806, -0.0985, 0.0012, -0.0078, -0.4565],
#                           [2.2110, -17.7295, -0.0777, -0.0087, -0.0301, -0.4642],
#                           [2.3110, -17.7770, -0.0922, -0.0091, -0.0359, -0.4628],
#                           [2.9351, -18.0958, -0.0708, -0.0168, -0.0421, -0.4662],
#                           [3.1411, -18.2043, -0.0679, -0.0181, -0.0438, -0.4640],
#                           [3.2362, -18.2500, -0.0627, -0.0191, -0.0430, -0.4662],
#                           [3.3268, -18.3039, -0.0491, -0.0159, -0.0454, -0.4654],
#                           [3.4281, -18.3607, -0.0465, -0.0132, -0.0511, -0.4636],
#                           [3.6316, -18.4537, -0.0386, -0.0073, -0.0587, -0.4660],
#                           [3.7334, -18.5049, -0.0413, -0.0096, -0.0601, -0.4665],
#                           [3.9173, -18.6072, -0.0293, -0.0116, -0.0545, -0.4688],
#                           [4.0193, -18.6567, -0.0332, -0.0118, -0.0528, -0.4677],
#                           [4.1082, -18.7119, -0.0327, -0.0113, -0.0503, -0.4681],
#                           [4.1806, -18.7482, -0.0263, -0.0081, -0.0364, -0.4682]
#     ])

#     controller = IdealDiffDriveMPC(parameter_map)
#     controller.update_path(Path(dummy_path))
#     controller.compute_desired_trajectory(robot_pose)

#     print('Target trajectory:', controller.target_trajectory)

#     # Plot trajectory
#     import matplotlib.pyplot as plt
#     plt.plot(dummy_path[:, 0], dummy_path[:, 1], 'ro-', label='Reference Path')
#     plt.scatter(controller.target_trajectory[0, 0], controller.target_trajectory[1, 0], c='g', label='Orthogonal Projection')
#     plt.plot(controller.target_trajectory[0, 1:], controller.target_trajectory[1, 1:], 'bo', label='Horizon Trajectory')
#     plt.scatter(robot_pose[0], robot_pose[1], c='k', label='Robot Pose')
    
#     # Display robot orientation (x-forward)
#     plt.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[2]), np.sin(robot_pose[2]), color='k')
    
#     plt.axis('equal')
#     plt.legend()
#     plt.show()


