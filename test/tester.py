from norlabcontrollib.controllers.controller_factory import ControllerFactory
from norlabcontrollib.path.path import Path
from norlabcontrollib.models.ideal_diff_drive import Ideal_diff_drive

import autograd.numpy as np
from autograd import grad
import time
import casadi as cas
import numdifftools as nd

controller_factory = ControllerFactory()
controller = controller_factory.load_parameters_from_yaml('test_parameters_smpc.yaml')
#
test_path_poses = np.load('../traj_a_int.npy')
test_path = Path(test_path_poses)
test_path.compute_metrics(controller.path_look_ahead_distance)
#
controller.update_path(test_path)
#
init_state = np.zeros(6)
init_state[0] = 0.0
init_state[1] = 2.5
init_state[5] = -1.5
# init_state[1] = 2
# init_state[0] = 9.703779
# init_state[1] = 5.218897
# init_state[2] = 1.4497365919889573
# controller.init_state = init_state
# controller.last_path_pose_id = 100
# controller.compute_command_vector(np.zeros(6))
# controller.predict_then_compute_cost(controller.previous_input_array)
# controller.compute_desired_trajectory(init_state)
t0 = time.time()
controller.compute_command_vector(init_state)
t1 = time.time()
print(t1 - t0)

############################ CASADI optimal control test
# x0 = cas.SX.sym('x0', 3, 1)
# x0[:, 0] = init_state
# u = cas.SX.sym('u', controller.horizon_length, 2)
# R = cas.SX.eye(3)
# R[0,0] = cas.cos(x0[2, 0])
# R[1,1] = cas.cos(x0[2, 0])
# R[0, 1] = -cas.sin((x0[2, 0]))
# R[1, 0] = cas.sin((x0[2, 0]))
# J = cas.SX(3, 2)
# J[0,:] = controller.wheel_radius / 2
# J[1, :] = 0
# J[2, 0] = -controller.wheel_radius / controller.baseline
# J[2, 1] = controller.wheel_radius / controller.baseline
#
# ideal_dd = Ideal_diff_drive(controller.wheel_radius, controller.baseline, 1/controller.rate)
# idd_state = np.zeros(6)
#
# casadi_x = cas.SX.sym('x', 3)
# casadi_u = cas.SX.sym('u', 2)
#
# R[0, 0] = cas.cos(casadi_x[2])
# R[1, 1] = cas.cos(casadi_x[2])
# R[0, 1] = -cas.sin((casadi_x[2]))
# R[1, 0] = cas.sin((casadi_x[2]))
# R_inv = R.T
#
# x_k = casadi_x + cas.mtimes(R_inv, cas.mtimes(J, casadi_u)) / controller.rate
#
# single_step_pred = cas.Function('single_step_pred', [casadi_x, casadi_u], [x_k])
# x_0 = cas.SX.sym('x_0', 3)
# # x_0[:] = init_state
# x_horizon_list = [x_0]
# u_horizon_flat = cas.SX.sym('u_horizon_flat',2*controller.horizon_length)
# u_horizon = cas.SX(controller.horizon_length, 2)
# u_horizon[:, 0] = u_horizon_flat[:controller.horizon_length]
# u_horizon[:, 1] = u_horizon_flat[controller.horizon_length:]
#
# x_ref_flat = cas.SX.sym('x_ref_flat', 3*controller.horizon_length)
# x_ref = cas.SX.zeros(3, controller.horizon_length)
# x_ref[0, :] = x_ref_flat[:controller.horizon_length]
# x_ref[1, :] = x_ref_flat[controller.horizon_length:2*controller.horizon_length]
# x_ref[2, :] = x_ref_flat[2*controller.horizon_length:3*controller.horizon_length]
# # x_ref[:, :] = controller.target_trajectory
# state_cost_matrix = cas.DM.zeros(3,3)
# state_cost_matrix[:, :] = controller.state_cost_matrix
# u_ref = cas.DM.zeros(2, controller.horizon_length)
# u_ref[:, :] = controller.previous_input_array
# u_ref = u_ref.T
# input_cost_matrix = cas.DM.zeros(2,2)
# input_cost_matrix[:, :] = controller.input_cost_matrix
# prediction_cost = cas.SX(0)
#
# for i in range(1, controller.horizon_length):
#     x_horizon_list.append(single_step_pred(x_horizon_list[i-1], u_horizon[i-1, :]))
#     x_error = x_ref[i]- x_horizon_list[i]
#     state_cost = cas.mtimes(cas.mtimes(x_error.T, state_cost_matrix), x_error)
#     u_error = u_ref[i-1, :] - u_horizon[i-1, :]
#     input_cost = cas.mtimes(cas.mtimes(u_error, input_cost_matrix), u_error.T)
#     prediction_cost = prediction_cost + state_cost + input_cost
#
# x_horizon = cas.hcat(x_horizon_list)
# horizon_pred = cas.Function('horizon_pred', [x_0, u_horizon_flat], [x_horizon])
# # pred_cost = cas.Function('pred_cost', [u_horizon_flat], [prediction_cost])
#
# casadi_single_step_pred = single_step_pred(init_state, controller.previous_input_array[:, 0])
# idd_single_step_pred = ideal_dd.predict(idd_state, controller.previous_input_array[:, 0])
#
# t0 = time.time()
# for i in range(1, controller.horizon_length):
#     idd_state = ideal_dd.predict(idd_state, controller.previous_input_array[:, i])
# t1 = time.time()
# t_idd = t1 - t0
# #
# # f = Function('f', [u], [x_k])
# t0 = time.time()
# # casadi_idd_pred = horizon_pred(controller.previous_input_array.T)
# # test_pred_cost = pred_cost(controller.previous_input_array.flatten())
# t1 = time.time()
# t_cas = t1 - t0
#
# params = cas.vertcat(x_0, x_ref_flat)
#
# optim_problem = {"f":prediction_cost, "x":u_horizon_flat, "p":params}
# opts = {}
# optim_problem_solver = cas.nlpsol("optim_problem_solver", "ipopt", optim_problem)

# optim_problem_solution = optim_problem_solver(x0=controller.previous_input_array.flatten())
# init_state = np.zeros(3)
# init_state[1] = 0.4
# init_state[2] = -1.6
# num_params = np.concatenate((init_state, controller.target_trajectory.flatten('C')))
# optim_problem_solution = optim_problem_solver(x0=np.zeros(controller.horizon_length*2), p=num_params)
# optim_control_input = optim_problem_solution['x']
# optim_trajectory = horizon_pred(init_state, optim_control_input)
# optim_trajectory_array = np.zeros((3, controller.horizon_length))
# optim_trajectory_array[0, :] = optim_trajectory[0, :]
# optim_trajectory_array[1, :] = optim_trajectory[1, :]
# optim_trajectory_array[2, :] = optim_trajectory[2, :]
# test_input_array = np.zeros(2*controller.horizon_length)
# test_input_array[controller.horizon_length:] = 5.0
# test_input_array[controller.horizon_length:] = 5.0
# test_trajectory = horizon_pred(init_state, test_input_array)
# test_trajectory_array = np.zeros((3, controller.horizon_length))
# test_trajectory_array[0, :] = test_trajectory[0, :]
# test_trajectory_array[1, :] = test_trajectory[1, :]
# test_trajectory_array[2, :] = test_trajectory[2, :]


# import matplotlib.pyplot as plt
# plt.scatter(controller.target_trajectory[1, :], controller.target_trajectory[0, :], s=10)
# plt.scatter(optim_trajectory_array[1, :], optim_trajectory_array[0, :], s=10)
# # plt.scatter(test_trajectory_array[1, :], test_trajectory_array[0, :], s=10)
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# # plt.scatter()
# plt.show()
#
#
print('test')

