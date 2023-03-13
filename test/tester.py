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
init_state = np.zeros(3)
init_state[1] = 0.5
# init_state[1] = 2
controller.init_state = init_state
controller.compute_desired_trajectory(init_state)
print(controller.target_trajectory)
# controller.predict_then_compute_cost(controller.previous_input_array)

############################ CASADI optimal control test
x0 = cas.SX(3, 1)
x0[:, 0] = init_state
u = cas.SX.sym('u', controller.horizon_length, 2)
R = cas.SX.eye(3)
R[0,0] = cas.cos(x0[2, 0])
R[1,1] = cas.cos(x0[2, 0])
R[0, 1] = -cas.sin((x0[2, 0]))
R[1, 0] = cas.sin((x0[2, 0]))
J = cas.SX(3, 2)
J[0,:] = controller.wheel_radius / 2
J[1, :] = 0
J[2, 0] = -controller.wheel_radius / controller.baseline
J[2, 1] = controller.wheel_radius / controller.baseline

ideal_dd = Ideal_diff_drive(controller.wheel_radius, controller.baseline, 1/controller.rate)
idd_state = np.zeros(6)

casadi_x = cas.SX.sym('x', 3)
casadi_u = cas.SX.sym('u', 2)

R[0, 0] = cas.cos(casadi_x[2])
R[1, 1] = cas.cos(casadi_x[2])
R[0, 1] = -cas.sin((casadi_x[2]))
R[1, 0] = cas.sin((casadi_x[2]))
x_k = casadi_x + cas.mtimes(R, cas.mtimes(J, casadi_u)) / controller.rate

single_step_pred = cas.Function('single_step_pred', [casadi_x, casadi_u], [x_k])
x_0 = cas.SX(3, 1)
x_0[:] = init_state
x_horizon_list = [x_0]
u_horizon_flat = cas.SX.sym('u_horizon_flat',2*controller.horizon_length)
u_horizon = cas.SX(controller.horizon_length, 2)
u_horizon[:, 0] = u_horizon_flat[:controller.horizon_length]
u_horizon[:, 1] = u_horizon_flat[controller.horizon_length:]

x_ref = cas.DM.zeros(3, controller.horizon_length)
x_ref[:, :] = controller.target_trajectory
state_cost_matrix = cas.DM.zeros(3,3)
state_cost_matrix[:, :] = controller.state_cost_matrix
u_ref = cas.DM.zeros(2, controller.horizon_length)
u_ref[:, :] = controller.previous_input_array
u_ref = u_ref.T
input_cost_matrix = cas.DM.zeros(2,2)
input_cost_matrix[:, :] = controller.input_cost_matrix
prediction_cost = cas.SX(0)

for i in range(1, controller.horizon_length):
    x_horizon_list.append(single_step_pred(x_horizon_list[i-1], u_horizon[i-1, :]))
    x_error = x_ref[i]- x_horizon_list[i]
    state_cost = cas.mtimes(cas.mtimes(x_error.T, state_cost_matrix), x_error)
    u_error = u_ref[i-1, :] - u_horizon[i-1, :]
    input_cost = cas.mtimes(cas.mtimes(u_error, input_cost_matrix), u_error.T)
    prediction_cost = prediction_cost + state_cost + input_cost

x_horizon = cas.hcat(x_horizon_list)
horizon_pred = cas.Function('horizon_pred', [u_horizon_flat], [x_horizon])
pred_cost = cas.Function('pred_cost', [u_horizon_flat], [prediction_cost])

casadi_single_step_pred = single_step_pred(init_state, controller.previous_input_array[:, 0])
idd_single_step_pred = ideal_dd.predict(idd_state, controller.previous_input_array[:, 0])

t0 = time.time()
for i in range(1, controller.horizon_length):
    idd_state = ideal_dd.predict(idd_state, controller.previous_input_array[:, i])
t1 = time.time()
t_idd = t1 - t0
#
# f = Function('f', [u], [x_k])
t0 = time.time()
# casadi_idd_pred = horizon_pred(controller.previous_input_array.T)
test_pred_cost = pred_cost(controller.previous_input_array.flatten())
t1 = time.time()
t_cas = t1 - t0

# Declare variables
x = cas.SX.sym("x",2)
f = x[0]**2 + x[1]**2 # objective
g = x[0]+x[1]-10      # constraint
prob = {'x':x, 'f':f}
optim_problem = {"f":prediction_cost, "x":u_horizon_flat}
solver = cas.nlpsol('solver', 'ipopt', prob)
opts = {}
optim_problem_solver = cas.nlpsol("optim_problem_solver", "ipopt", optim_problem)

# optim_problem_solution = optim_problem_solver(x0=controller.previous_input_array.flatten())
optim_problem_solution = optim_problem_solver(x0=np.zeros(controller.horizon_length*2))

print('test')

