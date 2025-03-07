from norlabcontrollib.controllers.controller import Controller
from norlabcontrollib.models.ideal_diff_drive import IdealDiffDrive
from norlabcontrollib.util.util_func import interp_angles, wrap2pi

import numpy as np
import casadi as cas

class IdealDiffDriveMPC(Controller):

    def __init__(self, parameter_map):
        super().__init__(parameter_map)

        self.number_states = 3
        self.number_inputs = 2

        self.horizon_length = parameter_map['horizon_length']
        self.id_window_size = parameter_map['id_window_size']

        self.linear_state_cost = parameter_map['linear_state_cost']
        self.angular_state_cost = parameter_map['angular_state_cost']
        self.velocity_delta_cost = parameter_map['velocity_delta_cost']
        
        self.wheel_radius = parameter_map['wheel_radius']
        self.baseline = parameter_map['baseline']

        self.linear_distance_to_goal = float('inf')
        self.euclidean_distance_to_goal = float('inf')
        self.angular_distance_to_goal = np.pi

        self.next_path_idx = 0
        self.next_command_id = 0
        
        self.init_casadi_model()


    def init_casadi_model(self):
        
        self.motion_model = IdealDiffDrive(self.wheel_radius, self.baseline, 1/self.rate)
        self.max_wheel_vel = self.motion_model.compute_wheel_vels(np.array([self.maximum_linear_velocity, 0]))[0]
        self.lower_bound_input = np.full(2 * self.horizon_length, -self.max_wheel_vel)
        self.upper_bound_input = np.full(2 * self.horizon_length, self.max_wheel_vel)
        self.optim_commands_wheels = np.zeros((2, self.horizon_length))
        self.optim_commands_body = np.zeros((2, self.horizon_length))
        self.target_trajectory = np.zeros((3, self.horizon_length))
        self.optim_trajectory = np.zeros((3, self.horizon_length))
  
        x = cas.SX.sym('x', 3)
        u = cas.SX.sym('u', 2)

        J = self.motion_model.jacobian_3x2
        R = cas.vertcat(
            cas.horzcat(cas.cos(x[2]), -cas.sin(x[2]), 0),
            cas.horzcat(cas.sin(x[2]), cas.cos(x[2]), 0),
            cas.horzcat(0, 0, 1)
        )

        x_0 = cas.SX.sym('x_0', 3, 1)
        x_k = x + cas.mtimes(R, cas.mtimes(J, u)) / self.rate
        single_step_pred = cas.Function('single_step_pred', [x, u], [x_k])

        u_horizon_flat = cas.SX.sym('u_horizon_flat', 2 * self.horizon_length)
        u_horizon = cas.reshape(u_horizon_flat, 2, self.horizon_length).T

        x_ref_flat = cas.SX.sym('x_ref', 3 * self.horizon_length)
        x_ref = cas.reshape(x_ref_flat, self.horizon_length, 3).T
        u_ref = cas.DM(self.optim_commands_wheels.T)

        velocity_delta_cost = cas.SX.sym('velocity_delta_cost', 1)
        linear_state_cost = cas.SX.sym('linear_state_cost', 1)
        angular_state_cost = cas.SX.sym('angular_state_cost', 1)

        velocity_cost_matrix = cas.diag(cas.repmat(velocity_delta_cost, 2, 1))
        state_cost_matrix = cas.diag(cas.vertcat(
            linear_state_cost,
            linear_state_cost,
            angular_state_cost
        ))
        
        x_horizon_list = [x_0]
        prediction_cost = cas.SX(0)

        for i in range(1, self.horizon_length):
            x_horizon_list.append(single_step_pred(x_horizon_list[i - 1], u_horizon[i - 1, :]))
            x_error = x_ref[:, i] - x_horizon_list[i]
            x_error[2] = cas.atan2(cas.sin(x_error[2]), cas.cos(x_error[2]))
            state_cost = cas.mtimes(cas.mtimes(x_error.T, state_cost_matrix), x_error)
            u_error = u_ref[i - 1, :] - u_horizon[i - 1, :]
            input_cost = cas.mtimes(cas.mtimes(u_error, velocity_cost_matrix), u_error.T)
            prediction_cost = prediction_cost + state_cost + input_cost

        x_horizon = cas.hcat(x_horizon_list)
        self.horizon_pred = cas.Function('horizon_pred', [x_0, u_horizon_flat], [x_horizon])

        nlp_params = cas.vertcat(
            x_0, 
            x_ref_flat, 
            velocity_delta_cost, 
            linear_state_cost, 
            angular_state_cost
        )
        optim_problem = {
            "f": prediction_cost, 
            "x": u_horizon_flat, 
            "p": nlp_params
        }
        nlpsol_opts = {
            'verbose_init': False, 
            'print_in': False, 
            'print_out': False, 
            'print_time': False,
            'verbose': False, 
            'ipopt': {'print_level': 0}
        }
        self.optim_problem_solver = cas.nlpsol("optim_problem_solver", "ipopt", optim_problem, nlpsol_opts)


    def compute_distance_to_goal(self, state, orthogonal_projection_id):

        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        distance_to_goal_path = self.path.distances_to_goal[orthogonal_projection_id]
        distance_to_next_node = np.linalg.norm(self.path.poses[orthogonal_projection_id, :2] - state[:2])
        self.linear_distance_to_goal =  distance_to_goal_path + distance_to_next_node 
        self.angular_distance_to_goal = np.abs(wrap2pi(self.path.poses[-1, 5] - state[5]))
        
        
    def compute_desired_trajectory(self, state):
        
        # Find closest point on path
        self.closest_pose, self.next_path_idx = self.path.compute_orthogonal_projection(
            state, self.next_path_idx, self.id_window_size, self.maximum_linear_velocity, self.maximum_angular_velocity
        )

        # Find the points on the path that are accessible within the horizon
        horizon_duration = self.horizon_length / self.rate
        horizon_poses, cumul_duration = self.path.compute_horizon(self.closest_pose, self.next_path_idx, horizon_duration, self.maximum_linear_velocity, self.maximum_angular_velocity)
        horizon_duration = min(horizon_duration, cumul_duration[-1])
        
        # Interpolate the poses to get the desired trajectory
        interp_duration = np.linspace(0, horizon_duration, self.horizon_length)
        interp_x = np.interp(interp_duration, cumul_duration, horizon_poses[:, 0])
        interp_y = np.interp(interp_duration, cumul_duration, horizon_poses[:, 1])
        interp_yaw = interp_angles(interp_duration, cumul_duration, horizon_poses[:, 2])   
        interp_poses = list(zip(interp_x, interp_y, interp_yaw))

        self.target_trajectory = np.array(interp_poses).T


    def compute_command_vector(self, state):

        self.planar_state = np.array([state[0], state[1], state[5]])
        self.compute_desired_trajectory(self.planar_state)

        nlp_params = np.concatenate((
            self.planar_state, 
            self.target_trajectory.flatten(),
            np.array([self.velocity_delta_cost]), 
            np.array([self.linear_state_cost]),
            np.array([self.angular_state_cost])
        )) 

        self.optim_control_solution = self.optim_problem_solver(
            x0=self.optim_commands_wheels.flatten(),
            p=nlp_params,
            lbx= self.lower_bound_input,
            ubx= self.upper_bound_input
        )['x']

        self.optim_commands_wheels = np.array(self.optim_control_solution).reshape(2, self.horizon_length)
        self.optim_commands_body = self.motion_model.compute_body_vel(self.optim_commands_wheels).astype('float64')
        self.optim_trajectory = np.array(self.horizon_pred(self.planar_state, self.optim_control_solution))

        self.compute_distance_to_goal(state, self.next_path_idx)
        self.next_command_id = 1

        return self.optim_commands_body[:, 0]
    

    def get_next_command(self):

        if self.next_command_id < self.horizon_length:
            next_command = self.optim_commands_body[:, self.next_command_id]
            self.next_command_id += 1
        else:
            next_command = np.array([0.0, 0.0]) # Stop if we never receive odom

        return next_command.reshape(2), self.next_command_id - 1


    def goal_reached(self):

        return (self.linear_distance_to_goal < self.linear_goal_tolerance and 
                self.angular_distance_to_goal < self.angular_goal_tolerance)       


if __name__ == "__main__":

    from norlabcontrollib.path.path import Path
    
    parameter_map = {
        'maximum_linear_velocity': 2.0,
        'minimum_linear_velocity': 0.1,
        'maximum_angular_velocity': 1.0,
        'linear_goal_tolerance': 0.5,
        'angular_goal_tolerance': 0.5,
        'id_window_size': 10,
        'horizon_length': 40,
        'linear_state_cost': 1.0,
        'angular_state_cost': 0.2,
        'velocity_delta_cost': 0.001,
        'wheel_radius': 0.3,
        'baseline': 1.2,
        'rate': 20.0,
    }

    dummy_path = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-0.3, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-0.6, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-0.9, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-1.2, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-1.5, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-1.8, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-2.1, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-2.4, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-2.7, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-3.0, 0.0, 0.0, 0.0, 0.0, np.pi],
                        [-3.0, 0.0, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -0.3, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -0.6, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -0.9, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -1.2, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -1.5, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -1.8, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -2.1, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -2.4, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -2.7, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -3.0, 0.0, 0.0, 0.0, -np.pi/2],
                        [-3.0, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-2.7, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-2.4, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-2.1, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.8, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.5, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.2, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.9, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.6, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.3, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -3.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -3.0, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -2.7, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -2.4, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -2.1, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -1.8, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -1.5, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -1.2, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -0.9, 0.0, 0.0, 0.0, np.pi/2],
                        [0.0, -0.6, 0.0, 0.0, 0.0, np.pi/2]
    ])

    def reverse_path(path):
        reversed_path = np.copy(path)
        reversed_path = reversed_path[::-1]
        for i, yaw in enumerate(reversed_path[:, 5]):
            reversed_path[i, 5] = wrap2pi(yaw + np.pi)
        return reversed_path

    dummy_path = reverse_path(dummy_path)

    robot_pose = np.array([0.2, -0.3, 0.0, 0.0, 0.0, -1.5])

    controller = IdealDiffDriveMPC(parameter_map)
    controller.update_path(Path(dummy_path))
    controller.compute_command_vector(robot_pose)

    def on_key_press(event):

        global controller, robot_pose
        
        robot_pose = [
            controller.optim_trajectory[0, 1],
            controller.optim_trajectory[1, 1],
            0.0,
            0.0,
            0.0,
            wrap2pi(controller.optim_trajectory[2, 1])
        ]
        controller.compute_command_vector(robot_pose)

        fig.clear()
        draw_plot()
        fig.canvas.draw()


    def draw_plot():
        plt.plot(dummy_path[:, 0], dummy_path[:, 1], 'ro-', label='Reference Path')
        plt.scatter(controller.target_trajectory[0, 0], controller.target_trajectory[1, 0], c='g', label='Orthogonal Projection')
        plt.plot(controller.target_trajectory[0, 1:], controller.target_trajectory[1, 1:], 'bo', label='Horizon Trajectory')
        plt.quiver(controller.target_trajectory[0, 1:], controller.target_trajectory[1, 1:], np.cos(controller.target_trajectory[2, 1:]), np.sin(controller.target_trajectory[2, 1:]), color='b', label='Horizon Trajectory')
        plt.plot(controller.optim_trajectory[0, 1:], controller.optim_trajectory[1, 1:], 'yo', label='Optimal Trajectory')
        plt.quiver(controller.optim_trajectory[0, 1:], controller.optim_trajectory[1, 1:], np.cos(controller.optim_trajectory[2, 1:]), np.sin(controller.optim_trajectory[2, 1:]), color='y', label='Optimal Trajectory')
        plt.scatter(robot_pose[0], robot_pose[1], c='k', label='Robot Pose')
        plt.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[5]), np.sin(robot_pose[5]), color='k')
        plt.axis('equal')
        plt.quiver(controller.closest_pose[0],controller.closest_pose[1],np.cos(controller.closest_pose[2]),np.sin(controller.closest_pose[2]),color="cyan",label="closest pose",zorder=10)
        plt.legend()
        plt.ylim(-3.5,0.0)
        plt.xlim(-1,0.5)


    # Plot trajectory
    import matplotlib.pyplot as plt
    fig = plt.figure()
    draw_plot()

    # Connect the event handler to the figure
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()


