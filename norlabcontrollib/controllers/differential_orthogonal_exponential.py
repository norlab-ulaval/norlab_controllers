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

        self.distance_to_goal = 100000

    def update_path(self, new_path):
        self.path = new_path

    def compute_distance_to_goal(self, state):
        self.distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        return None

    def compute_linear_velocity(self, orthogonal_projection_id):

        path_curvature = self.path.look_ahead_curvatures[orthogonal_projection_id]

        # print("Distance to goal :" + str(self.distance_to_goal))
        # print("Path curvature :" + str(path_curvature))

        command_linear_velocity = self.maximum_linear_velocity * np.exp(-self.gain_path_curvature_linear * path_curvature -
                                                                        -self.gain_distance_to_goal_linear / self.distance_to_goal)
        command_linear_velocity = np.clip(command_linear_velocity, self.minimum_linear_velocity, self.maximum_linear_velocity)
        #return command_linear_velocity
        return 0.2

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
        self.projection_pose_path_frame = self.apply_tf(state[:2],
                                                        self.path.world_to_path_tfs_array[orthogonal_projection_id])

        self.target_exponential_tangent_angle = np.arctan(-self.gain_path_convergence * self.projection_pose_path_frame[1])

        self.robot_yaw_path_frame = self.wrap2pi(state[5] - self.path.angles[orthogonal_projection_id])

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
        self.compute_distance_to_goal(state)
        command_linear_velocity = self.compute_linear_velocity(self.orthogonal_projection_id)
        command_angular_velocity = self.compute_angular_velocity(state, self.orthogonal_projection_id)
        return np.array([command_linear_velocity, command_angular_velocity])

        # TODO: Currently set up as a proportional controller for angular velocity control, investigate the need to use derivative component
