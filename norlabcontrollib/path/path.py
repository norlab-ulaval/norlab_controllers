import numpy as np
from math import ceil, floor
from norlabcontrollib.util.util_func import wrap2pi


class Path:
    def __init__(self, poses):
        self.going_forward = True
        self.poses = poses
        self.planar_poses = np.zeros((poses.shape[0], 3))
        self.planar_poses[:, 0] = poses[:, 0]
        self.planar_poses[:, 1] = poses[:, 1]
        self.n_poses = self.poses.shape[0]

        self.curvatures = np.zeros(self.n_poses)
        self.look_ahead_curvatures = np.zeros(self.n_poses)
        self.distances_to_goal = np.zeros(self.n_poses)
        self.angular_distances_to_goal = np.zeros(self.n_poses)
        self.angles = poses[:, 5]
        self.angles_spatial_window = 0.25
        self.world_to_path_tfs_array = np.ndarray((self.n_poses, 3, 3))
        self.path_to_world_tfs_array = np.ndarray((self.n_poses, 3, 3))

    def compute_curvatures(self):
        first_derivative_x = np.zeros(self.n_poses)
        first_derivative_y = np.zeros(self.n_poses)
        for i in range(1, self.n_poses - 1):
            backward_step_size = np.linalg.norm(
                self.poses[i, 0:2] - self.poses[i - 1, 0:2]
            )
            forward_step_size = np.linalg.norm(
                self.poses[i + 1, 0:2] - self.poses[i, 0:2]
            )
            first_derivative_x[i] = (
                backward_step_size**2 * self.poses[i - 1, 0]
                + (forward_step_size**2 - backward_step_size**2) * self.poses[i, 0]
                - forward_step_size**2 * self.poses[i + 1, 0]
            ) / (
                backward_step_size
                * forward_step_size
                * (backward_step_size + forward_step_size)
            )
            first_derivative_y[i] = (
                backward_step_size**2 * self.poses[i - 1, 1]
                + (forward_step_size**2 - backward_step_size**2) * self.poses[i, 1]
                - forward_step_size**2 * self.poses[i + 1, 1]
            ) / (
                backward_step_size
                * forward_step_size
                * (backward_step_size + forward_step_size)
            )

        second_derivative_x = np.zeros(self.n_poses)
        second_derivative_y = np.zeros(self.n_poses)
        for i in range(1, self.n_poses - 1):
            backward_step_size = np.linalg.norm(
                self.poses[i, 0:2] - self.poses[i - 1, 0:2]
            )
            forward_step_size = np.linalg.norm(
                self.poses[i + 1, 0:2] - self.poses[i, 0:2]
            )
            second_derivative_x[i] = (
                backward_step_size**2 * first_derivative_x[i - 1]
                + (forward_step_size**2 - backward_step_size**2) * first_derivative_x[i]
                - forward_step_size**2 * first_derivative_x[i + 1]
            ) / (
                backward_step_size
                * forward_step_size
                * (backward_step_size + forward_step_size)
            )
            second_derivative_y[i] = (
                backward_step_size**2 * first_derivative_y[i - 1]
                + (forward_step_size**2 - backward_step_size**2) * first_derivative_y[i]
                - forward_step_size**2 * first_derivative_y[i + 1]
            ) / (
                backward_step_size
                * forward_step_size
                * (backward_step_size + forward_step_size)
            )

        for i in range(0, self.n_poses):
            self.curvatures[i] = np.sqrt(
                second_derivative_x[i] ** 2 + second_derivative_y[i] ** 2
            )

        # curvatures_list = np.gradient(self.poses[:,:2])
        # self.curvatures = np.sqrt(np.square(curvatures_list[0]) + np.square(curvatures_list[1]))

    def compute_look_ahead_curvatures(self, look_ahead_distance=1.0):
        self.look_ahead_distance_counter_array = np.zeros(self.n_poses)
        for i in range(0, self.n_poses - 1):
            path_iterator = 0
            look_ahead_distance_counter = 0
            path_curvature_sum = 0
            while look_ahead_distance_counter <= look_ahead_distance:
                if i + path_iterator + 1 == self.n_poses:
                    break
                path_curvature_sum += np.abs(self.curvatures[i + path_iterator])
                look_ahead_distance_counter += np.abs(
                    self.distances_to_goal[i + path_iterator]
                    - self.distances_to_goal[i + path_iterator + 1]
                )
                path_iterator += 1
            self.look_ahead_curvatures[i] = path_curvature_sum
            self.look_ahead_distance_counter_array[i] = look_ahead_distance_counter

    def compute_distances_to_goal(self):
        distance_to_goal = 0
        angular_distance_to_goal = 0

        for i in range(self.n_poses - 1, 0, -1):
            distance_to_goal += np.linalg.norm(
                self.poses[i, :2] - self.poses[i - 1, :2]
            )
            angular_distance_to_goal = wrap2pi(angular_distance_to_goal + self.poses[i, 5] - self.poses[i - 1, 5])
            self.distances_to_goal[i - 1] = distance_to_goal
            self.angular_distances_to_goal[i-1] =  angular_distance_to_goal

    def compute_angles(self):
        distance_counter = 0
        for i in range(0, self.n_poses - 1):
            j = i
            while distance_counter <= self.angles_spatial_window:
                if j == self.n_poses - 1:
                    self.angles[i:] = self.angles[i - 1]
                    break
                j += 1
                distance_counter = self.distances_to_goal[i] - self.distances_to_goal[j]
            self.angles[i] = np.arctan2(
                self.poses[j, 1] - self.poses[i, 1], self.poses[j, 0] - self.poses[i, 0]
            )
            self.planar_poses[i, 2] = self.angles[i]
            distance_counter = 0

    def compute_world_to_path_frame_tfs(self):
        path_to_world_tf = np.eye(3)
        for i in range(0, self.n_poses):
            path_to_world_tf[0, 0] = np.cos(self.angles[i])
            path_to_world_tf[0, 1] = -np.sin(self.angles[i])
            path_to_world_tf[0, 2] = self.poses[i, 0]
            path_to_world_tf[1, 0] = np.sin(self.angles[i])
            path_to_world_tf[1, 1] = np.cos(self.angles[i])
            path_to_world_tf[1, 2] = self.poses[i, 1]
            self.path_to_world_tfs_array[i, :, :] = path_to_world_tf
            self.world_to_path_tfs_array[i, :, :] = np.linalg.inv(path_to_world_tf)

    def compute_metrics(self):
        self.compute_distances_to_goal()
        # self.compute_curvatures()
        # self.compute_look_ahead_curvatures(path_look_ahead_distance)
        # self.compute_angles()     # Angles taken from WILN
        self.compute_world_to_path_frame_tfs()
        return None
    

    def compute_orthogonal_projection(self, pose, last_id, window_size, linear_speed, angular_speed):
        
        first_idx = floor(max(0, last_id - window_size / 2))
        last_idx = ceil(min(self.n_poses, last_id + window_size / 2))
        window_points = self.planar_poses[first_idx:last_idx]
        window_angles = self.angles[first_idx:last_idx]

        min_distance = float("inf")
        closest_projection = None
        next_idx = 0

        for i in range(len(window_points) - 1):
            a = np.array(window_points[i])[:2]
            b = np.array(window_points[i + 1])[:2]
            projection = self.project_point_onto_line_segment(pose[:2], a, b)
            distance_linear = np.linalg.norm(pose[:2] - projection)
            distance_angular = np.abs(wrap2pi(pose[2] - window_angles[i]))
            distance_time = distance_linear/linear_speed + distance_angular/angular_speed
            if distance_time < min_distance:
                min_distance = distance_time
                closest_projection = projection
                next_idx = first_idx + i + 1

        closest_pose = np.array([closest_projection[0], closest_projection[1], pose[2]])
        return closest_pose, next_idx


    def project_point_onto_line_segment(self, p, a, b):
        """Project point p onto line segment ab.
        Args:
        - p (np.array): The point to project.
        - a (np.array): The start point of the line segment.
        - b (np.array): The end point of the line segment.

        Returns:
        - np.array: The projected point on the line segment.
        """
        ap = p - a
        ab = b - a
        ab_norm = np.dot(ab, ab)
        if ab_norm == 0:
            return a  # a and b are the same point
        t = np.dot(ap, ab) / ab_norm
        t = np.clip(t, 0, 1)
        projection = a + t * ab
        return projection


    def compute_horizon(self, initial_pose, start_idx, horizon_duration, linear_speed, angular_speed):

        horizon_poses = [initial_pose]
        stop_idx = start_idx
        cumul_duration = [0.0]
        while cumul_duration[-1] < horizon_duration and stop_idx < self.n_poses:
            linear_error = np.linalg.norm(self.poses[stop_idx, :2] - horizon_poses[-1][:2])
            angular_error = np.abs(wrap2pi(self.poses[stop_idx, 5] - horizon_poses[-1][2]))
            travel_time = max(linear_error / linear_speed, angular_error / angular_speed)
            cumul_duration.append(cumul_duration[-1] + travel_time)
            horizon_poses.append(
                np.array([
                        self.poses[stop_idx, 0],
                        self.poses[stop_idx, 1],
                        self.angles[stop_idx],
                ])
            )
            stop_idx += 1

        return np.array(horizon_poses), cumul_duration
