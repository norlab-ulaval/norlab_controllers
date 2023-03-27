import numpy as np
from scipy.spatial import KDTree


class Path:
    def __init__(self, poses):
        self.going_forward = True
        self.poses = poses
        self.n_poses = self.poses.shape[0]
        self.pose_kdtree = KDTree(poses[:, :2])

        self.curvatures = np.zeros(self.n_poses)
        self.look_ahead_curvatures = np.zeros(self.n_poses)
        self.distances_to_goal = np.zeros(self.n_poses)
        self.angles = np.zeros(self.n_poses)
        self.angles_spatial_window = 0.25
        self.world_to_path_tfs_array = np.ndarray((self.n_poses,3,3))
    def compute_curvatures(self):
        first_derivative_x = np.zeros(self.n_poses)
        first_derivative_y = np.zeros(self.n_poses)
        for i in range(1, self.n_poses - 1):
            backward_step_size = np.linalg.norm(self.poses[i, 0:2] - self.poses[i - 1, 0:2])
            forward_step_size = np.linalg.norm(self.poses[i + 1, 0:2] - self.poses[i, 0:2])
            first_derivative_x[i] = (backward_step_size ** 2 * self.poses[i - 1, 0] +
                                     (forward_step_size ** 2 - backward_step_size ** 2) * self.poses[i, 0] -
                                     forward_step_size ** 2 * self.poses[i + 1, 0]) / \
                                    (backward_step_size * forward_step_size * (backward_step_size + forward_step_size))
            first_derivative_y[i] = (backward_step_size ** 2 * self.poses[i - 1, 1] +
                                     (forward_step_size ** 2 - backward_step_size ** 2) * self.poses[i, 1] -
                                     forward_step_size ** 2 * self.poses[i + 1, 1]) / \
                                    (backward_step_size * forward_step_size * (backward_step_size + forward_step_size))

        second_derivative_x = np.zeros(self.n_poses)
        second_derivative_y = np.zeros(self.n_poses)
        for i in range(1, self.n_poses - 1):
            backward_step_size = np.linalg.norm(self.poses[i, 0:2] - self.poses[i - 1, 0:2])
            forward_step_size = np.linalg.norm(self.poses[i + 1, 0:2] - self.poses[i, 0:2])
            second_derivative_x[i] = (backward_step_size ** 2 * first_derivative_x[i - 1] +
                                      (forward_step_size ** 2 - backward_step_size ** 2) * first_derivative_x[i] -
                                      forward_step_size ** 2 * first_derivative_x[i + 1]) / \
                                     (backward_step_size * forward_step_size * (backward_step_size + forward_step_size))
            second_derivative_y[i] = (backward_step_size ** 2 * first_derivative_y[i - 1] +
                                      (forward_step_size ** 2 - backward_step_size ** 2) * first_derivative_y[i] -
                                      forward_step_size ** 2 * first_derivative_y[i + 1]) / \
                                     (backward_step_size * forward_step_size * (backward_step_size + forward_step_size))

        for i in range(0, self.n_poses):
            self.curvatures[i] = np.sqrt(second_derivative_x[i] ** 2 + second_derivative_y[i] ** 2)

        # curvatures_list = np.gradient(self.poses[:,:2])
        # self.curvatures = np.sqrt(np.square(curvatures_list[0]) + np.square(curvatures_list[1]))

    def compute_look_ahead_curvatures(self, look_ahead_distance=1.0):
        self.look_ahead_distance_counter_array = np.zeros(self.n_poses)
        for i in range(0, self.n_poses-1):
            path_iterator = 0
            look_ahead_distance_counter = 0
            path_curvature_sum = 0
            while look_ahead_distance_counter <= look_ahead_distance:
                if i + path_iterator + 1 == self.n_poses:
                    break
                path_curvature_sum += np.abs(self.curvatures[i + path_iterator])
                look_ahead_distance_counter += np.abs(self.distances_to_goal[i + path_iterator] - \
                                               self.distances_to_goal[i + path_iterator + 1])
                path_iterator += 1
            self.look_ahead_curvatures[i] = path_curvature_sum
            self.look_ahead_distance_counter_array[i] = look_ahead_distance_counter
    def compute_distances_to_goal(self):
        distance_to_goal = 0
        for i in range(self.n_poses-1, 0, -1):
            distance_to_goal += np.linalg.norm(self.poses[i, :2] - self.poses[i-1, :2])
            self.distances_to_goal[i-1] = distance_to_goal

    def compute_angles(self):
        distance_counter = 0
        for i in range(0, self.n_poses-1):
            j = i
            while distance_counter <= self.angles_spatial_window:
                if j == self.n_poses - 1:
                    self.angles[i:] = self.angles[i-1]
                    break
                j += 1
                distance_counter = self.distances_to_goal[i] - self.distances_to_goal[j]
            self.angles[i] = np.arctan2(self.poses[j, 1] - self.poses[i, 1], self.poses[j, 0] - self.poses[i, 0])
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
            self.world_to_path_tfs_array[i, :, :] = np.linalg.inv(path_to_world_tf)

    def compute_metrics(self, path_look_ahead_distance):
        self.compute_distances_to_goal()
        self.compute_curvatures()
        self.compute_look_ahead_curvatures(path_look_ahead_distance)
        self.compute_angles()
        self.compute_world_to_path_frame_tfs()
        return None

    def compute_orthogonal_projection(self, pose):
        orthogonal_projection_dist, orthogonal_projection_id = self.pose_kdtree.query(pose[:2])
        return orthogonal_projection_dist, orthogonal_projection_id

# TODO: find a way to split path into multiple directional paths to switch robot direction
