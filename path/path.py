import numpy as np

class Path:
    def __init__(self, poses):
        self.poses = poses

    def linear_interpolation(self):
        

    def compute_orthogonal_projection(self, pose):
        pass

    def compute_curvature(self, look_ahead_distance, pose):
        self.compute_orthogonal_projection()