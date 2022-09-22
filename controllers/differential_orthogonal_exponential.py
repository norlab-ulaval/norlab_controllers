from controllers.controller import Controller
import numpy as np

class DifferentialOrthogonalExponential(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.gain_path_convergence = parameter_map['gain_path_convergence']

    def update_path(self, new_path):
        self.path = new_path

    def compute_command_vector(self, state):
        orthogonal_projection_dist, orthogonal_projection_id = self.path.compute_orthogonal_projection(state)

        target_exponential_tangent_angle = np.arctan(-self.gain_path_convergence * orthogonal_projection_dist)

        error_angle = state[2] - self.path.angles[orthogonal_projection_id]

        target_rotation = target_exponential_tangent_angle + error_angle



        # TODO: Figure out transformation between cmd.rotation and cmd.rotational_velocity in gerona (is there one?)

        #TODO: validate orthogonal projection computation
    