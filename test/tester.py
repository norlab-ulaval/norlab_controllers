from norlabcontrollib.controllers.controller_factory import ControllerFactory
from norlabcontrollib.path.path import Path

import numpy as np

controller_factory = ControllerFactory()
controller = controller_factory.load_parameters_from_yaml('test_parameters.yaml')

test_path_poses = np.load('../traj_a_int.npy')
test_path = Path(test_path_poses)
test_path.compute_distances_to_goal()
test_path.compute_curvatures()
test_path.compute_look_ahead_curvatures(look_ahead_distance=1.0)
test_path.compute_angles()

controller.update_path(test_path)

controller.compute_command_vector(np.array([0.0, 0.0, 0.0]))

# fig, ax = plt.subplots()
# ax.plot(test_path.angles)
# plt.show()