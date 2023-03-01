from norlabcontrollib.controllers.controller_factory import ControllerFactory
from norlabcontrollib.path.path import Path

import numpy as np

controller_factory = ControllerFactory()
controller = controller_factory.load_parameters_from_yaml('test_parameters_smpc.yaml')

test_path_poses = np.load('../traj_a_int.npy')
test_path = Path(test_path_poses)
test_path.compute_metrics(controller.path_look_ahead_distance)

controller.update_path(test_path)

init_state = np.zeros(3)
controller.compute_command_vector(init_state)

print('test')

# controller.compute_command_vector(np.array([0.0, 0.0, 0.0]))

# fig, ax = plt.subplots()
# ax.plot(test_path.angles)
# plt.show()