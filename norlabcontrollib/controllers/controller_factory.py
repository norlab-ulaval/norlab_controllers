from norlabcontrollib.controllers.differential_orthogonal_exponential import DifferentialOrthogonalExponential
from norlabcontrollib.controllers.stochastic_linear_mpc import StochasticLinearMPC
import yaml

class ControllerFactory:
    def load_parameters_from_yaml(self, yaml_file_path):
        with open(yaml_file_path) as yaml_file:
            yaml_params = yaml.full_load(yaml_file)

            if yaml_params['controller_name'] == 'DifferentialOrthogonalExponential':
                controller = DifferentialOrthogonalExponential(yaml_params)
            elif yaml_params['controller_name'] == 'StochasticLinearMPC':
                controller = StochasticLinearMPC(yaml_params)

            else:
                raise RuntimeError("Undefined controller, please specify a valid controller name")
            return controller
