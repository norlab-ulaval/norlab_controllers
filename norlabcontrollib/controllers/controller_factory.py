import yaml

from norlabcontrollib.controllers import (
    DifferentialOrthogonalExponential,
    DifferentialRotationP,
    IdealDiffDriveMPC,
)


class ControllerFactory:
    def load_parameters_from_yaml(self, yaml_file_path):
        with open(yaml_file_path) as yaml_file:
            yaml_params = yaml.full_load(yaml_file)
            controller_name = yaml_params.get("controller_name", None)
            if controller_name is None:
                raise RuntimeError(
                    f"controller name was not specified in param file {yaml_file_path}"
                )
            if controller_name == "DifferentialOrthogonalExponential":
                controller = DifferentialOrthogonalExponential(yaml_params)
            elif controller_name == "DifferentialRotationP":
                controller = DifferentialRotationP(yaml_params)
            elif controller_name == "IdealDiffDriveMPC":
                controller = IdealDiffDriveMPC(yaml_params)
            else:
                raise RuntimeError(
                    "Undefined controller, please specify a valid controller name"
                )
            return controller
