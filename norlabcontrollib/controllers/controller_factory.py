from norlabcontrollib.controllers.differential_orthogonal_exponential import DifferentialOrthogonalExponential
from norlabcontrollib.controllers.differential_rotation_p import DifferentialRotationP
from norlabcontrollib.controllers.ideal_diff_drive_mpc import IdealDiffDriveMPC
from norlabcontrollib.controllers.pwrtrn_diff_drive_mpc import PwrtrnDiffDriveMPC
from norlabcontrollib.controllers.slip_blr_diff_drive_smpc import SlipBLRDiffDriveSMPC
import yaml


class ControllerFactory:
    def load_parameters_from_yaml(self, yaml_file_path):
        print(yaml_file_path)
        with open(yaml_file_path) as yaml_file:
            yaml_params = yaml.full_load(yaml_file)
            print(yaml_file)
            if yaml_params['controller_name'] == 'DifferentialOrthogonalExponential':
                controller = DifferentialOrthogonalExponential(yaml_params)
            elif yaml_params['controller_name'] == 'DifferentialRotationP':
                controller = DifferentialRotationP(yaml_params)
            elif yaml_params['controller_name'] == 'IdealDiffDriveMPC':
                controller = IdealDiffDriveMPC(yaml_params)
            elif yaml_params['controller_name'] == 'PwrtrnDiffDriveMPC':
                controller = PwrtrnDiffDriveMPC(yaml_params)
            elif yaml_params['controller_name'] == 'SlipBLRDiffDriveSMPC':
                controller = SlipBLRDiffDriveSMPC(yaml_params)
            else:
                raise RuntimeError(f"Undefined controller {yaml_params['controller_name']}, please specify a valid controller name")
            return controller
