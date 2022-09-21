from controllers.controller import Controller

class DifferentialOrthogonalExponential(Controller):
    def __int__(self, parameter_map):
        super(DifferentialOrthogonalExponential, self).__init__(parameter_map)
        self.parameter_map = parameter_map

    def compute_command_vector(self, state):

