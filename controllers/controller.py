from abc import ABCMeta, abstractmethod

class Controller(metaclass=ABCMeta):
    def __init__(self, parameter_map):
        self.rate = parameter_map['rate']
        self.minimum_linear_velocity = parameter_map['minimum_linear_velocity']
        self.maximum_linear_velocity = parameter_map['maximum_linear_velocity']
        self.maximum_angular_velocity = parameter_map['maximum_angular_velocity']

    @abstractmethod
    def compute_command_vector(self, state):
        pass

    def update_path(self, new_path):
        self.path = new_path
        pass
