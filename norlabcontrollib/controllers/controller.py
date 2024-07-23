from abc import ABCMeta, abstractmethod
import numpy as np

class Controller(metaclass=ABCMeta):
    def __init__(self, parameter_map):
        self.rate = parameter_map['rate']
        self.minimum_linear_velocity = parameter_map['minimum_linear_velocity']
        self.maximum_linear_velocity = parameter_map['maximum_linear_velocity']
        self.maximum_angular_velocity = parameter_map['maximum_angular_velocity']
        self.goal_tolerance = parameter_map['goal_tolerance']

    @abstractmethod
    def compute_command_vector(self, state):
        pass
    
    @abstractmethod
    def get_next_command(self):
        pass

    def update_path(self, new_path):
        self.path = new_path
        pass
