from abc import ABCMeta, abstractmethod
import numpy as np

class Controller(metaclass=ABCMeta):
    def __init__(self, parameter_map):
        self.rate = parameter_map['rate']
        self.minimum_linear_velocity = parameter_map['minimum_linear_velocity']
        self.maximum_linear_velocity = parameter_map['maximum_linear_velocity']
        self.maximum_angular_velocity = parameter_map['maximum_angular_velocity']
        self.linear_goal_tolerance = parameter_map['linear_goal_tolerance']
        self.angular_goal_tolerance = parameter_map["angular_goal_tolerance"]
        # For dynamic [parameter]
        self.function_to_re_init = False
        self.param_that_start_init = []
    @abstractmethod
    def compute_command_vector(self, state):
        pass

    def update_path(self, new_path):
        self.path = new_path
        self.path.compute_metrics()
        pass
