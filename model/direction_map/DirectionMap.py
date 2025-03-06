import numpy as np
from model.environment.environment_enum import Env
import model.navigator.navigator as nav


class DirectionMap:

    def __init__(self, direction_map: [[(int, int)]]):
        self.direction_map = direction_map

    def get_next_position(self, current_pos: (int, int)):
        (current_y, current_x) = current_pos
        point = self.direction_map[current_y][current_x]
        return point

    def get_direction(self, current_pos: (int, int)):
        return nav.get_direction_to_another_point(current_pos, self.get_next_position(current_pos))

    def get_step_size(self, current_pos: (int, int)):
        return nav.get_distance_beteween_points(current_pos, self.get_next_position(current_pos))

    def get_angle(self, current_pos: (int, int)):
        (desired_y, desired_x) = self.get_direction(current_pos)
        return np.arctan2(desired_y, desired_x)
