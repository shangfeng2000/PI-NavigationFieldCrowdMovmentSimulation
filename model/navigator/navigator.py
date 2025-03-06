import numpy as np


def get_direction_to_another_point(pos_1: (int, int), pos_2: (int, int)):
    return tuple(np.subtract(pos_2,pos_1))


def get_distance_beteween_points(pos_1: (int, int), pos_2: (int, int)):
    (desired_y, desired_x) = get_direction_to_another_point(pos_1, pos_2)
    return (desired_x ** 2 + desired_y ** 2) ** 0.5


def get_angle_of_direction_between_points(pos_1: (int, int), pos_2: (int, int)):
    (desired_y, desired_x) = get_direction_to_another_point(pos_1, pos_2)
    return np.arctan2(desired_y ,desired_x)
