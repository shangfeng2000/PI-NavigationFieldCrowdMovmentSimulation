import unittest

import numpy as np

from model.direction_map.DirectionMap import DirectionMap
import model.navigator.navigator as nav


class GettingPosTests(unittest.TestCase):

    def setUp(self):
        self.directions = np.zeros((100, 100)).tolist()

    def test_geting_next_position_1(self):
        self.directions[2][5] = (12, 12)

        direction_map = DirectionMap(self.directions)
        self.assertEqual((12, 12), direction_map.get_next_position((2, 5)))

    def test_geting_next_position_2(self):
        self.directions[30][70] = (12, 12)

        direction_map = DirectionMap(self.directions)
        self.assertEqual((12, 12), direction_map.get_next_position((30, 70)))

    def test_return_diffrent_values(self):
        self.directions[2][5] = (12, 12)

        direction_map = DirectionMap(self.directions)
        self.assertNotEqual((12, 12), direction_map.get_next_position((3, 5)))


class GettingDirectionTests(unittest.TestCase):

    def setUp(self):
        self.directions = np.zeros((100, 100)).tolist()

    def test_geting_direction(self):
        self.directions[2][5] = (12, 12)
        direction_map = DirectionMap(self.directions)

        direction = nav.get_direction_to_another_point((2, 5), (12, 12))
        self.assertEqual(direction, direction_map.get_direction((2, 5)))


class GettingStepSizeTests(unittest.TestCase):

    def setUp(self):
        self.directions = np.zeros((100, 100)).tolist()

    def test_geting_step_size(self):
        self.directions[2][5] = (12, 12)
        direction_map = DirectionMap(self.directions)

        step = nav.get_distance_beteween_points((2, 5), (12, 12))
        self.assertEqual(step, direction_map.get_step_size((2, 5)))


class GetingAngleTest(unittest.TestCase):

    def setUp(self):
        self.directions = np.zeros((100, 100)).tolist()

    def test_geting_angle(self):
        self.directions[2][5] = (12, 12)
        direction_map = DirectionMap(self.directions)

        angle = nav.get_angle_of_direction_between_points((2, 5), (12, 12))
        self.assertEqual(angle, direction_map.get_angle((2, 5)))
