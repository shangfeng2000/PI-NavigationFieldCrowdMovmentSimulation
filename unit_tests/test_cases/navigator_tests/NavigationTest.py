import unittest
import model.navigator.navigator as nav
import numpy as np


class TestDirection(unittest.TestCase):

#Using (y,x) notation

    def test_horizonta_direction(self):
        self.assertEqual(nav.get_direction_to_another_point((3, 0), (5, 0)), (2, 0), "Horizontal direction")

    def test_horizontal_direction(self):
        self.assertEqual(nav.get_direction_to_another_point((0, 10), (0, 3)), (0, -7), "Vertical direction")

    def test_cross_direction(self):
        self.assertEqual(nav.get_direction_to_another_point((2, 2), (5, 4)), (3, 2), "Cross direction")


class TestDistance(unittest.TestCase):

    def test_distane(self):
        self.assertEqual(nav.get_distance_beteween_points((0, 0), (3, 4)), 5, "Distance between points")

    def test_distane_cross(self):
        self.assertEqual(nav.get_distance_beteween_points((2, 2), (3, 7)), (1 ** 2 + 5 ** 2) ** (1 / 2),
                         "Distance between points")


class TestAngle(unittest.TestCase):
    def test_angle_from_orgin(self):
        self.assertEqual(nav.get_angle_of_direction_between_points((0, 0), (1, 1)), np.pi / 4,
                         "Angle of line orgin--(1,1)=45dgr")

    def test_vertical_angle(self):
        self.assertEqual(nav.get_angle_of_direction_between_points((0, 0), (4, 0)), np.pi / 2,
                         "Angle of line orgin--(4,1)=45dgr")

    def test_angle_ablove_90dgrs(self):
        self.assertEqual(nav.get_angle_of_direction_between_points((0, 0), (1, -1)), np.pi * (3 / 4),
                         "Angle of line orgin--(1,-1)=(90+45)dgr")

    def test_angle_ablove_110dgrs(self):
        self.assertAlmostEqual(nav.get_angle_of_direction_between_points((0, 0), ( 2 * 3 ** (1 / 2) / 2,-1)), np.deg2rad(120), 12,
                         "Angle of line")


if __name__ == '__main__':
    unittest.main()
