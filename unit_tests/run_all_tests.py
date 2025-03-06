import unittest

import unit_tests.test_cases.navigator_tests.NavigationTest as nav
import unit_tests.test_cases.collision_tests.tools_test as tools
import unit_tests.test_cases.direction_map_tests.DirectionMapTests as dm

if __name__ == '__main__':
    # Run this file to run all unit test
    # Other python running methods aren't much more better in my opinion

    nav_1 = nav.TestAngle
    nav_2 = nav.TestDirection
    nav_3 = nav.TestDistance

    tool_1 = tools.ObstacleAdding
    tool_2 = tools.TestMarking

    directmap_1 = dm.GetingAngleTest
    directmap_2 = dm.GettingDirectionTests
    directmap_3 = dm.GettingPosTests
    directmap_4 = dm.GettingStepSizeTests


    unittest.main()
