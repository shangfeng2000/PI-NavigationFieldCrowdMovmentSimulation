import copy
import numpy
import matplotlib.pyplot as plt
from .environment_enum import Env
from .line import Point, Line

from .a_star import astar, diagonal_distance_heuristics


def direction_map(environment, exit_points , step_size, quality_direction_map=False):
    # (Point,Env)[][] / (int[][], Point[], int, boolean)
    """ Return direction map with adjusted step_size"""

    # Mark map with None field or if 1 mark with obstacle
    mapped_environment = copy.deepcopy(environment)
    for i in range(0, len(mapped_environment)):
        for j in range(0, len(mapped_environment[i])):
            if mapped_environment[i][j] == 1:
                mapped_environment[i][j] = Env.OBSTACLE
            else:
                mapped_environment[i][j] = None

    # For step_size 1 or 2 non quality map is same as quality
    if quality_direction_map is True and step_size <= 2:
        quality_direction_map = False

    # We need to know where obstacles are if we want quality map
    obstacles = []
    if quality_direction_map is True:
        obstacles = get_obstacle_line_horizon(environment) + get_obstacle_line_vertical(environment)

    # Loop till all the field are not None
    for y in range(0, len(environment)):
        for x in range(0, len(environment[y])):
            if mapped_environment[y][x] is None:
                # Instead of counting a_star for each exit point and then choosing shortest path
                # I find exit point with shortest heuristic distance and then for this point counting a_star
                closest_point = exit_points[0]
                # shortest_distance = diagonal_distance_heuristics(Point(x, y), closest_point)
                # for point in exit_points:
                #     distance = diagonal_distance_heuristics(Point(x,y), point)
                #     if distance < shortest_distance:
                #         shortest_distance = distance
                #         closest_point = point
                shortest_path = astar(environment, Point(x, y), closest_point) #根据直线距离启发式选择一个出口点，A星算法规划路径
                # Looping through each point in shortest path
                for i in range(0, len(shortest_path)):

                    current_x = shortest_path[i].x
                    current_y = shortest_path[i].y

                    #  Check if it is pointless to continue mapping
                    if mapped_environment[current_y][current_x] is not None:
                        break

                    if quality_direction_map is False:

                        # If it is last point we mark it with EXIT
                        if i == len(shortest_path) - 1:
                            mapped_environment[current_y][current_x] = Env.EXIT
                        else:
                            # Check if next step is over exit
                            if i+step_size >= len(shortest_path)-1:
                                mapped_environment[current_y][current_x] = shortest_path[len(shortest_path)-1].y,shortest_path[len(shortest_path)-1].x
                            else:
                                mapped_environment[current_y][current_x] = shortest_path[i+step_size].y,shortest_path[i+step_size].x
                    else:
                        # If it is last point we mark it with EXIT
                        if i == len(shortest_path)-1:
                            mapped_environment[current_y][current_x] = Env.EXIT
                        else:
                            possible_step = step_size
                            possible_step_is_correct = True
                            while possible_step >= 1:
                                # Check if next step is not over the exit，如果超过则下一个移动位置为路径中最后一个网格
                                if i+possible_step < len(shortest_path):
                                    point_to_go = Point(shortest_path[i + possible_step].x, shortest_path[i + possible_step].y)
                                else:
                                    last_index = len(shortest_path)-1
                                    point_to_go = Point(shortest_path[last_index].x, shortest_path[last_index].y)
                                    possible_step = last_index - i

                                # Check if we don't go through obstacle
                                line = Line(Point(current_x, current_y), point_to_go)
                                for line_obstacle in obstacles:
                                    if line.intersect(line_obstacle):
                                        possible_step -= 1
                                        possible_step_is_correct = False
                                        break
                                    else:
                                        possible_step_is_correct = True

                                # if we don't go through obstacle and over the exit we accept this point
                                if possible_step_is_correct is True:
                                    mapped_environment[current_y][current_x] = (point_to_go.y, point_to_go.x)

                                    break
    return mapped_environment


def get_obstacle_line_vertical(environment):  # Line[] / int[][]
    is_line_started = False
    lines = []
    line_vertical = Line(None, None)
    for i in range(0, len(environment[0])):
        if is_line_started is True:
            lines.append(line_vertical)
            is_line_started = False
        line_vertical = Line(None, None)
        for j in range(0, len(environment)):
            if environment[j][i] == 1 and is_line_started is False:
                is_line_started = True
                line_vertical.point_start = Point(i, j)
                line_vertical.point_end = Point(i, j)
            elif environment[j][i] == 1 and is_line_started is True:
                line_vertical.point_end = Point(i, j)
            elif environment[j][i] == 0 and is_line_started is True:
                lines.append(line_vertical)
                is_line_started = False
    return lines #包含垂直方向上的障碍线段


def get_obstacle_line_horizon(environment):  # Line[] / int[][]
    is_line_started = False
    lines = []
    line_horizon = Point(None, None)
    for i in range(0, len(environment)):
        if is_line_started is True:
            lines.append(line_horizon)
            is_line_started = False
        line_horizon = Line(None, None)
        for j in range(0, len(environment[i])):
            if environment[i][j] == 1 and is_line_started is False:
                line_horizon.point_start = Point(j, i)
                line_horizon.point_end = Point(j, i)
                is_line_started = True
            elif environment[i][j] == 1 and is_line_started is True:
                line_horizon.point_end = Point(j, i)
            elif environment[i][j] == 0 and is_line_started is True:
                lines.append(line_horizon)
                is_line_started = False
    return lines #水平障碍线段


def path_distance(path):
    distance = 0
    for i in range(1, len(path)):
        if path[i].x != path[i-1].x and path[i].y != path[i-1].y:
            distance += numpy.sqrt(2)
        else:
            distance += 1
    return distance
