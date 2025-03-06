import os
import re
from model.environment.environment_enum import Env


def get_current_working_dir():
    print(os.path.dirname(os.path.realpath(__file__)))


def load_map_from_file(map_filename: str):
    with open(map_filename) as text_file:
        maze = [[int(x) for x in line.split()] for line in text_file]
        return maze



def load_direction_from_file(directions_filename: str):
    with open(directions_filename) as text_file:
        content = text_file.readlines()
    direction = []
    for line in content:
        splitted = re.split("\) \(|\) |\(| |, |\n", line)
        splitted = splitted[0::]
        splitted = splitted[:-2 or None]
        points = []
        pos_x = -1
        candidate = False
        for x in splitted:
            if candidate:
                candidate = False
                points.append((pos_x, int(x)))
            elif x == '':
                continue
            elif x == 'obstacle':
                points.append(Env.OBSTACLE)
            elif x == 'exit':
                points.append(Env.EXIT)
            else:  # x is a number
                candidate = True
                pos_x = int(x)
        direction.append(points)
    return direction
