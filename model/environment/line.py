from enum import Enum


class Line:

    def __init__(self, point_s, point_e):
        self.point_start = point_s
        self.point_end = point_e

    def __repr__(self):
        return "(" + str(self.point_start) + ", " + str(self.point_end) + ")"

    def intersect(self, line):
        """ Returns true if two line intersect"""
        # General case
        o1 = orientation(self.point_start, self.point_end, line.point_start)
        o2 = orientation(self.point_start, self.point_end, line.point_end)
        # Special case
        o3 = orientation(line.point_start, line.point_end, self.point_start)
        o4 = orientation(line.point_start, line.point_end, self.point_end)

        # General case
        if o1 != o2 and o3 != o4: #一个顺时针，一个逆时针则说明两者相交
            return True

        # Special case
        if o1 == Intersect.COLLINEAR and is_between(self.point_start, line.point_start, self.point_end):
            return True
        elif o2 == Intersect.COLLINEAR and is_between(self.point_start, line.point_end, self.point_end):
            return True
        elif o3 == Intersect.COLLINEAR and is_between(line.point_start, self.point_start, line.point_end):
            return True
        elif o4 == Intersect.COLLINEAR and is_between(line.point_start, self.point_end, line.point_end):
            return True
        else:
            return False


def is_between(p1, p2, p3):
    if min(p1.x, p3.x) <= p2.x <= max(p1.x, p3.x) and min(p1.y, p3.y) <= p2.y <= max(p1.y, p3.y):
        return True
    else:
        return False


def orientation(p1, p2, p3):
    value = (p2.y - p1.y) * (p3.x - p2.x) - (p2.x - p1.x) * (p3.y - p2.y) #斜率公式

    if value == 0:
        return Intersect.COLLINEAR

    if value > 0:
        return Intersect.CLOCKWISE
    else:
        return Intersect.COUNTERCLOCKWISE


class Intersect(Enum):
    COLLINEAR = 0 #共线
    CLOCKWISE = 1 #顺时针
    COUNTERCLOCKWISE = 2 #逆时针


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y)

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
