import numpy


from .line import Point


def astar(maze, start, end):
    """Returns a list of points in fastest path"""


    # Change points to tuples, since it works faster
    if isinstance(start, Point) and isinstance(end, Point):
        start = (start.y, start.x)
        end = (end.y, end.x)

    # Create start and end node
    if maze[start[0]][start[1]] != 0:
        return []
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            for index, ele in enumerate(path):
                path[index] = Point(ele[1], ele[0])
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

            # Create the f, g, and h values
            new_node.g = current_node.g + 1
            new_node.h = diagonal_distance_heuristics(Point(new_node.position[1], new_node.position[0]),
                                                      Point(end[1], end[0]))
            new_node.f = new_node.g + new_node.h

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:
                continue

            # Child is already in the open list
            skip = False
            for index, open_node in enumerate(open_list):
                if open_node.position == child.position:
                    if open_node.f > child.f or (child.f == open_node.f and child.h < open_node.f):
                        open_list.pop(index)
                        break
                    else:
                        skip = True

            if skip is True:
                continue

            # Add the child to the open list
            open_list.append(child)
    return []


def diagonal_distance_heuristics(current, end):  # (Point, Point)
    """ we have to use different heuristic since we can move just in 8 direction, more info:
    http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance"""

    # d = 1
    d2 = numpy.sqrt(2)
    dx = numpy.abs(end.x - current.x)
    dy = numpy.abs(end.y - current.y)

    if dx > dy:
        return (dx - dy) + d2 * dy
    else:
        return (dy - dx) + d2 * dx


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return str(self.position) + "f:" + str(self.f)
