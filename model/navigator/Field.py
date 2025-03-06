import numpy as np
import heapq
import random
from model.environment.environment_enum import Env
from resources.handling.reading import load_direction_from_file, load_map_from_file
def get_adjacent_cells(cell, grid_size):
    """
    获取给定单元格周围八个方向的单元格坐标
    """
    i, j = cell
    adjacent_cells = [(i-1, j-1), (i-1, j), (i-1, j+1),
                      (i, j-1),             (i, j+1),
                      (i+1, j-1), (i+1, j), (i+1, j+1)]
    return [(x, y) for x, y in adjacent_cells if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]]

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
def astar(array, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 4-directional movement
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []

def generate_direction_array(trajectory, grid_map, expand_steps=1):
    grid_size = grid_map.shape
    direction_array = np.empty(grid_size, dtype=object)
    expansion_counts = np.full((grid_size), None) #标记是第几次被扩展
    traj_set = set(map(tuple, trajectory))  # 将轨迹转换为集合以提高查询效率

    # 如果expand_steps为0，则直接标记轨迹上的网格
    if expand_steps == 0:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                current_pos = (i, j)
                if current_pos in traj_set:
                    indices = np.where((trajectory == current_pos).all(axis=1))[0]
                    if indices.size > 0:
                        index = indices[0]
                        if index < len(trajectory) - 1:
                            next_pos = trajectory[index + 1]
                            direction_array[i, j] = tuple(next_pos)
                            if  expansion_counts[i, j]==None:
                                expansion_counts[i, j] = expand_steps
                        else:
                            direction_array[i, j] = current_pos
                else:
                    if grid_map[i, j] == 1:
                        direction_array[i, j] = Env.OBSTACLE
                    else:
                        direction_array[i, j] = current_pos
    else:
        for step in range(expand_steps):
            newly_marked_cells = set()  # 用于存储本次扩展中新标记的网格

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    current_pos = (i, j)

                    if current_pos in traj_set:
                        indices = np.where((trajectory == current_pos).all(axis=1))[0]
                        if indices.size > 0:
                            index = indices[0]
                            if index < len(trajectory) - 1:
                                next_pos = trajectory[index + 1]
                                direction_array[i, j] = tuple(next_pos)
                                if  expansion_counts[i, j]==None:
                                    expansion_counts[i, j] = step
                            else:
                                direction_array[i, j] = Env.EXIT
                    else:
                        if grid_map[i, j] == 1:
                            direction_array[i, j] = Env.OBSTACLE
                        else:
                            # 找到周围八个方向中能够到达的轨迹经过的网格
                            reachable_cells = [adj_cell for adj_cell in get_adjacent_cells(current_pos, grid_size) if adj_cell in traj_set]
                            if reachable_cells:
                                chosen_cell = random.choice(reachable_cells)
                                direction_array[i, j] = chosen_cell
                                if  expansion_counts[i, j]==None:
                                    expansion_counts[i, j] = step+1
                                newly_marked_cells.add(current_pos)  # 将新标记的网格添加到集合中
                            else:
                                # 如果当前网格没有邻居可达，则指向自身
                                direction_array[i, j] = current_pos

            # 更新轨迹集合，以便下一次扩展
            traj_set.update(map(tuple, newly_marked_cells))
    return direction_array, expansion_counts

def cal_field_ByTra(maze, exits, traj):
    grid_map = np.array(maze)
    trajectory_grid = traj
    trajecory_astar_grid = []
    for point in trajectory_grid:
        # 将世界坐标转换为网格坐标
        grid_x = point[0]
        grid_y = point[1]
        trajecory_astar_grid.append((grid_y, grid_x))
    trajecory_astar_grid.append(exits)

    corrected_path = [trajecory_astar_grid[0]]
    for i in range(len(trajectory_grid) - 1):
        start = trajecory_astar_grid[i]
        goal = trajecory_astar_grid[i + 1]
        path = astar(grid_map, start, goal)
        path = path[::-1]
        # path.append(start)
        if path:
            corrected_path.extend(path)
        else:
            print(f"No path found from {start} to {goal}")
    # 将轨迹标记在地图上
    trajectory_new = []
    for point in corrected_path:
        trajectory_new.append([point[0], point[1]])
    trajectory_new = np.array(trajectory_new)
    expand_steps=30
    direction_array, expansion_counts = generate_direction_array(trajectory_new, grid_map, expand_steps)
    return direction_array

map_filename = "H:\\GRK\\CrowdMovmentSimulation-master\\resources\\ready\\galeria_krakowska_maze100x100.txt"
grid_map = load_map_from_file(map_filename)
trajectory_grid=[(0,20),(20,22),(30,60),(35,80),(40,40),(80,50),(70,80)]
exits=(99,90)
direction_array=cal_field_ByTra(grid_map, exits, trajectory_grid)
print(direction_array)