from copy import deepcopy
from random import randint

import model.navigator.navigator as nav
from model.agent.Agent import ExitReached
from model.environment.environment_enum import Env


class Agent:
    id = 0

    def __init__(self, id: int, direct_id: int, start_position: (int, int), exits: [(int, int)], gradient_maps, astar_map,
                 collision_map: [[(int, int)]], velocity, density_map,bound_size=2):
        self.id = id
        self.direct_id = direct_id
        self.start = start_position #起始网格
        self.exits = exits # 终点网格
        self.velocity = velocity
        self.current_pos = self.start
        self.front_collision_size = bound_size
        self.direction_map = gradient_maps
        self.astar_map = astar_map
        self.use_direction = gradient_maps
        self.facing_angle = nav.get_angle_of_direction_between_points(self.current_pos, exits[0])

        self.all_gradients = astar_map

        self.value_threshold = 4#控制受人群密度影响状态的拥堵值
        self.value = self.value_threshold

        self.gradient_space_size = 4
        self.move_counts = 0
        #更新行人密度场
        self.update_gradient(self.value,density_map,collision_map)
        self.block_point(start_position,collision_map)
        self.anger = 0

    def update_facing_angle(self, new_pos):
        self.facing_angle = nav.get_angle_of_direction_between_points(self.current_pos, new_pos)

    def get_available_moves(self,density_map,collision_map):
        available_spots = []

        for x in range(self.current_pos[0]-1, self.current_pos[0]+2):
            for y in range(self.current_pos[1]-1, self.current_pos[1]+2):

                # If this point is current point we skip
                if y == self.current_pos[1] and x == self.current_pos[0]:
                    continue

                # If we out of range we skip
                if y >= len(collision_map) or x >= len(collision_map[0]) or \
                        y < 0 or x < 0:
                    continue
                if self.use_direction[y][x] == Env.OBSTACLE or self.use_direction[y][x] == Env.EXIT:
                    continue

                # if spot has lower gradient value then the current_pos we add it 选择梯度值小的进行移动
                #print("错误0715", density_map[y][x],self.current_pos[0],self.current_pos[1])
                if self.use_direction[y][x]+density_map[y][x] < self.use_direction[self.current_pos[1]][self.current_pos[0]]+density_map[self.current_pos[1]][self.current_pos[0]]:
                    # we create list of ( gradient_value, (y, x) )
                    available_spots.append((self.use_direction[y][x], (x, y))) #存入梯度值与对应的坐标值

        available_spots.sort()

        return available_spots

    def get_best_move(self, available_spots: [(int, (int, int))],collision_map):
        if len(available_spots) == 0:
            return None

        for i in range(0, len(available_spots)):
            a_x, a_y = available_spots[i][1]
            if collision_map[a_y][a_x] == 0:
                return available_spots[i][1]
        return None

    def block_point(self, position, collision_map):
        collision_map[position[1]][position[0]] = 1

    def unblock_point(self, position, collision_map):
        collision_map[position[1]][position[0]] = 0

    def update_gradient(self, value, density_map,collision_map):
        for x in range(-self.gradient_space_size, self.gradient_space_size + 1):
            for y in range(-self.gradient_space_size, self.gradient_space_size + 1):

                local_y = self.current_pos[1] + y
                local_x = self.current_pos[0] + x

                if y >= 5 or y <= -5 or x >= 5 or x <= -5:
                    tmp_value = int(value/5)
                elif y == 4 or y == -4 or x == 4 or x == -4:
                    tmp_value = int(value/4)
                elif y == 3 or y == -3 or x == 3 or x == -3:
                    tmp_value = int(value/3)
                elif y == 2 or y == -2 or x == 2 or x == -2:
                    tmp_value = int(value/2)
                else:
                    tmp_value = value

                # If we this is current spot we double value here
                if local_y == self.current_pos[1] and local_x == self.current_pos[0]:
                    density_map[local_y][local_x] += tmp_value*2
                    continue

                # If we out of range we skip
                if local_y >= len(collision_map) or local_x >= len(collision_map[0]) or \
                        local_y <= 0 or local_x <= 0:
                    continue

                # If spot is obstacle or exit we skip
                if self.use_direction[local_y][local_x] == Env.EXIT or \
                        self.use_direction[local_y][local_x] == Env.OBSTACLE:
                    continue

                # Normal situation
                if self.use_direction[local_y][local_x] != 0:
                    density_map[local_y][local_x] += tmp_value * 2

    def move(self, density_map, collision_map):

        available_positions = self.get_available_moves(density_map,collision_map)

        best_pos = self.get_best_move(available_positions,collision_map)
        if self.move_counts < self.velocity:
            best_pos = self.current_pos
            self.move_counts += 1
        else:
            self.move_counts = 0
        if best_pos is None:
            best_pos = self.current_pos

        self.unblock_point(self.current_pos,collision_map)

        #print("测试0521",best_pos[1],self.direction_map[best_pos[0]][best_pos[1]])
        if self.use_direction[best_pos[1]][best_pos[0]] <= 0:
        #if best_pos == Env.EXIT or self.direction_map[best_pos[0]][best_pos[1]] < 15:
            self.update_gradient(-self.value,density_map,collision_map)
            print(best_pos,"exit reached")
            raise ExitReached

        self.update_facing_angle(best_pos)

        self.update_gradient(-self.value,density_map,collision_map) #减小梯度值，表示人将要移除

        self.current_pos = best_pos
        self.block_point(self.current_pos,collision_map)

        self.update_gradient(self.value,density_map,collision_map) #增加周围范围的梯度，表示人密度影响

        return 0


