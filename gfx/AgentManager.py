import copy
import numpy as np
import csv
from gfx.AgentGfx import AgentGfx
from model.agent.Agent import ExitReached
from model.gradient.gradient_map import gradient_from_direction_map

class AgentManager:
    def __init__(self, initial_tile_size: [float, float], client_width: int, client_height: int, map_offset: int,
                 maze, ped_data, direction_data,init_frame_count,freq):
        self.ped_data = ped_data
        self.direction_data = direction_data
        self.agent_list = list()
        self.tile_size = initial_tile_size
        self.agent_radius = (initial_tile_size[1] - initial_tile_size[1] / 5) / 2
        self.width = client_width
        self.height = client_height
        self.offset = map_offset
        self.maze = maze #迷宫障碍
        self.collision = copy.deepcopy(maze)
        self.maze_for_agent = copy.deepcopy(maze)
        maze_array = np.array(maze)
        self.density = np.zeros((maze_array.shape), dtype=int)
        self.agent_num = 0
        self.update_frame_count = init_frame_count
        self.Trajectory = [] #初始时候是空列表
        self.data_freq = freq
    def set_client_tile_size(self, client_width: int, client_height: int, tile_size: [float, float]):
        self.width = client_width
        self.height = client_height
        self.tile_size = tile_size
        self.agent_radius = (tile_size[1] - tile_size[1] / 5) /2

        for agent in self.agent_list:
            correct_pos = [
                0 + self.offset + 1 + (agent.map_position[1] * self.tile_size[0]) + (self.tile_size[0] / 2),
                self.height - self.offset - 1 - (agent.map_position[0] * self.tile_size[1]) - (self.tile_size[1] / 2)
            ]
            agent.position = correct_pos

    def updateDirection(self, cur_frame):
        for ped in self.agent_list:
            ped_direct_id = ped.agent.direct_id
            #寻找离当前frame最近的帧
            ped_frame_list = list(self.direction_data[ped_direct_id].keys())
            ped_frame_list.remove('exit_frame_index')
            cur_direction_frame=self.find_closest_number(ped_frame_list,cur_frame)
            direction_raw = self.direction_data[ped_direct_id][cur_direction_frame]
            direction = direction_raw
            ped.agent.use_direction = direction
    def draw_all(self):
        for agent in self.agent_list:
            agent.draw(self.agent_radius)

    def add_new(self,ped_id: int, position: [int, int], angle: float, color: [float, float, float]):
        #设置agent_id
        agent_id = ped_id
        agent_direct_id = ped_id
        #设置速度
        velocity=self.ped_data[agent_id]['initial_velocity']
        #设置出口
        exits = []
        for i in range(40, 91):
            exits.append((99, i))
        #设置最短路径场
        astar_map = self.ped_data[agent_id]['astar_direction']
        #设置引导场
        direction_map = self.ped_data[agent_id]['direction']
        correct_pos = [
            0 + self.offset + 1 + (position[0] * self.tile_size[0]) + (self.tile_size[0] / 2),
            self.height - self.offset - 1 - (position[1] * self.tile_size[1]) - (self.tile_size[1] / 2)
        ]#实际的坐标位置

        if self.maze_for_agent[position[1]][position[0]] == 0:
            #如果当前位置为可通行
            self.agent_num += 1
            self.agent_list.append(AgentGfx(agent_id, agent_direct_id, correct_pos, position, angle, color, self.maze_for_agent, velocity, astar_map, direction_map, exits, self.density, self.collision))#将agent初始化信息加入管理列表
        else:
            #如果当前位置不可通行
            print('Agent can not be adde on this pos')
    #新建一个add_new_gene函数表示由自己生成的人群
    def add_new_gene(self,ped_id: int, direct_id: int, position: [int, int], angle: float, color: [float, float, float]):
        #设置agent_id
        agent_id = ped_id
        agent_direct_id = direct_id
        #设置速度
        velocity=self.ped_data[agent_direct_id]['initial_velocity']
        #设置出口
        exits = []
        for i in range(40, 91):
            exits.append((99, i))
        #设置最短路径场
        astar_map = self.ped_data[agent_direct_id]['astar_direction']
        #设置引导场
        direction_map = self.ped_data[agent_direct_id]['direction']
        correct_pos = [
            0 + self.offset + 1 + (position[0] * self.tile_size[0]) + (self.tile_size[0] / 2),
            self.height - self.offset - 1 - (position[1] * self.tile_size[1]) - (self.tile_size[1] / 2)
        ]#实际的坐标位置

        if self.maze_for_agent[position[1]][position[0]] == 0:
            #如果当前位置为可通行
            self.agent_num += 1
            self.agent_list.append(AgentGfx(agent_id, agent_direct_id, correct_pos, position, angle, color, self.maze_for_agent, velocity, astar_map, direction_map, exits, self.density, self.collision))#将agent初始化信息加入管理列表
        else:
            #如果当前位置不可通行
            print('Agent can not be adde on this pos')
    def record_traj(self, frame_count, moving_list):
        #对于每个frame,以追加形式写入csv文件（frame, person,x_grid,y_grid）
        record_data = []
        for re_agent in moving_list:
            record_data.append([frame_count, re_agent.agent.id, re_agent.agent.current_pos[0], re_agent.agent.current_pos[1]])
        self.Trajectory.extend(record_data)


    def save_person_information(self,save_file_name,save_data):
        with open(save_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(save_data)
    def save_traj(self, save_file_name):
        save_trajectory = np.array(self.Trajectory)
        with open(save_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in save_trajectory:
                writer.writerow(row)
        print(f"数据成功写入成功")

    def step(self, frame_count : int):
        moving_lsit = sorted(self.agent_list, key=lambda agt: agt.agent.anger, reverse=True) #按躁动焦急程度锦绣排序
        #先记录位置
        self.record_traj(frame_count, moving_lsit)
        any_moved = True

        while any_moved:
            any_moved = False

            for agent in moving_lsit:
                try:
                    anger = agent.move(self.density,self.collision)
                except ExitReached:
                    self.agent_list.remove(agent)
                    moving_lsit.remove(agent)
                    # 程序如果没有出现错误，则执行else语句块，如果出现错误，则执行except块，else语句块将被跳过不执行
                else:
                    if anger == 0:
                        any_moved = True
                        moving_lsit.remove(agent)

                        agent.map_position = agent.agent.current_pos
                        agent.fx_pos = [
                            0 + self.offset + 1 + (agent.map_position[0] * self.tile_size[0]) + (self.tile_size[0] / 2),
                            self.height - self.offset - 1 - (agent.map_position[1] * self.tile_size[1]) - (
                                    self.tile_size[1] / 2)
                        ]
                        agent.position = agent.fx_pos
        if (frame_count-self.update_frame_count==self.data_freq):
            self.update_frame_count = frame_count
            self.updateDirection(frame_count)

    def find_closest_number(self, numbers, target):
        closest_number = None
        min_difference = float('inf')

        for number in numbers:
            difference = abs(number-target)
            if difference < min_difference:
                min_difference = difference
                closest_number = number

        return closest_number