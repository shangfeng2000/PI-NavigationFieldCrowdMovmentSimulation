#地图初始化
#数据集轨迹分析
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import random
import pickle
import heapq
import random
import math
import csv
from PIL import Image
from copy import deepcopy
from model.environment.environment_enum import Env
from model.gradient.gradient_map import gradient_from_direction_map
from enum import Enum
from resources.handling.reading import load_direction_from_file, load_map_from_file
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def cal_velocity(file_path):
    data = np.loadtxt(file_path)
    persons = np.unique(data[:, 1]).tolist()
    persons_data = []
    persons_data_v = []
    for p in persons:
        p_data = data[p == data[:, 1], :]
        p_data_rel = np.ones((p_data.shape[0], 2))
        for frame in range(p_data.shape[0] - 1):
            p_data_rel[frame, :] = (p_data[frame + 1, 2:4] - p_data[frame, 2:4]) / 0.4
        p_data_rel = np.hstack((p_data[:-1,:], p_data_rel[:-1, :]))
        persons_data_v.append(p_data_rel)
    persons_data = np.concatenate(persons_data_v, axis=0)
    sorted_indices = np.lexsort((persons_data[:, 1], persons_data[:, 0]))
    persons_data = persons_data[sorted_indices]
    return persons_data


def create_binary_matrix(image_path, threshold=128):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    width, height = image.size

    # Convert image to numpy array
    image_array = np.array(image)

    # Convert image array to binary matrix based on threshold
    binary_matrix = np.zeros((height, width), dtype=int)
    binary_matrix[image_array < threshold] = 1

    return binary_matrix


def create_grid_matrix(binary_matrix, m, n):
    # Dimensions of the binary matrix
    height, width = binary_matrix.shape

    # Calculate grid size
    grid_width = width // n
    grid_height = height // m

    # Initialize grid matrix
    grid_matrix = np.zeros((m, n), dtype=int)

    # Traverse each grid
    for i in range(m):
        for j in range(n):
            # Calculate grid boundaries in the binary matrix
            x1, y1 = j * grid_width, i * grid_height
            x2 = (j + 1) * grid_width if j < n - 1 else width  # 处理最后一列的情况
            y2 = (i + 1) * grid_height if i < m - 1 else height  # 处理最后一行的情况

            # Extract the corresponding part of the binary matrix
            grid = binary_matrix[y1:y2, x1:x2]

            # Check if grid contains any impassable area (white pixels)
            if np.any(grid == 0):
                grid_matrix[i, j] = 1  # Cannot pass
            else:
                grid_matrix[i, j] = 0  # Can pass

    return grid_matrix


def save_maze_to_txt(maze_matrix, output_file):
    # 将迷宫矩阵保存为 txt 文件
    np.savetxt(output_file, maze_matrix, fmt='%d')

def word_to_grid(min_x,max_x,min_y,max_y,row_num,col_num,word_x,word_y):
    grid_width = (max_x - min_x) / col_num
    grid_height = (max_y - min_y) / row_num
    # Calculate column index
    col_index = int((word_x - min_x) / grid_width)
    # Calculate row index
    row_index = int((max_y - word_y) / grid_height)
    grid_x = col_index
    grid_y = row_index
    return (grid_x,grid_y)

def grid_to_word(min_x,max_x,min_y,max_y,row_num,col_num,grid_x,grid_y):
    grid_width = (max_x - min_x) / col_num
    grid_height = (max_y - min_y) / row_num
    # Calculate world x coordinate
    word_x = min_x + grid_x * grid_width
    # Calculate world y coordinate, inverted due to Y increasing upwards
    word_y = max_y - grid_y * grid_height
    return (word_x,word_y)


def get_adjacent_cells(cell, grid_size):
    """
    获取给定单元格周围八个方向的单元格坐标
    """
    i, j = cell
    adjacent_cells = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                      (i, j - 1), (i, j + 1),
                      (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
    return [(x, y) for x, y in adjacent_cells if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]]


def clean_trajectory(trajectory):
    if len(trajectory) < 2:
        return trajectory

    cleaned_trajectory = []
    seen_points = set()

    # 从后往前遍历轨迹
    for point in reversed(trajectory):
        tuple_point = tuple(point)
        if tuple_point not in seen_points:
            # 如果点之前没见过，则加入结果列表和集合
            cleaned_trajectory.insert(0, point)
            seen_points.add(tuple_point)
    return cleaned_trajectory


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
    expansion_counts = np.full((grid_size), None)  # 标记是第几次被扩展
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
                            if expansion_counts[i, j] == None:
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
                                if expansion_counts[i, j] == None:
                                    expansion_counts[i, j] = step
                            else:
                                direction_array[i, j] = Env.EXIT
                    else:
                        if grid_map[i, j] == 1:
                            direction_array[i, j] = Env.OBSTACLE
                        else:
                            # 找到周围八个方向中能够到达的轨迹经过的网格
                            reachable_cells = [adj_cell for adj_cell in get_adjacent_cells(current_pos, grid_size) if
                                               adj_cell in traj_set]
                            if reachable_cells:
                                chosen_cell = random.choice(reachable_cells)
                                direction_array[i, j] = chosen_cell
                                if expansion_counts[i, j] == None:
                                    expansion_counts[i, j] = step + 1
                                newly_marked_cells.add(current_pos)  # 将新标记的网格添加到集合中
                            else:
                                # 如果当前网格没有邻居可达，则指向自身
                                direction_array[i, j] = current_pos

            # 更新轨迹集合，以便下一次扩展
            traj_set.update(map(tuple, newly_marked_cells))
    return direction_array, expansion_counts


def cal_field_ByTra(maze, exits, traj, expand_num):
    grid_map = np.array(maze)
    trajectory_grid = traj
    trajectory_astar_grid = []
    exits = (exits[1], exits[0])  # 矩阵和坐标索引是反的，坐标要变为（y,x）
    for point in trajectory_grid:
        # 将世界坐标转换为网格坐标
        grid_x = point[0]
        grid_y = point[1]
        trajectory_astar_grid.append((grid_y, grid_x))
    trajectory_astar_grid.append(exits)
    trajectory_astar_grid = list(dict.fromkeys(trajectory_astar_grid))
    corrected_path = [trajectory_astar_grid[0]]
    for i in range(len(trajectory_astar_grid) - 1):
        start = trajectory_astar_grid[i]
        goal = trajectory_astar_grid[i + 1]
        if (start[0] == goal[0] and start[1] == goal[1]):
            path = [start]
        else:
            path = astar(grid_map, start, goal)
            path = path[::-1]
        # path.append(start)
        if path:
            corrected_path.extend(path)
        else:
            print(f"No path found from {start} to {goal}")
    # 将轨迹标记在地图上
    trajectory_new = []
    corrected_path == list(dict.fromkeys(corrected_path))
    for point in corrected_path:
        trajectory_new.append([point[0], point[1]])
    trajectory_new = np.array(trajectory_new)
    trajectory_new = clean_trajectory(trajectory_new)
    trajectory_new = np.array(trajectory_new)
    expand_steps = expand_num
    direction_array, expansion_counts = generate_direction_array(trajectory_new, grid_map, expand_steps)
    return direction_array, expansion_counts

data_sorce="ETH"
data_stgcn="eth"
#场景离散化
data_eth = read_file('H:/GRK/STG/Social-STGCNN-master/datasets_icdm/eth/test/biwi_eth.txt')
frames = np.unique(data_eth[:, 0]).tolist()
persons = np.unique(data_eth[:,1]).tolist()#提取人id
frame_data = []
person_data = []
for frame in frames:
    frame_data.append(data_eth[frame == data_eth[:, 0], :])
for person in persons:
    person_data.append(data_eth[person == data_eth[:, 1], :])
max_y = np.amax(data_eth[:,2])
min_y = np.amin(data_eth[:,2])
max_x = np.amax(data_eth[:,3])
min_x = np.amin(data_eth[:,3])
row_num = math.ceil((max_y-min_y)/0.4)
col_num = math.ceil((max_x-min_x)/0.4)
# 定义输出文件路径
output_file = 'H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/maze.txt'
# Example usage:
image_path = 'H:/GRK/ewap_dataset/seq_'+data_stgcn+'/map.png'
m = row_num  # Number of rows
n = col_num   # Number of columns
binary_matrix = create_binary_matrix(image_path)
grid_matrix = create_grid_matrix(binary_matrix, m, n)
save_maze_to_txt(grid_matrix, output_file)
obs_len=12
pred_len=8
seq_len = obs_len+pred_len
skip=1
num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))  #可构建序列数量
remain_person = []
remain_sequence = 0
sequence_person = dict()
sequence_frame = dict()
for idx in range(0, num_sequences * skip + 1, skip):
    curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0) # 列拼接
    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
    max_peds_in_frame =len(peds_in_curr_seq)
    if (max_peds_in_frame>=2):
        valid_person = 0
        valid_person_id = []
        for ped_id in peds_in_curr_seq:
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            pad_front = frames.index(curr_ped_seq[0, 0])
            pad_end = frames.index(curr_ped_seq[-1, 0])
            if pad_end - pad_front+1 == seq_len:
                valid_person+=1
                valid_person_id.append(ped_id)
        if valid_person>=2:
            remain_sequence += 1
            remain_person.extend(valid_person_id)
            sequence_person[idx+1]=valid_person_id
            sequence_frame[idx+1]=frames[idx]
remain_person_id = np.unique(remain_person).tolist()
print(len(remain_person_id))
print(len(sequence_person))
#计算每个人的开始帧和结束帧
grid_map = load_map_from_file('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/maze.txt')
selc_persons = deepcopy(remain_person_id)
dest = dict()
#生成person列表
person_filename = "H:/GRK/CrowdMovmentSimulation-master/Data/"+data_sorce+"/person.origin.csv"
with open(person_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id','frame', 'frame_index','end_frame_index', 'x', 'y', 'vel','tar_x','tar_y','grid_x','grid_y','grid_tar_x','grad_tar_y','grid_vel'])
    for person in selc_persons:
        p_data = data_eth[person == data_eth[:, 1], :]
        pad_init_frame = p_data[7,0] #假设已知初始阶段轨迹
        pad_end_frame = p_data[-1,0]
        pad_end_frame_index = frames.index(p_data[-1,0])
        pad_init_frame_index = frames.index(p_data[7,0])
        pad_init_x = p_data[7,3]
        pad_init_y = p_data[7,2]#x和y转向
        grid_x,grid_y = word_to_grid(min_x,max_x,min_y,max_y,row_num,col_num,pad_init_x,pad_init_y)#离散化
        pad_target_x = p_data[-1,3]
        pad_target_y = p_data[-1,2]#x和y转向
        target_grid_x,target_grid_y = word_to_grid(min_x,max_x,min_y,max_y,row_num,col_num,pad_target_x,pad_target_y)
        pad_vel = math.sqrt((pad_init_y-p_data[0,2])**2+(pad_init_x-p_data[0,3])**2)/(8*0.4)
        pad_grid_v = int(pad_vel/0.4) #每秒移动几个网格
        if (target_grid_y>=len(grid_map)):
            target_grid_y=len(grid_map)-1
        if (target_grid_x>=len(grid_map[0])):
            target_grid_x=len(grid_map[0])-1
        dest[person]=(target_grid_x,target_grid_y)
        writer.writerow([person,pad_init_frame, pad_init_frame_index, pad_end_frame_index, pad_init_x, pad_init_y, pad_vel,pad_target_x,pad_target_y,grid_x,grid_y,target_grid_x,target_grid_y,pad_grid_v])
        if (grid_map[target_grid_y][target_grid_x]==1):
            grid_map[target_grid_y][target_grid_x]=0
# 定义输出文件路径
output_file = 'H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/maze_new.txt'
save_maze_to_txt(np.array(grid_map), output_file)
import pickle
with open('H:/GRK/STG/Social-STGCNN-master/Results/STGCNN/'+data_stgcn+'_data.pickle', 'rb') as file:
    pinn_dict = pickle.load(file)
with open('H:/GRK/STG/Social-STGCNN-master/Results/STGCNN/'+data_stgcn+'_tcss.pickle', 'rb') as file:
    record_dict = pickle.load(file)
maze_map = load_map_from_file('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/maze_new.txt')
step1=pinn_dict[1]
s1_obs=step1['obs']
s1_target=step1['trgt']
s1_pre=step1['pred']
v_pre=step1['velocity_pred']
#遍历所有的预测轨迹，筛选出其中ADE最小的，输出对应的网格坐标序列，生成字典，key1是人的id，key2是时间
All_direct = dict()
for p in remain_person_id:
    All_direct[p] = dict()
for step in pinn_dict.keys():
    step_data = pinn_dict[step]
    record_data = record_dict[step]
    step_pre = step_data['pred']
    step_person_num = step_pre[0].shape[1]#计算该片段里的行人数量
    for p in range(step_person_num):
        min_sample = record_data[p] #找到最小误差的采样样本
        pre_person_data = step_pre[min_sample] #所有人的
        pre_p_data = pre_person_data[:,p,:] #当前选定的人
        sequence_person_list = list(sequence_person.keys())
        p_id = sequence_person[sequence_person_list[step-1]][p]#找到行人id号
        seq_index = sequence_person_list[step-1]
        p_frame = sequence_frame[seq_index]#找到所在的frame
        #将坐标转化为网格坐标
        p_pre_grid_list = []
        for cor in pre_p_data:
            grid_x,grid_y = word_to_grid(min_x,max_x,min_y,max_y,row_num,col_num,cor[1],cor[0])#离散化,同样x和y要转向
            if (grid_x>=col_num or grid_y>=row_num or grid_x<0 or grid_y<0):
                continue
            if (grid_matrix[grid_y,grid_x]==0):
                p_pre_grid_list.append((grid_x,grid_y))
        #获得了预测轨迹序列
        #p_pre_grid_list.append(dest[p_id])#需要加上终点网格
        dest_grid=dest[p_id]
        expand_num = 100
        #考虑添加初始位置，增加连贯性
        p_pre_grid_list_new =list(dict.fromkeys(p_pre_grid_list))
        #调用函数计算输出导航场
        #检查出口是否是障碍，如果是障碍，需要将障碍网格进行修改
        p_direction_array,p_expansion_counts=cal_field_ByTra(maze_map, dest_grid, p_pre_grid_list_new,expand_num)#调用函数计算输出导航场
        #增加检验是否有指向自身而没有赋值矢量的网格
        All_direct[p_id][seq_index]=p_direction_array
#存储数据
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/direct_data.pickle', 'wb') as file:
    pickle.dump(All_direct, file)
#读取数据
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/direct_data.pickle', 'rb') as file:
    direction_dict = pickle.load(file)
print(len(direction_dict))
final_direction_dict = dict()
for p in remain_person_id:
    final_direction_dict[p] = dict()
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/person.origin.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    # 跳过标题行
    next(reader)  # 跳过第一行
    # 逐行读取并处理数据
    for row in reader:
        # 解析每一行数据
        person_id = int(float(row[0]))
        direction_frame = list(direction_dict[person_id].keys())
        pad_init_frame_index = direction_frame[0]  # 假设第三列是浮点数，转换为整数
        pad_exit_frame_index = pad_init_frame_index + 7
        pad_end_frame_index = direction_frame[-1]  # 假设第四列是浮点数，转换为整数
        final_direction_dict[person_id]['exit_frame_index'] = pad_exit_frame_index
        i = 0
        while (i < len(direction_dict[person_id])):
            pad_mid_frame_index = direction_frame[i]
            direction_raw = direction_dict[person_id][pad_init_frame_index]
            direction_raw_list = direction_raw.tolist()
            direction_field = gradient_from_direction_map(direction_raw_list)
            final_direction_dict[person_id][pad_mid_frame_index] = direction_field
            i += 1
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/final_direction.pickle', 'wb') as file:
        pickle.dump(final_direction_dict, file)

#生成A星纯路径规划算法的场
#读入数据
#遍历每个行人，获取起点和终点坐标，生成A星轨迹
#生成数值场
data_file = read_file('H:/GRK/STG/Social-STGCNN-master/datasets_icdm/'+data_stgcn+'/test/biwi_eth.txt')
save_file_name = 'H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/results/True_Trajectory.csv'
#提取帧数
frames = np.unique(data_eth[:, 0]).tolist()
persons = np.unique(data_eth[:,1]).tolist()#提取人id
frame_data = []
person_data = []
remain_person_id
sequence_person
print(len(frames))
with open(save_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    for frame in frames:
        current_frame_data=(data_eth[frame == data_eth[:, 0], :])
        for frame_p in current_frame_data:
            frame_p_id =frame_p[1]
            if frame_p_id in remain_person_id:
                frame_p_frame = frame_p[0]
                frame_p_index = frames.index(frame_p_frame)+1
                frame_p_x = frame_p[3]
                frame_p_y = frame_p[2]
                frame_grid_x,frame_grid_y = word_to_grid(min_x,max_x,min_y,max_y,row_num,col_num,frame_p_x,frame_p_y)#离散化
                row = [frame_p_index, frame_p_id, frame_grid_x, frame_grid_y]
                writer.writerow(row)