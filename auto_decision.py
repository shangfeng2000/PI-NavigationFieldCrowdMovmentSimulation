import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from resources.handling.reading import load_direction_from_file, load_map_from_file
from scipy.ndimage import label, generate_binary_structure, center_of_mass
from model.environment.environment_enum import Env

def normal_mat(matrix,k):
   row_min = matrix.min(axis=1, keepdims=False)
   row_min = np.min(np.where(matrix != 0, matrix, np.inf), axis=1)
   #row_min = matrix.mean(axis=1, keepdims=False)
   row_adjusted_matrix = np.zeros_like(matrix)
   for i in range(matrix.shape[0]):
       if row_min[i] != 0:
           row_adjusted_matrix[i, :] = abs(matrix[i, :] - row_min[i]) / row_min[i]
       else:
           row_adjusted_matrix[i, :] = abs(matrix[i, :] - row_min[i])/ row_min[i]


   # 计算每列的最小值
   col_min = matrix.min(axis=0, keepdims=False)
   col_min = np.min(np.where(matrix != 0, matrix, np.inf), axis=0)
   #col_min = matrix.mean(axis=0, keepdims=False)
   # 生成列处理后的新矩阵
   col_adjusted_matrix = np.zeros_like(matrix)

   for j in range(matrix.shape[1]):
       if col_min[j] != 0:
           col_adjusted_matrix[:, j] = abs(matrix[:, j] - col_min[j]) / col_min[j]
       else:
           col_adjusted_matrix[:, j] = abs(matrix[:, j] - col_min[j])/col_min[j]

   # 最终结果矩阵
   result_matrix = row_adjusted_matrix + col_adjusted_matrix
   return result_matrix


def find_valleys(matrix, threshold):
    # 创建低于阈值的二值矩阵
    binary_valleys = (matrix < threshold).astype(int)
    structure = generate_binary_structure(2, 2)
    labeled_valleys, num_valleys = label(binary_valleys, structure)

    # 计算每个山谷的中心坐标
    valley_centers = []
    for i in range(1, num_valleys + 1):
        coords = np.argwhere(labeled_valleys == i)
        center = center_of_mass(binary_valleys, labeled_valleys, i)
        valley_centers.append(center)

    return labeled_valleys, num_valleys, valley_centers


def find_valleys2(matrix):
    rows, cols = matrix.shape
    valleys = np.zeros_like(matrix, dtype=int)
    valley_label = 1  # 用于标记山谷的编号

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 获取周围8个邻域的高度
            neighbors = [
                matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i - 1, j + 1],
                matrix[i, j - 1], matrix[i, j + 1],
                matrix[i + 1, j - 1], matrix[i + 1, j], matrix[i + 1, j + 1]
            ]

            # 检查当前点是否是山谷
            if all(matrix[i, j] < neighbor for neighbor in neighbors):
                valleys[i, j] = valley_label

    # 对标记的山谷区域进行连通性分析
    labeled_valleys, num_valleys = label(valleys)

    # 计算每个山谷的中心坐标
    valley_centers = []
    for i in range(1, num_valleys + 1):
        center = center_of_mass(valleys, labeled_valleys, i)
        valley_centers.append(center)

    return labeled_valleys, num_valleys, valley_centers

def plot_contours(matrix):
    plt.figure(figsize=(10, 8))
    contour = plt.contour(matrix, levels=15, cmap='terrain')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.title('Terrain Contours')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.colorbar(label='Height')
    plt.gca().invert_yaxis()
    plt.show()


Data_sorce = 'ETH'
#初始化地图
maze_filename = 'H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+"/maze_decision.txt"
maze = load_map_from_file(maze_filename)
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+'/final_direction.pickle', 'rb') as file:
    direction_dict = pickle.load(file)
#初始化行人数据
ped_data = dict()
ped_filename = 'H:/GRK/CrowdMovmentSimulation-master/Data/Decision/person_decision_single.csv'  # 替换为你的 CSV 文件路径
with open(ped_filename, 'r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        if len(row) >= 5:  # 确保至少有行人id、时间、初始x坐标和初始y坐标这四列数据
            pedestrian_id = int(row[0])
            ped_enter_time = direction_dict[pedestrian_id]['exit_frame_index']
            ped_data[pedestrian_id] = {
                'ped_id': pedestrian_id,
                'enter_time': ped_enter_time
            }
# 初始化场数据
frame_data=dict()
for ped_id in ped_data.keys():
    direction_frame = list(direction_dict[ped_id].keys()) #'exit_frame_index', 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386]
    direction_raw = direction_dict[ped_id][direction_frame[1]] #初始化的第一个导航势能场
    direction = direction_raw
    ped_data[ped_id]['direction'] = direction

    for f in direction_frame[1:]:
        frame_data[f]=dict()
        frame_data[f][ped_id]=direction

ped_data_list = list(ped_data.keys())  #所有人的人员列表
direct_frame_list = list(frame_data.keys())


ped_enter_dict = dict()
ped_agent_dict = dict()
global_intensity = 100
ped_information_file = 'H:/GRK/CrowdMovmentSimulation-master/Data/' + 'Decision/single/auto/'+str(global_intensity)+'_person_information.csv'
with open(ped_information_file, mode='r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过第一行
    for row in reader:
        id = int(row[0])
        entrance_frame = int(row[1])
        agent_id = int(row[4])
        ped_enter_dict[id] = entrance_frame
        ped_agent_dict[id] = agent_id
ped_list = list(ped_enter_dict.keys())
T_min=980
T_max=T_min+100
attention_person = [] #用来存储在时间段里的人
total_Energy = np.zeros((len(direction),len(direction[0])))
end_point_list = []
for p in ped_list:
    if (ped_enter_dict[p]>=T_min and ped_enter_dict[p]<T_max):
        attention_person.append(p)
        direction_frame = list(direction_dict[ped_agent_dict[p]].keys())
        attention_direction = direction_dict[ped_agent_dict[p]][direction_frame[1]]
        temp_direction = attention_direction
        for i in range(len(temp_direction)):
            for j in range(len(temp_direction[i])):
                if temp_direction[i][j] == Env.OBSTACLE:
                    temp_direction[i][j] = 500
                if temp_direction[i][j] == 0:
                    end_point_list.append((i,j))
        temp_direction = np.array(temp_direction)
        normal_direction = normal_mat(temp_direction, 1500) #归一化处理
        total_Energy = total_Energy+normal_direction

print(attention_person)

# 识别山谷区域
threshold = 100  # 设置阈值
valleys, num_valleys, valley_centers = find_valleys(total_Energy,threshold)
print("识别出的山谷中心点")
print(valley_centers)
plot_contours(total_Energy)

# 显示山谷区域
plt.figure(figsize=(10, 8))
plt.imshow(valleys, cmap='Blues', alpha=0.5)
plt.title('Valleys Identified (Overlay)')
plt.colorbar(label='Valley Label')
plt.show()

plt.contourf(total_Energy,levels=50)
# for (q,k) in end_point_list:
#     plt.scatter(k, q, color='#DA635C', marker='*', s=200, edgecolor='black', linewidth=1, zorder=3)
# for center in valley_centers:
#     plt.scatter(center[1], center[0], color='w', marker='*', s=200, edgecolor='black', linewidth=1, zorder=3)
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
#plt.colorbar()  # 显示颜色条
plt.savefig('H:/GRK/STG论文/毕业设计/引导区域预测t='+str(T_min)+'.png',dpi=500)
plt.show()
#给定一个时间范围
#获取在该时间内的所有人的id
#将所有人的在该时间区间内的导航势能场加起来求和
#找到鞍点
#对该时间区间内的所有行人
#计算替换后的导航场
#将计算的导航场存入字典
#数据存入文件