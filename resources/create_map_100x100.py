import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model.environment.environment import direction_map
from model.environment.line import Point
from model.environment.environment_enum import Env
from resources.handling.generating import create_txt_form_direction_map
from resources.handling.reading import load_map_from_file
from model.gradient.gradient_map import gradient_from_direction_map

def draw_navigation_field(grid):
    # 将None替换为一个较小的值，比如负无穷
    rows=len(grid)
    cols =len(grid[0])
    plt.figure(figsize=(cols, rows))
    plt.axis('off')

    # 网格边长
    grid_size = 1.0

    # 箭头长度
    arrow_length = grid_size * 0.6

    # 绘制网格线
    for i in range(rows + 1):
        plt.plot([0, cols], [i, i], 'k-', linewidth=1)  # 横线
    for j in range(cols + 1):
        plt.plot([j, j], [0, rows], 'k-', linewidth=1)  # 竖线

    # 遍历每个网格
    for i in range(rows):
        for j in range(cols):
            #如果当前网格是障碍，则用黑色或灰色填充
            if grid[i][j] == Env.OBSTACLE:
                plt.fill([j, j, j + 1, j + 1], [i, i + 1, i + 1, i], color='black', alpha=1)
            # 如果当前网格是出口，则用红色或绿色填充
            elif grid[i][j] == Env.EXIT:
                plt.fill([j, j, j + 1, j + 1], [i, i + 1, i + 1, i], color='red', alpha=0.5)

            # 如果指向自身，则绘制圆点
            elif grid[i][j] == (i, j):
                # 获取当前网格指向的下一个网格的行号和列号
                next_i, next_j = grid[i][j]

                # 当前网格中心位置
                current_center = [j + 0.5, i + 0.5]

                # 下一个网格中心位置
                next_center = [next_j + 0.5, next_i + 0.5]
                plt.plot(current_center[0], current_center[1], 'ko', markersize=5)
            else:
                # 获取当前网格指向的下一个网格的行号和列号
                next_i, next_j = grid[i][j]

                # 当前网格中心位置
                current_center = [j + 0.5, i + 0.5]

                # 下一个网格中心位置
                next_center = [next_j + 0.5, next_i + 0.5]
                # 计算箭头的中点坐标
                arrow_center = [current_center[0] , current_center[1]]

                # 计算箭头的方向
                dx = next_center[0] - current_center[0]
                dy = next_center[1] - current_center[1]

                # 根据箭头长度缩放方向
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx /= length
                    dy /= length

                dx *= arrow_length
                dy *= arrow_length

                # 绘制箭头
                plt.arrow(arrow_center[0] - dx / 2, arrow_center[1] - dy / 2, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.gca().invert_yaxis()  # 反转 y 轴

    plt.show()



#生成A星纯路径规划算法的场
#读入数据
data_sorce = 'UNIV'
mazeGK = load_map_from_file('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/maze_new.txt')
person_filename = "H:/GRK/CrowdMovmentSimulation-master/Data/"+data_sorce+"/person.origin.csv"
astar_direction_dict = dict()
i=0
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/person.origin.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    # 跳过标题行
    next(reader)  # 跳过第一行
    # 逐行读取并处理数据
    for row in reader:
        # 解析每一行数据
        print(i)
        person_id = int(float(row[0]))
        person_end = [Point(int(float(row[11])),int(float(row[12])))]
        direction_raw = direction_map(mazeGK, person_end, 1)
        astar_field = gradient_from_direction_map(direction_raw)
        astar_direction_dict[person_id] = astar_field
        i+=1
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+data_sorce+'/astar_direction.pickle', 'wb') as file:
    pickle.dump(astar_direction_dict, file)
#create_txt_form_direction_map("ready/GK_directionmap_four_100x100.txt", directions)

