#读取数据
import pickle
import csv
from copy import deepcopy
from model.environment.environment_enum import Env
def gradient_from_direction_map(direction_map):
    """Returns gradient map based on distance counted from direction map"""
    #direction_map = load_direction_from_file(direction_map_txt)

    # Make gradient map same size as direction map
    gradient_map = deepcopy(direction_map)

    # Main loop, loop through all the points
    for y in range(0, len(direction_map)):
        for x in range(0, len(direction_map[y])):

            # Check if we already updated gradient map in [y][x]
            if gradient_map[y][x] != direction_map[y][x]:
                continue

            # If obstacle we skip
            if direction_map[y][x] == Env.OBSTACLE:
                continue

            # Get current position
            position = direction_map[y][x]

            # Initialize distance variable for distance from the start point
            distance = 0


            # Loop till EXIT found so distance is full
            find_count = 0
            while position != Env.EXIT:

                # Get next position
                find_count+=1
                next_position = direction_map[position[0]][position[1]]

                if next_position == Env.EXIT:  # TODO its bad break exit, find better solution for this
                    if position[0] == y or position[1] == x:
                        distance += 10
                    else:
                        distance += 10
                    break

                # Check if we move vertical (+10) or if diagonal (+14 (sqrt(2)*10))
                if position[0] == y or position[1] == x:
                    distance += 10
                else:
                    distance += 10
                if find_count >54*42:
                    distance = 1000
                    break
                position = next_position

            # Loop again and fill gradient map with correct distance
            gradient_map[y][x] = distance
    return gradient_map

with open('H:/GRK/CrowdMovmentSimulation-master/Data/ETH/direct_data.pickle', 'rb') as file:
    direction_dict = pickle.load(file)
print(len(direction_dict))
final_direction_dict = dict()
remain_person_id = list(direction_dict.keys())
for p in remain_person_id:
    final_direction_dict[p] = dict()
with open('H:/GRK/CrowdMovmentSimulation-master/Data/ETH/person.origin.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    # 跳过标题行
    next(reader)  # 跳过第一行
    # 逐行读取并处理数据
    for row in reader:
        # 解析每一行数据
        person_id = int(float(row[0]))
        direction_frame = list(direction_dict[person_id].keys())
        pad_init_frame_index = direction_frame[0]  # 假设第三列是浮点数，转换为整数
        pad_exit_frame_index = pad_init_frame_index+7
        pad_end_frame_index = direction_frame[-1]  # 假设第四列是浮点数，转换为整数
        final_direction_dict[person_id]['exit_frame_index'] = pad_exit_frame_index
        i=0
        while(i < len(direction_dict[person_id])):
            pad_mid_frame_index = direction_frame[i]
            direction_raw = direction_dict[person_id][pad_init_frame_index]
            direction_raw_list = direction_raw.tolist()
            direction_field = gradient_from_direction_map(direction_raw_list)
            final_direction_dict[person_id][pad_mid_frame_index]=direction_field
            i+=1

with open('H:/GRK/CrowdMovmentSimulation-master/Data/ETH/final_direction.pickle', 'wb') as file:
        pickle.dump(final_direction_dict, file)