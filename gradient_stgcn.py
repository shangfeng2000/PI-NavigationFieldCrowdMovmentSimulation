import csv
import glfw
import pickle
from OpenGL.GL import *
import random
from gfx.MazeTexture import MazeTexture
from model.direction_map.DirectionMap import DirectionMap
from model.environment.line import Point
from gfx.AgentManager import AgentManager
from resources.handling.reading import load_direction_from_file, load_map_from_file
from model.gradient.gradient_map import gradient_from_direction_map
from model.environment.environment_enum import Env
from random import randint

def initiallize(Data_sorce):
    #初始化地图
    maze_filename = "Data/"+Data_sorce+"/maze_new.txt"
    maze = load_map_from_file(maze_filename)
    #初始化导航场数据
    with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+'/final_direction.pickle', 'rb') as file:
        direction_dict = pickle.load(file)
    with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+'/astar_direction.pickle', 'rb') as file:
        astar_direction_dict = pickle.load(file)
    #初始化行人数据
    ped_data = dict()
    ped_filename = 'Data/'+Data_sorce+'/person.csv'  # 替换为你的 CSV 文件路径
    test_p = 0
    with open(ped_filename, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            if len(row) >= 5:  # 确保至少有行人id、时间、初始x坐标和初始y坐标这四列数据
                pedestrian_id = int(row[0])
                ped_enter_time = direction_dict[pedestrian_id]['exit_frame_index']
                initial_x = int(row[1])
                initial_y = int(row[2])
                init_velocity = int(row[3])
                ped_data[pedestrian_id] = {
                    'ped_id':pedestrian_id,
                    'enter_time': ped_enter_time,
                    'initial_position': (initial_x, initial_y),
                    'initial_velocity':init_velocity
                }
            test_p += 1
            #if test_p>20:
                #break
    #初始化场数据
    for ped_id in ped_data.keys():
        #direction_field_filename ='Data/Direct_field/'+str(ped_id)+'.txt'
        #astar_field_filename = 'Data/Astar_field/' + str(ped_id) + '.txt'
        direction_frame = list(direction_dict[ped_id].keys())
        direction_raw = direction_dict[ped_id][direction_frame[1]]
        direction = direction_raw
        astar_direction = astar_direction_dict[ped_id]
        ped_data[ped_id]['direction'] = direction
        ped_data[ped_id]['astar_direction'] = astar_direction
    return ped_data,maze,direction_dict

Data_sorce = 'univ'
Driven_freq = 12
ped_data,maze,direction_data= initiallize(Data_sorce)
maze_original = maze
#初始化glfw库
if not glfw.init():
    exit(1)

# global intensity
global global_intensity
global_intensity = 20
window = glfw.create_window(1280, 720, "Modelowanie i Symulacja Systemów - Symulacja (0 FPS)", None, None)#负责创建一个新的OpenGL环境和窗口
glfw.make_context_current(window) #使 OpenGL 上下文成为当前状态

simulation_running = True

if not window:
    glfw.terminate() #终止glfw
    exit(1)

w_prev = 1280
h_prev = 720
offset = 20 #窗口的外围边界厚度
tile_size = [(w_prev - 2 * (offset + 1)) / len(maze[0]), (h_prev - 2 * (offset + 1)) / len(maze)] #根据先前的窗口大小或固定值、偏移量和迷宫的行列数来计算每个迷宫单元格的宽度和高度


agents = AgentManager(tile_size, w_prev, h_prev, offset, maze, ped_data,direction_data ,0, Driven_freq)

mazeTexture = MazeTexture(maze_original, w_prev, h_prev, offset, tile_size)

# 编写鼠标回调函数
def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        pos_x, pos_y = glfw.get_cursor_pos(window) #获取光标的屏幕位置
        pos_x -= offset
        pos_y -= offset
        pos = [-1, -1]
        for it in range(len(maze)):
            if tile_size[1] * it < pos_y < tile_size[1] * (it + 1):
                pos[0] = it
        for it in range(len(maze[0])):
            if tile_size[0] * it < pos_x < tile_size[0] * (it + 1):
                pos[1] = it
        if pos[0] != -1 and pos[1] != -1 and maze[pos[0]][pos[1]] != 1:
            agents.add_new(pos, 33.0, [.0, .0, .9])
# 鼠标左键按下时，根据鼠标的位置在迷宫中找到对应的位置，并在该位置创建一个新的代理对象

#按键输入回调函数
def key_callback(window, key, scancode, action, mods):
    global global_intensity
    # 加号键盘，增加强度
    if key == glfw.KEY_KP_ADD and action == glfw.RELEASE:
        global_intensity += 10
        if global_intensity > 100:
            global_intensity = 100
    if key == glfw.KEY_KP_SUBTRACT and action == glfw.RELEASE:
        global_intensity -= 10
        if global_intensity < 0:
            global_intensity = 0
    if key == glfw.KEY_KP_ADD and action == glfw.RELEASE:
        print("Wcisnalem!")
    # 空格键控制运行状态
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        global simulation_running
        simulation_running = not (simulation_running and True)


glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_key_callback(window, key_callback)

old_step_time = glfw.get_time() #计时器
previous_time = glfw.get_time()
frame_count = 0
cur_ped_data=ped_data #初始化时刻当前行人信息等于初始行人信息

while not glfw.window_should_close(window):

    current_time = glfw.get_time() #程序运行的时间
    #print(current_time) #可以实时调取仿真当前时间
    frame_count += 1

    if simulation_running:
        agents.step(frame_count)

    if current_time - previous_time >= 1.0:
        title = "Crowd Simulation ( " + str(frame_count) + " FPS | Number Of Agents: " + str(
            len(agents.agent_list)) + " )" + " intensity: " + str(global_intensity)
        glfw.set_window_title(window, title)
        #frame_count = 0
        previous_time = current_time

    glfw.poll_events() #处理挂起的事件，然后立即返回

    glClearColor(0.0, 0.0, 0.0, 1.0) #设置颜色
    glClear(GL_COLOR_BUFFER_BIT) #清除颜色缓冲区

    w, h = glfw.get_window_size(window)

    if w != w_prev or h != h_prev:
        w_prev = w
        h_prev = h
        tile_size[0] = (w - 2 * (offset + 1)) / len(maze[0])
        tile_size[1] = (h - 2 * (offset + 1)) / len(maze)
        agents.set_client_tile_size(w, h, tile_size)
        mazeTexture.reconstruct(w, h, tile_size)

    glMatrixMode(GL_PROJECTION) #将当前矩阵指定为投影矩阵
    glLoadIdentity() #恢复初始坐标系，单位矩阵
    glOrtho(0, w, 0, h, -10, 10) #定义了一个正交投影矩阵，用于指定渲染的可见区域和深度范围

    glMatrixMode(GL_MODELVIEW) #当前矩阵模式切换回模型视图模式，用于后续的模型视图变换操作

    mazeTexture.draw()

    agents.draw_all()

    glfw.swap_buffers(window) #交换缓冲区，绘制图像

    intensity = randint(0, 100)
    keys_to_delete = []
    for ped in cur_ped_data.keys():
        if cur_ped_data[ped]['enter_time']<=frame_count:
            #创建该行人
            pos = [cur_ped_data[ped]['initial_position'][0],cur_ped_data[ped]['initial_position'][1]]
            agents.add_new(ped, pos, 33.0, [.0, .0, .9])
            keys_to_delete.append(ped)
    for key in keys_to_delete:
        del cur_ped_data[key]
    #如果所有人已经进入完并且当前没有人，则停止仿真
    #and len(agents.agent_list)<=0
    if (len(cur_ped_data)==0):
        file_name = 'H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+'/行人交互影响参数/Trajectory.csv'
        agents.save_traj(file_name)
        simulation_running = False
        end_time = glfw.get_time()
        simulate_time = end_time-old_step_time
        print("仿真运行时间",simulate_time)
        glfw.terminate()
        exit(1)

mazeTexture.release()
glfw.terminate()