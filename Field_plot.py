import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from model.environment.environment_enum import Env
Data_sorce = 'UNIV'
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+'/final_direction.pickle', 'rb') as file:
    direction_dict = pickle.load(file)
with open('H:/GRK/CrowdMovmentSimulation-master/Data/'+Data_sorce+'/astar_direction - 副本.pickle', 'rb') as file:
    astar_direction_dict = pickle.load(file)
person_id_list = list(direction_dict.keys())
plot_person = person_id_list[123]
print(plot_person)
person_data = direction_dict[plot_person]
person_frame = list(person_data.keys())
print(len(person_frame))
print(person_frame)
plot_frame = person_frame[57]
plot_field = person_data[plot_frame]
plot_astar_field = astar_direction_dict[plot_person]

for i in range(len(plot_field)):
    for j in range(len(plot_field[i])):
        if plot_field[i][j] == Env.OBSTACLE:
            plot_field[i][j] = 500

for i in range(len(plot_astar_field)):
    for j in range(len(plot_astar_field[i])):
        if plot_astar_field[i][j] == Env.OBSTACLE:
            plot_astar_field[i][j] = 500
plot_field = np.array(plot_field)
plot_astar_field = np.array(plot_astar_field)
print(plot_astar_field)
# plt.figure(figsize=(9, 6),dpi=500)
plt.contourf(plot_field,levels=25)
plt.scatter(41, 16, color='#DA635C', marker='*', s=200, edgecolor='black', linewidth=1,zorder=3)
#plt.clabel(plt.contour(plot_field), inline=True, fontsize=8)
plt.gca().invert_yaxis()
#plt.savefig('H:/GRK/STG论文/TCSS返修/fig9_field_165_'+str(plot_frame)+'.png',dpi=500)
#plt.savefig('H:/GRK/STG论文/TCSS返修/fig9_field_165_astar.png',dpi=500)
plt.show()

plt.imshow(plot_field, cmap='hot', interpolation='nearest')
plt.colorbar()  # 显示颜色条
plt.show()