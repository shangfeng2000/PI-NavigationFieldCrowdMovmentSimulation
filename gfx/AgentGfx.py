from OpenGL.GL import *
from math import *
from model.gradient_agent.AgentGradient import Agent
#from model.agent.Agent import Agent

class AgentGfx:
    def __init__(self, agent_id, direct_id, position: [float, float], map_position: [int, int], angle: float, color: [float, float, float],
                 maze, velocity, astar_map, direct, exits, density, collision):
        self.map_position = map_position
        self.position = position
        self.angle = radians(angle)
        self.color = color
        #exits = list(zip(range(90, 91), [99] * 1))
        self.agent = Agent(agent_id, direct_id, (map_position[0], map_position[1]), exits, direct, astar_map, collision, velocity, density)
        self.fx_pos = (0, 0)

    def move(self, density_map, collision_map):
        result = self.agent.move(density_map, collision_map)
        return result

    def draw(self, radius):
        direction = [cos(self.agent.facing_angle) + self.position[1], sin(self.agent.facing_angle) + self.position[0]]

        glColor3f(self.color[0], self.color[1], self.color[2])

        # draw circle

        posx, posy = self.fx_pos
        sides = 8

        # draw circle filling
        glBegin(GL_POLYGON)
        for vertex in range(sides):
            angle = float(vertex) * 2.0 * pi / sides
            glVertex2f(cos(angle) * radius + posx, sin(angle) * radius + posy)
        glEnd()

        # draw circle outline
        glLineWidth(0.1)
        glColor3f(1.0, 1.0, 1.0)

        glBegin(GL_LINE_LOOP)
        for vertex in range(sides):
            angle = float(vertex) * 2.0 * pi / sides
            glVertex2f(cos(angle) * radius + posx, sin(angle) * radius + posy)
        glEnd()

        # draw direction line
        vec = [(direction[1] - self.position[0]), (direction[0] - self.position[1])]

        vec_len = sqrt(pow(vec[0], 2) + pow(vec[1], 2)) / 5
        vec[0] = vec[0] / vec_len * radius
        vec[1] = vec[1] / vec_len * radius

        glLineWidth(0.1)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex2f(self.position[0], self.position[1])
        glVertex2f(self.position[0] + vec[0], self.position[1] - vec[1])
        glEnd()
