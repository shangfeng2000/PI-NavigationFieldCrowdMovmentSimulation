import moderngl
import numpy
import numpy as np
from OpenGL.GL import *
import time
import random
from math import *


def compute_final_pos(desired: [int, int]):
    return -1.0 + 2.0 * desired[0] / 800, -1.0 + 2.0 * desired[1] / 600


class InstancedAgentRenderer:
    def __init__(self):
        self.agents = list()
        self.dirty_update = False
        self.after_first_add = False

        self.ctx = moderngl.create_context()
        self.ctx.line_width = 0.1

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 v;
                in vec2 pos;
                in vec3 inColor;
                out vec3 color;
                void main() { gl_Position = vec4(pos + v, 0.0, 1.0); color = inColor; }
            ''',
            fragment_shader='''
                #version 330
                in vec3 color;
                out vec4 finalColor;
                void main() { finalColor = vec4(color, 1.0); }
            ''',
        )

        self.prog_dir = self.ctx.program(
            vertex_shader='''
                        #version 330
                        in vec2 v;
                        in vec2 pos;
                        in vec3 inColor;
                        in vec2 angle;
                        out vec3 color;
                        void main() { gl_Position = vec4(pos + v * angle, 0.0, 1.0); color = inColor; }
                    ''',
            fragment_shader='''
                        #version 330
                        in vec3 color;
                        out vec4 finalColor;
                        void main() { finalColor = vec4(color, 1.0); }
                    ''',
        )

        self.radius = 0.02
        self.sides = 16

        self.vertices = np.array([])
        self.colors = np.array([])
        self.positions = np.array([])

        self.colors_white = np.array([])

        self.angle = np.array([])

        self.vertices_direction = np.array([])
        self.colors_direction = np.array([])

        self.vertex_buffer = None
        self.color_buffer = None
        self.color_white_buffer = None
        self.pos_buffer = None
        self.vertex_dir_buffer = None
        self.color_dir_buffer = None
        self.angle_buffer = None

        self.vao_fill = None
        self.vao_outline = None
        self.vao_direction = None

    def add(self, agent):
        self.after_first_add = True
        self.dirty_update = True

        self.agents.append(agent)
        pass

    def remove(self, agent):
        self.dirty_update = True

        self.agents.remove(agent)
        pass

    def __update(self):
        self.dirty_update = False

        self.vertices = np.array([])
        self.colors = np.array([])
        self.positions = np.array([])
        self.colors_white = np.array([])
        self.angle = np.array([])

        for agent in self.agents:
            self.colors = np.append(self.colors, agent.color)
            self.positions = np.append(self.positions, compute_final_pos(agent.position))
            self.colors_white = np.append(self.colors_white, [1.0, 1.0, 1.0])
            self.angle = np.append(self.angle, [cos(agent.angle), sin(agent.angle)])

        for vertex in range(self.sides):
            angle = float(vertex) * 2.0 * pi / self.sides
            self.vertices = np.append(self.vertices, [cos(angle) * self.radius, sin(angle) * self.radius])

        self.vertices_direction = np.array([0.0, 0.0, self.radius, self.radius])
        self.colors_direction = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.vertex_buffer = self.ctx.buffer(self.vertices.astype('f4').tobytes())
        self.color_buffer = self.ctx.buffer(self.colors.astype('f4').tobytes())
        self.color_white_buffer = self.ctx.buffer(self.colors_white.astype('f4').tobytes())
        self.pos_buffer = self.ctx.buffer(self.positions.astype('f4').tobytes())
        self.vertex_dir_buffer = self.ctx.buffer(self.vertices_direction.astype('f4').tobytes())
        self.color_dir_buffer = self.ctx.buffer(self.colors_direction.astype('f4').tobytes())
        self.angle_buffer = self.ctx.buffer(self.angle.astype('f4').tobytes())

        vao_fill_content = [
            (self.vertex_buffer, '2f', 'v'),
            (self.color_buffer, '3f /i', 'inColor'),
            (self.pos_buffer, '2f /i', 'pos'),
        ]

        vao_outline_content = [
            (self.vertex_buffer, '2f', 'v'),
            (self.color_white_buffer, '3f /i', 'inColor'),
            (self.pos_buffer, '2f /i', 'pos'),
        ]

        vao_direction_content = [
            (self.vertex_dir_buffer, '2f', 'v'),
            (self.color_dir_buffer, '3f', 'inColor'),
            (self.pos_buffer, '2f /i', 'pos'),
            (self.angle_buffer, '2f /i', 'angle'),
        ]

        self.vao_fill = self.ctx.vertex_array(self.prog, vao_fill_content)
        self.vao_outline = self.ctx.vertex_array(self.prog, vao_outline_content)
        self.vao_direction = self.ctx.vertex_array(self.prog_dir, vao_direction_content)

    def draw(self):
        if not self.after_first_add:
            return
        if self.dirty_update:
            self.__update()

        self.ctx.viewport = (0, 0, 800, 600)

        self.vao_fill.render(instances=len(self.agents), mode=GL_POLYGON)
        self.vao_outline.render(instances=len(self.agents), mode=GL_LINE_LOOP)

        self.vao_direction.render(instances=len(self.agents), mode=GL_LINES)
