from OpenGL.GL import *
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *


# render a map once to boost performance
class MazeTexture:
    def __init__(self, maze, width, height, offset, tile_size):
        self.maze = maze
        self.w = width
        self.h = height
        self.offset = offset
        self.t_s = tile_size

        self.free_color = [5, 5, 32]
        self.collision_color = [64, 64, 255]

        self.__build()

    def draw(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.render_target)

        glBegin(GL_QUADS)
        glColor3ub(255, 255, 255)

        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(0, 1)
        glVertex2f(0, self.h)
        glTexCoord2f(1, 1)
        glVertex2f(self.w, self.h)
        glTexCoord2f(1, 0)
        glVertex2f(self.w, 0)

        glEnd()
        glFlush()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def reconstruct(self, width, height, tile_size):
        self.w = width
        self.h = height
        self.t_s = tile_size
        self.__build()

    def release(self):
        glDeleteTextures(self.texture)
        glDeleteTextures(self.render_target)
        glDeleteFramebuffers(1, self.fbo)

    def __build(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.w, 0, self.h, -1, 1)

        glMatrixMode(GL_MODELVIEW)

        # generate the texture we render to, and set parameters
        self.render_target = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.render_target)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h, 0, GL_RGBA, GL_UNSIGNED_INT, None)

        # generate a "Framebuffer" object and bind it
        self.fbo = 1
        glGenFramebuffers(1, self.fbo)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        # render to the texture
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, self.render_target, 0)

        # Align viewport to Texture dimensions
        glViewport(0, 0, self.w, self.h)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # Let the magic happens
        ##########################################################################################

        glColor3ub(255, 255, 255)
        glBegin(GL_LINE_LOOP)

        glVertex2f(0 + self.offset, 0 + self.offset)
        glVertex2f(0 + self.offset, self.h - self.offset)
        glVertex2f(self.w - self.offset, self.h - self.offset)
        glVertex2f(self.w - self.offset, 0 + self.offset)

        glEnd()

        # draw map
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                self.__draw_tile(i, j, self.maze[i][j], False)  # True for helper lines

        ##########################################################################################

        # do not render to texture anymore - "switch off" rtt
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0)

    # draw tile [n, m]
    def __draw_tile(self, n, m, block_type, draw_debug):
        if block_type == 0:
            glColor3ub(self.free_color[0], self.free_color[1], self.free_color[2])
        else:
            glColor3ub(self.collision_color[0], self.collision_color[1], self.collision_color[2])

        o = self.offset + 1

        glBegin(GL_TRIANGLES) #一个矩形方块分为两个三角形快绘制

        glVertex2f(0 + o + (m * self.t_s[0]), self.h - o - (n * self.t_s[1]))
        glVertex2f(0 + o + (m * self.t_s[0]), self.h - o - (n * self.t_s[1]) - self.t_s[1])
        glVertex2f(0 + o + (m * self.t_s[0]) + self.t_s[0], self.h - o - (n * self.t_s[1]))

        glVertex2f(0 + o + (m * self.t_s[0]) + self.t_s[0], self.h - o - (n * self.t_s[1]))
        glVertex2f(0 + o + (m * self.t_s[0]), self.h - o - (n * self.t_s[1]) - self.t_s[1])
        glVertex2f(0 + o + (m * self.t_s[0]) + self.t_s[0], self.h - o - (n * self.t_s[1]) - self.t_s[1])

        glEnd()
        if draw_debug:
            glColor3ub(60, 60, 90)
            glBegin(GL_LINE_LOOP)

            glVertex2f(0 + o + (m * self.t_s[0]), self.h - o - (n * self.t_s[1]))
            glVertex2f(0 + o + (m * self.t_s[0]), self.h - o - (n * self.t_s[1]) - self.t_s[1])
            glVertex2f(0 + o + (m * self.t_s[0]) + self.t_s[0], self.h - o - (n * self.t_s[1]) - self.t_s[1])
            glVertex2f(0 + o + (m * self.t_s[0]) + self.t_s[0], self.h - o - (n * self.t_s[1]))

            glEnd()
