from enum import Enum


class Env(Enum):
    OBSTACLE = 'obstacle'
    EXIT = 'exit'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
