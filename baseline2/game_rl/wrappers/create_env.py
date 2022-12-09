from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from . import *


def create_env(env,skip,shape, num_stack):
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = FrameStack(env, num_stack=num_stack)
    return env