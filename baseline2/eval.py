import torch
import datetime
from pathlib import Path
from game_rl.utils import MetricLogger
from game_rl.strategy import Mario
from gym.wrappers import FrameStack
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from game_rl.wrappers import *
import time

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human',apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None, bias_action=None)

mario.load('/media/soham/OS/Soham/3-1/popl/Popl_project/baseline2/checkpoints/2022-12-09T11-07-03/mario_net_0.chkpt')


episode = 5
for e in range(episode):
    with torch.no_grad():
        # Play the game!
        state = env.reset()
        while True:
            # Render Mario environment
            env.render()
            # Run agent on the state
            action = mario.act(state)
            
            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Update state
            state = next_state
            # Check if end of game
            if done or info["flag_get"]:
                break

