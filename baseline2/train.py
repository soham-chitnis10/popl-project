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

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, bias_action=None)

logger = MetricLogger(save_dir)

episodes = 10000
for e in range(episodes):

    state = env.reset()
    # Play the game!
    while True:
        # Render Mario environment
        # env.render()
        # Run agent on the state
        action = mario.act(state)
        
        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()
    mario.save()
    
    logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
