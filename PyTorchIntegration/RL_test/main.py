# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:52:11 2023

@author: admin
"""

from importconfig import (
    gym_super_mario_bros, 
    JoypadSpace, 
    FrameStack, 
    torch, 
)


# Torch and etc.
# import torch as torch

from pathlib import Path

import datetime

# OpenAI RL Tool gymnasium
# from gymnasium.wrappers import FrameStack

# NES emulator for OpenAi gymnasium
# from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI gymnasium
# import gym_super_mario_bros

# Get My Models
from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
from player import Mario
from logger import MetricLogger




env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", render_mode='rgb', apply_api_compatibility=True)

# To let JoypadSpace.reset accept 'seed' argument
class JoypadSpace(JoypadSpace):
    def reset(self, seed=None, options={}):
        return self.env.reset(seed=seed, options=options)

# Limit action space :: 0. walk right, 1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()

# next_state, reward, done, trunc, info = env.step(action=0)
# print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, num_stack=4)



use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)



mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

episodes = 100
for e in range(episodes):
    print(f"ep_{e:03d}")
    state = env.reset()

    # Play the game!
    while True:

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

    if e % 20 == 0 or e+1 == episodes:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)




