# -*- coding: utf-8 -*-
"""
Created on Thu May 18 18:19:41 2023

@author: admin
"""

from importconfig import (
    torch, 
    T, 
    gym, 
    Box, 
    np, 
)


# import torch
# from torchvision import transforms as T

# import gymnasium as gym
# from gymnasium.spaces import Box

# import numpy as np



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        Return only 
        """
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        """
        Repeat action, and sum reward
        """
        total_reward = .0
        for _ in range(self._skip):
            # Accumulate reward, and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        return obs, reward, done, trunc, info
    
    # def reset(self, options):
    #     return super().reset(options)



class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation
    
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation



class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation











