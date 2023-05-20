# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:56:37 2023

@author: admin
"""

from importconfig import (
    torch, 
    np, 
)


# import torch

# import numpy as np
from collections import deque
import random

from abstract_player import AbstractPlayer
from mario_net import MarioNet



class Mario(AbstractPlayer):
    def __init__(self, state_dim, action_dim, save_dir):
        """
        ACT
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Mario's DNN to predict the most optimal action
        #   we implement this in the Learn section
        
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        
        self.exploration_rate = 1
        self.exploration_rate_decay = .999_999_75
        self.exploration_rate_min = .1
        self.curr_step = 0
        
        self.save_every = 5e5
        
        """
        CACHE AND RECALL
        """
        self.memory = deque(maxlen=100_000)
        self.batch_size = 32
        
        """
        TD ESTIMATE && TD TARGET
        """
        self.gamma = .9
        
        """
        UPDATING THE MODEL
        """
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=.000_25)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        """
        MAIN LEARNING
        """
        self.burnin = 1e4 # min. experiences before training
        self.learn_every = 3 # no. of experiences between updates to Q_online
        self.sync_every = 1e4 # no. of experiences between Q_target & Q_online sync
        
        
    ###########################################################################
    
    """
    ACT
    """
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        
        Parameters
        ----------
        state : LazyFrame
            A single observation of the current state, dimension is (state_dim).
        
        Returns
        -------
        action_idx : int
            An integer representing which action Mario will perform.
        
        """
        
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        
        # EXPLOIT
        else:
            state = state[0].__array__().copy() if isinstance(state, tuple) else state.__array__().copy()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # increase step
        self.curr_step += 1
        return action_idx
    
    """
    CACHE AND RECALL
    """
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        
        Parameters
        ----------
        state : LazyFrame
        next_state : LazyFrame
        action : int
        reward : float
        done : bool
        
        Returns
        -------
        None.
        
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__().copy()
        next_state = first_if_tuple(next_state).__array__().copy()
        
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)
        
        self.memory.append((
            state, 
            next_state, 
            action, 
            reward, 
            done, 
        ))
    
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    """
    TD ESTIMATE && TD TARGET
    """
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return ( reward + ( ( ( 1 - done.float() ) * self.gamma ) * next_Q ) ).float()
    
    """
    UPDATE THE MODEL
    """
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    """
    SAVE CHECKPOINT
    """
    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            {
                "model": self.net.state_dict(), 
                "exploration_rate": self.exploration_rate,
            }, 
            save_path, 
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
        
    
    """
    PUTTING THOSE ALLTOGETHER
    """
    def learn(self):
        
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Sample from memory
        state, next_state, action, reward, done = self.recall()
        
        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        
        # Get TD target
        td_tgt = self.td_target(reward, next_state, done)
        
        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        
        return ( td_est.mean().item(), loss )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        