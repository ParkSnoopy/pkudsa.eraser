# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:19:49 2023

@author: admin
"""

class AbstractPlayer:
    
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action
        """
    
    def cache(self, experience):
        """
        Add the experience to memory
        """
    
    def recall(self):
        """
        Sample experiences from memory
        """
    
    def learn(self):
        """
        Update online action value (Q) function with a batch of experiences
        """
