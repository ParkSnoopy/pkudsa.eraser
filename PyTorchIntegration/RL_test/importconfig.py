# -*- coding: utf-8 -*-
"""
Created on Sun May 21 00:32:25 2023

@author: admin
"""

# Add `Local` path to sys.path
import sys
from pathlib import Path
master_path = Path.cwd().parent / 'Local'
sys.path.append(str(master_path))

# importlib for relative import from `Local`
# ( can be deprecated, alter with common `import` method )
from importlib import import_module
PACKAGE_ROOT = "Local"
def module(modulename):
    print(f'importing: {modulename} ...')
    return import_module(f"{modulename}")



# All packages && modules used in current test project

# import your packages here.


torch = module("torch")
nn = torch.nn

torchvision = module("torchvision")
T = torchvision.transforms
ToTensor = torchvision.transforms.ToTensor

np = module("numpy")
plt = module("matplotlib.pyplot")

gym = module("gymnasium")
Box = gym.spaces.Box
FrameStack = gym.wrappers.FrameStack

gym_super_mario_bros = module("gym_super_mario_bros")

JoypadSpace = module("nes_py.wrappers").JoypadSpace
