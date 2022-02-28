import numpy as np
import matplotlib.pyplot as plt
import torch

start = np.array([0.97720214, 0.81935125, 1.70960129])
goal = np.array([0.75211621,  0.68325767, -2.02301457])

print(np.linalg.norm(start[:2] - goal[:2]))
