import numpy as np
import matplotlib.pyplot as plt
from crlb import derive_CRLBs
from experiment import Experiment

a, b = 0.1, 0.3
size = (100, 100)

gamma = (b - a) * np.random.random(size) + a
theta = 2 * np.pi * np.random.random(size)

G = gamma * np.exp(1j * theta)

# Need a function that gets forward model parameters from experiment parameters
# to plug into derive_CRLBs. So implement that in generate_data, then bring it
# here.
