import numpy as np
import matplotlib.pyplot as plt
from dic.misc import make_phantom
from dic.experiment import Experiment

'''Using this script to experiment with propagating variance to OPL'''

OPL = make_phantom(512, 512)

dy, dx = np.gradient(OPL)  # generates cartesian gradient images

gamma = np.sqrt(dy**2 + dx**2)  # gradient magnitude
theta = np.arctan2(dy, dx)  # gradient azimuth on [-pi/2, pi/2]...
theta[theta < 0.0] += 2 * np.pi  # ...shifting to [0, 2pi]

experiment = Experiment(lens=40, weak_grad=True)
g_func, t_func = experiment.derive_CRLBs()

sigma_g = g_func(gamma, theta)
sigma_t = t_func(gamma, theta)

sigma_g[gamma == 0] = 0
sigma_t[gamma == 0] = 0

var_grad = np.exp(2 * 1j * theta) * (sigma_g**2 - gamma**2 * sigma_t**2)

Us = np.fft.fft2(var_grad)

plt.imshow(abs(Us))
plt.colorbar()
plt.show()
