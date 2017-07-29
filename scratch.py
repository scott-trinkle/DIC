import numpy as np
import matplotlib.pyplot as plt
from dic.experiment import Experiment

# Initial work in propagating variance to OPL

a, b = 0.1, 0.3
size = (100, 100)

np.random.seed(19680801)

gamma = (b - a) * np.random.random(size) + a
theta = 2 * np.pi * np.random.random(size)

G = gamma * np.exp(1j * theta)

expo = Experiment(lens=40,
                  weak_grad=False,
                  approaches=['2x2', '2x3', '2x4'],
                  k=0.5,
                  fromZero=True)

u2_gamma, u2_theta = expo.derive_CRLBs(equalize_dose=True, approach='2x2')

u2_G = np.exp(2j * theta) * (u2_gamma(gamma, theta) -
                             gamma**2 * u2_theta(gamma, theta))


# wx = np.zeros(size)
# wx[:, :] = np.arange(size[0]) + 1
# wy = wx.T


# FG = np.fft.fft2(G)
# OPL1 = np.fft.ifft2(G / (1j * (wx + 1j * wy)))
# OPL2 = np.fft.ifft2(np.fft.fft2(G) / (1j * (wx + 1j * wy)))
