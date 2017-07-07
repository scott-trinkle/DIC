'''
Version 1.0 

This calculates CRLB_gamma and CRLB_theta for a single physical setup and 
generates plots for each in its own figure. 

It does NOT include an equalized dose constraint, and the plots are not 
pretty. 
'''

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#############################################################################
#                      Choose physical setup                                #
#############################################################################

lens = 40  # Choose either 40x or 100x objective lens
weak_grad = False  # Choose a weak gradient specimen (True) or regular (False)
approach = '2x2'  # Choose '2x2', '2x3' or '2x4'
k = 1  # "illumination rate" constant

#############################################################################
#                      Variables                                            #
#############################################################################

lamda = 546  # illumination wavelength in nm

if not weak_grad:  # a regular specimen
    bias_j = 0.15 * lamda
else:  # a weak gradient specimen
    bias_j = 0.05 * lamda

if approach == '2x2':
    bias = [-bias_j, bias_j]
elif approach == '2x3':
    bias = [-bias_j, 0, bias_j]
elif approach == '2x4':
    bias = [0, lamda / 4, lamda / 2, 3 * lamda / 4]
else:
    print("Please select '2x2', '2x3' or '2x4' for 'approach'")

Nb = len(bias)

if lens == 40:
    d = 255  # shear distance in nm
    I_ent = 50 * k  # exposure time in us
elif lens == 100:
    d = 100  # shear distance in nm
    I_ent = 200 * k  # exposure time in us
else:
    print("Please select 40 or 100 for 'lens'")

Ic = 0.01 * I_ent  # stray light intensity

# 'vars' will be used to plug these numerical values into the final
# symbolic expressions
# vars = {'I_ent': I_ent, 'lamda': lamda, 'd': d, 'Ic': Ic,
#         'gamma': 0.2, 'theta': pi, 'Ic': Ic}

#############################################################################
#                Deriving the Fisher Matrix and Symbolic CRLBs              #
#############################################################################

# Declaring symbolic variables
# I_ent, lamda, d, gamma, theta, Ic = symbols(
#     'I_ent, lamda, d, gamma, theta, Ic')

gamma, theta = symbols('gamma theta')

# Forward model for shear measurement in shear in x direction.
# Yields a list of expressions for each value in 'bias'
I1 = [I_ent * sin(pi / lamda * (b + sqrt(2) * d * gamma *
                                cos(theta)))**2 + Ic for b in bias]

# Forward model for shear measurement in y direction
# Yields a list of expressions for each value in 'bias'
I2 = [I_ent * sin(pi / lamda * (b + sqrt(2) * d * gamma *
                                sin(theta)))**2 + Ic for b in bias]

# First derivatives
dI1dg = [I1[j].diff(gamma) for j in range(Nb)]
dI1dt = [I1[j].diff(theta) for j in range(Nb)]

dI2dg = [I2[j].diff(gamma) for j in range(Nb)]
dI2dt = [I2[j].diff(theta) for j in range(Nb)]

# Fisher matrix elements
F11 = np.array([1 / I1[j] * dI1dg[j]**2 + 1 / I2[j] *
                dI2dg[j]**2 for j in range(Nb)]).sum()

F22 = np.array([1 / I1[j] * dI1dt[j]**2 + 1 / I2[j] *
                dI2dt[j]**2 for j in range(Nb)]).sum()

F12 = np.array([1 / I1[j] * dI1dg[j] * dI1dt[j] + 1 / I2[j] *
                dI2dg[j] * dI2dt[j] for j in range(Nb)]).sum()

F = Matrix([[F11, F12], [F12, F22]])  # Fisher matrix itself
det = F.det()  # determinant, used to calcualte inverse

sigma_g = F22 / det  # equivalent to [F^-1]_11
sigma_t = F11 / det  # equivalent to [F^-1]_22


#############################################################################
#                      Plotting CRLBs on gamma-theta space                  #
#############################################################################

sigma_g_func = lambdify((gamma, theta), sigma_g, 'numpy')
sigma_t_func = lambdify((gamma, theta), sigma_t, 'numpy')

pix = 100

gamma = np.linspace(0.1, 0.3, pix)  # gradient magnitude in nm/nm
theta = np.linspace(0, 2 * np.pi, pix)  # gradient azimuth in radians

gamma, theta = np.meshgrid(gamma, theta)

sigma_g_num = sigma_g_func(gamma, theta)
sigma_t_num = sigma_t_func(gamma, theta)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.view_init(elev=35, azim=40)

surf1 = ax.plot_surface(gamma, theta, sigma_g_num, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
ax.set_xlabel(r'$\gamma(x,y)$')
ax.set_ylabel(r'$\theta (x,y)$')
ax.set_zlabel(r'$\sigma_{\gamma}^2$')
plt.show(block=False)


fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.view_init(elev=35, azim=40)

surf1 = ax.plot_surface(gamma, theta, sigma_t_num, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
ax.set_xlabel(r'$\gamma(x,y)$')
ax.set_ylabel(r'$\theta (x,y)$')
ax.set_zlabel(r'$\sigma_{\theta}^2$')
plt.show(block=False)
