'''
Version 2.0

User specifies lens and weak_gradient, the script calculates
CRLBs gamma (gradient magnitude) and theta (gradient azimuth)
for all acquisition approaches and for both "equal" and "non-equal"
dose conditions. Stores data into a single 2x3x100x100 np.array.

Creates a single figure with 4 cartesian surface plots of data: sigma_g
and sigma_t for equal and non-equal dose conditions. Each plot includes
all three acquisition approaches.
'''

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


def derive_CRLBs(I_ent, lamda, bias, d, Ic):
    # initializing symbolic variables
    gamma, theta = symbols('gamma theta')

    num_frames = len(bias)

    # Forward model for shear measurement in x direction.
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I1 = [I_ent * sin(pi / lamda * (b + sqrt(2) * d * gamma *
                                    cos(theta)))**2 + Ic for b in bias]

    # Forward model for shear measurement in y direction
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I2 = [I_ent * sin(pi / lamda * (b + sqrt(2) * d * gamma *
                                    sin(theta)))**2 + Ic for b in bias]

    # Calculates list of first partial derivatives for I1 and I2
    dI1dg = [I1[frame_num].diff(gamma) for frame_num in range(num_frames)]
    dI1dt = [I1[frame_num].diff(theta) for frame_num in range(num_frames)]

    dI2dg = [I2[frame_num].diff(gamma) for frame_num in range(num_frames)]
    dI2dt = [I2[frame_num].diff(theta) for frame_num in range(num_frames)]

    # Derives Fisher matrix elements for poisson variables
    # First generates a np.array containing the expressions for each frame_num,
    # then sums them into a single sympy expression
    F11 = np.array([1 / I1[frame_num] * dI1dg[frame_num]**2 + 1 / I2[frame_num] *
                    dI2dg[frame_num]**2 for frame_num in range(num_frames)]).sum()

    F22 = np.array([1 / I1[frame_num] * dI1dt[frame_num]**2 + 1 / I2[frame_num] *
                    dI2dt[frame_num]**2 for frame_num in range(num_frames)]).sum()

    F12 = np.array([1 / I1[frame_num] * dI1dg[frame_num] * dI1dt[frame_num] + 1 / I2[frame_num] *
                    dI2dg[frame_num] * dI2dt[frame_num] for frame_num in
                    range(num_frames)]).sum()

    F = Matrix([[F11, F12], [F12, F22]])  # generates the Fisher matrix itself
    det = F.det()  # determinant, used to calcualte inverse

    # Defining the CRLBs for sigma and theta based on standard inverse formula for
    # square matrices. Note: sympy has a matrix inverse function, but it is
    # very slow for these huge expressions.
    sigma_g = F22 / det  # equivalent to [F^-1]_11
    sigma_t = F11 / det  # equivalent to [F^-1]_22

    # Sympy function that converts a sympy expression into a numpy-based python
    # function
    sigma_g_func = lambdify((gamma, theta), sigma_g, 'numpy')
    sigma_t_func = lambdify((gamma, theta), sigma_t, 'numpy')

    return sigma_g_func, sigma_t_func


#############################################################################
#                      Choose physical setup                                #
#############################################################################

lens = 40  # Choose either 40x or 100x objective lens
weak_grad = False  # Choose a weak gradient specimen(True) or regular(False)
approaches = ['2x2', '2x3', '2x4']  # acquisition approaches
k = 0.1  # "illumination rate" constant

#############################################################################
#                      Variables                                            #
#############################################################################

sample_size = 100  # used later to sample gamma/theta space
lamda = 546  # illumination wavelength in nm
Na = len(approaches)  # 3, for now

# initializing data arrays:
# indeces are  (True_or_False, acq_approach, gamma_vals, theta_vals)
sigma_g_num = np.zeros((2, Na, sample_size, sample_size))
sigma_t_num = np.zeros((2, Na, sample_size, sample_size))

if not weak_grad:  # a regular specimen
    bias_j = 0.15 * lamda
else:  # a weak gradient specimen
    bias_j = 0.05 * lamda

if lens == 40:
    d = 255  # shear distance in nm
    I_ent = 50 * k  # exposure time in us (stand-in for entrance intensity)
elif lens == 100:
    d = 100  # shear distance in nm
    I_ent = 200 * k  # exposure time in us (stand-in for entrance intensity)
else:
    raise ValueError('Please enter lens = 40 or lens = 100')

# Generates numerical values for gamma/theta space
gamma = np.linspace(0.1, 0.3, sample_size)  # gradient magnitude in nm/nm
theta = np.linspace(0, 2 * np.pi, sample_size)  # gradient azimuth in radians
gamma, theta = np.meshgrid(gamma, theta)  # needed for 3D plots


# Top level loop: iterating over 'equal' or 'non-equal' dose options
for dose_num, dose_tf in enumerate((True, False)):

    equalize_dose = dose_tf  # bool

    # Second level loop: iterating over acquisition approaches
    for app_num, approach in enumerate(approaches):

        # gets the second numerical value in approach
        num_frames = int(approach.split('x')[-1])

        # defines the bias vector based on acquisiiton approach
        # also defines entrance intensity I_ent based on equalize_dose
        if approach == '2x2':
            bias = [-bias_j, bias_j]
            I_ent /= num_frames if equalize_dose is True else 1
        elif approach == '2x3':
            bias = [-bias_j, 0, bias_j]
            I_ent /= num_frames if equalize_dose is True else 1
        elif approach == '2x4':
            bias = [0, lamda / 4, lamda / 2, 3 * lamda / 4]
            I_ent /= num_frames if equalize_dose is True else 1

        Ic = 0.01 * I_ent  # stray light intensity

        # Grouping all physical parameters into a dictionary
        params = dict(I_ent=I_ent, lamda=lamda, bias=bias, d=d, Ic=Ic)

        # Generate the sigma_g and sigma_t functions
        sigma_g_func, sigma_t_func = derive_CRLBs(**params)

        # fills in numerical values for 'equal' or 'nonequal' dose,
        # and all acquisition approaches for a given lens and gradient type
        sigma_g_num[dose_num, app_num, :, :] = sigma_g_func(gamma, theta)
        sigma_t_num[dose_num, app_num, :, :] = sigma_t_func(gamma, theta)

elev, azim = 15, -70
c = ['maroon', 'seagreen', 'royalblue']


def make_surface_plots(ax, sigma_num, c=c):
    x_tick = np.arange(0.1, 0.35, 0.05)
    ax.set_xticks(x_tick)
    ax.set_xlabel(r'$\gamma(x,y)$')

    y_tick = np.arange(0, 2.50, 0.50)
    y_label = [r'0', r'$\frac{\pi}{2}$',
               r'$\pi$', r'$\frac{3\pi}{2}$',   r'$2\pi$']
    ax.set_yticks(y_tick * np.pi)
    ax.set_yticklabels(y_label)
    ax.set_ylabel(r'$\theta (x,y)$')

    ax.view_init(elev=elev, azim=azim)

    for jj in range(Na):
        ax.plot_surface(
            gamma, theta, sigma_num[jj], linewidth=0, color=c[jj],
            antialiased=True, alpha=0.70)

    return ax


fig = plt.figure(1, figsize=(14, 7.8))
fig.suptitle('{}x Objective, {} gradient'.format(
    lens, 'Weak' if weak_grad is True else 'Normal'), weight='bold')

ax_g1 = fig.add_subplot(221, projection='3d')
ax_g1 = make_surface_plots(ax_g1, sigma_g_num[0])
ax_g1.set_title(r'Equal dose CRLB for $\sigma_{\gamma}^2$')

ax_t1 = fig.add_subplot(222, projection='3d')
ax_t1 = make_surface_plots(ax_t1, sigma_t_num[0])
ax_t1.set_title(r'Equal dose CRLB for $\sigma_{\theta}^2$')

ax_g2 = fig.add_subplot(223, projection='3d')
ax_g2 = make_surface_plots(ax_g2, sigma_g_num[1])
ax_g2.set_title(r'Non-equal dose CRLB for $\sigma_{\gamma}^2$')

ax_t2 = fig.add_subplot(224, projection='3d')
ax_t2 = make_surface_plots(ax_t2, sigma_t_num[1])
ax_t2.set_title(r'Non-equal dose CRLB for $\sigma_{\theta}^2$')

fig.legend([Line2D([], [], linestyle='-', color=c[kk])
            for kk in range(Na)],
           [approach + ' frames' for approach in approaches], numpoints=1, loc=10,
           fontsize='large')

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# plt.savefig('grad_surface_plots/{}x_{}.png'.format(
#     lens, 'weak' if weak_grad is True else 'normal'), dpi=215)
