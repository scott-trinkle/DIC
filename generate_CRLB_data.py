'''
Version 3.0

User specifies lens (40 or 100) and weak_gradient bool, the script calculates
CRLBs for all acquisition approaches and for both "equal" and "non-equal"
dose conditions. Saves data into a single 2x3x100x100 np.array.

'''

import numpy as np
import sympy as sym

#############################################################################
#                      Choose physical setup                                #
#############################################################################

# Choose either 40x or 100x objective lens
lens = 40

# Choose a weak gradient specimen(True) or normal(False)
weak_grad = False

# acquisition approaches
approaches = ['2x2', '2x3', '2x4']

# "illumination rate" constant.
k = 0.1

# True: gamma range is 0.0-0.3, False: 0.1-0.3
fromZero = True

#############################################################################
#                      Function Definitions                                 #
#############################################################################


def derive_CRLBs(I_ent, lamda, bias, d, Ic):
    '''
    Returns numpy-based functions that calculate the CRLB for sigma_gamma and
    sigma_theta, both as a function of gamma and theta. 

    Input parameters are physical parameters of the system. 
    '''

    # initializing symbolic variables
    gamma, theta = sym.symbols('gamma theta')

    num_frames = len(bias)  # number of frames in acquisition approach

    # Forward model for shear measurement in x direction.
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I1 = [I_ent * sym.sin(sym.pi / lamda * (b + sym.sqrt(2) * d * gamma *
                                            sym.cos(theta)))**2 + Ic for b in bias]

    # Forward model for shear measurement in y direction
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I2 = [I_ent * sym.sin(sym.pi / lamda * (b + sym.sqrt(2) * d * gamma *
                                            sym.sin(theta)))**2 + Ic for b in bias]

    # Calculates lists of first partial derivatives for I1 and I2
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

    F12 = np.array([1 / I1[frame_num] * dI1dg[frame_num] * dI1dt[frame_num] +
                    1 / I2[frame_num] * dI2dg[frame_num] *
                    dI2dt[frame_num] for frame_num in range(num_frames)]).sum()

    # generates the Fisher matrix itself
    F = sym.Matrix([[F11, F12], [F12, F22]])
    det = F.det()  # determinant, used to calculate inverse

    # Defines the CRLBs for sigma and theta based on standard inverse formula
    # for square matrices. Note: sympy has a matrix inverse function, but it is
    # very slow for these huge expressions.
    sigma_g = F22 / det  # equivalent to [F^-1]_11
    sigma_t = F11 / det  # equivalent to [F^-1]_22

    # Sympy function that converts a sympy expression into a numpy-based python
    # function
    sigma_g_func = sym.lambdify((gamma, theta), sigma_g, 'numpy')
    sigma_t_func = sym.lambdify((gamma, theta), sigma_t, 'numpy')

    return sigma_g_func, sigma_t_func


#############################################################################
#                           Defining Variables                              #
#############################################################################


sample_size = 100  # used later to sample gamma/theta space
Na = len(approaches)  # 3, for now

# initializing data arrays:
# indices are  (equalize dose?, acq approach, gamma_vals, theta_vals)
sigma_g_num = np.zeros((2, Na, sample_size, sample_size))
sigma_t_num = np.zeros((2, Na, sample_size, sample_size))

lamda = 546  # illumination wavelength in nm

if not weak_grad:  # a normal specimen
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
if fromZero:
    start = 0.0
else:
    start = 0.1

gamma = np.linspace(start, 0.3, sample_size)  # gradient magnitude in nm/nm
theta = np.linspace(0, 2 * np.pi, sample_size)  # gradient azimuth in radians
gamma, theta = np.meshgrid(gamma, theta)  # needed for 3D plots


#############################################################################
#                      Generate and save CRLB data                          #
#############################################################################


# Top level loop: iterating over 'equal' or 'non-equal' dose options
for dose_num, dose_tf in enumerate((True, False)):

    equalize_dose = dose_tf  # bool

    # Second level loop: iterating over acquisition approaches
    for app_num, approach in enumerate(approaches):

        # gets the second numerical value in approach
        num_frames = int(approach.split('x')[-1])

        # defines entrance intensity I_ent based on equalize_dose
        I_ent /= num_frames if equalize_dose else 1
        Ic = 0.01 * I_ent  # stray light intensity

        # defines the bias vector based on acquisiiton approach
        if approach == '2x2':
            bias = [-bias_j, bias_j]
        elif approach == '2x3':
            bias = [-bias_j, 0, bias_j]
        elif approach == '2x4':
            bias = [0, lamda / 4, lamda / 2, 3 * lamda / 4]

        # Grouping all physical parameters into a dictionary
        params = dict(I_ent=I_ent, lamda=lamda, bias=bias, d=d, Ic=Ic)

        # Generate the sigma_g and sigma_t functions based on params
        sigma_g_func, sigma_t_func = derive_CRLBs(**params)

        # fills in numerical values for 'equal' or 'nonequal' dose,
        # and all acquisition approaches for a given lens and gradient type
        sigma_g_num[dose_num, app_num, :, :] = sigma_g_func(gamma, theta)
        sigma_t_num[dose_num, app_num, :, :] = sigma_t_func(gamma, theta)

# Saves data

if fromZero:
    folder = 'CRLB_data_fromZero/'
else:
    folder = 'CRLB_data/'

np.save(folder + '{}_{}_sigma_g'.format(
    lens, 'weak' if weak_grad else 'normal'), sigma_g_num)
np.save(folder + '{}_{}_sigma_t'.format(
    lens, 'weak' if weak_grad else 'normal'), sigma_t_num)
np.save(folder + 'gamma.npy', gamma)
np.save(folder + 'theta.npy', theta)
