import numpy as np
import sympy as sym
from dic.experiment import Experiment


def generate_data(expo=Experiment(), sample_size=100):
    '''
    Returns two numpy arrays containing data for the Cramer Rao Lower Bounds
    for the standard deviations of gamma and theta for OI-DIC.

    Parameters
    __________
    expo : Experiment instance
        Contains all the experiment parameters
    sample_size : int
        Sample size for gamma and theta

    Returns
    _______
    sigma_g : ndarray
        CRLB data for gamma for the specified physical parameters.
        Shape is (2, len(approaches), sample_size, sample_size).
    sigma_t : ndarray
        CRLB data for theta for the specified physical parameters.
        Shape is (2, len(approaches), sample_size, sample_size).
    expo : Experiment object
        Updated Experiment object
    '''

    print('Running...')

    # indices are (equalize dose?, acq approach, gamma_vals, theta_vals)
    sigma_g = np.zeros((2, expo.Na, sample_size, sample_size))
    sigma_t = np.zeros((2, expo.Na, sample_size, sample_size))

    if not expo.weak_grad:  # a normal specimen
        bias_j = 0.15 * expo.lamda
    else:  # a weak gradient specimen
        bias_j = 0.05 * expo.lamda

    if expo.lens == 40:
        d = 255  # shear distance in nm
        I_ent = 50 * expo.k  # exp time in us (future: entrance intensity)
    elif expo.lens == 100:
        d = 100
        I_ent = 200 * expo.k
    else:
        raise ValueError('Please enter expo.lens = 40 or expo.lens = 100')

    gamma, theta = expo.set_gamma_theta(sample_size)

    # iterating over 'equal' or 'non-equal' dose options
    for dose_num, equalize_dose in enumerate((True, False)):

        # iterating over acquisition approaches
        for app_num, approach in enumerate(expo.approaches):

            print('Calculating CRLBs: {} dose - {}'.format(
                'Equal' if equalize_dose else 'Non-equal', approach))

            # gets the second numerical value in approach
            num_frames = int(approach.split('x')[-1])

            # redefines entrance intensity I_ent based on equalize_dose
            I_ent /= num_frames if equalize_dose else 1
            Ic = 0.01 * I_ent  # stray light intensity

            # defines the bias vector based on acquisiiton approach
            if approach == '2x2':
                bias = [-bias_j, bias_j]
            elif approach == '2x3':
                bias = [-bias_j, 0, bias_j]
            elif approach == '2x4':
                bias = [0, expo.lamda / 4,
                        expo.lamda / 2, 3 * expo.lamda / 4]

            params = dict(I_ent=I_ent, lamda=expo.lamda,
                          bias=bias, d=d, Ic=Ic)

            # Calls derive_CRLBs function
            sigma_g_func, sigma_t_func = derive_CRLBs(**params)

            # stores data for 4x4 plots. Prevents divide by zero warning.
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_g[dose_num, app_num, :, :] = sigma_g_func(gamma, theta)
                sigma_t[dose_num, app_num, :, :] = sigma_t_func(gamma, theta)

    # Formats save settings
    if expo.save:
        print('Saving CRLB data...')
        np.save(expo.filepath +
                'sigma_gamma_{}x_{}grad_{}_{}x{}'.format(
                    expo.lens, 'weak' if expo.weak_grad else 'normal',
                    'fromZero' if expo.fromZero else None,
                    sample_size, sample_size), sigma_g)

        np.save(expo.filepath +
                'sigma_theta_{}x_{}grad_{}_{}x{}'.format(
                    expo.lens, 'weak' if expo.weak_grad else 'normal',
                    'fromZero' if expo.fromZero else None,
                    sample_size, sample_size), sigma_g)
    print('Calculating CRLBs: Done!')

    return np.sqrt(sigma_g), np.sqrt(sigma_t), expo


def derive_CRLBs(I_ent, lamda, bias, d, Ic):
    '''Analytically derives the CRLBs for gamma and theta based on the
    forward model for OI-DIC microscopy. Returns numpy-based functions that
    numerically calculate the CRLBs for a given gamma and theta.

    Parameters
    _________
    I_ent : float
        Entrance intensity in []
    lamda : int
        Wavelength of the light
    bias : list
        List of bias values to use in deriving the CRLBs
    d : int
        Shear distance
    Ic : float
        Stray light intensity

    Returns
    _______
    sigma_g_func : function
        Numpy-based function that calculates the CRLB for gamma as a function
        of gamma and theta.
    sigma_t_func : function
        Numpy-based function that calculates the CRLB for theta as a function
        of gamma and theta.
    '''

    gamma, theta = sym.symbols('gamma theta')

    num_frames = len(bias)  # number of frames in acquisition approach

    # Forward model for shear measurement in x direction.
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I1 = [I_ent * sym.sin(sym.pi /
                          lamda * (b + sym.sqrt(2) * d * gamma *
                                   sym.cos(theta)))**2 + Ic for b in bias]

    # Same for y
    I2 = [I_ent * sym.sin(sym.pi /
                          lamda * (b + sym.sqrt(2) * d * gamma *
                                   sym.sin(theta)))**2 + Ic for b in bias]

    # Lists of first partial derivatives for I1 and I2
    dI1dg = [I1[frame_num].diff(gamma) for frame_num in range(num_frames)]
    dI1dt = [I1[frame_num].diff(theta) for frame_num in range(num_frames)]

    dI2dg = [I2[frame_num].diff(gamma) for frame_num in range(num_frames)]
    dI2dt = [I2[frame_num].diff(theta) for frame_num in range(num_frames)]

    # Derives Fisher matrix elements for poisson variables
    F11 = np.array([1 / I1[frame_num] * dI1dg[frame_num]**2 + 1 / I2[frame_num]
                    * dI2dg[frame_num]**2 for frame_num in range(num_frames)]).sum()

    F22 = np.array([1 / I1[frame_num] * dI1dt[frame_num]**2 + 1 / I2[frame_num]
                    * dI2dt[frame_num]**2 for frame_num in range(num_frames)]).sum()

    F12 = np.array([1 / I1[frame_num] * dI1dg[frame_num] * dI1dt[frame_num] +
                    1 / I2[frame_num] * dI2dg[frame_num] *
                    dI2dt[frame_num] for frame_num in range(num_frames)]).sum()

    F = sym.Matrix([[F11, F12], [F12, F22]])
    det = F.det()  # determinant, used to calculate inverse

    # Note: sympy has a matrix inverse function, but it is
    # very slow for these huge expressions.
    sigma_g = F22 / det  # equivalent to [F^-1]_11
    sigma_t = F11 / det  # equivalent to [F^-1]_22

    # Sympy function that converts a sympy expression into a numpy-based python
    # function
    sigma_g_func = sym.lambdify((gamma, theta), sigma_g, 'numpy')
    sigma_t_func = sym.lambdify((gamma, theta), sigma_t, 'numpy')

    return sigma_g_func, sigma_t_func
