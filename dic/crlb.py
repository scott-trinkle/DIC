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

    gamma, theta = expo.set_gamma_theta(sample_size)

    # iterating over 'equal' or 'non-equal' dose options
    for dose_num, equalize_dose in enumerate((True, False)):

        # iterating over acquisition approaches
        for app_num, approach in enumerate(expo.approaches):

            # Calls derive_CRLBs function
            sigma_g_func, sigma_t_func = expo.derive_CRLBs(
                equalize_dose=equalize_dose, approach=approach)

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
