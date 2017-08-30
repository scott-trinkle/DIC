import numpy as np
from PIL import Image, ImageDraw


def make_phantom(rows, cols):
    '''This function generates a phantom for use in testing the propagation
    of variance in gamma and theta into the OPL

    Parameters
    __________
    rows : int
        Number of rows in the image
    cols : int
        Number of columns in the image

    Returns
    _______
    OPL : ndarray
        Phantom image as ndarray
    '''

    OPL = 50 * np.ones((rows, cols))
    OPL = Image.fromarray(OPL)

    draw = ImageDraw.Draw(OPL)

    # Rectangle 1
    x0 = cols * 0.045
    x1 = cols * 0.25
    y0 = rows * 0.08
    y1 = rows * 0.45
    draw.rectangle([x0, y0, x1, y1], fill=50.3)

    # Rectangle 2
    x0 = cols * 0.10
    x1 = cols * 0.2
    y0 = rows * 0.15
    y1 = rows * 0.40
    draw.rectangle([x0, y0, x1, y1], fill=49.9)

    # Rectangle 3
    x0 = cols * 0.15
    x1 = cols * 0.45
    y0 = rows * 0.6
    y1 = rows * 0.9
    draw.rectangle([x0, y0, x1, y1], fill=50.3)

    # Circle 1
    x0 = cols * 0.70
    x1 = cols * 0.75
    y0 = rows * 0.60
    y1 = rows * 0.65
    draw.ellipse([x0, y0, x1, y1], fill=49.8)

    # Circle 2
    x0 = cols * 0.80
    x1 = cols * 0.90
    y0 = rows * 0.50
    y1 = rows * 0.60
    draw.ellipse([x0, y0, x1, y1], fill=50.3)

    # Circle 3
    x0 = cols * 0.30
    x1 = cols * 0.55
    y0 = rows * 0.70
    y1 = rows * 0.95
    draw.ellipse([x0, y0, x1, y1], fill=50.2)

    # Ellipse 1
    x0 = cols * 0.45
    x1 = cols * 0.75
    y0 = rows * 0.25
    y1 = rows * 0.45
    draw.ellipse([x0, y0, x1, y1], fill=49.7)

    OPL = np.array(OPL)
    return OPL


def calculate_OPL_var(OPL, experiment):
    '''Takes a phantom (OPL) and an Experiment instance and calculates the 
    variance of the OPL image for equal and non-equal dose conditions for 
    all acquisition approaches. 

    Returns n x m "varOPL" array, where n is [True, False] dose conditions, 
    and m is experiment.approaches'''

    N, M = OPL.shape

    dy, dx = np.gradient(OPL)  # cartesian gradients
    gamma = np.sqrt(dy**2 + dx**2)  # gradient magnitude
    theta = np.arctan2(dy, dx)  # gradient azimuth on [-pi/2, pi/2]
    theta[theta < 0.0] += 2 * np.pi  # shifting to [0, 2pi]
    G = gamma * np.exp(1j * theta)

    varOPL = np.zeros((2, len(experiment.approaches)))

    # iterating over equal/nonequal dose and all approaches
    for dose_num, equalize_dose in enumerate([True, False]):
        for app_num, approach in enumerate(experiment.approaches):

            # NumPy generator functions for CRLB of sigma_gamma and sigma_theta
            gamma_func, theta_func = experiment.derive_CRLBs(equalize_dose=equalize_dose,
                                                             approach=approach)
            # Ignores divide by 0 errors
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_gamma = gamma_func(gamma, theta)
                sigma_theta = theta_func(gamma, theta)

            # Most of the image has gamma = 0, sigma_gamma = inf
            sigma_gamma[gamma == 0] = 0
            sigma_theta[gamma == 0] = 0

            # law of propagation of variance
            varG = sigma_gamma ** 2 + gamma ** 2 * sigma_theta ** 2
            K, L = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
            s = (1 / (K**2 + L**2)).sum()
            varOPL[dose_num, app_num] = (N * M)**-2 * s * varG.sum()

    return varOPL
