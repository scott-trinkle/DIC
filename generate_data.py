'''
add documentation
remove superfluous comments
remove warnings for divide by zero
add print statement
'''
import numpy as np
import sympy as sym


class Setup(object):

    def __init__(self, lens=40, weak_grad=False,
                 approaches=['2x2', '2x3', '2x4'],
                 k=0.1, lamda=546, fromZero=True,
                 save=False, filepath=None):
        self.lens = lens
        self.weak_grad = weak_grad
        self.approaches = approaches
        self.k = k
        self.lamda = lamda
        self.fromZero = fromZero
        self.save = save
        self.filepath = filepath

        self.Na = len(self.approaches)

    def set_gamma_theta(self, sample_size=100):

        if self.fromZero:
            start = 0.0
        else:
            start = 0.1

        self.gamma = np.linspace(start, 0.3, sample_size)
        self.theta = np.linspace(0, 2 * np.pi, sample_size)
        self.gamma, self.theta = np.meshgrid(self.gamma, self.theta)
        return self.gamma, self.theta


def generate_data(setup, sample_size=100):
    '''
    Returns two numpy arrays containing data for the Cramer Rao Lower Bounds
    for the standard deviations of gamma and theta for OI-DIC.

    Parameters
    __________
    lens : int
        Objective lens, either 40 or 100
    weak_grad : bool
        If True, calculate CRLBs for weak gradient specimens. Default is False.
    approaches : list of strings
        List of the acquisition approaches. Default is ['2x2', '2x3', '2x4']
    k : float
        "Illumination rate constant". Default is 0.1
    fromZero : bool
        If true, extend gamma range to 0.0-0.3. Otherwise it is 0.1-0.3.
        Default is True.
    save : bool
        If true, save data to filename. Default is False.
    filepath : str
        The folder to use when saving the data. Returns error if save=False.
        Default is None.
    sample_size : int
        Number of samples on the gamma, theta interval.
    lamda : int
        Wavelength of light used for the experiment (in nm).

    Returns
    _______
    sigma_g : ndarray
        CRLB data for gamma for the specified physical parameters.
        Shape is (2, len(approaches), sample_size, sample_size).
    sigma_t : ndarray
        CRLB data for theta for the specified physical parameters.
        Shape is (2, len(approaches), sample_size, sample_size).
    '''

    # indices are (equalize dose?, acq approach, gamma_vals, theta_vals)
    sigma_g = np.zeros((2, setup.Na, sample_size, sample_size))
    sigma_t = np.zeros((2, setup.Na, sample_size, sample_size))

    if not setup.weak_grad:  # a normal specimen
        bias_j = 0.15 * setup.lamda
    else:  # a weak gradient specimen
        bias_j = 0.05 * setup.lamda

    if setup.lens == 40:
        d = 255  # shear distance in nm
        # exposure time in us (stand-in for entrance intensity)
        I_ent = 50 * setup.k
    elif setup.lens == 100:
        d = 100  # shear distance in nm
        # exposure time in us (stand-in for entrance intensity)
        I_ent = 200 * setup.k
    else:
        raise ValueError('Please enter setup.lens = 40 or setup.lens = 100')

    gamma, theta = setup.set_gamma_theta(sample_size)

    # Top level loop: iterating over 'equal' or 'non-equal' dose options
    for dose_num, equalize_dose in enumerate((True, False)):

        # Second level loop: iterating over acquisition approaches
        for app_num, approach in enumerate(setup.approaches):

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
                bias = [0, setup.lamda / 4,
                        setup.lamda / 2, 3 * setup.lamda / 4]

            # Grouping all physical parameters into a dictionary
            params = dict(I_ent=I_ent, lamda=setup.lamda,
                          bias=bias, d=d, Ic=Ic)

            # Generate the sigma_g and sigma_t functions based on params
            sigma_g_func, sigma_t_func = derive_CRLBs(**params)

            # fills in numerical values for 'equal' or 'nonequal' dose,
            # and all acquisition approaches for a given lens and gradient type
            sigma_g[dose_num, app_num, :, :] = sigma_g_func(gamma, theta)
            sigma_t[dose_num, app_num, :, :] = sigma_t_func(gamma, theta)

    # Formats save settings
    if setup.save:
        np.save(setup.filepath +
                'sigma_gamma_{}x_{}grad_{}_{}x{}'.format(
                    setup.lens, 'weak' if setup.weak_grad else 'normal',
                    'fromZero' if setup.fromZero else None,
                    sample_size, sample_size), sigma_g)

        np.save(setup.filepath +
                'sigma_theta_{}x_{}grad_{}_{}x{}'.format(
                    setup.lens, 'weak' if setup.weak_grad else 'normal',
                    'fromZero' if setup.fromZero else None,
                    sample_size, sample_size), sigma_g)

    # passes physical parameters for future functions
    # params = dict(lens=setup.lens, weak_grad=setup.weak_grad, fromZero=setup.fromZero,
    #               approaches=self.approaches, filepath=setup.filepath,
    #               gamma=gamma, theta=theta, sample_size=sample_size)

    return sigma_g, sigma_t, setup


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

    # initializing symbolic variables
    gamma, theta = sym.symbols('gamma theta')

    num_frames = len(bias)  # number of frames in acquisition approach

    # Forward model for shear measurement in x direction.
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I1 = [I_ent * sym.sin(sym.pi /
                          lamda * (b + sym.sqrt(2) * d * gamma *
                                   sym.cos(theta)))**2 + Ic for b in bias]

    # Forward model for shear measurement in y direction
    # Yields a list of sympy expressions for each value 'b' in 'bias'
    I2 = [I_ent * sym.sin(sym.pi /
                          lamda * (b + sym.sqrt(2) * d * gamma *
                                   sym.sin(theta)))**2 + Ic for b in bias]

    # Calculates lists of first partial derivatives for I1 and I2
    dI1dg = [I1[frame_num].diff(gamma) for frame_num in range(num_frames)]
    dI1dt = [I1[frame_num].diff(theta) for frame_num in range(num_frames)]

    dI2dg = [I2[frame_num].diff(gamma) for frame_num in range(num_frames)]
    dI2dt = [I2[frame_num].diff(theta) for frame_num in range(num_frames)]

    # Derives Fisher matrix elements for poisson variables
    # First generates a np.array containing the expressions for each frame_num,
    # then sums them into a single sympy expression
    F11 = np.array([1 / I1[frame_num] * dI1dg[frame_num]**2 + 1 / I2[frame_num]
                    * dI2dg[frame_num]**2 for frame_num in range(num_frames)]).sum()

    F22 = np.array([1 / I1[frame_num] * dI1dt[frame_num]**2 + 1 / I2[frame_num]
                    * dI2dt[frame_num]**2 for frame_num in range(num_frames)]).sum()

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
