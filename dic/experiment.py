import numpy as np
import sympy as sym


class Experiment(object):
    '''Class which stores all of the experiment parameters, both for the
    physical setup of the OI-DIC experiment and for the CRLB calculation

    Attributes
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
    lamda : int
        Wavelength of light used for the experiment (in nm).
    '''

    def __init__(self, lens=40, weak_grad=False,
                 approaches=['2x2', '2x3', '2x4'],
                 k=0.1, fromZero=True, save=False, filepath=None):
        self.lens = lens
        self.weak_grad = weak_grad
        self.approaches = approaches
        self.k = k
        self.fromZero = fromZero
        self.save = save
        self.filepath = filepath
        self.polar = None

        self.Na = len(self.approaches)  # 3 for now

    def set_gamma_theta(self, sample_size=100):
        '''Initializes gamma and theta meshes of a given sample size

        Parameters
        __________
        sample size : int
            Sample size for gamma and theta from min to max values

        Returns
        _______
        self.gamma, self.theta : ndarray
            An npmeshgrid of gamma and theta
        '''

        if self.fromZero:
            start = 0.0
        else:
            start = 0.1

        self.gamma = np.linspace(start, 0.3, sample_size)
        self.theta = np.linspace(0, 2 * np.pi, sample_size)
        self.gamma, self.theta = np.meshgrid(self.gamma, self.theta)
        return self.gamma, self.theta

    def derive_CRLBs(self, equalize_dose=True, approach='2x2'):

        lamda = 546  # nm
        if not self.weak_grad:  # a normal specimen
            bias_j = 0.15 * lamda
        else:  # a weak gradient specimen
            bias_j = 0.05 * lamda

        if self.lens == 40:
            d = 255  # shear distance in nm
            I_ent = 50 * self.k  # exp time in us (future: entrance intensity)
        elif self.lens == 100:
            d = 100
            I_ent = 200 * self.k
        else:
            raise ValueError('Please enter expo.lens = 40 or expo.lens = 100')

        print('Calculating CRLBs: {} dose - {}'.format(
            'Equal' if equalize_dose else 'Non-equal', approach))

        # gets the second numerical value in approach
        num_frames = int(approach.split('x')[-1])

        # redefines entrance intensity I_ent based on equalize_dose
        I_ent /= num_frames if equalize_dose else 1
        Ic = 0.01 * I_ent   # stray light intensity

        # defines the bias vector based on acquisiiton approach
        if approach == '2x2':
            bias = [-bias_j, bias_j]
        elif approach == '2x3':
            bias = [-bias_j, 0, bias_j]
        elif approach == '2x4':
            bias = [0, lamda / 4,
                    lamda / 2, 3 * lamda / 4]

        gamma, theta = sym.symbols('gamma theta')

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

        # Sympy function that converts a sympy expression into a numpy-based
        # python function
        sigma_g_func = sym.lambdify((gamma, theta), sigma_g, 'numpy')
        sigma_t_func = sym.lambdify((gamma, theta), sigma_t, 'numpy')

        return sigma_g_func, sigma_t_func
