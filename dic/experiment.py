import numpy as np
import sympy as sym


class Experiment(object):
    '''Class which stores all of the experiment parameters and methods for
    deriving and calculating CRLB data.


    Attributes
    __________
    lens : int
        Objective lens, either 40 or 100
    weak_grad : bool
        If True, calculate CRLBs for weak gradient specimens. Default is False.
    lamda : int
        Wavelength of light used for the experiment (in nm). Note: 'lambda' is
        an existing Python function. Hence the misspelling.
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
    '''

    def __init__(self, lens=40, weak_grad=False, lamda=546,
                 approaches=['A: 2x2', 'A: 2x3', 'B: 2x3', 'B: 2x4'],
                 k=1e3, fromZero=True, save=False, filepath=None):
        self.lens = lens
        self.weak_grad = weak_grad
        self.lamda = lamda
        self.approaches = approaches
        self.k = k
        self.fromZero = fromZero
        self.save = save
        self.filepath = filepath
        self.polar = None

        self.Na = len(self.approaches)

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
            start = 1e-16
        else:
            start = 0.1

        self.gamma = np.linspace(start, 0.3, sample_size)
        self.theta = np.linspace(1e-16, 2 * np.pi, sample_size)
        self.gamma, self.theta = np.meshgrid(self.gamma, self.theta)
        return self.gamma, self.theta

    def derive_CRLBs(self, equalize_dose=True, approach='A: 2x2'):
        '''This function uses the SymPy symbolic math package to derive the
        expression for the CRLB for both gamma and theta for a given
        experimental condition.

        Attributes:
        ___________
        equalize_dose : bool
            If true, equalizes total dose across all acquisition frames. Default
            is true.
        approach : str
            Acquisition approach label.

        Returns:
        ________
        sigma_g_func : NumPy function
            NumPy function that generates numerical CRLB values for gamma
        sigma_t_func : NumPy function
            NumPy function that generates numerical CRLB values for theta'''

        if not self.weak_grad:  # a normal specimen
            bias_j = 0.15 * self.lamda
        else:  # a weak gradient specimen
            bias_j = 0.05 * self.lamda

        if self.lens == 40:
            d = 255  # shear distance in nm
            I_ent = self.k  # entrance intensity (40x get 50 us exposure)
        elif self.lens == 100:
            d = 100
            I_ent = self.k  # 100x gets 200 us exposure
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
        if approach == 'A: 2x2':
            bias = [-bias_j, bias_j]
        elif approach == 'A: 2x3':
            bias = [-bias_j, 0, bias_j]
        elif approach == 'B: 2x3':
            bias = [-self.lamda / 3, 0, self.lamda / 3]
        elif approach == 'B: 2x4':
            bias = [0, self.lamda / 4,
                    self.lamda / 2, 3 * self.lamda / 4]

        gamma, theta = sym.symbols('gamma theta')

        # Forward model for shear measurement in x direction.
        # Yields a list of sympy expressions for each value 'b' in 'bias'
        I1 = [I_ent * sym.sin(sym.pi /
                              self.lamda * (b + sym.sqrt(2) * d * gamma *
                                            sym.cos(theta)))**2 + Ic for b in bias]

        # Same for y
        I2 = [I_ent * sym.sin(sym.pi /
                              self.lamda * (b + sym.sqrt(2) * d * gamma *
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
        sigma_g = sym.sqrt(F22 / det)  # equivalent to [F^-1]_11
        sigma_t = sym.sqrt(F11 / det)  # equivalent to [F^-1]_22

        # Sympy function that converts a sympy expression into a numpy-based
        # python function
        sigma_g_func = sym.lambdify((gamma, theta), sigma_g, 'numpy')
        sigma_t_func = sym.lambdify((gamma, theta), sigma_t, 'numpy')

        return sigma_g_func, sigma_t_func

    def generate_data(self, sample_size=100):
        '''
        Returns two numpy arrays containing data for the Cramer Rao Lower Bounds
        for the standard deviations of gamma and theta for OI - DIC.

        Parameters
        __________
        sample_size: int
            Sample size for gamma and theta

        Returns
        _______
        sigma_g: ndarray
            CRLB data for gamma for the specified physical parameters.
            Shape is (2, len(approaches), sample_size, sample_size).
        sigma_t: ndarray
            CRLB data for theta for the specified physical parameters.
            Shape is (2, len(approaches), sample_size, sample_size).
        '''

        print('Running...')

        # indices are (equalize dose?, acq approach, gamma_vals, theta_vals)
        sigma_g = np.zeros((2, self.Na, sample_size, sample_size))
        sigma_t = np.zeros((2, self.Na, sample_size, sample_size))

        gamma, theta = self.set_gamma_theta(sample_size)

        # iterating over 'equal' or 'non-equal' dose options
        for dose_num, equalize_dose in enumerate((True, False)):

            # iterating over acquisition approaches
            for app_num, approach in enumerate(self.approaches):

                # Calls derive_CRLBs function
                sigma_g_func, sigma_t_func = self.derive_CRLBs(
                    equalize_dose=equalize_dose, approach=approach)

                # stores data for plots. Prevents divide by zero warning.
                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma_g[dose_num, app_num, :,
                            :] = sigma_g_func(gamma, theta)
                    sigma_t[dose_num, app_num, :,
                            :] = sigma_t_func(gamma, theta)

        # Formats save settings
        # if self.save:
        #     print('Saving CRLB data...')
        #     np.save(self.filepath +
        #             'sigma_gamma_{}x_{}grad_{}_{}x{}'.format(
        #                 self.lens, 'weak' if self.weak_grad else 'normal',
        #                 'fromZero' if self.fromZero else None,
        #                 sample_size, sample_size), sigma_g)

        #     np.save(self.filepath +
        #             'sigma_theta_{}x_{}grad_{}_{}x{}'.format(
        #                 self.lens, 'weak' if self.weak_grad else 'normal',
        #                 'fromZero' if self.fromZero else None,
        #                 sample_size, sample_size), sigma_g)
        print('Calculating CRLBs: Done!')

        return sigma_g, sigma_t
