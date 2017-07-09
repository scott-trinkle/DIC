import numpy as np


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
