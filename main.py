from dic.experiment import Experiment
from dic.plots import generate_plots

'''This script generates data for a given experiment condition and plots it in
a single figure. Plotting options are:

polar: Plots in polar coordinates if true, otherwise Cartesian. 
areaplot: Plots the "error area" (\gamma*\sigma_gamma*\sigma_theta) if true,
          otherwise, creates 4x4 variable-specific plots
show: Displays the figure if true, otherwise does not. Used when saving.'''

experiment = Experiment(lens=40,
                        weak_grad=False,
                        lamda=546,
                        approaches=['A: 2x2', 'A: 2x3', 'B: 2x3', 'B: 2x4'],
                        k=1e3,
                        fromZero=True,
                        save=False,
                        filepath=None)


sigma_g, sigma_t = experiment.generate_data(sample_size=100)

generate_plots(sigma_g, sigma_t, experiment,
               polar=False,
               areaplot=False,
               SNR=True,
               show=True)
