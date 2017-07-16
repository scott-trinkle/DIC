from dic.experiment import Experiment
from dic.plots import generate_plots

experiment = Experiment(lens=40,
                        weak_grad=False,
                        lamda=546,
                        approaches=['A: 2x2', 'A: 2x3', 'B: 2x3', 'B: 2x4'],
                        k=0.005,
                        fromZero=True,
                        save=False,
                        filepath=None)


sigma_g, sigma_t = experiment.generate_data(sample_size=100)

generate_plots(sigma_g, sigma_t, experiment,
               polar=True,
               areaplot=True,
               show=True)
