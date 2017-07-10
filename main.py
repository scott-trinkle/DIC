from dic.experiment import Experiment
from dic.crlb import generate_data
from dic.plots import generate_plots

expo = Experiment(lens=40,
                  weak_grad=False,
                  approaches=['2x2', '2x3', '2x4'],
                  k=0.01,
                  fromZero=True,
                  save=False,
                  filepath=None)


sigma_g, sigma_t, expo = generate_data(expo, sample_size=100)
generate_plots(sigma_g, sigma_t, expo,
               polar=True,
               areaplot=False,
               show=True)
