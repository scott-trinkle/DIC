from dic.experiment import Experiment
from dic.crlb import generate_data
from dic.plots import generate_plots, make_cartesian_plots, make_polar_plots
import numpy as np
import matplotlib.pyplot as plt

expo = Experiment(lens=100,
                  weak_grad=True,
                  approaches=['2x2', '2x3', '2x4'],
                  k=0.1,
                  lamda=546,
                  fromZero=True,
                  save=False,
                  filepath=None)


sigma_g, sigma_t, expo = generate_data(expo, sample_size=100)
generate_plots(sigma_g, sigma_t, expo,
               polar=True,
               areaplot=True)

# for lens in [40, 100]:
#     for weak_grad in [True, False]:
#         for polar in [True, False]:
#             for areaplot in [True, False]:
#                 print('\n lens={}'.format(lens),
#                       'weak_grad={}'.format(weak_grad),
#                       'polar={}'.format(polar),
#                       'areaplot={} \n'.format(areaplot))
#                 expo = Experiment(lens=lens,
#                                   weak_grad=weak_grad,
#                                   save=True,
#                                   filepath='images/')
#                 sigma_g, sigma_t, expo = generate_data(expo, sample_size=100)
#                 generate_plots(sigma_g, sigma_t, expo,
#                                polar=polar,
#                                areaplot=areaplot,
#                                show=False)
