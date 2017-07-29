from dic.experiment import Experiment
from dic.plots import generate_plots
import matplotlib.pyplot as plt

'''This script is a scratch file for looking at the "SNR" - evaluating how
the ratio of true value to standard deviation changes as a functin of gamma
and theta
'''


experiment = Experiment(lens=40,
                        weak_grad=False,
                        lamda=546,
                        approaches=['A: 2x2', 'A: 2x3', 'B: 2x3', 'B: 2x4'],
                        k=1e3,
                        fromZero=True,
                        save=False,
                        filepath=None)


sigma_g, sigma_t = experiment.generate_data(sample_size=100)

# sigma_g = experiment.gamma / sigma_g
# sigma_t = experiment.theta / sigma_t

generate_plots(sigma_g, sigma_t, experiment,
               polar=False,
               areaplot=False,
               show=True)

# SNR_g = experiment.gamma / sigma_g
# SNR_t = experiment.theta / sigma_t

# fig = plt.figure(1)
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# # Equal dose
# ax1.plot(experiment.gamma[0, :], sigma_g[0, 0, :, 50])
# ax2.plot(experiment.theta[0, :], sigma_t[0, 0, 50, :])

# plt.show()
