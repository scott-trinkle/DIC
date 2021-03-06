from dic.experiment import Experiment
from dic.plots import generate_plots, plot_OPL_var
from dic.misc import make_phantom, calculate_OPL_var

# Saves plots for all relevant experiment conditions.

count = 1

for lens in [40, 100]:
    for weak_grad in [True, False]:
        print('{}% done!'.format(count / 4 * 100),
              '\n lens={}'.format(lens),
              'weak_grad={}'.format(weak_grad))
        expo = Experiment(lens=lens,
                          weak_grad=weak_grad,
                          k=100,
                          save=True)
        expo.filepath = 'images/forreport/'
        sigma_g, sigma_t = expo.generate_data(sample_size=100)
        generate_plots(sigma_g, sigma_t, expo,
                       polar=False,
                       areaplot=False,
                       SNR=True,
                       report=True,
                       show=False)
        count += 1


# for lens in [40, 100]:
#     for weak_grad in [True, False]:

#         experiment = Experiment(lens=lens,
#                                 weak_grad=weak_grad,
#                                 k=1e3,
#                                 save=True,
#                                 filepath='images/varOPL/')

#         varOPL = calculate_OPL_var(make_phantom(512, 512), experiment)
#         plot_OPL_var(varOPL, experiment, show=False)

#         print('{}% done!'.format(count / 4 * 100),
#               '\n lens={}'.format(lens),
#               'weak_grad={}'.format(weak_grad))

#         count += 1
