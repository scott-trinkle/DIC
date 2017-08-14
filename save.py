from dic.experiment import Experiment
from dic.plots import generate_plots

# Saves plots for all relevant experiment conditions.

count = 1

# for lens in [40, 100]:
#     for weak_grad in [True, False]:
#         for polar in [True, False]:
#             for areaplot in [True, False]:
#                 print('{}% done!'.format(count / 8 * 100),
#                       '\n lens={}'.format(lens),
#                       'weak_grad={}'.format(weak_grad),
#                       'polar={}'.format(polar),
#                       'areaplot={} \n'.format(areaplot))
#                 expo = Experiment(lens=lens,
#                                   weak_grad=weak_grad,
#                                   k=0.005,
#                                   save=True)
#                 expo.filepath = 'images/area_plots/' if areaplot else 'images/raw_plots/'
#                 sigma_g, sigma_t, expo = generate_data(expo, sample_size=100)
#                 generate_plots(sigma_g, sigma_t, expo,
#                                polar=polar,
#                                areaplot=areaplot,
#                                show=False)
#                 count += 1


for lens in [40, 100]:
    for weak_grad in [True, False]:

        print('{}% done!'.format(count / 4 * 100),
              '\n lens={}'.format(lens),
              'weak_grad={}'.format(weak_grad))

        experiment = Experiment(lens=lens,
                                weak_grad=weak_grad,
                                k=1e3,
                                save=True)

        experiment.filepath = 'images/SNR/'

        sigma_g, sigma_t = experiment.generate_data(sample_size=100)

        generate_plots(sigma_g, sigma_t, experiment,
                       polar=False,
                       SNR=True,
                       show=False)
        count += 1
