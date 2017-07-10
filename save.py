from dic.experiment import Experiment
from dic.crlb import generate_data
from dic.plots import generate_plots

for lens in [40, 100]:
    for weak_grad in [True, False]:
        for polar in [True, False]:
            for areaplot in [True, False]:
                print('\n lens={}'.format(lens),
                      'weak_grad={}'.format(weak_grad),
                      'polar={}'.format(polar),
                      'areaplot={} \n'.format(areaplot))
                expo = Experiment(lens=lens,
                                  weak_grad=weak_grad,
                                  save=True)
                expo.filepath = 'images/area_plots/' if areaplot else 'images/raw_plots/'
                sigma_g, sigma_t, expo = generate_data(expo, sample_size=100)
                generate_plots(sigma_g, sigma_t, expo,
                               polar=polar,
                               areaplot=areaplot,
                               show=False)
