from dic.experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt

'''This script generates plots of median CRLB error area as a function of
entrance intensity'''

experiment = Experiment(lens=40,
                        weak_grad=False)
equal = 1

sample_size = 10

x = np.logspace(1, 5, sample_size)

E_g = np.zeros((3, sample_size))
E_t = np.zeros((3, sample_size))
for i, k in enumerate(x):
    experiment.k = k / 50 if experiment.lens == 40 else k / 200
    sigma_g, sigma_t = experiment.generate_data(experiment)
    for j in range(3):
        E_g[j, i] = np.median(sigma_g[equal, j])
        E_t[j, i] = np.median(sigma_t[equal, j])


poly_g = np.round(np.polyfit(np.log10(x), np.log10(E_g[equal]), deg=1), 3)
poly_t = np.round(np.polyfit(np.log10(x), np.log10(E_t[equal]), deg=1), 3)


colors = ['maroon', 'seagreen', 'royalblue']
labels = ['2x2', '2x3', '2x4']

fig, (ax_g, ax_t) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

for i in range(3):  # 'i' is approach
    ax_g.semilogx(x, E_g[i], '-+', color=colors[i],
                  label='{}'.format(labels[i]), alpha=1, linewidth=1.5)
    ax_t.semilogx(x, E_t[i], '-+', color=colors[i],
                  label='{}'.format(labels[i]), alpha=1, linewidth=1.5)

tenpower_g = r'{' + str(round(10**poly_g[1], 3)) + '}'
Ipower_g = r'{' + str(poly_g[0]) + '}'

tenpower_t = r'{' + str(round(10**poly_t[1], 3)) + '}'
Ipower_t = r'{' + str(poly_t[0]) + '}'

ax_g.set_title(r'$\gamma$')
ax_t.set_title(r'$\theta$')

ax_g.text(0.1, 0.8, r'Median error area $\approx$ 10$^{}I^{}$'.format(tenpower_g, Ipower_g),
          transform=ax_g.transAxes)
ax_t.text(0.1, 0.8, r'Median error area $\approx$ 10$^{}I^{}$'.format(tenpower_t, Ipower_t),
          transform=ax_t.transAxes)


ax_g.set_xlabel(r'Entrance intensity, I [photons]')
ax_t.set_xlabel(r'Entrance intensity, I [photons]')

ax_t.set_ylabel(r'$\sigma_{\theta, med}$ [nm/nm]')
ax_g.set_ylabel(r'$\sigma_{\gamma, med}$ [nm/nm]')
ax_g.legend()
ax_t.legend()

fig.suptitle('Median error as a function of entrance intensity \n' +
             '\nLens={}x, Gradient = {}, Dose = {}'.format(experiment.lens,
                                                           'Weak' if experiment.weak_grad else 'Normal',
                                                           'Non-equal'))

plt.subplots_adjust(top=0.80)

# plt.show()

plt.savefig('images/intensity/CRLB_vs_I_non_equal.png')
