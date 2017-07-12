from dic.experiment import Experiment
from dic.crlb import generate_data
import numpy as np
import matplotlib.pyplot as plt


expo = Experiment(lens=100,
                  weak_grad=True)

x = np.logspace(1, 5, 10)

E_g_med = np.zeros((3, 10))
E_g_max = np.zeros((3, 10))
E_g_min = np.zeros((3, 10))
for i, k in enumerate(x):
    expo.k = k / 50 if expo.lens == 40 else k / 200
    sigma_g, sigma_t, expo = generate_data(expo)
    err_area = expo.gamma * sigma_g * sigma_t
    for j in range(3):
        E_g_med[j, i] = np.median(err_area[0, j])
        # E_g_min[j, i] = err_area[0, j].min()
        # E_g_max[j, i] = err_area[0, j].max()

poly = np.round(np.polyfit(np.log10(x), np.log10(E_g_med[0]), deg=1))

colors = ['maroon', 'seagreen', 'royalblue']
labels = ['2x2', '2x3', '2x4']

fig, ax = plt.subplots(1)

for i in range(3):  # 'i' is approach
    # ax.loglog(x, E_g_max[i], '--x', color=colors[i],
    #           label='{}: max'.format(labels[i]), alpha=0.75, linewidth=0.75)
    ax.loglog(x, E_g_med[i], '-+', color=colors[i],
              label='{}'.format(labels[i]), alpha=1, linewidth=1.5)
    # ax.loglog(x, E_g_min[i], ':x', color=colors[i],
    #           label='{}: min'.format(labels[i]), alpha=0.75, linewidth=0.75)

power = r'{' + str(poly[0]) + '}'

ax.text(0.4, 0.8, r'Error area $\approx$ I$^{}$'.format(power),
        transform=ax.transAxes)
ax.text(0.15, 0.15, '\nLens={}x\nGradient = {}\nDose = {}'.format(expo.lens,
                                                                  'Weak' if expo.weak_grad else 'Normal',
                                                                  'Equal'), transform=ax.transAxes)

plt.title('Error area as a function of entrance intensity')
plt.xlabel(r'Entrance intensity, I [photons]')
plt.ylabel(r'Error area, $\gamma\sigma_{\gamma} \sigma_{\theta}$, [nm/nm]$^2$')
plt.legend()
# plt.show()

plt.savefig('images/intensity/{}x_{}grad_{}.png'.format(expo.lens, 'weak' if expo.weak_grad else 'normal',
                                                        'equal'), dpi=200)
