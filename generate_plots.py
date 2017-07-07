'''
Version 3.5 - added polar capability


Version 3.0

User specifies lens (40 or 100) and weak_gradient bool, the script loads
the relevant data and creates a single figure with 4 cartesian surface plots
of data: sigma_g and sigma_t for equal and non-equal dose conditions.
Each plot includes all three acquisition approaches.

User has the option to save the plot.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.ticker as tck


#############################################################################
#                      Choose physical setup                                #
#############################################################################

# Choose either 40x or 100x objective lens
lens = 40

# Choose a weak gradient specimen(True) or normal(False)
weak_grad = False

# choose whether or not to save the figure
save = False

# True: gamma range is 0.0-0.3, False: 0.1-0.3
fromZero = False

# Choose polar plot. 'False' is cartesian
polar = False
if polar and not fromZero:
    raise ValueError('Gamma range must be set to zero for polar plotting')

# acquisition approaches
approaches = ['2x2', '2x3', '2x4']

#############################################################################
#                               Variables                                   #
#############################################################################

if fromZero:
    datafolder = 'CRLB_data_fromZero/'
    if polar:
        plotfolder = 'grad_surface_plots_polar/'
    else:
        plotfolder = 'grad_surface_plots_fromZero/'
else:
    datafolder = 'CRLB_data/'
    plotfolder = 'grad_surface_plots/'

# Loads relevant data generated with generate_CRLB_data.py
sigma_g = np.load(datafolder + '{}_{}_sigma_g.npy'.format(
    lens, 'weak' if weak_grad else 'normal'))
sigma_t = np.load(datafolder + '{}_{}_sigma_t.npy'.format(
    lens, 'weak' if weak_grad else 'normal'))
gamma = np.load(datafolder + 'gamma.npy')
theta = np.load(datafolder + 'theta.npy')

Na = sigma_g.shape[1]  # gets number of acquisition approaches from data

# set view angle for surface plots
elev, azim = (1, -70) if polar else (15, -70)

# colors for different aquisition approaches within a single plot.
# Will need to extend if we ever look at more than 3 approaches.
c = ['maroon', 'seagreen', 'royalblue']


#############################################################################
#                          Define Functions                                 #
#############################################################################


def make_surface_plots(ax, sigma, var, c=c, elev=elev, azim=azim, gamma=gamma,
                       theta=theta):
    '''
    Returns Axes instance with formatted cartesian surface plots for all
    acquisition approaches.

    Does NOT format Axes titles or figure properties.

    Note: sigma needs to be a (Na, sample_size, sample_size) shaped array. 
    From the main function, pass in sigma[0] or sigma[1] as defined elsewhere. 
    '''

    # Customizes x, y, and z ticks and labels:

    ax.xaxis.set_major_locator(tck.MultipleLocator(0.1))
    ax.xaxis.set_label_text(r'$\gamma(x,y)$ [nm/nm]')

    ax.yaxis.set_major_locator(tck.LinearLocator(3))
    ax.yaxis.set_major_formatter(tck.FixedFormatter((r'0', r'$\pi$',
                                                     r'$2\pi$')))
    ax.yaxis.set_label_text(r'$\theta (x,y)$')

    ax.zaxis.set_major_locator(tck.MaxNLocator(5))
    if fromZero and var == 'theta':
        sigma[np.where(sigma == np.inf)] = sorted(set(sigma.flatten()))[-2]
        sigma = np.log10(sigma)

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    x, y = (gamma * np.cos(theta), gamma *
            np.sin(theta)) if polar else (gamma, theta)

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(Na):
        ax.plot_surface(
            x, y, sigma[app_num], color=c[app_num],
            antialiased=True, alpha=0.7, zorder=3 - app_num)

    if fromZero and var == 'theta':
        ax.zaxis.set_major_locator(tck.MultipleLocator(1))
        zticks = ax.get_zticks()
        ax.set_zticklabels(
            ['10$^{}$'.format(r'{' + str(int(tick)) + r'}') for tick in zticks])

    if polar:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$\gamma$ cos($\theta$)')
        ax.set_ylabel(r'$\gamma$ sin($\theta$)')
    return ax


def make_figure(fignum=1, size=(14, 7.8),  sigma_g=sigma_g, sigma_t=sigma_t,
                lens=lens, weak_grad=weak_grad):
    '''
    Creates a single figure with 4 Axes objects created with make_surface_plots
    '''

    fig = plt.figure(fignum, figsize=size)

    # Title for entire figure
    fig.suptitle('{}x Objective, {} gradient'.format(
        lens, 'Weak' if weak_grad else 'Normal'), weight='bold')

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    ax = []
    for dose_num, dose_tf in enumerate((True, False)):
        for var_num, var in enumerate(['gamma', 'theta']):

            # flattens two numerical indices to increment as [0, 1, 2, 3]
            ind = dose_num * 2 + var_num

            # adds to 4x4 grid of 3D subplots
            ax.append(fig.add_subplot(2, 2, ind + 1, projection='3d'))

            # adds surface plot to active subplot
            ax[ind] = make_surface_plots(
                ax[ind], sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                var=var)

            # sets title of subplot
            title = '{} dose CRLB for '.format(
                'Equal' if dose_tf else 'Non-equal') + r'$\sigma_{}^2$'.format(
                r'{\gamma}' if var == 'gamma' else r'{\theta}')
            ax[ind].set_title(title)

    # Adds legend to FIGURE, not to each axis, since they're the same.
    # There's not a great way to do this with surfaces, you have to set up
    # empty "proxy artists" with the same color.
    fig.legend([Line2D([], [], linestyle='-', color=c[kk])
                for kk in range(Na)],
               [approach + ' frames' for approach in approaches], numpoints=1,
               loc=10, fontsize='large')

    plt.tight_layout()  # makes things look nice
    plt.subplots_adjust(top=0.95)  # so subplots don't overlap the main title
    return


plt.close()  # clears any existing matplotlib backend stuff in the shell
make_figure()  # makes the figure

# saves figure if selected
if save:
    plt.savefig(plotfolder + '{}x_{}.png'.format(
        lens, 'weak' if weak_grad else 'normal'), dpi=215)

plt.show()  # shows figure regardless of saving
