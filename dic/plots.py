import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.ticker as tck
from dic.experiment import Experiment


def generate_plots(sigma_g, sigma_t, expo, polar=False, areaplot=True,
                   show=True, fignum=1, size=(14, 7.8)):
    ''' Generates and plots a single figure to visualize CRLB data

    Parameters
    __________
    sigma_g : ndarray
        sigma_g data
    sigma_t : ndarray
        sigma_t data
    expo : Experiment instance
        Contains all experiment parameters
    polar : bool
        If True, plots in polar coordinates. Default is False.
    fignum : int
        Figure numnber
    size : tuple
        Figure size
    '''

    expo.polar = polar

    if expo.polar and not expo.fromZero:
        raise ValueError('Gamma range must be set to zero for polar plotting')

    # colors for different aquisition approaches within a single plot.
    # Will need to extend if we ever look at more than 3 approaches.
    expo.colors = ['maroon', 'seagreen', 'royalblue']

    plt.close()  # clear any existing plots in the python interpreter

    fig = plt.figure(fignum, figsize=size)

    # Title for entire figure
    fig.suptitle('{}x Objective, {} gradient'.format(
        expo.lens, 'Weak' if expo.weak_grad else 'Normal'), weight='bold')

    print('Starting figure...')
    fig = make_area_plots(fig, sigma_g, sigma_t, expo) if areaplot else make_raw_plots(
        fig, sigma_g, sigma_t, expo)

    # saves figure if selected
    if expo.save:
        print('Saving figure...')
        plt.savefig(expo.filepath + '{}x_{}grad{}_{}_{}x{}_{}.png'.format(
            expo.lens, 'weak' if expo.weak_grad else 'normal',
            '_fromZero' if (expo.fromZero and not expo.polar) else '_from0.1',
            'polar' if expo.polar else 'cart',
            expo.gamma.shape[0], expo.gamma.shape[0],
            'areaplot' if areaplot else 'rawplot'), dpi=300)
    if show:
        print('Displaying figure...')
        plt.show()

    return


def make_raw_plots(fig, sigma_g, sigma_t, expo):
    ''' Makes a 2x2 figure that plots 3D CRLBS for gamma and theta under both
    equal and non-equal dose conditions.

    Parameters
    __________
    fig : matplotlib.figure.Figure
        Main figure
    sigma_g : ndarray
        CRLB data for gamma
    sigma_t : ndarray
        CRLB data for theta
    expo : Experiment instance
        Experiment parameters

    Returns
    _______
    fig : matplotlib.figure.Figure
        Updated figure
    '''
    ax = []

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    for dose_num, equalize_dose in enumerate((True, False)):
        for var_num, var in enumerate(['gamma', 'theta']):

            # increments as [0, 1, 2, 3]
            ind = dose_num * 2 + var_num

            print('Adding plot: {}'.format(ind + 1))

            # adds to 2x2 grid of 3D subplots
            ax.append(fig.add_subplot(2, 2, ind + 1, projection='3d'))

            if expo.polar:
                ax[ind] = make_polar_plots(
                    ax[ind],
                    sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                    expo)
            else:
                ax[ind] = make_cartesian_plots(
                    ax[ind],
                    sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                    expo)

            # sets title of subplot
            title = '{} dose CRLB for '.format(
                'Equal' if equalize_dose else 'Non-equal') + r'$\sigma_{}$'.format(
                r'{\gamma}' if var == 'gamma' else r'{\theta}')
            ax[ind].set_title(title)

    # Adds single legend
    print('Adding legend...')
    fig.legend([Line2D([], [], linestyle='-', color=expo.colors[kk])
                for kk in range(expo.Na)],
               [approach + ' frames' for approach in expo.approaches],
               numpoints=1, loc=10, fontsize='large')

    plt.tight_layout()  # makes things look nice

    plt.subplots_adjust(top=0.95)  # so subplots don't overlap the main title

    return fig


def make_area_plots(fig, sigma_g, sigma_t, expo):
    ''' Makes a 2x2 figure that plots the area product gamma*sigma_g*sigma_t 
    under both equal and non-equal dose conditions.

    Parameters
    __________
    fig : matplotlib.figure.Figure
        Main figure
    sigma_g : ndarray
        CRLB data for gamma
    sigma_t : ndarray
        CRLB data for theta
    expo : Experiment instance
        Experiment parameters

    Returns
    _______
    fig : matplotlib.figure.Figure
        Updated figure
    '''
    ax = []

    err_area = expo.gamma * sigma_g * sigma_t

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    for dose_num, equalize_dose in enumerate((True, False)):

        print('Adding plot: {}'.format(dose_num + 1))

        # adds to 2x2 grid of 3D subplots
        ax.append(fig.add_subplot(1, 2, dose_num + 1, projection='3d'))

        if expo.polar:
            ax[dose_num] = make_polar_plots(
                ax[dose_num], err_area[dose_num], expo)
        else:
            ax[dose_num] = make_cartesian_plots(
                ax[dose_num], err_area[dose_num], expo)

        # sets title of subplot
        title = '{} dose CRLB for '.format(
            'Equal' if equalize_dose else 'Non-equal') + r'$\gamma\sigma_{\gamma}\sigma_{\theta}$'
        ax[dose_num].set_title(title)

    # Adds single legend
    print('Adding legend...')
    fig.legend([Line2D([], [], linestyle='-', color=expo.colors[kk])
                for kk in range(expo.Na)],
               [approach + ' frames' for approach in expo.approaches],
               numpoints=1, loc=(0.45, 0.08), fontsize='large')

    plt.tight_layout()  # makes things look nice

    # so subplots don't overlap the main title
    plt.subplots_adjust(top=0.95)

    return fig


def setup_ax(ax, elev, azim):
    ''' Basic formatting for 3D axes

    Parameters
    _________
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes instance
    elev, azim : int
        elevation and azimuth angles for viewing 3D axis

    Returns
    _______
    ax : matplotlib.axes._subplots.AxesSubplot
        Updated Axes instance
    '''
    ax.view_init(elev=elev, azim=azim)  # Set view angle

    # Gamma ticks and label
    ax.xaxis.set_major_locator(tck.MultipleLocator(0.1))
    ax.xaxis.set_label_text(r'$\gamma(x,y)$ [nm/nm]')

    # Theta ticks and label
    ax.yaxis.set_major_locator(tck.LinearLocator(3))
    ax.yaxis.set_major_formatter(tck.FixedFormatter((r'0', r'$\pi$',
                                                     r'$2\pi$')))
    ax.yaxis.set_label_text(r'$\theta (x,y)$')

    # CRLB ticks
    ax.zaxis.set_major_locator(tck.MaxNLocator(5))

    return ax


def make_cartesian_plots(ax, sigma, expo):
    ''' Plots CRLB on 3D cartesian grid

    Parameters
    _________
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes instance
    sigma : ndarray
        Shape needs to be (Na, sample_size, sample_size) where Na is the
        number of approaches, and sample_size is the sample size for gamma
        and theta.
    var : str
        Either 'gamma' or 'theta'
    expo : Experiment instance
        Experiment parameters

    Returns
    _______
    ax : matplotlib.axes._subplots.AxesSubplot
        Updated Axes instance
    '''

    setup_ax(ax, elev=15, azim=-70)

    # Fixes dividing by zero errors. Sets 'inf' to the highest non-inf value
    if expo.fromZero and sigma.max() == np.inf:
        sigma[np.where(sigma == np.inf)] = sorted(set(sigma.flatten()))[-2]
        sigma = np.log10(sigma)

        # Set z axis to log scale
        ax.zaxis.set_major_locator(tck.MultipleLocator(1))
        zticks = ax.get_zticks()
        ax.set_zticklabels(
            ['10$^{}$'.format(r'{' + str(int(tick)) +
                              r'}') for tick in zticks])

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(expo.Na):
        ax.plot_surface(
            expo.gamma, expo.theta, sigma[app_num],
            color=expo.colors[app_num], antialiased=True, alpha=0.7)

    return ax


def make_polar_plots(ax, sigma, expo):
    ''' Plots CRLB on 3D polar grid

    Parameters
    _________
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes instance
    sigma : ndarray
        Shape needs to be (Na, sample_size, sample_size) where Na is the
        number of approaches, and sample_size is the sample size for gamma
        and theta.
    var : str
        Either 'gamma' or 'theta'
    expo : Experiment instance
        Experiment parameters

    Returns
    _______
    ax : matplotlib.axes._subplots.AxesSubplot
        Updated Axes instance
    '''

    setup_ax(ax, elev=1, azim=-70)

    # Fixes dividing by zero errors. Sets 'inf' to the highest non-inf value
    if sigma.max() == np.inf:
        sigma[np.where(sigma == np.inf)] = sorted(set(sigma.flatten()))[-2]
        sigma = np.log10(sigma)

        # Set z axis to log scale
        ax.zaxis.set_major_locator(tck.MultipleLocator(1))
        zticks = ax.get_zticks()
        ax.set_zticklabels(
            ['10$^{}$'.format(r'{' + str(int(tick)) +
                              r'}') for tick in zticks])

    # Puts x and y in polar coordinates
    x, y = (expo.gamma * np.cos(expo.theta), expo.gamma *
            np.sin(expo.theta))

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(expo.Na):
        ax.plot_surface(
            x, y, sigma[app_num], color=expo.colors[app_num],
            antialiased=True, alpha=0.7)

    # Adjusts ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$\gamma$ cos($\theta$)')
    ax.set_ylabel(r'$\gamma$ sin($\theta$)')
    return ax
