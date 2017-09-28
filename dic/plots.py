import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.ticker as tck
from dic.experiment import Experiment


def generate_plots(sigma_g, sigma_t, experiment, polar=False, areaplot=True,
                   SNR=False, report=False, show=True, fignum=1, size=(14, 7.8)):
    ''' Generates and plots a single figure to visualize CRLB data

    Parameters
    __________
    sigma_g : ndarray
        sigma_g data
    sigma_t : ndarray
        sigma_t data
    experiment : Experiment instance
        Contains all experiment parameters and methods
    polar : bool
        If True, plots in polar coordinates. Default is False.
    fignum : int
        Figure numnber
    size : tuple
        Figure size
    '''

    experiment.polar = polar
    experiment.SNR = SNR
    # colors for different aquisition approaches within a single plot.
    experiment.colors = ['maroon', 'seagreen', 'gold', 'royalblue']

    plt.close()  # clear any existing plots in the python interpreter

    fig = plt.figure(fignum, figsize=size)

    # Title for entire figure
    fig.suptitle('{}x Objective, {} gradient'.format(
        experiment.lens, 'Weak' if experiment.weak_grad else 'Normal'), weight='bold')

    print('Starting figure...')
    if experiment.SNR:
        if report:
            fig = make_report_SNR_plots(fig, sigma_g, sigma_t, experiment)
        else:
            fig = make_SNR_plots(fig, sigma_g, sigma_t, experiment)
    else:
        fig = make_area_plots(fig, sigma_g, sigma_t, experiment) if areaplot else make_raw_plots(
            fig, sigma_g, sigma_t, experiment)

    # saves figure if selected
    if experiment.save:
        print('Saving figure...')

        plt.savefig(experiment.filepath + '{}x_{}grad{}_{}_{}x{}_{}.png'.format(
            experiment.lens, 'weak' if experiment.weak_grad else 'normal',
            '_fromZero' if (
                experiment.fromZero and not experiment.polar) else '_from0.1',
            'polar' if experiment.polar else 'cart',
            experiment.gamma.shape[0], experiment.gamma.shape[0],
            'areaplot' if areaplot else 'rawplot'), dpi=300)
    if show:
        print('Displaying figure...')
        plt.show()

    return


def make_raw_plots(fig, sigma_g, sigma_t, experiment):
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
    experiment : Experiment instance
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

            if experiment.polar:
                ax[ind] = make_polar_plots(
                    ax[ind],
                    sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                    experiment)
            else:
                ax[ind] = make_cartesian_plots(
                    ax[ind],
                    sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                    experiment)

            # sets title of subplot
            title = '{} dose CRLB for '.format(
                'Equal' if equalize_dose else 'Non-equal') + r'$\sigma_{}$'.format(
                    r'{\gamma}' if var == 'gamma' else r'{\theta}')
            ax[ind].set_title(title)

    # Adds single legend
    print('Adding legend...')
    fig.legend([Line2D([], [], linestyle='-', color=experiment.colors[kk])
                for kk in range(experiment.Na)],
               [approach + ' frames' for approach in experiment.approaches],
               numpoints=1, loc=10, fontsize='large')

    plt.tight_layout()  # makes things look nice

    plt.subplots_adjust(top=0.95)  # so subplots don't overlap the main title

    return fig


def make_area_plots(fig, sigma_g, sigma_t, experiment):
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
    experiment : Experiment instance
        Experiment parameters

    Returns
    _______
    fig : matplotlib.figure.Figure
        Updated figure
    '''
    ax = []

    err_area = experiment.gamma * sigma_g * sigma_t

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    for dose_num, equalize_dose in enumerate((True, False)):

        print('Adding plot: {}'.format(dose_num + 1))

        # adds to 2x2 grid of 3D subplots
        ax.append(fig.add_subplot(1, 2, dose_num + 1, projection='3d'))

        if experiment.polar:
            ax[dose_num] = make_polar_plots(
                ax[dose_num], err_area[dose_num], experiment)
            ax[dose_num].set_zticklabels([])
        else:
            ax[dose_num] = make_cartesian_plots(
                ax[dose_num], err_area[dose_num], experiment)

        # sets title of subplot
        title = '{} dose CRLB for - '.format(
            'Equal' if equalize_dose else 'Non-equal') + r'$\gamma\sigma_{\gamma}\sigma_{\theta}$'
        ax[dose_num].set_title(title)

    # Adds single legend
    print('Adding legend...')
    fig.legend([Line2D([], [], linestyle='-', color=experiment.colors[kk])
                for kk in range(experiment.Na)],
               [approach + ' frames' for approach in experiment.approaches],
               numpoints=1, loc=(0.45, 0.08), fontsize='large')

    plt.tight_layout()  # makes things look nice

    # so subplots don't overlap the main title
    plt.subplots_adjust(top=0.95)

    return fig


def make_SNR_plots(fig, sigma_g, sigma_t, experiment):
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
    experiment : Experiment instance
        Experiment parameters

    Returns
    _______
    fig : matplotlib.figure.Figure
        Updated figure
    '''
    ax = []

    SNR_gamma = experiment.gamma / sigma_g
    SNR_area = 1 / (sigma_g * sigma_t)

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    for dose_num, equalize_dose in enumerate((True, False)):
        for var_num, var in enumerate(['gamma', 'area']):

            # increments as [0, 1, 2, 3]
            ind = dose_num * 2 + var_num

            print('Adding plot: {}'.format(ind + 1))

            # adds to 2x2 grid of 3D subplots
            ax.append(fig.add_subplot(2, 2, ind + 1, projection='3d'))

            if experiment.polar:
                ax[ind] = make_polar_plots(
                    ax[ind],
                    SNR_gamma[dose_num] if var == 'gamma' else SNR_area[dose_num],
                    experiment)
            else:
                ax[ind] = make_cartesian_plots(
                    ax[ind],
                    SNR_gamma[dose_num] if var == 'gamma' else SNR_area[dose_num],
                    experiment)

            # sets title of subplot
            if var == 'gamma':
                title = r'{} dose "$\gamma$ SNR" - '.format(
                    'Equal' if equalize_dose else 'Non-equal') + r'${} / \sigma_{}$'.format(
                        r'{\gamma}', r'{\gamma}')
            else:
                title = '{} dose "Area SNR" - '.format(
                    'Equal' if equalize_dose else 'Non-equal') + r'$ 1 / \sigma_{\gamma}\sigma_{\theta}$'

            ax[ind].set_title(title)

    # Adds single legend
    print('Adding legend...')
    fig.legend([Line2D([], [], linestyle='-', color=experiment.colors[kk])
                for kk in range(experiment.Na)],
               [approach + ' frames' for approach in experiment.approaches],
               numpoints=1, loc=10, fontsize='large')

    plt.tight_layout()  # makes things look nice

    plt.subplots_adjust(top=0.95)  # so subplots don't overlap the main title

    return fig


def make_report_SNR_plots(fig, sigma_g, sigma_t, experiment):
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
    experiment : Experiment instance
        Experiment parameters

    Returns
    _______
    fig : matplotlib.figure.Figure
        Updated figure
    '''
    ax = []

    SNR_gamma = experiment.gamma / sigma_g
    SNR_area = 1 / (sigma_g * sigma_t)

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    for dose_num, equalize_dose in enumerate((True, False)):

        ind = dose_num + 1

        print('Adding plot: {}'.format(ind))

        # adds to 2x2 grid of 3D subplots
        ax.append(fig.add_subplot(1, 2, ind, projection='3d'))

        # if experiment.polar:
        #     ax[ind] = make_polar_plots(
        #         ax[ind],
        #         SNR_gamma[dose_num] if var == 'gamma' else SNR_area[dose_num],
        #         experiment)
        # else:

        ax[ind - 1] = make_cartesian_plots(ax[ind - 1],
                                           SNR_area[dose_num],
                                           experiment)

        # sets title of subplot
        # if var == 'gamma':
        #     title = r'{} dose "$\gamma$ SNR" - '.format(
        #         'Equal' if equalize_dose else 'Non-equal') + r'${} / \sigma_{}$'.format(
        #             r'{\gamma}', r'{\gamma}')
        # else:
        title = '{} dose "Area SNR" - '.format(
            'Equal' if equalize_dose else 'Non-equal') + r'$ 1 / \sigma_{\gamma}\sigma_{\theta}$'

        ax[ind - 1].set_title(title)

    # Adds single legend
    print('Adding legend...')
    fig.legend([Line2D([], [], linestyle='-', color=experiment.colors[kk])
                for kk in range(experiment.Na)],
               [approach + ' frames' for approach in experiment.approaches],
               numpoints=1, loc=8, fontsize='large')

    plt.tight_layout()  # makes things look nice

    plt.subplots_adjust(top=0.95)  # so subplots don't overlap the main title

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


def make_cartesian_plots(ax, sigma, experiment):
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
    experiment : Experiment instance
        Experiment parameters

    Returns
    _______
    ax : matplotlib.axes._subplots.AxesSubplot
        Updated Axes instance
    '''

    setup_ax(ax, elev=15, azim=-70)

    # Fixes dividing by zero errors. Sets 'inf' to the highest non-inf value
    if experiment.fromZero and sigma.max() == np.inf:
        sigma[np.where(sigma == np.inf)] = sorted(set(sigma.flatten()))[-2]
        sigma = np.log10(sigma)

        # Set z axis to log scale
        ax.zaxis.set_major_locator(tck.MultipleLocator(1))
        zticks = ax.get_zticks()
        ax.set_zticklabels(
            ['10$^{}$'.format(r'{' + str(int(tick)) +
                              r'}') for tick in zticks])

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(experiment.Na):
        ax.plot_surface(
            experiment.gamma, experiment.theta, sigma[app_num],
            color=experiment.colors[app_num], antialiased=True, alpha=0.7)

    return ax


def make_polar_plots(ax, sigma, experiment):
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
    experiment : Experiment instance
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
    x, y = (experiment.gamma * np.cos(experiment.theta), experiment.gamma *
            np.sin(experiment.theta))

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(experiment.Na):
        ax.plot_surface(
            x, y, sigma[app_num], color=experiment.colors[app_num],
            antialiased=True, alpha=0.7)

    # Adjusts ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$\gamma$ cos($\theta$)')
    ax.set_ylabel(r'$\gamma$ sin($\theta$)')
    return ax


def plot_gradients(im):
    ''' This function takes a 2D image and displays the gradient images.'''

    dy, dx = np.gradient(im)
    gamma = np.sqrt(dy**2 + dx**2)
    theta = np.arctan2(dy, dx)
    theta[theta < 0.0] += 2 * np.pi

    fig = plt.figure(1, figsize=(10, 7))

    ax1 = fig.add_subplot(221)
    imx = ax1.imshow(dx, cmap='gray')
    fig.colorbar(imx, ax=ax1)
    ax1.set_title('X Gradient')

    ax2 = fig.add_subplot(222)
    imy = ax2.imshow(dy, cmap='gray')
    fig.colorbar(imy, ax=ax2)
    ax2.set_title('Y Gradient')

    ax3 = fig.add_subplot(223)
    img = ax3.imshow(gamma, cmap='gray')
    fig.colorbar(img, ax=ax3)
    ax3.set_title('Gradient Magnitude')

    ax4 = fig.add_subplot(224)
    imt = ax4.imshow(theta, cmap='gray')
    fig.colorbar(imt, ax=ax4)
    ax4.set_title('Gradient Azimuth')
    fig.tight_layout()
    plt.show()


def plot_OPL_var(varOPL, experiment, show=True):
    '''For a given varOPL array and Experiment instance, plots
    the equal and non-equal dose image variances for each 
    acquisition approach'''

    width = 0.30
    n = len(experiment.approaches)

    plt.close()
    plt.bar(np.arange(n), np.sqrt(varOPL[0, :]), width, label='Equal Dose')
    plt.bar(np.arange(n) + width,
            np.sqrt(varOPL[1, :]), width, label='Non-equal Dose')
    plt.legend()
    plt.title('{}x Objective, {} gradient'.format(
        experiment.lens, 'Weak' if experiment.weak_grad else 'Normal'))
    plt.xlabel('Acquisition Approach')
    plt.ylabel(r'$\sigma_{OPL}$')
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.xticks(np.arange(n) + width / 2, experiment.approaches)
    plt.tight_layout()

    if show:
        plt.show()

    if experiment.save:
        plt.savefig(experiment.filepath + 'var_OPL_{}x_{}_gradient'.format(
            experiment.lens, 'Weak' if experiment.weak_grad else 'Normal'),
            dpi=400)
