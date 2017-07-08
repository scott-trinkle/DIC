'''
add documentation
remove superfluous comments
add print statements
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.ticker as tck


def generate_plots(sigma_g, sigma_t, setup, polar=False,
                   fignum=1, size=(14, 7.8)):

    setup.polar = polar

    if polar and not setup.fromZero:
        raise ValueError('Gamma range must be set to zero for polar plotting')

    # colors for different aquisition approaches within a single plot.
    # Will need to extend if we ever look at more than 3 approaches.
    setup.colors = ['maroon', 'seagreen', 'royalblue']

    plt.close()
    fig = plt.figure(fignum, figsize=size)

    # Title for entire figure
    fig.suptitle('{}x Objective, {} gradient'.format(
        setup.lens, 'Weak' if setup.weak_grad else 'Normal'), weight='bold')

    fig = make_figure(fig, sigma_g, sigma_t, setup)

    # saves figure if selected
    if setup.save:
        plt.savefig(setup.filepath + '{}x_{}.png'.format(
            setup.lens, 'weak' if setup.weak_grad else 'normal'), dpi=215)

    plt.show()  # shows figure regardless of saving

    return


def make_figure(fig, sigma_g, sigma_t, setup):

    # iterates through equalize_dose=True and False as well as
    # variables (gamma and theta) to add Axes instances to fig
    ax = []
    for dose_num, equalize_dose in enumerate((True, False)):
        for var_num, var in enumerate(['gamma', 'theta']):

            # flattens two numerical indices to increment as [0, 1, 2, 3]
            ind = dose_num * 2 + var_num

            # adds to 4x4 grid of 3D subplots
            ax.append(fig.add_subplot(2, 2, ind + 1, projection='3d'))

            # adds surface plot to active subplot

            if setup.polar:
                ax[ind] = make_polar_plots(
                    ax[ind],
                    sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                    var, setup)
            else:
                ax[ind] = make_cartesian_plots(
                    ax[ind],
                    sigma_g[dose_num] if var == 'gamma' else sigma_t[dose_num],
                    var, setup)

            # sets title of subplot
            title = '{} dose CRLB for '.format(
                'Equal' if equalize_dose else 'Non-equal') + r'$\sigma_{}^2$'.format(
                r'{\gamma}' if var == 'gamma' else r'{\theta}')
            ax[ind].set_title(title)

    # Adds legend to FIGURE, not to each axis, since they're the same.
    # There's not a great way to do this with surfaces, you have to set up
    # empty "proxy artists" with the same color.
    fig.legend([Line2D([], [], linestyle='-', color=setup.colors[kk])
                for kk in range(setup.Na)],
               [approach + ' frames' for approach in setup.approaches],
               numpoints=1, loc=10, fontsize='large')

    plt.tight_layout()  # makes things look nice

    # so subplots don't overlap the main title
    plt.subplots_adjust(top=0.95)

    return fig


def setup_ax(ax, elev, azim):

    ax.view_init(elev=elev, azim=azim)  # Set view angle

    ax.xaxis.set_major_locator(tck.MultipleLocator(0.1))
    ax.xaxis.set_label_text(r'$\gamma(x,y)$ [nm/nm]')

    ax.yaxis.set_major_locator(tck.LinearLocator(3))
    ax.yaxis.set_major_formatter(tck.FixedFormatter((r'0', r'$\pi$',
                                                     r'$2\pi$')))
    ax.yaxis.set_label_text(r'$\theta (x,y)$')

    ax.zaxis.set_major_locator(tck.MaxNLocator(5))

    return ax


def make_cartesian_plots(ax, sigma, var, setup):

    setup_ax(ax, elev=15, azim=-70)

    if setup.fromZero and var == 'theta':
        sigma[np.where(sigma == np.inf)] = sorted(set(sigma.flatten()))[-2]
        sigma = np.log10(sigma)

        ax.zaxis.set_major_locator(tck.MultipleLocator(1))
        zticks = ax.get_zticks()
        ax.set_zticklabels(
            ['10$^{}$'.format(r'{' + str(int(tick)) +
                              r'}') for tick in zticks])

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(setup.Na):
        ax.plot_surface(
            setup.gamma, setup.theta, sigma[app_num],
            color=setup.colors[app_num], antialiased=True, alpha=0.7)

    return ax


def make_polar_plots(ax, sigma, var, setup):

    setup_ax(ax, elev=1, azim=-70)

    if var == 'theta':
        sigma[np.where(sigma == np.inf)] = sorted(set(sigma.flatten()))[-2]
        sigma = np.log10(sigma)

        ax.zaxis.set_major_locator(tck.MultipleLocator(1))
        zticks = ax.get_zticks()
        ax.set_zticklabels(
            ['10$^{}$'.format(r'{' + str(int(tick)) +
                              r'}') for tick in zticks])

    x, y = (setup.gamma * np.cos(setup.theta), setup.gamma *
            np.sin(setup.theta))

    # iterates through approaches to plot Na surfaces on each Axes object
    for app_num in range(setup.Na):
        ax.plot_surface(
            x, y, sigma[app_num], color=setup.colors[app_num],
            antialiased=True, alpha=0.7, zorder=3 - app_num)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$\gamma$ cos($\theta$)')
    ax.set_ylabel(r'$\gamma$ sin($\theta$)')
    return ax
