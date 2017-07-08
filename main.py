import generate_data as data
import generate_plots as plots

setup = data.Setup(lens=40,
                   weak_grad=False,
                   approaches=['2x2', '2x3', '2x4'],
                   k=0.1,
                   lamda=546,
                   fromZero=True,
                   save=False,
                   filepath=None)


sigma_g, sigma_t, setup = data.generate_data(setup, sample_size=100)
plots.generate_plots(sigma_g, sigma_t, setup, polar=False)
