import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_data(n_signals = 400):
    '''
    Generate a set of values for signal and background. Input arguement sets
    the number of signal events, and can be varied (default to higgs-like at
    announcement).

    The background amplitude is fixed to 9e5 events, and is modelled as an exponential,
    hard coded width. The signal is modelled as a gaussian on top (again, hard
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals


def generate_signal(N, mu, sig):
    '''
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()


def generate_background(N, tau):
    '''
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()


def get_B_chi(vals, mass_range, nbins, A, lamb):
    '''
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form,
    using the hard coded scale of the exp. That depends on the binning, so pass
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0

    # Loop over bins - all of them for now.
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator

    return chi/float(nbins-2) # B has 2 parameters.

def get_SB_chi(vals, mass_range, nbins, A, lamb, mu, sig, amp):
    """
    Calculates reduced chi square for 5 params (sig plus background)
    """
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, mu, sig, amp)
    chi = 0

    # Loop over bins - all of them for now.
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator

    return chi/float(nbins-5) # SB has 5 parameters.


def get_B_expectation(xs, A, lamb):
    '''
    Return a set of expectation values for the background distribution for the
    passed in x values.
    '''
    return [A*np.exp(-x/lamb) for x in xs]


def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys


