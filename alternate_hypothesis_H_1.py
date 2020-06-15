# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 01:24:58 2020

@author: AlexanderSinclairTeixeira
Generates gaussian signal data amongst an exponential background distribution.
Models BG distribution in mass range below 120, showing it can be effectively
paramterised (chi square value close to 1).
"""
#%%
import STOM_higgs_tools as stom
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as op
import scipy.stats as st
import numpy as np

#%%
def SpBG(x, A, lamb, mu, sig, signal_amp):
    B = A * sp.exp( - x / lamb)
    S = stom.signal_gaus(x, mu, sig, signal_amp)
    return S + B

np.random.seed(1)
start = 104
stop = 150
bin_num = 30
edges = sp.linspace(start, stop, num=bin_num+1, endpoint=True)
width = edges[1]-edges[0]
x_mid = (width/2 + edges)[:-1]

iters = 250
SpBG_opts = sp.zeros((iters, 5))
SpBG_chisqs = sp.zeros(iters)
SpBG_ps = sp.zeros(iters)
mus = sp.zeros(iters)
mu_stds = sp.zeros(iters)
freq_hist = sp.zeros((iters, bin_num))

for i in range(0,iters):
    vals = stom.generate_data()
    means = sp.zeros(bin_num)
    freqs = sp.zeros(bin_num)
    bins = [ [] for i in range(bin_num) ]
    print(iters - i)
    for val in vals:
        if start < val and val < stop:
            bins[sp.searchsorted(edges, val)-1].append(val)
    for j, Bin in enumerate(bins):
        means[j] = sp.mean(Bin)
        freqs[j] = len(Bin)
    freq_hist[i] = freqs

    SpBG_opt, SpBG_cov = op.curve_fit(f=SpBG, xdata=means, ydata=freqs,
                                  p0=(45000, 30, 125, 1.25, 700))
    SpBG_opts[i] = SpBG_opt
    mus[i] = SpBG_opt[2]
    SpBG_chisqs[i], SpBG_ps[i] = st.chisquare(f_obs=freqs,
               f_exp=stom.get_SB_expectation(x_mid, *SpBG_opt), ddof=4)

SpBG_red_chis = SpBG_chisqs / (bin_num - 5)
#%%
SpBG_opt_means = sp.mean(SpBG_opts, axis=0)
SpBG_chi_mean, SpBG_chi_std = sp.mean(SpBG_chisqs), sp.std(SpBG_chisqs)
SpBG_red_chi_mean, SpBG_red_chi_std = sp.mean(SpBG_red_chis), sp.std(SpBG_red_chis)
SpBG_p_mean, SpBG_p_std = sp.mean(SpBG_ps), sp.std(SpBG_ps)
mu_mean, mu_std = sp.mean(mus), sp.std(mus)
freq_means = freq_hist.mean(axis=0)
freq_errors = freq_hist.std(axis=0)/sp.sqrt(iters)

#%%
print("SIGNAL PLUS BACKGROUND")
print("%.i iterations with %.i bins over a mass range from %.i to %.i Gev"
      % (iters, bin_num, start, stop) )
print("The mean chi square is %.4f +/- %.4f"
      % (SpBG_chi_mean, SpBG_chi_std/sp.sqrt(iters)) )
print("The mean reduced chi square is %.4f +/- %.4f"
      % (SpBG_red_chi_mean, SpBG_red_chi_std/sp.sqrt(iters)) )
print("The mean p value is %.4f +/- %.4f"
      % (SpBG_p_mean, SpBG_p_std/sp.sqrt(iters)) )
print("The mean mu is %.4f +/- %.4f GeV"
      % (mu_mean, mu_std / sp.sqrt(iters)) )

#%%
"""
SAMPLE RUNS
250 iterations with 30 bins over a mass range from 104 to 150 Gev
The mean chi square is 24.3523 +/- 0.4305
The mean reduced chi square is 0.9741 +/- 0.0172
The mean p value is 0.5243 +/- 0.0182
The mean mu is 124.9933 +/- 0.0213 GeV
"""