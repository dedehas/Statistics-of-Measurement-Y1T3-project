# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:11:21 2020

@author: AlexanderSinclairTeixeira

Generates gaussian signal data amongst an exponential background distribution.
Models BG distribution in mass range below 120, showing it can be effectively
paramterised (chi square value close to 1). Apply the same test to the
104 - 150 GeV range (change start and stop, lines 26-27) to show that this is
a weak hypothesis for this range (p value rejects H_0).
"""
#%%
import STOM_higgs_tools as stom
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as op
import scipy.stats as st
import numpy as np

#%%
def BG(x, A, lamb):
    BG = A * sp.exp( - x / lamb)
    return BG

np.random.seed(1)
start = 104
stop = 150
bin_num = 30
edges = sp.linspace(start, stop, num=bin_num+1, endpoint=True)
width = edges[1]-edges[0]
x_mid = (width/2 + edges)[:-1]

iters = 250
BG_opts = sp.zeros((iters, 2))
BG_chisqs = sp.zeros(iters)
BG_ps = sp.zeros(iters)

for i in range(iters):
    vals = stom.generate_data()
    means = sp.zeros(bin_num)
    freqs = sp.zeros(bin_num)
    bins = [ [] for i in range(bin_num) ]
    print(iters - i) #countdown
    for val in vals:
        if start < val and val < stop:
            bins[sp.searchsorted(edges, val)-1].append(val)
    for j, Bin in enumerate(bins):
        means[j] = sp.mean(Bin)
        freqs[j] = len(Bin)

    BG_opt, BG_cov = op.curve_fit(f=BG, xdata=means, ydata=freqs,
                                  p0=(20000, 30) )
    BG_opts[i] = BG_opt
    BG_chisqs[i], BG_ps[i] = st.chisquare(f_obs=freqs,
                 f_exp=stom.get_B_expectation(x_mid, *BG_opt), ddof=1)#params-1

BG_red_chis = BG_chisqs / (bin_num - 2) #div by degrees of freedom
#%%
BG_opt_means = sp.mean(BG_opts, axis=0)
BG_chi_mean, BG_chi_std = sp.mean(BG_chisqs), sp.std(BG_chisqs)
BG_red_chi_mean, BG_red_chi_std = sp.mean(BG_red_chis), sp.std(BG_red_chis)
BG_p_mean, BG_p_std = sp.mean(BG_ps), sp.std(BG_ps)

#%%
print("BACKGROUND ONLY")
print("%.i iterations with %.i bins over a mass range from %.i to %.i Gev"
      % (iters, bin_num, start, stop) )
print("The mean chi square is %.4f +/- %.4f"
      % (BG_chi_mean, BG_chi_std/sp.sqrt(iters)) )
print("The mean reduced chi square is %.4f +/- %.4f"
      % (BG_red_chi_mean, BG_red_chi_std/sp.sqrt(iters)) )
print("The mean p value is %.4f +/- %.4f"
      % (BG_p_mean, BG_p_std/sp.sqrt(iters)) )

#%%
"""
SAMPLE RUNS
250 iterations with 30 bins over a mass range from 104 to 120 Gev
The mean chi square is 28.0245 +/- 0.4910
The mean reduced chi square is 1.0009 +/- 0.0175
The mean p value is 0.5040 +/- 0.0190

250 iterations with 30 bins over a mass range from 104 to 150 Gev
The mean chi square is 77.3825 +/- 0.8981
The mean reduced chi square is 2.7637 +/- 0.0321
The mean p value is 0.0006 +/- 0.0002
"""