# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:01:32 2020

@author: alesi
"""
#%%
import STOM_higgs_tools as stom
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as op

#%%
start = 0.0
stop = 120
bin_num = 120
edges = sp.linspace(start, stop, num=bin_num+1, endpoint=True)

def BG(x, A, lamb):
    B = A * sp.exp( - x / lamb)
    return B

def SpBG(x, A, lamb, mu, sig, signal_amp):
    B = A * sp.exp( - x / lamb)
    S = stom.signal_gaus(x, mu, sig, signal_amp)
    return S + B


#%%
iters = 10
chiList = sp.zeros(iters)
BG_opt_list = sp.zeros((iters, 2))
#%%
for i in range(0,iters):
    means = sp.zeros(bin_num)
    freqs = sp.zeros(bin_num)
    vals = stom.generate_data()
    print(i)
    bins = [ [] for i in range(bin_num) ]
    for val in vals:
        if start < val and val < stop:
            bins[sp.searchsorted(edges, val)-1].append(val)

    for j, Bin in enumerate(bins):
        means[j] = sp.mean(Bin)
        freqs[j] = len(Bin)

    BG_opt, BG_cov = op.curve_fit(f=BG, xdata=means, ydata=freqs,
                                  p0=(18000, 30) )
    B_chi = stom.get_B_chi(vals, mass_range=(start, stop), nbins=bin_num,
                           A=BG_opt[0], lamb=BG_opt[1])
    chiList[i] = B_chi
    BG_opt_list[i] = BG_opt

#%%
chiMean, chiStd = sp.mean(chiList), sp.std(chiList)
BG_opt_means = sp.mean(BG_opt_list, axis=0)
#%%
#vals = stom.generate_data()
x = sp.linspace(start, stop, bin_num)
plt.plot(x, BG(x, *BG_opt_means) )
plt.hist(vals, range = [start,stop], bins = bin_num)
