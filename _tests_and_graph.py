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
import numpy as np

#%%
start = 104
stop = 150
bin_num = 93
edges = sp.linspace(start, stop, num=bin_num+1, endpoint=True)

def BG(x, A, lamb):
    B = A * sp.exp( - x / lamb)
    return B

def SpBG(x, A, lamb, mu, sig, signal_amp):
    B = A * sp.exp( - x / lamb)
    S = stom.signal_gaus(x, mu, sig, signal_amp)
    return S + B


#%%                           Background only from here
iters = 25
chiList = sp.zeros(iters)
BG_opt_list = sp.zeros((iters, 2))
#%% 
np.random.seed(1)
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
chiMean_B, chiStd_BG = sp.mean(chiList), sp.std(chiList)
BG_opt_means = sp.mean(BG_opt_list, axis=0)
#%%
#vals = stom.generate_data()
x = sp.linspace(start, stop, bin_num)
plt.plot(x, BG(x, *BG_opt_means), label= u'H\u2080' )
#plt.hist(vals, range = [start,stop], bins = bin_num)

#%% #%%                           Background plus Signal from here
iters = 25
chiList = sp.zeros(iters)
SpBG_opt_list = sp.zeros((iters, 5))
#%% 
np.random.seed(1)
mus = sp.zeros(iters)
mu_stds = sp.zeros(iters)
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

    SpBG_opt, SpBG_cov = op.curve_fit(f=SpBG, xdata=means, ydata=freqs,
                                  p0=(18000, 30, 125, 1.25, 200))
    SpBG_chi = stom.get_SB_chi(vals, mass_range=(start, stop), nbins=bin_num,
                           A=SpBG_opt[0], lamb=SpBG_opt[1], mu =SpBG_opt[2] , sig = SpBG_opt[3], amp = SpBG_opt[4])
    mus[i] = SpBG_opt[2]
    mu_stds[i] = sp.sqrt(SpBG_cov[2,2])
    chiList[i] = SpBG_chi
    SpBG_opt_list[i] = SpBG_opt

#%%
chiMean_SpBG, chiStd_SpBG = sp.mean(chiList), sp.std(chiList)
SpBG_opt_means = sp.mean(SpBG_opt_list, axis=0)
#%%
#vals = stom.generate_data()

print("BACKGROUND ONLY")
print("The mean background-only reduced chi square is",chiMean_B)
print("The mean background-only  chi square is",chiMean_B*(bin_num-5))
print("The background only standard error is", chiStd_BG/sp.sqrt(iters))
print("")

print("SIGNAL PLUS BACKGROUND")
print("The mean signal plus background reduced chi square is",chiMean_SpBG)
print("The mean signal plus bakcground chi square is",chiMean_SpBG*(bin_num-5))
print("The signal plus background only standard error is", chiStd_SpBG/sp.sqrt(iters))


x = sp.linspace(start, stop, bin_num)
plt.plot(x, SpBG(x, *SpBG_opt_means), label = u'H\u2081')
#plt.hist(vals, range = [start,stop], bins = bin_num)
plt.title("Histogram of rest-mass energies of events", fontname = "Times New Roman", fontsize=15)
plt.ylabel("Frequency of event (F)", fontname = "Times New Roman", fontsize=13)
plt.xlabel("Rest-mass energy of event in GeV", fontname = "Times New Roman", fontsize=13)
plt.legend(loc='upper right')
plt.show()



#%% 





