import ROOT
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import scipy.optimize as opt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

import momentum_track_reconstruction as constr


# Arrays and files #
case = input("chose the case to study by entering : 1 for antimuon, 2 for positron, 3 for proton  \n")
if case == 1 :
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T.root")
    tree = inf.Get("B5")
    name_file = "mu_100GeV_05T"
    B = 0.5
    q = 1.
    mass = 105.66E-3 #MeV
elif case == 2:
    inf = ROOT.TFile.Open("B5_out_e_100GeV_05T.root")
    tree = inf.Get("B5")
    name_file = "e_100GeV_05T"
    B = 0.5
    q = 1.
    mass = 0.51E-3 #MeV 
elif case == 3 :
    inf = ROOT.TFile.Open("B5_out_p_100GeV_05T.root")
    tree = inf.Get("B5")
    name_file = "p_100GeV_05T"
    B = 0.5
    q = 1.
    mass = 938.27E-3 #MeV     
else :
    print("not defined case")

ecal_energy = []
hcal_energy = []
momentum = []

def plotEnergy(energy,name): # Plots reconstructed energy as a histogram
    plt.clf()
    n, bins, patches = plt.hist(energy, bins=50, edgecolor = "black", color = "blue", histtype = 'stepfilled')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel("Reconstructed energy [GeV]", fontsize = 16)
    plt.ylabel("Number of events", fontsize = 16)
    plt.show()
    plt.savefig("Reconstructed_energy_"+name+".pdf")
    return n, bins


def theorical_energy(momentum,mass):
    theo_energy = [np.sqrt(i**2 + mass**2) for i in momentum]
    return theo_energy


for event in tree: # Main function to run
    if event.ECEnergy != 0:
        ecal_energy.append(event.ECEnergy)
    if event.HCEnergy != 0:
        hcal_energy.append(event.HCEnergy)
    if len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5:
        m1,c1,position1 = constr.trackReconstruction(event, 1)
        m2,c2,position2 = constr.trackReconstruction(event, 2)
        recMom = constr.determineMomentum(m1,c1,m2,c2)
        momentum.append(abs(recMom))


energy_total = ecal_energy+hcal_energy
energy_theo = theorical_energy(momentum,mass)


n, bins = plotEnergy(ecal_energy,name=name_file+"EMcal")

n2, bins2 = plotEnergy(hcal_energy,name=name_file+"Hcal")

n3, bins3 = plotEnergy(energy_nocorr,name=name_file+"total_energy")

n4, bins4 = plotEnergy(energy_theo,name=name_file+"total_theo")
