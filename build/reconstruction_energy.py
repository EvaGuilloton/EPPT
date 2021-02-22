import ROOT
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import log10, floor
import scipy.optimize as opt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

import momentum_track_reconstruction as constr


# Choosing the case to study #
case = input("chose the case to study by entering : 1 for antimuon, 2 for positron, 3 for proton  \n")
if case == 1 :
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T.root")
    tree = inf.Get("B5")
    name_file = "mu_100GeV_05T" #Label to differenciate the output
    B = 0.5 # Stenght of the magnetic field in Tesla
    q = 1. # Charge of one particle
    f = 1. # Factor specific to the type of particle
    mass = 105.66E-3 #Mass of the particle in MeV
elif case == 2:
    inf = ROOT.TFile.Open("B5_out_e_100GeV_05T.root")
    tree = inf.Get("B5")
    name_file = "e_100GeV_05T" #Label to differenciate the output
    B = 0.5 # Stenght of the magnetic field in Tesla 
    q = 1. # Charge of one particle 
    f = 1. # Factor specific to the type of particle
    mass = 0.51E-3 #Mass of the particle in MeV
elif case == 3 :
    inf = ROOT.TFile.Open("B5_out_p_100GeV_05T.root")
    tree = inf.Get("B5")
    name_file = "p_100GeV_05T" #Label to differenciate the output
    B = 0.5 # Stenght of the magnetic field in Tesla 
    q = 1. # Charge of one particle 
    f = -1. # Factor specific to the type of particle
    mass = 938.27E-3 #Mass of the particle in MeV
else :
    print("not defined case")

 #Lists#
ecal_energy = [] #List to stock energy from the electromagnetic calorimeter
hcal_energy = [] #List to stock energy from the hadronic calorimeter
ecal_energy_corr = [] #List to stock energy from the electromagnetic calorimeter with correction
hcal_energy_corr = [] #List to stock energy from the hadronic calorimeter with correction
momentum = [] #List to stock momentum values
energy_nocorr = [] #List to stock the total energy from both calorimeters
energy_allcorr = [] #List to stock the total energy from both calorimeters with correction

 #Constants#
ratio = 20 #ratio of the active to passive layers of the detector
Z_CsI = 54 #Atomic number for the CsI scintillator (mean between the Cs atomic number and I atomic number)
X0_CsI = 1.86E-2 #Radiation lenght for CsI scintillator
E0 = 1.3 #in GeV
stock = 0 #To stock value to help


def round_to(x, y): #Function to round x to the fist non-zero number of y
    return round(x, -int(floor(log10(abs(y)))))

def plotEnergy(energy,name): # Plots reconstructed energy as a histogram
    plt.clf() #Close previous plot canvas
    
    y, binEdges = np.histogram(energy, 100)
    x = (binEdges[:-1] + binEdges[1:]) / 2
    x_width = (x[-1] - x[0]) / len(x)

#    popt, pcov = opt.curve_fit(constr.gauss, x, y, [100,np.std(energy), np.mean(energy)]) #Find the best amplitude, sigma and mu for a gaussian fit
    
    plt.bar(x, y, x_width, color="blue", edgecolor="black") #Plot the energy histo

#    x_int = np.linspace(x[0], x[-1], 10*len(x))
#    y_int = constr.gauss(x_int, *popt)

#    plt.plot(x_int, y_int, label="Gaussian fit", color="red") #Plot the gaussian fit
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel("Reconstructed energy [GeV]", fontsize = 16)
    plt.ylabel("Number of events", fontsize = 16)
#    plt.legend()
#    plt.annotate("sigma (GeV): " + str(round_to(popt[1], pcov[1,1])) + " $\pm$ " + str(round_to(pcov[1,1], pcov[1,1])) + '\n' + "mu (GeV): " + str(round_to(popt[2], pcov[2,2])) + " $\pm$ " + str(round_to(pcov[2,2], pcov[2,2])), xy=(0.025, 0.8), xycoords='axes fraction')
    #    plt.show() 
    plt.savefig("Reconstructed_energy_"+name+".pdf")
    #print("[amplitude, sigma, mu]")
    #print(popt)

def EMCal_correction(energy,f): #Compute the correction for the electromagnetic calorimeter (explanation in the overleaf)
    epsilon = 0.61 / (Z_CsI + 1.24)
    depth = 0.3 / (X0_CsI)
    tmax = np.log(energy/1000/epsilon) - f*0.5
    t95 = tmax + 0.08*54 + 9.6
    epsilon_corr = (t95/tmax)*epsilon
    return epsilon_corr * ( 1 - np.exp(t95) ) / ( 1 - np.exp(depth) )

def theorical_energy(momentum,mass): #Compute theorical energy for the particle with relativistic relation
    theo_energy = [np.sqrt(i**2 + mass**2) for i in momentum]
    return theo_energy


for event in tree: # Loop over the events
    if event.ECEnergy != 0:
        ecal_energy.append(event.ECEnergy/1000) #Add the energy value from the electromagnetic calorimeter (/1000 for units)
        ecal_energy_for_corr = EMCal_correction(event.ECEnergy,f) #Compute and stock the correction for the electromagnetic calorimeter
        ecal_energy_corr.append(event.ECEnergy/1000+ecal_energy_for_corr) #Stock the energy with the correction for the electromagnetic calorimeter
    if event.HCEnergy != 0:
        hcal_energy.append(event.HCEnergy/1000) #Add the energy value from the hadronic calorimeter (/1000 for units) 
        hcal_energy_for_corr = event.HCEnergy/1000 #Stock the value of this event energy from the hadronic calorimeter
        if hcal_energy_for_corr>E0 : stock = hcal_energy_for_corr*E0 #First correction for the ionisation lost
        hcal_energy_for_corr=event.HCEnergy/1000*ratio + stock #Second correction for ratio lost applied and adding of the first correction
        hcal_energy_corr.append(hcal_energy_for_corr) #Stock in a list
    energy_nocorr.append(event.ECEnergy/1000 + event.HCEnergy/1000) #List for energy from electromagnetic and hadronic calorimeter without any correction
    energy_allcorr.append((event.ECEnergy)/1000 + hcal_energy_for_corr + ecal_energy_for_corr) #List for energy from electromagnetic and hadronic calorimeter with all corrections                                                           

    if len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5: #Only for events with 5 hits in both chambers
        m1,c1,position1 = constr.trackReconstruction(event, 1) #Compute the best factor for linear fit and position of the hits in the first arm
        m2,c2,position2 = constr.trackReconstruction(event, 2) #Compute the best factor for linear fit and position of the hits in the second arm
        computedMom = constr.determineMomentum(m1,c1,m2,c2) #Compute the momentum value of the particle
        momentum.append(abs(computedMom)) #Stock the value in a list

energy_theo = theorical_energy(momentum,mass) #Compute of the theorical value of energy


#Plots of the different energy, from electromagnetic and hadronic calorimeter, with and without the correction
plotEnergy(ecal_energy,name=name_file+"EMcal_wo_corr")

plotEnergy(hcal_energy,name=name_file+"Hcal_wo_corr")

plotEnergy(energy_nocorr,name=name_file+"total_energy_wo_corr")

#plotEnergy(energy_theo,name=name_file+"total_theo")

#plotEnergy(energy_allcorr,name=name_file+"total_energy_allcorr")

plotEnergy(ecal_energy_corr,name=name_file+"EMcal_corr")

plotEnergy(hcal_energy_corr,name=name_file+"Hcal_corr")
