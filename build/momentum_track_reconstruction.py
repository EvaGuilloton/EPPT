import ROOT
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import scipy.optimize as opt
import matplotlib.mlab as mlab
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial


# Arrays and files #
case = input("1: mu,100GeV,0.5T; 2: mu,100GeV,0.25T; 3: mu,100GeV,1T; 4: mu,50GeV,0.5T; 5: mu,200GeV,0.5T; 6: with delta_x = 0; 7: with lead layer before magnet; 8: with lead layer after magnet")

if case == 1:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5
    delta_x = 1E-4
    name = "mu_100GeV_05T"
elif case == 2:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_025T.root")
    tree = inf.Get("B5")
    B = 0.25
    delta_x = 1E-4
    name = "mu_100GeV_025T"
elif case == 3:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_1T.root")
    tree = inf.Get("B5")
    B = 1.
    delta_x = 1E-4
    name = "mu_100GeV_1T"
elif case == 4:
    inf = ROOT.TFile.Open("B5_out_mu_50GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5
    delta_x = 1E-4
    name = "mu_50GeV_05T"
elif case == 5:
    inf = ROOT.TFile.Open("B5_out_mu_200GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5
    delta_x = 1E-4
    name = "mu_200GeV_05T"
elif case == 6:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5
    delta_x = 0
    name = "mu_100GeV_05T_precise"
elif case == 7:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T_bef_lead.root")
    tree = inf.Get("B5")
    B = 0.5
    delta_x = 1E-4
    name = "mu_100GeV_05T_bef_lead"
elif case == 8:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T_aft_lead.root")
    tree = inf.Get("B5")
    B = 0.5
    delta_x = 1E-4
    name = "mu_100GeV_05T_aft_lead"
else : 
    print("input invalide")

p = []

# Constants #
L = 2.0 # Length of magnetic field chamber
dz = 0.5 # Drift chamber wire separation distances


def linear(m,x,c): # Straight line function (y=mx+c) to plot reconstructed xz-plane track
    return m*x+c

def gauss(x, a, sigma, mean):
    amplitude = a / ( sigma * np.sqrt(2 * np.pi) )
    part_2 = (x - mean) / sigma
    return amplitude * np.exp( -0.5 * (part_2**2) )

def trackReconstruction(event, chamber): # Reconstruct the xz-plane tracks to output the intercept and gradient of the best fit lline
    if chamber == 1:
        xz_pos = [event.Dc1HitsVector_z,event.Dc1HitsVector_x]
    elif chamber == 2:
        xz_pos = [event.Dc2HitsVector_z,event.Dc2HitsVector_x]
    coordinates = []
    for i in range(len(xz_pos[0])):
        coordinates.append([xz_pos[0][i]*dz, xz_pos[1][i]/1000])
    def fit(next):
        nextx = 0
        for z,x in coordinates:
            xp = next[0]*z+next[1]
            nextx += (x-xp)**2 
        return nextx
    m = (coordinates[-1][1]-coordinates[0][1])/(coordinates[-1][0]-coordinates[0][0])
    c = coordinates[0][1] 
    minimum = opt.minimize(fit, x0=[m,c])
    m = minimum["x"][0]
    c = minimum["x"][1]

    return m, c, coordinates

def plotTrack(m1,c1,position1,m2,c2,position2,name):
    plt.clf()
    plt.figure()
    h = plt.gca()
    z1 = np.linspace(-6.25,0.5,676)
    z2 = np.linspace(-0.5,4.25,476)
    x1 = []
    x2 = []
    for i in z1 :
        x1.append(linear(m1,i,c1))
    for i in z2 :
        x2.append(linear(m2,i,c2))
    plt.plot(z1,x1)
    plt.plot(z2,x2)
    h.add_patch(Rectangle((-1,-1), 2, 2, edgecolor='black',facecolor='none'))
    plt.scatter(position1[0],position1[1])
    plt.scatter(position2[0],position2[1])
    plt.xlabel("z (m)")
    plt.ylabel("x (m)")
    plt.savefig("track_recons_"+name+".pdf")
#    plt.show()



def determineMomentum(m1,c1,m2,c2): # Calculates sigma, dx and momentum using Eq.(5) in the notes
    sigma = math.atan((m2-m1)/(1+m1*m2))
    dx = abs((m1*(-1)+c1)-(m2*(1)+c2))
    return (0.3*B*math.sqrt(L**2 + dx**2))/sigma


def plotMomenta(momenta,name): # Plots reconstructed momentum as a histogram and calculates the resolution (width)
    plt.clf()
  
    y, binEdges = np.histogram(momenta, 50)
    x = (binEdges[:-1] + binEdges[1:]) / 2
    x_width = (x[-1] - x[0]) / len(x)

    popt, pcov = opt.curve_fit(gauss, x, y, [300,np.std(momenta), np.mean(momenta)])

    plt.bar(x, y, x_width, color="blue", edgecolor="black")
    
    x_int = np.linspace(x[0], x[-1], 10*len(x))
    y_int = gauss(x_int, *popt)
    plt.plot(x_int, y_int, label="Gaussian fit", color="red") 
    

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel("Reconstructed momentum [GeV/c]", fontsize = 16)
    plt.ylabel("Number of events", fontsize = 16)
    #    plt.show()
    plt.savefig("Recons_momentum_"+name+".pdf")
    print("[amplitude, sigma, mu]")
    print(popt)

#def resolution(momenta):
    
#    return resMomenta



for event in tree:
    if len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5:
        m1,c1,position1 = trackReconstruction(event, 1)
        m2,c2,position2 = trackReconstruction(event, 2)
        plotTrack(m1,c1,position1,m2,c2,position2,name)
        recMom = determineMomentum(m1,c1,m2,c2)
        p.append(abs(recMom))
plotMomenta(p,name)
#resMom=resolution(momenta)
