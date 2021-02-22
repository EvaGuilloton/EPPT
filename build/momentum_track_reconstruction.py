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


# Choosing the case to study #
case = input("Choose the case to study : 1: 100GeV,0.5T; 2: 100GeV,0.25T; 3: 100GeV,1T; 4: 50GeV,0.5T; 5: 200GeV,0.5T; 6: with delta_x = 0; 7: with lead layer before magnet; 8: with lead layer after magnet   ")

if case == 1:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 1E-4 #Precision on position in x-coordinate
    name = "mu_100GeV_05T" #label to help differenciate the ouputs
elif case == 2:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_025T.root")
    tree = inf.Get("B5")
    B = 0.25 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 1E-4 #Precision on position in x-coordinate
    name = "mu_100GeV_025T" #label to help differenciate the ouputs
elif case == 3:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_1T.root")
    tree = inf.Get("B5")
    B = 1. #Strenght of the magnetic field in Tesla
    res_x = 1E-4 #Precision on position in x-coordinate
    q = 1. #Charge of one particle
    name = "mu_100GeV_1T" #label to help differenciate the ouputs
elif case == 4:
    inf = ROOT.TFile.Open("B5_out_mu_50GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 1E-4 #Precision on position in x-coordinate
    name = "mu_50GeV_05T" #label to help differenciate the ouputs
elif case == 5:
    inf = ROOT.TFile.Open("B5_out_mu_200GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 1E-4 #Precision on position in x-coordinate
    name = "mu_200GeV_05T" #label to help differenciate the ouputs
elif case == 6:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T.root")
    tree = inf.Get("B5")
    B = 0.5 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 0 #Precision on position in x-coordinate
    name = "mu_100GeV_05T_precise" #label to help differenciate the ouputs
elif case == 7:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T_bef_lead.root")
    tree = inf.Get("B5")
    B = 0.5 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 1E-4 #Precision on position in x-coordinate
    name = "mu_100GeV_05T_bef_lead" #label to help differenciate the ouputs
elif case == 8:
    inf = ROOT.TFile.Open("B5_out_mu_100GeV_05T_aft_lead.root")
    tree = inf.Get("B5")
    B = 0.5 #Strenght of the magnetic field in Tesla
    q = 1. #Charge of one particle
    res_x = 1E-4 #Precision on position in x-coordinate
    name = "mu_100GeV_05T_aft_lead" #label to help differenciate the ouputs
else : 
    print("input invalide")

#Lists#

momentum = [] #List to stock values of momentum

#Constants#
L = 2.0 # Length of magnetic field chamber
dz = 0.5 # Drift chamber wire separation distances
larm_2 = 4.25 # Length of the second arm (taken between the center of the magnetic field chamber and the last wire

def linear(m,x,c): # Straight line function (y=mx+c) to plot reconstructed xz-plane track
    return m*x+c

def gauss(x, a, sigma, mean): #Gaussian function to fit the momentum data
    amplitude = a / ( sigma * np.sqrt(2 * np.pi) )
    part_2 = (x - mean) / sigma
    return amplitude * np.exp( -0.5 * (part_2**2) )

def trackReconstruction(event, chamber): # Reconstruct the xz-plane tracks and find the best linear fit for each track
    if chamber == 1:
        xz_pos = [event.Dc1HitsVector_z,event.Dc1HitsVector_x]
    elif chamber == 2:
        xz_pos = [event.Dc2HitsVector_z,event.Dc2HitsVector_x]
    coordinates = [] #List to stocks the coordinates of the hits in the xz-plane 
    for i in range(len(xz_pos[0])):
        coordinates.append([xz_pos[0][i]*dz, xz_pos[1][i]/1000]) #Adding of the the z position, and then the x position of the hit (/1000 for units)
    def fit(next): #Chi2 method to find the best linear fit
        nextx = 0
        for z,x in coordinates:
            xp = next[0]*z+next[1]
            nextx += (x-xp)**2 
        return nextx
    m = (coordinates[-1][1]-coordinates[0][1])/(coordinates[-1][0]-coordinates[0][0]) #Compute an initial m value to help the fitting
    c = coordinates[0][1] #Compute an initial c value to help the fitting
    minimum = opt.minimize(fit, x0=[m,c]) #Find the best m and c value for the fit by chi2 method
    m = minimum["x"][0] #Stock the best m value
    c = minimum["x"][1] #Stock the best c value

    return m, c, coordinates

def plotTrack(m1,c1,position1,m2,c2,position2,name): #Plot the fitted tracks for one event
    plt.clf() #Close previous plot canvas
    plt.figure() #Open a new canvas
    h = plt.gca()
    z1 = np.linspace(-6.25,0.5,676) #z values for the tracks in the first arm
    z2 = np.linspace(-0.5,4.25,476) #z values for the tracks in the secnd arm
    x1 = [] #List to stock x values corresponding to the first arm
    x2 = [] #List to stock x values corresponding to the second
    for i in z1 :
        x1.append(linear(m1,i,c1)) #Adding the x position corresponding to a linear fit of the tracks in the first arm
    for i in z2 :
        x2.append(linear(m2,i,c2)) #Adding the x position corresponding to a linear fit of the tracks in the second arm 
    plt.plot(z1,x1) #Plot the track in the first arm
    plt.plot(z2,x2) #Plot the track in the second arm
    h.add_patch(Rectangle((-1,-1), 2, 2, edgecolor='black',facecolor='none')) #Add a square corresponding to the magnetic field chamber
    plt.scatter(position1[0],position1[1])
    plt.scatter(position2[0],position2[1])
    plt.xlabel("z (m)") 
    plt.ylabel("x (m)")
    plt.savefig("track_recons_"+name+".pdf")
#    plt.show()



def determineMomentum(m1,c1,m2,c2): # Calculate the momentum (equation explanation in the overleaf)
    sigma = math.atan((m2-m1)/(1+m1*m2))
    dx = abs((m1*(-1)+c1)-(m2*(1)+c2))
    return (0.3*B*math.sqrt(L**2 + dx**2))/(2*math.sin(sigma/2))


def plotMomenta(momenta,name): #Hist of the momentum
    plt.clf() #Close previous plot canvas
  
    y, binEdges = np.histogram(momenta, 50)
    x = (binEdges[:-1] + binEdges[1:]) / 2
    x_width = (x[-1] - x[0]) / len(x)

    popt, pcov = opt.curve_fit(gauss, x, y, [300,np.std(momenta), np.mean(momenta)]) #Compute the best amplitude, sigma and mu for a gaussian fit

    plt.bar(x, y, x_width, color="blue", edgecolor="black") #Plot the momentum data
    
    x_int = np.linspace(x[0], x[-1], 10*len(x))
    y_int = gauss(x_int, *popt)
    plt.plot(x_int, y_int, label="Gaussian fit", color="red") #Plot the gaussian fit
    

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel("Reconstructed momentum [GeV/c]", fontsize = 16)
    plt.ylabel("Number of events", fontsize = 16)
    #    plt.show()
    plt.savefig("Recons_momentum_"+name+".pdf")
    print("[amplitude, sigma, mu]")
    print(popt)
    print("Error for the same variables")
    print(pcov)

def resolution(momenta): #Compute the momentum resolution of each event(equation in the overleaf) and do the mean of it
    
    ampli = res_x/(0.3*q*B*L*larm_2)
    resMomenta = [ampli*i for i in momenta]
    mean_res = sum(resMomenta) 
    mean_res = mean_res/len(resMomenta)
    return mean_res



def plotRes(resMomenta,name): # Plot the momentum resolution                      
    plt.clf() #Close previous plot canvas
    plt.plot(resMomenta)
    plt.xlabel("Reconstructed energy [GeV/c]", fontsize = 16)
    plt.ylabel("Number of events", fontsize = 16)
    #    plt.show()                                                                                                                           
    plt.savefig("Recons_resol_"+name+".pdf")



for event in tree: #Loop over the events
    if len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5: #if there is at least 5 hits in each chamber (possible to do something with less hits in each chamber, however I didn't test it)
        m1,c1,position1 = trackReconstruction(event, 1) #Linear factors and positions of tracks in the first chamber
        m2,c2,position2 = trackReconstruction(event, 2) #Linear factors and positions of tracks in the second chamber 
        computeMom = determineMomentum(m1,c1,m2,c2) #Compute momentum
        momentum.append(abs(computeMom)) #Stock momentum value
plotTrack(m1,c1,position1,m2,c2,position2,name) #Plot the fitted tracks
plotMomenta(momentum,name) #Plot the momentum with a gaussian fit
resMom=resolution(momentum) #Compute the mean of the momentum resolution
print(resMom)
plotRes(resMom,name) #Plot the momentum resolution
