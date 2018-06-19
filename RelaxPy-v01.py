# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:03:51 2018

@author: C.J. Wilkinson & Y. Mauro
   Implementation of paper: Journal of American Ceramic Society,
   : 10.1111/jace.15272, 9/29/2017

   Calculation of the nonequailibrium viscosity

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from collections import defaultdict

import time
import os
import sys

def Tf_tot(Tfi, wi):  # equation (A.1)
	return np.sum(Tfi*wi) ### Composite Fictive Temperature

def ergodicity(T, Tf, p): # equation (2)
	return (min(T, Tf)/max(T, Tf))**p ### Ergodicity Calculation

def log10_eta_eq(Tf, eta_inf, Tg, m): # equation (3)
	return eta_inf+(12.0-eta_inf)*(Tg/Tf)*np.exp((m/(12.0-eta_inf)-1.0)*(Tg/Tf-1.0)) ### Equilibrium Viscosity

def log10_eta_ne(T, Tf, eta_inf, Tg, m, A, B, C): # equation (4)
	return A+B/T-C*np.exp(-(Tg/Tf)*(m/(12.0-eta_inf)-1.0)) ### Nonequailibrium Viscosity

def log10_viscosity(y, visc_eq, visc_ne): # equation (1)
	return y*visc_eq + (1-y)*visc_ne ### 'Real' Viscosity

def log10_tao_K(log10_eta,log10Ks): # equation (A.5)
	return np.log10((10**log10_eta)/(10**log10Ks)) ### Relaxation time

def sim(Temp, wiN, kiN, eta_inf, m, Tg, A, B, C, p, dt): ### Actual Calculation
	print ("Starting Simulation")
	print ("Using Following Parameters:")
	print ("	dt: "+str(dt))
	print ("	Fragility: "+str(m))
	print ("	Transition Temp: "+str(Tg))
	print ("	C [S/kln10]: "+str(C))
	print ("Using the Following Constants:")
	print ("	Eta_Inf: "+str(eta_inf))
	print ("	A: "+str(A)) #These ones are constants
	print ("	B: "+str(B))
	print ("	p: "+str(p)) ## Printing Parameters


	numT = Temp.size
	N = wiN.size
	log10Ks = np.log10(35*10**9)

	Tfi = np.zeros((N, numT)) ### Array Terms of prony x Number of data points
	visc = np.zeros((numT)) ### Viscosity for each Data point
	taoK = np.zeros((numT)) ### Tao K for each data point
	STf = np.zeros((numT))
	# initialization at time t = 0, where Tfi(0) = T_thermal[0, 1]

	Tfi[:,0] = np.repeat(Temp[0], N) # intitial value at t = 0 N times

	Tf0 = Tf_tot(Tfi[:,0], wiN) # Sum of Tf components x Weight of each component
	visc_eq0 = log10_eta_eq(Tf0, eta_inf, Tg, m) # Equalibrium Viscosity
	visc_ne0 = log10_eta_ne(Temp[0], Tf0, eta_inf, Tg, m, A, B, C) # none nonequailibrium viscocity
	ergo0 = ergodicity(Temp[0], Tf0, p) ### Ergodicity function
	visc[0] = log10_viscosity(ergo0, visc_eq0, visc_ne0) ### Real viscocity
	taoK[0] = log10_tao_K(visc[0],log10Ks) ### Tao K

	STf[0] = Tf0

	for i in range(1,numT): #### Loop over time
		Tf = Tf_tot(Tfi[:,i-1], wiN) ### Current Fictive is this
		STf[i] = Tf
		visc_eq = log10_eta_eq(Tf, eta_inf, Tg, m) ### Equilibrium
		visc_ne = log10_eta_ne(Temp[i], Tf, eta_inf, Tg, m, A, B, C) ### Non-Eqaul
		ergo = ergodicity(Temp[i], Tf, p) ### Ergodicity
		visc[i] = log10_viscosity(ergo, visc_eq, visc_ne) ### Real Viscocity
		taoK[i] = log10_tao_K(visc[i],log10Ks)     ### TaoK
		R = log10_tao_K(visc[i],log10Ks)
		for j in range(N):
			Tfi[j,i] = Temp[i] - ((Temp[i]-Tfi[j,i-1])*np.exp(kiN[j]*dt/10**taoK[i])) ### A.2
			# Tfi[j,i] = Temp[i] + 1 - ((Temp[i]-Tfi[j,i-1])*np.exp(dt*kiN[j]/(10**taoK[i]))) ### A.2
			# Tfi[j,i] = Temp[i] - ((Temp[i]-Temp[i-1])*10**taoK[i])/(kiN[j]*dt)
	        ###  Just trying to get around the expression always be 0
	return visc,taoK,STf,Tfi

if (len(sys.argv)!=3 and len(sys.argv)!=2 and len(sys.argv) != 4):
	print ("Error: Incorrect Use")
	print (" Correct Use is [ python Tf_Calc.py InputFile.txt Output.csv ]")
	print (" Correct Use is [ python Tf_Calc.py InputFile.txt] if no printing is wanted")
	print (" Correct Use is [ python Tf_Calc.py InputFile.txt -All] if no printing is wanted but all Tf")

	quit() ### Setting up command line args

input = ""
output = ""
flag = False

for x in range(1,len(sys.argv)): ### Command line args
	if sys.argv[x]!='-All' and sys.argv[x]!='-all':
		if input == "":
			input = sys.argv[x]
		else:
			output = sys.argv[x]
	else:
		flag = True

print ("")
print ("")

print("**************************************************")
print("        Fictive Temperature Calculator            ")
print("        By C.J. Wilkinson and Y. Mauro            ")
print("")
print("   See X. Guo et al. (2017) for details on model  ")
print("**************************************************")

print ("")

Ns,I,wi_3over7,ki_3over7,wi_1over2, ki_1over2,wi_3over5, ki_3over5 = np.loadtxt('fit_params_PronyToKWW.txt', skiprows=1, unpack=True) ### Load Beta Parameters
N_unique = np.unique(Ns)

eta_inf = -2.9
Tg = 734.5
m = 35.3
p = 10.88
A = 45.19
B = 4136.7
C= 135.09
Beta = '3/7'
N = 8

f = open(input,'r')
i=0
Temp = np.empty((0),float)
dt = 1
for line in f:
	if (len(line) != 0 and line[0]!='#'):
		J = line.split()
		if i==0:
			x = 0
			while x < len(J):
				if (J[x]=="eta_inf:"):
					x=x+1
					eta_inf = float(J[x])
				elif (J[x]=="Tg:"):
					x=x+1
					Tg = float(J[x])
				elif (J[x] == "m:"):
					x=x+1
					m = float(J[x])
					p = m*0.3082153
				elif (J[x]=="Beta:"):
					x=x+1
					Beta = J[x]
				elif (J[x] == "N:"):
					x=x+1
					N = int(J[x])
				elif (J[x] == "A:"):
					x=x+1
					A = float(J[x])
				elif (J[x] == "B (dH/kln10):"):
					x=x+1
					B = float(J[x])
				elif (J[x] == "C (dS/K*ln10):"):
					x=x+1
					C = float(J[x])
				elif (J[x] == "dt:"):
					x=x+1
					dt = float(J[x])
				elif (J[x] == "p:"):
					x=x+1
					p = float(J[x])
				x=x+1
		else:
			J = np.empty((3))
			if "," in line:
				J = line.rstrip().split(",")
			Temp = np.append(Temp,np.linspace(float(J[0])+273,float(J[1])+273,int(J[2])),axis=0)
		i=i+1 # SETTING UP Simulation
		
wiidx = np.where(Ns==N)
if Beta=='3/7':
	wiN = wi_3over7[wiidx]
	kiN = ki_3over7[wiidx]
elif Beta=='1/2':
	wiN = wi_1over2[wiidx]
	kiN = ki_1over2[wiidx]
elif Beta=='3/5':
	wiN = wi_3over5[wiidx]
	kiN = ki_3over5[wiidx]


t = np.linspace(0,Temp.size*dt,Temp.size)
visc, taoK, Tf, Tfc = sim(Temp,wiN,kiN,eta_inf,m,Tg,A,B,C,p,dt)

plt.figure(num=None, figsize=(20, 8), facecolor='w', edgecolor='k')

plt.subplot(2,2,1)
T, = plt.plot(t, Temp, linewidth=3.0,color="green",label='Thermal History')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')

plt.subplot(2,2,2)
V, = plt.plot(t, visc, linewidth=3.0, color="red", label = 'Viscocity')
plt.xlabel('Time [s]')
plt.ylabel(r'log10 [$\eta$'+' (Pa s)]')

plt.subplot(2,2,3)
K, = plt.plot(t, taoK, linewidth=3.0, color="purple", label = r'$\tau_k$')
plt.xlabel('Time [s]')
plt.ylabel(r'log10 [$\tau_k$'+' (s)]')

plt.subplot(2,2,4)
if flag:
	for i in range(0,Tfc.shape[0]):
		J, = plt.plot(t, Tfc[i,:], linewidth=wiN[i]*10, color="orange",label=r'T$_f$')
T, = plt.plot(t, Temp, linewidth=3.0,color="green",label='Glass')
S, = plt.plot(t, Tf, linewidth=3.0, color="blue",label=r'$\sum$W$_i$T$_f$')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')

if flag:
	leg = plt.legend([T,S,J],['Thermal History','Composite Fictive Temperature','Fictive Temperature Components'],loc=0)
else:
	leg = plt.legend([T,S],['Thermal History','Fictive Temperature'],loc=0)

for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)


if (output != ""):
	print ("")
	if flag:
		np.savetxt(output, np.vstack((t,Temp,Tf,taoK,visc,Tfc)).T , delimiter=",", header="Time,Temp,Fictive,TaoK,Viscocity,Tf Components")
	else:
		np.savetxt(output, np.vstack((t,Temp,Tf,taoK,visc)).T , delimiter=",", header="Time,Temp,Fictive,TaoK,Viscocity")
	print "File Saved!"

plt.show()

print ("")
print ("")



#####################################################
#####################################################
