# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 08:41:53 2021
Python implementering af 
Johan Claesson: "Radial Heat Flow for a Pipe in a Borehole in Ground Using Laplace Solutions. Report on Mathematical Background" 
Report 2011:4, Appendix A4.1 Laplace solution for Tf(t) for a pipe in borehole in ground
@author: KART
"""

import numpy as np
from scipy.special import jv, yn
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pythermonet.core.fThermonetDim import ep,ils

qinj = 10
kb = 1.5
rhocb = 1550*2000
k = 3.0
rhoc = 2500*750

rp = 0.02*np.sqrt(2)
rb = 0.055
ab = kb/rhocb
a = k/rhoc
Cp = np.pi*rp**2*4.18e6

dpw = 0.0023
kp = 0.42
hp = 725
Rp = 1/(2*np.pi*kp)*np.log(rp/(rp-dpw)) + 1/(2*rp*hp)

t0 = 3600
print('CP: ' + str(Cp))
print('Rp: ' + str(Rp))
print('ab: ' + str(ab))
print('a: ' + str(a))

tau_p = rp/np.sqrt(ab*t0)
tau_b = rb/np.sqrt(ab*t0)
tau_g = rb/np.sqrt(a*t0)

Kbt = lambda u: 4*kb / (jv(0,tau_p*u)*yn(0,tau_b*u) - yn(0,tau_p*u)*jv(0,tau_b*u))

Kbp = lambda u: 4*kb* (0.5*np.pi*tau_p*u*(jv(1,tau_p*u)*yn(0,tau_b*u) - yn(1,tau_p*u)*jv(0,tau_b*u))-1) / (jv(0,tau_p*u)*yn(0,tau_b*u) - yn(0,tau_p*u)*jv(0,tau_b*u))

Kbb = lambda u: 4*kb* (0.5*np.pi*tau_b*u*(jv(1,tau_b*u)*yn(0,tau_p*u) - yn(1,tau_b*u)*jv(0,tau_p*u))-1) / (jv(0,tau_p*u)*yn(0,tau_b*u) - yn(0,tau_p*u)*jv(0,tau_b*u))

Kbg = lambda u: 2*np.pi*k*tau_g*u *(jv(1,tau_g*u)-1j*yn(1,tau_g*u)) / (jv(0,tau_g*u)-1j*yn(0,tau_g*u))

Lu = lambda u: np.imag(-1 / (Cp*(-u**2/t0) + 1/(Rp + 1/(Kbp(u) + 1/( 1/Kbt(u) + 1/(Kbb(u)+Kbg(u)))))))

integrand = lambda t, u: (1-np.exp(-u**2*t/t0))/u *Lu(u)
Tf = lambda t: 2/np.pi * quad(lambda u: integrand(t,u), 0, np.inf)[0]

tv = 4*3600
T1 = Tf(tv);
print(qinj*T1)
print(qinj*(ils(a,tv,rb)/k+1/2/np.pi/kb*np.log(rb/rp)+Rp))

#Tff = ep(kb,rhocb,k,rhoc,0.02,rb,dpw,kp,Rp,tv);
#print(qinj*Tff)

#Tfff = bp(k,rhoc,0.02,dpw,kp,Rp,tv);
#print(Tfff)
    

# # Analytisk afledte
# der_integrand = lambda t, u: (np.exp(-u**2*t/t0)) * (u/t0) * Lu(u)
# dTfdt = lambda t: 2/np.pi * quad(lambda u: der_integrand(t,u), 0, np.inf)[0]

# # Compare Mathcad implementation - OK
# print('Compare fluid temperatures from Mathcad implementation:')
# print([Tf(0), Tf(60), Tf(600), Tf(3600), Tf(10*3600), Tf(100*3600), Tf(1000*3600)])

# # Plot
# t = np.linspace(1e-6, 1e3, 100)
# Tfl = np.nan*np.ones(len(t))
# Tfl10 = np.nan*np.ones(len(t))
# Tfl100 = np.nan*np.ones(len(t))
# der_Tfl = np.nan*np.ones(len(t))
# for i in range(len(t)):
#     Tfl[i] = Tf(t[i])
#     Tfl10[i] = Tf(10*t[i])
#     Tfl100[i] = Tf(100*t[i])
#     der_Tfl[i] = dTfdt(t[i])
    
# plt.figure()
# plt.plot(t,Tfl,'g')
# plt.plot(t,Tfl10,'b')
# plt.plot(t,Tfl100,'r')


# # Sammenlign numerisk afledte
# dTdt_num = np.gradient(Tfl,t)

# plt.figure()
# plt.plot(t, der_Tfl, '*')
# plt.plot(t, dTdt_num,'--')
# plt.title('Afledte af temperatur')
# plt.legend(('Analytisk','Numerisk'))