# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:49:20 2022

@author: SOEB
"""
import math as mt;
import scipy.special as f;
import numpy as np;
import mpmath as mpt;
from scipy.integrate import quad

def ils(a,t,r):
    return 1/4/mt.pi*f.exp1(r**2/4/a/t);

def ps(PT,COP):
    # Computes ground load from total building load and COP
    #   pt = total building thermal load (W);
    #   COP = Coefficient of Performance (-);
    return (COP-1)*PT/COP;

def Re(rho,dyn,v,d):
    #Re Calculates Reynolds number
    return rho*v*d/dyn;
    
def fD_beier(REN):
    #Beregner Darcys friktionsfaktor til beregning af tryktab
    return 8*(1/((8/REN)**10 + (REN/36500)**20)**0.5 + (2.21*np.log(REN/7))**10)**(-1/5);
    # NB! Beier ser ud til at bruge Fanning friction factor, Darcy er 4 gange
    # større!!
    
def dp(rho,mu,Q,Di):
# Computes the pressure loss per length of pipe (typically Pa/m)
# Update with reference to Beier
    v = Q/mt.pi/(Di/2)**2;                            #Flow velocity (m/s)
    K = rho/2*v**2;                                   #Computational constants (Pa)
    REN = Re(rho,mu,v,Di);                            #Reynolds number (-)
    fD = fD_beier(REN);                               #Darcy friction factor (-)
    return fD*K/Di;                                   #Pressure loss per meter pipe (Pa/m)

def Rp(Di,Do,REN,Pr,lb,lp):
    # Compute the combined conductive and convective thermal resistance of a
    # pipe with a flowing fluid with a certain Reynolds number ´. Unit: m*K/W
    #   For laminar flow equation see p. 70 in "Advances in GSHP systems".     
    #################### Compute heat transfer coefficients #############
    if REN>2300:                                # Test for transitional or turbulent flow regime
        fD = fD_beier(REN);                     # Darcy friction factor
        h = lb*0.125*fD*(REN-1000)*Pr/(1+12.7*np.sqrt(0.125*fD)*(Pr**(2/3)-1))/Di;      
    else:
        h = (3.657+4.364)/2*lb/Di;          #For laminar flow, the Nusselt number is constant and equal to 3.66 and 4.36 for uniform heat flux and convection with fixed temperature, respectively. Source: https://archive.org/details/fundamentalsheat00incr_617
    return 1/(2*mt.pi*0.5*Di*h) + 1/2/mt.pi/lp*np.log(Do/Di); #The total resistance is the sum of the conductive and convective resistance
    
def CSM(r,r0,t,a):
    fCSM = lambda u, r, r0, t, a: ((mt.exp(-a*u**2*t)-1)*(f.jv(0,u*r)*f.yn(1,u*r0)-f.yn(0,u*r)*f.jv(1,u*r0))/(u**2*(f.jv(1,u*r0)**2+f.yn(1,u*r0)**2)));
    NT = len(t);
    G = np.zeros(NT);
    for i in range(NT):
        G[i] = quad(fCSM, 0, np.Inf, args=(r,r0,t[i],a))[0]
    return G/mt.pi**2/r0

def RbMP(lb,lp,lg,lss,rb,rp,ri,s,RENBHE,Pr):
    # Multipole computation of the borehole thermal resistance
    b = 2*mt.pi*lg*Rp(2*ri,2*rp,RENBHE,Pr,lb,lp); 
    C1 = s/2/rb; 
    C2 = rb/rp;
    C3 = 1/2/C1/C2;
    si = (lg-lss)/(lg+lss);
    return 1/4/mt.pi/lg*(b + np.log(C2/2/C1/(1-C1**4)**si) - C3**2*(1-(4*si*C1**4)/(1-C1**4))**2/((1+b)/(1-b) + C3**2*(1+16*si*C1**4/(1-C1**4)**2)));

def GCLS(Fo):
    # Computes the approximate composite cylinder source G-function
    return 10.**(-0.89129+0.36081*mt.log10(Fo)-0.05508*mt.log10(Fo)**2+0.00359617*mt.log10(Fo)**3);

def RbMPflc(lb,lp,lg,lss,rhob,cb,rb,rp,ri,LBHE,s,QBHE,RENBHE,Pr):
    # Multipole computation of the borehole thermal resistance considering flow and length effects
    b = 2*mt.pi*lg*Rp(2*ri,2*rp,RENBHE,Pr,lb,lp); 
    C1 = s/2/rb; 
    C2 = rb/rp;
    C3 = 1/2/C1/C2;
    si = (lg-lss)/(lg+lss);
    Rb1 = RbMP(lb,lp,lg,lss,rb,rp,ri,s,RENBHE,Pr)
    Ra = 1/mt.pi/lg*(b + np.log((1+C1**2)**si/C3/(1-C1**2)**si) - C3**2*(1-C1**4 + 4*si*C1**2)**2/((1+b)/(1-b)*(1-C1**4)**2-C3**2*(1-C1**4)**2 + 8*si*C1**2*C3**2*(1+C1**4))); #Eq. 3.62 Advances in GSHP systems
    dRb1 = 1/3/Ra*(LBHE/rhob/cb/QBHE)**2;               #Eq. 3.67 Advances in GSHP systems
    R1b = 2*Rb1;                                        #Eq. 3.58 Advances in GSHP systems
    R12 = 2*Ra*R1b/(2*R1b - Ra);                        #Eq. 3.14 Advances in GSHP systems
    nu = (LBHE/rhob/cb/QBHE)*1/Rb1*np.sqrt(1+4*Rb1/R12);  #Eq. 3.69 Advances in GSHP systems
    Rb2 = Rb1 * nu * mpt.coth(nu);                           #Eq. 3.68 Advances in GSHP systems
    Rb1 = Rb1 + dRb1;                                    #Use average of the two corrections
    return 0.5*(Rb1+Rb2);