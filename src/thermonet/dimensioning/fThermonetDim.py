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

def ils(a, t, r):
    """
    Calculate the dimensionless line source model.
    
    This function calculates the dimensionless line source model based on the input parameters.
    
    Args:
        a (float): Soil thermal diffusivity (m^2/s).
        t (float): Time (s).
        r (float): Radius (m).
        
    Returns:
        float: The dimensionless line source model temperature (-).
    """
    return 1/4/mt.pi*f.exp1(r**2/4/a/t)


def Re(rho,dyn,v,d):
    # Computes the Reynolds number for a pipe (-)
    # rho : fluid density (kg/m3)
    # dyn : dynamic viscosity (Pa*s)
    # v : fluid velocity (m/s)
    # d : pipe diameter (m)
    return rho*v*d/dyn                     
    
def fD_beier(REN):
    # Computes the Darcy friction factor (-)
    # REN : Reynolds number (-)
    return 8*(1/((8/REN)**10 + (REN/36500)**20)**0.5 + (2.21*np.log(REN/7))**10)**(-1/5)
        
def dp(rho,mu,Q,Di):    
    # Computes the pressure loss per length of pipe (typically Pa/m)
    # rho : fluid density (kg/m3)
    # REN : Reynolds number (-)
    v = Q/mt.pi/(Di/2)**2                             # Flow velocity (m/s)
    K = rho/2*v**2                                    # Computational constants (Pa)
    REN = Re(rho,mu,v,Di)                             # Reynolds number (-)
    fD = fD_beier(REN)                                # Darcy friction factor (-)
    return fD*K/Di                                    # Pressure loss per meter pipe (Pa/m)

def Rp(Di,Do,REN,Pr,lb,lp):
    # Compute the combined conductive and convective thermal resistance of a
    # pipe with a fluid with a certain Reynolds number. Unit: m*K/W
    # For laminar flow equation see p. 70 in "Advances in GSHP systems".     
    # Di : pipe inner diameter (m)
    # Do : pipe outer diameter (m)
    # REN : Reynolds number (-)
    # Pr : Prandtl number (-)
    # lb : fluid thermal conductivity (W/m/K)
    # lp : pipe material thermal conductivity (W/m/K)
    
    #################### Compute heat transfer coefficients #############
    # Reference: 
    if REN > 2300:                                      # Test for laminar or transitional/turbulent flow regime
        fD = fD_beier(REN)                              # Darcy friction factor (-)
        h = lb*0.125*fD*(REN-1000)*Pr/(1+12.7*np.sqrt(0.125*fD)*(Pr**(2/3)-1))/Di       
    else:
        h = (3.657+4.364)/2*lb/Di                       # For laminar flow, the Nusselt number is constant and equal to 3.66 and 4.36 for uniform heat flux and convection with fixed temperature, respectively. Source: https://archive.org/details/fundamentalsheat00incr_617
    return 1/(2*mt.pi*0.5*Di*h) + 1/2/mt.pi/lp*np.log(Do/Di)    # The total resistance is the sum of the conductive and convective resistance
    
def CSM(r,r0,t,a):
    # Computes the Cylindrical Source Model (CSM) (-)   
    # Reference: INGERSOLL L. R. et al. (1954). Heat conduction with engineering, geological, and other applications. New York, McGraw-Hill
    # r : computation radius (m)
    # r0 : pipe outer radius (m)
    # t : time (s)
    # a : thermal diffusivity (m2/s)
    p = r/r0
    z = a*t/r**2
    fCSM = lambda b, p, z: ((np.exp(-b**2*z)-1)*(f.jv(0,p*b)*f.yn(1,b)-f.yn(0,p*b)*f.jv(1,b))/(b**2*(f.jv(1,b)**2+f.yn(1,b)**2)))
    
    if np.ndim(t) == 0:
        G = quad(fCSM, 0, np.inf, args=(p,z))[0]
        
    else:
        NT = len(t)
        G = np.zeros(NT)
        for i in range(NT):
            G[i] = quad(fCSM, 0, np.inf, args=(p,z[i]))[0]
    
    
    return G/mt.pi**2

def VFLS(x, y, H, a, U, t):
    rr = x * x + y * y
    
    # Allocate g-function
    NT = len(t)
    G = np.zeros(NT)
    UU = U * U
    aa = a * a

    def erfi(x):
        return x * f.erf(x) - (1 - np.exp(-x**2)) / np.sqrt(np.pi)
    
    # Define analytical solution
    def fun(s):    
        return np.exp(-UU / (16 * aa * s * s) - rr * s * s) * 2 * erfi(H * s) / (H * s * s)
    
    for i in range(0, NT):
        #print(t[i])
        G[i], _ = quad(fun, 1 / np.sqrt(4 * a * t[i]), np.inf)
    
    G = f.iv(0, x * U / (2 * a)) * G  # Modified Bessel function of the first kind
    return G/4/mt.pi

def RbMP(lb,lp,lg,lss,rb,rp,ri,s,RENBHE,Pr):
    # Multipole computation of the borehole thermal resistance (K*m/W)
    b = 2*mt.pi*lg*Rp(2*ri,2*rp,RENBHE,Pr,lb,lp)        # Eq. 3.47 Advances in GSHP systems
    C1 = s/2/rb                                         # Eq. 3.29 Advances in GSHP systems
    C2 = rb/rp                                          # Eq. 3.30 Advances in GSHP systems
    C3 = 1/2/C1/C2                                      # Eq. 3.31 Advances in GSHP systems
    si = (lg-lss)/(lg+lss)                              # Eq. 3.32 Advances in GSHP systems
    return 1/4/mt.pi/lg*(b + np.log(C2/2/C1/(1-C1**4)**si) - C3**2*(1-(4*si*C1**4)/(1-C1**4))**2/((1+b)/(1-b) + C3**2*(1+16*si*C1**4/(1-C1**4)**2)))    # Eq. 3.60 Advances in GSHP systems


def RbMPflc(lb,lp,lg,lss,rhob,cb,rb,rp,ri,LBHE,s,QBHE,RENBHE,Pr):
    # Multipole computation of the borehole thermal resistance considering flow and length effects
    b = 2*mt.pi*lg*Rp(2*ri,2*rp,RENBHE,Pr,lb,lp) 
    C1 = s/2/rb 
    C2 = rb/rp
    C3 = 1/2/C1/C2
    si = (lg-lss)/(lg+lss)
    Rb1 = RbMP(lb,lp,lg,lss,rb,rp,ri,s,RENBHE,Pr)
    Ra = 1/mt.pi/lg*(b + np.log((1+C1**2)**si/C3/(1-C1**2)**si) - C3**2*(1-C1**4 + 4*si*C1**2)**2/((1+b)/(1-b)*(1-C1**4)**2-C3**2*(1-C1**4)**2 + 8*si*C1**2*C3**2*(1+C1**4)))  # Eq. 3.62 Advances in GSHP systems
    dRb1 = 1/3/Ra*(LBHE/rhob/cb/QBHE)**2                    # Eq. 3.67 Advances in GSHP systems
    R1b = 2*Rb1                                             # Eq. 3.58 Advances in GSHP systems
    R12 = 2*Ra*R1b/(2*R1b - Ra)                             # Eq. 3.14 Advances in GSHP systems
    nu = (LBHE/rhob/cb/QBHE)*1/Rb1*np.sqrt(1+4*Rb1/R12)     # Eq. 3.69 Advances in GSHP systems
    Rb2 = Rb1 * nu * mpt.coth(nu)                           # Eq. 3.68 Advances in GSHP systems
    Rb1 = Rb1 + dRb1                                        # Use average of the two corrections as recommended in Advances in GSHP systems
    return 0.5*(Rb1+Rb2) 


def Halley(x,dx,f1,f2,f3):
    # Computes one iteration of Halleys rootfinding method
    # x: inital guess of x
    # dx: step in x to compute numerical derivatives
    # f1: f(x-dx)
    # f2: f(x)
    # f3: f(x+dx)
    
    # df: first derivative of f with respect to x
    # ddf: second derivative of f with respect to x
    # xn: updated estimate of x 
    df = (f3 - f1)/2/dx
    ddf = (f3 - 2*f2 + f1)/dx**2
    xn = x - 2*f2*df/(2*df**2-f2*ddf)
    return xn  