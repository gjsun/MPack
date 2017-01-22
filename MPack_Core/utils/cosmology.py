import numpy as np
import matplotlib.pyplot as plt
import hmf
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad



# ========== All constants are in cgs units ========== #
### Cosmology ###
c_light = 2.99792458*1.0e10   # in [cm/s]
km_per_pc = 3.08568e13
km_per_mpc = km_per_pc*1e6
Mpc_to_cm = 3.08567758e24
Lsun_to_cgs = 3.828e33
Jansky_to_cgs = 1.0e-23
Omega_m_0 = 0.27
Omega_l_0 = 0.73
hubble_0 = 0.71
H_0 = hubble_0 * 100 / km_per_mpc
MpcPerh_to_cm = 3.086e24/hubble_0



def EvolutionFunction(z):
    return np.sqrt(Omega_m_0 * (1.0 + z)**3  + Omega_l_0)

def HubbleParameter(z):
    # Hubble parameter in [s^-1]
    return H_0 * EvolutionFunction(z)

def T_CMB(z):
    return 2.725 * (1.+z)

# Luminosity Distance. An analytic fit from Adachi & Kasai (2012)
def d_L(Z_INPUT):

    def x(z):
        return (1.0-Omega_m_0)/Omega_m_0 / (1.0+z)**3

    def Phi(x):
        return (1.0+1.320*x+0.4415*x**2+0.02656*x**3) / (1.0+1.392*x+0.5121*x**2+0.03944*x**3)

    temp = Phi(x(0))-1.0/np.sqrt(1.0+Z_INPUT)*Phi(x(Z_INPUT))

    temp = 2*c_light*(1.0+Z_INPUT)/H_0/np.sqrt(Omega_m_0) * temp / Mpc_to_cm

    return temp

# Comoving Angular Diameter Distance
def d_A_comov(z):
    return d_L(z)/(1.+z)

# Halo mass function
def dNdlog10M(M_in, z_in):
    hmf_sample = hmf.MassFunction(Mmin=8.0,Mmax=13.0,hmf_model='ST',z=z_in)
    xs = hmf_sample.M
    ys = hmf_sample.dndlog10m
    sm_func = InterpolatedUnivariateSpline(xs, ys, k=2)
    return sm_func(M_in)

def halo_bias(M_in, z_in):
    # Get the smooth function from interpolation
    hmf_sample = hmf.MassFunction(Mmin=8.0,Mmax=13.0,hmf_model='ST',z=z_in)
    _p = 0.3
    _q = 0.75
    xs = hmf_sample.M
    # Here, the Sheth-Tormen mass function is assumed
    ys = 1. + (_q * hmf_sample.nu - 1.) / hmf_sample.delta_c + \
         (2. * _p / hmf_sample.delta_c) / (1. + (_q * hmf_sample.nu) ** _p)
    sm_func = InterpolatedUnivariateSpline(xs, ys, k=2)
    
    return sm_func(M_in)