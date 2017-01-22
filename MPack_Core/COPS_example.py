import numpy as np
import matplotlib.pyplot as plt
import hmf
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import itertools


from utils.cosmology import *
from utils.kvector_conversion import *

# ========== Specify the redshift of C[II] signal here ========== #
z_CII = 6.5
print 'Computing CO power spectra at redshift z = %.1f...' % z_CII


# Define stellar mass function. For z<0.2, the 0.2-0.5 values work fine (see Muzzin et al. Fig.6) so we extend 
# the same values to z=0. 
def SMF(M, z, gal_type):
    
    logM = np.log10(M)
    
    z_bins = np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0])
    ind = np.where((z_bins[0:-1]-z)*(z_bins[1::]-z)<=0)[0][0]
    
    if gal_type == 'sf':
        logMstar = np.array([10.81,10.78,10.76,10.85,10.80,11.06,11.56])
        alpha = np.array([-1.34,-1.26,-1.21,-1.16,-0.53,-1.03,-1.51])
        alpha2 = np.array([0.,0.,0.,0.,0.,0.,0.])
        phi = np.array([11.35,12.71,8.87,5.68,3.72,1.39,0.06])*1.0E-4
        phi2 = np.array([0.,0.,0.,0.,0.,0.,0.])*1.0E-4
        
    elif gal_type == 'qt':
        logMstar = np.array([10.92,10.84,10.73,10.67,10.87,10.80,10.85])
        alpha = np.array([-0.38,-0.36,-0.17,0.03,-0.71,-0.39,0.46])
        alpha2 = np.array([-1.52,-2.32,0.,0.,0.,0.,0.])
        phi = np.array([19.68,14.55,7.48,3.61,1.14,0.66,0.05])*1.0E-4
        phi2 = np.array([0.58,0.005,0.,0.,0.,0.,0.])*1.0E-4
    
    temp = np.log(10)*phi[ind]*10.**((logM-logMstar[ind])*(1.+alpha[ind])) * np.exp(-10.**(logM-logMstar[ind]))
    temp += np.log(10)*phi2[ind]*10.**((logM-logMstar[ind])*(1.+alpha2[ind])) * np.exp(-10.**(logM-logMstar[ind]))
    
    return temp
SMF = np.vectorize(SMF)


# ================================================================================================ #
# ===================================== C[II] Power Spectrum ===================================== #
# ================================================================================================ #
    
lambda_CII_rf = 1.577E-2
nu_CII_rf = c_light/lambda_CII_rf
h_pl = 6.626E-27
k_B = 1.381E-16

### C[II] Line Parameters ###
g_u = 4.
g_l = 2.
A_ul = 2.36E-6   # in [s^-1]
T_star_ul = 91.2
T_star_lu = 91.2

# =================   Gong's Model   ================= #

# =====   Part 1: C[II] Mean Intensity   ===== #

def gamma_lu(T_k):
    # Effective collision strength
    if np.round(np.log10(T_k),0) == 2:
        temp = 1.58
    elif np.round(np.log10(T_k),0) == 3:
        temp = 1.60
    elif np.round(np.log10(T_k),0) == 4:
        temp = 2.11
    else:
        raise ValueError('Input electron kinetic temperature must be between 100 and 10000.')
    return temp

def C_lu(T_k):
    return 8.629E-6/g_l/np.sqrt(T_k) * gamma_lu(T_k) * np.exp(-T_star_lu/T_k)

def C_ul(T_k):
    temp = g_u/g_l * np.exp(-T_star_ul/T_k)
    temp = C_lu(T_k)/temp
    return temp
    
def I_nu(z, nu=nu_CII_rf):
    # Intensity of CMB at the given frequency => for resonant scattering
    _T = T_CMB(z)
    temp = 2.*h_pl*nu**3/c_light**2 
    temp /= np.exp(h_pl*nu/k_B/_T) - 1.
    return temp

def TSpin_ul(T_k, n_e, z):
    temp = A_ul * (1.+I_nu(z)*c_light**2/2./h_pl/nu_CII_rf**3) + n_e*C_ul(T_k)
    temp /= A_ul * I_nu(z)*c_light**2/2./h_pl/nu_CII_rf**3 + n_e*C_ul(T_k)*np.exp(-T_star_ul/T_k)
    temp = np.log(temp)
    temp = T_star_ul/temp
    return temp
TSpin_ul = np.vectorize(TSpin_ul)


def n_b_mean(z):
    # Mean baryon density, in [cm^-3]
    return 2.0E-7 * (1.+z)**3

def n_gas_mean(z):
    _f_gas_hot = 0.3
    _f_gas_cr = 0.1
    return _f_gas_hot * _f_gas_cr * n_b_mean(z)

def f_CII(z):
    # Global metallicity at z
    return 10.**(-0.4*z)

def n_CII(z):
    # Number density of CII ions at z
    _Z_C = 3.7E-4
    return f_CII(z) * _Z_C * n_gas_mean(z)

def I_nu_CII(T_k, n_e, z):
    _f_grd_CII = 0.333   # Good for ISM only
    temp = g_u/g_l*A_ul * _f_grd_CII * n_CII(z)
    temp *= h_pl*c_light/4./np.pi/HubbleParameter(z)/(1.+z)**3
    temp *= np.exp(-T_star_ul/TSpin_ul(T_k, n_e, z))
    temp *= (1. - (np.exp(T_star_ul/TSpin_ul(T_k, n_e, z))-1.)/(2.*h_pl*nu_CII_rf**3/c_light**2/I_nu(z)))
    temp *= 1.0E23   # Convert cgs units to Jansky!
    return temp
I_nu_CII = np.vectorize(I_nu_CII)

# =====   Part 2: C[II] Power Spectrum   ===== #

def M_Z(M,z):
    # Mass in metal as a function of halo mass and redshift
    if z==6.0:
        _M0 = 1.1E9
        _Mc = 3.5E11
        _b = 3.6
        _d = 3.25
    elif z==6.5:
        _M0 = 1.05E9
        _Mc = 3.5E11
        _b = 3.5
        _d = 3.175
    elif z==7.0:
        _M0 = 1.0E9
        _Mc = 3.5E11
        _b = 3.4
        _d = 3.1
    elif z==8.0:
        _M0 = 1.6E9
        _Mc = 3.7E11
        _b = 3.4
        _d = 3.6
    else:
        raise ValueError('%.2f is out of bounds!' % z)
    
    return _M0 * (M/_Mc)**_b * (1.+M/_Mc)**-_d

def M_CII(M,z):
    # C[II] mass 
    return 0.21 * M_Z(M,z)

def bias_CII_mean(z_in):
    # Calculate the average bias for C[II]
    integrand_num = lambda log10M: dNdlog10M(10**log10M, z_in) * M_CII(10**log10M, z_in) * halo_bias(10**log10M, z_in)
    integrand_dem = lambda log10M: dNdlog10M(10**log10M, z_in) * M_CII(10**log10M, z_in)
    xpts = np.linspace(8.0,13.0,100)
    temp = np.sum((integrand_num(xpts)[0:-1] + integrand_num(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))
    temp /= np.sum((integrand_dem(xpts)[0:-1] + integrand_dem(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))
    return temp

def P_shot(z_in):
    integrand = lambda log10M: dNdlog10M(10**log10M,z_in) * \
                               (M_Z(10**log10M,z_in)*100.63*(Lsun_to_cgs/Jansky_to_cgs/MpcPerh_to_cm**2)\
                               *(lambda_CII_rf/MpcPerh_to_cm)/4./np.pi/HubbleParameter(z_in))**2
    xpts = np.linspace(8.0,13.0,1000)
    temp = np.sum((integrand(xpts)[0:-1] + integrand(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))
    return temp

def delta_CII_shot(k,z):
    return P_shot(z) * k**3 / (2.*np.pi**2)

k_CII_list = hmf.Transfer(z=z_CII).k
Pl_CII_list = hmf.Transfer(z=z_CII).delta_k   # linear dark matter power spectrum

P_CII_clust = I_nu_CII(1.0E3, 1.0E2, z_CII)**2 * bias_CII_mean(z_CII)**2 * Pl_CII_list
P_CII_shot = delta_CII_shot(k_CII_list,z_CII)



# ============================================================================================= #
# ===================================== CO Power Spectrum ===================================== #
# ============================================================================================= #


def nu_CO(J):
    nu_CO_10 = 1.15271208e11   # in [Hz]
    return (J*(J+1) - (J-1)*J)/2 * nu_CO_10

def nu_obs(z_CII):
    lambda_CII_rest = 1.577e-2   # in [cm]
    temp = c_light/((1.0+z_CII)*lambda_CII_rest)
    return temp

def z_obs_CO(J,z_CII):
    temp = nu_CO(J)/nu_obs(z_CII) - 1.0
    temp = round(temp,2)
    return temp


def k_CO_equiv(k_CII,z_CII,J): 
    temp = 2*(d_A_comov(z_CII)/d_A_comov(z_obs_CO(z_CII,J)))**2 /3
    #temp += (1.0+z_CII)**2/(1.0+z_obs_CO(z_CII,J))**2 \
    temp += (1.0+z_CII)**1/(1.0+z_obs_CO(z_CII,J))**1 \
                * HubbleParameter(z_obs_CO(z_CII,J))**2/HubbleParameter(z_CII)**2 /3
    temp = k_CII * np.sqrt(temp)
    return temp
k_CO_equiv = np.vectorize(k_CO_equiv)




def logLIR_logM_z(logM, z, gal_type):
    # Takes two 1D arrays and returns a 1D array
    _logM = logM
    _z = z
    if gal_type == 'sf':
        #y_matrix = np.array([[-7.248,3.160,-0.137],[-1.634,0.335,-0.009],[-7.758,1.374,-0.062]])
        y_matrix = np.array(
                    [[ -4.12823189e+01,   9.51368029e+00,  -4.35397951e-01],
                     [  8.72098081e+00,  -1.73766938e+00,   9.47591623e-02],
                     [ -1.52804792e+00,   2.36956587e-01,  -1.02684100e-02]] 
                            )
        M_vector = np.array([_logM**0,_logM**1,_logM**2])
        z_vector = np.array([_z**0,_z**1,_z**2]).T
    elif gal_type == 'qt':
        y_matrix = np.array([[2.672,0.624],[1.430,-0.056]])
        M_vector = np.array([_logM**0,_logM**1])
        z_vector = np.array([_z**0,_z**1]).T
    x_vector = np.dot(y_matrix,M_vector)
    _logL = np.dot(z_vector,x_vector)
    
    return _logL
logLIR_logM_z = np.vectorize(logLIR_logM_z)



def logLIR_logM_z_mono(logM, z, gal_type):
    # We extrapolate the 2nd order poly fit linearly beyond logM_crit = 11.0 at any given z, 
    # where the evolution with mass is poorly constrained due to our binning scheme. 
    logM_c = 11.0
    if logM <= logM_c:
        return logLIR_logM_z(logM,z,gal_type)
    else:
        return logLIR_logM_z(logM_c,z,gal_type) + (logM - logM_c) * \
               (logLIR_logM_z(logM_c,z,gal_type) - logLIR_logM_z(logM_c-0.1,z,gal_type)) / 0.1
logLIR_logM_z_mono = np.vectorize(logLIR_logM_z_mono)



def L_M_z(M, z, rand, sigma=0.35, gal_type=None):
    
    _logM = np.log10(M)
    _z = z
    _logL = logLIR_logM_z_mono(_logM, _z, gal_type)       
    
    if rand == None:
        return 10**_logL
    else:
        return 10**(_logL + rand.normal(0.,sigma))
L_M_z = np.vectorize(L_M_z)



def LIR_to_LCO(LIR, J, rand, sigma=0.3, fit_type='CW'):
    
    # 'CW' for Carilli & Walter (2013)
    if fit_type == 'CW':
        a1 = 1.27
        b1 = 0.73
        log10_LCOp1 = a1 + b1*np.log10(LIR)
        r = np.zeros(10)
        r[0:6] = np.array([1.0,0.85,0.66,0.46,0.39,0.39])
        ans = 10.**(log10_LCOp1+rand.normal(0.,sigma)) * r[J-1] * 3.0E-11 * (nu_CO(J)/1.0E9)**3
    
    return ans


def LIR_to_LCO_MEAN(LIR, J, sigma_t=0.5):
    """ Return LCO at given LIR in [Lsun] """
    mu = 1.27 + 0.73 * np.log10(LIR)
    r = np.zeros(10)
    r[0:6] = np.array([1.0,0.85,0.66,0.46,0.39,0.39])
    prob = lambda x: 1./sigma_t/np.sqrt(2.*np.pi) * np.exp(-(x-mu)**2/2./sigma_t**2)
    integrand = lambda x: 10**x * prob(x) * r[J-1] * 3.0E-11 * (nu_CO(J)/1.0E9)**3
    temp = quad(integrand, -20., 20.)
    return temp[0]
LIR_to_LCO_MEAN = np.vectorize(LIR_to_LCO_MEAN)


def LIR_to_LCO_MEAN_CU(LIR, J, sigma_t=0.5):
    """ Return LCO in common unit [K km s^-1 pc^2] at given LIR in [Lsun] """
    mu = 1.27 + 0.73 * np.log10(LIR)
    r = np.zeros(10)
    r[0:6] = np.array([1.0,0.85,0.66,0.46,0.39,0.39])
    prob = lambda x: 1./sigma_t/np.sqrt(2.*np.pi) * np.exp(-(x-mu)**2/2./sigma_t**2)
    integrand = lambda x: 10**x * prob(x) * r[J-1]
    temp = quad(integrand, -20., 20.)
    return temp[0]
LIR_to_LCO_MEAN_CU = np.vectorize(LIR_to_LCO_MEAN_CU)



class COPowerSpectrum(object):
    
    def __init__(self, sigma_t, pop='sf', rand_IR=None, rand_CO=None):
        self.sigma_t = sigma_t
        self.pop = pop
        if (rand_IR != None) and (rand_CO != None): 
            self.rand1 = rand_IR
            self.rand2 = rand_CO
    
    
    #####   Calculation of Power Spectrum   #####

    def b_CO(self,z_in):

        integrand_num = lambda log10M: 10**log10M * dNdlog10M(10**log10M,z_in) * halo_bias(10**log10M,z_in)
        integrand_dem = lambda log10M: 10**log10M * dNdlog10M(10**log10M,z_in)

        xpts = np.linspace(9.0,13.0,100)
        temp = np.sum((integrand_num(xpts)[0:-1] + integrand_num(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))
        temp /= np.sum((integrand_dem(xpts)[0:-1] + integrand_dem(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))

        return temp



    def ICO(self,z_in,J,log10M_max=13.0):
        if hasattr(self, 'rand_IR') and hasattr(self, 'rand_CO'):
            integrand = lambda log10M: SMF(10**log10M,z_in,self.pop) * \
                                       LIR_to_LCO(L_M_z(10**log10M,z_in,self.rand1,gal_type=self.pop), J, self.rand2)*(Lsun_to_cgs/Jansky_to_cgs/MpcPerh_to_cm**2)\
                                       *(c_light/nu_CO(J)/MpcPerh_to_cm)/4./np.pi/HubbleParameter(z_in)
        else:
            integrand = lambda log10M: SMF(10**log10M,z_in,self.pop) * \
                                       LIR_to_LCO_MEAN(L_M_z(10**log10M,z_in,None,gal_type=self.pop), J, sigma_t=self.sigma_t)*(Lsun_to_cgs/Jansky_to_cgs/MpcPerh_to_cm**2)\
                                       *(c_light/nu_CO(J)/MpcPerh_to_cm)/4./np.pi/HubbleParameter(z_in)

        xpts = np.linspace(8.0,log10M_max,1000)
        temp = np.sum((integrand(xpts)[0:-1] + integrand(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))
        return temp

    def PCO_shot(self,z_in,J,log10M_max=13.0):
        if hasattr(self, 'rand_IR') and hasattr(self, 'rand_CO'):
            integrand = lambda log10M: SMF(10**log10M,z_in,self.pop) * \
                                       (LIR_to_LCO(L_M_z(10**log10M,z_in,self.rand1,gal_type=self.pop), J, self.rand2)*(Lsun_to_cgs/Jansky_to_cgs/MpcPerh_to_cm**2)\
                                       *(c_light/nu_CO(J)/MpcPerh_to_cm)/4./np.pi/HubbleParameter(z_in))**2
        else:
            integrand = lambda log10M: SMF(10**log10M,z_in,self.pop) * \
                                       (LIR_to_LCO_MEAN(L_M_z(10**log10M,z_in,None,gal_type=self.pop), J, sigma_t=self.sigma_t)*(Lsun_to_cgs/Jansky_to_cgs/MpcPerh_to_cm**2)\
                                       *(c_light/nu_CO(J)/MpcPerh_to_cm)/4./np.pi/HubbleParameter(z_in))**2

        xpts = np.linspace(8.0,log10M_max,1000)
        temp = np.sum((integrand(xpts)[0:-1] + integrand(xpts)[1::])/2. * abs(xpts[1]-xpts[0]))
        return temp
    
    
    def PCO_clust(self,k,z_in,J,log10M_max=13.0):
        kf_list_interp = hmf.Transfer(z=z_in).k
        Pl_list_interp = hmf.Transfer(z=z_in).power

        smooth_ps = InterpolatedUnivariateSpline(kf_list_interp, Pl_list_interp, k=2)
        
        Pl_list = smooth_ps(k)
        PCO_clust_list = self.ICO(z_in,J,log10M_max=log10M_max)**2 * self.b_CO(z_in)**2 * Pl_list
        return PCO_clust_list
        
        
        
    
    def delta_CO_shot(self,k,z,J,log10M_max=13.0):
        return self.PCO_shot(z,J,log10M_max=log10M_max) * k**3 / (2.*np.pi**2)
    
    # ---------- Dimensionless Power Spectrum ---------- #
    def PCO_dl_obs(self,J,log10M_max=13.0):

        z_CO = z_obs_CO(J,z_CII)

        ks_list = hmf.Transfer(z=z_CII).k
        kf_list = k_CO_equiv(ks_list,z_CII,J)

        kf_list_interp = hmf.Transfer(z=z_CO).k
        Pl_list_interp = hmf.Transfer(z=z_CO).power

        smooth_ps = InterpolatedUnivariateSpline(kf_list_interp, Pl_list_interp, k=2)

        Pl_list = smooth_ps(kf_list)
        PCO_clust_list = self.ICO(z_CO,J,log10M_max=log10M_max)**2 * self.b_CO(z_CO)**2 * Pl_list
        PCO_shot_list = self.PCO_shot(z_CO,J,log10M_max=log10M_max)
        PCO_obs_list = (PCO_clust_list+PCO_shot_list) * \
                       (d_A_comov(z_CII)/d_A_comov(z_CO))**2 * \
                       (1.+z_CII)**2/(1.+z_CO)**2 * HubbleParameter(z_CO)/HubbleParameter(z_CII)

        PCO_dl_obs_list = PCO_obs_list * ks_list**3 / (2.*np.pi**2)

        return [ks_list, PCO_dl_obs_list]
    
    # ---------- kP(k) Power Spectrum ---------- #
    def PCO_kPk_obs(self,J,log10M_max=13.0):

        z_CO = z_obs_CO(J,z_CII)

        ks_list = hmf.Transfer(z=z_CII).k
        kf_list = k_CO_equiv(ks_list,z_CII,J)

        kf_list_interp = hmf.Transfer(z=z_CO).k
        Pl_list_interp = hmf.Transfer(z=z_CO).power

        smooth_ps = InterpolatedUnivariateSpline(kf_list_interp, Pl_list_interp, k=2)

        Pl_list = smooth_ps(kf_list)
        PCO_clust_list = self.ICO(z_CO,J,log10M_max=log10M_max)**2 * self.b_CO(z_CO)**2 * Pl_list
        PCO_shot_list = self.PCO_shot(z_CO,J,log10M_max=log10M_max)
        PCO_obs_list = (PCO_clust_list+PCO_shot_list) * \
                       (d_A_comov(z_CII)/d_A_comov(z_CO))**2 * \
                       (1.+z_CII)**2/(1.+z_CO)**2 * HubbleParameter(z_CO)/HubbleParameter(z_CII)

        PCO_kPk_obs_list = PCO_obs_list * ks_list

        return [ks_list, PCO_kPk_obs_list]
        
        
        
        
# ==================================================================================== #
# ===================================== PLOTTING ===================================== #
# ==================================================================================== # 
        
# This part of code is lengthy as I wanted to make everything clear then. Need to be simplified later. 
        
# ------------------   LEFT: UNMASKED   ------------------ #

PS_generated_32_sf = [COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=3)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=3)[1]
                      ]

PS_generated_32_qt = [COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=3)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=3)[1]
                      ]

PS_generated_43_sf = [COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=4)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=4)[1]
                      ]

PS_generated_43_qt = [COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=4)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=4)[1]
                      ]

PS_generated_54_sf = [COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=5)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=5)[1]
                      ]

PS_generated_54_qt = [COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=5)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=5)[1]
                      ]

PS_generated_65_sf = [COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=6)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=6)[1]
                      ]

PS_generated_65_qt = [COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=6)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=6)[1]
                      ]

# ------------------   RIGHT: MASKING LEVELS   ------------------ #

PS_generated_IL_sf = [COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=3,log10M_max=9.0)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=3,log10M_max=9.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=4,log10M_max=9.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=5,log10M_max=9.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=6,log10M_max=9.0)[1]
                      ]
PS_generated_IL_qt = [COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=3,log10M_max=9.0)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=3,log10M_max=9.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=4,log10M_max=9.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=5,log10M_max=9.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=6,log10M_max=9.0)[1]
                      ]
PS_generated_IH_sf = [COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=3,log10M_max=10.0)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=3,log10M_max=10.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=4,log10M_max=10.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=5,log10M_max=10.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='sf').PCO_kPk_obs(J=6,log10M_max=10.0)[1]
                      ]
PS_generated_IH_qt = [COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=3,log10M_max=10.0)[0],
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=3,log10M_max=10.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=4,log10M_max=10.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=5,log10M_max=10.0)[1] + \
                      COPowerSpectrum(sigma_t=0.5, pop='qt').PCO_kPk_obs(J=6,log10M_max=10.0)[1]
                      ]


PS_generated_IIL_sf = [COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=3,log10M_max=9.0)[0],
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=3,log10M_max=9.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=4,log10M_max=9.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=5,log10M_max=9.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=6,log10M_max=9.0)[1]
                       ]
PS_generated_IIL_qt = [COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=3,log10M_max=9.0)[0],
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=3,log10M_max=9.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=4,log10M_max=9.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=5,log10M_max=9.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=6,log10M_max=9.0)[1]
                       ]
PS_generated_IIH_sf = [COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=3,log10M_max=10.0)[0],
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=3,log10M_max=10.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=4,log10M_max=10.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=5,log10M_max=10.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='sf').PCO_kPk_obs(J=6,log10M_max=10.0)[1]
                       ]
PS_generated_IIH_qt = [COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=3,log10M_max=10.0)[0],
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=3,log10M_max=10.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=4,log10M_max=10.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=5,log10M_max=10.0)[1] + \
                       COPowerSpectrum(sigma_t=0.3, pop='qt').PCO_kPk_obs(J=6,log10M_max=10.0)[1]
                       ]


PS_generated_IIIL_sf = [COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=3,log10M_max=9.0)[0],
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=3,log10M_max=9.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=4,log10M_max=9.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=5,log10M_max=9.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=6,log10M_max=9.0)[1]
                        ]

PS_generated_IIIL_qt = [COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=3,log10M_max=9.0)[0],
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=3,log10M_max=9.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=4,log10M_max=9.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=5,log10M_max=9.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=6,log10M_max=9.0)[1]
                        ]
PS_generated_IIIH_sf = [COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=3,log10M_max=10.0)[0],
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=3,log10M_max=10.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=4,log10M_max=10.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=5,log10M_max=10.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='sf').PCO_kPk_obs(J=6,log10M_max=10.0)[1]
                        ]
PS_generated_IIIH_qt = [COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=3,log10M_max=10.0)[0],
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=3,log10M_max=10.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=4,log10M_max=10.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=5,log10M_max=10.0)[1] + \
                        COPowerSpectrum(sigma_t=0.1, pop='qt').PCO_kPk_obs(J=6,log10M_max=10.0)[1]
                        ]
                        

xs_t3, ys_t3a = PS_generated_32_sf
xs_t3, ys_t3b = PS_generated_32_qt
xs_t4, ys_t4a = PS_generated_43_sf
xs_t4, ys_t4b = PS_generated_43_qt
xs_t5, ys_t5a = PS_generated_54_sf
xs_t5, ys_t5b = PS_generated_54_qt
xs_t6, ys_t6a = PS_generated_65_sf
xs_t6, ys_t6b = PS_generated_65_qt

xs0, ys0 = PS_generated_IL_sf
xs1, ys1 = PS_generated_IL_qt
xs2, ys2 = PS_generated_IH_sf
xs3, ys3 = PS_generated_IH_qt

xs4, ys4 = PS_generated_IIL_sf
xs5, ys5 = PS_generated_IIL_qt
xs6, ys6 = PS_generated_IIH_sf
xs7, ys7 = PS_generated_IIH_qt

xs8, ys8 = PS_generated_IIIL_sf
xs9, ys9 = PS_generated_IIIL_qt
xs10, ys10 = PS_generated_IIIH_sf
xs11, ys11 = PS_generated_IIIH_qt




# Read in Gong's C[II] model
x_CII_Gong = np.array(k_CII_list)
y_CII_Gong = (np.array(P_CII_clust) + np.array(P_CII_shot)) / x_CII_Gong**2 * (2.*np.pi**2)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12,6), dpi=200)
plt.subplots_adjust(wspace=0.05,hspace=0.)

lco, = ax[0].plot(xs_t6, ys_t3a+ys_t3b+ys_t4a+ys_t4b+ys_t5a+ys_t5b+ys_t6a+ys_t6b, 'k-', lw=2, 
                  label=r'$\mathrm{CO_{tot}},\ \sigma_{\mathrm{tot}}=0.5$')
lco3, = ax[0].plot(xs_t3, ys_t3a+ys_t3b, 'k-', alpha=0.5, label=r'$z_{\mathrm{\ CO(3-2)}}=%.2f$'%z_obs_CO(3,z_CII))
lco4, = ax[0].plot(xs_t4, ys_t4a+ys_t4b, 'k--', alpha=0.5, label=r'$z_{\mathrm{\ CO(4-3)}}=%.2f$'%z_obs_CO(4,z_CII))
lco5, = ax[0].plot(xs_t5, ys_t5a+ys_t5b, 'k-.', alpha=0.5, label=r'$z_{\mathrm{\ CO(5-4)}}=%.2f$'%z_obs_CO(5,z_CII))
lco6, = ax[0].plot(xs_t6, ys_t6a+ys_t6b, 'k:', alpha=0.5, label=r'$z_{\mathrm{\ CO(6-5)}}=%.2f$'%z_obs_CO(6,z_CII))

with open('./Silva_Results/silva_cii_m2.csv') as file:
    data_Silva = np.asarray([[float(digit) for digit in line.split(',')] for line in itertools.islice(file, 1, None)])

x_SilvaL_CII_z6p5 = data_Silva[:,0]
y_SilvaL_CII_z6p5 = data_Silva[:,1]
y_SilvaL_CII_z6p5 = y_SilvaL_CII_z6p5 / x_SilvaL_CII_z6p5**2 * 2.*np.pi**2
lcii_silva, = ax[0].plot(x_SilvaL_CII_z6p5,y_SilvaL_CII_z6p5,color='y',ls='None',marker='+',ms=8,mew=2,lw=2,alpha=0.8,
                         label=r'$\mathrm{C[II]}$, Silva+2015')

lcii_gong, = ax[0].plot(x_CII_Gong[0::5],y_CII_Gong[0::5],ls='None',marker='+',color='blue',ms=8,mew=2,lw=2,alpha=0.8,
                        label=r'$\mathrm{C[II]}$, Gong+2012')

x_SilvaL_CO_z6p5 = np.array([0.05,0.07,0.10,0.14,0.19,0.27,0.37,0.48,0.62,0.82,1.12,1.49,1.99])
y_SilvaL_CO_z6p5 = np.array([9955.,16541.,37873.,82204.,173718.,367110.,840552.,
                            1683858.,3558420.,8368171.,21321615.,48818902.,117914368.])
y_SilvaL_CO_z6p5 = y_SilvaL_CO_z6p5 / x_SilvaL_CO_z6p5**2 * 2.*np.pi**2
lco_silva_sim, = ax[0].plot(x_SilvaL_CO_z6p5,y_SilvaL_CO_z6p5,'-',color='darkorchid',lw=2,alpha=0.8,
                            label=r'$\mathrm{CO_{sim}}$, Silva+2015')

x_SilvaH_CO_z6p5 = np.array([0.050,0.066,0.086,0.115,0.159,0.211,0.286,0.392,0.526,0.708,0.927,1.234,1.610,2.00])
y_SilvaH_CO_z6p5 = np.array([329894.,562967.,1013450.,1639457.,2873518.,5036486.,8827574.,16321721.,31834743.,
                             63773781.,124387735.,262862846.,555495891.,1027082704.])
y_SilvaH_CO_z6p5 = y_SilvaH_CO_z6p5 / x_SilvaH_CO_z6p5**2 * 2.*np.pi**2
lco_silva_obs, = ax[0].plot(x_SilvaH_CO_z6p5,y_SilvaH_CO_z6p5,'-',color='steelblue',lw=2,alpha=0.8,
                            label=r'$\mathrm{CO_{obs}}$, Silva+2015')



with open('./YunTing_Results/P_k_z6_j3.dat') as file:
    yt_j3 = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 1, None)])
with open('./YunTing_Results/P_k_z6_j4.dat') as file:
    yt_j4 = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 1, None)])
with open('./YunTing_Results/P_k_z6_j5.dat') as file:
    yt_j5 = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 1, None)])
with open('./YunTing_Results/P_k_z6_j6.dat') as file:
    yt_j6 = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 1, None)])


yt_xs = yt_j3[:,0]
yt_ys = yt_xs * (yt_j3[:,1]+yt_j4[:,1]+yt_j5[:,1]+yt_j6[:,1])

from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_weight('bold')
font.set_style('italic')
ax[0].text(1.2E-2,2.0E5,'z = %.1f\nunmasked'%z_CII,fontsize=15,fontproperties=font)

ax[1].plot(xs0, ys0+ys1, 'r-', lw=2)
ax[1].plot(xs2, ys2+ys3, 'k-', lw=2, label=r'$\sigma_{\mathrm{tot}}=0.5$')
ax[1].plot(xs4, ys4+ys5, 'r--', lw=2)
ax[1].plot(xs6, ys6+ys7, 'k--', lw=2, label=r'$\sigma_{\mathrm{tot}}=0.3$')
ax[1].plot(xs8, ys8+ys9, 'r-.', lw=2)
ax[1].plot(xs10, ys10+ys11, 'k-.', lw=2, label=r'$\sigma_{\mathrm{tot}}=0.1$')

ax[1].plot(x_CII_Gong[0::5],y_CII_Gong[0::5],ls='None',marker='+',color='blue',ms=8,mew=2,lw=2,alpha=0.8)
ax[1].plot(0.1*np.ones(10),np.logspace(5.,11.,10),'k:',alpha=0.8)
ax[1].plot(x_SilvaL_CII_z6p5,y_SilvaL_CII_z6p5,color='y',ls='None',marker='+',ms=8,mew=2,lw=2,alpha=0.8)

ax[1].text(2.0E-1, 2.5E10, '$\log M_\mathrm{{*,thres}}=10.0$',color='black',fontsize=15)
ax[1].text(2.0E-1, 1.0E10, '$\log M_\mathrm{{*,thres}}=9.0$',color='red',fontsize=15)
ax[1].text(1.2E-2,2.0E5,'z = %.1f\nmasked'%z_CII,fontsize=15,fontproperties=font)

for i in range(2):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_xlim([0.9E-2, 1.1E1])
    ax[i].set_ylim([1.0E5, 1.0E11])
    ax[i].tick_params(axis='both', which='major', labelsize=12)
    ax[i].tick_params(axis='both', which='minor', labelsize=12)
    if i==0:
        lgd0 = ax[i].legend(handles=[lco,lco3,lco4,lco5,lco6],loc='lower right',frameon=False,numpoints=1,ncol=1,prop={'size':14})
        ax[i].add_artist(lgd0)
        lgd = ax[i].legend(handles=[lcii_silva,lcii_gong,lco_silva_sim,lco_silva_obs],loc='upper left',
                           frameon=False,numpoints=1,ncol=1,prop={'size':11})
        plt.setp(lgd.get_texts()[0], color = 'y')
        plt.setp(lgd.get_texts()[1], color = 'blue')
        plt.setp(lgd.get_texts()[2], color = 'darkorchid')
        plt.setp(lgd.get_texts()[3], color = 'steelblue')
    else:
        ax[i].legend(loc='upper left',frameon=False,numpoints=1,ncol=1,prop={'size':14})
    ax[i].set_xlabel('$k_{\mathrm{C[II]}}\ [h/\mathrm{Mpc}]$', fontsize=18)
    #ax[0].set_ylabel('$\Delta^{2}(k)\ \mathrm{[(Jy/sr)^2]}$', fontsize=18)
    ax[0].set_ylabel('$kP(k)\ [(\mathrm{Jy/sr})^2(\mathrm{Mpc}/h)^2]$', fontsize=18)
    
plt.savefig('./mean_co_ps.png',dpi=200)
#plt.show()