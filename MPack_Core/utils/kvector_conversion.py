#####   Obtain the angular averaged, equivalent k_CO for a given k_C[II]   #####

import numpy as np
from cosmology import *
from scipy.integrate import quad

def nu_CO(J):
    nu_CO_10 = 1.15271208e11   # in [Hz]
    return (J*(J+1) - (J-1)*J)/2 * nu_CO_10

def z_obs_CO(J,z_CII):
    temp = nu_CO(J)/nu_obs(z_CII) - 1.0
    temp = round(temp,2)
    return temp
    
def nu_obs(z_CII):
    lambda_CII_rest = 1.577e-2   # in [cm]
    temp = c_light/((1.0+z_CII)*lambda_CII_rest)
    return temp

def k_CO_equiv_avg(k_CII,z_CII,J): 
    ''' Calculate the mu_CII averaged k-vector for CO at a given k_CII '''
    z_CO = z_obs_CO(z_CII,J)
    k_CII_prl = lambda mu: mu * k_CII
    k_CII_prp = lambda mu: (1.-mu**2)**0.5 * k_CII
    k_CO_prl = lambda mu: (1.0+z_CII)/(1.0+z_CO) * HubbleParameter(z_CO)/HubbleParameter(z_CII) * k_CII_prl(mu)
    k_CO_prp = lambda mu: d_A_comov(z_CII)/d_A_comov(z_CO) * k_CII_prp(mu)
    
    #mu = np.sqrt(3.)/3.
    
    k_CO_mu = lambda mu: (k_CO_prl(mu)**2 + k_CO_prp(mu)**2)**0.5
    
    #temp = k_CO_mu(mu)
    
    temp = quad(k_CO_mu, 0., 1.)[0]

    return temp
k_CO_equiv_avg = np.vectorize(k_CO_equiv_avg)



