import numpy as np
import os as os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, interp1d

# Get C[II] redshift
from COPS_example import z_CII, z_obs_CO

# Get the path of the master catalog file
_HOME = os.environ.get('HOME')

rand = np.random.RandomState(42)

np.set_printoptions(precision=4,suppress=True)








###############     Mass Estimator (from R. Quadri)     ###############
# Data points for interpolation
zmeans = [0.200, 0.400, 0.600, 0.800, 1.050, 1.350, 1.650, 2.000, 2.400, 2.900, 3.500]
intercepts = [18.2842,18.9785,19.2706,19.1569,20.5633,21.5504,19.6128,19.8258,19.8795,23.1529,22.1678]
slopes = [-0.454737,-0.457170,-0.454706,-0.439577,-0.489793,-0.520825,-0.436967,-0.447071,-0.443592,-0.558047,-0.510875]
slopes_cols = [0.0661783,-0.0105074,0.00262891,0.140916,0.0321968,0.0601271,0.470524,0.570098,0.455855,0.0234542,0.0162301]

intercepts_b = [18.3347,18.9626,19.2789,19.6839,20.7085,21.8991,22.9160,24.1886,22.6673,23.1514,21.6482]
slopes_b = [-0.456550,-0.456620,-0.455029,-0.460626,-0.495505,-0.534706,-0.570496,-0.617651,-0.543646,-0.556633,-0.487324]

data_list = [zmeans, intercepts, slopes, slopes_cols, intercepts_b, slopes_b]


# Interpolate the data points, and force the extrapolation to asymptote the mean of data points. 
def intercept_full(z, flag='color_magnitude'):
    dist = np.maximum((z - 0.2)*(z - 3.5), 0)
    # Interpolating data
    if flag == 'color_magnitude':
        sm = InterpolatedUnivariateSpline(zmeans, intercepts, k=3)
    elif flag == 'magnitude_only':
        sm = InterpolatedUnivariateSpline(zmeans, intercepts_b, k=3)
    else:
        raise ValueError('Invalid flag name!')
    # Forcing extrapolation to asymptote
    ans = sm(z) * np.exp(-dist) + np.mean(intercepts) * (1.-np.exp(-dist))
    return ans

def slope_full(z, flag='color_magnitude'):
    dist = np.maximum((z - 0.2)*(z - 3.5), 0)
    if flag == 'color_magnitude':
        sm = InterpolatedUnivariateSpline(zmeans, slopes, k=3)
    elif flag == 'magnitude_only':
        sm = InterpolatedUnivariateSpline(zmeans, slopes_b, k=3)
    else:
        raise ValueError('Invalid flag name!')
    ans = sm(z) * np.exp(-dist) + np.mean(slopes) * (1.-np.exp(-dist))
    return ans

def slope_col_full(z, flag='color_magnitude'):
    dist = np.maximum((z - 0.2)*(z - 3.5), 0)
    if flag == 'color_magnitude':
        sm = InterpolatedUnivariateSpline(zmeans, slopes_cols, k=3)
        ans = sm(z) * np.exp(-dist) + np.mean(slopes_cols) * (1.-np.exp(-dist))
        return ans
    elif flag == 'magnitude_only':
        return 0
    else:
        raise ValueError('Invalid flag name!')

func_list = [intercept_full, slope_full, slope_col_full]


# Define the mass estimation function
def Mass(K, JK, z):
    """ Return the mass estimate for the given K-band magnitude, J-K color and redshift """
    if (JK < 0.) or (JK > 5.):
        flag = 'magnitude_only'
    else:
        flag = 'color_magnitude'
    model = slope_full(z, flag) * K + intercept_full(z, flag) + slope_col_full(z, flag) * JK
    return model
Mass = np.vectorize(Mass)

# Flux to magnitude conversion adopted by the UltraVISTA catalog
def FluxToMagnitude(flux, ap_corr):
    return 25.0 - 2.5*np.log10(flux*ap_corr)
    




####################   Extended Bootstrapping Technique (EBT)   ####################
# --- EBT assembles new bins for stacking, rather than drawing from original bins
# Step 1: draw simulated redshifts from the photometric redshift probability distribution
# Step 2: estimate the mass using the perturbed redshift and observed K magnitude and J-K color
# Step 3: a simulated catalog is split up into (original) bins and calculate new stacked flux densities
# Step 4: repeat Step 1-3 many (>1000) times to complete the "bootstrapping"

n_bt = 1000   # Number of bootstrapping

path_cat = '/Desktop/Caltech_OBSCOS/DataCollection/simstack_catalogs/UVISTA/DR2/UVISTA_DR2_master_v2.1_USE.csv'
path_cat = _HOME + path_cat

col_to_read = ['ra','dec','z_peak','l68','u68','J','Ks','ap_corr','lmass','rf_U_V','rf_V_J']

df_cat_in = pd.read_csv(path_cat,usecols=col_to_read)
header_list = list(df_cat_in.columns.values)

cat_in = df_cat_in.as_matrix(columns=df_cat_in.columns)
n_sources = cat_in.shape[0]; n_params = cat_in.shape[1]
#print 'Size Read-In: ', cat_in.shape

c_z_peak = header_list.index('z_peak')
c_z_l68 = header_list.index('l68')
c_z_u68 = header_list.index('u68')
c_J = header_list.index('J')
c_Ks = header_list.index('Ks')
c_ap_corr = header_list.index('ap_corr')
c_lmass = header_list.index('lmass')



###     Redshift of C[II] Signal     ###
z_CII = 6.5

z_CO_32 = z_obs_CO(3,z_CII)
z_CO_43 = z_obs_CO(4,z_CII)


#if z_CII == 6.0:
#	z_CO_32 = 0.27
#if z_CII == 6.5:
#	z_CO_32 = 0.36
#elif z_CII == 7.0:
#	z_CO_32 = 0.46


inds3 = np.where( (cat_in[:,c_z_peak]>=z_CO_32-0.01) & (cat_in[:,c_z_peak]<=z_CO_32+0.01) )[0]

#print '----- Ks-logM relation is estimated at z in [%.2f, %.2f] with %d galaxies -----' % (z_CO_32-0.01, z_CO_32+0.01, np.size(inds3))

xpts3 = cat_in[inds3,c_lmass]
ypts3 = FluxToMagnitude(cat_in[inds3,c_Ks], cat_in[inds3,c_ap_corr])

fit_coeff3 = np.polyfit(xpts3,ypts3,1)

def fit_Ks_logM_3(logM):
    return fit_coeff3[1] + fit_coeff3[0] * logM
    

def fit_logM_Ks_3(Ks):
	return (Ks - fit_coeff3[1]) / fit_coeff3[0]
    
#print 'J32: ', fit_Ks_logM_3(9.0)
#print 'J32: ', fit_logM_Ks_3(22.0)

#xss = np.linspace(8.,11.,100)
#plt.plot(xpts3, ypts3, 'b+')
#plt.plot(xss, fit_Ks_logM_3(xss), 'r-')
#plt.show()





#if z_CII == 6.0:
#	z_CO_43 = 0.70
#if z_CII == 6.5:
#	z_CO_43 = 0.82
#elif z_CII == 7.0:
#	z_CO_43 = 0.94

inds4 = np.where( (cat_in[:,c_z_peak]>=z_CO_43-0.01) & (cat_in[:,c_z_peak]<=z_CO_43+0.01) )[0]

#print '----- Ks-logM relation is estimated at z in [%.2f, %.2f] with %d galaxies -----' % (z_CO_43-0.01, z_CO_43+0.01, np.size(inds4))

xpts4 = cat_in[inds4,c_lmass]
ypts4 = FluxToMagnitude(cat_in[inds4,c_Ks], cat_in[inds4,c_ap_corr])



fit_coeff4 = np.polyfit(xpts4,ypts4,1)

def fit_Ks_logM_4(logM):
    return fit_coeff4[1] + fit_coeff4[0] * logM
    

def fit_logM_Ks_4(Ks):
	return (Ks - fit_coeff4[1]) / fit_coeff4[0]
	
	
#print 'J43: ', fit_Ks_logM_4(9.0)