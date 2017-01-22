import pdb
import numpy as np
import pandas as pd
import sys, os
import math
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from collections import OrderedDict

from ..utils import parameter_file as master_pf
from ..simstack.simstack_modified import stack_libraries_in_redshift_slices
from ..simstack.skymaps_sfqt import Skymaps, Field_catalogs





binning_pf = master_pf.BinningParameterFile()

z_nodes = binning_pf['z_nodes']
nz = len(z_nodes) - 1
m_nodes_sf = binning_pf['m_nodes_sf']
m_nodes_qt = binning_pf['m_nodes_qt']
nm_sf = len(m_nodes_sf) - 1
nm_qt = len(m_nodes_qt) - 1


# Decide which Maps to include
wv0 = np.array([1,1,1,1,1,1,1,1])
indstack = np.where(wv0 == 1)




## Map Directories 

## Dictionary Names
library_keys =['mips24'
               ,'pacs_green'
               ,'pacs_red'
               ,'spire_PSW'
               ,'spire_PMW'
               ,'spire_PLW'
               ,'scuba850'
               ,'aztec'
              ]

wavelength=[24,100,160,250,350,500,850,1100]
nwv = np.sum(wv0) 
fwhm =[6.32, 7.4, 11.3, 18.1, 25.2, 36.6, 15., 18.]
efwhm=[6.32, 6.7, 11.2, 17.6, 23.9, 35.2, 14.5, 18.] # want to the measured effective FWHM later
#color_correction=[1.25,1.01,23.58,23.82,1.018,0.9914,0.95615,1.0]
color_correction=[1.,1.,1.,1.,1.,1.,1.0E-3,1.]
#beam_area = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.] #sr
beam_area = [1.547E-9,1.,1.,1.,1.,1.,1.,1.] #sr

maps = []
noises = []
masks = []

for (ii,jj) in enumerate(wv0):
	map_pf = master_pf.MapsParameterFile()
	maps.append(map_pf[str(wavelength[ii])+'_img'])
	noises.append(map_pf[str(wavelength[ii])+'_rms'])
	masks.append(map_pf[str(wavelength[ii])+'_msk'])




psf_input = [6.32, 
			 6.7, 
			 11.2, 
			 17.6, 
			 23.9, 
			 35.2, 
			 14.5, 
			 18.]

print psf_input


#Different libraries for Actual and Gaussian Approximated PSFs
sky_library_gaussians=OrderedDict({})
for t in indstack[0]:
    #sky_library_gaussians[library_keys[t]] = Skymaps(maps[t],noises[t],masks[t],efwhm[t],color_correction=color_correction[t])
    sky_library_gaussians[library_keys[t]] = Skymaps(maps[t],noises[t],masks[t],psf_input[t],color_correction=color_correction[t])
    sky_library_gaussians[library_keys[t]].add_wavelength(wavelength[t])
    sky_library_gaussians[library_keys[t]].add_fwhm(efwhm[t])
    if beam_area[t] != 1: sky_library_gaussians[library_keys[t]].beam_area_correction(beam_area[t])


catalog_pf = master_pf.CatalogsParameterFile()
catalog_to_use = catalog_pf[master_pf.GetFields()]



tbl = pd.read_table(catalog_to_use,sep=',')
if 'lmass' in tbl:
	tbl['LMASS']=tbl['lmass']
elif 'mass' in tbl:
	tbl['LMASS']=tbl['mass']

CATALOG = Field_catalogs(tbl)

# This function creates a column 'sfg' based on its UVJ colors, where sf == 1 and qt == 0
CATALOG.separate_sf_qt()


#Stack in redshift bins, with Layers divided by Stellar Mass and Star-Forming/Quiescent Galaxies
pop = ['sf','qt']
npop = len(pop)


CATALOG.get_sf_qt_mass_redshift_bins(z_nodes,m_nodes_sf,m_nodes_qt)
radec_m_z_p = CATALOG.subset_positions(CATALOG.id_z_ms)



stacked_fluxes_psfs =  None
n_sources_max = None

stacked_fluxes_psfs = stack_libraries_in_redshift_slices(sky_library_gaussians,radec_m_z_p)



all_stacked_fluxes_sf = np.zeros([nwv,nz,nm_sf])
all_stacked_fluxes_qt = np.zeros([nwv,nz,nm_qt])

args = radec_m_z_p.keys()
for iwv in range(nwv):
	stacked_fluxes_wv = stacked_fluxes_psfs[str(wavelength[indstack[0][iwv]])]
	for iz in range(nz):
		for jm1 in range(nm_sf):
			arg = 'z_'+str(z_nodes[iz])+'-'+str(z_nodes[iz+1])+\
				  '__m_'+str(m_nodes_sf[jm1])+'-'+str(m_nodes_sf[jm1+1])+'_sf'
			all_stacked_fluxes_sf[iwv,iz,jm1] = stacked_fluxes_wv[arg.replace('.','p').replace('-','_')].value
		for jm2 in range(nm_qt):
			arg = 'z_'+str(z_nodes[iz])+'-'+str(z_nodes[iz+1])+\
				  '__m_'+str(m_nodes_qt[jm2])+'-'+str(m_nodes_qt[jm2+1])+'_qt'
			all_stacked_fluxes_qt[iwv,iz,jm2] = stacked_fluxes_wv[arg.replace('.','p').replace('-','_')].value


def WriteFluxesToFiles():
	
	current_path = os.path.dirname(__file__)
	
	for iwv in range(nwv):

		sf_fit = all_stacked_fluxes_sf[iwv,:,:].reshape((nz,nm_sf)) * 1.0E3
		qt_fit = all_stacked_fluxes_qt[iwv,:,:].reshape((nz,nm_qt)) * 1.0E3

		np.savetxt(current_path + '/' + '../results/mean_flux_matrices/flux_matrix_sf_%d.dat' % wavelength[indstack[0][iwv]], 
					   sf_fit, fmt='%.6f', delimiter='\t')
		np.savetxt(current_path + '/' + '../results/mean_flux_matrices/flux_matrix_qt_%d.dat' % wavelength[indstack[0][iwv]],
					   qt_fit, fmt='%.6f', delimiter='\t')
	
	return 0














