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




def RunStacking(WR=True):

	# Get binning information from the corresponding parameter function
	binning_pf = master_pf.BinningParameterFile()

	z_nodes = binning_pf['z_nodes']
	nz = len(z_nodes) - 1
	m_nodes_sf = binning_pf['m_nodes_sf']
	m_nodes_qt = binning_pf['m_nodes_qt']
	nm_sf = len(m_nodes_sf) - 1
	nm_qt = len(m_nodes_qt) - 1


	# Get wavelength information from the corresponding parameter function
	wvl_pf = master_pf.WavelengthsParameterFile()

	wavelength = wvl_pf['wavelength']
	nwv = wvl_pf['nwv']
	indstack = wvl_pf['indstack']
	library_keys = wvl_pf['library_keys']
	fwhm = wvl_pf['fwhm']
	efwhm = wvl_pf['efwhm']
	color_correction = wvl_pf['color_correction']
	beam_area = wvl_pf['beam_area']


	# Get map information from the corresponding parameter function
	maps = []; noises = []; masks = []
	map_pf = master_pf.MapsParameterFile()
	for ii in range(len(wavelength)):
		maps.append(map_pf[str(wavelength[ii])+'_img'])
		noises.append(map_pf[str(wavelength[ii])+'_rms'])
		masks.append(map_pf[str(wavelength[ii])+'_msk'])

	psf_input = efwhm

	# Different libraries for Actual and Gaussian Approximated PSFs
	sky_library_gaussians=OrderedDict({})
	for t in indstack[0]:
		#sky_library_gaussians[library_keys[t]] = Skymaps(maps[t],noises[t],masks[t],efwhm[t],color_correction=color_correction[t])
		sky_library_gaussians[library_keys[t]] = Skymaps(maps[t],noises[t],masks[t],psf_input[t],color_correction=color_correction[t])
		sky_library_gaussians[library_keys[t]].add_wavelength(wavelength[t])
		sky_library_gaussians[library_keys[t]].add_fwhm(efwhm[t])
		if beam_area[t] != 1: sky_library_gaussians[library_keys[t]].beam_area_correction(beam_area[t])

	# Get catalog information from the corresponding parameter function
	catalog_pf = master_pf.CatalogsParameterFile()
	catalog_to_use = catalog_pf[master_pf.SetFields()]



	tbl = pd.read_table(catalog_to_use,sep=',')
	if 'lmass' in tbl:
		tbl['LMASS']=tbl['lmass']
	elif 'mass' in tbl:
		tbl['LMASS']=tbl['mass']

	CATALOG = Field_catalogs(tbl)

	# This function creates a column 'sfg' based on its UVJ colors, where sf == 1 and qt == 0
	CATALOG.separate_sf_qt()


	# Stack in redshift bins, with Layers divided by Stellar Mass and Star-Forming/Quiescent Galaxies
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
	
	
	if WR == True:
		# Write stacked mean fluxes to files
		current_path = os.path.dirname(__file__)
	
		for iwv in range(nwv):

			sf_fit = all_stacked_fluxes_sf[iwv,:,:].reshape((nz,nm_sf)) * 1.0E3
			qt_fit = all_stacked_fluxes_qt[iwv,:,:].reshape((nz,nm_qt)) * 1.0E3

			np.savetxt(current_path + '/' + '../results/mean_flux_matrices/flux_matrix_sf_%d.dat' % wavelength[indstack[0][iwv]], 
						   sf_fit, fmt='%.6f', delimiter='\t')
			np.savetxt(current_path + '/' + '../results/mean_flux_matrices/flux_matrix_qt_%d.dat' % wavelength[indstack[0][iwv]],
						   qt_fit, fmt='%.6f', delimiter='\t')
	else:
		pass
	
	print '\n----- Stacking Done -----\n'
	
	return 0














