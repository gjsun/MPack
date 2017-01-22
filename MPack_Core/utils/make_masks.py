import pdb
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from parameter_file import *


# def CreateMask(wavelength):
	



# RMS from eyeballing
# 24um: 
# 70um: 0.0018
# 250um: 0.0017
# 350um: 0.0023
# 450um: 4.1
# 500um: 0.0040
# 850um: 0.8

wavelength = [24,100,160,250,350,500,850,1100]
# Specify wavelengths at which we want to make masks
wv0 = np.array([1,1,1,1,1,1,1,1])

for (ii,jj) in enumerate(wv0):
	if jj==1:
		
		print '\nCreating mask for %dum...\n' % wavelength[ii]
		
		map_pf = UDSMapsParameterFile()
		map_name = map_pf[str(wavelength[ii])+'_img']
		noise_name = map_pf[str(wavelength[ii])+'_rms']
		mask_name = map_pf[str(wavelength[ii])+'_msk']
		#print [map_name, noise_name, mask_name]
		
		
		cmap, chd = fits.getdata(map_name, 0, header = True)
		nmap, nhd = fits.getdata(noise_name, 0, header = True)
		cms = np.shape(cmap)
		
		rms = np.median(nmap[cms[0]/4:cms[0]*3/4, cms[1]/4:cms[1]*3/4].flatten())
		#print 'rms is %.8f' % rms

		new_map = np.ones_like(cmap)
		#new_map = np.zeros_like(cmap)

		new_map[np.where((nmap>3.*rms) + np.isnan(nmap) + (nmap==0))] = 0
		#new_map[cms[0]/2-cms[0]/4:cms[0]/2+cms[0]/4, cms[1]/2-cms[1]/4:cms[1]/2+cms[1]/4] = 1

		# Filter isolated ones
		for i in range(cms[0]):
			row_mean = (new_map[i,0:-2]+new_map[i,1:-1]+new_map[i,2::])/3.
			ind_0s = np.where(row_mean<0.5)[0] + 1
			new_map[i,ind_0s] = 0
		for j in range(cms[1]):
			col_mean = (new_map[0:-2,j]+new_map[1:-1,j]+new_map[2::,j])/3.
			ind_0s = np.where(col_mean<0.5)[0] + 1
			new_map[ind_0s,j] = 0

		fits.writeto(mask_name, new_map, nhd, clobber=True)
		print 'Created the mask of %dum map to .fits file.\n' % wavelength[ii]

print '----- DONE -----'


#for i in enumerate(wv0):
#	UDSMapsParameterFile()['%s' % str(wavelength)]

'''
map_path = '/Users/guochaosun/Desktop/Caltech_OBSCOS/DataCollection/simstack_maps/uds_maps_to_use/'
mask_path = map_path + 'masks/'

pixsize_suffix = '4.0_arcsec_pixels'

map_name = map_path+'sxdf_aste_kscott20100924_map.fits'
noise_name = map_path+'sxdf_aste_kscott20100924_map_noise.fits'
mask_name = mask_path+'uds_aztec_mask.fits'



cmap, chd = fits.getdata(map_name, 0, header = True)
nmap, nhd = fits.getdata(noise_name, 0, header = True)
cms = np.shape(cmap)

rms = np.median(nmap[cms[0]/4:cms[0]*3/4, cms[1]/4:cms[1]*3/4].flatten())
print 'rms is %.8f' % rms

new_map = np.ones_like(cmap)
#new_map = np.zeros_like(cmap)

new_map[np.where((nmap>3.*rms) + np.isnan(nmap) + (nmap==0))] = 0
#new_map[cms[0]/2-cms[0]/4:cms[0]/2+cms[0]/4, cms[1]/2-cms[1]/4:cms[1]/2+cms[1]/4] = 1

# Filter isolated ones
for i in range(cms[0]):
    row_mean = (new_map[i,0:-2]+new_map[i,1:-1]+new_map[i,2::])/3.
    ind_0s = np.where(row_mean<0.5)[0] + 1
    new_map[i,ind_0s] = 0
for j in range(cms[1]):
    col_mean = (new_map[0:-2,j]+new_map[1:-1,j]+new_map[2::,j])/3.
    ind_0s = np.where(col_mean<0.5)[0] + 1
    new_map[ind_0s,j] = 0


fits.writeto(mask_name, new_map, nhd, clobber=True)
print '\nDONE!\n'

'''