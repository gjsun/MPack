import numpy as np
import sys, os


def SetFields():
	
	Fields = ['uds']
	#Fields = ['uds','uvsita_dr2']
	
	if len(Fields)==1:
		# Stacking in a single field
		field_to_stack = Fields[0]
	else:
		# Stacking in multiple fields
		raise ValueError('--- Cannot stack on multiple fields yet ---')
		
	return field_to_stack




def MapsParameterFile():
	
	current_path = os.path.dirname(__file__)
	
	field_to_stack = SetFields()
	
	map_path = current_path + '/../data/maps/%s_maps/' % field_to_stack
	mask_path = map_path + 'masks/'
	pixsize_suffix = '4.0_arcsec_pixels'
	
	# Pixel sizes in arcsec (UDS/UVISTA): 
	# 24um: 1.2/1.2
	# 100um: 2.0/1.2
	# 160um: 3.0/2.4
	# 250um: 4.0/6.0
	# 350um: 4.0/6.0
	# 500um: 4.0/6.0
	# 850um: 2.0/1.0
	# 1100um: 3.0/6.0
	
	pf = {
		  '24_img': map_path + 'mips_mosaic.fits'
		 ,'24_rms': map_path + 'mips_mosaic_unc.fits'
		 ,'24_msk': mask_path + 'uds_mips24_mask.fits'
		 ,'100_img': map_path + 'UDS_level4_GRN_Unimap_img_wgls.fits'
		 ,'100_rms': map_path + 'UDS_level4_GRN_Unimap_cove_gls.fits'
		 ,'100_msk': mask_path + 'uds_pacsG_mask.fits'
		 ,'160_img': map_path + 'UDS_level4_RED_Unimap_img_wgls.fits'
		 ,'160_rms': map_path + 'UDS_level4_RED_Unimap_cove_gls.fits'
		 ,'160_msk': mask_path + 'uds_pacsR_mask.fits'
		 ,'250_img': map_path + 'uds_v4_itermap_'+pixsize_suffix+'_PSW.signal.cutout.fits'
		 ,'250_rms': map_path + 'uds_v4_itermap_'+pixsize_suffix+'_PSW.noise.cutout.fits'
		 ,'250_msk': mask_path + 'uds_spirePSW_mask.fits'
		 ,'350_img': map_path + 'uds_v4_itermap_'+pixsize_suffix+'_PMW.signal.cutout.fits'
		 ,'350_rms': map_path + 'uds_v4_itermap_'+pixsize_suffix+'_PMW.noise.cutout.fits'
		 ,'350_msk': mask_path + 'uds_spirePMW_mask.fits'
		 ,'500_img': map_path + 'uds_v4_itermap_'+pixsize_suffix+'_PLW.signal.cutout.fits'
		 ,'500_rms': map_path + 'uds_v4_itermap_'+pixsize_suffix+'_PLW.noise.cutout.fits'
		 ,'500_msk': mask_path + 'uds_spirePLW_mask.fits'
		 ,'850_img': map_path + 'f_UDS_8_flux_2015-02-16_new_header.fits'
		 ,'850_rms': map_path + 'f_UDS_8_rms_2015-02-16_new_header.fits'
		 ,'850_msk': mask_path + 'uds_scuba850_mask.fits'
		 ,'1100_img': map_path + 'sxdf_aste_kscott20100924_map.fits'
		 ,'1100_rms': map_path + 'sxdf_aste_kscott20100924_map_noise.fits'
		 ,'1100_msk': mask_path + 'uds_aztec_mask.fits'
			}
	
	return pf


def CatalogsParameterFile():
	
	current_path = os.path.dirname(__file__)
	
	catalog_path = current_path + '/../data/catalogs/'
	
	pf = {
		  'uds': catalog_path + 'uds8.csv'
		 ,'uvista_dr2': catalog_path + 'UVISTA_DR2_master_v2.1_USE.csv'
			}
	
	return pf
	
	
	
def WavelengthsParameterFile():
	
	# Decide which maps to include
	wavelength=[24,100,160,250,350,500,850,1100]
	wv0 = np.array([0,0,0,1,0,0,0,0])
	indstack = np.where(wv0 == 1)

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
	
	nwv = np.sum(wv0) 
	fwhm =[6.32, 7.4, 11.3, 18.1, 25.2, 36.6, 15., 18.]
	efwhm=[6.32, 6.7, 11.2, 17.6, 23.9, 35.2, 14.5, 18.] # want to the measured effective FWHM later
	#color_correction=[1.25,1.01,23.58,23.82,1.018,0.9914,0.95615,1.0]
	color_correction=[1.,1.,1.,1.,1.,1.,1.0E-3,1.]
	#beam_area = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.] #sr
	beam_area = [1.547E-9,1.,1.,1.,1.,1.,1.,1.] #sr
	
	pf = {
		  'wavelength': wavelength
		 ,'nwv': nwv
		 ,'indstack': indstack
		 ,'library_keys': library_keys
		 ,'fwhm': fwhm
		 ,'efwhm': efwhm
		 ,'color_correction': color_correction
		 ,'beam_area': beam_area
		}
	
	return pf

	
	
def BinningParameterFile(Binning_Scheme = "UNI"):
	
	if Binning_Scheme == "UNI":
		z_nodes = np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
		m_nodes_sf = np.array([8.5,9.5,10.0,10.5,11.0,13.0])
		m_nodes_qt = np.array([8.5,10.0,11.0,13.0])
	elif Binning_Scheme == "OPT":
		z_nodes = np.array([0.,0.37,0.53,0.72,0.97,1.31,1.83,2.74,5.])
		m_nodes_sf = np.array([8.5,9.5,10.,10.24,10.51,13.])
		m_nodes_qt = np.array([8.0,10.0,11.0,13.0])
	else:
		raise ValueError('--- Binning scheme unsupported ---')
	
	pf = {
		  'z_nodes': z_nodes
		 ,'m_nodes_sf': m_nodes_sf
		 ,'m_nodes_qt': m_nodes_qt
		  	}
		
	return pf
	
	
	
	
	
	
	
	
	
