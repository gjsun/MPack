import numpy as np
import sys, os


def GetFields():
	
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
	
	field_to_stack = GetFields()
	
	map_path = current_path + '/../data/maps/%s_maps/' % field_to_stack
	mask_path = map_path + 'masks/'
	pixsize_suffix = '4.0_arcsec_pixels'
	
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
	
	
	
	
	
def BinningParameterFile():
	
	Binning_Scheme = "UNI"
	
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
	
	
	
	
	
	
	
	
	
