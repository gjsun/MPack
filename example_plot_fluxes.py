import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter, FixedLocator
import itertools

from MPack_Core.utils import parameter_file as master_pf



################################################################################
### Plot the stacked flux density as a function of stellar mass and redshift ###
################################################################################



wvl_pf = master_pf.WavelengthsParameterFile()
binning_pf = master_pf.BinningParameterFile()

wavelength = wvl_pf['wavelength']
nwv = len(wavelength)
z_nodes = binning_pf['z_nodes']
z_mids = (z_nodes[0:-1]+z_nodes[1::])/2.
m_nodes_sf = binning_pf['m_nodes_sf']
m_mids_sf = (m_nodes_sf[0:-1]+m_nodes_sf[1::])/2.
m_nodes_qt = binning_pf['m_nodes_qt']

fig, ax = plt.subplots(2, 4, sharey=False, figsize=(16,8), dpi=200)

for i in range(nwv):
	
	print 'Plotting %dum...' % wavelength[i]
	
	path_uds = './MPack_Core/results/mean_flux_matrices/uds_uni/flux_matrix_sf_%d.dat' % wavelength[i]
	path_uvista = './MPack_Core/results/mean_flux_matrices/uvista_uni/flux_matrix_sf_%d.dat' % wavelength[i]
	with open(path_uds) as file: 
		data_uds = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 0, None)])
	with open(path_uvista) as file: 
		data_uvista = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 0, None)])

	t1 = i/(nwv/2)
	t2 = i%(nwv/2)
	
	# UDS
	ax[t1,t2].plot(z_mids, data_uds[:,-1], 'ko-', mfc='none', mec='k', mew=2, label=r'$\log(M/M_{\odot})=%.1f$' % m_mids_sf[-1])
	ax[t1,t2].plot(z_mids, data_uds[:,-2], 'mo-', mfc='none', mec='m', mew=2, label=r'$\log(M/M_{\odot})=%.1f$' % m_mids_sf[-2])
	ax[t1,t2].plot(z_mids, data_uds[:,-3], 'ro-', mfc='none', mec='r', mew=2, label=r'$\log(M/M_{\odot})=%.1f$' % m_mids_sf[-3])
	ax[t1,t2].plot(z_mids, data_uds[:,-4], 'go-', mfc='none', mec='g', mew=2, label=r'$\log(M/M_{\odot})=%.1f$' % m_mids_sf[-4])
	# UVISTA
	ax[t1,t2].plot(z_mids, data_uvista[:,-1], 'ks--', mfc='none', mec='k', mew=2, alpha=0.5)
	ax[t1,t2].plot(z_mids, data_uvista[:,-2], 'ms--', mfc='none', mec='m', mew=2, alpha=0.5)
	ax[t1,t2].plot(z_mids, data_uvista[:,-3], 'rs--', mfc='none', mec='r', mew=2, alpha=0.5)
	ax[t1,t2].plot(z_mids, data_uvista[:,-4], 'gs--', mfc='none', mec='g', mew=2, alpha=0.5)
	
	ax[t1,t2].text(0.2,0.015,'%dum' % wavelength[i], color='black')
	
	
	ax[1,t2].set_xlabel('Redshift', fontsize=15)
	ax[t1,0].set_ylabel('Stacked Flux Density [mJy]', fontsize=15)
	ax[t1,t2].set_yscale('log')
	ax[t1,t2].set_ylim([0.01,40.])
	
	nbinsx = len(ax[0,0].get_xticklabels())
	nbinsy = len(ax[0,0].get_yticklabels())
	
	ax[0,t2].xaxis.set_major_formatter( NullFormatter() )
	#ax[0,0].get_yticklabels()[1].set_visible(False)
	
	if t2!=0:
		ax[1,t2].xaxis.set_major_locator(MaxNLocator(nbins=nbinsx, prune='lower'))
		ax[0,t2].yaxis.set_major_formatter( NullFormatter() )
		ax[1,t2].yaxis.set_major_formatter( NullFormatter() )

ax[0,0].legend(loc='upper right', numpoints=1, frameon=False)

t_i = ax[0,1].text(3.5,20., 'UDS', withdash=True, dashlength=80, dashpad=15)
t_ii = ax[0,1].text(3.5,10., 'UVISTA', alpha=0.5, withdash=True, dashlength=80, dashpad=15)
dash_i = t_ii.dashline
dash_i.set_color('k')
dash_i.set_dashes((6,6))

plt.suptitle('Stacked Flux Density of Star-Forming Galaxies', fontsize=18)
plt.subplots_adjust(wspace=0.001,hspace=0.001)
plt.savefig('./example_plots/flux_comp_all.png', dpi=200)

print '\n----- ALL DONE -----\n'