from COPS_example import *

k_pt = 0.1

N_logM = 9
N_sigma = 3
logM_max_list = np.linspace(8.5,12.5,N_logM)
print logM_max_list
sigma_list = np.array([0.1,0.5,1.0])
alpha_list = np.linspace(0.2,0.8,N_logM)



#COPS_list = np.zeros((N_logM, N_sigma))
COPS32_list = np.zeros((N_logM, N_sigma))
COPS43_list = np.zeros((N_logM, N_sigma))

for j in range(N_sigma):
    for i in range(N_logM):

        PS_generated_32_sf = [COPowerSpectrum(sigma_t=sigma_list[j], pop='sf').PCO_kPk_obs(J=3,log10M_max=logM_max_list[i])[0],
                              COPowerSpectrum(sigma_t=sigma_list[j], pop='sf').PCO_kPk_obs(J=3,log10M_max=logM_max_list[i])[1]]

        PS_generated_32_qt = [COPowerSpectrum(sigma_t=sigma_list[j], pop='qt').PCO_kPk_obs(J=3,log10M_max=logM_max_list[i])[0],
                              COPowerSpectrum(sigma_t=sigma_list[j], pop='qt').PCO_kPk_obs(J=3,log10M_max=logM_max_list[i])[1]]

        PS_generated_43_sf = [COPowerSpectrum(sigma_t=sigma_list[j], pop='sf').PCO_kPk_obs(J=4,log10M_max=logM_max_list[i])[0],
                              COPowerSpectrum(sigma_t=sigma_list[j], pop='sf').PCO_kPk_obs(J=4,log10M_max=logM_max_list[i])[1]]

        PS_generated_43_qt = [COPowerSpectrum(sigma_t=sigma_list[j], pop='qt').PCO_kPk_obs(J=4,log10M_max=logM_max_list[i])[0],
                              COPowerSpectrum(sigma_t=sigma_list[j], pop='qt').PCO_kPk_obs(J=4,log10M_max=logM_max_list[i])[1]]


        
        xs_t3i, ys_t3i = PS_generated_32_sf
        xs_t3ii, ys_t3ii = PS_generated_32_qt
        
        xs_t4i, ys_t4i = PS_generated_43_sf
        xs_t4ii, ys_t4ii = PS_generated_43_qt

        # Determine the index of k=0.1 h/cMpc
        ind3 = np.argmin(abs(xs_t3i-k_pt))
        ind4 = np.argmin(abs(xs_t4i-k_pt))

        k3_axis = (xs_t3i+xs_t3ii)/2.
        ps3_axis = ys_t3i+ys_t3ii
        k4_axis = (xs_t4i+xs_t4ii)/2.
        ps4_axis = ys_t4i+ys_t4ii

        COPS32_list[i,j] = ps3_axis[ind3]
        COPS43_list[i,j] = ps4_axis[ind4]
        
        
        
        
        
        
        
        
fig, ax = plt.subplots(2, 1, sharey=True, figsize=(24,24), dpi=200)
plt.subplots_adjust(wspace=0.05,hspace=0.5)






# ======================================   Upper Panel   ====================================== #
ax1 = ax[0]
ax2 = ax1.twiny()

l_dashes = [15,5,15,5]
ls_dashes = [15,10,5,10]


# First plot C[II] models
ax1.plot(logM_max_list, np.ones_like(logM_max_list)*5.0E8, 'b-', lw=5, alpha=0.8, label='$\mathrm{C[II]_{z=6.5},\ Gong+12}$')
ax1.plot(logM_max_list, np.ones_like(logM_max_list)*4.0E7, 'r-', lw=5, alpha=0.8, label='$\mathrm{C[II]_{z=6.5},\ Silva+15}$')


# High S/N
ax1.annotate('',fontsize=20,
            xy=(9.01, 5.0e8),
            xytext=(9.01, 1.5e6),
            arrowprops=dict(arrowstyle="<->",ec='grey',linewidth=5.0,
                            connectionstyle="arc3"))
ax1.text(9.05, 1.0e7, r'$\frac{P_{\mathrm{C[II]}}}{P_{\mathrm{CO(3-2)}}} \sim 300$', 
         fontsize=26, color='grey')

ax1.text(8.6, 4.0e9, '$z_{\mathrm{C[II]}}=6.5$', color='black', fontsize=40)

# Low S/N
ax1.annotate('',fontsize=20,
            xy=(8.99, 4.0e7),
            xytext=(8.99, 1.5e6),
            arrowprops=dict(arrowstyle="<->",ec='black',linewidth=5.0,
                            connectionstyle="arc3"))
ax1.text(8.55, 2.0e6, r'$\frac{P_{\mathrm{C[II]}}}{P_{\mathrm{CO(3-2)}}} \sim 25$', 
         fontsize=26)

# Second plot CO models
l0, = ax1.plot(logM_max_list, COPS32_list[:,0], '-', color='orange', lw=5, zorder=10,
               label='$\mathrm{CO,\ }\sigma_{\mathrm{tot}} = %.1f$'%sigma_list[0])
l1, = ax1.plot(logM_max_list, COPS32_list[:,1], '--', color='orange', lw=5, 
               label='$\mathrm{CO,\ }\sigma_{\mathrm{tot}} = %.1f$'%sigma_list[1])
l1.set_dashes(l_dashes)
l2, = ax1.plot(logM_max_list, COPS32_list[:,2], '--', color='orange', lw=5, 
               label='$\mathrm{CO,\ }\sigma_{\mathrm{tot}} = %.1f$'%sigma_list[2])
l2.set_dashes(ls_dashes)


ax1.set_yscale('log')
ax1.set_ylim([1.0E3,2.0E10])
ax1.set_xlabel(r"$\log (M_{\mathrm{*}}/M_{\odot})$", fontsize=25)
ax1.set_ylabel('$kP_{\mathrm{CO(3-2)}}(k=0.1\ h\mathrm{/Mpc})\ \mathrm{[(Jy/sr)^2(Mpc/}h)^2]$', fontsize=32)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', which='minor', labelsize=20)
lgd = ax1.legend(loc='lower center',frameon=False,numpoints=1,ncol=2,prop={'size':30})
plt.setp(lgd.get_texts()[0], color = 'blue')
plt.setp(lgd.get_texts()[1], color = 'red')

#new_tick_locations = np.array([8.75, 9.25, 9.75, 10.25, 10.75])
new_tick_labels = np.array([22., 21., 20., 19., 18., 17., 16.])

from Ks_Mstar_Estimate import fit_Ks_logM_3 as fit_Ks_logM_3
from Ks_Mstar_Estimate import fit_logM_Ks_3 as fit_logM_Ks_3

def inverse_tick_function(Ks):
    val = np.zeros_like(Ks)
    for i in range(np.size(Ks)):
        val[i] = "%.2f" % fit_logM_Ks_3(Ks[i])
    return val


print fit_Ks_logM_3(9.0)

ax2.set_xlim(ax1.get_xlim())
#ax2.set_xticks(inverse_tick_function(new_tick_labels))
#ax2.set_xticklabels(new_tick_labels)
ax2.set_xticks([8.73,9.35,9.87,10.23,10.7,11.3])
ax2.set_xticklabels([15,10,5,3,1,0.1])
ax2.set_xlabel(r"$f_{\mathrm{mask}}\ [\%]$", fontsize=28, labelpad=12)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='minor', labelsize=20)
#for label in ax3.get_yticklabels():
#    label.set_color("red")
    
    
#ax3.tick_params(axis='both', which='major', labelsize=25)
#ax3.tick_params(axis='both', which='minor', labelsize=25)


# --- Add another axis at top ---
newax1 = ax1.twiny()

newax1.set_frame_on(True)
newax1.patch.set_visible(False)
newax1.xaxis.set_ticks_position('top')
newax1.xaxis.set_label_position('top')
newax1.spines['top'].set_position(('outward', 65))

newax1.set_xlim(ax1.get_xlim())
#newax1.set_xticks([8.73,9.35,9.87,10.23,10.7,11.3])
newax1.set_xticks(inverse_tick_function(new_tick_labels))
#newax1.set_xticklabels([15,10,5,3,1,0.1])
newax1.set_xticklabels(new_tick_labels)
newax1.tick_params(axis='both', which='major', labelsize=20)
newax1.tick_params(axis='both', which='minor', labelsize=20)
#newax1.set_xlabel(r"$f_{\mathrm{mask}}\ [\%]$", fontsize=32)
newax1.set_xlabel(r"$\mathbf{m_{\mathrm{K}}^{\mathrm{AB}}}$", fontsize=25, labelpad=15)


for spinename, spine in newax1.spines.iteritems():
    if spinename != 'top':
        spine.set_visible(False)


# --- Add another axis at bottom ---
newax2 = ax1.twiny()

newax2.set_frame_on(True)
newax2.patch.set_visible(False)
newax2.xaxis.set_ticks_position('bottom')
newax2.xaxis.set_label_position('bottom')
newax2.spines['bottom'].set_position(('outward', 65))

newax2.set_xlim(ax1.get_xlim())
z_CO_32 = 0.36
newax2_tickf = lambda logM: np.log10(L_M_z(10**logM, z_CO_32, None, sigma=0.01, gal_type='sf'))
newax2_ticklabels = np.array([8.5,9.0,9.5,10.0,10.5,11.0,12.0])
newax2.set_xticks(newax2_tickf(newax2_ticklabels))
newax2.set_xticklabels(newax2_ticklabels)
newax2.tick_params(axis='both', which='major', labelsize=20)
newax2.tick_params(axis='both', which='minor', labelsize=20)
newax2.set_xlabel(r"$\log (L_{\mathrm{IR}}/L_{\odot})$", fontsize=25)


for spinename, spine in newax2.spines.iteritems():
    if spinename != 'bottom':
        spine.set_visible(False)
        
ax1.grid(True)






# ======================================   Lower Panel   ====================================== #

ax1b = ax[1]
ax2b = ax1b.twiny()

l_dashes = [15,5,15,5]
ls_dashes = [15,10,5,10]

# First plot C[II] models
ax1b.plot(logM_max_list, np.ones_like(logM_max_list)*5.0E8, 'b-', lw=5, alpha=0.8, label='$\mathrm{C[II]_{z=6.5},\ Gong+12}$')
ax1b.plot(logM_max_list, np.ones_like(logM_max_list)*4.0E7, 'r-', lw=5, alpha=0.8, label='$\mathrm{C[II]_{z=6.5},\ Silva+15}$')

# High S/N
ax1b.annotate('',fontsize=20,
            xy=(9.01, 5.0e8),
            xytext=(9.01, 2.0e5),
            arrowprops=dict(arrowstyle="<->",ec='grey',linewidth=5.0,
                            connectionstyle="arc3"))
ax1b.text(9.05, 5.0e6, r'$\frac{P_{\mathrm{C[II]}}}{P_{\mathrm{CO(4-3)}}} \sim 2000$', 
         fontsize=26, color='grey')

ax1b.text(8.6, 4.0e9, '$z_{\mathrm{C[II]}}=6.5$', color='black', fontsize=40)

# Low S/N
ax1b.annotate('',fontsize=20,
            xy=(8.99, 4.0e7),
            xytext=(8.99, 2.0e5),
            arrowprops=dict(arrowstyle="<->",ec='black',linewidth=5.0,
                            connectionstyle="arc3"))
ax1b.text(8.55, 3.0e5, r'$\frac{P_{\mathrm{C[II]}}}{P_{\mathrm{CO(4-3)}}} \sim 200$', 
         fontsize=26)

# Second CO models
l0, = ax1b.plot(logM_max_list, COPS43_list[:,0], '-', color='orange', lw=5, zorder=10,
               label='$\mathrm{CO,\ }\sigma_{\mathrm{tot}} = %.1f$'%sigma_list[0])
l1, = ax1b.plot(logM_max_list, COPS43_list[:,1], '--', color='orange', lw=5, 
               label='$\mathrm{CO,\ }\sigma_{\mathrm{tot}} = %.1f$'%sigma_list[1])
l1.set_dashes(l_dashes)
l2, = ax1b.plot(logM_max_list, COPS43_list[:,2], '--', color='orange', lw=5, 
               label='$\mathrm{CO,\ }\sigma_{\mathrm{tot}} = %.1f$'%sigma_list[2])
l2.set_dashes(ls_dashes)






ax1b.set_yscale('log')
ax1b.set_ylim([1.0E3,2.0E10])
ax1b.set_xlabel(r"$\log (M_{\mathrm{*}}/M_{\odot})$", fontsize=25)
ax1b.set_ylabel('$kP_{\mathrm{CO(4-3)}}(k=0.1\ h\mathrm{/Mpc})\ \mathrm{[(Jy/sr)^2(Mpc/}h)^2]$', fontsize=32)
ax1b.tick_params(axis='both', which='major', labelsize=20)
ax1b.tick_params(axis='both', which='minor', labelsize=20)
lgd = ax1b.legend(loc='lower center',frameon=False,numpoints=1,ncol=2,prop={'size':30})
plt.setp(lgd.get_texts()[0], color = 'blue')
plt.setp(lgd.get_texts()[1], color = 'red')


#new_tick_locations = np.array([8.75, 9.25, 9.75, 10.25, 10.75])
new_tick_labels = np.array([23., 22., 21., 20., 19., 18., 17.])

from Ks_Mstar_Estimate import fit_Ks_logM_4 as fit_Ks_logM_4
from Ks_Mstar_Estimate import fit_logM_Ks_4 as fit_logM_Ks_4

def inverse_tick_function(Ks):
    val = np.zeros_like(Ks)
    for i in range(np.size(Ks)):
        val[i] = "%.2f" % fit_logM_Ks_4(Ks[i])
    return val

ax2b.set_xlim(ax1.get_xlim())
#ax2.set_xticks(inverse_tick_function(new_tick_labels))
#ax2.set_xticklabels(new_tick_labels)
ax2b.set_xticks([8.73,9.35,9.87,10.23,10.7,11.3])
ax2b.set_xticklabels([15,10,5,3,1,0.1])
ax2b.set_xlabel(r"$f_{\mathrm{mask}}\ [\%]$", fontsize=28, labelpad=12)
ax2b.tick_params(axis='both', which='major', labelsize=20)
ax2b.tick_params(axis='both', which='minor', labelsize=20)
#for label in ax3.get_yticklabels():
#    label.set_color("red")
    
    
#ax3.tick_params(axis='both', which='major', labelsize=25)
#ax3.tick_params(axis='both', which='minor', labelsize=25)


# --- Add another axis at top ---
newax1b = ax1b.twiny()

newax1b.set_frame_on(True)
newax1b.patch.set_visible(False)
newax1b.xaxis.set_ticks_position('top')
newax1b.xaxis.set_label_position('top')
newax1b.spines['top'].set_position(('outward', 65))

newax1b.set_xlim(ax1.get_xlim())
#newax1.set_xticks([8.73,9.35,9.87,10.23,10.7,11.3])
newax1b.set_xticks(inverse_tick_function(new_tick_labels))
#newax1.set_xticklabels([15,10,5,3,1,0.1])
newax1b.set_xticklabels(new_tick_labels)
newax1b.tick_params(axis='both', which='major', labelsize=20)
newax1b.tick_params(axis='both', which='minor', labelsize=20)
#newax1.set_xlabel(r"$f_{\mathrm{mask}}\ [\%]$", fontsize=32)
newax1b.set_xlabel(r"$\mathbf{m_{\mathrm{K}}^{\mathrm{AB}}}$", fontsize=25, labelpad=15)


for spinename, spine in newax1.spines.iteritems():
    if spinename != 'top':
        spine.set_visible(False)


# --- Add another axis at bottom ---
newax2b = ax1b.twiny()

newax2b.set_frame_on(True)
newax2b.patch.set_visible(False)
newax2b.xaxis.set_ticks_position('bottom')
newax2b.xaxis.set_label_position('bottom')
newax2b.spines['bottom'].set_position(('outward', 65))

newax2b.set_xlim(ax1.get_xlim())
z_CO_43 = 0.82
newax2b_tickf = lambda logM: np.log10(L_M_z(10**logM, z_CO_43, None, sigma=0.01, gal_type='sf'))
newax2b_ticklabels = np.array([8.5,9.0,9.5,10.0,10.5,11.0,12.0])
newax2b.set_xticks(newax2b_tickf(newax2b_ticklabels))
newax2b.set_xticklabels(newax2_ticklabels)
newax2b.tick_params(axis='both', which='major', labelsize=20)
newax2b.tick_params(axis='both', which='minor', labelsize=20)
newax2b.set_xlabel(r"$\log (L_{\mathrm{IR}}/L_{\odot})$", fontsize=25)


for spinename, spine in newax2.spines.iteritems():
    if spinename != 'bottom':
        spine.set_visible(False)


    
ax1b.grid(True)



plt.savefig('./MaskingTwoPanels.png', dpi=200)
#plt.show()