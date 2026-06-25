def plot_sav_files(file):
    sav_fname = file
    target_name = file.split('_calibch1')[0]
    sav_data = readsav(sav_fname)
    fig, ax = plt.subplots(figsize=(10,6))
    #Plotting part
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.tick_params(axis='x', colors='black')  # Change x-axis tick label color
    ax.tick_params(axis='y', colors='black')  # Change y-axis tick label color
    ax.xaxis.label.set_color('black')         # Change x-axis label color
    ax.yaxis.label.set_color('black')         # Change y-axis label color
    ax.title.set_color('white')
    #plotting data binned
    plt.plot(sav_data['bint'][:,0], sav_data['targfcal'],linestyle='none',marker='.',label='Data',color='black')
    #plotting error for data
    plt.errorbar(sav_data['bint'][:,0], np.squeeze(sav_data['targfcal']),yerr=binflux_error_array,alpha=0.2,color='grey')
    #plotting best fit sin function
    plt.plot(sav_data['bint'][:,0], sav_data['mod1'],linestyle='-',label='Best fit sin',color='crimson',linewidth=3)
    plt.ylabel('Relative Flux',fontsize=19)
    plt.xlabel('Elapsed')
    plt.ylabel('Relative Flux',fontsize=19)
    plt.xlabel('Elapsed Time (hr)',fontsize=19)
    plt.tight_layout()
    plt.savefig(f"{target_name}.pdf")
    
#example    
    
    
plot_sav_files('2M2343-3640_calibch1_bin5_ap_opt.sav')
