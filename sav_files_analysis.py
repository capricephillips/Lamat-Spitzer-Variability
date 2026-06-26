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
    
#plot_sav_files('2M2343-3640_calibch1_bin5_ap_opt.sav')


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from scipy.stats import spearmanr

def stability_measure_sav_file(file, rotation_period):
    """
    Analyzes atmospheric variability in Spitzer light curves by breaking continuous 
    photometry into discrete rotational cycles. Each subsequent rotation period is statistically 
    evaluated against the first cycle (the baseline) using non-parametric correlation 
    and a chi-squared goodness-of-fit test to categorize atmospheric evolution.
    """
    # getting the target name from the files from Dr. Johanna Vos
    file_name = os.path.basename(file)
    target_name = file_name.split('_calibch1')[0]
    
    # we are just reading the data here and getting out time, calibrated flux and best fit sin function 
    sav_data = readsav(file)
    time = np.squeeze(sav_data['bint'][:, 0])    # hours
    flux = np.squeeze(sav_data['targfcal'])    # Normalized, calibrated light curve flux values
    model = np.squeeze(sav_data['mod1'])       # best fit sin function
    
    # Calculate flux uncertainity from script Dr. Vos provideded
    binflux_error = np.std(flux - np.roll(flux, 1)) / np.sqrt(2)
    binflux_error_array = binflux_error * (0 * time + 1) # Vectorized error array for plotting bounds
    
    # Just setting up plotting functions here
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', colors='black', labelsize=12)
    
    # breaking up the entire time window into the rotation period of the brown dwarf | which is why knowing the rotation period is key work here
    total_duration = time[-1] - time[0]
    num_segments = int(np.floor(total_duration / rotation_period))
    
    # Starting report header
    print(f"--- Comprehensive Stability Report for {target_name} ---")
    print(f"Total Duration: {total_duration:.2f} hours | Period: {rotation_period} hours")
    print(f"{'Comparison':<14}{'Time (hr)':<16}{'Spearman Rho':<15}{'p-value':<12}{'Red. Chi2':<12}{'Status':<10}")
    print("-" * 85)
    
    # Alternating purple hex color tones because i like purple lol
    colors_pool = ['#f3eef7', '#e6daf0']
    
    # Define a standard 100-point phase grid to normalize varying sample sizes across loops
    common_phase_grid = np.linspace(0.0, 1.0, 100)
    
    # --- STEP 1: ISOLATE THE REFERENCE TEMPLATE (First Rotation) ---
    start_c1 = time[0]
    end_c1 = start_c1 + rotation_period
    mask_c1 = (time >= start_c1) & (time < end_c1)
    
    time_c1 = time[mask_c1]
    flux_c1 = flux[mask_c1]
    
    if len(flux_c1) <= 5:
        print("Error: Insufficient data points in Cycle 1.")
        return
        
    # We are phase folding the light curve here based on the known rotation period of the brown dwarf
    phase_c1 = (time_c1 - start_c1) / rotation_period
    # Interpolate Cycle (Rotation) 1 onto the uniform 100-point array to act as the base template
    flux_baseline = np.interp(common_phase_grid, phase_c1, flux_c1)
    
    #just setting up things so the text at plot does over run here
    flux_min = np.min(flux)
    flux_max = np.max(flux)
    flux_range = flux_max - flux_min
    

    ax.set_ylim(flux_min - 0.05 * flux_range, flux_max + 0.15 * flux_range)
    
    
    text_y_position = flux_max + 0.02 * flux_range
    
    evolving_cycles_count = 0
    valid_segments = 0
    
    # --- STEP 2: LOOP AND ANALYZE SUBSEQUENT ROTATIONS ---
    for i in range(num_segments):
        start_time = time[0] + (i * rotation_period)
        end_time = start_time + rotation_period
        
        # Plotting alternative colors for each rotation period
        ax.axvspan(start_time, end_time, color=colors_pool[i % 2], alpha=0.7, zorder=0)
        
        # Mask and slice the current cycle's data points
        mask = (time >= start_time) & (time < end_time)
        seg_time = time[mask]
        seg_flux = flux[mask]
        
        mid_time = start_time + (rotation_period / 2)
        
        if i == 0:
            # Establish self-comparison values for the baseline rotation period | making assumption here
            print(f"Cycle 1 vs 1   {f'{start_time:.1f}-{end_time:.1f}':<16}{1.000:<15.3f}{0.000:<12.3e}{0.000:<12.3f}{'BASELINE':<10}")
            ax.text(mid_time, text_y_position, f"Cycle 1\nBASELINE", 
                    color='#4a154b', fontsize=9, ha='center', weight='bold')
            continue
            
        if len(seg_flux) > 5:
            valid_segments += 1
            
            # Normalize the current cycle's timestamps to standard 0.0-1.0 phase space
            seg_phase = (seg_time - start_time) / rotation_period
            # Interpolate onto the standard 100-point array for direct comparison with the template
            flux_current = np.interp(common_phase_grid, seg_phase, seg_flux)
            
            # METRIC A (Morphology): Spearman test evaluates similarity of wave patterns (shape changes)
            rho, p_val = spearmanr(flux_baseline, flux_current)
            
            # METRIC B (Scale): Reduced Chi-Squared tracks absolute variations in amplitude/noise
            dof = len(common_phase_grid) - 1 # Degrees of Freedom
            chi2_val = np.sum(((flux_current - flux_baseline) / binflux_error) ** 2)
            reduced_chi2 = chi2_val / dof
            
            # CRITICAL BALANCING LOGIC here | based on spearman correlation steps
            # A cycle is classified as STABLE only if the wave morphology/shape correlates well (rho > 0.65) 
            # AND the absolute amplitude variation remains within statistical noise thresholds (chi2 < 2.5)
            is_shape_stable = (rho > 0.65) and (p_val < 0.05)
            is_amplitude_stable = (reduced_chi2 < 2.5)
            
            if is_shape_stable and is_amplitude_stable:
                cycle_status = "STABLE"
                text_color = '#5c5061' # Slate purple-grey for coherent cycles
            else:
                evolving_cycles_count += 1
                cycle_status = "CHANGING"
                text_color = 'crimson' # Crimson alert text for active atmospheric drift
                
            print(f"Cycle 1 vs {i+1:<3}{f'{start_time:.1f}-{end_time:.1f}':<16}{rho:<15.3f}{p_val:<12.3e}{reduced_chi2:<12.3f}{cycle_status:<10}")
            
            # just label stuff
            ax.text(mid_time, text_y_position, fr"Cycle {i+1}" + "\n" + fr"$\rho$={rho:.2f}" + "\n" + fr"$\chi^2_\nu$={reduced_chi2:.1f}", 
                    color=text_color, fontsize=8, ha='center', weight='bold')
        else:
            print(f"Cycle 1 vs {i+1:<3}{f'{start_time:.1f}-{end_time:.1f}':<16}{'Insufficient data':<39}")
            
    print("-" * 85)
    # Perform a global correlation test across the full, uninterrupted time span
    global_rho, global_p = spearmanr(time, flux)
    print(f"Global Scan (Entire 21 hrs): Spearman Rho = {global_rho:.3f}, p-value = {global_p:.3e}")
    print("-" * 85)
    
    # --- STEP 3: FINALLY VERDICT PRINTED HERE---
    # Trigger an EVOLVING classification if a persistent global trend exists OR if >25% of the cycles morph
    if (abs(global_rho) > 0.35 and global_p < 0.05) or (valid_segments > 0 and (evolving_cycles_count / valid_segments) > 0.25):
        classification_status = "EVOLVING"
        status_color = "crimson"
    else:
        classification_status = "STABLE"
        status_color = "darkgreen"
        
    print(f"FINAL CLASSIFICATION: {classification_status}")
    print("-" * 85 + "\n")
    
    # --- STEP 4: PLOT MAKING STRUCTURE ---
    # Error bar plot
    ax.errorbar(time, flux, yerr=binflux_error_array, alpha=0.2, color='grey', fmt='none', zorder=1)
    # plotting the data
    ax.plot(time, flux, linestyle='none', marker='.', label='Data Points', color='black', alpha=0.8, zorder=2)
    # Overlay the BEST FIT SIN FUNCTION HERE
    ax.plot(time, model, linestyle='-', label='Best Fit Sinusoid', color='#6a0dad', linewidth=2.5, zorder=3)
    
    # settign up titles for viewing
    ax.set_title(f"{target_name} | Atmosphere Status: {classification_status} (Period = {rotation_period}h)", 
                 color=status_color, fontsize=15, pad=25, weight='bold')
    ax.set_xlabel('Elapsed Time (hr)', fontsize=14)
    ax.set_ylabel('Relative Flux', fontsize=14)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='lightgrey')
    
    # Saving the figure for you to see
    plt.tight_layout()
    pdf_filename = f"{target_name}_comprehensive_stability.pdf"
    plt.savefig(pdf_filename, dpi=300)
    plt.show()

#Testing script
stability_measure_sav_file('2M0031+5749_calibch1_bin5_ap_opt.sav', rotation_period=1.64)
