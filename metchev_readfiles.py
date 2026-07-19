def calculate_uncertainty(residuals):
    """Calculates uncertainty as RMS of (i - i+1) / sqrt(2) per email instructions from Stan Metchev."""
    if len(residuals) < 2: return 0.005
    diffs = residuals[:-1] - residuals[1:]
    rms = np.sqrt(np.mean(diffs**2))
    return rms / np.sqrt(2)

def plot_metchev_files(f_path):
    data = np.genfromtxt(f_path).T
    flux = data[4]
    print(len(flux))
    time = data[0]
    file_name = os.path.basename(f_path)
    target_name = file_name.split('_fit')[0]
    # Col 0: Time, Col 3: Residuals, Col 4: Corrected Flux, Col 5: Ch1 Astro Model
    data_error_ch1 = calculate_uncertainty(data[3][0:362])
    data_error_ch2 = calculate_uncertainty(data[3][363:549])

    #plotting here
    fig, ax = plt.subplots(figsize=(10,6))
    ax.tick_params(axis='x', colors='black')  # Change x-axis tick label color
    ax.tick_params(axis='y', colors='black')  # Change y-axis tick label color
    ax.xaxis.label.set_color('black')         # Change x-axis label color
    ax.yaxis.label.set_color('black')         # Change y-axis label color
    ax.title.set_color('black')
    ax.set_title(f"{target_name}", fontsize=15, weight='bold')
    plt.ylabel('Relative Flux',fontsize=19)
    plt.xlabel('Elapsed Time (hr)',fontsize=19)
    
    #channel 1 plotting
    ax.errorbar(time[0:362],flux[0:362], yerr=np.abs(data_error_ch1), fmt='o', color='purple', ecolor ='plum',
                alpha=0.3, markersize=3, label='Ch1 Data (3.6μm)')
    ax.plot(time[0:362], data[5][0:362], '-', color='indigo', linewidth=2, label='Ch1 Fit')
    
    #channel 2
    ax.errorbar(time[363:549],flux[363:549], yerr=np.abs(data_error_ch2), fmt='s', color='teal', ecolor ='aqua',
                alpha=0.3, markersize=3, label='Ch1 Data (4.5μm)')
    ax.plot(time[363:549], data[5][363:549], '-', color='blue', linewidth=2, label='Ch2 Fit')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(target_name+'.pdf') #saving the file here


#plot_metchev_files('Stan_Metchev_Variable_Sample/2M1721_fit2out02.txt')    #example testing

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def stability_measure_stan_file(f_path, rotation_period, channel_config=None):
    """
    Analyzes atmospheric variability in Spitzer light curves by breaking continuous 
    photometry into discrete rotational cycles. Dynamically splits and evaluates stability 
    by specific wavelength channels using an optional  dictionary.
    """
    # Getting the target name from the files from Dr. Stan Metchev
    data = np.genfromtxt(f_path).T
    flux = data[4]
    time = data[0]
    file_name = os.path.basename(f_path)
    target_name = file_name.split('_fit')[0]
    
    # --- reading in the data ---
    # Col 0: Time, Col 3: Residuals, Col 4: Corrected Flux, Col 5: Ch1 Astro Model
    error = calculate_uncertainty(data[3])
    
    # this is where we set to just channel 1 if nothing provided
    channel_map = np.ones(len(time), dtype=int)
    ch1_label, ch2_label = '3.6um', '4.5um'
    data_error_ch1 = error
    data_error_ch2 = error
    
    if channel_config is not None:
        ch1_label = channel_config.get('ch1_name', '3.6um')
        ch2_label = channel_config.get('ch2_name', '4.5um')
        ch1_start, ch1_end = channel_config['ch1_indices']
        ch2_start, ch2_end = channel_config['ch2_indices']
        
        # Calculate localized uncertainties using uncertainiti function from Stan's file
        data_error_ch1 = calculate_uncertainty(data[3][ch1_start:ch1_end])
        data_error_ch2 = calculate_uncertainty(data[3][ch2_start:ch2_end])
        
        # Mapping out indices
        channel_map = np.zeros(len(time), dtype=int)
        channel_map[ch1_start:ch1_end] = 1
        channel_map[ch2_start:ch2_end] = 2
    
    # Just setting up plotting functions here
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', colors='black', labelsize=12)
    
    # Breaking up the entire time window into the rotation period of the brown dwarf
    total_duration = time[-1] - time[0]
    num_segments = int(np.floor(total_duration / rotation_period))
    
    # Starting report header
    print(f"--- Comprehensive Stability Report for {target_name} ---")
    print(f"Total Duration: {total_duration:.2f} hours | Period: {rotation_period} hours")
    if channel_config:
        print(f"Dynamic Channels Enabled: Ch1 ({ch1_label}) & Ch2 ({ch2_label})")
    print(f"{'Comparison':<14}{'Time (hr)':<16}{'Spearman Rho':<15}{'p-value':<12}{'Red. Chi2':<12}{'Status':<10}")
    print("-" * 85)
    
    # Alternating purple hex color tones because i like purple lol
    colors_pool = ['#f3eef7', '#e6daf0']
    common_phase_grid = np.linspace(0.0, 1.0, 100)
    
    # --- STEP 1: ISOLATE REFERENCE TEMPLATES FOR EACH PRESSED CHANNEL ---
    # Channel 1 Template Definition
    start_c1 = time[0]
    end_c1 = start_c1 + rotation_period
    mask_c1 = (time >= start_c1) & (time < end_c1) & (channel_map == 1)
    flux_baseline_ch1 = np.interp(common_phase_grid, (time[mask_c1] - start_c1) / rotation_period, flux[mask_c1]) if np.sum(mask_c1) > 5 else None

    # Channel 2 Template Definition (Finds the very first cycle after the channel switch)
    flux_baseline_ch2 = None
    start_time_ch2 = None
    idx_baseline_ch2 = None  # To track which cycle number becomes the Ch2 baseline
    
    if 2 in channel_map:
        time_ch2_all = time[channel_map == 2]
        if len(time_ch2_all) > 0:
            start_time_ch2 = time_ch2_all[0]
            end_time_ch2 = start_time_ch2 + rotation_period
            mask_c2 = (time >= start_time_ch2) & (time < end_time_ch2) & (channel_map == 2)
            if np.sum(mask_c2) > 5:
                flux_baseline_ch2 = np.interp(common_phase_grid, (time[mask_c2] - start_time_ch2) / rotation_period, flux[mask_c2])
                # Figure out which rotation period this time corresponds to
                idx_baseline_ch2 = int(np.floor((start_time_ch2 - time[0]) / rotation_period))
    
    # trying to make it pretty in plot
    flux_min, flux_max = np.min(flux), np.max(flux)
    flux_range = flux_max - flux_min
    ax.set_ylim(flux_min - 0.05 * flux_range, flux_max + 0.22 * flux_range)
    text_y_position = flux_max + 0.02 * flux_range
    
    # Tracking metrics separated by physical channel properties
    ch1_valid, ch1_stable, ch1_changing = 0, 0, 0
    ch2_valid, ch2_stable, ch2_changing = 0, 0, 0
    
    # --- STEP 2: looping through and going through each segment ---
    for i in range(num_segments):
        start_time = time[0] + (i * rotation_period)
        end_time = start_time + rotation_period
        mid_time = start_time + (rotation_period / 2)
        
        # Plotting alternative colors for each rotation period
        ax.axvspan(start_time, end_time, color=colors_pool[i % 2], alpha=0.7, zorder=0)
        
        # Mask and slice the current cycle's data points
        mask = (time >= start_time) & (time < end_time)
        seg_time, seg_flux = time[mask], flux[mask]
        seg_channels = channel_map[mask]
        
        # Figure out which channel rules this segment
        current_channel = 1 if np.sum(seg_channels == 1) >= np.sum(seg_channels == 2) else 2
        
        # Compute dynamic staggered Y position for plotting text to stop collisions
        if i % 2 == 0:
            staggered_y = text_y_position + (0.04 * flux_range)
        else:
            staggered_y = text_y_position

        # Establish Baseline identities on output logs
        if i == 0 and current_channel == 1:
            print(f"Cycle 1 vs 1   {f'{start_time:.1f}-{end_time:.1f}':<16}{1.000:<15.3f}{0.000:<12.3e}{0.000:<12.3f}{'BASELINE 1':<10}")
            ax.text(mid_time, staggered_y, f"Cy 1 ({ch1_label})\nBASELINE 1", color='#4a154b', fontsize=8, ha='center', weight='bold')
            continue
            
        if idx_baseline_ch2 is not None and i == idx_baseline_ch2:
            print(f"Cycle {i+1} vs {i+1}   {f'{start_time:.1f}-{end_time:.1f}':<16}{1.000:<15.3f}{0.000:<12.3e}{0.000:<12.3f}{'BASELINE 2':<10}")
            ax.text(mid_time, staggered_y, f"Cy {i+1} ({ch2_label})\nBASELINE 2", color='#004b49', fontsize=8, ha='center', weight='bold')
            continue
            
        if len(seg_flux) > 5:
            seg_phase = (seg_time - start_time) / rotation_period
            flux_current = np.interp(common_phase_grid, seg_phase, seg_flux)
            
            # Dynamically lock baseline and metrics to the active channel that we are working on
            baseline_template = flux_baseline_ch1 if current_channel == 1 else flux_baseline_ch2
            current_error = data_error_ch1 if current_channel == 1 else data_error_ch2
            wave_label = ch1_label if current_channel == 1 else ch2_label
            
            # boring printing stuff
            baseline_name = f"Cycle 1" if current_channel == 1 else f"Cycle {idx_baseline_ch2 + 1}"
            
            if baseline_template is None:
                print(f"{baseline_name:<12} vs {i+1:<3}{f'{start_time:.1f}-{end_time:.1f}':<16}{'Missing Baseline':<39}")
                continue
            
            # METRIC A (Morphology) & METRIC B (Scale)
            rho, p_val = spearmanr(baseline_template, flux_current)
            dof = len(common_phase_grid) - 1 
            chi2_val = np.sum(((flux_current - baseline_template) / current_error) ** 2)
            reduced_chi2 = chi2_val / dof
            
            is_shape_stable = (rho > 0.65) and (p_val < 0.05)
            is_amplitude_stable = (reduced_chi2 < 2.5)
            
            if is_shape_stable and is_amplitude_stable:
                cycle_status = "STABLE"
                text_color = '#5c5061' 
                if current_channel == 1: ch1_stable += 1
                else: ch2_stable += 1
            else:
                cycle_status = "CHANGING"
                text_color = 'crimson' 
                if current_channel == 1: ch1_changing += 1
                else: ch2_changing += 1
                
            if current_channel == 1: ch1_valid += 1
            else: ch2_valid += 1
                
            # Print dynamically mapping terminal feedback to real isolated pairings
            print(f"{baseline_name:<12} vs {i+1:<3}{f'{start_time:.1f}-{end_time:.1f}':<16}{rho:<15.3f}{p_val:<12.3e}{reduced_chi2:<12.3f}{cycle_status:<10}")
            
            # Formatted box metrics utilizing shortened strings, staggering, and a soft background mask
            ax.text(mid_time, staggered_y, 
                    f"Cy {i+1} ({wave_label})\n" + fr"$\rho$={rho:.2f}" + "\n" + fr"$\chi^2_\nu$={reduced_chi2:.1f}", 
                    color=text_color, fontsize=7.5, ha='center', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        else:
            baseline_name = f"Cycle 1" if current_channel == 1 else f"Cycle {idx_baseline_ch2 + 1}"
            print(f"{baseline_name:<12} vs {i+1:<3}{f'{start_time:.1f}-{end_time:.1f}':<16}{'Insufficient data':<39}")
            
    print("-" * 85)
    global_rho, global_p = spearmanr(time, flux)
    print(f"Global Scan (Entire 21 hrs): Spearman Rho = {global_rho:.3f}, p-value = {global_p:.3e}")
    print("-" * 85)
    
    # --- STEP 3: AUTOMATED SCORE CALCULATION & VERDICT ---
    score_ch1 = (ch1_stable / ch1_valid) * 100 if ch1_valid > 0 else 0.0
    score_ch2 = (ch2_stable / ch2_valid) * 100 if ch2_valid > 0 else 0.0
    
    total_valid = ch1_valid + ch2_valid
    total_changing = ch1_changing + ch2_changing
    
    print(f"METRIC SUMMARY FOR {target_name}:")
    print(f" -> {ch1_label} Stability Score: {score_ch1:.2f}% ({ch1_stable}/{ch1_valid} cycles stable)")
    if 2 in channel_map:
        print(f" -> {ch2_label} Stability Score: {score_ch2:.2f}% ({ch2_stable}/{ch2_valid} cycles stable)")
    print("-" * 85)
    
    if (abs(global_rho) > 0.35 and global_p < 0.05) or (total_valid > 0 and (total_changing / total_valid) > 0.25):
        classification_status = "EVOLVING"
        status_color = "crimson"
    else:
        classification_status = "STABLE"
        status_color = "darkgreen"
        
    print(f"FINAL CLASSIFICATION: {classification_status}")
    print("-" * 85 + "\n")
    
    # --- STEP 4: PLOT MAKING STRUCTURE ---
    ax.errorbar(time, flux, yerr=error, alpha=0.2, color='grey', fmt='none', zorder=1)
    ax.plot(time, flux, linestyle='none', marker='.', label='Data Points', color='black', alpha=0.8, zorder=2)
    ax.plot(time, data[5], linestyle='-', label='Best Fit Sinusoid', color='#6a0dad', linewidth=2.5, zorder=3)
    
    title_str = f"{target_name} | {ch1_label}: {score_ch1:.1f}%"
    if 2 in channel_map:
        title_str += f" | {ch2_label}: {score_ch2:.1f}%"
    ax.set_title(title_str, color=status_color, fontsize=18, pad=25, weight='bold')
    
    ax.set_xlabel('Elapsed Time (hr)', fontsize=14)
    ax.set_ylabel('Relative Flux', fontsize=14)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='lightgrey')
    
    plt.tight_layout()
    plt.savefig(f"{target_name}_comprehensive_stability.pdf", dpi=300)
    plt.show()  