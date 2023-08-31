# MIT License

# Copyright (c) 2023 Sarah H. Needleman

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# # # # # # 
# Sarah H. Needleman
# University College London

# Script for running ICA and plotting/saving the results.

import functions_RunICA_andPlot__NiftyReg_2022_08_26




# # # # # MRI acquisition parameters
# Voxel dimensions
Vox1 = 96
Vox2 = 96
# Dynamic imaging temporal resolution /s:
TempRes = 1.5
# TE_1 and TE_2 = echo times /ms, in the form of a list.
TE_1 = 0.71
TE_2 = 1.2
TE_s = [TE_1, TE_2]
# Number of echoes acquired = length of TE_s list.
NumEchoes = len(TE_s) # 2
Cycled3 = 1 # Were gases switched multiple times

NumSlices = 4
NumDyn = 420
# Gas switching period /number of dynamics:
gas_switching = 60
# Number of images to average over (the last av_im_num of images) in each gas cycle
# to create the mean oxygen image:
av_im_num = 5

# Also, for both SingleCycle and Cycled3, set GasSwitch to a list of
# [Cycled3 Y/N, list of switching times if SingleCycle or single value if Cycled3]
GasSwitch = [Cycled3, gas_switching]

# Registered image series name ending
NiftyReg_RegIDs=['cpg3_be0p005']



# # # # #  ICA settings
# Number of ICA components to find: from num_c_start to num_c_stop.
num_c_start = 22
num_c_stop = 72
# Minimum number of components to consider when identifying the OE ICA component. (e.g. 21 for 22 and greater).
g_num = 21






# # # # # What to run:
# Which plots to create: 0/1 for Y/N.
do_runICA = 1 #1               # Run ICA.
do_ICA_ordering = 0 #0            # *ONLY* calculate ICA ordering, requires ICA to have been run (if do_runICA = 1 the ordering will be calculated automatically).
do_ICA_PSE_calculation = 1 #1     # Use ordering information, reconstruct ICA OE component, calculate PSE etc.
do_anyPlotting = 1 #1             # Perform any plotting.


# # # # # # What to plot:
# DPI value when saving plots
dpi_save = 300
plot_ReconOE = 1
plot_MRI_SI = 1
# # ** Which plots to create: 0/1 for Y/N.
plot_PSE_timeseries = 1###1             # Plot PSE timeseries (reconstructed OE ICA component and/or MRI SI data, see above).
plot_OE_ICA = 1###1                     # Plot specifically/only the OE ICA component frequency spectrum and timeseries (will run with any 1/0 in plot_ReconOE and plot_MRI_SI).
plot_ICA_components = 0###0#1             # Plot all ICA component maps for the run of ICA which contained the OE component (will run with any 1/0 in plot_ReconOE and plot_MRI_SI).
plot_all_component_FreqScaled = 0###1   # Plot the frequency spectra of all ICA components, scaled by the gas cycling frequency as used in the frequency RMS calculation, for the run of ICA which contained the OE component (will run with any 1/0 in plot_ReconOE and plot_MRI_SI). - keep as 0, and only plot scaled version.
plot_PSE_overlay = 1 ###1                # Plot PSE maps overlay (SI images) (reconstructed OE ICA component and/or MRI SI data, see above). **Alternative to plotting PSE maps.
plot_PSE_overlay_LungMask = 0#NOT SETUP       # Plot PSE maps overlay - within *LUNG MASK* only (SI images) (reconstructed OE ICA component and/or MRI SI data, see above). **Alternative to plotting PSE maps.
plot_OE_component_overlay = 0###1       # Plot OE component maps overlay (SI images) (OE ICA component only). **Alternative to plotting the OE component maps.
plot_clims_set = 1    ###1              # For the PSE OE and SI maps etc, plot set colourbar limits (**same** for all subjects).
plot_clims_multi = 0  ###0              # For the PSE OE and SI maps etc, plot colourbar limits scaled by multiplying factors (different for each subject).
plot_FreqRMS_vs_NumC = 0 ###1            # Plot the number of components vs the Frequency RMS value.

# # # # # Show plots and/or save plots:
# 0/1 for Y/N.
save_plots = 1
showplot = 0




# Frequency spectra: scaling plotting options - will be plotted if plot_all_component_FreqScaled = 1.
# freq_spectra_plotting_options = a list of 1/0 for Y/N for plotting:
# - [0] frequency spectra scaled by the mean OE gas cycling frequency *amplitude*;
# - [1] frequency spectra scaled by the maximum frequency *amplitude* (per spectra);
# - [2] unscaled - as output by ICA - but could have any scaling.
# - [3] frequency spectra scaled by the mean frequency *amplitude* (per spectra); 
# - [4] frequency spectra scaled by the median frequency *amplitude* (per spectra); 
plot_freq_scaled_by_OE = 0
plot_freq_scaled_by_max_freq = 1
plot_freq_unscaled_ICAoutput = 0
plot_freq_scaled_by_mean = 0
plot_freq_scaled_by_median = 0
freq_spectra_plotting_options = [plot_freq_scaled_by_OE, plot_freq_scaled_by_max_freq, plot_freq_unscaled_ICAoutput, \
    plot_freq_scaled_by_mean, plot_freq_scaled_by_median]


# # # # # Save HDF5 and csv files:
# 0/1 for Y/N.
# Automatically saves subject-specific HDF5 file output after running ICA.
# Save subject-wide PSE information:
save_PSE_subject_wide = 1##1 #1
# Save/store HDF5 info of PCA *and* ICA:
save_ordering_HDF5 = 0
# Save comparison between ICA and PCA in a **csv** file:
save_csv_PCAICA = 0 # Not using any more, see few later...
# Save the PSE stats for a specific metric to a **csv** file (otherwise stored in a subject-wide HDF5 file, see save_PSE_subject_wide from above):
save_csv_PSE_a_metric = 1 #1
# # ^^ After running for different subjects, then set to 1 to stop creating for each subject/repetition.
save_csv_OE_metric_ICA = 0 # Save the OE component metric value to a csv file for ICA components for subjects in the subjectID_list.







# # # # # File locations:

# ** MRI scan data directory
# For the directory containing the MRI images (dir_images) and the masks (dir_images_masks)
# as follows...
# dir_images = dir_base + dir_date + '/' + dir_date + '_' + dir_subj + dir_end + dir_gas + '/' + dir_image_ending + '/'
# dir_images_mask = dir_base + dir_date + '/' + dir_date + '_' + dir_subj + dir_end + dir_gas + '/' + dir_mask_ending + '/'

dir_gas = 'Cycled3' # Gas cycling
dir_base = 'C:/Documents/MRI_Scanning/OE_MR_'
dir_end = '/DICOM/'
dir_image_ending = 'reg'
dir_mask_ending = 'Masks_reg'
# dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
# dir_images_mask = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[4] + '/'
regCorr_name = '_vedit_alphaedit'  # Name of the density correction is applied.
dir_image_nifty = '_nifty'

# # ** Image/figure saving directory
# The directory where plots are to be saved.
# image_saving_directory = image_save_loc_general + image_dir_name + date + '/'
# Within this directory, subject-specific directories: dir_date + '_' + dir_subj + '/'
image_save_loc_general = 'C:/Documents/ICA_Images/'
image_dir_name = 'images_ComponentAnalysis_'

# # ** File names and locations of: mask dictionary and HDF5 files.
# Location of csv files to be saved:
file_loc = 'C:/Documents/ICA_Outputs/'
hdf5_loc = "C:/Documents/MRI_Scanning/OE_MR_HDF5_structures/"

hdf5_file_name_base = 'hdf5_structure_'

# Dictionary location (contains the names of the image masks for each subject):
dictionary_loc = 'C:/Documents/ICA_Coregistration/'
# Full dictionary name - subject info and names of masks (cardiac and lung)
dictionary_name = dictionary_loc + '_masks_HealthyVolunteers.json'

# Name of the HDF5 structure to contain the PSE stats across all subjects:
PSE_subjectWide_base = "PSE_stats_HDF5_structure"
subject_wide_HDF5_name = PSE_subjectWide_base + "_HealthyVolunteers_SubjectWide"

# Name of the csv file to contain the PSE stats for a particular 
PSE_file_wrtCycle1_start = 'PSE_wrtCycle1_'
PSE_file_wrtCycle1_ending = '__HealthyVolunteers'

# Name of the csv file containing comparisons of ICA to PCA:
file_name='Comparison_ICA_PCA_metrics_numC'
subject_wide_PCA_HDF5_file_name='PCA_PSE__HealthyVolunteers'



# # # # # Further plotting parameters:
# # Upper and lower colourbar limits for subject-wide colourbar scaled for PSE map plotting (reconstructed OE ICA component and MRI SI data):
clim_lower = [35, 30, 25, 20] # Negative of this PSE will be used.
clim_upper = [5]
# For plotting ICA component timeseries and frequency spectra, the total number of subplots per figure,
# and the number of subplots to be contained within each row can be specified.
# Needs to be in the form of a list - all numbers in list will be plotted.
subplots_num_in_figure_list = [16]
subplots_num_row_list = [4]
# # For plotting component maps don't plot all within the list. Choose a large number with equal number of subplots in the rows and columns.
subplots_num_in_figure = 16
subplots_num_row = 4

# # # # # Colourbar limits and multipliers for overlay plotting:
# Will loop over, if not plotting, leave as empty.
# Multipliers to PSE overlay plots.
clims_multi_lower_list = [0.8, 0.5, 0.4]
clims_multi_upper_list = [0.25, 0.2, 0.1]
# Upper and lower limits of PSE overlay plots.
clims_lower_list = [25, 20, 25, 20, 15, 20, 25]
clims_upper_list = [5, 5, 3, 3, 15, 20, 25]
# How many figures to plot containing component subplots:
num_figures_plot_components = 3
# Multipliers to ICA component overlay plots (symmetrical).
clims_multi_componentOverlay_list = [0.7, 0.4]
plot_PSE_yaxis_limited = 1 # Whether to plot the y-axis of the PSE timeseries with a set range.
PSE_yaxis_lower = -15
PSE_yaxis_upper = 5
PSE_yaxis_range = [plot_PSE_yaxis_limited, PSE_yaxis_lower, PSE_yaxis_upper]
# For plotting purposes, number of timepoints of whitespace -ve and +ve sides of time plots:
time_plot_side = 10







plot_all_component_TimeFreq = 0
plot_SI_timeseries = 0
plot_PSE_map = 0###1 #1 # Use if plotting PSE map from OE component (not overlay).
echo_num_no=0; densityCorr=0; ANTs_used=0
ordering_method_list_list=["S_ica_SpearmanCorr"]
invert_metric=1 # Invert metric - Spearman correlation
CA_type_list = ["ICA"]; num_c_use=250
plot_PCA_variances=0              # Plot PCA variance explained vs the number of components.
do_ICAPCA_comparison=0 #0      # Run PCA and compare to ICA. - - leave at 0 if not running PCA.
save_csv_OE_metric_PCA=0 #0 # Save the OE component metric value to a csv file for PCA components for subjects in the subjectID_list.
save_csv_PSE_PCA=0 #0 # Save the OE component PSE stats to a csv file for PCA components for subjects in the subjectID_list.
plot_SI_images=0



# # # # # # # # # # # # # 
# # # # # Lists of arguments:
MRI_scan_directory = [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending]
images_saving_info_list = [image_save_loc_general, image_dir_name]
argument_running = [do_runICA, do_ICA_PSE_calculation, do_ICAPCA_comparison, do_anyPlotting, do_ICA_ordering] # 2022/10/10 - Ordering
what_to_plot = [plot_ReconOE, plot_MRI_SI]
which_plots = [plot_PSE_timeseries, plot_SI_timeseries, plot_PSE_map, plot_OE_ICA, \
    plot_ICA_components, plot_all_component_TimeFreq, plot_all_component_FreqScaled, \
    plot_SI_images, plot_PSE_overlay, plot_OE_component_overlay, plot_clims_set, \
    plot_clims_multi, plot_FreqRMS_vs_NumC, plot_PCA_variances, plot_PSE_overlay_LungMask]
save_show_plots = [showplot, save_plots]
file_details_echo_num = [subject_wide_HDF5_name, hdf5_loc, echo_num_no, dictionary_name]
# file_details_create = [file_loc, PSE_file_wrtCycle1, file_name]
saving_YN = [save_PSE_subject_wide, save_ordering_HDF5, save_csv_PCAICA, save_csv_PSE_a_metric, \
    save_csv_OE_metric_ICA, save_csv_OE_metric_PCA, save_csv_PSE_PCA]
component_plotting_params = [CA_type_list, subplots_num_in_figure_list, subplots_num_row_list, \
    subplots_num_in_figure, subplots_num_row]
plotting_list_argument = [dpi_save, what_to_plot, which_plots, images_saving_info_list, clim_lower, clim_upper]
MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes, Vox1, Vox2]
clims_overlay = [clims_multi_lower_list, clims_multi_upper_list, \
    clims_lower_list, clims_upper_list, clims_multi_componentOverlay_list]
other_parameters = [NumEchoes, num_c_start, num_c_stop, ordering_method_list_list, num_c_use, av_im_num]


# # # # # # # # # # # # # 
# Subjects to be investigated:
# # As a list - loop over

echo_num_list = [1,2]
RunNum_use = 1
j_range = 1

subjectID_list = ['subject_V1']


for j in range(j_range):
    # RunNum_use = RunNum_use_list[j]
    subjectID_list = list(subjectID_list)[1:]
    print(subjectID_list)
    print('Run' + str(RunNum_use))
    # # # # # # # # # # # # # 
    for ordering_method_list in ordering_method_list_list:
        PSE_file_wrtCycle1 = PSE_file_wrtCycle1_start + '_' + ordering_method_list + PSE_file_wrtCycle1_ending
        file_details_create = [file_loc, PSE_file_wrtCycle1, file_name, PSE_file_wrtCycle1_start, PSE_file_wrtCycle1_ending, invert_metric]
        other_parameters = [NumEchoes, num_c_start, num_c_stop, [ordering_method_list], num_c_use, av_im_num]
        # Run
        for echo_num in echo_num_list:
            file_details_echo_num = [subject_wide_HDF5_name, hdf5_loc, echo_num, dictionary_name]
            #
            if ANTs_used == 0:
                for NiftyRegID in NiftyReg_RegIDs:
                    print(NiftyRegID)
                    functions_RunICA_andPlot__NiftyReg_2022_08_26.run_ICA_PCA_full(subjectID_list, \
                        saving_YN, other_parameters, file_details_echo_num, file_details_create, \
                        component_plotting_params, plotting_list_argument, \
                        argument_running, save_show_plots, \
                        MRI_scan_directory, MRI_params, clims_overlay, \
                        NiftyRegID, densityCorr, regCorr_name, \
                        ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                        freq_spectra_plotting_options, RunNum_use, g_num, num_figures_plot_components, \
                        subject_wide_PCA_HDF5_file_name, PSE_yaxis_range)
                    print(NiftyRegID + '   ' + str(echo_num) + '   ' + ordering_method_list)
            else:
                # Any NiftyReg ID can be used.
                functions_RunICA_andPlot__NiftyReg_2022_08_26.run_ICA_PCA_full(subjectID_list, \
                    saving_YN, other_parameters, file_details_echo_num, file_details_create, \
                    component_plotting_params, plotting_list_argument, \
                    argument_running, save_show_plots, \
                    MRI_scan_directory, MRI_params, clims_overlay, \
                    'cpg3_be0p005', densityCorr, regCorr_name, \
                    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                    freq_spectra_plotting_options, RunNum_use, g_num, num_figures_plot_components, \
                    subject_wide_PCA_HDF5_file_name, PSE_yaxis_range)
                print('ANTs' + '   ' + str(echo_num) + '   ' + ordering_method_list)