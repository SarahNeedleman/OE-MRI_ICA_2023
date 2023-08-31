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

# Functions for plotting ICA components/reconstructed components, MRI SI 
# parameters and other outputs of the pipeline.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from datetime import date
import os
import h5py
import Sarah_ICA
import functions_PlottingICAetc_testing_Subplots
import functions_Calculate_stats
import functions_loadHDF5_ordering
import functions_preproc_MRI_data
import functions_loadHDF5_ordering__NiftyReg_2022_08_26

def HDF5_Loop_CalculatePSEwrtCycle1__plotting_reconOE_PSE(hdf5_loc, dir_date, dir_subj, \
    metric_chosen, av_im_num, dictionary_name, \
    echo_num, \
    save_plots, showplot, \
    clim_lower, clim_upper, \
    dpi_save, \
    what_to_plot, which_plots, \
    images_saving_info_list, MRI_scan_directory, \
    subplots_num_in_figure, subplots_num_row, \
    MRI_params, clims_overlay, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use, g_num, other_parameters, \
    PSE_yaxis_range, plot_mean_median):
    """
    hdf5_loc = file_details_echo_num[1]
    metric_chosen = other_parameters[3]
    av_im_num = other_parameters[5]
    dictionary_name = file_details_echo_num[3]
    echo_num = file_details_echo_num[2]
    save_plots = save_show_plots[1]
    showplot = save_show_plots[0]
    clim_lower = plotting_list_argument[4]
    clim_upper = plotting_list_argument[5]
    dpi_save = plotting_list_argument[0]
    what_to_plot = plotting_list_argument[1]
    which_plots = plotting_list_argument[2]
    images_saving_info_list = plotting_list_argument[3]
    subplots_num_in_figure = component_plotting_params[3]
    subplots_num_row = component_plotting_params[4]

    Function to plot:
    - Plot PSE timeseries.
    - Plot SI timeseries (MRI SI data and reconstructed OE component).
    - Plot frequency spectra and timeseries of the OE component.
    - Plot PSE maps.
    - Plot all component maps.
    for OE component (reconstructed) and MRI SI data.
    - Plot components/PSE overlay on SI images.

    Arguments:
    hdf5_loc = location of the HDF5 files.
    dir_date = subject scanning date.
    dir_subj = subject ID.
    metric_chosen = frequency RMS ordering metric used - as a list.
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    dictionary_name = name and location of the dictionary containing the masks and the 
    names of the reference images for the particular subject(s).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    save_plots = 0 if No; 1 if Yes - whether to save plots.
    showplot = 0 if No; 1 if Yes - whether to show plots.
    clim_lower = lower limit of the PSE map plots.
    clim_upper = upper limit of the PSE map plots.
    # # FOR PSE MAP PLOTTING, e.g.
    # clim_lower = [35, 30, 25, 20] # Negative of this
    # clim_upper = [5]
    # Used for subject-wide PSE colourbars.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    what_to_plot = [plot_ReconOE, plot_MRI_SI] - 0/1 in list for Yes/No to plot the
    reconstructed OE component plots and/or the MRI SI data.
    which_plots = [plot_PSE_timeseries, plot_SI_timeseries, plot_PSE_map, plot_OE_ICA, plot_ICA_components, \
    plot_all_component_TimeFreq, plot_all_component_FreqScaled, plot_SI_images, plot_PSE_overlay, \
    plot_OE_component_overlay, plot_clims_set, plot_clims_multi] - 0/1 in list for Yes/No
    to plot some of the plots contained within this function.
    images_saving_info_list = details of where to save all images. As a 
    list with values [image_save_loc_general, image_dir_name].
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes]
    clims_overlay = [clims_multi_lower_list, clims_multi_upper_list, \
        clims_lower_list, clims_upper_list, clims_multi_componentOverlay_list]
    RunNum_use = specific RunNum to use, e.g. 1.
    g_num = number of components greater than when ordering (e.g. 21 for 22 and greater).
    other_parameters = for details of the lower and upper number of components to consider
    when extracting the components.
    PSE_yaxis_range = [plot_PSE_yaxis_limited, PSE_yaxis_lower, PSE_yaxis_upper] = for
    if plotting a set y-axis range on the PSE timeseries plots.
    # See script version for details of the argument lists.
    Returns:
    None - plots saved and CSV/HDF5 files created as required.

    # # # Here, subplot numbers for *component maps* only.
    # As component maps, plot 20 subplots per figure
    # ...
    # # subplots_num_in_figure & subplots_num_row info:
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row

    """
    # Directory for saving subject-specific plots.
    today = date.today()
    d1 = today.strftime("%Y_%m_%d")

    # Directory - subject specific.
    image_saving_dir_subjSpec = images_saving_info_list[0] + images_saving_info_list[1] + d1 + '/' + \
        dir_date + '_' + dir_subj + '/'
    image_saving_pre_subjSpec = image_saving_dir_subjSpec + \
        dir_date + '_' + dir_subj + '_'

    if save_plots == 1:
        # Check to see if directory exists, if not, create one for the
        # date of plotting and subject.
        if os.path.isdir(images_saving_info_list[0] + images_saving_info_list[1] + d1 + '/') == False:
            os.mkdir(images_saving_info_list[0] + images_saving_info_list[1] + d1)
        #
        if os.path.isdir(image_saving_dir_subjSpec) == False:
            # Make the directory
            os.mkdir(image_saving_dir_subjSpec[:-1])

    plotting_param = 'SI_regEcho' + str(echo_num)
    component_analysis = 'ICA'
    # Saving details - dir and name
    saving_details = []
    saving_details.append(save_plots)
    # saving_details.append(image_saving_pre_allSubj + component_analysis + 'on') # Instead, save for each subject in dir
    saving_details.append(image_saving_dir_subjSpec + dir_date + '_' + dir_subj + '_' + \
        component_analysis + '_')
    saving_details.append(plotting_param)
    saving_details.append(metric_chosen[0])
    # saving_details.append(metric_chosen)


    # saving_details[2] - append info about reg and registration type.
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # # Add in SingleCycle or Cycled3 identifier
    if MRI_params[3][0] == 1:
        # Cycled3
        saving_detail_add = 'Cycled3'

    saving_details[2] = saving_details[2] + '__' + reg_applied + '__' + reg_type + '__' + saving_detail_add + '_'

    # # # Also, if PSE maps etc are for alternative oxy mean....
    # Do this if len(av_im_num) == 4.
    # And the name to use in the subgroup name is av_im_num[3]
    if MRI_params[3][0] == 0:
        if len(av_im_num) == 4:
            saving_details[2] = saving_details[2] + av_im_num[3]

    # Remove ICA references in saving_details list for SI plots.
    saving_details_SI = saving_details[:]
    saving_details_SI[1] = saving_details_SI[1][:-4]

    # Initialise OE component number
    OE_component_number = None

    # Call generate_freq - to generate frequencies and timepoints for plotting.
    # Use plot_frequencies for plotting.
    freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(MRI_params[2], MRI_params[0])
    
    if MRI_params[3][0] == 1:
        # Cycled 3 PSE calculation
        stats_eachSlice_all_lung, stats_ALLSlice_all_lung, PSE_timeseries_withinLung_EachSlice, PSE_timeseries_withinLung_ALLSlice, \
            maskedMaps_lung, maskedMaps_lung_MeanEachSlice, maskedMaps_lung_MeanALLSlice, PSE_image_OverallandCycled, load_data_echo1_regOnly, \
            sorted_array_metric_sorted, sorted_array, S_ica_array, freq_S_ica_array, masks_reg_data_cardiac, stats_list \
            = PSE_Calculation__CalculatePSE_Cycled3(\
            hdf5_loc, dir_date, dir_subj, \
            metric_chosen, dictionary_name, \
            echo_num, MRI_scan_directory, MRI_params, \
            NiftyReg_ID, densityCorr, regCorr_name, \
            ANTs_used, dir_image_nifty, hdf5_file_name_base, \
            RunNum_use, g_num, other_parameters)


    # "Unpack" the returned lists to use for when plotting etc... 
    stats_eachSlice_PSE_SI_all_lung = stats_eachSlice_all_lung[0]
    stats_eachSlice_PSE_recon_all_lung = stats_eachSlice_all_lung[1]

    stats_ALLSlice_PSE_SI_all_lung = stats_ALLSlice_all_lung[0]
    stats_ALLSlice_PSE_recon_all_lung = stats_ALLSlice_all_lung[1]

    # MEDIAN as well as mean
    # ALLSlice as well as each slice
    PSE_timeseries_SI_lungMask_MeanEachSlice = PSE_timeseries_withinLung_EachSlice[0]
    PSE_timeseries_SI_lungMask_MedianEachSlice = PSE_timeseries_withinLung_EachSlice[1]
    PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice = PSE_timeseries_withinLung_EachSlice[2]
    PSE_timeseries_recon_unMean_unSc_lungMask_MedianEachSlice = PSE_timeseries_withinLung_EachSlice[3]

    PSE_timeseries_SI_lungMask_MeanALLSlice = PSE_timeseries_withinLung_ALLSlice[0]
    PSE_timeseries_SI_lungMask_MedianALLSlice = PSE_timeseries_withinLung_ALLSlice[1]
    PSE_timeseries_recon_unMean_unSc_lungMask_MeanALLSlice = PSE_timeseries_withinLung_ALLSlice[2]
    PSE_timeseries_recon_unMean_unSc_lungMask_MedianALLSlice = PSE_timeseries_withinLung_ALLSlice[3]

    load_data_echo1_regOnly_maskedMaps_lung = maskedMaps_lung[0]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung = maskedMaps_lung[1]

    # MEDIAN as well as mean
    # ALLSlice as well as each slice
    load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = maskedMaps_lung_MeanEachSlice[0]
    load_data_echo1_regOnly_maskedMaps_lungMask_MedianEachSlice = maskedMaps_lung_MeanEachSlice[1]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice = maskedMaps_lung_MeanEachSlice[2]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MedianEachSlice = maskedMaps_lung_MeanEachSlice[3]

    load_data_echo1_regOnly_maskedMaps_lungMask_MeanALLSlice = maskedMaps_lung_MeanALLSlice[0]
    load_data_echo1_regOnly_maskedMaps_lungMask_MedianALLSlice = maskedMaps_lung_MeanALLSlice[1]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanALLSlice = maskedMaps_lung_MeanALLSlice[2]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MedianALLSlice = maskedMaps_lung_MeanALLSlice[3]


    PSE_image_SI_all = PSE_image_OverallandCycled[0]
    PSE_image_unMean_unSc_all = PSE_image_OverallandCycled[1]

    size_echo_data = np.shape(load_data_echo1_regOnly)


    ######################################################################################################################################

    # # # # # # # # # Plotting

    # Titles/saving details for OE vs SI plotting.
    plotting_param_name_list = ['Recon_OE', 'SI']
    plotting_param_name_list_titles = ['Recon OE', 'MRI SI']
    # plotting_param_chosen use 0 or Recon OE and 1 for MRI SI.
    stat_plotted = ['mean', 'median']

    # Number of components - to include in plotting title
    NumC_title = '_C' + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0])))
    # If SI, use '' instead.

    if plot_mean_median[0] == 1:
        # # # PLOT MEAN
        stat_plotted_string = stat_plotted[0]
        if which_plots[0] == 1:

            # For OE component plotting PSE - within the lung mask.
            if what_to_plot[0] == 1:
                PSE_timeseries_ALLSlice = np.array(PSE_timeseries_recon_unMean_unSc_lungMask_MeanALLSlice)
                plotting_param_chosen = 0
                plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
                    PSE_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use, stat_plotted_string)
                # Plot with limited y-axis range?
                if PSE_yaxis_range[0] == 1:
                    plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
                        PSE_timeseries_ALLSlice, \
                        plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                        saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
                        PSE_yaxis_range, RunNum_use, stat_plotted_string)
                #
                # Plot frequency spectrum, scaled by the maximum frequency
                #     
                freq_Plot_initial = np.squeeze(Sarah_ICA.freq_spec(1, np.transpose(np.array([PSE_timeseries_ALLSlice]), [1,0])))
                # 20230811 - Set zero to zero, then calculate max from :halfTimepoints
                freq_Plot_initial[0] = 0
                max_scaling_freq_Plot_initial = np.max(np.abs(freq_Plot_initial[:halfTimepoints]))
                freq_Plot_scaled = np.divide(freq_Plot_initial, max_scaling_freq_Plot_initial)
                plot_singleMetric_PSEtimeseries_EditPlot__FreqVersion_Updated(freqPlot, echo_num, \
                    freq_Plot_scaled, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
                    RunNum_use, halfTimepoints, stat_plotted_string)

            # For MRI SI PSE plotting - within the lung mask.
            if what_to_plot[1] == 1:
                PSE_timeseries_ALLSlice = np.array(PSE_timeseries_SI_lungMask_MeanALLSlice)
                plotting_param_chosen = 1
                plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
                    PSE_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details_SI, showplot, dpi_save, metric_chosen, '', 0, stat_plotted_string)
                # Plot with limited y-axis range?
                if PSE_yaxis_range[0] == 1:
                    plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
                    PSE_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details_SI, showplot, dpi_save, metric_chosen, '', \
                    PSE_yaxis_range, 0, stat_plotted_string)
                #
                # Plot frequency spectrum, scaled by the maximum frequency
                #     
                freq_Plot_initial = np.squeeze(Sarah_ICA.freq_spec(1, np.transpose(np.array([PSE_timeseries_ALLSlice]), [1,0])))
                # 20230811 - Set zero to zero, then calculate max from :halfTimepoints
                freq_Plot_initial[0] = 0
                max_scaling_freq_Plot_initial = np.max(np.abs(freq_Plot_initial[:halfTimepoints]))
                freq_Plot_scaled = np.divide(freq_Plot_initial, max_scaling_freq_Plot_initial)
                plot_singleMetric_PSEtimeseries_EditPlot__FreqVersion_Updated(freqPlot, echo_num, \
                    freq_Plot_scaled, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, '', 0, halfTimepoints, stat_plotted_string)

        # # # MEAN SI TIMESERIES PLOTS:
        if which_plots[1] == 1:
            # For OE component timeseries plotting - within the lung mask.
            if what_to_plot[0] == 1:
                param_timeseries_ALLSlice = np.array(X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanALLSlice)
                plotting_param_chosen = 0
                plot_singleMetric_parametertimeseries_EditPlot(plot_time_value, echo_num, \
                    param_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use, stat_plotted_string)
            # For MRI SI plotting - within the lung mask.
            if what_to_plot[1] == 1:
                param_timeseries_ALLSlice = np.array(load_data_echo1_regOnly_maskedMaps_lungMask_MeanALLSlice)
                plotting_param_chosen = 1
                plot_singleMetric_parametertimeseries_EditPlot(plot_time_value, echo_num, \
                    param_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details_SI, showplot, dpi_save, metric_chosen, '', 0, stat_plotted_string)


    if plot_mean_median[1] == 1:
        # # # PLOT MEDIAN
        stat_plotted_string = stat_plotted[1]
        if which_plots[0] == 1:
            # For OE component PSE plotting - within the lung mask.
            if what_to_plot[0] == 1:
                PSE_timeseries_ALLSlice = np.array(PSE_timeseries_recon_unMean_unSc_lungMask_MedianALLSlice)
                plotting_param_chosen = 0
                plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
                    PSE_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use, stat_plotted_string)
                # Plot with limited y-axis range?
                if PSE_yaxis_range[0] == 1:
                    plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
                        PSE_timeseries_ALLSlice, \
                        plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                        saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
                        PSE_yaxis_range, RunNum_use, stat_plotted_string)
                #
                # Plot frequency spectrum, scaled by the maximum frequency
                #     
                freq_Plot_initial = np.squeeze(Sarah_ICA.freq_spec(1, np.transpose(np.array([PSE_timeseries_ALLSlice]), [1,0])))
                # 20230811 - Set zero to zero, then calculate max from :halfTimepoints
                freq_Plot_initial[0] = 0
                max_scaling_freq_Plot_initial = np.max(np.abs(freq_Plot_initial[:halfTimepoints]))
                freq_Plot_scaled = np.divide(freq_Plot_initial, max_scaling_freq_Plot_initial)
                plot_singleMetric_PSEtimeseries_EditPlot__FreqVersion_Updated(freqPlot, echo_num, \
                    freq_Plot_scaled, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
                    RunNum_use, halfTimepoints, stat_plotted_string)

            # For MRI SI PSE plotting - within the lung mask.
            if what_to_plot[1] == 1:
                PSE_timeseries_ALLSlice = np.array(PSE_timeseries_SI_lungMask_MedianALLSlice)
                plotting_param_chosen = 1
                plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
                    PSE_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details_SI, showplot, dpi_save, metric_chosen, '', 0, stat_plotted_string)
                # Plot with limited y-axis range?
                if PSE_yaxis_range[0] == 1:
                    plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
                    PSE_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details_SI, showplot, dpi_save, metric_chosen, '', \
                    PSE_yaxis_range, 0, stat_plotted_string)
                #
                # Plot frequency spectrum, scaled by the maximum frequency
                #     
                freq_Plot_initial = np.squeeze(Sarah_ICA.freq_spec(1, np.transpose(np.array([PSE_timeseries_ALLSlice]), [1,0])))
                # 20230811 - Set zero to zero, then calculate max from :halfTimepoints
                freq_Plot_initial[0] = 0
                max_scaling_freq_Plot_initial = np.max(np.abs(freq_Plot_initial[:halfTimepoints]))
                freq_Plot_scaled = np.divide(freq_Plot_initial, max_scaling_freq_Plot_initial)
                plot_singleMetric_PSEtimeseries_EditPlot__FreqVersion_Updated(freqPlot, echo_num, \
                    freq_Plot_scaled, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, '', 0, halfTimepoints, stat_plotted_string)



        # # # MEAN SI TIMESERIES PLOTS:
        if which_plots[1] == 1:
            # For OE component timeseries plotting - within the lung mask.
            if what_to_plot[0] == 1:
                param_timeseries_ALLSlice = np.array(X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MedianALLSlice)
                plotting_param_chosen = 0
                plot_singleMetric_parametertimeseries_EditPlot(plot_time_value, echo_num, \
                    param_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use, stat_plotted_string)
            # For MRI SI plotting - within the lung mask.
            if what_to_plot[1] == 1:
                param_timeseries_ALLSlice = np.array(load_data_echo1_regOnly_maskedMaps_lungMask_MedianALLSlice)
                plotting_param_chosen = 1
                plot_singleMetric_parametertimeseries_EditPlot(plot_time_value, echo_num, \
                    param_timeseries_ALLSlice, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    saving_details_SI, showplot, dpi_save, metric_chosen, '', 0, stat_plotted_string)


    # # # OE COMPONENT TIMESERIES AND FREQUENCY SPECTRUM PLOTS:
    if which_plots[3] == 1:
        parameter_spectrum = np.array(freq_S_ica_array)
        plotting_param_chosen = 0
        plot_singleMetric_parameter_TimeAndFrequency(plot_time_value, freqPlot, echo_num, \
            np.squeeze(S_ica_array), np.abs(parameter_spectrum), halfTimepoints, \
            plotting_param_name_list[plotting_param_chosen], \
            saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use)

    # # # PSE MAP PLOTS:
    if which_plots[2] == 1:

        # For OE component plotting.
        if what_to_plot[0] == 1:
            PSE_image_plot = np.array(PSE_image_unMean_unSc_all)
            plotting_param_chosen = 0
            # Plot colourbar limits set, multiplied, or both versions.
            if which_plots[11] == 1:
                # Multi
                plot_singleMetric_PSEmaps_SIandRecon_EditPlot(PSE_image_plot, \
                MRI_params[1], saving_details, showplot, dpi_save, \
                plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                metric_chosen, NumC_title, RunNum_use, \
                clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25])
            
            if which_plots[10] == 1:
                # Set
                plot_singleMetric_PSEmaps_SIandRecon_setCBAR_EditPlot(PSE_image_plot, \
                    MRI_params[1], saving_details, showplot, dpi_save, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    metric_chosen, NumC_title, RunNum_use, \
                    clims_overlay, \
                    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25])

        # For MRI SI plotting.
        if what_to_plot[1] == 1:
            PSE_image_plot = np.array(PSE_image_SI_all)
            plotting_param_chosen = 1

            # Plot colourbar limits set, multiplied, or both versions.
            if which_plots[11] == 1:
                plot_singleMetric_PSEmaps_SIandRecon_EditPlot(PSE_image_plot, \
                    MRI_params[1], saving_details_SI, showplot, dpi_save, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    metric_chosen, '', RunNum_use, \
                    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25])

            if which_plots[10] == 1:
                plot_singleMetric_PSEmaps_SIandRecon_setCBAR_EditPlot(PSE_image_plot, \
                    MRI_params[1], saving_details_SI, showplot, dpi_save, \
                    plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                    metric_chosen, '', RunNum_use, \
                    clims_overlay, \
                    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25])


    # Extraction of all of the ICA components for the
    # particular run of ICA containing the OE component.
    recon_all_components, ordered_S_ica, ordered_A_ica, ordered_A_ica_reshape = \
        ICAcomponent_LoadandRecon_nifty(size_echo_data, hdf5_loc, dir_date, dir_subj,\
        echo_num, sorted_array_metric_sorted, metric_chosen, \
        MRI_params[1], MRI_params[2], masks_reg_data_cardiac, \
        NiftyReg_ID, densityCorr, regCorr_name, \
        ANTs_used, dir_image_nifty, hdf5_file_name_base, \
        RunNum_use)
    del recon_all_components, ordered_S_ica

    # # # PLOTTING OF ALL ICA COMPONENTS:
    # Plotting the component maps of all ICA components extracted.
    if which_plots[4] == 1:
        # Plotting of all of the ICA components.
        # Here, cmap = 'bwr'
        plot_maps_components_EditPlot(ordered_A_ica_reshape, np.shape(ordered_A_ica_reshape)[0], saving_details, showplot, \
            dpi_save, RunNum_use, 'bwr', subplots_num_in_figure, subplots_num_row, clims_multi=0.75)

        
    # # # PLOTTING OVERLAYS:
    # Alternative to plotting PSE maps etc...
    if which_plots[8] == 1:
        import functions_Plot_Overlay

        # For OE component plotting.
        if what_to_plot[0] == 1:
            plotting_param_chosen = 0
            functions_Plot_Overlay.plot_Map_Overlay_onSI_Scaling(load_data_echo1_regOnly, np.squeeze(PSE_image_unMean_unSc_all[0,:,:,:]), dir_date, dir_subj, \
                MRI_params[1], echo_num, NumC_title, metric_chosen, \
                plotting_param_name_list[plotting_param_chosen]+'_PSE', masks_reg_data_cardiac, \
                save_plots, showplot, dpi_save, \
                saving_details, which_plots, \
                clims_overlay[0], clims_overlay[1], \
                clims_overlay[2], clims_overlay[3], \
                RunNum_use)

        # For SI PSE plotting.
        if what_to_plot[1] == 1:
            plotting_param_chosen = 1
            functions_Plot_Overlay.plot_Map_Overlay_onSI_Scaling(load_data_echo1_regOnly, np.squeeze(PSE_image_SI_all[0,:,:,:]), dir_date, dir_subj, \
                MRI_params[1], echo_num, '', metric_chosen, \
                plotting_param_name_list[plotting_param_chosen]+'_PSE', masks_reg_data_cardiac, \
                save_plots, showplot, dpi_save, \
                saving_details, which_plots, \
                clims_overlay[0], clims_overlay[1], \
                clims_overlay[2], clims_overlay[3], \
                0)

    # Alternative to plotting OE component map.
    # Plot as an overlay.
    # Clims_multi for the component map.
    if which_plots[9] == 1:
        import functions_Plot_Overlay
        functions_Plot_Overlay.plot_Map_Overlay_onSI_Scaling_ComponentOnly(load_data_echo1_regOnly, np.squeeze(ordered_A_ica_reshape[0,:,:,:]), dir_date, dir_subj, \
            MRI_params[1], echo_num, NumC_title, metric_chosen, \
            'OEc', masks_reg_data_cardiac, \
            save_plots, showplot, dpi_save, \
            saving_details, \
            clims_overlay[4], \
            RunNum_use)

    # Plot Frequency RMS vs the number of components (ICA only).
    if which_plots[12] == 1:
        functions_loadHDF5_ordering.plot_componentNum_vs_metricVal_wrtCycle1(\
            sorted_array, metric_chosen, RunNum_use, \
            saving_details, showplot, dpi_save)

    
    return






def plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
    PSE_timeseries_ALLSlice, \
    plotting_param_name, plotting_param_title, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
    RunNum_use, stat_plotted_string):
    """
    Plot mean PSE (mean PSE within lung mask over all slices). Can be for any input array
    e.g. SI or reconstructed OE component.
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries.

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    # PSE_timeseries_EachSlice = timeseries of the mean PSE value within
    # the lung masks for each slice of the SI PSE. Shape (NumDyn, NumSlices).
    PSE_timeseries_ALLSlice --> now assuming ALLSlice already, so (NumDyn,) only. 20230809.
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    Returns:
    None, figure plotted and saved if required.

    """
    # # Calculate mean PSE over all 4 slices
    # PSE_timeseries_EachSlice_overall = np.mean(PSE_timeseries_EachSlice, axis=1)
    # NO, assuming already over ALL slices

    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # # Plot timeseries of PSE and single slice.
    fig1, (ax5) = plt.subplots(1, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax5.plot(plot_time_value, PSE_timeseries_ALLSlice)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel('PSE (%)')
    ax5.set_title(stat_plotted_string + plotting_param_title + " PSE over all slices, echo " + str(np.int32(np.rint(echo_num))))

    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_PSEtimeseries___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_PSEtimeseries___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    



def plot_singleMetric_PSEtimeseries_EditPlot__FreqVersion(plot_freq_value, echo_num, \
    PSE_freq_spectra, \
    plotting_param_name, plotting_param_title, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
    RunNum_use, halfTimepoints):
    """
    Plot mean PSE (mean PSE within lung mask over all slices). Can be for any input array
    e.g. SI or reconstructed OE component.
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries.

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    PSE_timeseries_EachSlice = timeseries of the mean PSE value within
    the lung masks for each slice of the SI PSE. Shape (NumDyn, NumSlices).
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    Returns:
    None, figure plotted and saved if required.

    """
    # Frequency spectrum - plot positive frequencies only
    # parameter_spectrum = np.squeeze(PSE_freq_spectra)
    parameter_spectrum = np.abs(PSE_freq_spectra[0:halfTimepoints])


    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # # Plot timeseries of PSE and single slice.
    fig1, (ax5) = plt.subplots(1, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax5.plot(plot_freq_value, parameter_spectrum)
    ax5.set_xlabel("Frequency (Hz)"); ax5.set_ylabel('Amplitude (a.u.)')
    ax5.set_title(plotting_param_title + " PSE frequency spectrum , echo " + str(np.int32(np.rint(echo_num))))

    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_PSEFreq___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_PSEfreq___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    


# Updated - median etc argument
def plot_singleMetric_PSEtimeseries_EditPlot__FreqVersion_Updated(plot_freq_value, echo_num, \
    PSE_freq_spectra, \
    plotting_param_name, plotting_param_title, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
    RunNum_use, halfTimepoints, stat_plotted_string):
    """
    Plot mean PSE (mean PSE within lung mask over all slices). Can be for any input array
    e.g. SI or reconstructed OE component.
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries.

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    PSE_timeseries_EachSlice = timeseries of the mean PSE value within
    the lung masks for each slice of the SI PSE. Shape (NumDyn, NumSlices).
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    Returns:
    None, figure plotted and saved if required.

    """
    # Frequency spectrum - plot positive frequencies only
    # parameter_spectrum = np.squeeze(PSE_freq_spectra)
    parameter_spectrum = np.abs(PSE_freq_spectra[0:halfTimepoints])


    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # # Plot timeseries of PSE and single slice.
    fig1, (ax5) = plt.subplots(1, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax5.plot(plot_freq_value, parameter_spectrum)
    ax5.set_xlabel("Frequency (Hz)"); ax5.set_ylabel('Amplitude (a.u.)')
    ax5.set_title(plotting_param_title + " PSE frequency spectrum , echo " + str(np.int32(np.rint(echo_num))))

    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_PSEFreq___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_PSEfreq___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    


def plot_singleMetric_parametertimeseries_EditPlot(plot_time_value, echo_num, \
    parameter_timeseries_ALLSlice, \
    plotting_param_name, plotting_param_title, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use, stat_plotted_string):
    """
    Plot the mean SI (mean within the lung mask over all slices). Can be for any input array
    e.g. SI or reconstructed OE component.
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries.

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    # parameter_timeseries_EachSlice = timeseries of the mean SI value within
    # the lung masks for each slice of the SI. Shape (NumDyn, NumSlices).
    parameter_timeseries_ALLSlice --> now assuming ALLSlice already, so (NumDyn,) only. 20230809.
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    Returns:
    None, figure plotted and saved if required.

    """
    # # Calculate mean parameter over all 4 slices
    # parameter_timeseries_EachSlice_overall = np.mean(parameter_timeseries_ALLSlice, axis=1)

    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # Plot timeseries of PSE as 4x2 subplots.
    fig1, (ax5) = plt.subplots(1, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax5.plot(plot_time_value, parameter_timeseries_ALLSlice)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel('SI (a.u.)')
    ax5.set_title(stat_plotted_string + plotting_param_title + " over all slices, echo " + str(np.int32(np.rint(echo_num))))

    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_timeseries___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_timeseries___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '.png', dpi=dpi_save, bbox_inches='tight')
    
    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    


def plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
    PSE_timeseries_ALLSlice, \
    plotting_param_name, plotting_param_title, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title, \
    PSE_yaxis_range, RunNum_use, stat_plotted_string):
    """
    Plot mean PSE (mean PSE within lung mask over all slices). Can be for any input array
    e.g. SI or reconstructed OE component.
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries.

    ** With a set y-axis range**

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    # PSE_timeseries_EachSlice = timeseries of the mean PSE value within
    # the lung masks for each slice of the SI PSE. Shape (NumDyn, NumSlices).
    PSE_timeseries_ALLSlice --> now assuming ALLSlice already, so (NumDyn,) only. 20230809.
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    PSE_yaxis_range = [plot_PSE_yaxis_limited, PSE_yaxis_lower, PSE_yaxis_upper] = for
    if plotting a set y-axis range on the PSE timeseries plots.
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # # Plot timeseries of PSE and single slice.
    fig1, (ax5) = plt.subplots(1, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax5.plot(plot_time_value, PSE_timeseries_ALLSlice)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel('PSE (%)')
    ax5.set_title(stat_plotted_string + plotting_param_title + " PSE over all slices, echo " + str(np.int32(np.rint(echo_num))))
    ax5.set_ylim(PSE_yaxis_range[1], PSE_yaxis_range[2])


    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_PSEtimeseries___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '_ylim_' + str(PSE_yaxis_range[1]) + '_' + str(PSE_yaxis_range[2]) + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_PSEtimeseries___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '_' + stat_plotted_string + '_ylim_' + str(PSE_yaxis_range[1]) + '_' + str(PSE_yaxis_range[2]) + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    


def plot_singleMetric_parameterFrequencySpectrum(freqPlot, echo_num, \
    parameter_spectrum, halfTimepoints, \
    plotting_param_name, plotting_param_title, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title):
    """
    Plot the frequency spectrum.
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries.

    Arguments:
    freqPlot = x-axis values for the positive frequencies timepoints, shape 
    of (halfTimepoints,).
    echo_num = the echo number being investigated.
    parameter_spectrum = frequency spectrum to be plotted. With shape (420,1) which includes the
    positive and negative frequencies. halfTimepoints will be used for indexing so that
    only the positive frequencies are plotted. (np.squeeze)
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    Returns:
    None, figure plotted and saved if required.

    """
    # Frequency spectrum - plot positive frequencies only
    parameter_spectrum = np.squeeze(parameter_spectrum)
    parameter_spectrum = np.abs(parameter_spectrum[0:halfTimepoints])

    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # Plot frequency spectra.
    fig1, (ax5) = plt.subplots(1, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax5.plot(freqPlot, parameter_spectrum); ax5.set_xlabel("Frequency (Hz)")
    ax5.set_title("Frequency spectrum, echo " + str(np.int32(np.rint(echo_num))))
    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        fig1.savefig(saving_details[1] + saving_details[2] + '_FreqSpectrum___' + \
            metric_chosen[0] + NumC_title + \
            plotting_param_name + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    

    
def plot_singleMetric_parameter_TimeAndFrequency(plot_time_value, freqPlot, echo_num, \
    S_ica_array, parameter_spectrum, halfTimepoints, \
    plotting_param_name, \
    saving_details, showplot, dpi_save, metric_chosen, NumC_title, RunNum_use):
    """
    Plot the timeseries and frequency spectrum. 
    Based on functions_PlottingICAetc_testing_Subplots.plot_singleMetric_PSEtimeseries

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    freqPlot = x-axis values for the positive frequencies timepoints, shape 
    of (halfTimepoints,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    S_ica_array = parameter_timeseries_EachSlice = timeseries of the mean SI value within
    the lung masks **across all slices**. Shape (NumDyn,).
    parameter_spectrum = frequency spectrum to be plotted. With shape (420,1) which includes the
    positive and negative frequencies. halfTimepoints will be used for indexing so that
    only the positive frequencies are plotted. (np.squeeze)
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    halfTimepoints
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    Returns:
    None, figure plotted and saved if required.

    """
    # Frequency spectrum - plot positive frequencies only
    parameter_spectrum = np.squeeze(parameter_spectrum)
    parameter_spectrum = np.abs(parameter_spectrum[0:halfTimepoints])


    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # Plot frequency spectra.
    fig1, (ax1, ax2) = plt.subplots(2, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    ax1.plot(plot_time_value, S_ica_array); ax1.set_xlabel("Time (s)")
    ax1.set_title("ICA component timeseries, echo " + str(np.int32(np.rint(echo_num))))
    ax2.plot(freqPlot, parameter_spectrum); ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title("Frequency spectrum, echo " + str(np.int32(np.rint(echo_num))))
    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_TimeFreq___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_TimeFreq___' + \
                metric_chosen[0] + NumC_title + \
                plotting_param_name + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    


def plot_singleMetric_PSEmaps_SIandRecon_EditPlot(PSE_image_plot, \
    NumSlices, saving_details, showplot, dpi_save, \
    plotting_param_name, plotting_param_title, \
    metric_chosen, NumC_title, RunNum_use, \
    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25]):
    """
    MULTI
    Plot PSE maps over all NumSlices, using colourbar limits common
    across all subjects.
    Based on plot_singleMetric_PSEmaps_SIandRecon_setCBAR
 
    Arguments:
    PSE_image_plot = PSE image calculated for the MRI registered input data for 
    a specific echo. Calculated for each slice. Shape of 
    (4 for overall and the three cycles, vox, vox, NumSlices).
    NumSlices = number of slices acquired.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    clims_multi = colourbar limits multiplier on maximum value that is plotted.
    vlims_multi = multiplier of the colourbar *range* plotted, multiplying the lower limit
    of the colourbar. Default value = 0.8.
    vlims_multi2 = (possible array, each will be plotted and saved according to the 
    index into the array) multiplier of the colourbar *range* plotted, multiplying the upper limit
    of the colourbar. Default value = [0.5, 0.25].
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # Colourmap scaling
    cmap = cm.bwr

    subplots_num_column = 2 # Assume. These are the number of subplots *per* column - assume 2 subplots per column, i.e. 2 rows.
    subplots_num_row = np.int32(np.ceil(np.divide(NumSlices, 2))) #; subplots_num_row = subplots_num_row[0]

    # # # 1) Plot each slice for recon or SI PSE for the mean oxygen 
    # calculation.
    for k in range(len(vlims_multi2)):
        min_max_plotted = [np.min(PSE_image_plot[0,:,:,:]), np.max(PSE_image_plot[0,:,:,:])]
        min_max_abs = np.max(np.abs(min_max_plotted))
        norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)
        #
        # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
        # Also include the metric, component number used and metric value.
        fig1 = plt.figure(figsize=(19.20,10.80))
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of axes (axs) identifiers to plot the axes in a loop. Also, image identifier (ims).
        axs={}
        ims={}
        # # axes_c = a counter for the axis number to be used when plotting the 
        # # axis titles.
        # axes_c = 0
        # Lopo over slices/axes to be plotted
        for slice_plot in range(NumSlices):
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
            # Subplot number - added 1, in case needs to start from 1 instead of zero.
            # axs[slice_plot] = fig1.add_subplot(subplots_num_row, subplots_num_column, slice_plot+1)
            axs[slice_plot] = fig1.add_subplot(subplots_num_column, subplots_num_row, slice_plot+1)
            ims[slice_plot] = axs[slice_plot].imshow(PSE_image_plot[0,:,:,slice_plot], norm=norm_unC, cmap=cmap)
            axs[slice_plot].invert_yaxis(); fig1.colorbar(ims[slice_plot], ax=axs[slice_plot])
            ims[slice_plot].set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            axs[slice_plot].set_title("Median " + plotting_param_title + " PSE")
        fig1.tight_layout()

        if saving_details[0] == 1:
            if RunNum_use > 1:
                fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_PSE___' + \
                metric_chosen[0] + "_" + NumC_title + '_vlims_L' + str(vlims_multi) + '_U' + str(vlims_multi2[k]) + \
                '_Plot_' + plotting_param_name + '_' + str(k) + '.png', dpi=dpi_save, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + '_PSE___' + \
            metric_chosen[0] + "_" + NumC_title + '_vlims_L' + str(vlims_multi) + '_U' + str(vlims_multi2[k]) + \
            '_Plot_' + plotting_param_name + '_' + str(k) + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_singleMetric_PSEmaps_SIandRecon_setCBAR_EditPlot(PSE_image_plot, \
    NumSlices, saving_details, showplot, dpi_save, \
    plotting_param_name, plotting_param_title, \
    metric_chosen, NumC_title, RunNum_use, \
    clims_overlay, \
    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25]):
    """
    SET 
    Plot PSE maps over all NumSlices, using colourbar limits common
    across all subjects.
    Based on plot_singleMetric_PSEmaps_SIandRecon_setCBAR
 
    Arguments:
    PSE_image_plot = PSE image calculated for the MRI registered input data for 
    a specific echo. Calculated for each slice. Shape of 
    (4 for overall and the three cycles, vox, vox, NumSlices).
    NumSlices = number of slices acquired.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    plotting_param_name = name/ID of the plot to go in the figure name when saved,
    one of ['Recon_OE', 'SI'] - for the parameter to be plotted.
    plotting_param_title = name/ID of the plot to go in the figure title,
    one of ['Recon OE', 'MRI SI'] - for the parameter to be plotted.
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumC_title = number of components to go in title.
    clims_overlay = [clims_multi_lower_list, clims_multi_upper_list, \
        clims_lower_list, clims_upper_list, clims_multi_componentOverlay_list]
    clims_lower_list, clims_upper_list = a list of values to set/threshold the colourmaps
    and colourbars (for use when plotting PSE). The list values will be 
    looped over. e.g. [30, 25, 20] and [5, 5, 5] = clims_lower_list, clims_upper_list.
    - If list is empty, [], these will not be plotted.
    # clim_lower = lower limit of the PSE map plots.
    # clim_upper = upper limit of the PSE map plots.
    # # FOR PSE MAP PLOTTING, e.g.
    # clim_lower = [35, 30, 25, 20] # Negative of this
    # clim_upper = [5]
    clims_multi = colourbar limits multiplier on maximum value that is plotted.
    vlims_multi = multiplier of the colourbar *range* plotted, multiplying the lower limit
    of the colourbar. Default value = 0.8.
    vlims_multi2 = (possible array, each will be plotted and saved according to the 
    index into the array) multiplier of the colourbar *range* plotted, multiplying the upper limit
    of the colourbar. Default value = [0.5, 0.25].
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # Colourmap scaling
    cmap = cm.bwr

    subplots_num_column = 2 # Assume. These are the number of subplots *per* column - assume 2 subplots per column, i.e. 2 rows.
    subplots_num_row = np.int32(np.ceil(np.divide(NumSlices, 2))) #; subplots_num_row = subplots_num_row[0]

    # # # 1) Plot each slice for recon and SI PSE for the mean oxygen calculation.
    for k in range(len(clims_overlay[2])):
        min_max_plotted = [np.min(PSE_image_plot[0,:,:,:]), np.max(PSE_image_plot[0,:,:,:])]
        min_max_abs = np.max(np.abs(min_max_plotted))
        norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

        # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
        # Also include the metric, component number used and metric value.
        fig1 = plt.figure(figsize=(19.20,10.80))
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of axes (axs) identifiers to plot the axes in a loop. Also, image identifier (ims).
        axs={}
        ims={}
        # # axes_c = a counter for the axis number to be used when plotting the 
        # # axis titles.
        # axes_c = 0
        # Lopo over slices/axes to be plotted
        for slice_plot in range(NumSlices):
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
            # Subplot number - added 1, in case needs to start from 1 instead of zero.
            # axs[slice_plot] = fig1.add_subplot(subplots_num_row, subplots_num_column, slice_plot+1)
            axs[slice_plot] = fig1.add_subplot(subplots_num_column, subplots_num_row, slice_plot+1)
            ims[slice_plot] = axs[slice_plot].imshow(PSE_image_plot[0,:,:,slice_plot], norm=norm_unC, cmap=cmap)
            axs[slice_plot].invert_yaxis(); fig1.colorbar(ims[slice_plot], ax=axs[slice_plot])
            ims[slice_plot].set_clim(vmin=-clims_overlay[2][k], vmax=clims_overlay[3][k])
            axs[slice_plot].set_title("Median " + plotting_param_title + " PSE")

        fig1.tight_layout()

        if saving_details[0] == 1:
            if RunNum_use > 1:
                fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_PSE___' + \
                metric_chosen[0] + "_" + NumC_title + '_Plot_' + plotting_param_name + '_' + '_ClimsSet_' + str(clims_overlay[2][k]) + '_' + str(clims_overlay[3][k]) + '.png', dpi=dpi_save, bbox_inches='tight')
            else:
                fig1.savefig(saving_details[1] + saving_details[2] + '_PSE___' + \
                metric_chosen[0] + "_" + NumC_title + '_Plot_' + plotting_param_name + '_' + '_ClimsSet_' + str(clims_overlay[2][k]) + '_' + str(clims_overlay[3][k]) + '.png', dpi=dpi_save, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_maps_components_EditPlot(\
    map_data, num_components, saving_details, showplot, \
    dpi_save, RunNum_use, cmap='bwr', subplots_num_in_figure=9, subplots_num_row=3, clims_multi=0.75):
    """
    Plot the component maps for all ICA components.
    Based on functions_PlottingICAetc_testing_Subplots.plot_maps_components.
    Plot and save (for any number of maps). Plot a set of maps for each slice.
    (Assume DPI = 300.)

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    clims_multi = multiplier of the colour plot scale, default = 0.75. It will multiply the
    colour bar scale which is calculated separately *for each component in each slice*.
    Returns:
    None - just saves figures and plots.

    # # subplots_num_in_figure & subplots_num_row info:
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))

    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[3]):
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # Create array of figure identifiers to loop over.
        separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        # Loop over number of separate figures
        for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
            figs[idx]=plt.figure(figsize=(19.20,10.80))
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            for j in range(subplots_num_in_figure):
                # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
                axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
                ims[axes_c] = axs[axes_c].imshow(map_data[axes_c,:,:,slice_plot], cmap)
                axs[axes_c].invert_yaxis(); figs[idx].colorbar(ims[axes_c], ax=axs[axes_c])
                ims[axes_c].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])))
                #
                axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                axs[axes_c].set_xticks([])
                axs[axes_c].set_yticks([])

                # figs[idx].tight_layout()
                # Increase the axis counter.
                axes_c = axes_c + 1

                figs[idx].tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if RunNum_use > 1:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')


        # Next plot the final partially filled figure. Only do this if there are
        # remainder subplots, i.e. subplots_num[1] != 0
        if subplots_num[1] != 0:
            figs_sub = plt.figure(figsize=(19.20,10.80))
            # Again, create a dictionary of axes (axs) identifiers to plot the axes
            # of the final figure. And for the images, an_ims_sub.
            an_axs_sub = {}
            an_ims_sub = {}

            # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
            # remainder axes or not) to maintain the correct scaling/sizes of the axes.
            for j in range(subplots_num_in_figure):
                # an_axs_sub[j] relates to each axis.
                # Add an axis for all subplots whether they are to be plotted or not.
                an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
                if j < subplots_num[1]:
                    # If the subplot axis is one of the remainder axes, plot.
                    # an_axs_sub[j] relates to each axis. ims[axes_c] relates to each image.
                    an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
                    an_ims_sub[j] = an_axs_sub[j].imshow(map_data[axes_c,:,:,slice_plot], cmap)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_ims_sub[j].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])))
                    an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                else:
                    # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                    # to maintain the axes scaling/sizes.
                    # Maybe plot as zeros so square shape
                    an_ims_sub[j] = an_axs_sub[j].imshow(np.multiply(map_data[0,:,:,slice_plot],0), cmap, vmin=0, vmax=0)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_axs_sub[j].set_title(" ")
                    an_axs_sub[j].set_aspect('equal')
                axes_c = axes_c + 1
                figs_sub.tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if RunNum_use > 1:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')
                else:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return

def plot_maps_components_EditPlot_origOrder(map_data, num_components, saving_details, showplot, \
    dpi_save, RunNum_use, cmap='bwr', subplots_num_in_figure=9, subplots_num_row=3, clims_multi=0.75):
    """
    Plot the component maps for all ICA components.
    Based on functions_PlottingICAetc_testing_Subplots.plot_maps_components.
    Plot and save (for any number of maps). Plot a set of maps for each slice.
    (Assume DPI = 300.)

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    clims_multi = multiplier of the colour plot scale, default = 0.75. It will multiply the
    colour bar scale which is calculated separately *for each component in each slice*.
    Returns:
    None - just saves figures and plots.

    # # subplots_num_in_figure & subplots_num_row info:
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))

    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[3]):
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # Create array of figure identifiers to loop over.
        separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        # Loop over number of separate figures
        for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
            figs[idx]=plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            for j in range(subplots_num_in_figure):
                # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
                axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
                ims[axes_c] = axs[axes_c].imshow(map_data[axes_c,:,:,slice_plot], cmap)
                axs[axes_c].invert_yaxis(); figs[idx].colorbar(ims[axes_c], ax=axs[axes_c])
                ims[axes_c].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])))
                axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                # Increase the axis counter.
                axes_c = axes_c + 1

                figs[idx].tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if RunNum_use > 1:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')

        # Next plot the final partially filled figure. Only do this if there are
        # remainder subplots, i.e. subplots_num[1] != 0
        if subplots_num[1] != 0:
            figs_sub = plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # Again, create a dictionary of axes (axs) identifiers to plot the axes
            # of the final figure. And for the images, an_ims_sub.
            an_axs_sub = {}
            an_ims_sub = {}

            # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
            # remainder axes or not) to maintain the correct scaling/sizes of the axes.
            for j in range(subplots_num_in_figure):
                # an_axs_sub[j] relates to each axis.
                # Add an axis for all subplots whether they are to be plotted or not.
                an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
                if j < subplots_num[1]:
                    # If the subplot axis is one of the remainder axes, plot.
                    # an_axs_sub[j] relates to each axis. ims[axes_c] relates to each image.
                    an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
                    an_ims_sub[j] = an_axs_sub[j].imshow(map_data[axes_c,:,:,slice_plot], cmap)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_ims_sub[j].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])))
                    an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                else:
                    # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                    # to maintain the axes scaling/sizes.
                    # Maybe plot as zeros so square shape
                    an_ims_sub[j] = an_axs_sub[j].imshow(np.multiply(map_data[0,:,:,slice_plot],0), cmap, vmin=0, vmax=0)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_axs_sub[j].set_title(" ")
                    an_axs_sub[j].set_aspect('equal')
                axes_c = axes_c + 1
                figs_sub.tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if RunNum_use > 1:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')
                else:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot + 1) \
                        + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def ICAcomponent_LoadandRecon_nifty(size_echo_data, hdf5_loc, dir_date, dir_subj,\
    echo_num, sorted_array_metric_sorted, metric_chosen, \
    NumSlices, NumDyn, masks_reg_data_cardiac, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use):
    """
    Function to load and reconstruct all ICA components for a particular run of ICA
    (specific number of ICA components, NumC, e.g. the number in which the OE component
    was found). The reconstruction includes unmean and unscaling.
    The reconstructed components are ordered, as are component timeseries and maps.

    Arguments:
    size_echo_data = shape of the input MRI data.
    hdf5_loc = location of the HDF5 files.
    dir_date = subject scanning date.
    dir_subj = subject ID.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    sorted_array_metric_sorted = array of the component numbers and metric value
    for the metric under investigation, sorted by the metric value. For use
    when describing the metric in the plot titles. Shape of 
    (number of converged ICA runs, 2, 1); with the first index running over
    the different numbers of components for which ICA reached convergence;
    the second index containing  the component numbers 
    sorted_array_metric_sorted[:,0,:] and their corresponding metric values 
    sorted_array_metric_sorted[:,1,:]; the third index previously ran over the
    different metrics under investigation.
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    NumSlices = number of slices acquired.
    NumDyn = number of OE-MRI dynamic images acquired.
    masks_reg_data_cardiac = cardiac masks with shape (vox,vox,NumSlices).
    RunNum_use = the specific RunNum to use.
    Returns:
    recon_all_components = all components reconstructed, with shape
    (vox,vox,NumDyn,NumSlices,NumComponents).
    ordered_S_ica = ordered array of the component timeseries, with
    shape (NumDyn,NumComponents).
    ordered_A_ica = ordered array of the component voxels amplitudes, 
    with shape (NonZ{{vox,vox,Slices}},NumComponents).
    ordered_A_ica_reshape = ordered array of the component maps (reshaped
    to form component maps), with shape (vox,vox,NumSlices,NumComponents).
    
    """
    # Load the HDF5 file and read in (read only).
    f = h5py.File(hdf5_loc + hdf5_file_name_base + dir_date + '_' + dir_subj, 'r')

    # Know from the argument sorted_array_metric_sorted the number of components 
    # of the ICA run containing the OE component. Use this to extract S_ica and
    # A_ica from the subject-specific HDF5 file (for the specific number of components
    # for the ICA run that contains the OE component).
    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num) + '/CardiacMasks'

    # Additional group names
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    additional_group_names = reg_applied + "/" + reg_type 
    group_name_full = group_name + '/' + additional_group_names


    numC_OE = np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))
    # Check runs to find the run that contains the metric of interest.
    # # Use RunNum provided.
    # for RunNumber in f[group_name_full + '/NumC_' + str(numC_OE)].keys():
    # Check if the desired metric is present in the run. If not, the for loop over
    # the different RunNumbers will continue until the RunNumber with the desired
    # metric is found.
    if group_name_full + '/NumC_' + str(numC_OE) + '/Run' + str(RunNum_use) + '/ComponentSorting/' \
            + metric_chosen[0] in f:
        # Extract the indices/component ordering from the HDF5 file.
        to_sort_indices = f[group_name_full + '/NumC_' + str(numC_OE) + '/' + '/Run' + str(RunNum_use) + '/ComponentSorting/' \
            + metric_chosen[0] + '/argsort_indices'][:]
        # Extract the ICA components S_ica and A_ica.
        S_ica_all = f[group_name_full + '/NumC_' + str(numC_OE) + '/' + '/Run' + str(RunNum_use) + '/S_ica_tVox'][:]
        A_ica_all = f[group_name_full + '/NumC_' + str(numC_OE) + '/' + '/Run' + str(RunNum_use) + '/A_ica_tVox'][:]
        # Also extract details of ICA pre-processing steps for use when reconstructing components 
        # (specifically the scaling and mean contained within the HDF5 file).
        ica_scaling = f[group_name_full + '/ica_scaling'][:]
        ica_tVox_mean = f[group_name_full + '/ica_tVox_mean'][:]
        # break

    f.close()


    # Sort/order S_ica_all and A_ica_all before reconstructing.
    ordered_S_ica = functions_Calculate_stats.order_components(S_ica_all, to_sort_indices)
    ordered_A_ica = functions_Calculate_stats.order_components(A_ica_all, to_sort_indices)

    # Reconstruct the components
    # Can use functions_Calculate_stats.reconstruct_component_icaMean but need to loop over the 
    # components as function only reconstructs a single component at a time.
    # Create empty array to store the reconstructed component dynamic maps.
    recon_all_components = np.zeros((size_echo_data[0],size_echo_data[1],size_echo_data[2],size_echo_data[3],numC_OE))
    for j in range(numC_OE):
        recon_collapsed, recon_map = functions_Calculate_stats.reconstruct_component_icaMean(\
            ordered_S_ica, ordered_A_ica, ica_tVox_mean, ica_scaling, j, \
            size_echo_data, NumSlices, NumDyn, masks_reg_data_cardiac, add_mean=1, un_scale=1, create_maps=1)
        del recon_collapsed
        recon_all_components[:,:,:,:,j] = recon_map
        del recon_map

    # Finally, reshape the ordered A_ica to form component maps for plotting (ordered_A_ica_reshape).
    ordered_A_ica_reshape = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, NumSlices, numC_OE, masks_reg_data_cardiac, ordered_A_ica)

    return recon_all_components, ordered_S_ica, ordered_A_ica, ordered_A_ica_reshape


def Plot_ICAPCA_final_scaled_nifty(dir_date, dir_subj, \
    metric_chosen, hdf5_loc, hdf5_subjectWide_name, showplot, save_plots, \
    subplots_num_in_figure_list, subplots_num_row_list, \
    echo_num, CA_type, dictionary_name, dpi_save, \
    images_saving_info_list, MRI_scan_directory, MRI_params, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    freq_spectra_plotting_options, \
    RunNum_use, invert_metric_values):
    """
    Function to plot the ICA/PCA components for a particular subject. 
    Both the components and their frequency spectra will be plotted,
    **BUT** the frequency spectra will be plotted as scaled by the amplitude (mean)
    of the gas cycling/OE frequency - as this is what is used in the frequency 
    RMS calculation.

    ** AND PLOTTING DIFFERENT SCALINGS DEPENDING ON INPUT ARGUMENTS ***

    Arguments:
    dir_date = subject scanning date.
    dir_subj = subject ID.
    metric_chosen = name of the frequency RMS ordering method being investigated.
    hdf5_loc = location of the HDF5 file which contains the metric information.
    subject_wide_HDF5_name = the name of the subject-wide HDF5 file containing 
    the PSE stats for all subjects which is to be opened and read from.
    save_plots = 0 if No; 1 if Yes - whether to save plots.
    showplot = 0 if No; 1 if Yes - whether to show plots.
    subplots_num_in_figure_list = list of the different numbers of subplots per figure
    to be plotted, associated with the list subplots_num_row_list.
    subplots_num_row_list = list of the different numbers of *ROWS* of subplots, 
    associated with the list subplots_num_in_figure_list.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    CA_type = list of the component analyses to be plotted, e.g.["ICA", "PCA"].
    dictionary_name = name and location of the mask dictionary.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    images_saving_info_list = details of where to save all images. As a 
    list with values [image_save_loc_general, image_dir_name].
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes]
    freq_spectra_plotting_options = a list of 1/0 for Y/N for plotting:
    - [0] frequency spectra scaled by the mean OE gas cycling frequency *amplitude*;
    - [1] frequency spectra scaled by the maximum frequency *amplitude* (per spectra);
    - [2] unscaled - as output by ICA - but could have any scaling.
    RunNum_use = specific RunNum to use, e.g. 1.
    invert_metric_values = file_details_create[5] = 1/0 for Y/N - whether to invert the ordering metric
    value. As original code minimised the ordering metric value, so for when 
    ordering the correlation value, inverted the correlation value so that the minimised
    inverted correlation value (metric value) was equivalent to maximising the 
    correlation value. Invert if want to plot with inverted metric value, e.g. for 
    correlation ordering.
    Returns:
    None - plot are created and saved if required.

    # # subplots_num_in_figure & subplots_num_row info:
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row

    """
    # print('PLOTTING')

    # Directory for saving subject-specific plots.
    today = date.today()
    d1 = today.strftime("%Y_%m_%d")

    # $$££$$££ NEW SUBGROUP - registered, is density correction applied?
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    # $$££$$££ NEW SUBGROUP - registered, is density correction applied?
    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # Directory - subject specific.
    image_saving_dir_subjSpec = images_saving_info_list[0] + images_saving_info_list[1] + d1 + '/' + \
        dir_date + '_' + dir_subj + '/'
    image_saving_pre_subjSpec = image_saving_dir_subjSpec + \
        dir_date + '_' + dir_subj + '_'

    if save_plots == 1:
        # If *saving plots* check to see if plot/image saving directory exists, if not
        # create one for the date of plotting and the subject investigated.
        if os.path.isdir(images_saving_info_list[0] + images_saving_info_list[1] + d1 + '/') == False:
            os.mkdir(images_saving_info_list[0] + images_saving_info_list[1] + d1)
        #
        if os.path.isdir(image_saving_dir_subjSpec) == False:
            # Make the directory
            os.mkdir(image_saving_dir_subjSpec[:-1])


    if MRI_params[3][0] == 1:
        # Cycled3
        saving_detail_add = 'Cycled3'


    plotting_param = 'reg'
    saving_details = [save_plots, \
        image_saving_dir_subjSpec + dir_date + '_' + dir_subj + '_' + CA_type + '_' + plotting_param + 'Echo' + str(echo_num) + '_SCALED'] # Slightly different append from usual, including plotting_param in 2nd list element.
    saving_details[1] = saving_details[1] + '__' + reg_applied + '__' + reg_type + '__' + saving_detail_add + '_'

    # 1) Load HDF5 for a specific subject.
    # 2) Identify lowest RMS component number for metric: RMS_aboveOEFreq_W2_WrtOEFreq.
    # 3) Extract component timeseries and ordering metric.
    # 4) Perform ordering of timeseries and also order metrics (simple ordering for metrics).
    # 5) Calculate frequency spectra.
    # 6) PLOT.

    # # 1) Load HDF5 for a specific subject.
    f = h5py.File(hdf5_loc + hdf5_file_name_base + dir_date + '_' + dir_subj, 'r')
    group_name = "group_" + CA_type + "_SI" + "_" + "Echo" + str(echo_num) + '/CardiacMasks'

    group_name_full = group_name + '/' + reg_applied + '/' + reg_type

    if CA_type == 'ICA':
        # ***ICA***
        # # 2) Identify lowest RMS component number for the specific metric.
        # These are contained within the subject-wide HDF5 files.
        # fwide = h5py.File(hdf5_loc + "PSE_stats_HDF5_structure_SubjectWide_Echo" + str(echo_num), 'r')
        fwide = h5py.File(hdf5_loc + hdf5_subjectWide_name + '_Echo' + str(echo_num), 'r')
        if RunNum_use == 1:
            # Original
            NumC = fwide[dir_date + '_' + dir_subj + '/' + metric_chosen][reg_applied + '_' + reg_type].attrs['NumC']
        else:
            NumC = fwide[dir_date + '_' + dir_subj + '/' + metric_chosen][reg_applied + '_' + reg_type]['Run' + str(RunNum_use)].attrs['NumC']
        numC_key = 'NumC_' + str(NumC)
        fwide.close()
        #
        # Check if ICA converged for the number of components and the specific run.
        if f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use) + '/converged_test'][:] == 1:
            # Check if the desired metric is present in the run. If not, the for loop over
            # the different RunNumbers will continue until the RunNumber with the desired
            # metric is found.
            if group_name_full + '/' + numC_key + '/Run' + str(RunNum_use) + '/ComponentSorting/' + metric_chosen in f:
                # Extract RMS metrics and ordering.
                metric_values = f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use) + '/ComponentSorting/' + metric_chosen + '/sorting_metrics'][:]
                argsort_indices = f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use) + '/ComponentSorting/' + metric_chosen + '/argsort_indices'][:]
                # Extract component timeseries.
                S_ica = f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use) + '/S_ica_tVox'][:]
                # print(RunNumber)
                # break
        #
        # # 4) Perform ordering of timeseries and also order metrics (simple ordering for metrics).
        # Use functions_Calculate_stats.order_components(array_to_order, arg_sort_indices) to perform the ordering.
        metric_values_ordered = np.squeeze(functions_Calculate_stats.order_components(np.array([metric_values]), argsort_indices))
        S_ordered = functions_Calculate_stats.order_components(S_ica, argsort_indices)
        # # 5) Calculate frequency spectra.
        freq_S_ordered = Sarah_ICA.freq_spec(np.shape(S_ordered)[1], S_ordered)

    f.close()

    # # 6) PLOT.

    freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(MRI_params[2], MRI_params[0])
    # Invert metric values?
    inverted_values = ''
    if invert_metric_values == 1:
        metric_values_ordered = np.divide(1, metric_values_ordered)
        inverted_values = '_inv_'

    metric_args_plot = [metric_chosen, metric_values_ordered]
    # If inverting, include in saving image name.
    saving_details[1] = saving_details[1] + inverted_values

    for j in range(len(subplots_num_in_figure_list)):
        subplots_num_in_figure = subplots_num_in_figure_list[j]
        subplots_num_row = subplots_num_row_list[j]
        # Save as different names, e.g. 1 and 2 etc...

        # Timeseries plot
        further_details = ["time series", "Time (s)", "timeseries"]
        functions_PlottingICAetc_testing_Subplots.plot_lineplot_RMSordered_metric(\
            plot_time_value, S_ordered, np.shape(S_ordered)[1], \
            saving_details, showplot, further_details, \
            halfTimepoints, metric_args_plot, dpi_save, RunNum_use, \
            subplots_num_in_figure, subplots_num_row)


    # SCALE FREQUENCY SPECTRA...
    # Frequency spectra plotting
    further_details = ["frequency spectra", "Frequency (Hz)", "freq"]

    ##### ALSO CHANGE NAME - for saving the plots
    # Save in loop, changing//adding further_details onto.
    further_details = ["frequency spectra", "Frequency (Hz)", "freq"]
    if freq_spectra_plotting_options[0] == 1:
        # Normalise by the mean OE cycling frequency.
        # Save as different names, e.g. 1 and 2 etc...
        further_details[2] = "freq_scale_OEFreq"
        #
        # # Also normalise the frequencies relative to the mean OE frequency amplitude.
        # # OE frequency ~ 0.00556 Hz.
        # # Frequency array values: 0, 0.00158, 0.00317, 0.00476, 0.00635, 0.00794, 0.00952, ...
        # # Therefore take the mean OE frequency amplitude as the mean value of the 
        # # absolute amplitudes of the 0.00476, 0.00635 Hz frequency elements.
        # # These are 4th and 5th in the array (3, 4 Python counting from zero).
        # OE_freq_index = np.array((4, 5)) # Counting normally, convert to Python
        # OE_freq_index = OE_freq_index - 1
        # # Calculate the relative frequency compared to the OE frequency amplitude.
        # OE_freq_S_ica = np.array((freq_S_ica_tVox_regOnly_P[OE_freq_index[0],:], freq_S_ica_tVox_regOnly_P[OE_freq_index[1],:]))
        # freq_S_ica_tVox_regOnly_P_value_relOEAmp = np.mean(np.abs(OE_freq_S_ica), axis=0)
        # freq_S_ica_tVox_regOnly_P_relOEAmp = np.divide(freq_S_ica_tVox_regOnly_P, freq_S_ica_tVox_regOnly_P_value_relOEAmp)
        OE_freq_index = np.array((4, 5)) # Counting normally, convert to Python
        OE_freq_index = OE_freq_index - 1
        # Calculate the relative frequency compared to the OE frequency amplitude.
        OE_freq_S_ica = np.array((freq_S_ordered[OE_freq_index[0],:], freq_S_ordered[OE_freq_index[1],:]))
        #
        freq_S_ordered_meanOEAmp = np.mean(np.abs(OE_freq_S_ica), axis=0)
        freq_S_ordered_relOEAmp = np.divide(freq_S_ordered, freq_S_ordered_meanOEAmp)
        # freq_S_ordered_original_scaling = np.array(freq_S_ordered)
        # freq_S_ordered = np.array(freq_S_ordered_relOEAmp)
        freq_spectra_plotting = np.array(freq_S_ordered_relOEAmp)
        #
        # And run plotting
        for j in range(len(subplots_num_in_figure_list)):
            subplots_num_in_figure = subplots_num_in_figure_list[j]
            subplots_num_row = subplots_num_row_list[j]
            functions_PlottingICAetc_testing_Subplots.plot_lineplot_RMSordered_metric(\
                freqPlot, np.abs(freq_spectra_plotting[0:halfTimepoints]), np.shape(freq_spectra_plotting[0:halfTimepoints])[1], \
                saving_details, showplot, further_details, \
                halfTimepoints, metric_args_plot, dpi_save, RunNum_use, \
                subplots_num_in_figure, subplots_num_row)

    if freq_spectra_plotting_options[1] == 1:
        # Normalise by the maximum frequency amplitude present
        # Save as different names, e.g. 1 and 2 etc...
        further_details[2] = "freq_scale_MaxFreq"
        #
        # freq_S_ordered_max_amp = np.max(np.abs(OE_freq_S_ica), axis=0)
        freq_S_ordered_max_amp = np.max(np.abs(freq_S_ordered), axis=0)
        freq_S_ordered_relMaxAmp = np.divide(freq_S_ordered, freq_S_ordered_max_amp)
        freq_spectra_plotting = np.array(freq_S_ordered_relMaxAmp)
        #
        # And run plotting
        for j in range(len(subplots_num_in_figure_list)):
            subplots_num_in_figure = subplots_num_in_figure_list[j]
            subplots_num_row = subplots_num_row_list[j]
            functions_PlottingICAetc_testing_Subplots.plot_lineplot_RMSordered_metric(\
                freqPlot, np.abs(freq_spectra_plotting[0:halfTimepoints]), np.shape(freq_spectra_plotting[0:halfTimepoints])[1], \
                saving_details, showplot, further_details, \
                halfTimepoints, metric_args_plot, dpi_save, RunNum_use, \
                subplots_num_in_figure, subplots_num_row)

    if freq_spectra_plotting_options[2] == 1:
        # Plot as output by ICA.
        # Save as different names, e.g. 1 and 2 etc...
        further_details[2] = "freq_ICAoutput"
        # Just run plotting
        for j in range(len(subplots_num_in_figure_list)):
            subplots_num_in_figure = subplots_num_in_figure_list[j]
            subplots_num_row = subplots_num_row_list[j]
            functions_PlottingICAetc_testing_Subplots.plot_lineplot_RMSordered_metric(\
                freqPlot, np.abs(freq_S_ordered[0:halfTimepoints]), np.shape(freq_S_ordered[0:halfTimepoints])[1], \
                saving_details, showplot, further_details, \
                halfTimepoints, metric_args_plot, dpi_save, RunNum_use, \
                subplots_num_in_figure, subplots_num_row)

    if freq_spectra_plotting_options[3] == 1:
        # Normalise by the mean frequency amplitude present
        # Save as different names, e.g. 1 and 2 etc...
        further_details[2] = "freq_scale_mean"
        #
        # freq_S_ordered_max_amp = np.max(np.abs(OE_freq_S_ica), axis=0)
        freq_S_ordered_mean_amp = np.mean(np.abs(freq_S_ordered), axis=0)
        freq_S_ordered_relMeanAmp = np.divide(freq_S_ordered, freq_S_ordered_mean_amp)
        freq_spectra_plotting = np.array(freq_S_ordered_relMeanAmp)
        #
        # And run plotting
        for j in range(len(subplots_num_in_figure_list)):
            subplots_num_in_figure = subplots_num_in_figure_list[j]
            subplots_num_row = subplots_num_row_list[j]
            functions_PlottingICAetc_testing_Subplots.plot_lineplot_RMSordered_metric(\
                freqPlot, np.abs(freq_spectra_plotting[0:halfTimepoints]), np.shape(freq_spectra_plotting[0:halfTimepoints])[1], \
                saving_details, showplot, further_details, \
                halfTimepoints, metric_args_plot, dpi_save, RunNum_use, \
                subplots_num_in_figure, subplots_num_row)

    if freq_spectra_plotting_options[4] == 1:
        # Normalise by the median frequency amplitude present
        # Save as different names, e.g. 1 and 2 etc...
        further_details[2] = "freq_scale_median"
        #
        # freq_S_ordered_max_amp = np.max(np.abs(OE_freq_S_ica), axis=0)
        freq_S_ordered_median_amp = np.median(np.abs(freq_S_ordered), axis=0)
        freq_S_ordered_relMedianAmp = np.divide(freq_S_ordered, freq_S_ordered_median_amp)
        freq_spectra_plotting = np.array(freq_S_ordered_relMedianAmp)
        #
        # And run plotting
        for j in range(len(subplots_num_in_figure_list)):
            subplots_num_in_figure = subplots_num_in_figure_list[j]
            subplots_num_row = subplots_num_row_list[j]
            functions_PlottingICAetc_testing_Subplots.plot_lineplot_RMSordered_metric(\
                freqPlot, np.abs(freq_spectra_plotting[0:halfTimepoints]), np.shape(freq_spectra_plotting[0:halfTimepoints])[1], \
                saving_details, showplot, further_details, \
                halfTimepoints, metric_args_plot, dpi_save, RunNum_use, \
                subplots_num_in_figure, subplots_num_row)


    return




def PSE_Calculation__CalculatePSE_Cycled3(hdf5_loc, dir_date, dir_subj, \
    metric_chosen, dictionary_name, \
    echo_num, MRI_scan_directory, MRI_params, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use, g_num, other_parameters):
    """
    **Cycled3** PSE calculations

    MEDIAN CALCULATION...

    Function to:
    - Load ICA outputs from an individual subject's HDF5 file.
    - Calculate ordering of the components - for a *single* component.
    - Reconstruct data from a specific component (unMean and unScale options).
    - Calculate the PSE of the reconstructed data and compare to the PSE of
        the registered SI MRI data. (Both maps and timeseries, can plot both.)
    - Calculate stats for the reconstructed and SI PSE values (maps specifically).
    - Assume g21 (> 21 components).

    "V2"
    ** PSE calculation are wrt the mean air image during cycle 1.**
    ** PSE calculations are also done for each oxygen cycle and for
    a mean of the oxygen cycles (last av_im_num of images in each cycle).
    ** Stats here are [mean, median, range, IQR, stdev].

    Arguments:
    hdf5_loc = location of the HDF5 file which contains the metric information.
    dir_date = scan date for the particular subject.
    dir_subj = subject ID for the particular subject.
    metric_chosen = the metric being investigated (only one now), in the form of a 
    list containing only one element (the name of the metric being investigated).
    av_im_num = the number of images to average over at the end of each oxygen cycle.
    dictionary_name = name and location of the mask dictionary.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.    
    subject_wide_HDF5_name = name of the HDF5 file to contain the "V2" PSE
    stats for *ALL* subjects.
    hdf5_loc = location of the HDF5 files.
    images_saving_info_list = details of where to save all images. As a 
    list with values [image_save_loc_general, image_dir_name].
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes]
    save_PSE_subject_wide = 1 if Yes; 0 if No. Whether to save the PSE stats extracted
    for a particular subject 
    RunNum_use = specific RunNum to use, e.g. 1.
    g_num = number of components greater than when ordering (e.g. 21 for 22 and greater).
    other_parameters = for future identification of the NumC to use when extracting 
    components and performing calculations.
    save_plots = 0 for No; 1 for Yes - whether to save the plots. Default = 0 (No).
    do_plots = whether to plot anything. Default = 0 (No).
    do_plots_2 = whether to plot the PSE slices with subject-wide colourbar scales.
    showplot = 0 for No; 1 for Yes - whether to show the plots. If No, the
    plots will be closed. Default = 0 (No).
    plot_components_vs_metrics = 0 for No; 1 for Yes - whether to plot components vs metric
    values to observe convergence/the optimal component number. Default = 1 (Yes).
    Returns:

    Cycled3
    Return stats_eachSlice_all_lung, stats_ALLSlice_all_lung, PSE_timeseries_withinLung, ...
        maskedMaps_lung, maskedMaps_lung_MeanEachSlice, PSE_image_OverallandCycled, load_data_echo1_regOnly
    # For *Cycled3* at least
    # # # When saving PSE stats - PSE (ICA and SI) within the lung mask:
    # stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung, ...
    # stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung
    # # # When plotting:
    # # PSE timeseries (ICA and SI) within the lung mask:
    # PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice
    # # SI/reconstructed SI timeseries within the lung mask:
    # load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung
    # load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice
    # # PSE maps (ICA and SI) within the cardiac mask:
    # PSE_image_SI_all, PSE_image_unMean_unSc_all
    # # Also:
    # load_data_echo1_regOnly
    # #
    # # Sizes:
    # # PSE stats
    # stats_ALLSlice_..._all_lung = np.zeros((4,len(stats_list)))
    # stats_eachSlice_..._all_lung = np.zeros((4, MRI_params[1],len(stats_list)))
    stats_eachSlice_all_lung = [stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung]
    stats_ALLSlice_all_lung = [stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung]
    # # PSE timeseries (wihtin the lung mask)
    # (420,4)
    PSE_timeseries_withinLung = [PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice]
    # maskedMaps_lung 
    # (vox,vox,NumDyn,NumSlices)
    maskedMaps_lung = [load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung]
    # maskedMaps_lungMask_MeanEachSlice
    # (NumDyn, NumSlices)
    maskedMaps_lung_MeanEachSlice = [load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice]
    # PSE image all
    # (4, vox,vox,NumSlices) - where 4 is for overall cycles and then for each of the three oxygen cycles.
    PSE_image_OverallandCycled = [PSE_image_SI_all, PSE_image_unMean_unSc_all]
    # load_data_echo1_regOnly
    # (vox,vox,NumDyn,NumSlices)
    # sorted_array_metric_sorted
    # sorted_array
    # S_ica_array
    # freq_S_ica_array
    # masks_reg_data_cardiac
    # stats_list


    """
    # MRI images directory and mask directory.
    # dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_mask = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[4] + '/'

    # MRI images directory and mask directory.
    # dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_A = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/'

    # $$££$$££
    # NiftyReg or ANTs - for first part

    if ANTs_used == 1:
        if densityCorr == 0:
            dir_images_B = MRI_scan_directory[3] + '/'
        else:
            dir_images_B = MRI_scan_directory[3] + 'corr' + regCorr_name + '/'
        dir_images = dir_images_A + dir_images_B
        # Here call load_MRI_data_raw and load the registered first echo data.
        load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8])
    else:
        if densityCorr == 0:
            dir_images_B = MRI_scan_directory[3] + dir_image_nifty + '_' + NiftyReg_ID + '/'
        else:
            dir_images_B = MRI_scan_directory[3] + 'corr' + regCorr_name + '_' + NiftyReg_ID + '/'
        dir_images = dir_images_A +  dir_images_B
        # Here call load_MRI_data_raw and load the registered first echo data.
        load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw_nii(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8], \
            densityCorr)



    # Here call load_MRI_data_raw and load the registered first echo data.
    size_echo_data = np.shape(load_data_echo1_regOnly)

    # Load image masks.
    # First find the names of the masks to be loaded for the particular subject.
    subject = dir_date + '_' + dir_subj
    mask_names_func_lung = functions_preproc_MRI_data.load_mask_subj_names(dictionary_name, 1, subject)
    mask_names_func_cardiac = functions_preproc_MRI_data.load_mask_subj_names(dictionary_name, 0, subject)
    # Load the image masks.
    masks_reg_data_lung = Sarah_ICA.load_MRI_reg_masks(dir_images_mask, mask_names_func_lung, MRI_params[1], \
        (size_echo_data[0],size_echo_data[1],size_echo_data[3]))
    masks_reg_data_cardiac = Sarah_ICA.load_MRI_reg_masks(dir_images_mask, mask_names_func_cardiac, MRI_params[1], \
        (size_echo_data[0],size_echo_data[1],size_echo_data[3]))

    # Calculate the number of non-zero mask voxels for use when calculating mean values within the masks.
    # Calculate the number of non-zero mask voxels over all slices and for each slice.
    nonZ_cardiac = np.int32(np.rint(np.sum(masks_reg_data_cardiac)))
    nonZ_lung = np.int32(np.rint(np.sum(masks_reg_data_lung)))
    nonZ_cardiac_slices = np.int32(np.rint(np.sum(masks_reg_data_cardiac, axis=0))); nonZ_cardiac_slices = np.int32(np.rint(np.sum(nonZ_cardiac_slices, axis=0)))
    nonZ_lung_slices = np.int32(np.rint(np.sum(masks_reg_data_lung, axis=0))); nonZ_lung_slices = np.int32(np.rint(np.sum(nonZ_lung_slices, axis=0)))
    # Single array to store the numbers of non-zero values over all slices for both masks.
    # Store in np.array([[], []]) format for accessing later.
    nonZ_masks = np.array([[nonZ_cardiac], [nonZ_lung]])
    # Single array to store the numbers of non-zero values for each slice for both masks.
    nonZ_masks_slices = np.array([nonZ_cardiac_slices, nonZ_lung_slices])
    # Ensure voxel numbers within the masks are integers.
    nonZ_masks = np.int32(np.rint(nonZ_masks))
    nonZ_masks_slices = np.int32(np.rint(nonZ_masks_slices))



    ## NaNs
    # Presence of NaNs outside cardiac masks (in background) - set these values to zero to avoid
    # issues when calculating PSE etc...
    # # Maybe earlier for this also NaN to zero if outside mask...
    load_data_echo1_regOnly_NaNs = np.isnan(load_data_echo1_regOnly)
    # And then e.g. multiply by inverted mask
    load_data_echo1_regOnly_NaNs_outsideCardiac = np.transpose(np.multiply(np.transpose(load_data_echo1_regOnly_NaNs, [2,0,1,3]), (1-masks_reg_data_cardiac)), [1,2,0,3])
    # PSE_timeseries_SI_lungMask_noNan_ousideCardiac = np.multiply()
    # np.sum(np.isnan(PSE_timeseries_SI_lungMask_NaNs_ousideCardiac))
    load_data_echo1_regOnly_NaNs_outsideCardiac = load_data_echo1_regOnly_NaNs_outsideCardiac == 1 # Boolean
    load_data_echo1_regOnly_noNan_ousideCardiac = np.array(load_data_echo1_regOnly)
    load_data_echo1_regOnly_noNan_ousideCardiac[load_data_echo1_regOnly_NaNs_outsideCardiac] = 0
    # np.sum(np.isnan(load_data_echo1_regOnly_noNan_ousideCardiac))
    load_data_echo1_regOnly_o = np.array(load_data_echo1_regOnly); del load_data_echo1_regOnly
    load_data_echo1_regOnly = np.array(load_data_echo1_regOnly_noNan_ousideCardiac); del load_data_echo1_regOnly_noNan_ousideCardiac




    # stats_list is a list of stats to be found.
    stats_list = ["mean", "median", "range", "IQR", "stdev"]


    # # Aim: identify component with lowest frequency RMS metric as this component
    # is most likely to be the OE component (due to the lack of high frequencies
    # such as those due to cardiac and respiratory motion and their motion-induced
    # density changes, and blood flow).
    # # In addition, the OE component (or component most likely to be the OE component)
    # with the lowest frequency RMS over ICA applied with different numbers of components
    # found is most likely to be the 'best' OE component. By having the least high frequency
    # contributions, it is likely to have most noise/unwanted sources well extracted by ICA.

    # Number of lowest metric values to be stored is given by num_lowest_metrics.
    num_lowest_metrics = 1
    # Extract the SINGLE METRIC OF INTEREST for a particular subject from the subject's 
    # HDF5 file. The number of components and metric values are stored and returned.
    print(metric_chosen)
    components_metrics_etc_new = functions_loadHDF5_ordering__NiftyReg_2022_08_26.extract_metrics_nifty(metric_chosen, \
        num_lowest_metrics, echo_num, hdf5_loc, dir_date, dir_subj, \
        NiftyReg_ID, densityCorr, regCorr_name, \
        ANTs_used, dir_image_nifty, hdf5_file_name_base, \
        RunNum_use, other_parameters[1], other_parameters[2])

    # Order components in terms of increasing component number for each metric being investigated.
    sorted_array = functions_loadHDF5_ordering.order_component_number(metric_chosen, components_metrics_etc_new)

    # **ASSUMING** g21.
    # Find location (row number) at which the component number in sorted_array > 21.
    # Originally, used == 21 components, but for some subjects the ICA algorithm may not
    # have reached convergence for this number of components. Hence, find lowest
    # component number that is greater than 21 and take as the lowest index.
    row_num = np.squeeze(np.where(sorted_array[:,0,0] > g_num))
    # Require integer for using as an index.
    row_num = np.int32(np.rint(row_num[0]))
    # Store only those > 21.
    sorted_array = sorted_array[row_num:,:,:]

    # Order by the lowest component metric value for each metric being investigated.
    sorted_array_metric_sorted = functions_loadHDF5_ordering.order_component_metric(metric_chosen, sorted_array)

    # # Reconstruct the lowest metric value component(s).
    # First, extract S_ica and A_ica for the component being reconstructed 
    # from the HDF5 file.
    # # Use the above information on the minimum metric value and the index and 
    # component number to access S_ica and A_ica (for the mimimum metric component).
    # Additionally, return the component number and metric of the component being 
    # reconstructed (to include in the plots), where component number is the 
    # number of components run for that particular ICA, assuming the 
    # reconstructed component is the one with the lowest metric value.

    # minNumberLookat = number of lowest metric value components to look at.
    minNumberLookat = 1 # was 3
    # Array of S_ica and A_ica for the single metric being investigated (previously, 
    # the different metrics had run over the third dimension). The second dimension
    # consists of multiple lowest components (as set by minNumberLookat).
    # $$££$$££
    S_ica_array, A_ica_array, min_timeseries_metrics_component_numbers, min_timeseries_metrics_values = \
        functions_loadHDF5_ordering__NiftyReg_2022_08_26.HDF5_extract_ICAoutputs_nifty(minNumberLookat, MRI_params[2], sorted_array_metric_sorted, \
        echo_num, metric_chosen, np.squeeze(nonZ_masks[0,:]), \
        hdf5_loc, dir_date, dir_subj, \
        NiftyReg_ID, densityCorr, regCorr_name, \
        ANTs_used, dir_image_nifty, hdf5_file_name_base, \
        RunNum_use)

    freq_S_ica_array = Sarah_ICA.freq_spec(len(metric_chosen), S_ica_array[:,0,:])


    # # COMPONENT RECONSTRUCTION

    # Reconstruct the data (unMean and * scaling as well) for a single ICA component each time.
    # $$££$$££
    recon_num = 1
    X_recon_OE_data, X_recon_OE_data_unMean, X_recon_OE_data_unMean_unSc = \
        functions_loadHDF5_ordering__NiftyReg_2022_08_26.reconstruct_single_ICA_component_nifty(hdf5_loc, dir_date, dir_subj, \
        MRI_params[2], np.squeeze(nonZ_masks[0,:]), metric_chosen, echo_num, \
        S_ica_array, A_ica_array, recon_num, \
        NiftyReg_ID, densityCorr, regCorr_name, \
        ANTs_used, dir_image_nifty, hdf5_file_name_base)

    del recon_num
    # Reshape the reconstructed data into image maps.
    # Need to reshape dynamic data, therefore use the (newly created) reshape_maps_DYNAMICS() 
    # function.
    X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps = \
        functions_loadHDF5_ordering.reshape_maps_DYNAMICS_noSqueeze(MRI_params[2], metric_chosen, \
        size_echo_data, masks_reg_data_cardiac, \
        X_recon_OE_data, X_recon_OE_data_unMean, X_recon_OE_data_unMean_unSc)

    # Apply masks to the loaded SI data for comparing to the reconstructed data.
    load_data_echo1_regOnly_maskedMaps = functions_Calculate_stats.apply_mask_maps(load_data_echo1_regOnly, masks_reg_data_cardiac)


    ######################################################################



    # **CALCULATING PSE with respect to the mean air cycle 1 image (baseline)**
    # For each oxygen cycle individually and for the mean of those three oxygen
    # cycles, taking the last av_im_num from each cycle as the oxygen image 
    # from each cycle.

    # # CALCULATE PSE 
    # PSE calculated as a timeseries and map/image for the metrics and SI data.

    # # PSE maps
    # av_im_num = the number of images to average over when calculating the mean
    # oxy image for each cycle.

    av_im_num = other_parameters[5]
    GasSwitching = MRI_params[3][1]

    # Only considering a single metric, so can use same PSE stat calculation.
    # New function was created to calculate for each cycle and average over all cycles
    # with respect to the initial air cycle - hence _airCycle1 in function name.
    PercentageChange_image, PSE_image_unMean_unSc, Difference_image, Air_map_baseline, \
        Mean_Oxy_map, PSE_image_unMean_unSc_cycle1, PSE_image_unMean_unSc_cycle2, PSE_image_unMean_unSc_cycle3 = \
        functions_loadHDF5_ordering.PSE_map_airCycle1(X_recon_OE_unMean_unSc_maps, GasSwitching, av_im_num)
    del PercentageChange_image, Difference_image, Air_map_baseline, Mean_Oxy_map

    # Similarly, calculate the PSE of SI data (_airCycle1 function).
    PercentageChange_image, PSE_image_SI, Difference_image, Air_map_baseline, \
        Mean_Oxy_map, PSE_image_SI_cycle1, PSE_image_SI_cycle2, PSE_image_SI_cycle3 = \
        functions_loadHDF5_ordering.PSE_map_airCycle1(load_data_echo1_regOnly_maskedMaps, GasSwitching, av_im_num)
    del PercentageChange_image, Difference_image, Air_map_baseline, Mean_Oxy_map

    # Store as PSE_image_all = [overall, cycle1, cycle2, cycle3]
    PSE_image_unMean_unSc_all = np.array((PSE_image_unMean_unSc, PSE_image_unMean_unSc_cycle1, \
        PSE_image_unMean_unSc_cycle2, PSE_image_unMean_unSc_cycle3))
    # Squeeze to get rid of extra recon dimension.
    PSE_image_unMean_unSc_all = np.squeeze(PSE_image_unMean_unSc_all)
    PSE_image_SI_all = np.array((PSE_image_SI, PSE_image_SI_cycle1, \
        PSE_image_SI_cycle2, PSE_image_SI_cycle3))




    # # STATS ANALYSIS
    # PSE METRICS STATS - extract from PSE maps (calculated from mean air vs mean oxy),
    # rather than from the timeseries dynamic PSE (calculated later).
    # Stats per slice and for all slices include:
    # stats_list = ["mean", "median", "range", "IQR", "stdev"]

    # Values within the lung masks are to be saved and plotted.

    # Calculate the stats within the lung masks.
    PSE_image_SI_all_lung = np.transpose(\
        functions_Calculate_stats.apply_mask_maps(np.transpose(PSE_image_SI_all, (1,2,0,3)), masks_reg_data_lung), \
        (2,0,1,3))
    PSE_image_unMean_unSc_all_lung = np.transpose(\
        functions_Calculate_stats.apply_mask_maps(np.transpose(PSE_image_unMean_unSc_all, (1,2,0,3)), masks_reg_data_lung), \
        (2,0,1,3))

    # Calculate stats as above
    # Calculate stats for SI PSE and recon unMean unScaled PSE - for 
    # each oxygen cycle and for the mean oxygen image.
    # Empty array to store PSE stats for overall and each oxygen cycle.
    stats_ALLSlice_PSE_SI_all_lung = np.zeros((4,len(stats_list)))
    stats_eachSlice_PSE_SI_all_lung = np.zeros((4, MRI_params[1],len(stats_list)))
    stats_ALLSlice_PSE_recon_all_lung = np.zeros((4,len(stats_list)))
    stats_eachSlice_PSE_recon_all_lung = np.zeros((4, MRI_params[1],len(stats_list)))

    # Calculate for overall and each cycle PSE - loop over variations:
    for k in range(np.shape(PSE_image_SI_all)[0]):
        # SI data
        stats_ALLSlice_PSE_SI_lung, stats_eachSlice_PSE_SI_lung = \
            functions_loadHDF5_ordering.calc_stats_slices_IQR(PSE_image_SI_all_lung[k,:,:,:], stats_list)
        stats_ALLSlice_PSE_SI_all_lung[k,:] = stats_ALLSlice_PSE_SI_lung
        stats_eachSlice_PSE_SI_all_lung[k,:,:] = stats_eachSlice_PSE_SI_lung
        del stats_ALLSlice_PSE_SI_lung, stats_eachSlice_PSE_SI_lung
        # And recon data
        stats_ALLSlice_PSE_recon_lung, stats_eachSlice_PSE_recon_lung = \
            functions_loadHDF5_ordering.calc_stats_slices_IQR(PSE_image_unMean_unSc_all_lung[k,:,:,:], stats_list)
        stats_ALLSlice_PSE_recon_all_lung[k,:] = stats_ALLSlice_PSE_recon_lung
        stats_eachSlice_PSE_recon_all_lung[k,:,:] = stats_eachSlice_PSE_recon_lung
        del stats_ALLSlice_PSE_recon_lung, stats_eachSlice_PSE_recon_lung


    # # PSE time series
    # Calculate PSE time series as a PSE dynamic image series (within 
    # cardiac masks due to reconstructed data) - again, _wrtCycle1 functions.
    PSE_timeseries_X_recon_OE_maps, PSE_timeseries_X_recon_OE_unMean_maps, PSE_timeseries_X_recon_OE_unMean_unSc_maps = \
        functions_loadHDF5_ordering.calc_PSE_timeseries_metrics_wrtCycle1(X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps, \
        metric_chosen, GasSwitching)
    # Squeeze all to remove dimension of 1 (due to choice of single metric).
    PSE_timeseries_X_recon_OE_maps = np.squeeze(PSE_timeseries_X_recon_OE_maps)
    PSE_timeseries_X_recon_OE_unMean_maps = np.squeeze(PSE_timeseries_X_recon_OE_unMean_maps)
    PSE_timeseries_X_recon_OE_unMean_unSc_maps = np.squeeze(PSE_timeseries_X_recon_OE_unMean_unSc_maps)
    PSE_timeseries_SI = functions_Calculate_stats.calc_PSE_timeseries_wrtCycle1(load_data_echo1_regOnly_maskedMaps, GasSwitching)




    # Apply lung masks and calculate within the lung masks.
    # As only single metric, now same SI method as recon method.
    # SI
    PSE_timeseries_SI_lungMask = functions_Calculate_stats.apply_mask_maps(PSE_timeseries_SI, masks_reg_data_lung)
    PSE_timeseries_SI_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(PSE_timeseries_SI_lungMask, nonZ_masks_slices[1,:])
    # Recon
    PSE_timeseries_recon_lungMask = functions_Calculate_stats.apply_mask_maps(np.squeeze(PSE_timeseries_X_recon_OE_maps), masks_reg_data_lung)
    PSE_timeseries_recon_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(PSE_timeseries_recon_lungMask, nonZ_masks_slices[1,:])

    PSE_timeseries_recon_unMean_lungMask = functions_Calculate_stats.apply_mask_maps(np.squeeze(PSE_timeseries_X_recon_OE_unMean_maps), masks_reg_data_lung)
    PSE_timeseries_recon_unMean_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(PSE_timeseries_recon_unMean_lungMask, nonZ_masks_slices[1,:])

    PSE_timeseries_recon_unMean_unSc_lungMask = functions_Calculate_stats.apply_mask_maps(np.squeeze(PSE_timeseries_X_recon_OE_unMean_unSc_maps), masks_reg_data_lung)
    PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(PSE_timeseries_recon_unMean_unSc_lungMask, nonZ_masks_slices[1,:])

    # SI mean across each lung masked slice - for reference when plotting the PSE timeseries etc.
    # As some subjects have irregular SI traces etc... useful to check.
    load_data_echo1_regOnly_maskedMaps_lung = functions_Calculate_stats.apply_mask_maps(load_data_echo1_regOnly, masks_reg_data_lung)
    load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(load_data_echo1_regOnly_maskedMaps_lung, nonZ_masks_slices[1,:])
    # Similarly for OE component timeseries (recon timeseries mean).
    # X_recon_OE_data_unMean_unSc is *collapsed already*
    # Just need to squeeze as has shape (NumDyn,NonZ{{vox,vox,NumSlices}},1), and then same mean axis=1 should be fine as in the function.
    # BUT in the cardiac mask --> need to use map, apply lung mask and then do same steps as above for the SI.
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung = functions_Calculate_stats.apply_mask_maps(np.squeeze(X_recon_OE_unMean_unSc_maps), masks_reg_data_lung)
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(X_recon_OE_unMean_unSc_maps_maskedMaps_lung, nonZ_masks_slices[1,:])









    # (A)
    # (A) Function to collapse maps but separately for each slice, with dynamics
    # # functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(input_data, masks)
    # Based on functions_Calculate_stats.collapse_maps_Fast
    PSE_timeseries_SI_lungMask_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(PSE_timeseries_SI, masks_reg_data_lung)
    PSE_timeseries_recon_lungMask_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(PSE_timeseries_X_recon_OE_maps, masks_reg_data_lung)
    PSE_timeseries_recon_unMean_lungMask_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(PSE_timeseries_X_recon_OE_unMean_maps, masks_reg_data_lung)
    PSE_timeseries_recon_unMean_unSc_lungMask_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(PSE_timeseries_X_recon_OE_unMean_unSc_maps, masks_reg_data_lung)



    # (B)
    # (B) Function to calculate the mean and median for each slice using the collapsed maps from (A) which 
    # have been collapsed separately for each slice and contain all dynamics
    # Calculate stats now for each slice of the collapsed (masked) data...
    # Based on functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied
    # functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(input_ARRAY_masked_collapsed_EachSlice_Dynamic, size_echo_data):
    PSE_timeseries_SI_lungMask__MeanEachSlice, PSE_timeseries_SI_lungMask__MedianEachSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(PSE_timeseries_SI_lungMask_Collapsed_EachSlice, size_echo_data)
    PSE_timeseries_recon_lungMask__MeanEachSlice, PSE_timeseries_recon_lungMask__MedianEachSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(PSE_timeseries_recon_lungMask_Collapsed_EachSlice, size_echo_data)
    PSE_timeseries_recon_unMean_lungMask__MeanEachSlice, PSE_timeseries_recon_unMean_lungMask__MedianEachSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(PSE_timeseries_recon_unMean_lungMask_Collapsed_EachSlice, size_echo_data)
    PSE_timeseries_recon_unMean_unSc_lungMask__MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask__MedianEachSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(PSE_timeseries_recon_unMean_unSc_lungMask_Collapsed_EachSlice, size_echo_data)




    # (C)
    # (C) Function to calculate the mean and median across all slices.
    # Based on functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied
    # BASED ON functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian
    # But instead, collapse all slices and then calculate mean and median - ALLSlice
    # Use functions_Calculate_stats.collapse_maps_Fast
    # For collapsed...
    # Use functions_Calculate_stats.collapse_maps_Fast(masked_data, masks)
    PSE_timeseries_SI_lungMask_Collapsed = functions_Calculate_stats.collapse_maps_Fast(PSE_timeseries_SI, masks_reg_data_lung)
    PSE_timeseries_recon_lungMask_Collapsed = functions_Calculate_stats.collapse_maps_Fast(PSE_timeseries_X_recon_OE_maps, masks_reg_data_lung)
    PSE_timeseries_recon_unMean_lungMask_Collapsed = functions_Calculate_stats.collapse_maps_Fast(PSE_timeseries_X_recon_OE_unMean_maps, masks_reg_data_lung)
    PSE_timeseries_recon_unMean_unSc_lungMask_Collapsed = functions_Calculate_stats.collapse_maps_Fast(PSE_timeseries_X_recon_OE_unMean_unSc_maps, masks_reg_data_lung)
    PSE_timeseries_SI_lungMask__MeanALLSlice, PSE_timeseries_SI_lungMask__MedianALLSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(PSE_timeseries_SI_lungMask_Collapsed)
    PSE_timeseries_recon_lungMask__MeanALLSlice, PSE_timeseries_recon_lungMask__MedianALLSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(PSE_timeseries_recon_lungMask_Collapsed)
    PSE_timeseries_recon_unMean_lungMask__MeanALLSlice, PSE_timeseries_recon_unMean_lungMask__MedianALLSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(PSE_timeseries_recon_unMean_lungMask_Collapsed)





    # SI mean across each lung masked slice - for reference when plotting the PSE timeseries etc.
    load_data_echo1_regOnly_maskedMaps_lung = functions_Calculate_stats.apply_mask_maps(load_data_echo1_regOnly, masks_reg_data_lung)
    load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(load_data_echo1_regOnly_maskedMaps_lung, nonZ_masks_slices[1,:])
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung = functions_Calculate_stats.apply_mask_maps(np.squeeze(X_recon_OE_unMean_unSc_maps), masks_reg_data_lung)
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice = \
        functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(X_recon_OE_unMean_unSc_maps_maskedMaps_lung, nonZ_masks_slices[1,:])

    load_data_echo1_regOnly_maskedMaps_lung_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(load_data_echo1_regOnly_maskedMaps_lung, masks_reg_data_lung)
    load_data_echo1_regOnly_maskedMaps_lung__MeanEachSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianEachSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(load_data_echo1_regOnly_maskedMaps_lung_Collapsed_EachSlice, size_echo_data)
    load_data_echo1_regOnly_maskedMaps_lung_Collapsed = functions_Calculate_stats.collapse_maps_Fast(load_data_echo1_regOnly_maskedMaps_lung, masks_reg_data_lung)
    load_data_echo1_regOnly_maskedMaps_lung__MeanALLSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianALLSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(load_data_echo1_regOnly_maskedMaps_lung_Collapsed)

    X_recon_OE_unMean_unSc_maps_maskedMaps_lung_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(X_recon_OE_unMean_unSc_maps_maskedMaps_lung, masks_reg_data_lung)
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MedianEachSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(X_recon_OE_unMean_unSc_maps_maskedMaps_lung_Collapsed_EachSlice, size_echo_data)
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung_Collapsed = functions_Calculate_stats.collapse_maps_Fast(X_recon_OE_unMean_unSc_maps_maskedMaps_lung, masks_reg_data_lung)
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MeanALLSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MedianALLSlice = \
        functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(X_recon_OE_unMean_unSc_maps_maskedMaps_lung_Collapsed)



    # RETURN:

    # # # When saving PSE stats - PSE (ICA and SI) within the lung mask:
    # stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung, ...
    # stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung
    # # # When plotting:
    # # PSE timeseries (ICA and SI) within the lung mask:
    # PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice
    # # SI/reconstructed SI timeseries within the lung mask:
    # load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung
    # load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice
    # # PSE maps (ICA and SI) within the cardiac mask:
    # PSE_image_SI_all, PSE_image_unMean_unSc_all
    # # Also:
    # load_data_echo1_regOnly

    # # # Sizes:
    # # # PSE stats
    # # stats_ALLSlice_..._all_lung = np.zeros((4,len(stats_list)))
    # # stats_eachSlice_..._all_lung = np.zeros((4, MRI_params[1],len(stats_list)))
    # stats_eachSlice_all_lung = [stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung]
    # stats_ALLSlice_all_lung = [stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung]
    # # # PSE timeseries (wihtin the lung mask)
    # # (420,4)
    # PSE_timeseries_withinLung = [PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice]
    # # maskedMaps_lung 
    # # (vox,vox,NumDyn,NumSlices)
    # maskedMaps_lung = [load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung]
    # # maskedMaps_lungMask_MeanEachSlice
    # # (NumDyn, NumSlices)
    # maskedMaps_lung_MeanEachSlice = [load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice]
    # # PSE image all
    # # (4, vox,vox,NumSlices) - where 4 is for overall cycles and then for each of the three oxygen cycles.
    # PSE_image_OverallandCycled = [PSE_image_SI_all, PSE_image_unMean_unSc_all]
    # # load_data_echo1_regOnly
    # # (vox,vox,NumDyn,NumSlices)

    stats_eachSlice_all_lung = [stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung]
    stats_ALLSlice_all_lung = [stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung]

    # # PSE timeseries (wihtin the lung mask)
    # (420,4)
    # 20230809 - mean and **median** added in.
    # 20230809 - **each slice and all slice added in**
    # # HERE PSE mean and median... for **EACH SLICE**
    PSE_timeseries_withinLung_EachSlice = [PSE_timeseries_SI_lungMask__MeanEachSlice, PSE_timeseries_SI_lungMask__MedianEachSlice, \
                                 PSE_timeseries_recon_unMean_unSc_lungMask__MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask__MedianEachSlice]
    # 20230809 - mean and **median** added in.
    # # HERE PSE mean and median... for **ALL SLICE**
    PSE_timeseries_withinLung_ALLSlice = [PSE_timeseries_SI_lungMask__MeanALLSlice, PSE_timeseries_SI_lungMask__MedianALLSlice, \
                                 PSE_timeseries_recon_unMean_lungMask__MeanALLSlice, PSE_timeseries_recon_unMean_lungMask__MedianALLSlice]
   
    # maskedMaps_lung 
    # (vox,vox,NumDyn,NumSlices)
    maskedMaps_lung = [load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung]
    # maskedMaps_lungMask_MeanEachSlice
    # (NumDyn, NumSlices)
    # 20230809 - mean and **median** added in.
    # 20230809 - **each slice and all slice added in**
    # # HERE PSE mean and median... for **EACH SLICE**
    maskedMaps_lung_MeanEachSlice = [load_data_echo1_regOnly_maskedMaps_lung__MeanEachSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianEachSlice, \
                                     X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MedianEachSlice]
    # 20230809 - mean and **median** added in.
    # # HERE PSE mean and median... for **ALL SLICE**
    maskedMaps_lung_MeanALLSlice = [load_data_echo1_regOnly_maskedMaps_lung__MeanALLSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianALLSlice, \
                                    X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MeanALLSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lung__MedianALLSlice]

    # PSE image all
    # (4, vox,vox,NumSlices) - where 4 is for overall cycles and then for each of the three oxygen cycles.
    # # # AND ADD IN PSE dynamic image...
    PSE_image_OverallandCycled = [PSE_image_SI_all, PSE_image_unMean_unSc_all, PSE_timeseries_SI, PSE_timeseries_X_recon_OE_unMean_unSc_maps]
    # load_data_echo1_regOnly
    # (vox,vox,NumDyn,NumSlices)


    return stats_eachSlice_all_lung, stats_ALLSlice_all_lung, PSE_timeseries_withinLung_EachSlice, PSE_timeseries_withinLung_ALLSlice, \
        maskedMaps_lung, maskedMaps_lung_MeanEachSlice, maskedMaps_lung_MeanALLSlice, PSE_image_OverallandCycled, load_data_echo1_regOnly, \
        sorted_array_metric_sorted, sorted_array, S_ica_array, freq_S_ica_array, masks_reg_data_cardiac, stats_list





def plot_MRI_SI_PSE_maps_timeseries(dir_date, dir_subj, \
    metric_chosen, av_im_num, dictionary_name, \
    echo_num, \
    save_plots, showplot, \
    clim_lower, clim_upper, \
    dpi_save, \
    what_to_plot, which_plots, \
    images_saving_info_list, MRI_scan_directory, \
    subplots_num_in_figure, subplots_num_row, \
    MRI_params, clims_overlay, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use, g_num, other_parameters, \
    PSE_yaxis_range, plot_mean_median):

    # Directory for saving subject-specific plots.
    today = date.today()
    d1 = today.strftime("%Y_%m_%d")

    # Directory - subject specific.
    image_saving_dir_subjSpec = images_saving_info_list[0] + images_saving_info_list[1] + d1 + '/' + \
        dir_date + '_' + dir_subj + '/'
    image_saving_pre_subjSpec = image_saving_dir_subjSpec + \
        dir_date + '_' + dir_subj + '_'

    if save_plots == 1:
        # HERE ONLY IF SAVING THOUGH...
        if os.path.isdir(images_saving_info_list[0] + images_saving_info_list[1] + d1 + '/') == False:
            os.mkdir(images_saving_info_list[0] + images_saving_info_list[1] + d1)
        #
        if os.path.isdir(image_saving_dir_subjSpec) == False:
            # Make the directory
            os.mkdir(image_saving_dir_subjSpec[:-1])

    plotting_param = 'SI_regEcho' + str(echo_num)
    component_analysis = 'ICA'
    # Saving details - dir and name
    saving_details = []
    saving_details.append(save_plots)
    # saving_details.append(image_saving_pre_allSubj + component_analysis + 'on') # Instead, save for each subject in dir
    saving_details.append(image_saving_dir_subjSpec + dir_date + '_' + dir_subj + '_' + \
        component_analysis + '_')
    saving_details.append(plotting_param)
    saving_details.append(metric_chosen[0])
    # saving_details.append(metric_chosen)

    # $$££$$££ saving_details[2] - append info about reg and registration type.
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # # Add in SingleCycle or Cycled3 identifier
    if MRI_params[3][0] == 1:
        # Cycled3
        saving_detail_add = 'Cycled3'

    saving_details[2] = saving_details[2] + '__' + reg_applied + '__' + reg_type + '__' + saving_detail_add + '_'


    if MRI_params[3][0] == 0:
        if len(av_im_num) == 4:
            saving_details[2] = saving_details[2] + av_im_num[3]

    # Remove ICA references in saving_details list for SI plots.
    saving_details_SI = saving_details[:]
    saving_details_SI[1] = saving_details_SI[1][:-4]

    # Call generate_freq - to generate frequencies and timepoints for plotting.
    # Use plot_frequencies for plotting.
    freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(MRI_params[2], MRI_params[0])
    # # Here call load_MRI_data_raw and load the registered first echo data.
    # # load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8])
    # size_echo_data = np.shape(load_data_echo1_regOnly)


    # MRI images directory and mask directory.
    # dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_mask = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[4] + '/'

    # MRI images directory and mask directory.
    # dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_A = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/'

    if ANTs_used == 1:
        if densityCorr == 0:
            dir_images_B = MRI_scan_directory[3] + '/'
        else:
            dir_images_B = MRI_scan_directory[3] + 'corr' + regCorr_name + '/'
        dir_images = dir_images_A + dir_images_B
        # Here call load_MRI_data_raw and load the registered first echo data.
        load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8])
    else:
        if densityCorr == 0:
            dir_images_B = MRI_scan_directory[3] + dir_image_nifty + '_' + NiftyReg_ID + '/'
        else:
            dir_images_B = MRI_scan_directory[3] + 'corr' + regCorr_name + '_' + NiftyReg_ID + '/'
        dir_images = dir_images_A +  dir_images_B
        # Here call load_MRI_data_raw and load the registered first echo data.
        load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw_nii(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8], \
            densityCorr)
        # load_data_echo1_regOnly = load_MRI_data_raw_nii(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8], \
        #     densityCorr)

        # Here call load_MRI_data_raw and load the registered first echo data.
    size_echo_data = np.shape(load_data_echo1_regOnly)

    # Load image masks.
    # First find the names of the masks to be loaded for the particular subject.
    subject = dir_date + '_' + dir_subj
    mask_names_func_lung = functions_preproc_MRI_data.load_mask_subj_names(dictionary_name, 1, subject)
    mask_names_func_cardiac = functions_preproc_MRI_data.load_mask_subj_names(dictionary_name, 0, subject)
    # Load the image masks.
    masks_reg_data_lung = Sarah_ICA.load_MRI_reg_masks(dir_images_mask, mask_names_func_lung, MRI_params[1], \
        (size_echo_data[0],size_echo_data[1],size_echo_data[3]))
    masks_reg_data_cardiac = Sarah_ICA.load_MRI_reg_masks(dir_images_mask, mask_names_func_cardiac, MRI_params[1], \
        (size_echo_data[0],size_echo_data[1],size_echo_data[3]))

    # Calculate the number of non-zero mask voxels for use when calculating mean values within the masks.
    # Calculate the number of non-zero mask voxels over all slices and for each slice.
    nonZ_cardiac = np.int32(np.rint(np.sum(masks_reg_data_cardiac)))
    nonZ_lung = np.int32(np.rint(np.sum(masks_reg_data_lung)))
    nonZ_cardiac_slices = np.int32(np.rint(np.sum(masks_reg_data_cardiac, axis=0))); nonZ_cardiac_slices = np.int32(np.rint(np.sum(nonZ_cardiac_slices, axis=0)))
    nonZ_lung_slices = np.int32(np.rint(np.sum(masks_reg_data_lung, axis=0))); nonZ_lung_slices = np.int32(np.rint(np.sum(nonZ_lung_slices, axis=0)))
    # Single array to store the numbers of non-zero values over all slices for both masks.
    # Store in np.array([[], []]) format for accessing later.
    nonZ_masks = np.array([[nonZ_cardiac], [nonZ_lung]])
    # Single array to store the numbers of non-zero values for each slice for both masks.
    nonZ_masks_slices = np.array([nonZ_cardiac_slices, nonZ_lung_slices])
    # Ensure voxel numbers within the masks are integers.
    nonZ_masks = np.int32(np.rint(nonZ_masks))
    nonZ_masks_slices = np.int32(np.rint(nonZ_masks_slices))


    # NaNs
    # Presence of NaNs outside cardiac masks (in background) - set these values to zero to avoid
    # issues when calculating PSE etc...
    # # Maybe earlier for this also NaN to zero if outside mask...
    load_data_echo1_regOnly_NaNs = np.isnan(load_data_echo1_regOnly)
    # And then e.g. multiply by inverted mask
    load_data_echo1_regOnly_NaNs_outsideCardiac = np.transpose(np.multiply(np.transpose(load_data_echo1_regOnly_NaNs, [2,0,1,3]), (1-masks_reg_data_cardiac)), [1,2,0,3])
    # PSE_timeseries_SI_lungMask_noNan_ousideCardiac = np.multiply()
    # np.sum(np.isnan(PSE_timeseries_SI_lungMask_NaNs_ousideCardiac))
    load_data_echo1_regOnly_NaNs_outsideCardiac = load_data_echo1_regOnly_NaNs_outsideCardiac == 1 # Boolean
    load_data_echo1_regOnly_noNan_ousideCardiac = np.array(load_data_echo1_regOnly)
    load_data_echo1_regOnly_noNan_ousideCardiac[load_data_echo1_regOnly_NaNs_outsideCardiac] = 0
    # np.sum(np.isnan(load_data_echo1_regOnly_noNan_ousideCardiac))
    load_data_echo1_regOnly_o = np.array(load_data_echo1_regOnly); del load_data_echo1_regOnly
    load_data_echo1_regOnly = np.array(load_data_echo1_regOnly_noNan_ousideCardiac); del load_data_echo1_regOnly_noNan_ousideCardiac

    # stats_list is a list of stats to be found.
    stats_list = ["mean", "median", "range", "IQR", "stdev"]

    # Apply masks to the loaded SI data for comparing to the reconstructed data.
    load_data_echo1_regOnly_maskedMaps = functions_Calculate_stats.apply_mask_maps(load_data_echo1_regOnly, masks_reg_data_cardiac)









    if MRI_params[3][0] == 1:
        av_im_num = other_parameters[5]
        GasSwitching = MRI_params[3][1]


        # Similarly, calculate the PSE of SI data (_airCycle1 function).
        PercentageChange_image, PSE_image_SI, Difference_image, Air_map_baseline, \
            Mean_Oxy_map, PSE_image_SI_cycle1, PSE_image_SI_cycle2, PSE_image_SI_cycle3 = \
            functions_loadHDF5_ordering.PSE_map_airCycle1(load_data_echo1_regOnly_maskedMaps, GasSwitching, av_im_num)
        del PercentageChange_image, Difference_image, Air_map_baseline, Mean_Oxy_map
        PSE_image_SI_all = np.array((PSE_image_SI, PSE_image_SI_cycle1, \
            PSE_image_SI_cycle2, PSE_image_SI_cycle3))

        # # STATS ANALYSIS
        # PSE METRICS STATS - extract from PSE maps (calculated from mean air vs mean oxy),
        # rather than from the timeseries dynamic PSE (calculated later).
        # Stats per slice and for all slices include:
        # stats_list = ["mean", "median", "range", "IQR", "stdev"]
        # Calculate the stats within the lung masks.
        PSE_image_SI_all_lung = np.transpose(\
            functions_Calculate_stats.apply_mask_maps(np.transpose(PSE_image_SI_all, (1,2,0,3)), masks_reg_data_lung), \
            (2,0,1,3))

        # Calculate stats as above
        # Calculate stats for SI PSE and recon unMean unScaled PSE - for 
        # each oxygen cycle and for the mean oxygen image.
        # Empty array to store PSE stats for overall and each oxygen cycle.
        stats_ALLSlice_PSE_SI_all_lung = np.zeros((4,len(stats_list)))
        stats_eachSlice_PSE_SI_all_lung = np.zeros((4, MRI_params[1],len(stats_list)))


        # Calculate for overall and each cycle PSE - loop over variations:
        for k in range(np.shape(PSE_image_SI_all)[0]):
            # SI data
            stats_ALLSlice_PSE_SI_lung, stats_eachSlice_PSE_SI_lung = \
                functions_loadHDF5_ordering.calc_stats_slices_IQR(PSE_image_SI_all_lung[k,:,:,:], stats_list)
            stats_ALLSlice_PSE_SI_all_lung[k,:] = stats_ALLSlice_PSE_SI_lung
            stats_eachSlice_PSE_SI_all_lung[k,:,:] = stats_eachSlice_PSE_SI_lung
            del stats_ALLSlice_PSE_SI_lung, stats_eachSlice_PSE_SI_lung

        # # PSE time series
        PSE_timeseries_SI = functions_Calculate_stats.calc_PSE_timeseries_wrtCycle1(load_data_echo1_regOnly_maskedMaps, GasSwitching)

        # SI
        PSE_timeseries_SI_lungMask = functions_Calculate_stats.apply_mask_maps(PSE_timeseries_SI, masks_reg_data_lung)
        PSE_timeseries_SI_lungMask_MeanEachSlice = \
            functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(PSE_timeseries_SI_lungMask, nonZ_masks_slices[1,:])
        # SI mean across each lung masked slice - for reference when plotting the PSE timeseries etc.
        # As some subjects have irregular SI traces etc... useful to check.
        load_data_echo1_regOnly_maskedMaps_lung = functions_Calculate_stats.apply_mask_maps(load_data_echo1_regOnly, masks_reg_data_lung)
        load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = \
            functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(load_data_echo1_regOnly_maskedMaps_lung, nonZ_masks_slices[1,:])




        # (A)
        # (A) Function to collapse maps but separately for each slice, with dynamics
        # # functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(input_data, masks)
        # Based on functions_Calculate_stats.collapse_maps_Fast
        PSE_timeseries_SI_lungMask_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(PSE_timeseries_SI, masks_reg_data_lung)

        # (B)
        # (B) Function to calculate the mean and median for each slice using the collapsed maps from (A) which 
        # have been collapsed separately for each slice and contain all dynamics
        # Calculate stats now for each slice of the collapsed (masked) data...
        # Based on functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied
        # functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(input_ARRAY_masked_collapsed_EachSlice_Dynamic, size_echo_data):
        PSE_timeseries_SI_lungMask__MeanEachSlice, PSE_timeseries_SI_lungMask__MedianEachSlice = \
            functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(PSE_timeseries_SI_lungMask_Collapsed_EachSlice, size_echo_data)

        # (C)
        # (C) Function to calculate the mean and median across all slices.
        # Based on functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied
        # BASED ON functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian
        # But instead, collapse all slices and then calculate mean and median - ALLSlice
        # Use functions_Calculate_stats.collapse_maps_Fast
        # For collapsed...
        # Use functions_Calculate_stats.collapse_maps_Fast(masked_data, masks)
        PSE_timeseries_SI_lungMask_Collapsed = functions_Calculate_stats.collapse_maps_Fast(PSE_timeseries_SI, masks_reg_data_lung)
        # # And now, calculate mean/median over all...
        # NOT worrying about zeros etc...
        # Similar to functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian
        # USING
        # Newly created functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian
        PSE_timeseries_SI_lungMask__MeanALLSlice, PSE_timeseries_SI_lungMask__MedianALLSlice = \
            functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(PSE_timeseries_SI_lungMask_Collapsed)




        # And for ... (similar to the above)
        # SI mean across each lung masked slice - for reference when plotting the PSE timeseries etc.
        # As some subjects have irregular SI traces etc... useful to check.
        load_data_echo1_regOnly_maskedMaps_lung = functions_Calculate_stats.apply_mask_maps(load_data_echo1_regOnly, masks_reg_data_lung)
        load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = \
            functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(load_data_echo1_regOnly_maskedMaps_lung, nonZ_masks_slices[1,:])
        # # MRI SI
        load_data_echo1_regOnly_maskedMaps_lung_Collapsed_EachSlice = functions_Calculate_stats.collapse_maps_Fast_dynamic_EachSliceArray(load_data_echo1_regOnly_maskedMaps_lung, masks_reg_data_lung)
        load_data_echo1_regOnly_maskedMaps_lung__MeanEachSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianEachSlice = \
            functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(load_data_echo1_regOnly_maskedMaps_lung_Collapsed_EachSlice, size_echo_data)
        load_data_echo1_regOnly_maskedMaps_lung_Collapsed = functions_Calculate_stats.collapse_maps_Fast(load_data_echo1_regOnly_maskedMaps_lung, masks_reg_data_lung)
        load_data_echo1_regOnly_maskedMaps_lung__MeanALLSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianALLSlice = \
            functions_Calculate_stats.timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(load_data_echo1_regOnly_maskedMaps_lung_Collapsed)

        # RETURN:

        # For *Cycled3* at least

        # # # When saving PSE stats - PSE (ICA and SI) within the lung mask:
        # stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung, ...
        # stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung
        # # # When plotting:
        # # PSE timeseries (ICA and SI) within the lung mask:
        # PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice
        # # SI/reconstructed SI timeseries within the lung mask:
        # load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung
        # load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice
        # # PSE maps (ICA and SI) within the cardiac mask:
        # PSE_image_SI_all, PSE_image_unMean_unSc_all
        # # Also:
        # load_data_echo1_regOnly

        # # # Sizes:
        # # # PSE stats
        # # stats_ALLSlice_..._all_lung = np.zeros((4,len(stats_list)))
        # # stats_eachSlice_..._all_lung = np.zeros((4, MRI_params[1],len(stats_list)))
        # stats_eachSlice_all_lung = [stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung]
        # stats_ALLSlice_all_lung = [stats_ALLSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_recon_all_lung]
        # # # PSE timeseries (wihtin the lung mask)
        # # (420,4)
        # PSE_timeseries_withinLung = [PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice]
        # # maskedMaps_lung 
        # # (vox,vox,NumDyn,NumSlices)
        # maskedMaps_lung = [load_data_echo1_regOnly_maskedMaps_lung, X_recon_OE_unMean_unSc_maps_maskedMaps_lung]
        # # maskedMaps_lungMask_MeanEachSlice
        # # (NumDyn, NumSlices)
        # maskedMaps_lung_MeanEachSlice = [load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice, X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice]
        # # PSE image all
        # # (4, vox,vox,NumSlices) - where 4 is for overall cycles and then for each of the three oxygen cycles.
        # PSE_image_OverallandCycled = [PSE_image_SI_all, PSE_image_unMean_unSc_all]
        # # load_data_echo1_regOnly
        # # (vox,vox,NumDyn,NumSlices)

        stats_eachSlice_all_lung = [stats_eachSlice_PSE_SI_all_lung]
        stats_ALLSlice_all_lung = [stats_ALLSlice_PSE_SI_all_lung]

        # # PSE timeseries (wihtin the lung mask)
        # (420,4)
        # 20230809 - mean and **median** added in.
        # 20230809 - **each slice and all slice added in**
        # # HERE PSE mean and median... for **EACH SLICE**
        PSE_timeseries_withinLung_EachSlice = [PSE_timeseries_SI_lungMask__MeanEachSlice, PSE_timeseries_SI_lungMask__MedianEachSlice]
        # 20230809 - mean and **median** added in.
        # # HERE PSE mean and median... for **ALL SLICE**
        PSE_timeseries_withinLung_ALLSlice = [PSE_timeseries_SI_lungMask__MeanALLSlice, PSE_timeseries_SI_lungMask__MedianALLSlice]
    
        # maskedMaps_lung 
        # (vox,vox,NumDyn,NumSlices)
        maskedMaps_lung = [load_data_echo1_regOnly_maskedMaps_lung]
        # maskedMaps_lungMask_MeanEachSlice
        # (NumDyn, NumSlices)
        # 20230809 - mean and **median** added in.
        # 20230809 - **each slice and all slice added in**
        # # HERE PSE mean and median... for **EACH SLICE**
        maskedMaps_lung_MeanEachSlice = [load_data_echo1_regOnly_maskedMaps_lung__MeanEachSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianEachSlice]
        # 20230809 - mean and **median** added in.
        # # HERE PSE mean and median... for **ALL SLICE**
        maskedMaps_lung_MeanALLSlice = [load_data_echo1_regOnly_maskedMaps_lung__MeanALLSlice, load_data_echo1_regOnly_maskedMaps_lung__MedianALLSlice]

        # PSE image all
        # (4, vox,vox,NumSlices) - where 4 is for overall cycles and then for each of the three oxygen cycles.
        PSE_image_OverallandCycled = [PSE_image_SI_all]
        # load_data_echo1_regOnly
        # (vox,vox,NumDyn,NumSlices)




    # # "Unpack" the returned lists to use for when plotting etc... 
    # stats_eachSlice_PSE_SI_all_lung = stats_eachSlice_all_lung[0]
    # stats_eachSlice_PSE_recon_all_lung = stats_eachSlice_all_lung[1]

    # stats_ALLSlice_PSE_SI_all_lung = stats_ALLSlice_all_lung[0]
    # stats_ALLSlice_PSE_recon_all_lung = stats_ALLSlice_all_lung[1]

    PSE_timeseries_SI_lungMask_MeanEachSlice = PSE_timeseries_withinLung_EachSlice[0]
    PSE_timeseries_SI_lungMask_MedianEachSlice = PSE_timeseries_withinLung_EachSlice[1]

    PSE_timeseries_SI_lungMask_MeanALLSlice = PSE_timeseries_withinLung_ALLSlice[0]
    PSE_timeseries_SI_lungMask_MedianALLSlice = PSE_timeseries_withinLung_ALLSlice[1]

    load_data_echo1_regOnly_maskedMaps_lung = maskedMaps_lung[0]


    load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = maskedMaps_lung_MeanEachSlice[0]
    load_data_echo1_regOnly_maskedMaps_lungMask_MedianEachSlice = maskedMaps_lung_MeanEachSlice[1]

    load_data_echo1_regOnly_maskedMaps_lungMask_MeanALLSlice = maskedMaps_lung_MeanALLSlice[0]
    load_data_echo1_regOnly_maskedMaps_lungMask_MedianALLSlice = maskedMaps_lung_MeanALLSlice[1]

    PSE_image_SI_all = PSE_image_OverallandCycled[0]

    size_echo_data = np.shape(load_data_echo1_regOnly)





    # # # "Unpack" the returned lists to use for when plotting etc...
    # # stats_eachSlice_PSE_SI_all_lung = np.array(stats_eachSlice_PSE_SI_all_lung)
    # # stats_ALLSlice_PSE_SI_all_lung = np.array(stats_ALLSlice_PSE_SI_all_lung)
    # PSE_timeseries_SI_lungMask_MeanEachSlice = np.array(PSE_timeseries_SI_lungMask_MeanEachSlice)
    # load_data_echo1_regOnly_maskedMaps_lung = np.array(load_data_echo1_regOnly_maskedMaps_lung)
    # load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = np.array(load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice)

    # PSE_image_SI_all = np.array(PSE_image_SI_all)
    # size_echo_data = np.shape(load_data_echo1_regOnly)


    # # # # # # # # # Plotting

    # Titles/saving details for OE vs SI plotting.
    plotting_param_name_list = ['Recon_OE', 'SI']
    plotting_param_name_list_titles = ['Recon OE', 'MRI SI']
    # plotting_param_chosen use 0 or Recon OE and 1 for MRI SI.

    # plotting_param_chosen use 0 or Recon OE and 1 for MRI SI.
    stat_plotted = ['mean', 'median']


    if plot_mean_median[0] == 1:
        # # # PLOT MEAN
        stat_plotted_string = stat_plotted[0]
        # # # MEAN PSE TIMESERIES PLOTS:
        PSE_timeseries_EachSlice = np.array(PSE_timeseries_SI_lungMask_MeanALLSlice)
        #
        plotting_param_chosen = 1
        plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
            PSE_timeseries_EachSlice, \
            plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
            saving_details_SI, showplot, dpi_save, metric_chosen, '', 0, stat_plotted_string)
        #
        # Plot with limited y-axis range?
        # if PSE_yaxis_range[0] == 1:
        plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
            PSE_timeseries_EachSlice, \
            plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
            saving_details_SI, showplot, dpi_save, metric_chosen, '', \
            PSE_yaxis_range, 0, stat_plotted_string)

    if plot_mean_median[1] == 1:
        # # # PLOT MEDIAN NOW
        stat_plotted_string = stat_plotted[1]
        # # # MEDIAN PSE TIMESERIES PLOTS:
        PSE_timeseries_EachSlice = np.array(PSE_timeseries_SI_lungMask_MedianALLSlice)
        #
        plotting_param_chosen = 1
        plot_singleMetric_PSEtimeseries_EditPlot(plot_time_value, echo_num, \
            PSE_timeseries_EachSlice, \
            plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
            saving_details_SI, showplot, dpi_save, metric_chosen, '', 0, stat_plotted_string)
        # Plot with limited y-axis range?
        # if PSE_yaxis_range[0] == 1:
        plot_singleMetric_PSEtimeseries_EditPlot_RANGE(plot_time_value, echo_num, \
            PSE_timeseries_EachSlice, \
            plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
            saving_details_SI, showplot, dpi_save, metric_chosen, '', \
            PSE_yaxis_range, 0, stat_plotted_string)






    # # # PSE MAP PLOTS:
    if which_plots[2] == 1:
        # # For MRI SI plotting.
        # if what_to_plot[1] == 1:
        PSE_image_plot = np.array(PSE_image_SI_all)
        plotting_param_chosen = 1
        # Plot colourbar limits set, multiplied, or both versions.
        if which_plots[11] == 1:
            plot_singleMetric_PSEmaps_SIandRecon_EditPlot(PSE_image_plot, \
                MRI_params[1], saving_details_SI, showplot, dpi_save, \
                plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                metric_chosen, '', RunNum_use, \
                clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25])

        if which_plots[10] == 1:
            plot_singleMetric_PSEmaps_SIandRecon_setCBAR_EditPlot(PSE_image_plot, \
                MRI_params[1], saving_details_SI, showplot, dpi_save, \
                plotting_param_name_list[plotting_param_chosen], plotting_param_name_list_titles[plotting_param_chosen], \
                metric_chosen, '', RunNum_use, \
                clims_overlay, \
                clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25])


    # # # PLOTTING OVERLAYS:
    # Alternative to plotting PSE maps etc...
    if which_plots[8] == 1:
        import functions_Plot_Overlay
        # For SI PSE plotting.
        # if what_to_plot[1] == 1:
        plotting_param_chosen = 1
        functions_Plot_Overlay.plot_Map_Overlay_onSI_Scaling(load_data_echo1_regOnly, np.squeeze(PSE_image_SI_all[0,:,:,:]), dir_date, dir_subj, \
            MRI_params[1], echo_num, '', metric_chosen, \
            plotting_param_name_list[plotting_param_chosen]+'_PSE', masks_reg_data_cardiac, \
            save_plots, showplot, dpi_save, \
            saving_details, which_plots, \
            clims_overlay[0], clims_overlay[1], \
            clims_overlay[2], clims_overlay[3], \
            0)
    
    return





if __name__ == "__main__":
    # ...
    a = 1
