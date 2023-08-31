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

# Main function called for running ICA, saving the results, and
# plotting.

import h5py
import os
import functions_LoopSave_ICA_HDF5__NiftyReg_2022_08_26
import functions_LoadHDF5_scriptLoop__NiftyReg_2022_08_26
import functions_RunICA_andPlot_additional__NiftyReg_2022_08_26

def run_ICA_PCA_full(subjectID_list, \
    saving_YN, other_parameters, file_details_echo_num, file_details_create, \
    component_plotting_params, plotting_list_argument, \
    argument_running, save_show_plots, \
    MRI_scan_directory, MRI_params, clims_overlay, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    freq_spectra_plotting_options, RunNum_use, \
    g_num, num_figures_plot_components, \
    subject_wide_PCA_HDF5_file_name, PSE_yaxis_range, \
    plot_MRI_PSE_maps_timeseries_only, plot_mean_median):
    """
    Function to:
    - Run ICA repeatedly over an increasing number of components. 
    - PCA is run once for each subject.
    - The ICA components can be saved to HDF5 files.
    - For both component analysis methods, the components are ordered and 
    the OE component is identified.
    - The OE component is reconstructed, and the PSE is calculated and compared
    to the PSE of motion-corrected MRI SI data.
    - The PSE stats are saved to HDF5 files and can be written to a csv file.
    
    Plotting:
    - Component timeseries and frequency spectra, frequency spectra can be plotted
    scaled by the OE gas cycling frequency.
    - SI images.
    (In functions_RunICA_andPlot_additional.py, for reconstructed OE component and/or
    MRI SI data...
    - Plot PSE timeseries.
    - Plot SI timeseries (MRI SI data and reconstructed OE component).
    - Plot frequency spectra and timeseries of the OE component.
    - Plot PSE maps.
    - Plot all component maps.
    - Plot components/PSE overlay on SI images.


    Arguments:
    subjectID_list = list of subject(s) to be investigated.
    saving_YN = [save_PSE_subject_wide, save_ordering_HDF5, save_csv_PCAICA, save_csv_PSE_a_metric, \
        save_csv_OE_metric_ICA, save_csv_OE_metric_PCA, save_csv_PSE_PCA].
    other_parameters = [NumEchoes, num_c_start, num_c_stop, ordering_method_list, num_c_use, av_im_num].
    file_details_echo_num = [subject_wide_HDF5_name, hdf5_loc, echo_num, dictionary_name].
    file_details_create = [file_loc, PSE_file_wrtCycle1, file_name, PSE_file_wrtCycle1_start, PSE_file_wrtCycle1_ending, invert_metric]
    component_plotting_params = [CA_type_list, subplots_num_in_figure_list, subplots_num_row_list, \
        subplots_num_in_figure, subplots_num_row]
    plotting_list_argument = [dpi_save, what_to_plot, which_plots, images_saving_info_list, clim_lower, clim_upper].
    argument_running = [do_runICA, do_ICA_PSE_calculation, do_ICAPCA_comparison, do_anyPlotting, do_ICA_ordering]
    save_show_plots = [showplot, save_plots] - 0/1 for No/Yes for the main plotting functions.
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes]
    clims_overlay = [clims_multi_lower_list, clims_multi_upper_list, \
        clims_lower_list, clims_upper_list, clims_multi_componentOverlay_list]
    freq_spectra_plotting_options = a list of 1/0 for Y/N for plotting:
    - [0] frequency spectra scaled by the mean OE gas cycling frequency *amplitude*;
    - [1] frequency spectra scaled by the maximum frequency *amplitude* (per spectra);
    - [2] unscaled - as output by ICA - but could have any scaling.
    RunNum_use = the specific RunNum to use when calculating PSE/ordering.
    g_num = number of components greater than when ordering (e.g. 21 for 22 and greater).
    PSE_yaxis_range = [plot_PSE_yaxis_limited, PSE_yaxis_lower, PSE_yaxis_upper] = for
    if plotting a set y-axis range on the PSE timeseries plots.
    #
    GasSwitch = [Cycled3, gas_switching], where Cycled3 = 1/0 for Yes/No for cycled oxygen
    use. If 0, assume SingleCycle. gas_switching may be a list of switching times if SingleCycle,
    of a single repeated number of dynamics at which the gases are switched if cyclic.
    av_im_num = av_im_num if Cycled3. Otherwise = [[air_mean, oxy_mean], assume_alpha], where
    assume_alpha = assumed upslope/downslope (s) times for creating synthetic input function.
    #
    # See script version for details of the argument lists.
    #
    Returns:
    None - plots saved and CSV/HDF5 files created as required.

    """
    save_plots = 0; showplot = 0; do_plots = 0; do_plots_2 = 0; do_plots_original = 0
    plot_cmap = 0; plot_all = 0; plot_components_vs_metrics = 0; plot_variance = 0
    component_recon = 1 

    # If saving subject-wide HDF5 for PSE stats (save_PSE_subject_wide):
    if saving_YN[0] == 1:
        # If not present, need to create for the first time.
        if os.path.exists(file_details_echo_num[1] + file_details_echo_num[0] + '_Echo' + str(file_details_echo_num[2])) == False:
            # Create file and then close so can be written to in the future.
            f = h5py.File(file_details_echo_num[1] + file_details_echo_num[0] + '_Echo' + str(file_details_echo_num[2]), 'w')
            f.close()

    # Loop over all subjects in subjectID_list supplied.
    for subjectID_date in subjectID_list:
        dir_date = subjectID_date[:8]
        dir_subj = subjectID_date[9:]

        if argument_running[0] == 1:
            # # # # 1. Run ICA - for all subjects in subjectID_list supplied.
            functions_LoopSave_ICA_HDF5__NiftyReg_2022_08_26.loop_ICA_save_metrics_details(\
                dir_date, dir_subj, other_parameters[1], other_parameters[2], other_parameters[3], \
                file_details_echo_num[2], file_details_echo_num[3], \
                file_details_echo_num[1], plotting_list_argument[3], MRI_scan_directory, MRI_params, \
                NiftyReg_ID, densityCorr, regCorr_name, \
                ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                save_plots, showplot, plot_cmap, plot_all, \
                other_parameters)

        if argument_running[4] == 1:
            # Calculate ordering - only if ICA has already been run.
            if argument_running[0] == 0:
                functions_LoopSave_ICA_HDF5__NiftyReg_2022_08_26.loopICA_HDF5save_plot_order_nifty_ONLY_ORDERING(\
                    other_parameters[1], other_parameters[2], other_parameters[3], \
                    MRI_params[2], MRI_params[0], MRI_params[3], \
                    file_details_echo_num[1], dir_date, dir_subj, 'SI', 'cardiac', file_details_echo_num[2], \
                    NiftyReg_ID, densityCorr, regCorr_name, \
                    ANTs_used, hdf5_file_name_base, \
                    RunNum_use, \
                    other_parameters, MRI_params)

        if argument_running[1] == 1:
            # Reconstruct ICA components and calculate PSE
            functions_LoadHDF5_scriptLoop__NiftyReg_2022_08_26.HDF5_Loop_CalculatePSEwrtCycle1(file_details_echo_num[1], dir_date, dir_subj, \
                other_parameters[3], other_parameters[5], file_details_echo_num[3], \
                file_details_echo_num[2], file_details_echo_num[0], saving_YN[0], \
                plotting_list_argument[3], MRI_scan_directory, MRI_params, \
                NiftyReg_ID, densityCorr, regCorr_name, \
                ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                RunNum_use, g_num, other_parameters, \
                save_plots, showplot, do_plots, do_plots_2, do_plots_original, \
                plot_components_vs_metrics, component_recon)

        if argument_running[3] == 1:
            # # # # PLOTTING
            # if plotting_list_argument[2][6] == 1:
            if plotting_list_argument[2][6] == 1:
                # Plot ICA and PCA component timeseries and frequency spectra, 
                # with the frequency spectra scaled by the amplitude of the gas cycling frequency.
                for CA_type in component_plotting_params[0]:
                    # $$££$$££
                    functions_RunICA_andPlot_additional__NiftyReg_2022_08_26.Plot_ICAPCA_final_scaled_nifty(dir_date, dir_subj, \
                        other_parameters[3][0], file_details_echo_num[1], file_details_echo_num[0], save_show_plots[0], save_show_plots[1], \
                        component_plotting_params[1], component_plotting_params[2], \
                        file_details_echo_num[2], CA_type, file_details_echo_num[3], plotting_list_argument[0], \
                        plotting_list_argument[3], MRI_scan_directory, MRI_params, \
                        NiftyReg_ID, densityCorr, regCorr_name, \
                        ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                        freq_spectra_plotting_options, RunNum_use, file_details_create[5])
       
            # # # # Plotting:
            # PSE timeseries, SI timeseries, PSE maps (for OE component and 
            # MRI SI), the OE ICA component and all components (for ICA only).
            functions_RunICA_andPlot_additional__NiftyReg_2022_08_26.HDF5_Loop_CalculatePSEwrtCycle1__plotting_reconOE_PSE(\
                file_details_echo_num[1], dir_date, dir_subj, \
                other_parameters[3], other_parameters[5], file_details_echo_num[3], \
                file_details_echo_num[2], \
                save_show_plots[1], save_show_plots[0], \
                plotting_list_argument[4], plotting_list_argument[5], \
                plotting_list_argument[0], \
                plotting_list_argument[1], plotting_list_argument[2], \
                plotting_list_argument[3], MRI_scan_directory, \
                component_plotting_params[3], component_plotting_params[4], \
                MRI_params, clims_overlay, \
                NiftyReg_ID, densityCorr, regCorr_name, \
                ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                RunNum_use, g_num, other_parameters, PSE_yaxis_range, plot_mean_median)

        print(dir_date + '_' + dir_subj)


        if plot_MRI_PSE_maps_timeseries_only == 1:
            # Plot MRI SI PSE timeseries and maps...
            functions_RunICA_andPlot_additional__NiftyReg_2022_08_26.plot_MRI_SI_PSE_maps_timeseries(dir_date, dir_subj, \
                other_parameters[3], other_parameters[5], file_details_echo_num[3], \
                file_details_echo_num[2], \
                save_show_plots[1], save_show_plots[0], \
                plotting_list_argument[4], plotting_list_argument[5], \
                plotting_list_argument[0], \
                plotting_list_argument[1], plotting_list_argument[2], \
                plotting_list_argument[3], MRI_scan_directory, \
                component_plotting_params[3], component_plotting_params[4], \
                MRI_params, clims_overlay, \
                NiftyReg_ID, densityCorr, regCorr_name, \
                ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                RunNum_use, g_num, other_parameters, PSE_yaxis_range, plot_mean_median)
        
    # Save the PSE stats to a csv file (currently stored in a subject-wide HDF5 file).
    if saving_YN[3] == 1:
        if MRI_params[3][0] == 1:
            functions_LoadHDF5_scriptLoop__NiftyReg_2022_08_26.savePSE_metrics_dictionary_excel_wrtCycle1_nifty(\
                file_details_echo_num[1], file_details_echo_num[0], \
                other_parameters[3], file_details_create[0], file_details_create[1], file_details_echo_num[2], \
                NiftyReg_ID, densityCorr, regCorr_name, \
                ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                RunNum_use, subjectID_list)

    if saving_YN[4] == 1:
        # ICA OE metric value (metric/inverse of correlation value)
        functions_LoadHDF5_scriptLoop__NiftyReg_2022_08_26.save_OEmetric_dictionary_nifty(\
            file_details_echo_num[1], file_details_echo_num[0], \
            other_parameters[3], file_details_create, file_details_echo_num[2], \
            NiftyReg_ID, densityCorr, regCorr_name, \
            ANTs_used, dir_image_nifty, hdf5_file_name_base, \
            RunNum_use, subjectID_list)




    return




if __name__ == "__main__":
    # ...
    a = 1