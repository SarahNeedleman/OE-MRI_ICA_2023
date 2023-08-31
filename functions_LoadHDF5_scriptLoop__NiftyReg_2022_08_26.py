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

# Functions to load HDF5 file containing the ICA components
# and reconstruct the components and calculate their ordering.

import numpy as np
from datetime import date
import os
import json
import h5py
import csv
import Sarah_ICA
import functions_PlottingICAetc_testing_Subplots
import functions_Calculate_stats
import functions_loadHDF5_ordering
import functions_loadHDF5_ordering__NiftyReg_2022_08_26
import functions_preproc_MRI_data
import functions_RunICA_andPlot_additional__NiftyReg_2022_08_26

# For caclulating stats wrt the mean image of air cycle 1.
def HDF5_Loop_CalculatePSEwrtCycle1(hdf5_loc, dir_date, dir_subj, \
    metric_chosen, av_im_num, dictionary_name, \
    echo_num, subject_wide_HDF5_name, save_PSE_subject_wide, \
    images_saving_info_list, MRI_scan_directory, MRI_params, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use, g_num, other_parameters, \
    save_plots=0, showplot=0, do_plots=0, do_plots_2=1, do_plots_original=0, \
    plot_components_vs_metrics=1, component_recon=0):
    """
    Function to:
    - Load ICA outputs from an individual subject's HDF5 file.
    - Calculate ordering of the components - for a *single* component.
    - Reconstruct data from a specific component (unMean and unScale options).
    - Calculate the PSE of the reconstructed data and compare to the PSE of
        the registered SI MRI data. (Both maps and timeseries, can plot both.)
    - Calculate stats for the reconstructed and SI PSE values (maps specifically).
    - Save PSE to each subject's HDF5 file.
    - Assume g21 (> 21 components).

    "V2"
    ** PSE calculation are wrt the mean air image during cycle 1.**
    ** PSE calculations are also done for each oxygen cycle and for
    a mean of the oxygen cycles (last av_im_num of images in each cycle).
    ** Stats here are [mean, median, range, IQR, stdev].
    # Different from later (V1) function HDF5_Loop_all.
    

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
    None, saves/plots figures and saves PSE if required.
    
    """
    if MRI_params[3][0] == 1:
        # Cycled 3 PSE calculation
        stats_PSE_order = ["allCycles", "Cycle1", "Cycle2", "Cycle3"]
        stats_eachSlice_all_lung, stats_ALLSlice_all_lung, PSE_timeseries_withinLung, \
            maskedMaps_lung, maskedMaps_lung_MeanEachSlice, PSE_image_OverallandCycled, load_data_echo1_regOnly, \
            sorted_array_metric_sorted, sorted_array, S_ica_array, freq_S_ica_array, masks_reg_data_cardiac, stats_list \
            = functions_RunICA_andPlot_additional__NiftyReg_2022_08_26.PSE_Calculation__CalculatePSE_Cycled3(\
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

    PSE_timeseries_SI_lungMask_MeanEachSlice = PSE_timeseries_withinLung[0]
    PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice = PSE_timeseries_withinLung[1]

    load_data_echo1_regOnly_maskedMaps_lung = maskedMaps_lung[0]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lung = maskedMaps_lung[1]

    load_data_echo1_regOnly_maskedMaps_lungMask_MeanEachSlice = maskedMaps_lung_MeanEachSlice[0]
    X_recon_OE_unMean_unSc_maps_maskedMaps_lungMask_MeanEachSlice = maskedMaps_lung_MeanEachSlice[1]

    PSE_image_SI_all = PSE_image_OverallandCycled[0]
    PSE_image_unMean_unSc_all = PSE_image_OverallandCycled[1]

    size_echo_data = np.shape(load_data_echo1_regOnly)

    # # Save PSE stats to subject-wide HDF5 file.
    # Use the created function.
    
    if save_PSE_subject_wide == 1:
        add_PSE_subjectWideHDF5_wrtCycle1_nifty(hdf5_loc, subject_wide_HDF5_name, \
            echo_num, dir_date, dir_subj, metric_chosen, sorted_array_metric_sorted, \
            stats_list, stats_PSE_order, \
            stats_eachSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_SI_all_lung, \
            stats_eachSlice_PSE_recon_all_lung, stats_ALLSlice_PSE_recon_all_lung, \
            NiftyReg_ID, densityCorr, regCorr_name, \
            ANTs_used, dir_image_nifty, hdf5_file_name_base, \
            RunNum_use, other_parameters[5])


    print(dir_date + '_' + dir_subj, metric_chosen)
    return


def add_PSE_subjectWideHDF5_wrtCycle1_nifty(hdf5_loc, subject_wide_HDF5_name, \
    echo_num, dir_date, dir_subj, metric_chosen, sorted_array_metric_sorted, \
    stats_list, stats_PSE_order, \
    stats_eachSlice_PSE_SI_all_lung, stats_ALLSlice_PSE_SI_all_lung, \
    stats_eachSlice_PSE_recon_all_lung, stats_ALLSlice_PSE_recon_all_lung, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use, av_im_num):
    """
    Function to save a particular subject's PSE stats ("V2") to the subject-wide HDF5 file.
    For a specific metric, assuming g21 and stats calculated within the lung masks.

    *** WILL OVERWRITE **

    Arguments:
    hdf5_loc = location of the HDF5 file which contains the metric information.
    subject_wide_HDF5_name = name of the HDF5 file to contain the "V2" PSE
    stats for *ALL* subjects.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.    
    dir_date = scan date for the particular subject.
    dir_subj = subject ID for the particular subject.
    metric_chosen = the metric being investigated (only one now), in the form of a 
    list containing only one element (the name of the metric being investigated).
    sorted_array_metric_sorted = sorted component details - sorted by metric value (2nd 
    column) and including the corresponding number of components (1st column). For
    g21 and a specific metric. Shape of (different_component_Num, 2,1) with the final
    index previously running over the different metrics, however only a single metric is under
    investigation here.
    stats_list = list of the stats to be found. "V2" ==> ["mean", "median", "range", "IQR", "stdev"].
    stats_PSE_order = list of the order in which the PSE stats are being saved (in array form). Again,
    "V2" ==> ["allCycles", "Cycle1", "Cycle2", "Cycle3"].
    stats_eachSlice_PSE_..._all_lung = PSE stats calculated for each slice (within the lung masks).
    stats_ALLSlice_PSE_..._all_lung = PSE stats calculated over all slices (within the lung masks).
    Both stats_..._PSE_..._all_lung arrays will be for SI data and reconstructed data.
    stats_eachSlice_PSE_SI_all_lung has shape (NumSlices, len(stats_list)).
    stats_AllSlice_PSE_SI_all_lung has shape (,len(stats_list))
    stats_eachSlice_PSE_recon_all_lung has shape (NumSlices, len(stats_list)).
    stats_AllSlice_PSE_recon_all_lung has shape (,len(stats_list)).
    i.e. same shapes for SI and recon for each type as only investigating a single metric.
    RunNum_use = number of the Run to save the PSE stats under.
    Returns:
    None, HDF5 added to and closed.
    
    """
    # Save PSE stats to subject-wide HDF5 file.
    # Open subject wide PSE HDF5 file.
    f = h5py.File(hdf5_loc + subject_wide_HDF5_name + '_Echo' + str(echo_num), 'r+')
    # Adding stats as:
    # 1) Subject ID
    # 2) Metric name
    # 3) As attributes for the metric and subject, add in the NumC and metric_value
    # for this component.
    # Also adding as an attribute the stats ordering:
    # stats_list = ["Mean", "Median", "Range", "IQR", "stdev"]
    # And also the order of the PSE stats:
    # stats_PSE_order = ["allCycles", "Cycle1", "Cycle2", "Cycle3"]
    # Add both of these to the file's attributes (directly to f).
    # Only do once - e.g. when file is created, or repeat to make sure.
    # 4) PSE each type (all slices vs each slice and Recon vs SI --> x4)
    # **Assuming within the lung** and g21. Also for a specific metric.

    subject_key = dir_date + '_' + dir_subj

    # 1) Check if subject is (not) present. If present, rewrite data later, if not, create group.
    if not subject_key in f:
        f.create_group(subject_key)

    # 2) Check if metric is (not) present. If not, create metric subgroup.
    if not subject_key + '/' + metric_chosen[0] in f:
        f[subject_key].create_group(metric_chosen[0])


    # Within this group, include registered and registration type...
    # Additional group names
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID


    additional_group_names = reg_applied + "_" + reg_type 
    group_name_full = subject_key + '/' + metric_chosen[0] + '/' + additional_group_names

    # Add main subgroups one at a time as/when required, but before adding, need to check
    # if the required subgroup is already present.
    if not group_name_full in f:
        subgroup_ICA = f.create_group(group_name_full) # 2) ICA applied to SI
    else:
        # Set variable to the subgroup name for use later when referring to the subgroup.
        subgroup_ICA = f[group_name_full]


    # # Here, if earlier oxy mean calculation... add in indicator in the subgroup_ICA name to 
    # differentiate/identify.
    # Do this if len(av_im_num) == 4.
    # And the name to use in the subgroup name is av_im_num[3]
    if len(av_im_num) == 4:
        # "Alternative//ealier oxy mean image calculation".
        # 2023/08/04 - need to add NumC_OE above Cycled3Version...
        group_name_orig = group_name_full[:]
        group_name_full = group_name_full + "/" + av_im_num[3]
        # Copied from above...
        if not group_name_full in f:
            subgroup_ICA = f.create_group(group_name_full) # 2) ICA applied to SI
        else:
            # Set variable to the subgroup name for use later when referring to the subgroup.
            subgroup_ICA = f[group_name_full]
            # And include oxy mean numbers for future reference.
            f[group_name_full].attrs['mean_oxy_dyn'] = av_im_num[0][1]


    # # If RunNum_use == 1, add PSE stats following original method.
    # If not, add under a RunNum group...
    if RunNum_use == 1:
        # 3) Add attributes for this subject - NumC and metric_value.
        f[group_name_full].attrs['NumC'] = np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))
        f[group_name_full].attrs['metric_value'] = sorted_array_metric_sorted[0,1,0]
        f.attrs["stats_list"] = stats_list
        f.attrs["stats_PSE_order"] = stats_PSE_order
        f[group_name_orig].attrs['NumC'] = np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))
        f[group_name_orig].attrs['metric_value'] = sorted_array_metric_sorted[0,1,0]
        #
        # 4) Add the datasets - PSE stats over all slices and for each slice, for SI and recon.
        # (Including all cycles PSE stats within each, in the order allCycles, Cycle1, Cycle2, Cycle3).
        # Check if present (subgroups) - delete datasets if present.
        # subgroup_general = subject_key + '/' + metric_chosen[0]
        #
        # # Additional
        #
        # First, PSE stats for each slice of the SI data.
        if not group_name_full + '/' + 'PSE_stats_EachSlice_SI' in f:
            f[group_name_full].create_dataset('PSE_stats_EachSlice_SI', data=stats_eachSlice_PSE_SI_all_lung)
        else:
            del f[group_name_full + '/' + 'PSE_stats_EachSlice_SI']
            f[group_name_full].create_dataset('PSE_stats_EachSlice_SI', data=stats_eachSlice_PSE_SI_all_lung)
        #
        # Second, PSE stats for all slices of the SI data.
        if not group_name_full + '/' + 'PSE_stats_Overall_SI' in f:
            f[group_name_full].create_dataset('PSE_stats_Overall_SI', data=stats_ALLSlice_PSE_SI_all_lung)
        else:
            del f[group_name_full + '/' + 'PSE_stats_Overall_SI']
            f[group_name_full].create_dataset('PSE_stats_Overall_SI', data=stats_ALLSlice_PSE_SI_all_lung)
        #
        # Third, PSE stats for each slice of the recon (unMean and unScaled) data.
        if not group_name_full + '/' + 'PSE_stats_EachSlice_recon' in f:
            f[group_name_full].create_dataset('PSE_stats_EachSlice_recon', data=stats_eachSlice_PSE_recon_all_lung)
        else:
            del f[group_name_full + '/' + 'PSE_stats_EachSlice_recon']
            f[group_name_full].create_dataset('PSE_stats_EachSlice_recon', data=stats_eachSlice_PSE_recon_all_lung)
        #
        # Fourth, PSE stats for all slices of the recon (unMean and unScaled) data.
        if not group_name_full + '/' + 'PSE_stats_Overall_recon' in f:
            f[group_name_full].create_dataset('PSE_stats_Overall_recon', data=stats_ALLSlice_PSE_recon_all_lung)
        else:
            del f[group_name_full + '/' + 'PSE_stats_Overall_recon']
            f[group_name_full].create_dataset('PSE_stats_Overall_recon', data=stats_ALLSlice_PSE_recon_all_lung)
    else:
        # # Add RunNum as a subgroup, and then add the usual/original PSE stats etc within.
        # Add main subgroups one at a time as/when required, but before adding, need to check
        # if the required subgroup is already present.
        group_name_full_Run = group_name_full + '/Run' + str(RunNum_use)
        if not group_name_full_Run in f:
            subgroup_ICA_Run = f.create_group(group_name_full_Run) # 2) ICA applied to SI
        else:
            # Set variable to the subgroup name for use later when referring to the subgroup.
            subgroup_ICA_Run = f[group_name_full_Run]
        #
        #
        # 3) Add attributes for this subject - NumC and metric_value.
        f[group_name_full_Run].attrs['NumC'] = np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))
        f[group_name_full_Run].attrs['metric_value'] = sorted_array_metric_sorted[0,1,0]
        f[group_name_orig].attrs['NumC'] = np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))
        f[group_name_orig].attrs['metric_value'] = sorted_array_metric_sorted[0,1,0]
        #
        #
        #
        #
        # 4) Add the datasets - PSE stats over all slices and for each slice, for SI and recon.
        # (Including all cycles PSE stats within each, in the order allCycles, Cycle1, Cycle2, Cycle3).
        # Check if present (subgroups) - delete datasets if present.
        # subgroup_general = subject_key + '/' + metric_chosen[0]
        #
        # # Additional
        #
        # First, PSE stats for each slice of the SI data.
        if not group_name_full_Run + '/' + 'PSE_stats_EachSlice_SI' in f:
            f[group_name_full_Run].create_dataset('PSE_stats_EachSlice_SI', data=stats_eachSlice_PSE_SI_all_lung)
        else:
            del f[group_name_full_Run + '/' + 'PSE_stats_EachSlice_SI']
            f[group_name_full_Run].create_dataset('PSE_stats_EachSlice_SI', data=stats_eachSlice_PSE_SI_all_lung)
        #
        # Second, PSE stats for all slices of the SI data.
        if not group_name_full_Run + '/' + 'PSE_stats_Overall_SI' in f:
            f[group_name_full_Run].create_dataset('PSE_stats_Overall_SI', data=stats_ALLSlice_PSE_SI_all_lung)
        else:
            del f[group_name_full_Run + '/' + 'PSE_stats_Overall_SI']
            f[group_name_full_Run].create_dataset('PSE_stats_Overall_SI', data=stats_ALLSlice_PSE_SI_all_lung)
        #
        # Third, PSE stats for each slice of the recon (unMean and unScaled) data.
        if not group_name_full_Run + '/' + 'PSE_stats_EachSlice_recon' in f:
            f[group_name_full_Run].create_dataset('PSE_stats_EachSlice_recon', data=stats_eachSlice_PSE_recon_all_lung)
        else:
            del f[group_name_full_Run + '/' + 'PSE_stats_EachSlice_recon']
            f[group_name_full_Run].create_dataset('PSE_stats_EachSlice_recon', data=stats_eachSlice_PSE_recon_all_lung)
        #
        # Fourth, PSE stats for all slices of the recon (unMean and unScaled) data.
        if not group_name_full_Run + '/' + 'PSE_stats_Overall_recon' in f:
            f[group_name_full_Run].create_dataset('PSE_stats_Overall_recon', data=stats_ALLSlice_PSE_recon_all_lung)
        else:
            del f[group_name_full_Run + '/' + 'PSE_stats_Overall_recon']
            f[group_name_full_Run].create_dataset('PSE_stats_Overall_recon', data=stats_ALLSlice_PSE_recon_all_lung)


    # Close the HDF5 file (subject-wide).
    f.close()
    # Nothing to return, file re-written/data added to and file saved.
    return



def savePSE_metrics_dictionary_excel_wrtCycle1_nifty(hdf5_loc, subject_wide_HDF5_name, \
    metric_chosen, file_loc, PSE_file_wrtCycle1, echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum, subjectID_list):
    """
    **For Cycled3 PSE storage**

    Function to extract PSE metrics from the HDF5 files of different subjects, 
    store in a dictionary and write the dictionary to a csv file which can 
    be opened in Excel.

    **EACH** metric to be saved separately. SPECIFIC metric saved is
    determined by the input metric_chosen.
    V2 method: Assuming > 21 components; IQR included in stats.

    2022/09/28 - reduce stats output - not all cycles, etc...
    PSE for _Overall_ = for all slices, and have shape (4,5) with 
    4 corresponding to the PSE stats value over all cycles and then each
    over of the three oxygen cycles, and 5 corresponding to each PSE metric - 
    ["mean", "median", "range", "IQR", "stdev"].
    --> Reduce to overall slices and overall oxygen cycles, so that
    the PSE stats shape will have 1 row. ((1st row))
    --> Reduce to median value only, so that the PSE stats shape
    will have 1 column. ((2nd column))
    --> Reduced shape to (1,1).

    # Excel:
    # 1. =clean(cells)
    # 2. Copy text only (avoid formulae).
    # 3. Delete columns from inputs of steps 1 and 2.
    # 4. Find and Replace: (,),{,},[,], array, with space.
    # 5. > Data > Text-to-columns. Select delimited with Tab, Comma, Space and treat consecutive delimiters as one. Text qualifier is '.
    # 6. Mean, Median, Min, Max, Range, Stdev
    # 7. Can copy smoking info after ordering subjects by value (column).
    # 8. Then sort by current smoker/smoking status


    Arguments:
    hdf5_loc = location of the HDF5 files.
    subject_wide_HDF5_name = the name of the subject-wide HDF5 file containing 
    the PSE stats for all subjects which is to be opened and read from.
    metric_chosen = the metric being investigated (only one now), in the form of a 
    list containing only one element (the name of the metric being investigated).
    file_loc = location of the csv file to be saved.
    PSE_file_wrtCycle1 = name of the PSE csv file to be saved.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    Returns:
    None, extracts PSE data from each subject's HDF5 file and saves as a csv file.

    """
    # Open the HDF5 file which contains the PSE stats for **all** subjects for
    # the **specific** metric of interest.
    f = h5py.File(hdf5_loc + subject_wide_HDF5_name + '_Echo' + str(echo_num), 'r')

    # Single dictionary to store the PSE stats for SI and recon data, over all slices
    # and for each slice (to store for all subjects).
    PSE_dictionary = {}

    # Loop over all subjects (keys) in the HDF5 file and extract the PSE stats.
    # for subject_key in f.keys():
    for subject_key in subjectID_list:
        print(subject_key)
        for reg_type in f[subject_key][metric_chosen[0]].keys():
            print(reg_type)
            PSE_reg_type_SI = "PSE_SI_Overall_" +  reg_type
            PSE_reg_type_recon = "PSE_recon_Overall_" +  reg_type
            if not PSE_reg_type_SI in PSE_dictionary:
                if RunNum == 1:
                    # Usual
                    PSE_dictionary[PSE_reg_type_SI] = {\
                        subject_key : f[subject_key][metric_chosen[0]][reg_type]['PSE_stats_Overall_SI'][0,1]}
                    PSE_dictionary[PSE_reg_type_recon] = {\
                        subject_key : f[subject_key][metric_chosen[0]][reg_type]['PSE_stats_Overall_recon'][0,1]}
                else:
                    PSE_dictionary[PSE_reg_type_SI] = {\
                        subject_key : f[subject_key][metric_chosen[0]][reg_type]['Run' + str(RunNum)]['PSE_stats_Overall_SI'][0,1]}
                    PSE_dictionary[PSE_reg_type_recon] = {\
                        subject_key : f[subject_key][metric_chosen[0]][reg_type]['Run' + str(RunNum)]['PSE_stats_Overall_recon'][0,1]}
            else:
                # Check for metric chosen and then reg_type etc...
                if RunNum == 1:
                    # Usual
                    PSE_dictionary[PSE_reg_type_SI].update({subject_key : f[subject_key][metric_chosen[0]][reg_type]['PSE_stats_Overall_SI'][0,1]})
                    PSE_dictionary[PSE_reg_type_recon].update({subject_key : f[subject_key][metric_chosen[0]][reg_type]['PSE_stats_Overall_recon'][0,1]})
                else:
                    PSE_dictionary[PSE_reg_type_SI].update({subject_key : f[subject_key][metric_chosen[0]][reg_type]['Run' + str(RunNum)]['PSE_stats_Overall_SI'][0,1]})
                    PSE_dictionary[PSE_reg_type_recon].update({subject_key : f[subject_key][metric_chosen[0]][reg_type]['Run' + str(RunNum)]['PSE_stats_Overall_recon'][0,1]})

    # Close the HDF5 file for the current subject.
    f.close()


    if RunNum == 1:
        # Usual name 
        csv_file_name = file_loc + PSE_file_wrtCycle1 + '_Echo' + str(echo_num) + '.csv'
    else:
        csv_file_name = file_loc + PSE_file_wrtCycle1 + '_Echo' + str(echo_num) + '_Run' + str(RunNum) + '.csv'

    # To save as a CSV file.
    with open(csv_file_name, 'w') as output:
        writer = csv.writer(output)
        for key, value in PSE_dictionary.items():
            writer.writerow([key, value])

    return #PSE_dictionary




def save_OEmetric_dictionary_nifty(hdf5_loc, subject_wide_HDF5_name, \
    metric_chosen, file_details_create, echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum, subjectID_list):
    """
    Function to extract PSE metrics from the HDF5 files of different subjects, 
    store in a dictionary and write the dictionary to a csv file which can 
    be opened in Excel.

    # Metric value (so for correlation, will be the inverse of the Spearman
    # correlation) of the OE component.
    --> --> Only for the specific NiftyReg_ID, densityCorr, regCorr_name, ANTs_used input.

    **EACH** metric to be saved separately. SPECIFIC metric saved is
    determined by the input metric_chosen.
    V2 method: Assuming > 21 components; IQR included in stats.

    2022/09/28 - reduce stats output - not all cycles, etc...
    PSE for _Overall_ = for all slices, and have shape (4,5) with 
    4 corresponding to the PSE stats value over all cycles and then each
    over of the three oxygen cycles, and 5 corresponding to each PSE metric - 
    ["mean", "median", "range", "IQR", "stdev"].
    --> Reduce to overall slices and overall oxygen cycles, so that
    the PSE stats shape will have 1 row. ((1st row))
    --> Reduce to median value only, so that the PSE stats shape
    will have 1 column. ((2nd column))
    --> Reduced shape to (1,1).

    # Excel:
    # 1. =clean(cells)
    # 2. Copy text only (avoid formulae).
    # 3. Delete columns from inputs of steps 1 and 2.
    # 4. Find and Replace: (,),{,},[,], array, with space.
    # 5. > Data > Text-to-columns. Select delimited with Tab, Comma, Space and treat consecutive delimiters as one. Text qualifier is '.
    # 6. Mean, Median, Min, Max, Range, Stdev
    # 7. Can copy smoking info after ordering subjects by value (column).
    # 8. Then sort by current smoker/smoking status


    Arguments:
    hdf5_loc = location of the HDF5 files.
    subject_wide_HDF5_name = the name of the subject-wide HDF5 file containing 
    the PSE stats for all subjects which is to be opened and read from.
    metric_chosen = the metric being investigated (only one now), in the form of a 
    list containing only one element (the name of the metric being investigated).
    file_loc = location of the csv file to be saved.
    PSE_file_wrtCycle1 = name of the PSE csv file to be saved.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    file_details_create = [file_loc, PSE_file_wrtCycle1, file_name, PSE_file_wrtCycle1_start, PSE_file_wrtCycle1_ending]
    Returns:
    None, extracts PSE data from each subject's HDF5 file and saves as a csv file.

    """
    invert_metric = file_details_create[5]

    # Open the HDF5 file which contains the PSE stats for **all** subjects for
    # the **specific** metric of interest.
    f = h5py.File(hdf5_loc + subject_wide_HDF5_name + '_Echo' + str(echo_num), 'r')

    # 3) Within this group, include registered and registration type...
    # Additional group names
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    additional_group_names = reg_applied + "_" + reg_type 

    # Single dictionary to store the PSE stats for SI and recon data, over all slices
    # and for each slice (to store for all subjects).
    PSE_dictionary = {}
    # Loop over all subjects (keys) in the HDF5 file and extract the PSE stats.
    # for subject_key in f.keys():
    for subject_key in subjectID_list:
        print(subject_key)
        # f[subject_key][metric_chosen[0]][additional_group_names]
        PSE_reg_type_recon = "OE_metric_value"
        if not PSE_reg_type_recon in PSE_dictionary:
            # if not subject_key in PSE_dictionary:
            if RunNum == 1:
                # Usual
                if invert_metric == 0:
                    PSE_dictionary[PSE_reg_type_recon] = {\
                        subject_key : f[subject_key][metric_chosen[0]][additional_group_names].attrs['metric_value']}
                else:
                    PSE_dictionary[PSE_reg_type_recon] = {\
                        subject_key : 1/f[subject_key][metric_chosen[0]][additional_group_names].attrs['metric_value']}
            else:
                if invert_metric == 0:
                    PSE_dictionary[PSE_reg_type_recon] = {\
                        subject_key : f[subject_key][metric_chosen[0]][additional_group_names]['Run' + str(RunNum)].attrs['metric_value']}
                else:
                    PSE_dictionary[PSE_reg_type_recon] = {\
                        subject_key : 1/f[subject_key][metric_chosen[0]][additional_group_names]['Run' + str(RunNum)].attrs['metric_value']}
        else:
            # Check for metric chosen and then reg_type etc...
            if RunNum == 1:
                # Usual
                if invert_metric == 0:
                    PSE_dictionary[PSE_reg_type_recon].update({subject_key : f[subject_key][metric_chosen[0]]\
                        [additional_group_names].attrs['metric_value']})
                else:
                    PSE_dictionary[PSE_reg_type_recon].update({subject_key : 1/f[subject_key][metric_chosen[0]]\
                        [additional_group_names].attrs['metric_value']})
            else:
                if invert_metric == 0:
                    PSE_dictionary[PSE_reg_type_recon].update({subject_key : f[subject_key][metric_chosen[0]]\
                        [additional_group_names]['Run' + str(RunNum)].attrs['metric_value']})
                else:
                    PSE_dictionary[PSE_reg_type_recon].update({subject_key : 1/f[subject_key][metric_chosen[0]]\
                        [additional_group_names]['Run' + str(RunNum)].attrs['metric_value']})

    # Close the HDF5 file for the current subject.
    f.close()

    # CSV filename
    # file_details_create = [file_loc, PSE_file_wrtCycle1, file_name, PSE_file_wrtCycle1_start, PSE_file_wrtCycle1_ending]
    file_loc = file_details_create[0]
    PSE_file_wrtCycle1_start = file_details_create[3]
    PSE_file_wrtCycle1_ending = file_details_create[4]
    csv_file_name_middle = 'OEmetricValue' + '___ICA___' + metric_chosen[0] + PSE_file_wrtCycle1_ending + '_' + NiftyReg_ID + '_Echo' + str(echo_num)

    if RunNum == 1:
        # Usual name 
        csv_file_name = file_loc + csv_file_name_middle
    else:
        csv_file_name = file_loc + csv_file_name_middle + '_Run' + str(RunNum)

    if invert_metric == 1:
        csv_file_name = csv_file_name + '_MetricInverted'

    # To save as a CSV file.
    with open(csv_file_name + '.csv', 'w') as output:
        writer = csv.writer(output)
        for key, value in PSE_dictionary.items():
            writer.writerow([key, value])

    return #PSE_dictionary



if __name__ == "__main__":
    # ...
    a = 1

