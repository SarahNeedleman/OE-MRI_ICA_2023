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

# Functions to loop over different numbers of ICA components, apply ICA
# and save the extracted ICA components to an HDF5 file.

import numpy as np
import os
import json
import h5py
import csv
from datetime import date
import Sarah_ICA
import functions_PlottingICAetc_testing_Subplots
import functions_Calculate_stats
import functions_testing_pythonFastICA_errorsWarnings
import functions_preproc_MRI_data

def loop_ICA_save_metrics_details(dir_date, dir_subj, num_c_start, num_c_stop, ordering_method_list, \
    echo_num, dictionary_name, 
    hdf5_loc, images_saving_info_list, MRI_scan_directory, MRI_params, \
    save_plots=0, showplot=0, plot_cmap=0, plot_all=0):
    """
    Function initially loads and preprocesses data ready for ICA application. ICA is
    then applied for a range of different component numbers and the ICA outputs can
    be saved to HDF5 file. Optionally, the components (timeseries, frequency spectra and
    maps) can be plotted and saved.

    Arguments:
    dir_date = subject scanning date.
    dir_subj = subject ID.
    num_c_start = lower number of components to be found with ICA.
    num_c_stop = upper number of components to be found with ICA.
    ordering_method_list = list of different ordering methods (as strings)
    to be investigated.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    dictionary_name = name and location of the dictionary containing the masks and the 
    names of the reference images for the particular subject(s).
    hdf5_loc = location of the HDF5 files.
    images_saving_info_list = details of where to save all images. As a 
    list with values [image_save_loc_general, image_dir_name].
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes]
    save_plots = 0 if No; 1 if Yes - whether to save plots.
    showplot = 0 if No; 1 if Yes - whether to show plots.
    plot_cmap = 0 if No; 1 if Yes - whether to generate component map plots.
    plot_all = 0 if No; 1 if Yes - whether to perform the plotting at all. If No,
    none will be saved as the plotting functions will not be run.
    Returns:
    None - details will be saved to HDF5 file and plots can be saved.

    """
    # MRI images directory and mask directory.
    dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_mask = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[4] + '/'

    # Directory for saving subject-specific plots.
    today = date.today()
    d1 = today.strftime("%Y_%m_%d")

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

    # Info for plotting - whether to save, title, saving directory and saving name details etc...
    plotting_param = 'SI_regEcho' + str(echo_num)
    component_analysis = 'ICA'
    # Saving details - dir and name
    saving_details = []
    saving_details.append(save_plots)
    # # # # SAVING DETAILS AND APPEND MAY BE DIFFERENT FOR SUBJECTS
    saving_details.append(image_saving_dir_subjSpec + dir_date + '_' + dir_subj + '_' + \
        component_analysis + '_')
    saving_details.append(plotting_param)

    ICA_applied = "SI"

    # Initialise OE component number
    OE_component_number = None

    # Call generate_freq - to generate frequencies and timepoints for plotting.
    # Use plot_frequencies for plotting.
    freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(MRI_params[2], MRI_params[0])



    # Here call load_MRI_data_raw and load the registered first echo data.
    load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw(dir_images, MRI_params[1], echo_num, MRI_params[2], MRI_params[7], MRI_params[8])
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

    # Apply masks to the loaded data.
    mask_applied = 'cardiac'
    if mask_applied == 'cardiac':
        masks_reg_data = masks_reg_data_cardiac
    else: masks_reg_data = masks_reg_data_lung

    masked_data_echo1_regOnly = Sarah_ICA.apply_masks(masks_reg_data, load_data_echo1_regOnly)



    # # Perform ICA...
    # Scale the masked data to the maximum SI amplitude (within the mask).
    X_scale_tVox_regOnly, data_scaling = Sarah_ICA.preproc_ICA(masked_data_echo1_regOnly)

    # Generate array of components numbers for ICA to find, this will be looped over.
    num_components_list = np.int32(np.rint(np.linspace(num_c_start, num_c_stop, num_c_stop-num_c_start+1)))
    RunNum, RunNum_2 = functions_testing_pythonFastICA_errorsWarnings.loopICA_HDF5save_plot_order(\
        num_components_list, ordering_method_list, \
        X_scale_tVox_regOnly, size_echo_data, MRI_params[1], \
        masks_reg_data, halfTimepoints, freqPlot, plot_time_value, \
        MRI_params[2], MRI_params[0], MRI_params[3], data_scaling, \
        hdf5_loc, subject, ICA_applied, mask_applied, echo_num, \
        saving_details, showplot, plot_cmap, plot_all)

    return




def hdf5_save_ICA_metricsOnly(hdf5_loc, subject, num_components, mask_applied, ICA_applied, \
    echo_num, RunNumber=0, metrics_name=0, argsort_indices=0, sorting_metrics=0):
    """
    Function to save ICA component ordering metrics (only) to a HDF5 (.h5) file.

    Arguments:
    hdf5_loc = location of the HDF5 file (to be saved/opened and written to).
    subject = subject ID for saving HDF5 file.
    num_components = number of components used in ICA.
    mask_applied = 'cardiac' or 'lung' depending on the mask used.
    ICA_applied = what ICA was applied to: (param/data_type).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    RunNumber = the Run#. Default=0, meaning that the highest run number present
    in the HDF5 file will be used. It is the actual ***NUMBER***.
    metrics_name = name of the metric used to order the components. e.g. "RMS_freq". 
    If zero, do not save the metrics, hence set to zero as a default.
    argsort_indices = array of indices of the sorted array according to the sorting method.
    sorting_metrics = array of values of the metric used to sort the ICA components.
    Returns:
    RunNumReturn = the run number to be used for plotting, if required.

    """
    # HDF5 subject-specific file name.
    hdf5_file_name = "hdf5_structure_" + subject

    # The subject-specific HDF5 file must exist from ICA component writing performed
    # in hdf5_save_ICA, which would have been called before this function.
    newHDF5file = h5py.File(hdf5_loc + hdf5_file_name, 'r+')

    # Use ICA_applied to identify the data that ICA was performed on (param/data_type),
    # and if ICA was applied to SI data ('SI' input), include the echo number.
    if ICA_applied == 'SI':
        group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)
    else: group_name = str(ICA_applied)

    # The main  subgroups are likely to be present. But check and add if not.
    if not group_name in newHDF5file:
        subgroup_ICA = newHDF5file.create_group(group_name) # 2) ICA applied to SI
    else:
        # Set variable to the subgroup name for use later when referring to the subgroup.
        subgroup_ICA = newHDF5file[group_name]

    # Type of mask: 'cardiac' or 'lung'. 
    # (Expect cardiac masks to have been applied.)
    if mask_applied == 'cardiac':
        mask_naming = 'CardiacMasks'
    else: mask_naming = 'LungMasks'
    subgroup_ICA_Masks = newHDF5file[subgroup_ICA.name + "/" + mask_naming]

    # a)) Subgroup for NumC will already be present.

    # b)) Run details - if RunNum == 0 (default), assume information to be saved relates to the
    # **maximum run number present** - i.e. the most recent run which would have been saved using the
    # hdf5_save_ICA function earlier.
    list_ofRuns_inNumC = list(newHDF5file[newHDF5file[newHDF5file[subgroup_ICA_Masks.name].name + '/' + 'NumC_' + str(num_components)].name].keys())
    if RunNumber == 0:
        # b))ii/. if run details does not exists, find maximum previous run value.
        list_num_runs = []
        for j in list_ofRuns_inNumC:
            list_num_runs.append(np.int32(j[3:]))

        # Sort Run# numerical values and subsequently the 'Run#' array.
        list_num_runs = np.array(list_num_runs); list_ofRuns_inNumC = np.array(list_ofRuns_inNumC)
        list_num_runs_argsort_indices = np.argsort(list_num_runs, axis=0)
        list_num_runs_argsort_Sorted = np.take_along_axis(list_ofRuns_inNumC, list_num_runs_argsort_indices, axis=0)
        # 'Previous' maximum Run# is the final list element.
        RunNum_indicator = list_num_runs_argsort_Sorted[-1]
        # Numerical value - add one for new/current Run# when creating the subgroup.
        RunNum_indicator = np.int32(RunNum_indicator[3:])
        RunNum = 'Run' + str(RunNum_indicator)
        # # Add subgroup
    else:
        # b))i/. use RunNum input as the RunNumber
        RunNum = str(RunNumber)

    # 'Save' name of subgroup (numC and Runs) to be used when storing the new data.
    subgroup_new_name = newHDF5file[subgroup_ICA_Masks.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + RunNum

    # Add ordering information.
    # Add subgroup for ordering if not present already.
    ordering_subgroup = newHDF5file[subgroup_ICA_Masks.name].name + '/' + 'NumC_' + str(num_components) \
    + '/' + RunNum + '/' + 'ComponentSorting'
    if not ordering_subgroup in newHDF5file:
        newHDF5file.create_group(newHDF5file[subgroup_ICA_Masks.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + RunNum + '/' + 'ComponentSorting')

    # New subgroup for the ordering method to be saved:
    newHDF5file.create_group(newHDF5file[ordering_subgroup].name + '/' + metrics_name)
    # Add data:
    # Argsort indices
    newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('argsort_indices', data=argsort_indices)
    # Metric values (not ordered)
    newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('sorting_metrics', data=sorting_metrics)

    # Close file
    newHDF5file.close()
    # Return RunNum
    return RunNum






if __name__ == "__main__":
    # ...
    a = 1