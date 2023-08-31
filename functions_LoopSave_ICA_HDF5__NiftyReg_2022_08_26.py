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

# Functions to loop over different numbers of ICA components, apply ICA,
# and save the extracted ICA components to an HDF5 file.

import numpy as np
import sys
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
import functions_LoopSave_ICA_HDF5
import functions_loadHDF5_ordering
import functions_RunICA_andPlot_additional__NiftyReg_2022_08_26

def loop_ICA_save_metrics_details(dir_date, dir_subj, num_c_start, num_c_stop, ordering_method_list, \
    echo_num, dictionary_name, 
    hdf5_loc, images_saving_info_list, MRI_scan_directory, MRI_params, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    save_plots, showplot, plot_cmap, plot_all, \
    other_parameters):
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
    other_parameters = [NumEchoes, num_c_start, num_c_stop, ordering_method_list, num_c_use, av_im_num].
    Returns:
    None - details will be saved to HDF5 file and plots can be saved.
    #
    #
    #
    GasSwitch = [Cycled3, gas_switching], where Cycled3 = 1/0 for Yes/No for cycled oxygen
    use. If 0, assume SingleCycle. gas_switching may be a list of switching times if SingleCycle,
    of a single repeated number of dynamics at which the gases are switched if cyclic.
    av_im_num = av_im_num if Cycled3. Otherwise = [[air_mean, oxy_mean], assume_alpha], where
    assume_alpha = assumed upslope/downslope (s) times for creating synthetic input function.

    """
    # MRI images directory and mask directory.
    # dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_A = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/'

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


    # # CHECK FOR NON- BINARY MASKS:
    # # Check, either one or zero...
    ones_LungMask = np.sum(masks_reg_data_lung == 1)
    zeros_LungMask = np.sum(masks_reg_data_lung == 0)
    ones_CardiacMask = np.sum(masks_reg_data_cardiac == 1)
    zeros_CardiacMask = np.sum(masks_reg_data_cardiac == 0)
    total_voxels = size_echo_data[0]*size_echo_data[1]*size_echo_data[3]
    LungMask_tot = ones_LungMask + zeros_LungMask
    CardiacMask_tot = ones_CardiacMask + zeros_CardiacMask
    LungMask_TrueQ = LungMask_tot == total_voxels
    CardiacMask_TrueQ = CardiacMask_tot == total_voxels
    LungMask_TrueQ_val = 9; CardiacMask_TrueQ_val = 9
    if LungMask_TrueQ == True:
        LungMask_TrueQ_val = 1
    else:
        LungMask_TrueQ_val = 0
    #
    if CardiacMask_TrueQ == True:
        CardiacMask_TrueQ_val = 1
    else:
        CardiacMask_TrueQ_val = 0
    Masks_ALL_TrueQ_val = LungMask_TrueQ_val + CardiacMask_TrueQ_val
    if Masks_ALL_TrueQ_val != 2:
        print('Mask not binary: ' + subject)
        # sys.exit(1)
        return

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
    # Loop over the different numbers of components and perform ICA.
    # Optionally, can save to HDF5; optionally can plot figures and save.
    RunNum, RunNum_2 = loopICA_HDF5save_plot_order_nifty(\
        num_components_list, ordering_method_list, \
        X_scale_tVox_regOnly, size_echo_data, MRI_params[1], \
        masks_reg_data, halfTimepoints, freqPlot, plot_time_value, \
        MRI_params[2], MRI_params[0], MRI_params[3], data_scaling, \
        hdf5_loc, subject, ICA_applied, mask_applied, echo_num, \
        NiftyReg_ID, densityCorr, regCorr_name, \
        ANTs_used, dir_image_nifty, hdf5_file_name_base, \
        saving_details, showplot, plot_cmap, plot_all, \
        other_parameters, MRI_params)

    return




def loopICA_HDF5save_plot_order_nifty(num_components_list, ordering_method_list, X_scale_tVox, size_echo_data, NumSlices, \
    masks_reg_data, halfTimepoints, freqPlot, plot_time_value, \
    NumDyn, TempRes, GasSwitch, data_scaling, \
    hdf5_loc, subject, ICA_applied, mask_applied, echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    saving_details, showplot, plot_cmap, plot_all, \
    other_parameters, MRI_params):
    """
    Function to perform ICA for different num_components (supplied as a list),
    save the components in an HDF5 file and optionally plot the components 
    if convergence is reached.

    Arguments:
    num_components_list = list of different components for ICA to find.
    ordering_method_list = list of different ordering methods (as strings)
    to be investigated.
    X_scale_tVox = ICA input ('pre-processed' loaded MRI data that
    has been masked, scaled and collapsed).
    size_echo_data = np.shape(echo_data_load).
    NumSlices = the number of slices acquired.
    masks_reg_data = the array containing the masks of all of the slices,
    which has shape (vox,vox,NumSlices).
    halfTimepoints = half of the time points value to be used as an index when plotting
    frequencies (freqPlot[0:halfTimepoints]) to plot only one side of the frequency spectrum
    (i.e. the positive frequencies). freqPlot is already adjusted for this, halfTimepoints is
    needed to index the final value of the dependent variable that is being plotted
    on the y-axis.
    freqPlot = frequency values for plotting along the x-axis (only for the positive
    frequencies).
    plot_time_value = (timepoints) time points for plotting along the x-axis.
    NumDyn = the number of dynamic images acquired.
    TempRes = the temporal imaging resolution /s.
    GasSwitch = dynamic number the gases are cycled at.
    data_scaling = scaling applied to the loaded MRI data before ICA, performed and
    calculated during data preprocessing.
    hdf5_loc = location of the HDF5 file to be read/written which will/does contain 
    details of the subject, ICA components (param/data_type) and parameter fits.
    subject = subject ID.
    ICA_applied = what ICA was applied to: (param/data_type).
    mask_applied = which mask was used, 'cardiac or 'lung'.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    saving_details = list of three elements. saving_details[0] = 0 for No or 1 for Yes - 
    whether to save the figures. saving_details[1] is the directory of where the 
    figures are to be saved to with the start of the figure saving name (e.g.
    'ICAon' and the subject ID). saving_details[2] is information about the
    loaded data (e.g. 'regOnly').
    showplot = 0 for No or 1 for Yes - whether to show the plots. If No, the
    plots will be closed.
    plot_cmap = whether to plot the component map. Default = 0 (No).
    plot_all = whether to plot the linear plots. Default = 0 (No).
    other_parameters = [NumEchoes, num_c_start, num_c_stop, ordering_method_list, num_c_use, av_im_num].
    Returns:
    RunNum, RunNum_2 = RunNumber of the ICA analysis for the particular component.
    May be used for plotting saving names and saving components/details/metrics
    to the HDF5 file.
    Also used to check later.
    #
    GasSwitch = [Cycled3, gas_switching], where Cycled3 = 1/0 for Yes/No for cycled oxygen
    use. If 0, assume SingleCycle. gas_switching may be a list of switching times if SingleCycle,
    of a single repeated number of dynamics at which the gases are switched if cyclic.
    av_im_num = av_im_num if Cycled3. Otherwise = [[air_mean, oxy_mean], assume_alpha], where
    assume_alpha = assumed upslope/downslope (s) times for creating synthetic input function.
    #
    """
    # Set RunNum and RunNum_2 = 0 so that if not saving, values will be returned and
    # no errors will be returned.
    RunNum = 0; RunNum_2 = 0

    # Loop over all components in the components list supplied. For reach loop,
    # apply ICA to the input data, and optionally plot and save the results.
    for k in num_components_list:
        # Perform ICA.
        ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean, \
            iteration_loop, converged_test = Sarah_ICA.apply_ICA(X_scale_tVox, k)
        
        # Save original components and details of whether the ICA algorithm converged.
        # Save regardless of whether ICA converges - the appropriate details will
        # be saved by the hdf5_save_ICA function, and it is useful/important to know
        # if ICA did not converge vs whether an error occurred and the code did not
        # run properly.
        RunNum = hdf5_save_ICA_nifty(hdf5_loc, subject, k, mask_applied, ICA_applied, \
            converged_test, iteration_loop, echo_num, S_ica_tVox, A_ica_tVox, ica_tVox_mean, data_scaling, \
            NiftyReg_ID, densityCorr, regCorr_name, \
            ANTs_used, dir_image_nifty, hdf5_file_name_base, \
            metrics_name=0, argsort_indices=0, sorting_metrics=0)

        # Plot S_ica timeseries and frequency spectra, and A_ica component maps 
        # **ONLY** if ICA converges.
        if converged_test == 1:
            #  Calculate the frequency spectra of the ICA components (spectra of S_ica).
            freq_S_ica_tVox = Sarah_ICA.freq_spec(k, S_ica_tVox)

            # Plot component time series, frequency spectra and component maps in 
            # their original ICA output ordering *only if* the ICA algorithm has converged.
            # Plot component maps once for each num_component and for all slices;
            # plot time series and frequency spectra according to the original
            # ICA output and for each ordering metric.


        # Calculate component ordering and metric values.
        # Save the ordering information to the HDF5 file and plot the ordered
        # components, if desired.
        # ALL ONLY IF THE ICA ALGORITHM CONVERGED:
        if converged_test == 1:
            # Calculate component ordering/metric values.
            # Loop over the different ordering methods.
            for j in ordering_method_list:
                S_ica_tVox_Sorted, freq_S_ica_tVox_Sorted, RMS_allFreq_P_relMax_argsort_indices, \
                    RMS_allFreq_P_relMax, RMS_allFreq_P_relMax_Sorted = \
                    functions_Calculate_stats.calc_plot_component_ordering(j, S_ica_tVox, A_ica_tVox, \
                        freq_S_ica_tVox, ica_tVox, size_echo_data, NumSlices, NumDyn, halfTimepoints, \
                        freqPlot, masks_reg_data, k, \
                        data_scaling, \
                        other_parameters, MRI_params)
                # Save components and ordering information.
                RunNum_2 = hdf5_save_ICA_metricsOnly_nifty(hdf5_loc, subject, k, mask_applied, ICA_applied, \
                    echo_num, \
                    NiftyReg_ID, densityCorr, regCorr_name, \
                    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
                    RunNumber=RunNum, metrics_name=j, argsort_indices=np.array(RMS_allFreq_P_relMax_argsort_indices), \
                    sorting_metrics=np.array(RMS_allFreq_P_relMax))


        print(str(k)) # Print num_component in loop.

    # Nothing to return - saved and plotted components and ordering if required.
    return RunNum, RunNum_2


def hdf5_save_ICA_nifty(hdf5_loc, subject, num_components, mask_applied, ICA_applied, \
    converged_test, iteration_loop, echo_num, \
    S_ica_tVox, A_ica_tVox, ica_tVox_mean, ica_scaling, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    metrics_name=0, argsort_indices=0, sorting_metrics=0):
    """
    Function to save ICA components to a HDF5 (.h5) file.

    Arguments:
    hdf5_loc = location of the HDF5 file (to be saved/opened and written to).
    subject = subject ID for saving HDF5 file.
    num_components = number of components used in ICA.
    mask_applied = 'cardiac' or 'lung' depending on the mask used.
    ICA_applied = what ICA was applied to: (param/data_type).
    converged_test = 0 if No; 1 if Yes - whether ICA converged. If ICA successfully converges,
    the components etc will be saved.
    iteration_loop = (tolerance, max_iterations) ICA settings, form of a tuple.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean = ICA outputs to be saved.
    ica_scaling = scaling factor applied to the data before ICA was applied.
    metrics_name = name of the metric used to order the components. e.g. "RMS_freq". 
    If zero, do not save the metrics, hence set to zero as a default.
    argsort_indices = array of indices of the sorted array according to the sorting method.
    sorting_metrics = array of values of the metric used to sort the ICA components.
    Returns:
    RunNumReturn = the run number to be used for plotting, if required.

    """
    # HDF5 subject-specific file name.
    hdf5_file_name = hdf5_file_name_base + subject

    # Check if subject-specific HDF5 is present. If it is not present, create a 
    # writable file. Otherwise, open the existing HDF5 file.
    # Use file_exist = 0 for No; 1 = Yes to identify for future data additions.
    file_exist = 0

    if os.path.isfile(hdf5_loc + hdf5_file_name):
        # File exists - read in.
        newHDF5file = h5py.File(hdf5_loc + hdf5_file_name, 'r+')
        file_exist = 1
    else:
        # File does not exist - create.
        newHDF5file = h5py.File(hdf5_loc + hdf5_file_name, "w")
        file_exist = 0

    # Set subject ID as an attribute.
    if file_exist == 0:
        # If exists, assume subject attribute is set. Otherwise, set/add ...
        # 1) Subject details
        # Add Subject ID as an attribute.
        newHDF5file.attrs['Subject'] = subject

    # Create the main group structures first and before filling.
    # Create subgroup structure: firstly for all main subgroups.
    # Use ICA_applied to identify the data that ICA was performed on (param/data_type),
    # and if ICA was applied to SI data ('SI' input), include the echo number.
    if ICA_applied == 'SI':
        group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)
    else: group_name = str(ICA_applied)

    # Add main subgroups one at a time as/when required, but before adding, need to check
    # if the required subgroup is already present.
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

    # The mask type is used as a further subgroup. Again, need to check if it is already
    # present as a subgroup.
    if not subgroup_ICA.name + "/" + mask_naming in newHDF5file:
        subgroup_ICA_Masks = newHDF5file.create_group(subgroup_ICA.name + "/" + mask_naming) # 2) ICA applied to SI
    else:
        # Still set variable to this.
        subgroup_ICA_Masks = newHDF5file[subgroup_ICA.name + "/" + mask_naming]


    # $$££$$££ NEW SUBGROUP - registered, is density correction applied?
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    # $$££$$££ NEW SUBGROUP - registration type - ANTs or NiftyReg_ and settings.
    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # Add subgroup.
    if not subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type in newHDF5file:
        subgroup_ICA_reg = newHDF5file.create_group(subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type) # 2) ICA applied to SI
    else:
        # Still set variable to this.
        subgroup_ICA_reg = newHDF5file[subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type]

    # Write ICA input mean and scaling factor used to the HDF5 file. These will
    # both be required when reconstructing ICA components. These values are for
    # a subject within a specific mask (and can be used for ICA reconstruction
    # with any number of components).
    # Similar to previous steps, only add if not already present.
    if not subgroup_ICA_reg.name + "/" + 'ica_scaling' in newHDF5file:
        # print(ica_tVox_mean)
        # newHDF5file[subgroup_ICA_reg.name].create_dataset('ica_tVox_mean', data=ica_tVox_mean)
        newHDF5file[subgroup_ICA_reg.name].create_dataset('ica_scaling', data=np.array([ica_scaling]))

    # a)) Check if subgroup for NumC (number of components ICA was run with) is present, if not, create.
    if not newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) in newHDF5file:
        # Create NumC_# subgroup
        newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components))

    # To save multiple runs of ICA, save each run as Run# with the # increasing from 1.
    # b)) RunDetails
    # b))i/. if no run details exist at all (i.e. first run of ICA), create subgroup for Run# = 1.
    # b))ii/. if run details does exist, create Run#+1 by using # = the maximum (previous) Run value.
    list_ofRuns_inNumC = list(newHDF5file[newHDF5file[newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components)].name].keys())
    if len(list_ofRuns_inNumC) == 0:
        # No Run# subgroups saved therefore start with Run1. Create subgroup for Run1.
        newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + 'Run1')
        # In this case, prev_maxRuns = 0
        prev_maxRuns = 0
    else:
        # b))ii/.
        # Run# present, therefore find the largest Run# present and + 1 to create the
        # new Run# value.
        # Run# numerical values.
        list_num_runs = []
        for j in list_ofRuns_inNumC:
            list_num_runs.append(np.int32(j[3:]))
        #
        # Sort Run# numerical values and subsequently the 'Run#' array.
        list_num_runs = np.array(list_num_runs); list_ofRuns_inNumC = np.array(list_ofRuns_inNumC)
        list_num_runs_argsort_indices = np.argsort(list_num_runs, axis=0)
        list_num_runs_argsort_Sorted = np.take_along_axis(list_ofRuns_inNumC, list_num_runs_argsort_indices, axis=0)
        # 'Previous' maximum Run# is the final list element (of the list ordered by increasing run number).
        prev_maxRuns = list_num_runs_argsort_Sorted[-1]
        # Numerical value - add one for new/current Run# when creating the subgroup.
        prev_maxRuns = np.int32(prev_maxRuns[3:])
        # Add subgroup
        newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + 'Run' + str(prev_maxRuns + 1))

    # 'Save' name of newsubgroup (numC and Runs) to be used when storing the new data.
    subgroup_new_name = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + 'Run' + str(prev_maxRuns + 1)

    # c)) Add data as subgroups within NumC and Run#. Add the data by creating datasets (to simultaneously create/add
    # subgroups appropriately for each dataset to be added).
    # First, add converged_test - to identify whether the ICA algorithm converged for this NumC and Run#.
    conv_add = newHDF5file[subgroup_new_name].create_dataset('converged_test', data=np.array(([converged_test])))

    # If converged_test is not equal to zero, the ICA algorithm converged and the rest of the data
    # should be added. In this case, ica returns were None and are not added to the HDF5 file.
    if converged_test != 0:
        # ICA has converged - add ICA outputs etc.
        newHDF5file[subgroup_new_name].create_dataset('iteration_loop', data=iteration_loop)
        # Add ICA outputs
        # if not subgroup_ICA_reg.name + 'ica_tVox_mean' in newHDF5file:
        if not 'ica_tVox_mean' in newHDF5file[subgroup_ICA_reg.name]:
            newHDF5file[subgroup_ICA_reg.name].create_dataset('ica_tVox_mean', data=ica_tVox_mean)

        newHDF5file[subgroup_new_name].create_dataset('S_ica_tVox', data=S_ica_tVox)
        newHDF5file[subgroup_new_name].create_dataset('A_ica_tVox', data=A_ica_tVox)
        # Add ordering information if ICA ordering has been performed. Do this if
        # metrics_name is NOT equal to zero.
        if metrics_name != 0:
            # Add subgroup for ordering
            newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
                + '/' + 'Run' + str(prev_maxRuns + 1) + '/' + 'ComponentSorting')
            # 'Save' name of newsubgroup (numC and Runs) to be used when storing the new data.
            Csort_new_name = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
                + '/' + 'Run' + str(prev_maxRuns + 1)  + '/' + 'ComponentSorting'
            # New subgroup for the ordering method to be saved:
            newHDF5file.create_group(newHDF5file[Csort_new_name].name + '/' + metrics_name)
            # Add data:
            # Argsort indices
            newHDF5file[newHDF5file[Csort_new_name].name + '/' + metrics_name].create_dataset('argsort_indices', data=argsort_indices)
            # Metric values ***(not ordered)***
            newHDF5file[newHDF5file[Csort_new_name].name + '/' + metrics_name].create_dataset('sorting_metrics', data=sorting_metrics)

    # Close file
    newHDF5file.close()
    # Return RunNum
    RunNumReturn = 'Run' + str(prev_maxRuns + 1)
    return RunNumReturn


def hdf5_save_ICA_metricsOnly_nifty(hdf5_loc, subject, num_components, mask_applied, ICA_applied, \
    echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNumber=0, metrics_name=0, argsort_indices=0, sorting_metrics=0):
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
    hdf5_file_name = hdf5_file_name_base + subject

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


    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID


    # Add subgroup.
    if not subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type in newHDF5file:
        subgroup_ICA_reg = newHDF5file.create_group(subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type) # 2) ICA applied to SI
    else:
        # Still set variable to this.
        subgroup_ICA_reg = newHDF5file[subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type]



    # a)) Subgroup for NumC will already be present.

    # b)) Run details - if RunNum == 0 (default), assume information to be saved relates to the
    # **maximum run number present** - i.e. the most recent run which would have been saved using the
    # hdf5_save_ICA function earlier.
    list_ofRuns_inNumC = list(newHDF5file[newHDF5file[newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components)].name].keys())
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
    subgroup_new_name = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + RunNum

    # Add ordering information.
    # Add subgroup for ordering if not present already.
    ordering_subgroup = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
    + '/' + RunNum + '/' + 'ComponentSorting'
    if not ordering_subgroup in newHDF5file:
        newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
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



def loopICA_HDF5save_plot_order_nifty_ONLY_ORDERING(\
    num_c_start, num_c_stop, ordering_method_list, \
    NumDyn, TempRes, GasSwitch, \
    hdf5_loc, dir_date, dir_subj, ICA_applied, mask_applied, echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, hdf5_file_name_base, \
    RunNum_use, \
    other_parameters, MRI_params):
    """
    Function to *ONLY* calculate ordering after ICA has been saved for different num_components (supplied as a list),
    and to save the ordering method's details in an HDF5 file.

    # Overwrites if function is re-run for a subject - i.e. at Run1.
    # # Unless give argument...

    Arguments:
    num_c_start = lower number of components to be found with ICA.
    num_c_stop = upper number of components to be found with ICA.
    ordering_method_list = list of different ordering methods (as strings)
    to be investigated.
    NumDyn = the number of dynamic images acquired.
    TempRes = the temporal imaging resolution /s.
    GasSwitch = dynamic number the gases are cycled at.
    hdf5_loc = location of the HDF5 file to be read/written which will/does contain 
    details of the subject, ICA components (param/data_type) and parameter fits.
    dir_date = subject scanning date.
    dir_subj = subject ID.
    ICA_applied = what ICA was applied to: (param/data_type).
    mask_applied = which mask was used, 'cardiac or 'lung'.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    av_im_num = average number of images to average over at the end of each oxygen cycle
    when calculating mean oxygen values.
    Returns:
    RunNum, RunNum_2 = RunNumber of the ICA analysis for the particular component.
    May be used for plotting saving names and saving components/details/metrics
    to the HDF5 file.
    Also used to check later.
    
    """
    # # Set RunNum and RunNum_2 = 0 so that if not saving, values will be returned and
    # # no errors will be returned.
    # RunNum = 0; RunNum_2 = 0
    RunNum = 'Run' + str(RunNum_use)

    # Generate array of components numbers for ICA to find, this will be looped over.
    num_components_list = np.int32(np.rint(np.linspace(num_c_start, num_c_stop, num_c_stop-num_c_start+1)))

    # Call generate_freq - to generate frequencies and timepoints for plotting.
    # Use plot_frequencies for plotting.
    freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(NumDyn, TempRes)

    subject = dir_date + '_' + dir_subj

    hdf5_file_name = hdf5_file_name_base + subject

    # Use ICA_applied to identify the data that ICA was performed on (param/data_type),
    # and if ICA was applied to SI data ('SI' input), include the echo number.
    if ICA_applied == 'SI':
        group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)
    else: group_name = str(ICA_applied)

    # Type of mask: 'cardiac' or 'lung'. 
    # (Expect cardiac masks to have been applied.)
    if mask_applied == 'cardiac':
        mask_naming = 'CardiacMasks'
    else: mask_naming = 'LungMasks'

    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # Subgroup name for which the keys are NumC_#
    subgroup_name = group_name + '/' + mask_naming + '/' + reg_applied + '/' + reg_type
    # newHDF5file[subgroup_name]

    # Loop over all components in the components list supplied. For reach loop,
    # apply ICA to the input data, and optionally plot and save the results.
    for k in num_components_list:
        # print((k, echo_num))
        newHDF5file = h5py.File(hdf5_loc + hdf5_file_name, 'r')
        subgroup_name_component = subgroup_name + '/NumC_' + str(k)
        # Set converged test value
        subgroup_name_component_Run = subgroup_name_component + '/' + RunNum
        converged_test = np.squeeze(newHDF5file[subgroup_name_component_Run + '/converged_test'][:])

        # Plot S_ica timeseries and frequency spectra, and A_ica component maps 
        # **ONLY** if ICA converges.
        if converged_test == 1:
            S_ica_tVox = newHDF5file[subgroup_name_component_Run + '/S_ica_tVox'][:]
            A_ica_tVox = newHDF5file[subgroup_name_component_Run + '/A_ica_tVox'][:]
            #  Calculate the frequency spectra of the ICA components (spectra of S_ica).
            freq_S_ica_tVox = Sarah_ICA.freq_spec(k, S_ica_tVox)

        newHDF5file.close()

        # Calculate component ordering and metric values.
        # Save the ordering information to the HDF5 file and plot the ordered
        # components, if desired.
        # ALL ONLY IF THE ICA ALGORITHM CONVERGED:
        if converged_test == 1:
            # Calculate component ordering/metric values.
            # Loop over the different ordering methods.
            for j in ordering_method_list:
                S_ica_tVox_Sorted, freq_S_ica_tVox_Sorted, RMS_allFreq_P_relMax_argsort_indices, \
                    RMS_allFreq_P_relMax, RMS_allFreq_P_relMax_Sorted = \
                    functions_Calculate_stats.calc_plot_component_ordering(j, S_ica_tVox, A_ica_tVox, \
                        freq_S_ica_tVox, 0, 0, 0, NumDyn, halfTimepoints, \
                        freqPlot, 0, k, 0, \
                        other_parameters, MRI_params)

                # Save components and ordering information.
                # **OVERWRITING**
                RunNum_2 = hdf5_save_ICA_metricsOnly_nifty__overwrite(hdf5_loc, subject, k, mask_applied, ICA_applied, \
                    echo_num, \
                    NiftyReg_ID, densityCorr, regCorr_name, \
                    ANTs_used, 0, hdf5_file_name_base, \
                    RunNum, metrics_name=j, argsort_indices=np.array(RMS_allFreq_P_relMax_argsort_indices), \
                    sorting_metrics=np.array(RMS_allFreq_P_relMax))

    # Nothing to return - saved and plotted components and ordering if required.
    return


def loopICA_HDF5save_plot_order_nifty_ONLY_ORDERING_specRun(num_c_start, num_c_stop, ordering_method_list, \
    NumDyn, TempRes, GasSwitch, \
    hdf5_loc, dir_date, dir_subj, ICA_applied, mask_applied, echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, hdf5_file_name_base, \
    RunNum, \
    other_parameters, MRI_params):
    """
    Function to *ONLY* calculate ordering after ICA has been saved for different num_components (supplied as a list),
    and to save the ordering method's details in an HDF5 file.

    Overwrites if function is re-run for a subject - **BUT FOR SPECIFIC RunNum**

    Arguments:
    num_c_start = lower number of components to be found with ICA.
    num_c_stop = upper number of components to be found with ICA.
    ordering_method_list = list of different ordering methods (as strings)
    to be investigated.
    NumDyn = the number of dynamic images acquired.
    TempRes = the temporal imaging resolution /s.
    GasSwitch = dynamic number the gases are cycled at.
    hdf5_loc = location of the HDF5 file to be read/written which will/does contain 
    details of the subject, ICA components (param/data_type) and parameter fits.
    dir_date = subject scanning date.
    dir_subj = subject ID.
    ICA_applied = what ICA was applied to: (param/data_type).
    mask_applied = which mask was used, 'cardiac or 'lung'.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    av_im_num = average number of images to average over at the end of each oxygen cycle
    when calculating mean oxygen values.
    RunNum = specific RunNumber to calculate the ordering of.
    Returns:
    None.
    
    """
    # Generate array of components numbers for ICA to find, this will be looped over.
    num_components_list = np.int32(np.rint(np.linspace(num_c_start, num_c_stop, num_c_stop-num_c_start+1)))

    # Call generate_freq - to generate frequencies and timepoints for plotting.
    # Use plot_frequencies for plotting.
    freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(NumDyn, TempRes)

    subject = dir_date + '_' + dir_subj

    # Load subject-specific HDF5 file
    # Group ... ANTs or NiftyReg, and was density correction applied etc...
    # HDF5 subject-specific file name.
    hdf5_file_name = hdf5_file_name_base + subject

    # Use ICA_applied to identify the data that ICA was performed on (param/data_type),
    # and if ICA was applied to SI data ('SI' input), include the echo number.
    if ICA_applied == 'SI':
        group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)
    else: group_name = str(ICA_applied)

    # Type of mask: 'cardiac' or 'lung'. 
    # (Expect cardiac masks to have been applied.)
    if mask_applied == 'cardiac':
        mask_naming = 'CardiacMasks'
    else: mask_naming = 'LungMasks'

    # NEW SUBGROUP - registered, is density correction applied?
    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    # NEW SUBGROUP - registration type - ANTs or NiftyReg_ and settings.
    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # Subgroup name for which the keys are NumC_#
    subgroup_name = group_name + '/' + mask_naming + '/' + reg_applied + '/' + reg_type
    # newHDF5file[subgroup_name]

    # Loop over all components in the components list supplied. For reach loop,
    # apply ICA to the input data, and optionally plot and save the results.
    for k in num_components_list:
        # print((k, echo_num))
        newHDF5file = h5py.File(hdf5_loc + hdf5_file_name, 'r')

        subgroup_name_component = subgroup_name + '/NumC_' + str(k)

        # Set converged test value
        subgroup_name_component_Run = subgroup_name_component + '/' + 'Run' + str(RunNum)
        converged_test = np.squeeze(newHDF5file[subgroup_name_component_Run + '/converged_test'][:])

        # Plot S_ica timeseries and frequency spectra, and A_ica component maps 
        # **ONLY** if ICA converges.
        if converged_test == 1:
            S_ica_tVox = newHDF5file[subgroup_name_component_Run + '/S_ica_tVox'][:]
            A_ica_tVox = newHDF5file[subgroup_name_component_Run + '/A_ica_tVox'][:]
            #  Calculate the frequency spectra of the ICA components (spectra of S_ica).
            freq_S_ica_tVox = Sarah_ICA.freq_spec(k, S_ica_tVox)

        newHDF5file.close()

        # Calculate component ordering and metric values.
        # Save the ordering information to the HDF5 file and plot the ordered
        # components, if desired.
        # ALL ONLY IF THE ICA ALGORITHM CONVERGED:
        if converged_test == 1:
            # Calculate component ordering/metric values.
            # Loop over the different ordering methods.
            for j in ordering_method_list:
                S_ica_tVox_Sorted, freq_S_ica_tVox_Sorted, RMS_allFreq_P_relMax_argsort_indices, \
                    RMS_allFreq_P_relMax, RMS_allFreq_P_relMax_Sorted = \
                    functions_Calculate_stats.calc_plot_component_ordering(j, S_ica_tVox, A_ica_tVox, \
                        freq_S_ica_tVox, 0, 0, 0, NumDyn, halfTimepoints, \
                        freqPlot, GasSwitch, TempRes, 0, k, \
                        0, \
                        other_parameters, MRI_params)
                        
                # Save components and ordering information.
                # **OVERWRITING**
                hdf5_save_ICA_metricsOnly_nifty__overwrite_specRunNum(hdf5_loc, subject, k, mask_applied, ICA_applied, \
                    echo_num, \
                    NiftyReg_ID, densityCorr, regCorr_name, \
                    ANTs_used, 0, hdf5_file_name_base, \
                    RunNum, metrics_name=j, argsort_indices=np.array(RMS_allFreq_P_relMax_argsort_indices), \
                    sorting_metrics=np.array(RMS_allFreq_P_relMax))

    # Nothing to return - saved and plotted components and ordering if required.
    return



def hdf5_save_ICA_metricsOnly_nifty__overwrite(hdf5_loc, subject, num_components, mask_applied, ICA_applied, \
    echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNumber, metrics_name=0, argsort_indices=0, sorting_metrics=0):
    """
    Function to save ICA component ordering metrics (only) to a HDF5 (.h5) file.

    **Overwriting if re-run**
    --> set RunNum to zero.

    Arguments:
    hdf5_loc = location of the HDF5 file (to be saved/opened and written to).
    subject = subject ID for saving HDF5 file.
    num_components = number of components used in ICA.
    mask_applied = 'cardiac' or 'lung' depending on the mask used.
    ICA_applied = what ICA was applied to: (param/data_type).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    RunNumber = the 'Run#'. As a string.
    metrics_name = name of the metric used to order the components. e.g. "RMS_freq". 
    If zero, do not save the metrics, hence set to zero as a default.
    argsort_indices = array of indices of the sorted array according to the sorting method.
    sorting_metrics = array of values of the metric used to sort the ICA components.
    Returns:
    RunNumReturn = the run number to be used for plotting, if required.

    """
    # HDF5 subject-specific file name.
    hdf5_file_name = hdf5_file_name_base + subject

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


    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # Add subgroup.
    if not subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type in newHDF5file:
        subgroup_ICA_reg = newHDF5file.create_group(subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type) # 2) ICA applied to SI
    else:
        # Still set variable to this.
        subgroup_ICA_reg = newHDF5file[subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type]


    # 'Save' name of subgroup (numC and Runs) to be used when storing the new data.
    subgroup_new_name = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + RunNumber


    ## DELETE AND REWRITE IF ALREADY PRESENT, if not, just save normally.

    # Add ordering information.
    # Add subgroup for ordering if not present already.
    ordering_subgroup = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
    + '/' + RunNumber + '/' + 'ComponentSorting'
    if not ordering_subgroup in newHDF5file:
        newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + RunNumber + '/' + 'ComponentSorting')

    metric_ordering_subgroup = newHDF5file[ordering_subgroup].name + '/' + metrics_name
    if not metric_ordering_subgroup in newHDF5file:
        # Ordering method **NOT** present --> save using normal method...
        # New subgroup for the ordering method to be saved:
        newHDF5file.create_group(newHDF5file[ordering_subgroup].name + '/' + metrics_name)
        # Add data:
        # Argsort indices
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('argsort_indices', data=argsort_indices)
        # Metric values (not ordered)
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('sorting_metrics', data=sorting_metrics)
    else:
        # Ordering method is present --> **OVER-WRITE**
        # Need to delete data first and then re-save data.
        del newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name]['argsort_indices']
        del newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name]['sorting_metrics']
        # # New subgroup for the ordering method to be saved:
        # newHDF5file.create_group(newHDF5file[ordering_subgroup].name + '/' + metrics_name)
        # Add data:
        # Argsort indices
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('argsort_indices', data=argsort_indices)
        # Metric values (not ordered)
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('sorting_metrics', data=sorting_metrics)

    # Close file
    newHDF5file.close()
    # Return RunNum
    return RunNumber



def hdf5_save_ICA_metricsOnly_nifty__overwrite_specRunNum(hdf5_loc, subject, num_components, mask_applied, ICA_applied, \
    echo_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum, metrics_name=0, argsort_indices=0, sorting_metrics=0):
    """
    Function to save ICA component ordering metrics (only) to a HDF5 (.h5) file.
    
    **Overwriting if re-run** - - FOR A **SPECIFIC RunNum**

    Arguments:
    hdf5_loc = location of the HDF5 file (to be saved/opened and written to).
    subject = subject ID for saving HDF5 file.
    num_components = number of components used in ICA.
    mask_applied = 'cardiac' or 'lung' depending on the mask used.
    ICA_applied = what ICA was applied to: (param/data_type).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    RunNum = the Run#. It is the actual ***NUMBER***.
    metrics_name = name of the metric used to order the components. e.g. "RMS_freq". 
    If zero, do not save the metrics, hence set to zero as a default.
    argsort_indices = array of indices of the sorted array according to the sorting method.
    sorting_metrics = array of values of the metric used to sort the ICA components.
    Returns:
    RunNumReturn = the run number to be used for plotting, if required.

    """
    # HDF5 subject-specific file name.
    hdf5_file_name = hdf5_file_name_base + subject

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

    if densityCorr == 0:
        reg_applied = 'Reg'
    else:
        reg_applied = 'RegCorr' + regCorr_name

    if ANTs_used == 1:
        reg_type = 'ANTs'
    else:
        reg_type = 'NiftyReg_' + NiftyReg_ID

    # Add subgroup.
    if not subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type in newHDF5file:
        subgroup_ICA_reg = newHDF5file.create_group(subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type) # 2) ICA applied to SI
    else:
        # Still set variable to this.
        subgroup_ICA_reg = newHDF5file[subgroup_ICA.name + "/" + mask_naming + "/" + reg_applied + "/" + reg_type]


    # RunNum = 'Run1'

    # 'Save' name of subgroup (numC and Runs) to be used when storing the new data.
    subgroup_new_name = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
        + '/' + 'Run' + str(RunNum)


    ## DELETE AND REWRITE IF ALREADY PRESENT, if not, just save normally.

    # Add ordering information.
    # Add subgroup for ordering if not present already.
    ordering_subgroup = newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
        + '/' + 'Run' + str(RunNum) + '/' + 'ComponentSorting'
    if not ordering_subgroup in newHDF5file:
        newHDF5file.create_group(newHDF5file[subgroup_ICA_reg.name].name + '/' + 'NumC_' + str(num_components) \
            + '/' + 'Run' + str(RunNum) + '/' + 'ComponentSorting')

    metric_ordering_subgroup = newHDF5file[ordering_subgroup].name + '/' + metrics_name
    if not metric_ordering_subgroup in newHDF5file:
        # Ordering method **NOT** present --> save using normal method...
        # New subgroup for the ordering method to be saved:
        newHDF5file.create_group(newHDF5file[ordering_subgroup].name + '/' + metrics_name)
        # Add data:
        # Argsort indices
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('argsort_indices', data=argsort_indices)
        # Metric values (not ordered)
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('sorting_metrics', data=sorting_metrics)
    else:
        # Ordering method is present --> **OVER-WRITE**
        # Need to delete data first and then re-save data.
        del newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name]['argsort_indices']
        del newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name]['sorting_metrics']
        # # New subgroup for the ordering method to be saved:
        # newHDF5file.create_group(newHDF5file[ordering_subgroup].name + '/' + metrics_name)
        # Add data:
        # Argsort indices
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('argsort_indices', data=argsort_indices)
        # Metric values (not ordered)
        newHDF5file[newHDF5file[ordering_subgroup].name + '/' + metrics_name].create_dataset('sorting_metrics', data=sorting_metrics)

    # Close file
    newHDF5file.close()
    
    return


if __name__ == "__main__":
    # ...
    a = 1