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

# Functions to load HDF5 and calculate the ordering of components.

import numpy as np
import Sarah_ICA
import h5py
import matplotlib.pyplot as plt
import functions_Calculate_stats

def extract_metrics_nifty(component_sort_metric_list, num_lowest_metrics, \
    echo_num, hdf5_loc, dir_date, dir_subj, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, \
    RunNum_use, num_c_start, num_c_stop):
    """
    Extract metrics for a particular subject from the HDF5 structure. All
    metrics (as provided in the component_sort_metric_list) will be extracted.
    The number of components and metric values are stored and returned.

    Arguments:
    component_sort_metric_list = list of the metric names to be extracted.
    num_lowest_metrics = number of lowest metric values to be stored.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    hdf5_loc = location of the HDF5 file which contains the metric information.
    dir_date = scan date for the particular subject.
    dir_subj = subject ID for the particular subject.
    RunNum_use = specific RunNum to use, e.g. 1.
    num_c_start, num_c_stop = starting NumC and ending NumC - for calculations
    over this component range.
    Returns:
    components_metrics_etc_new = which contains the ICA component numbers (for
    cases in which ICA reached convergence) in the first column, and the 
    'num_lowest_metrics' lowest frequency RMS metrics in the next 
    'num_lowest_metrics' columns. The third dimension runs over the metrics of
    interest (of which there are len(component_sort_metric_list)). Hence, 
    shape of (#converged_ICA, 1 + num_lowest_metrics, len(component_sort_metric_list)).
    
    """
    # Generate array of components numbers for ICA to find, this will be looped over.
    num_components_list = np.int32(np.rint(np.linspace(num_c_start, num_c_stop, num_c_stop-num_c_start+1)))

    # Load the HDF5 file and read in (read only).
    f = h5py.File(hdf5_loc + hdf5_file_name_base + dir_date + '_' + dir_subj, 'r')
    # Store details of the ICA run, the number of components and the 'num_lowest_metrics' 
    # lowest frequency RMS  values for each and all metrics of interest.
    # Store these details in components_metrics_etc, which contains ICA component numbers
    # in the first column and the num_lowest_metrics lowest frequency RMS metrics in the 
    # next num_lowest_metrics columns. The third dimension runs over the different 
    # metrics being investigated, of which there are len(component_sort_metric_list).
    # All values are stored, with zeros left if convergence is not reached. The cases
    # for which ICA did not converge will be removed later before returning the array.
    # Given shape of len(list(f['group_ICA_SI/CardiacMasks'].keys())) - 2, although have
    # added more subgroups (key entries) to f['group_ICA_SI/CardiacMasks'] recently.
    # However, these entries will not be filled and will be removed later when the non-converged
    # ICA runs are removed.

    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)

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
    group_name_full = group_name + '/CardiacMasks/' + additional_group_names


    components_metrics_etc = np.zeros((len(list(f[group_name_full].keys())) - 2, \
            (1 + num_lowest_metrics), len(component_sort_metric_list)))


    # Loop over all component sorting metrics of interest (i.e. those contained within
    # component_sort_metric_list).
    for j in range(len(component_sort_metric_list)):
        # Counter for position (row number, down the number of components column) 
        # in components_metrics_etc as not all of the different ICA component numbers
        # may have converged.
        comp_counter = 0
        for numC_value in num_components_list:
            numC_key = 'NumC_' + str(numC_value)
            if f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use) + '/converged_test'][:] == 1:
                # Extract the metric values and indices from the HDF5 file.
                metrics = f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use)  + \
                    '/ComponentSorting/' + component_sort_metric_list[j] + '/sorting_metrics'][:]
                indices = f[group_name_full + '/' + numC_key + '/Run' + str(RunNum_use)  + \
                    '/ComponentSorting/' + component_sort_metric_list[j] + '/argsort_indices'][:]
                # Calculate the sorted metrics and store in components_metrics_etc.
                sorted_metrics = functions_Calculate_stats.order_components(np.array([metrics]), np.array([indices]))
                components_metrics_etc[comp_counter, 0, j] = numC_key[5:]
                # Create empty array for storing the metrics in the correct locations.
                empty_array = np.zeros((np.shape(components_metrics_etc)[1]))
                # - 1 as first column is for c_num
                min_to_fill = np.min([np.shape(components_metrics_etc)[1]-1, np.int32(numC_key[5:])])
                empty_array[1:min_to_fill+1] = np.squeeze(sorted_metrics[:,0:min_to_fill])
                components_metrics_etc[comp_counter, :, j] = empty_array
                # Store component number now.
                components_metrics_etc[comp_counter, 0, j] = numC_key[5:]
                # components_metrics_etc[comp_counter, 1:4, j] = sorted_metrics[:,0:3]
                # + 1 to comp_counter as convergence was reached for this number of components.
                comp_counter = comp_counter + 1
                # Continue as have identified and sorted the metrics for the lowest RunNumber of 
                # a specific number of ICA components and for the lowest RunNumber (for which ICA
                # convergence was reached and the metric of interest was present for the RunNumber).
                # Hence, no further loops over the RunNumber are required - use 'break' to break 
                # out of the loop and continue for the next iteration of the for loop of num_C.
                # break

    # Likely that not all ICA applications reached convergence. Remove these 
    # entries in components_metrics_etc. The lack of convergence can be 
    # identified by metric values == 0 (not saved in the HDF5 file and therefore
    # not extracted either and not set in the components_metrics_etc array).
    components_metrics_etc_new = components_metrics_etc[0:np.shape(components_metrics_etc)[0]-np.sum(components_metrics_etc[:,0,0] == 0),:,:]

    # Close the HDF5 file.
    f.close()
    # Return 
    return components_metrics_etc_new




def HDF5_extract_ICAoutputs_nifty(minNumberLookat, NumDyn, sorted_array_metric_sorted, \
    echo_num, component_sort_metric_list, nonZ_cardiac, \
    hdf5_loc, dir_date, dir_subj, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base, 
    RunNum_use):
    """
    Function to extract ICA outputs (S_ica_arry, A_ica_array).
    **For multiple metric sorting methods**

    Arguments:
    minNumberLookat = minimum number of components/metrics to be extracted.
    NumDyn = number of dynamic images acquired.
    sorted_array_metric_sorted = array of component numbers and their metrics sorted
    by the component number. Array contains the ICA component numbers 
    (for cases in which ICA reached convergence) in the first column, and 
    the 'num_lowest_metrics' lowest frequency RMS metrics in the next 
    'num_lowest_metrics' columns. The third dimension runs over the
    metrics of interest (of which there are len(component_sort_metric_list)).
    Hence, shape of (#converged_ICA, 1 + num_lowest_metrics, 
    len(component_sort_metric_list)).
    Sorted by component metric value.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    component_sort_metric_list = list of the metric names to be extracted.
    nonZ_cardiac = number of non-zero voxels within the mask.
    hdf5_loc = location of the HDF5 file which contains the metric information.
    dir_date = scan date for the particular subject.
    dir_subj = subject ID for the particular subject.
    RunNum_use = specific RunNumber to use.
    Returns:
    S_ica_array = array of the extracted S_ica time series. The minNumberLookat
    most smallest metric value components are extracted for each metric being
    investigated. The minNumberLookat run along the columns and the metrics being
    investigated along the third dimension. Shape of 
    (NumDyn, minNumberLookat, len(component_sort_metric_list)).
    A_ica_array = array of the extracted A_ica component maps. The minNumberLookat
    most smallest metric value components are extracted for each metric being
    investigated. The minNumberLookat run along the columns and the metrics being
    investigated along the third dimension. Shape of 
    (nonZ{{vox,vox,NumSlices}}, minNumberLookat, len(component_sort_metric_list)).
    min_timeseries_metrics_component_numbers = array containing the number of components
    found by ICA for each extracted ICA output.
    Of shape (minNumberLookat, len(component_sort_metric_list))
    min_timeseries_metrics_values = array containing the corresponding metric value for 
    the minimum metric value for the particular number of components found by ICA for 
    each extracted ICA output. 
    Of shape (minNumberLookat, len(component_sort_metric_list))

    """
    # Load the HDF5 file and read in (read only).
    f = h5py.File(hdf5_loc + hdf5_file_name_base + dir_date + '_' + dir_subj, 'r')
    # Empty array to store the timeseries (S_ica) in, and the maps (A_ica).
    S_ica_array = np.zeros((NumDyn, minNumberLookat, len(component_sort_metric_list)))
    A_ica_array = np.zeros((nonZ_cardiac, minNumberLookat, len(component_sort_metric_list)))
    # Empty arrays to store the corresponding metric values and component numbers.
    min_timeseries_metrics_values = np.zeros((1, minNumberLookat, len(component_sort_metric_list)))
    min_timeseries_metrics_component_numbers = np.zeros((1, minNumberLookat, len(component_sort_metric_list)))

    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num) 

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
    group_name_full = group_name + '/CardiacMasks/' + additional_group_names



    # Loop over metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Loop over #(minNumberLookat) most minimum metric vales.
        for k in range(minNumberLookat):
            # print(k)
            # Number of components for minimum metric value is given by the first column
            # of the sorted array value.
            numC_key_of_interest = "NumC_" + str(np.int32(np.rint(sorted_array_metric_sorted[k,0,j])))
            # Save the subgroup name for future reference.
            subgroup_numC = group_name_full + '/' + numC_key_of_interest
            # Extract from HDF5 file. Loop over the Run# as may not be present 
            # RunNum as given...
            # if subgroup_numC + '/' + RunNumber + '/ComponentSorting/' + component_sort_metric_list[j] in f:
            # Extract S_ica and A_ica.
            subgroup_component_of_interest = subgroup_numC + '/Run' + str(RunNum_use) + '/S_ica_tVox'
            subgroup_component_MAP_of_interest = subgroup_numC + '/Run' + str(RunNum_use) + '/A_ica_tVox'
            # Access argsort indices to find the minimum metric value component for this particular
            # metric type, NumC and RunNum.
            subgroup_metrics_of_interest = subgroup_numC + '/Run' + str(RunNum_use) + '/ComponentSorting/' + \
                component_sort_metric_list[j] + '/argsort_indices'
            component_min_index = f[subgroup_metrics_of_interest][:][0]
            # Extract the minimum metric component of S_ica and A_ica.
            S_ica_component_of_interest = f[subgroup_component_of_interest][:][:,np.int32(np.rint(component_min_index))]
            S_ica_component_of_interest = np.squeeze(S_ica_component_of_interest)
            A_ica_component_of_interest = f[subgroup_component_MAP_of_interest][:][:,np.int32(np.rint(component_min_index))]
            A_ica_component_of_interest = np.squeeze(A_ica_component_of_interest)
            # Store these values in the created arrays.
            S_ica_array[:,k,j] = S_ica_component_of_interest
            A_ica_array[:,k,j] = A_ica_component_of_interest
            # Store metric values and component number.
            subgroup_metrics_of_interest_values = subgroup_numC + '/Run' + str(RunNum_use) + '/ComponentSorting/' + \
                component_sort_metric_list[j] + '/sorting_metrics'
            min_timeseries_metrics_values[:,k,j] = f[subgroup_metrics_of_interest_values][component_min_index]
            min_timeseries_metrics_component_numbers[:,k,j] = np.int32(numC_key_of_interest[5:])
            # Continue as have identified and extracted the ICA outputs for the lowest RunNumber of 
            # a specific number of ICA components and for the lowest RunNumber (for which ICA
            # convergence was reached and the metric of interest was present for the RunNumber).
            # Hence, no further loops over the RunNumber are required - use 'break' to break 
            # out of the loop and continue for the next iteration of the for loop of num_C.
            # break

    # Squeeze metric values and the corresponding component number arrays to remove the empty dimension.
    min_timeseries_metrics_values = np.squeeze(min_timeseries_metrics_values)
    min_timeseries_metrics_component_numbers = np.int32(np.rint(np.squeeze(min_timeseries_metrics_component_numbers)))

    # Close the HDF5 file.
    f.close()
    # Return 
    return S_ica_array, A_ica_array, min_timeseries_metrics_component_numbers, min_timeseries_metrics_values



def reconstruct_single_ICA_component_nifty(hdf5_loc, dir_date, dir_subj, \
    NumDyn, nonZ_cardiac, component_sort_metric_list, echo_num, \
    S_ica_array, A_ica_array, recon_num, \
    NiftyReg_ID, densityCorr, regCorr_name, \
    ANTs_used, dir_image_nifty, hdf5_file_name_base):
    """
    Function to reconstruct a single component for each of the metrics being
    investigated. Reconstructed arrays have the shape
    (NumDyn, nonZ{{vox,vox,NumSlices}}, len(component_sort_metric_list)).
    Three reconstructions are output: reconstruction, + mean and + mean * unScaling.
    Scaling and mean are read from the HDF5 file.
    *Multiple metrics*

    Arguments:
    hdf5_loc = location of the HDF5 file which contains the metric information.
    dir_date = scan date for the particular subject.
    dir_subj = subject ID for the particular subject.
    NumDyn = number of dynamic images acquired.
    nonZ_cardiac = number of non-zero voxels within the mask.
    component_sort_metric_list = list of the metric names to be extracted.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    S_ica_array = array of the extracted S_ica time series. The minNumberLookat
    most smallest metric value components are extracted for each metric being
    investigated. The minNumberLookat run along the columns and the metrics being
    investigated along the third dimension. Shape of 
    (NumDyn, minNumberLookat, len(component_sort_metric_list)).
    A_ica_array = array of the extracted A_ica component maps. The minNumberLookat
    most smallest metric value components are extracted for each metric being
    investigated. The minNumberLookat run along the columns and the metrics being
    investigated along the third dimension. Shape of 
    (nonZ{{vox,vox,NumSlices}}, minNumberLookat, len(component_sort_metric_list)).
    recon_num = component (lowest) to be reconstructed, to index the second dimension
    of S_ica_array and A_ica_array (the minNumberLookat part).
    Returns:
    X_recon_OE_data = reconstructed components for each metric in 
    component_sort_metric_list, for the recon_num th component (lowest).
    Shape of (NumDyn, nonZ{{vox,vox,NumSlices}}, len(component_sort_metric_list)).
    X_recon_OE_data_unMean = unMean reconstructed components for each metric in 
    component_sort_metric_list, for the recon_num th component (lowest).
    Shape of (NumDyn, nonZ{{vox,vox,NumSlices}}, len(component_sort_metric_list)).
    X_recon_OE_data_unMean_unSc = unMean and * Scaling factor reconstructed 
    components for each metric in component_sort_metric_list, for the 
    recon_num th component (lowest). Shape of 
    (NumDyn, nonZ{{vox,vox,NumSlices}}, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
    
    """
    # Load the HDF5 file and read in (read only).
    f = h5py.File(hdf5_loc + hdf5_file_name_base + dir_date + '_' + dir_subj, 'r')
    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)

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
    group_name_full = group_name + '/CardiacMasks/' + additional_group_names

    # Access the data pre-processing and ICA prep details: mean and scaling. Both
    # were saved in the HDF5 file for use when reconstructing the data.
    mean_to_add = f[group_name_full + '/ica_tVox_mean'][:]
    data_scaling = f[group_name_full + '/ica_scaling'][:]

    # Reconstruct the data using a single ICA component (each time).
    # Create empty arrays to store the reconstructed data and for the unMean/uncentred
    # reconstructed data. Later, the data scaling will also be reversed to generate
    # the un-centred reconstructed data.
    X_recon_OE_data = np.zeros((NumDyn, nonZ_cardiac, len(component_sort_metric_list)))
    X_recon_OE_data_unMean = np.zeros((NumDyn, nonZ_cardiac, len(component_sort_metric_list)))

    # Loop over the different metrics being investigated and reconstruct the
    # recon_num lowest component.
    for j in range(len(component_sort_metric_list)):
        # Reconstruct the data using S_ica and A_ica.
        # Need to transpose/permute to have the correct dimensions for reconstruction.
        X_recon_OE = functions_Calculate_stats.reconstruct_component_noMean(\
            np.transpose(np.array([S_ica_array[:,recon_num-1,j]]), (1,0)), \
            np.transpose(np.array([A_ica_array[:,recon_num-1,j]]), (1,0)), 0)
        # Store the reconstructed data for a particular metric.
        X_recon_OE_data[:,:,j] = X_recon_OE
        # Calculate the un-centred reconstructed data by adding on the centring value.
        # This needs to be done within the loop.
        X_recon_OE_data_unMean[:,:,j] = X_recon_OE_data[:,:,j] + mean_to_add

    # Reverse the scaling on the un-centred reconstructed data to 'fully' reconstruct
    # the data for a particular component.
    X_recon_OE_data_unMean_unSc = np.multiply(data_scaling, X_recon_OE_data_unMean)

    # Close the HDF5 file.
    f.close()
    # Return
    return X_recon_OE_data, X_recon_OE_data_unMean, X_recon_OE_data_unMean_unSc











if __name__ == "__main__":
    # ...
    a = 1