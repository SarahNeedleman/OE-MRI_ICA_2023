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

# Functions for loading HDF5 and calculating the ordering of components etc...

import numpy as np
import Sarah_ICA
import h5py
import matplotlib.pyplot as plt
import functions_Calculate_stats


def order_component_number(component_sort_metric_list, components_metrics_etc_new):
    """
    Order components_metrics_etc_new array by the component numbers (useful for 
    plotting the component number vs metric value).

    Arguments:
    component_sort_metric_list = list of metrics being investigated.
    components_metrics_etc_new = array of component numbers and their metrics for
    the different metrics being investigated. components_metrics_etc_new contains 
    the ICA component numbers (for cases in which ICA reached convergence) 
    in the first column, and the 'num_lowest_metrics' lowest frequency RMS 
    metrics in the next 'num_lowest_metrics' columns. The third dimension 
    runs over the metrics of interest (of which there are 
    len(component_sort_metric_list)). Hence, shape of 
    (#converged_ICA, 1 + num_lowest_metrics, len(component_sort_metric_list)).
    Returns:
    sorted_array = array of component numbers and their metrics sorted
    by the component number. This array has the same shape as 
    components_metrics_etc_new.
    
    """
    # Order components_metrics_etc_new in terms of increasing component number 
    # (number of ICA components found, contained within the 1st column).
    # Create empty array to form the sorted array of the same size as the
    # input array components_metrics_etc_new which is being sorted.
    sorted_array = np.multiply(0,components_metrics_etc_new)

    # Loop over the different metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Order manually due to direction/orientation (vs ordering function 
        # previously created).
        # Calculate ordering indices - sorting by component number which 
        # is contained in the first column (0 in Python).
        components_metrics_argsort_indices = np.argsort(components_metrics_etc_new[:,0,j])
        # Empty array to store indices with the 'full' shape of the array
        # to be re-ordered.
        arg_sort_indices_ALL = np.zeros(np.shape(components_metrics_etc_new[:,:,j]))
        # Loop over the first (Python = 2) dimension of the array to be ordered 
        # (likely to be NumDyn), and 'extend' the ordering indices from a single dimensional
        # array, to an array of the same shape of that to be re-ordered.
        # An array containing the indices having the same shape as the array to 
        # be re-ordered is required by python.
        for k in range(np.shape(components_metrics_etc_new[:,:,j])[1]):
            arg_sort_indices_ALL[:,k] = components_metrics_argsort_indices
            # Ensure sorting indices are integers.
            arg_sort_indices_ALL = np.int32(np.rint(arg_sort_indices_ALL))
            # Perform and store the sorted components.
            sorted_array[:,:,j] = np.take_along_axis(components_metrics_etc_new[:,:,j], arg_sort_indices_ALL, axis=0)
    # Return the sorted array.
    return sorted_array


def order_component_metric(component_sort_metric_list, sorted_array):
    """
    Order components_metrics_etc_new/sorted_array array by the 
    metric values. To be used for identifying the component with the 
    lowest metric value.

    Arguments:
    component_sort_metric_list = list of metrics being investigated.
    sorted_array = array of component numbers and their metrics for
    the different metrics being investigated. components_metrics_etc_new contains 
    the ICA component numbers (for cases in which ICA reached convergence) 
    in the first column, and the 'num_lowest_metrics' lowest frequency RMS 
    metrics in the next 'num_lowest_metrics' columns. The third dimension 
    runs over the metrics of interest (of which there are 
    len(component_sort_metric_list)). Hence, shape of 
    (#converged_ICA, 1 + num_lowest_metrics, len(component_sort_metric_list)).
    Sorted by component number from order_component_number.
    Returns:
    sorted_array_metric_sorted = array of component numbers and their metrics sorted
    by the component number. This array has the same shape as 
    components_metrics_etc_new.
    
    """
    # Sort components_metrics_etc_new//sorted_array by the metric value
    # (lowest metric value in the second column).
    # Create an empty array to store the sorted metric values for all metrics, and
    # to store the sorting indices for sorting the array.
    arg_sort_indices_metric = np.zeros((np.shape(sorted_array)[0],np.shape(sorted_array)[1]))
    sorted_array_metric_sorted = np.zeros(np.shape(sorted_array))

    # Loop over all metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Calculate ordering indices - sort by second column (1 in Python).
        sorted_array_metric_argsort_indices = np.argsort(sorted_array[:,1,j])
        # Loop over the first (Python = 2) dimension of the array to be ordered 
        # (likely to be NumDyn), and 'extend' the ordering indices from a single dimensional
        # array, to an array of the same shape of that to be re-ordered.
        # An array containing the indices having the same shape as the array to 
        # be re-ordered is required by python.
        for k in range(np.shape(sorted_array)[1]):
            arg_sort_indices_metric[:,k] = sorted_array_metric_argsort_indices
        # Ensure sorting indices are integers.
        arg_sort_indices_metric = np.int32(arg_sort_indices_metric)
        # Perform and store the sorted components.
        sorted_array_metric_sorted[:,:,j] = np.take_along_axis(sorted_array[:,:,j], arg_sort_indices_metric, axis=0)
    # Return the sorted array.
    return sorted_array_metric_sorted


def order_component_metric_V2_specificNumber(component_sort_metric_list, sorted_array_V2, spec_Num):
    """
    Sorts by spec_Num metric value (column) in sorted_array_V2, finding the 'next' lowest
    component metrics.
    For spec_Num > lowest numC, may have metric values = 0, therefore an additional step
    is required here to set these metric values to 99 so they will not be assigned
    as being the lowest metric value.    
    
    Arguments:
    component_sort_metric_list = list of metrics being investigated.
    sorted_array_V2 = array of component numbers and their metrics for
    the different metrics being investigated. components_metrics_etc_new contains 
    the ICA component numbers (for cases in which ICA reached convergence) 
    in the first column, and the 'num_lowest_metrics' lowest frequency RMS 
    metrics in the next 'num_lowest_metrics' columns. The third dimension 
    runs over the metrics of interest (of which there are 
    len(component_sort_metric_list)). Hence, shape of 
    (#converged_ICA, 1 + num_lowest_metrics, len(component_sort_metric_list)).
    Sorted by component number from order_component_number.
    spec_Num = metric number (column) to sort by.
    Returns:
    sorted_array_metric_sorted_V3 = array of component numbers and their metrics
    values sorted by the spec_Num th metric value. This array has the same shape
    as components_metrics_etc_new//sorted_array_V2.

    """
    # First, find the locations of zero metric values in the sorted array and
    # set these zero values to 99 (so they are not automatically found as being
    # the lowest metric values).
    # Use indexing for speed.
    loc_zeros_index = np.where(sorted_array_V2 == 0)
    # Array of the input sorting array in which 0 metric values are to be set to
    # 99.
    sorted_array_V3 = np.array((sorted_array_V2))
    # Loop over the zero values and assign these values to have a value of 99.
    for j in range(np.int32(np.rint(np.sum(sorted_array_V2 == 0)))):
        # Identify the coordinates of the jth metric value which equals zero.
        coords_index = np.array((loc_zeros_index[0][j], loc_zeros_index[1][j], loc_zeros_index[2][j]))
        # Assign the metric value to instead take the value of 99.
        sorted_array_V3[coords_index[0],coords_index[1],coords_index[2]] = 99

    # Sort components_metrics_etc_new by the metric value
    # (lowest metric value in the spec_Num column).
    # Create an empty array to store the sorted metric values for all metrics
    # and an array to store the sorting indices.
    arg_sort_indices_metric = np.zeros((np.shape(sorted_array_V3)[0],np.shape(sorted_array_V3)[1]))
    sorted_array_metric_sorted_V3 = np.zeros(np.shape(sorted_array_V3))

    # Loop over the metrics to be investigated.
    for j in range(len(component_sort_metric_list)):
        # Calculate ordering indices - sort by spec_Num column (1 in Python).
        sorted_array_metric_argsort_indices = np.argsort(sorted_array_V3[:,spec_Num,j])
        # Loop over the first (Python = 2) dimension of the array to be ordered 
        # (likely to be NumDyn), and 'extend' the ordering indices from a single dimensional
        # array, to an array of the same shape of that to be re-ordered.
        for k in range(np.shape(sorted_array_V3)[1]):
            arg_sort_indices_metric[:,k] = sorted_array_metric_argsort_indices

        # Ensure sorting indices are integers.
        arg_sort_indices_metric = np.int32(arg_sort_indices_metric)
        # Perform and store the sorted components.
        sorted_array_metric_sorted_V3[:,:,j] = np.take_along_axis(sorted_array_V3[:,:,j], arg_sort_indices_metric, axis=0)

    # Return the sorted array.
    return sorted_array_metric_sorted_V3


def HDF5_extract_ICAoutputs(minNumberLookat, NumDyn, sorted_array_metric_sorted, \
    echo_num, component_sort_metric_list, nonZ_cardiac, \
    hdf5_loc, dir_date, dir_subj):
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
    f = h5py.File(hdf5_loc + "hdf5_structure_" + dir_date + '_' + dir_subj, 'r')
    # Empty array to store the timeseries (S_ica) in, and the maps (A_ica).
    S_ica_array = np.zeros((NumDyn, minNumberLookat, len(component_sort_metric_list)))
    A_ica_array = np.zeros((nonZ_cardiac, minNumberLookat, len(component_sort_metric_list)))
    # Empty arrays to store the corresponding metric values and component numbers.
    min_timeseries_metrics_values = np.zeros((1, minNumberLookat, len(component_sort_metric_list)))
    min_timeseries_metrics_component_numbers = np.zeros((1, minNumberLookat, len(component_sort_metric_list)))

    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)

    # Loop over metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Loop over #(minNumberLookat) most minimum metric vales.
        for k in range(minNumberLookat):
            # Number of components for minimum metric value is given by the first column
            # of the sorted array value.
            numC_key_of_interest = "NumC_" + str(np.int32(np.rint(sorted_array_metric_sorted[k,0,j])))
            # Save the subgroup name for future reference.
            subgroup_numC = group_name + '/CardiacMasks/' + numC_key_of_interest
            # Extract from HDF5 file. Loop over the Run# as may not be present 
            # for one particular metric. Assume use of the lowest Run#.
            for RunNumber in f[subgroup_numC].keys():
                # Check if the particular metric is present for the number of 
                # components and RunNumber. If not, the for loop will run to the 
                # next RunNumber.
                if subgroup_numC + '/' + RunNumber + '/ComponentSorting/' + component_sort_metric_list[j] in f:
                    # Extract S_ica and A_ica.
                    subgroup_component_of_interest = subgroup_numC + '/' + RunNumber + '/S_ica_tVox'
                    subgroup_component_MAP_of_interest = subgroup_numC + '/' + RunNumber + '/A_ica_tVox'
                    # Access argsort indices to find the minimum metric value component for this particular
                    # metric type, NumC and RunNum.
                    subgroup_metrics_of_interest = subgroup_numC + '/' + RunNumber + '/ComponentSorting/' + \
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
                    subgroup_metrics_of_interest_values = subgroup_numC + '/' + RunNumber + '/ComponentSorting/' + \
                        component_sort_metric_list[j] + '/sorting_metrics'
                    min_timeseries_metrics_values[:,k,j] = f[subgroup_metrics_of_interest_values][component_min_index]
                    min_timeseries_metrics_component_numbers[:,k,j] = np.int32(numC_key_of_interest[5:])
                    # Continue as have identified and extracted the ICA outputs for the lowest RunNumber of 
                    # a specific number of ICA components and for the lowest RunNumber (for which ICA
                    # convergence was reached and the metric of interest was present for the RunNumber).
                    # Hence, no further loops over the RunNumber are required - use 'break' to break 
                    # out of the loop and continue for the next iteration of the for loop of num_C.
                    break

    # Squeeze metric values and the corresponding component number arrays to remove the empty dimension.
    min_timeseries_metrics_values = np.squeeze(min_timeseries_metrics_values)
    min_timeseries_metrics_component_numbers = np.int32(np.rint(np.squeeze(min_timeseries_metrics_component_numbers)))
    
    # Close the HDF5 file.
    f.close()
    # Return 
    return S_ica_array, A_ica_array, min_timeseries_metrics_component_numbers, min_timeseries_metrics_values



def HDF5_extract_S_ica(minNumberLookat, NumDyn, \
    sorted_array_metric_sorted, echo_num, component_sort_metric_list, \
    spec_Num, hdf5_loc, dir_date, dir_subj):
    """
    Function to extract S_ica output.

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
    spec_Num = specific lowest number to be extracted, if the lowest
    value is not necessarily desired.
    But it will be, e.g. 2nd lowest for the components in the sorted
    array. The sorted array will be sorted on metric value and can be sorted
    for e.g. second lowest metric value (3rd column) which can match up.
    hdf5_loc = location of the HDF5 file which contains the metric information.
    dir_date = scan date for the particular subject.
    dir_subj = subject ID for the particular subject.
    Returns:
    S_ica_array = array of the extracted S_ica time series. The minNumberLookat
    most smallest metric value components are extracted for each metric being
    investigated. The minNumberLookat run along the columns and the metrics being
    investigated along the third dimension. Shape of 
    (NumDyn, minNumberLookat, len(component_sort_metric_list)).  

    """
    # Load the HDF5 file and read in (read only).
    f = h5py.File(hdf5_loc + "hdf5_structure_" + dir_date + '_' + dir_subj, 'r')

    # Empty array to store the timeseries (S_ica) in.
    S_ica_array = np.zeros((NumDyn, minNumberLookat, len(component_sort_metric_list)))
    # Empty arrays to store the corresponding metric values and component numbers.
    min_timeseries_metrics_values = np.zeros((1, minNumberLookat, len(component_sort_metric_list)))
    min_timeseries_metrics_component_numbers = np.zeros((1, minNumberLookat, len(component_sort_metric_list)))

    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)

    # Loop over metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Loop over #(minNumberLookat) most minimum metric vales.
        for k in range(minNumberLookat):
            # Number of components for minimum metric value is given by the first column
            # of the sorted array value.
            numC_key_of_interest = "NumC_" + str(np.int32(np.rint(sorted_array_metric_sorted[k,0,j])))
            # Save the subgroup name for future reference.
            subgroup_numC = group_name + '/CardiacMasks/' + numC_key_of_interest
            # Extract from HDF5 file. Loop over the Run# as may not be present 
            # for one particular metric. Assume use of the lowest Run#.
            for RunNumber in f[subgroup_numC].keys():
                # Check if the particular metric is present for the number of 
                # components and RunNumber. If not, the for loop will run to the 
                # next RunNumber.
                if str(subgroup_numC) + '/' + str(RunNumber) + '/ComponentSorting/' + str(component_sort_metric_list[j]) in f:
                    # Extract S_ica.
                    subgroup_component_of_interest = subgroup_numC + '/' + RunNumber + '/S_ica_tVox'
                    # Access argsort indices to find the minimum metric value component for this particular
                    # metric type, NumC and RunNum.
                    subgroup_metrics_of_interest = subgroup_numC + '/' + RunNumber + '/ComponentSorting/' + \
                        component_sort_metric_list[j] + '/argsort_indices'
                    # Find the index (row) of the minimum metric valued component, but
                    # choose spec_Num-1 for the spec_Num lowest.
                    component_min_index = f[subgroup_metrics_of_interest][:][spec_Num-1]
                    # Extract the minimum metric component of S_ica.
                    S_ica_component_of_interest = f[subgroup_component_of_interest][:][:,np.int32(np.rint(component_min_index))]
                    S_ica_component_of_interest = np.squeeze(S_ica_component_of_interest)
                    # Store these values in the created arrays.
                    S_ica_array[:,k,j] = S_ica_component_of_interest
                    # Store metric values and component number.
                    subgroup_metrics_of_interest_values = subgroup_numC + '/' + RunNumber + '/ComponentSorting/' + \
                        component_sort_metric_list[j] + '/sorting_metrics'
                    min_timeseries_metrics_values[:,k,j] = f[subgroup_metrics_of_interest_values][component_min_index]
                    min_timeseries_metrics_component_numbers[:,k,j] = np.int32(np.rint(sorted_array_metric_sorted[k,0,j]))
                    # Continue as have identified and extracted the ICA outputs for the lowest RunNumber of 
                    # a specific number of ICA components and for the lowest RunNumber (for which ICA
                    # convergence was reached and the metric of interest was present for the RunNumber).
                    # Hence, no further loops over the RunNumber are required - use 'break' to break 
                    # out of the loop and continue for the next iteration of the for loop of num_C.
                    break

    # Squeeze metric values and the corresponding component number arrays to remove the empty dimension.
    min_timeseries_metrics_values = np.squeeze(min_timeseries_metrics_values)
    min_timeseries_metrics_component_numbers = np.int32(np.rint(np.squeeze(min_timeseries_metrics_component_numbers)))
    
    # Close the HDF5 file.
    f.close()
    # Return 
    return S_ica_array, min_timeseries_metrics_component_numbers, min_timeseries_metrics_values


def reconstruct_single_ICA_component(hdf5_loc, dir_date, dir_subj, \
    NumDyn, nonZ_cardiac, component_sort_metric_list, echo_num, \
    S_ica_array, A_ica_array, recon_num):
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
    f = h5py.File(hdf5_loc + "hdf5_structure_" + dir_date + '_' + dir_subj, 'r')
    group_name = "group_ICA_SI" + "_" + "Echo" + str(echo_num)

    # Access the data pre-processing and ICA prep details: mean and scaling. Both
    # were saved in the HDF5 file for use when reconstructing the data.
    mean_to_add = f[group_name + '/CardiacMasks/ica_tVox_mean'][:]
    data_scaling = f[group_name + '/CardiacMasks/ica_scaling'][:]

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


def reshape_maps_DYNAMICS(NumDyn, component_sort_metric_list, size_echo_data, \
    masks_reg_data_cardiac, \
    X_recon_OE_data, X_recon_OE_data_unMean, X_recon_OE_data_unMean_unSc):
    """
    Reshape component maps of a dynamic series. Reshaping collapsed data using
    masks.
    *For multiple reconstructed OE map versions*
    *For multiple component sorting methods?*
    
    Arguments:
    NumDyn = number of dynamic images acquired.
    component_sort_metric_list = list of the metric names to be extracted.
    size_echo_data = shape of the loaded MRI data to base the reshaped maps on.
    masks_reg_data_cardiac = masks applied to the original data to use 
    for reshaping the collapsed array.
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
    Returns:
    X_recon_OE_maps = maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_maps = unMean maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_unSc_maps = unMean and * Scaling factor maps of the 
    reconstructed component over the dynamic series. Shape 
    (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
        
    """
    # Reshape the reconstructed data to obtain a dynamic series of reconstructed images.
    # Create empty arrays to store the reshaped dynamic series in.
    # Require the empty arrays to have shape (NumDyn, len(numMetrics), vox,vox,NumSlices)
    # due to the form of the reshaping function.
    X_recon_OE_maps = np.zeros((size_echo_data[2], len(component_sort_metric_list), \
        size_echo_data[0], size_echo_data[1], size_echo_data[3]))
    X_recon_OE_unMean_maps = np.zeros((size_echo_data[2], len(component_sort_metric_list), \
        size_echo_data[0], size_echo_data[1], size_echo_data[3]))
    X_recon_OE_unMean_unSc_maps = np.zeros((size_echo_data[2], len(component_sort_metric_list), \
        size_echo_data[0], size_echo_data[1], size_echo_data[3]))

    # Loop over the dynamic time points (due to the nature of the reshaping function).
    # NumSlices -- same as size_echo_data[3]
    for j in range(NumDyn):
        X_recon_OE_maps[j,:,:,:] = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, size_echo_data[3], \
            len(component_sort_metric_list), masks_reg_data_cardiac, np.squeeze(X_recon_OE_data[j,:,:]))
        X_recon_OE_unMean_maps[j,:,:,:] = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, size_echo_data[3], \
            len(component_sort_metric_list), masks_reg_data_cardiac, np.squeeze(X_recon_OE_data_unMean[j,:,:]))
        X_recon_OE_unMean_unSc_maps[j,:,:,:] = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, size_echo_data[3], \
            len(component_sort_metric_list), masks_reg_data_cardiac, np.squeeze(X_recon_OE_data_unMean_unSc[j,:,:]))

    # Need to permute (np.transpose()) the maps for the shape (vox,vox,NumDyn,NumSlices, metric_num).
    X_recon_OE_maps = np.transpose(X_recon_OE_maps, (2,3,0,4,1))
    X_recon_OE_unMean_maps = np.transpose(X_recon_OE_unMean_maps, (2,3,0,4,1))
    X_recon_OE_unMean_unSc_maps = np.transpose(X_recon_OE_unMean_unSc_maps, (2,3,0,4,1))
    return X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps


def reshape_maps_DYNAMICS_noSqueeze(NumDyn, component_sort_metric_list, size_echo_data, \
    masks_reg_data_cardiac, \
    X_recon_OE_data, X_recon_OE_data_unMean, X_recon_OE_data_unMean_unSc):
    """
    Reshape component maps of a dynamic series. Reshaping collapsed data using
    masks.
    *For multiple reconstructed OE map versions*
    *For SINGLE component metric sorting method*
    
    Arguments:
    NumDyn = number of dynamic images acquired.
    component_sort_metric_list = list of the metric names to be extracted.
    size_echo_data = shape of the loaded MRI data to base the reshaped maps on.
    masks_reg_data_cardiac = masks applied to the original data to use 
    for reshaping the collapsed array.
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
    Returns:
    X_recon_OE_maps = maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_maps = unMean maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_unSc_maps = unMean and * Scaling factor maps of the 
    reconstructed component over the dynamic series. Shape 
    (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
        
    """
    # Reshape the reconstructed data to obtain a dynamic series of reconstructed images.
    # Create empty arrays to store the reshaped dynamic series in.
    # Require the empty arrays to have shape (NumDyn, len(numMetrics), vox,vox,NumSlices)
    # due to the form of the reshaping function.
    X_recon_OE_maps = np.zeros((size_echo_data[2], len(component_sort_metric_list), \
        size_echo_data[0], size_echo_data[1], size_echo_data[3]))
    X_recon_OE_unMean_maps = np.zeros((size_echo_data[2], len(component_sort_metric_list), \
        size_echo_data[0], size_echo_data[1], size_echo_data[3]))
    X_recon_OE_unMean_unSc_maps = np.zeros((size_echo_data[2], len(component_sort_metric_list), \
        size_echo_data[0], size_echo_data[1], size_echo_data[3]))

    # Loop over the dynamic time points (due to the nature of the reshaping function).
    # NumSlices -- same as size_echo_data[3]
    for j in range(NumDyn):
        X_recon_OE_maps[j,:,:,:] = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, size_echo_data[3], \
            len(component_sort_metric_list), masks_reg_data_cardiac, X_recon_OE_data[j,:,:])
        X_recon_OE_unMean_maps[j,:,:,:] = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, size_echo_data[3], \
            len(component_sort_metric_list), masks_reg_data_cardiac, X_recon_OE_data_unMean[j,:,:])
        X_recon_OE_unMean_unSc_maps[j,:,:,:] = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, size_echo_data[3], \
            len(component_sort_metric_list), masks_reg_data_cardiac, X_recon_OE_data_unMean_unSc[j,:,:])

    # Need to permute (np.transpose()) the maps for the shape (vox,vox,NumDyn,NumSlices, metric_num).
    X_recon_OE_maps = np.transpose(X_recon_OE_maps, (2,3,0,4,1))
    X_recon_OE_unMean_maps = np.transpose(X_recon_OE_unMean_maps, (2,3,0,4,1))
    X_recon_OE_unMean_unSc_maps = np.transpose(X_recon_OE_unMean_unSc_maps, (2,3,0,4,1))
    return X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps


def calc_PSE_maps_metrics(av_im_num, \
    component_sort_metric_list, GasSwitch, \
    X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps):
    """ 
    Calculating PSE image for each slice of an input dynamic series of image 
    maps.
    *For multiple reconstructed OE map versions*
    *For MULTIPLE metrics?*

    Arguments:
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    component_sort_metric_list = list of the metric names being investigated.
    The PSE will be calculated for each reconstructed dynamic series for
    each metric.
    GasSwitch = dynamic image number at which the gases were switched.
    X_recon_OE_maps = maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_maps = unMean maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_unSc_maps = unMean and * Scaling factor maps of the 
    reconstructed component over the dynamic series. Shape 
    (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
    Returns:
    PSE_image_map = PSE image maps calculated for each slice and for each metric 
    being investigated. Shape (vox, vox, NumSlices, len(component_sort_metric_list)).
    ** This will be abut zero, for bwr plotting.
    PSE_image_map_unMean = unMean PSE image maps calculated for each slice and for each metric 
    being investigated. Shape (vox, vox, NumSlices, len(component_sort_metric_list)).
    PSE_image_map_unMean_unSc = unMean and unscaling PSE image maps calculated for 
    each slice and for each metric being investigated. Shape 
    (vox, vox, NumSlices, len(component_sort_metric_list)).
    **This will have the same magnitude/be comparable to the MRI SI data.

    """
    # Create empty arrays to store the PSE images in.
    PSE_image_map = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))
    PSE_image_map_unMean = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))
    PSE_image_map_unMean_unSc = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))

    # Loop over the different metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Call the Sarah_ICA.PSE_map() function to calculate the PSE map, and pass the number
        # of images to average over when calculating the mean air and oxy images.
        # Then store the PSE map for each metric.
        PercentageChange_image, PSE_image_map_single, Difference_image, Air_map_cycle_1, Mean_Air_map, Mean_Oxy_map = \
            Sarah_ICA.PSE_map(X_recon_OE_maps[:,:,:,:,j], GasSwitch, av_im_num)
        PSE_image_map[:,:,:,j] = PSE_image_map_single
        
        PercentageChange_image, PSE_image_map_unMean_single, Difference_image, Air_map_cycle_1, Mean_Air_map, Mean_Oxy_map = \
            Sarah_ICA.PSE_map(X_recon_OE_unMean_maps[:,:,:,:,j], GasSwitch, av_im_num)
        PSE_image_map_unMean[:,:,:,j] = PSE_image_map_unMean_single
        
        PercentageChange_image, PSE_image_map_unMean_unSc_single, Difference_image, Air_map_cycle_1, Mean_Air_map, Mean_Oxy_map = \
            Sarah_ICA.PSE_map(X_recon_OE_unMean_unSc_maps[:,:,:,:,j], GasSwitch, av_im_num)
        PSE_image_map_unMean_unSc[:,:,:,j] = PSE_image_map_unMean_unSc_single

    # Return PSE map calculations.
    return PSE_image_map, PSE_image_map_unMean, PSE_image_map_unMean_unSc


def PSE_map_airCycle1(data_map, GasSwitch, av_im_num, baseline_delay=0):
    """
    Calculation of the PSE map with respect to the baseline air image formed from 
    the average air image during the initial air cycle (0:GasSwitch).
    **Alternative** function to PSE_map in Sarah_ICA.

    data_map = dynamic series of images for which the PSE is to be calculated.
    With shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    PercentageChange_image = percentage change image, relative enhancement ratio,
    equal to Oxy_mean / Air_mean. With shape (vox,vox,NumSlices).    
    PSE_image = PSE image, percentage signal enhancement (difference) image,
    equal to ([Oxy_mean - Air_mean] / Air_mean)*100. With shape (vox,vox,NumSlices).
    Difference_image = difference (subtraction) image, Oxy_mean - Air_mean.
    Air_map_baseline = mean air image during the first cycle (vox,vox,NumSlices). This
    image is taken as the baseline image which is used for calculating the PSE
    over the dynamic series (vs mean oxy in each cycle).
    Mean_Oxy_map = mean air image over all three cycles (vox,vox,NumSlices). 
    PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3 = PSE images for 
    each gas cycle, using the average image from the last av_im_num timepoints
    of each oxy gas cycle. Shape of (vox,vox,NumSlices).

    """
    # Calculate the mean baseline and plateau images, with the 'plateau'/oxy images
    # calculated for ecah cycle, over the *last* av_im_num of each cycle.
    Air_map_baseline = np.mean(data_map[:,:,baseline_delay:GasSwitch,:], axis=2)
    Oxy_map_cycle_1 = np.mean(data_map[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:], axis=2)
    Oxy_map_cycle_2 = np.mean(data_map[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:], axis=2)
    Oxy_map_cycle_3 = np.mean(data_map[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:], axis=2)
    
    # Mean of oxy cycles
    Mean_Oxy_map = np.divide(Oxy_map_cycle_1 + Oxy_map_cycle_2 + Oxy_map_cycle_3, 3)
    
    # Calculate PSE for mean Oxy and for each oxy cycle.
    PSE_image = np.multiply(np.divide((Mean_Oxy_map - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    PSE_image_cycle1 = np.multiply(np.divide((Oxy_map_cycle_1 - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    PSE_image_cycle2 = np.multiply(np.divide((Oxy_map_cycle_2 - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    PSE_image_cycle3 = np.multiply(np.divide((Oxy_map_cycle_3 - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    
    # Calculate PercentageChange_image - for the mean oxy map.
    PercentageChange_image = np.divide(Mean_Oxy_map, Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0)
    # Calculate Difference_image - for the mean oxy map.
    Difference_image = np.array(Mean_Oxy_map - Air_map_baseline)
    
    # Return PSE maps
    return PercentageChange_image, PSE_image, Difference_image, Air_map_baseline, \
        Mean_Oxy_map, PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3


def PSE_map_airCycle1_excludeVoxels(data_map, GasSwitch, av_im_num, voxels_maps_exclude, baseline_delay=0):
    """
    Calculation of the PSE map with respect to the baseline air image formed from 
    the average air image during the initial air cycle (0:GasSwitch).
    **Alternative** function to PSE_map in Sarah_ICA.
    **AND EXCLUDING VOXELS e.g. not fitted to**

    data_map = dynamic series of images for which the PSE is to be calculated.
    With shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    voxels_maps_exclude = Boolean array of voxels that the fitting is not performed on,
    in the form of the input data (representing both echoes in a single array dimension), with 
    shape (vox,vox,NumDyn,NumSlices).
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    PercentageChange_image = percentage change image, relative enhancement ratio,
    equal to Oxy_mean / Air_mean. With shape (vox,vox,NumSlices).    
    PSE_image = PSE image, percentage signal enhancement (difference) image,
    equal to ([Oxy_mean - Air_mean] / Air_mean)*100. With shape (vox,vox,NumSlices).
    Difference_image = difference (subtraction) image, Oxy_mean - Air_mean.
    Air_map_baseline = mean air image during the first cycle (vox,vox,NumSlices). This
    image is taken as the baseline image which is used for calculating the PSE
    over the dynamic series (vs mean oxy in each cycle).
    Mean_Oxy_map = mean air image over all three cycles (vox,vox,NumSlices). 
    PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3 = PSE images for 
    each gas cycle, using the average image from the last av_im_num timepoints
    of each oxy gas cycle. Shape of (vox,vox,NumSlices).

    """
    # Calculate the mean baseline and plateau images, with the 'plateau'/oxy images
    # calculated for ecah cycle, over the *last* av_im_num of each cycle.
    Air_map_baseline = np.sum(data_map[:,:,baseline_delay:GasSwitch,:], axis=2)
    Oxy_map_cycle_1 = np.sum(data_map[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:], axis=2)
    Oxy_map_cycle_2 = np.sum(data_map[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:], axis=2)
    Oxy_map_cycle_3 = np.sum(data_map[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:], axis=2)

    # For mean, take into account the number of voxels present (and/or not present
    # due to lack of fitting).
    # voxels_maps_exclude = np.array(voxels_maps_exclude, dtype=float)

    Air_map_baseline = np.divide(Air_map_baseline, np.sum(voxels_maps_exclude[:,:,baseline_delay:GasSwitch,:] < 1, axis=2), out=np.zeros_like(Air_map_baseline), where=np.sum(voxels_maps_exclude[:,:,baseline_delay:GasSwitch,:] < 1, axis=2)!=0)
    Oxy_map_cycle_1 = np.divide(Oxy_map_cycle_1, np.sum(voxels_maps_exclude[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:] < 1, axis=2), out=np.zeros_like(Oxy_map_cycle_1), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:] < 1, axis=2)!=0)
    Oxy_map_cycle_2 = np.divide(Oxy_map_cycle_2, np.sum(voxels_maps_exclude[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:] < 1, axis=2), out=np.zeros_like(Oxy_map_cycle_2), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:] < 1, axis=2)!=0)
    Oxy_map_cycle_3 = np.divide(Oxy_map_cycle_3, np.sum(voxels_maps_exclude[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:] < 1, axis=2), out=np.zeros_like(Oxy_map_cycle_3), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:] < 1, axis=2)!=0)

    # Mean of oxy cycles
    Mean_Oxy_map = np.divide(Oxy_map_cycle_1 + Oxy_map_cycle_2 + Oxy_map_cycle_3, 3)

    # Calculate PSE for mean Oxy and for each oxy cycle.
    PSE_image = np.multiply(np.divide((Mean_Oxy_map - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    PSE_image_cycle1 = np.multiply(np.divide((Oxy_map_cycle_1 - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    PSE_image_cycle2 = np.multiply(np.divide((Oxy_map_cycle_2 - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)
    PSE_image_cycle3 = np.multiply(np.divide((Oxy_map_cycle_3 - Air_map_baseline), Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0), 100)

    # Calculate PercentageChange_image - for the mean oxy map.
    PercentageChange_image = np.divide(Mean_Oxy_map, Air_map_baseline, out=np.zeros_like(Air_map_baseline), where=Air_map_baseline!=0)
    # Calculate Difference_image - for the mean oxy map.
    Difference_image = np.array(Mean_Oxy_map - Air_map_baseline)

    # Return PSE maps
    return PercentageChange_image, PSE_image, Difference_image, Air_map_baseline, \
        Mean_Oxy_map, PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3


def gas_maps_each_cycle(data_map, GasSwitch, av_im_num, baseline_delay=0):
    """
    Calculation of image/parameter maps during each gas duration.

    data_map = dynamic series of images for which the PSE is to be calculated.
    With shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    Air_map_cycle_1, Air_map_cycle_2, Air_map_cycle_3, Air_map_cycle_4,
    Oxy_map_cycle_1, Oxy_map_cycle_2, Oxy_map_cycle_3
    With shapes of (vox,vox,NumSlices).

    """
    # Calculate the mean baseline and plateau images, with the 'plateau'/oxy images
    # calculated for ecah cycle, over the *last* av_im_num of each cycle.
    Air_map_cycle_1 = np.mean(data_map[:,:,baseline_delay:GasSwitch,:], axis=2)
    Air_map_cycle_2 = np.mean(data_map[:,:,GasSwitch*3-av_im_num:GasSwitch*3,:], axis=2)
    Air_map_cycle_3 = np.mean(data_map[:,:,GasSwitch*5-av_im_num:GasSwitch*5,:], axis=2)
    Air_map_cycle_4 = np.mean(data_map[:,:,GasSwitch*7-av_im_num:GasSwitch*7,:], axis=2)

    Oxy_map_cycle_1 = np.mean(data_map[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:], axis=2)
    Oxy_map_cycle_2 = np.mean(data_map[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:], axis=2)
    Oxy_map_cycle_3 = np.mean(data_map[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:], axis=2)
    
    # Return maps
    return Air_map_cycle_1, Air_map_cycle_2, Air_map_cycle_3, Air_map_cycle_4, \
        Oxy_map_cycle_1, Oxy_map_cycle_2, Oxy_map_cycle_3


def gas_maps_each_cycle_excludeVoxels(data_map, GasSwitch, av_im_num, voxels_maps_exclude, baseline_delay=0):
    """
    Calculation of image/parameter maps during each gas duration.
    **AND EXCLUDING VOXELS e.g. not fitted to**

    data_map = dynamic series of images for which the PSE is to be calculated.
    With shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    voxels_maps_exclude = Boolean array of voxels that the fitting is not performed on,
    in the form of the input data (representing both echoes in a single array dimension), with 
    shape (vox,vox,NumDyn,NumSlices).
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    Air_map_cycle_1, Air_map_cycle_2, Air_map_cycle_3, Air_map_cycle_4,
    Oxy_map_cycle_1, Oxy_map_cycle_2, Oxy_map_cycle_3
    Each with shape (vox,vox,NumSlices).

    """
    # Calculate the mean baseline and plateau images, with the 'plateau'/oxy images
    # calculated for each cycle, over the *last* av_im_num of each cycle.
    Air_map_cycle_1 = np.sum(data_map[:,:,baseline_delay:GasSwitch,:], axis=2)
    Air_map_cycle_2 = np.sum(data_map[:,:,GasSwitch*3-av_im_num:GasSwitch*3,:], axis=2)
    Air_map_cycle_3 = np.sum(data_map[:,:,GasSwitch*5-av_im_num:GasSwitch*5,:], axis=2)
    Air_map_cycle_4 = np.sum(data_map[:,:,GasSwitch*7-av_im_num:GasSwitch*7,:], axis=2)

    Oxy_map_cycle_1 = np.sum(data_map[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:], axis=2)
    Oxy_map_cycle_2 = np.sum(data_map[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:], axis=2)
    Oxy_map_cycle_3 = np.sum(data_map[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:], axis=2)

    # For mean, take into account the number of voxels present (and/or not present
    # due to lack of fitting).
    voxels_maps_exclude = voxels_maps_exclude.astype('float64')

    Air_map_cycle_1 = np.divide(Air_map_cycle_1, np.sum(voxels_maps_exclude[:,:,baseline_delay:GasSwitch,:] < 1, axis=2), out=np.zeros_like(Air_map_cycle_1), where=np.sum(voxels_maps_exclude[:,:,baseline_delay:GasSwitch,:] < 1, axis=2)!=0)
    Air_map_cycle_2 = np.divide(Air_map_cycle_2, np.sum(voxels_maps_exclude[:,:,GasSwitch*3-av_im_num:GasSwitch*3,:] < 1, axis=2), out=np.zeros_like(Air_map_cycle_2), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*3-av_im_num:GasSwitch*3,:] < 1, axis=2)!=0)
    Air_map_cycle_3 = np.divide(Air_map_cycle_3, np.sum(voxels_maps_exclude[:,:,GasSwitch*5-av_im_num:GasSwitch*5,:] < 1, axis=2), out=np.zeros_like(Air_map_cycle_3), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*5-av_im_num:GasSwitch*5,:] < 1, axis=2)!=0)
    Air_map_cycle_4 = np.divide(Air_map_cycle_4, np.sum(voxels_maps_exclude[:,:,GasSwitch*7-av_im_num:GasSwitch*7,:] < 1, axis=2), out=np.zeros_like(Air_map_cycle_4), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*7-av_im_num:GasSwitch*7,:] < 1, axis=2)!=0)
    Oxy_map_cycle_1 = np.divide(Oxy_map_cycle_1, np.sum(voxels_maps_exclude[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:] < 1, axis=2), out=np.zeros_like(Oxy_map_cycle_1), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:] < 1, axis=2)!=0)
    Oxy_map_cycle_2 = np.divide(Oxy_map_cycle_2, np.sum(voxels_maps_exclude[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:] < 1, axis=2), out=np.zeros_like(Oxy_map_cycle_2), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:] < 1, axis=2)!=0)
    Oxy_map_cycle_3 = np.divide(Oxy_map_cycle_3, np.sum(voxels_maps_exclude[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:] < 1, axis=2), out=np.zeros_like(Oxy_map_cycle_3), where=np.sum(voxels_maps_exclude[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:] < 1, axis=2)!=0)
    
    # Return maps
    return Air_map_cycle_1, Air_map_cycle_2, Air_map_cycle_3, Air_map_cycle_4, \
        Oxy_map_cycle_1, Oxy_map_cycle_2, Oxy_map_cycle_3


def calc_PSE_maps_metrics_airCycle1(av_im_num, \
    component_sort_metric_list, GasSwitch, \
    X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps):
    """ 
    Calculating PSE image for each slice of an input dynamic series of image 
    maps.
    With respect to the baseline image of air during the entire first air gas
    period (0:GasSwitch). Using function PSE_map_airCycle1 instead of Sarah_ICA funciton.
    *For multiple reconstructed OE map versions*
    *For MULTIPLE metrics?*

    Arguments:
    av_in_num = average number of images to average over when calculating
    the mean air and oxy images for the PSE, taken for each cycle.
    component_sort_metric_list = list of the metric names being investigated.
    The PSE will be calculated for each reconstructed dynamic series for
    each metric.
    GasSwitch = dynamic image number at which the gases were switched.
    X_recon_OE_maps = maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_maps = unMean maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_unSc_maps = unMean and * Scaling factor maps of the 
    reconstructed component over the dynamic series. Shape 
    (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
    Returns:
    PSE_image_map = PSE image maps calculated for each slice and for each metric 
    being investigated. Shape (vox, vox, NumSlices, len(component_sort_metric_list)).
    ** This will be abut zero, for bwr plotting.
    PSE_image_map_unMean = unMean PSE image maps calculated for each slice and for each metric 
    being investigated. Shape (vox, vox, NumSlices, len(component_sort_metric_list)).
    PSE_image_map_unMean_unSc = unMean and unscaling PSE image maps calculated for 
    each slice and for each metric being investigated. Shape 
    (vox, vox, NumSlices, len(component_sort_metric_list)).
    **This will have the same magnitude/be comparable to the MRI SI data.
    ** AND ALSO RETURNS**
    PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3,
    PSE_image_unMean_cycle1, PSE_image_unMean_cycle2, PSE_image_unMean_cycle3,
    PSE_image_unMean_unSc_cycle1, PSE_image_unMean_unSc_cycle2, PSE_image_unMean_unSc_cycle3
    These are the PSE maps for each cycle of oxygen with respect to the baseline mean air 
    (over the entire first air gas period), separately calculated for each cycle.

    """
    # Create empty arrays to store the PSE images in.
    PSE_image_map = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))
    PSE_image_map_unMean = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))
    PSE_image_map_unMean_unSc = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))

    # Loop over the different metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Call the PSE_map_airCycle1() function to calculate the PSE map, and pass the number
        # of images to average over when calculating the mean air and oxy images.
        # Then store the PSE map for each metric.
        PercentageChange_image, PSE_image_map_single, Difference_image, Air_map_cycle_1, Mean_Oxy_map, \
            PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3 = \
            PSE_map_airCycle1(X_recon_OE_maps[:,:,:,:,j], GasSwitch, av_im_num)
        PSE_image_map[:,:,:,j] = PSE_image_map_single
        
        PercentageChange_image, PSE_image_map_unMean_single, Difference_image, Air_map_cycle_1, Mean_Oxy_map, \
            PSE_image_unMean_cycle1, PSE_image_unMean_cycle2, PSE_image_unMean_cycle3 = \
            PSE_map_airCycle1(X_recon_OE_unMean_maps[:,:,:,:,j], GasSwitch, av_im_num)
        PSE_image_map_unMean[:,:,:,j] = PSE_image_map_unMean_single
        
        PercentageChange_image, PSE_image_map_unMean_unSc_single, Difference_image, Air_map_cycle_1, Mean_Oxy_map, \
            PSE_image_unMean_unSc_cycle1, PSE_image_unMean_unSc_cycle2, PSE_image_unMean_unSc_cycle3 = \
            PSE_map_airCycle1(X_recon_OE_unMean_unSc_maps[:,:,:,:,j], GasSwitch, av_im_num)
        PSE_image_map_unMean_unSc[:,:,:,j] = PSE_image_map_unMean_unSc_single

    # Return PSE map calculations.
    return PSE_image_map, PSE_image_map_unMean, PSE_image_map_unMean_unSc, \
        PSE_image_cycle1, PSE_image_cycle2, PSE_image_cycle3, \
        PSE_image_unMean_cycle1, PSE_image_unMean_cycle2, PSE_image_unMean_cycle3, \
        PSE_image_unMean_unSc_cycle1, PSE_image_unMean_unSc_cycle2, PSE_image_unMean_unSc_cycle3


def Calc_Maps_airCycle1(param_map, \
    dyn_1, dyn_2):
    """
    Function to calculate mean map between two supplied dynamic 
    timepoints (e.g. could be used to calculate the mean air 
    baseline maps if dyn_1 = 0 and dyn_2 = GasSwitch).

    Arguments:
    param_map = dynamic parameter map to calculate the mean over dynamic 
    values, with shape (vox,vox,NumDyn,NumSlices).
    dyn_1 = dynamic timepoint to start the mean calculation over.
    dyn_2 = dynamic timepoint to end the mean calculation over.
    dyn_ values are the dynamics numbers counting from 1, need to -1
    to convert to python indexing.

    Returns:
    mean_map = map of the mean image over the dynamic timepoints specified,
    with shape (vox,vox,NumSlices).

    """
    # Calculate the mean map.
    mean_map = np.mean(param_map[:,:,dyn_1-1:dyn_2-1+1,:], axis=2)

    return mean_map


def Calc_Maps_airCycle1_excludeVoxels(param_map, voxels_maps_exclude, \
    dyn_1, dyn_2):
    """
    Function to calculate mean map between two supplied dynamic 
    timepoints (e.g. could be used to calculate the mean air 
    baseline maps if dyn_1 = 0 and dyn_2 = GasSwitch).
    **AND EXCLUDING VOXELS e.g. not fitted to**

    Arguments:
    param_map = dynamic parameter map to calculate the mean over dynamic 
    values, with shape (vox,vox,NumDyn,NumSlices).
    voxels_maps_exclude = Boolean array of voxels that the fitting is not performed on,
    in the form of the input data (representing both echoes in a single array dimension), with 
    shape (vox,vox,NumDyn,NumSlices).
    dyn_1 = dynamic timepoint to start the mean calculation over.
    dyn_2 = dynamic timepoint to end the mean calculation over.
    dyn_ values are the dynamics numbers counting from 1, need to -1
    to convert to python indexing.

    Returns:
    mean_map = map of the mean image over the dynamic timepoints specified,
    with shape (vox,vox,NumSlices).

    """
    # Calculate the mean map.
    mean_map = np.sum(param_map[:,:,dyn_1-1:dyn_2-1+1,:], axis=2)
    # For mean, take into account the number of voxels present (and/or not present
    # due to lack of fitting).
    voxels_maps_exclude = voxels_maps_exclude.astype('float64')
    mean_map = np.divide(mean_map, np.sum(voxels_maps_exclude[:,:,dyn_1-1:dyn_2,:] < 1, axis=2), out=np.zeros_like(mean_map), where=np.sum(voxels_maps_exclude[:,:,dyn_1-1:dyn_2,:] < 1, axis=2)!=0)

    return mean_map


def Calc_MapsStats_airCycle1(param_map, baseline_map):
    """
    Function to calculate the delta map and PSE map between two 
    supplied dynamic timepoints (e.g. could be used to calculate the
    mean air baseline maps if dyn_1 = 0 and dyn_2 = GasSwitch).

    Arguments:
    param_map = dynamic parameter map to calculate the mean over dynamic 
    values, with shape (vox,vox,NumDyn,NumSlices).
    baseline_map = map of the baseline values to use when calculating
    the PSE and delta values, with shape (vox,vox,NumSlices).
    Returns:
    mean_PSE_map = map of the mean PSE image over the dynamic timepoints
    specified, with shape (vox,vox,NumSlices). 
    mean_delta_map = map of the mean delta/subtraction image over the dynamic
    timepoints specified, with shape (vox,vox,NumSlices). 

    """
    # Calculate the PSE and delta maps.
    # PSE_image = np.multiply(np.divide((np.param_map[:,:,dyn_1-1:dyn_2-1+1,:] - baseline_map), baseline_map, out=np.zeros_like(baseline_map), where=baseline_map!=0), 100)
    # delta_image = param_map[:,:,dyn_1-1:dyn_2-1+1,:] - baseline_map
    # Remove dyn parts
    PSE_image = np.multiply(np.divide((param_map - baseline_map), baseline_map, out=np.zeros_like(baseline_map), where=baseline_map!=0), 100)
    delta_image = param_map - baseline_map
    return PSE_image, delta_image




def calc_PSE_timeseries_metrics(X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps, \
    component_sort_metric_list):
    """ 
    Calculating the dynamic PSE timeseries for each slice of different metrics.
    ***BUT only using last av_im_num for mean air image of cycle 1.
    see later function for this calculated using the entire first air cycle --> calc_PSE_timeseries_metrics_wrtCycle1.
    *For multiple reconstructed OE map versions*
    *For MULTIPLE metrics?*

    Arguments:
    X_recon_OE_maps = maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_maps = unMean maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_unSc_maps = unMean and * Scaling factor maps of the 
    reconstructed component over the dynamic series. Shape 
    (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
    component_sort_metric_list = list of the metric names being investigated.
    Returns:
    PSE_timeseries_X_recon_OE_maps = dynamic PSE maps for each slice and for
    each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_maps = unMean dynamic PSE maps for each 
    slice and for each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_unSc_maps = unMean and * scaling 
    dynamic PSE maps for each slice and for each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.

    """
    # Calculate PSE time series as a PSE dynamic image series (within 
    # the cardiac masks due to reconstructed data from collapsed ICA outputs).

    # Empty arrays to store the dynamic PSE maps.
    PSE_timeseries_X_recon_OE_maps = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], \
        np.shape(X_recon_OE_maps)[2], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))
    PSE_timeseries_X_recon_OE_unMean_maps = np.zeros((np.shape(X_recon_OE_unMean_maps)[0], np.shape(X_recon_OE_unMean_maps)[1], \
        np.shape(X_recon_OE_unMean_maps)[2], np.shape(X_recon_OE_unMean_maps)[3], len(component_sort_metric_list)))
    PSE_timeseries_X_recon_OE_unMean_unSc_maps = np.zeros((np.shape(X_recon_OE_unMean_unSc_maps)[0], np.shape(X_recon_OE_unMean_unSc_maps)[1], \
        np.shape(X_recon_OE_unMean_unSc_maps)[2], np.shape(X_recon_OE_unMean_unSc_maps)[3], len(component_sort_metric_list)))

    # Loop over the metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Use functions_Calculate_stats.calc_PSE_timeseries() to calculate the PSE maps
        # over the dynamic series.
        # Then, store only the second output argument which calculated the PSE using
        # a the baseline of the mean of the air only image from the *first* cycle, but
        # these are only the last five images from cycle 1, not the mean over the entire
        # first cycle of air.
        PSE_timeseries2, PSE_timeseries = functions_Calculate_stats.calc_PSE_timeseries(X_recon_OE_maps[:,:,:,:,j])
        PSE_timeseries_X_recon_OE_maps[:,:,:,:,j] = PSE_timeseries
        PSE_timeseries2, PSE_timeseries = functions_Calculate_stats.calc_PSE_timeseries(X_recon_OE_unMean_maps[:,:,:,:,j])
        PSE_timeseries_X_recon_OE_unMean_maps[:,:,:,:,j] = PSE_timeseries
        PSE_timeseries2, PSE_timeseries = functions_Calculate_stats.calc_PSE_timeseries(X_recon_OE_unMean_unSc_maps[:,:,:,:,j])
        PSE_timeseries_X_recon_OE_unMean_unSc_maps[:,:,:,:,j] = PSE_timeseries

    # Return dynamic PSE series.
    return PSE_timeseries_X_recon_OE_maps, PSE_timeseries_X_recon_OE_unMean_maps, PSE_timeseries_X_recon_OE_unMean_unSc_maps


def calc_PSE_timeseries_metrics_wrtCycle1(X_recon_OE_maps, X_recon_OE_unMean_maps, X_recon_OE_unMean_unSc_maps, \
    component_sort_metric_list, GasSwitch):
    """ 
    Calculating the dynamic PSE timeseries for each slice of different metrics.
    *Using the mean air image from the first cycle only - this is the 
    mean air image over the ENTIRE first air cycle (all GasSwitch number
    of images averaged).
    *For multiple reconstructed OE map versions*
    *For MULTIPLE metrics?*

    Arguments:
    X_recon_OE_maps = maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_maps = unMean maps of the reconstructed component over the dynamic
    series. Shape (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    X_recon_OE_unMean_unSc_maps = unMean and * Scaling factor maps of the 
    reconstructed component over the dynamic series. Shape 
    (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
    component_sort_metric_list = list of the metric names being investigated.
    Returns:
    PSE_timeseries_X_recon_OE_maps = dynamic PSE maps for each slice and for
    each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_maps = unMean dynamic PSE maps for each 
    slice and for each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_unSc_maps = unMean and * scaling 
    dynamic PSE maps for each slice and for each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.

    """
    # Calculate PSE time series as a PSE dynamic image series (within 
    # the cardiac masks due to reconstructed data from collapsed ICA outputs).

    # Empty arrays to store the dynamic PSE maps.
    PSE_timeseries_X_recon_OE_maps = np.zeros((np.shape(X_recon_OE_maps)[0], np.shape(X_recon_OE_maps)[1], \
        np.shape(X_recon_OE_maps)[2], np.shape(X_recon_OE_maps)[3], len(component_sort_metric_list)))
    PSE_timeseries_X_recon_OE_unMean_maps = np.zeros((np.shape(X_recon_OE_unMean_maps)[0], np.shape(X_recon_OE_unMean_maps)[1], \
        np.shape(X_recon_OE_unMean_maps)[2], np.shape(X_recon_OE_unMean_maps)[3], len(component_sort_metric_list)))
    PSE_timeseries_X_recon_OE_unMean_unSc_maps = np.zeros((np.shape(X_recon_OE_unMean_unSc_maps)[0], np.shape(X_recon_OE_unMean_unSc_maps)[1], \
        np.shape(X_recon_OE_unMean_unSc_maps)[2], np.shape(X_recon_OE_unMean_unSc_maps)[3], len(component_sort_metric_list)))

    # Loop over the metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        # Use functions_Calculate_stats.calc_PSE_timeseries_wrtCycle1() to calculate the PSE maps
        # over the dynamic series.
        PSE_timeseries = functions_Calculate_stats.calc_PSE_timeseries_wrtCycle1(X_recon_OE_maps[:,:,:,:,j], GasSwitch)
        PSE_timeseries_X_recon_OE_maps[:,:,:,:,j] = PSE_timeseries
        PSE_timeseries = functions_Calculate_stats.calc_PSE_timeseries_wrtCycle1(X_recon_OE_unMean_maps[:,:,:,:,j], GasSwitch)
        PSE_timeseries_X_recon_OE_unMean_maps[:,:,:,:,j] = PSE_timeseries
        PSE_timeseries = functions_Calculate_stats.calc_PSE_timeseries_wrtCycle1(X_recon_OE_unMean_unSc_maps[:,:,:,:,j], GasSwitch)
        PSE_timeseries_X_recon_OE_unMean_unSc_maps[:,:,:,:,j] = PSE_timeseries

    # Return dynamic PSE series.
    return PSE_timeseries_X_recon_OE_maps, PSE_timeseries_X_recon_OE_unMean_maps, PSE_timeseries_X_recon_OE_unMean_unSc_maps


def calc_stats_slices(PSE_image_data):
    """
    Calculate a variety of stats over each slice and over all slices 
    (for parameter contained within the input array, e.g. image SI or PSE).
    Stats: mean, median, min, max, range, stdev.

    Arguments:
    PSE_image_data = array of image data of shape (vox, vox, NumSlices)
    for which the statistics are being calculated of.
    Returns:
    stats_ALLSlice = array of the different stats measures calculated over
    all slices. With shape (numStats,).
    stats_eachSlice = array of the different stats measures calculated for
    each slice. With shape (NumSlices,numStats).
    
    """
    # Number of statistics parameters to be calculated. Assume 6: mean, median, 
    # min, max, range, stedev.
    stats_metrics = 6 
    # Create empty arrays to store the stats for each slice and for over all slices.
    stats_ALLSlice = np.zeros((stats_metrics))
    stats_eachSlice = np.zeros((np.shape(PSE_image_data)[2], stats_metrics))


    # # Stats over *ALL* slices:
    # Collapse all dimensions to enable a calculation over all values.
    array_ALL_slice = np.reshape(PSE_image_data, \
        (np.shape(PSE_image_data)[0]*np.shape(PSE_image_data)[1]*np.shape(PSE_image_data)[2]))
    # Calculate the stats measures over non-zero values (i.e. within the cardiac or lung mask).
    array_ALL_slice = array_ALL_slice[array_ALL_slice != 0]

    # Calculate statistic measures and store.
    stats_ALLSlice[0] = np.mean(array_ALL_slice)
    stats_ALLSlice[1] = np.median(array_ALL_slice)
    stats_ALLSlice[2] = np.min(array_ALL_slice)
    stats_ALLSlice[3] = np.max(array_ALL_slice)
    stats_ALLSlice[4] = stats_ALLSlice[3] - stats_ALLSlice[2]
    stats_ALLSlice[5] = np.std(array_ALL_slice)


    # # Stats over *EACH* slice:
    # Collapse first and second (voxel) dimensions to enable calculations 
    # over each slice.
    array_each_slice = np.reshape(PSE_image_data, \
        (np.shape(PSE_image_data)[0]*np.shape(PSE_image_data)[1],np.shape(PSE_image_data)[2]))

    # Loop over all slices and calculate the stats values separately for each slice.
    for k in range(np.shape(PSE_image_data)[2]):
        # Select a single slice to calculate the stats measures over.
        array_single_slice = array_each_slice[:,k]
        # Calculate the stats measures over non-zero values (i.e. within the cardiac or lung mask).
        array_single_slice = array_single_slice[array_single_slice != 0]
        # Calculate statistics measures and store.
        stats_eachSlice[k,0] = np.mean(array_single_slice)
        stats_eachSlice[k,1] = np.median(array_single_slice)
        stats_eachSlice[k,2] = np.min(array_single_slice)
        stats_eachSlice[k,3] = np.max(array_single_slice)
        stats_eachSlice[k,4] = stats_eachSlice[k,3] - stats_eachSlice[k,2]
        stats_eachSlice[k,5] = np.std(array_single_slice)

    # Return statistics measures.
    return stats_ALLSlice, stats_eachSlice


def calc_stats_slices_metrics(PSE_image_map_Recon, \
    component_sort_metric_list):
    """
    Calculate a variety of stats over each slice and over all slices 
    for different metrics.
    (Calcualted for parameter contained within the input array, e.g. image SI or PSE).
    Stats: mean, median, min, max, range, stdev.
    #For multiple component ordering metrics.*

    Arguments:
    PSE_image_data_Recon = array of image data of shape 
    (vox, vox, NumSlices, len(component_sort_metric_list))
    for which the statistics are being calculated of.
    Returns:
    stats_ALLSlice_metrics = array of the different stats measures calculated over
    all slices. With shape (numStats, len(component_sort_metric_list)).
    stats_eachSlice_metrics = array of the different stats measures calculated for
    each slice. With shape (NumSlices, numStats, len(component_sort_metric_list)).

    """
    # Number of statistics parameters to be calculated. Assume 6: mean, median, 
    # min, max, range, stedev.
    stats_metrics = 6 
    # Create empty arrays to store the stats for each slice and for over all slices.
    stats_ALLSlice_metrics = np.zeros((stats_metrics, len(component_sort_metric_list)))
    stats_eachSlice_metrics = np.zeros((np.shape(PSE_image_map_Recon)[2], stats_metrics, len(component_sort_metric_list)))

    # Loop over the metrics being investigated and call calc_stats_slices function.
    for j in range(len(component_sort_metric_list)):
        stats_ALLSlice, stats_eachSlice = calc_stats_slices(PSE_image_map_Recon[:,:,:,j])
        stats_ALLSlice_metrics[:,j] = stats_ALLSlice
        stats_eachSlice_metrics[:,:,j] = stats_eachSlice
    
    # Return stats measures for all metrics being investigated.
    return stats_ALLSlice_metrics, stats_eachSlice_metrics


def calc_stats_slices_IQR(PSE_image_data, stats_list):
    """
    Calculate a variety of stats over each slice and over all slices 
    (for parameter contained within the input array, e.g. image SI or PSE).
    Stats: mean, median, (min - max) = range, IQR, stdev.

    Arguments:
    PSE_image_data = array of image data of shape (vox, vox, NumSlices)
    for which the statistics are being calculated of.
    Returns:
    stats_ALLSlice = array of the different stats measures calculated over
    all slices. With shape (numStats,).
    stats_eachSlice = array of the different stats measures calculated for
    each slice. With shape (NumSlices,numStats).
    
    """
    stats_ALLSlice = np.zeros((len(stats_list)))
    stats_eachSlice = np.zeros((np.shape(PSE_image_data)[2], len(stats_list)))


    # # Stats over *ALL* slices:
    # Collapse all dimensions to enable a calculation over all values.
    array_ALL_slice = np.reshape(PSE_image_data, \
        (np.shape(PSE_image_data)[0]*np.shape(PSE_image_data)[1]*np.shape(PSE_image_data)[2]))
    # Calculate the stats measures over non-zero values (i.e. within the cardiac or lung mask).
    array_ALL_slice = array_ALL_slice[array_ALL_slice != 0]

    # Calculate statistic measures and store.
    stats_ALLSlice[0] = np.mean(array_ALL_slice)
    stats_ALLSlice[1] = np.median(array_ALL_slice)
    stats_ALLSlice[2] = np.max(array_ALL_slice) - np.min(array_ALL_slice)
    # IQR and percentile calculation
    q3, q1 = np.percentile(array_ALL_slice, [75 ,25])
    stats_ALLSlice[3] = q3 - q1
    stats_ALLSlice[4] = np.std(array_ALL_slice)


    # # Stats over *EACH* slice:
    # Collapse first and second (voxel) dimensions to enable calculations 
    # over each slice.
    array_each_slice = np.reshape(PSE_image_data, \
        (np.shape(PSE_image_data)[0]*np.shape(PSE_image_data)[1],np.shape(PSE_image_data)[2]))
    #
    # Loop over all slices and calculate the stats values separately for each slice.
    for k in range(np.shape(PSE_image_data)[2]):
        # Select a single slice to calculate the stats measures over.
        array_single_slice = array_each_slice[:,k]
        # Calculate the stats measures over non-zero values (i.e. within the cardiac or lung mask).
        array_single_slice = array_single_slice[array_single_slice != 0]
        # Calculate statistics measures and store.
        stats_eachSlice[k,0] = np.mean(array_single_slice)
        stats_eachSlice[k,1] = np.median(array_single_slice)
        stats_eachSlice[k,2] = np.max(array_single_slice) - np.min(array_single_slice)
        q3, q1 = np.percentile(array_single_slice, [75 ,25])
        stats_eachSlice[k,3] = q3 - q1
        stats_eachSlice[k,4] = np.std(array_single_slice)

    return stats_ALLSlice, stats_eachSlice


def calc_stats_slices_IQR_masked(PSE_image_data, stats_list, image_mask):
    """
    2022/08/05 - masked stats slices.***********

    Calculate a variety of stats over each slice and over all slices 
    (for parameter contained within the input array, e.g. image SI or PSE).
    Stats: mean, median, min, max, range, stdev.
    **Here providing image mask - for e.g. delta may have zero values**

    Arguments:
    PSE_image_data = array of image data of shape (vox, vox, NumSlices)
    for which the statistics are being calculated of.
    image_mask = mask applied to image data with shape (vox,vox,NumSlices).
    Returns:
    stats_ALLSlice = array of the different stats measures calculated over
    all slices. With shape (numStats,).
    stats_eachSlice = array of the different stats measures calculated for
    each slice. With shape (NumSlices,numStats).
    
    """
    stats_ALLSlice = np.zeros((len(stats_list)))
    stats_eachSlice = np.zeros((np.shape(PSE_image_data)[2], len(stats_list)))

    # Stats over *ALL SLICES* - extract non-mask voxels for each slice and 
    # append to an array.
    # Stats over *EACH SLICE* - similar to above, but calculate within the
    # extraction loop.

    # Empty array to store masked input image in:
    array_ALL_slice = np.array(([]))

    # Loop over each slice.
    for j in range(np.shape(PSE_image_data)[2]):
        PSE_image_data_slice = PSE_image_data[:,:,j]
        nonZ_voxel_values = PSE_image_data_slice[image_mask[:,:,j] != 0]
        #
        # Append to the array to use for calculating the stats over *ALL SLICES*.
        array_ALL_slice = np.append(array_ALL_slice, nonZ_voxel_values)
        # 
        # Calculate stats for this slice and add to *EACH SLICE* calculations.
        stats_eachSlice[j,0] = np.mean(nonZ_voxel_values)
        stats_eachSlice[j,1] = np.median(nonZ_voxel_values)
        stats_eachSlice[j,2] = np.max(nonZ_voxel_values) - np.min(nonZ_voxel_values)
        q3, q1 = np.percentile(nonZ_voxel_values, [75 ,25])
        stats_eachSlice[j,3] = q3 - q1
        stats_eachSlice[j,4] = np.std(nonZ_voxel_values)
        #
        #
        del PSE_image_data_slice, nonZ_voxel_values


    # Calculate statistic measures and store.
    stats_ALLSlice[0] = np.mean(array_ALL_slice)
    stats_ALLSlice[1] = np.median(array_ALL_slice)
    stats_ALLSlice[2] = np.max(array_ALL_slice) - np.min(array_ALL_slice)
    # IQR and percentile calculation
    q3, q1 = np.percentile(array_ALL_slice, [75 ,25])
    stats_ALLSlice[3] = q3 - q1
    stats_ALLSlice[4] = np.std(array_ALL_slice)


    return stats_ALLSlice, stats_eachSlice


def calc_stats_within_mask(PSE_timeseries_X_recon_OE_maps, PSE_timeseries_X_recon_OE_unMean_maps, PSE_timeseries_X_recon_OE_unMean_unSc_maps,\
    component_sort_metric_list, NumDyn, NumSlices, nonZ_masks_slices, \
    masks_reg_data_lung=0):
    """
    Calculate the different metrics being investigated within a specific mask, the mask of which can be applied.
    For a dynamic series of maps.
    *For different OE reconstruction methods*
    
    Arguments:
    PSE_timeseries_X_recon_OE_maps = dynamic PSE maps for each slice and for
    each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_maps = unMean dynamic PSE maps for each 
    slice and for each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_unSc_maps = unMean and * scaling 
    dynamic PSE maps for each slice and for each metric being investigated.
    Shape of (vox, vox, NumDyn, NumSlices, len(component_sort_metric_list)).
    This will have the same magnitude/be comparable to the MRI SI data.
    component_sort_metric_list = list of the metric names being investigated.
    NumDyn = Number of dynamic images acquired.
    NumSlices = Number of image slices acquired.
    nonZ_masks_slices = number of voxels present within the mask per slice.
    masks_reg_data_lung = whether to apply lung masks to the data to
    calculate the mean PSE only within the lung mask. = 0 if No to apply masks,
    = the lung mask array if Yes to apply masks.
    Returns:
    Calculation of the PSE for each slice at every timepoint for all metrics 
    being investigated. Depending on the input masks_reg_data_lung argument,
    the calculation may be for within the lung mask. All have shape
    (NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice = calculation
    for reconstructed data.
    PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice = calculation
    for reconstructed data that has unMean.
    PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice = calculation
    for reconstructed data that has unMean and * scaling.
    This will have the same magnitude/be comparable to the MRI SI data.
    
    """
    # If masks_reg_data_lung != 0 (use sum as may potentially be an array),
    # apply the lung masks to the data.
    if np.sum(masks_reg_data_lung) != 0:
        # Create arrays to store the masked data in.
        PSE_timeseries_X_recon_OE_maps_lungMask = np.zeros((np.shape(masks_reg_data_lung)[0], np.shape(masks_reg_data_lung)[1], \
            NumDyn, np.shape(masks_reg_data_lung)[2], len(component_sort_metric_list)))
        PSE_timeseries_X_recon_OE_unMean_maps_lungMask = np.zeros((np.shape(masks_reg_data_lung)[0], np.shape(masks_reg_data_lung)[1], \
            NumDyn, np.shape(masks_reg_data_lung)[2], len(component_sort_metric_list)))
        PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask = np.zeros((np.shape(masks_reg_data_lung)[0], np.shape(masks_reg_data_lung)[1], \
            NumDyn, np.shape(masks_reg_data_lung)[2], len(component_sort_metric_list)))
        # Apply the lung masks for all metrics.
        for j in range(len(component_sort_metric_list)):
            PSE_timeseries_X_recon_OE_maps_lungMask[:,:,:,:,j] = functions_Calculate_stats.apply_mask_maps(PSE_timeseries_X_recon_OE_maps[:,:,:,:,j], masks_reg_data_lung)
            PSE_timeseries_X_recon_OE_unMean_maps_lungMask[:,:,:,:,j] = functions_Calculate_stats.apply_mask_maps(PSE_timeseries_X_recon_OE_unMean_maps[:,:,:,:,j], masks_reg_data_lung)
            PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask[:,:,:,:,j] = functions_Calculate_stats.apply_mask_maps(PSE_timeseries_X_recon_OE_unMean_unSc_maps[:,:,:,:,j], masks_reg_data_lung)

        # For calculating the mean PSE in each slice, need to know the number of masked 
        # voxels within the lung masks (for each slice separately), given by 
        # [1,:] of nonZ_masks_slices.
        nonZ_lung_slices = nonZ_masks_slices[1,:]
    else:
        # If not applying the lung mask, calculating PSE of data which has already had
        # the cardiac masks applied.
        PSE_timeseries_X_recon_OE_maps_lungMask = np.array(PSE_timeseries_X_recon_OE_maps)
        PSE_timeseries_X_recon_OE_unMean_maps_lungMask = np.array(PSE_timeseries_X_recon_OE_unMean_maps)
        PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask = np.array(PSE_timeseries_X_recon_OE_unMean_unSc_maps)
        # For calculating the mean PSE in each slice, need to know the number of masked 
        # voxels within the cardiac masks (for each slice separately), given by 
        # [0,:] of nonZ_masks_slices.
        nonZ_lung_slices = nonZ_masks_slices[0,:]

    # Empty arrays to store the mean PSE value over each slice. These will be calculated (mean for each slice) 
    # for each dynamic image in the series and for all metrics being investigated.
    PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice = np.zeros((NumDyn, NumSlices, len(component_sort_metric_list)))
    PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice = np.zeros((NumDyn, NumSlices, len(component_sort_metric_list)))
    PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice = np.zeros((NumDyn, NumSlices, len(component_sort_metric_list)))

    # Loop over the metrics to be investigated.
    for j in range(len(component_sort_metric_list)):
        # Calculate the mean PSE in each slice by summing along both voxel dimensions and diving
        # by the number of mask voxels (nonZ_lung_slices).
        # Then store the PSE timeseries in all slices for the particular metric.
        PSE_calc_in_progress = \
            functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(\
                PSE_timeseries_X_recon_OE_maps_lungMask[:,:,:,:,j], nonZ_lung_slices)
        PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice[:,:,j] = PSE_calc_in_progress
        # Calculate the mean PSE in each slice by summing along both voxel dimensions and diving
        # by the number of mask voxels (nonZ_lung_slices).
        # Then store the PSE timeseries in all slices for the particular metric.
        PSE_calc_in_progress = \
            functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(\
                PSE_timeseries_X_recon_OE_unMean_maps_lungMask[:,:,:,:,j], nonZ_lung_slices)
        PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice[:,:,j] = PSE_calc_in_progress
        # Calculate the mean PSE in each slice by summing along both voxel dimensions and diving
        # by the number of mask voxels (nonZ_lung_slices).
        # Then store the PSE timeseries in all slices for the particular metric.
        PSE_calc_in_progress = \
            functions_Calculate_stats.timeCalc_perSlice_fromMaps_nonZSupplied(\
                PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask[:,:,:,:,j], nonZ_lung_slices)
        PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,:,j] = PSE_calc_in_progress

    # Return the PSE calculations over each slice for all timeseries points and all metrics
    # being investigated.
    return PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice, PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice


def stats_perSlice_andOverall_calc(mean_timeseries_in_each_slice):
    """ 
    Better to calculate stats from the PSE image for entire timeseries (found
    using mean air and oxy from each cycle) rather than PSE calculations
    from the timeseries.

    Arguments:
    mean_timeseries_in_each_slice = array of values for each point in a timeseries
    for each slice, for each metric. Shape of 
    (NumDyn, NumSlices, len(component_sort_metric_list)).
    Returns:
    mean_timeseries_overTime_allSlices = shape (NumSlices, len(component_sort_metric_list)).
    mean_overall = shape (len(component_sort_metric_list),).

    """
    # Calculate the mean over all timepoints per slice.
    mean_timeseries_overTime_allSlices = np.mean(mean_timeseries_in_each_slice, axis=0)
    # Calculate the mean over all timepoints and all slices.
    mean_overall = np.mean(mean_timeseries_overTime_allSlices, axis=0)
    return mean_timeseries_overTime_allSlices, mean_overall


def stats_perSlice_andOverall_calc_Recon(PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice, \
    PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice, \
    PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice):
    """
    Calculate stats per slice over time and over all slices with time.
    For different recon options (+ mean and * scaling etc).
    *For multiple OE reconstruction methods*

    Again, better to calculate stats from the PSE image for entire timeseries (found
    using mean air and oxy from each cycle) rather than PSE calculations
    from the timeseries.

    Arguments:
    PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice = array of values for 
    each point in a timeseries for each slice, for each metric. 
    Shape of (NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice = unMean array 
    of values for each point in a timeseries for each slice, for each metric. 
    Shape of (NumDyn, NumSlices, len(component_sort_metric_list)).
    PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice = 
    unMean * scaling array of values for each point in a timeseries 
    for each slice, for each metric. Shape of 
    (NumDyn, NumSlices, len(component_sort_metric_list)).
    Returns:
    mean_timeseries_overTime_allSlices = shape (NumSlices, len(component_sort_metric_list)).
    mean_overall = shape (len(component_sort_metric_list),).
    But for each input unMean and unScaled:
    mean_PSE_metrics_allSlices, mean_PSE_metrics_overall,
    mean_PSE_metrics_unMean_allSlices, mean_PSE_metrics_unMean_overall,
    mean_PSE_metrics_unMean_unSc_allSlices, mean_PSE_metrics_unMean_unSc_overall

    """
    # Calculate the mean over all timepoints per slice, and all slices (for each metric).
    mean_PSE_metrics_allSlices, mean_PSE_metrics_overall = \
        stats_perSlice_andOverall_calc(PSE_timeseries_X_recon_OE_maps_lungMask_MeanEachSlice)
    mean_PSE_metrics_unMean_allSlices, mean_PSE_metrics_unMean_overall = \
        stats_perSlice_andOverall_calc(PSE_timeseries_X_recon_OE_unMean_maps_lungMask_MeanEachSlice)
    mean_PSE_metrics_unMean_unSc_allSlices, mean_PSE_metrics_unMean_unSc_overall = \
        stats_perSlice_andOverall_calc(PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice)
    
    # Return mean values.
    return mean_PSE_metrics_allSlices, mean_PSE_metrics_overall, \
        mean_PSE_metrics_unMean_allSlices, mean_PSE_metrics_unMean_overall, \
        mean_PSE_metrics_unMean_unSc_allSlices, mean_PSE_metrics_unMean_unSc_overall



# Plotting

def plot_componentNum_vs_metricVal(sorted_array, component_sort_metric_list, \
    saving_details, showplot, dpi_val=300):
    """
    Plot component number vs metric value to observe at which component number
    the lowest metric value is reached.

    Arguments:
    sorted_array = sorted array of increasing components numbers (first column)
    and their corresponding metric values, with the metric running over the third
    dimension. Of shape 
    (Number of components, 1+num metrics stored, len(component_sort_metric_list)).
    component_sort_metric_list = list of the metrics being investigated.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    Returns:
    None, plots and optionally saves the figures(s).

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # Plot components within the sorted array.
    fig1 = plt.figure(figsize=(19.20,10.80))
    ax1 = fig1.add_subplot(1, 1, 1)
    # Loop over the metrics being investigated.
    for j in range(len(component_sort_metric_list)):
        plt.plot(np.int32(np.rint(sorted_array[:,0,j])), sorted_array[:,1,j], '-x')

    # Add legend, axes titles and set axes limits.
    ax1.set_xlim(0, 75); ax1.set_xlabel('Number of ICA components'); ax1.set_ylabel('Metric value')
    ax1.legend(component_sort_metric_list)
    fig1.tight_layout()

    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]
    
    # Save figure if required.
    if saving_details[0] == 1:
        fig1.savefig(saving_details[1] + saving_details[2] + 'NumC_MetricValues' + saving_end_detail + '.png', dpi=dpi_val, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_componentNum_vs_metricVal_wrtCycle1(sorted_array, metric_chosen, RunNum_use, \
    saving_details, showplot, dpi_save=300):
    """
    Plot component number vs metric value to observe at which component number
    the lowest metric value is reached.
    ** V2 - for a single metric **
    Assuming >21 components, for a single metric investigated.

    Arguments:
    sorted_array = sorted array of increasing components numbers (first column)
    and their corresponding metric values, with the metric running over the third
    dimension. Of shape 
    (Number of components, 2, 1).
    metric_chosen = the *single* metric being investigated, in the form of a list.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    dpi_save = dpi value to use when saving. Default = 300.
    Returns:
    None, plots and optionally saves the figures(s).

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # Plot components within the sorted array.
    fig1 = plt.figure(figsize=(19.20,10.80))
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.plot(np.int32(np.rint(sorted_array[:,0,0])), sorted_array[:,1,0], '-x')

    # Add legend, axes titles and set axes limits.
    ax1.set_xlim(21-3, 75); ax1.set_xlabel('Number of ICA components'); ax1.set_ylabel('Metric value')
    ax1.set_title('Number of components vs metric value for ' + metric_chosen[0])
    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        if RunNum_use > 1:
            fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + 'NumC_MetricValues' + '_' + metric_chosen[0] + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')
        else:
            fig1.savefig(saving_details[1] + saving_details[2] + 'NumC_MetricValues' + '_' + metric_chosen[0] + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



if __name__ == "__main__":
    # ...
    a = 1