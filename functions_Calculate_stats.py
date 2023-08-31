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

# Functions for calculating stats.

import numpy as np
import Sarah_ICA
from scipy import stats
from scipy.fft import fft, ifft, fftfreq

def calc_PSE_timeseries(image_data, baseline_image=0):
    """
    Calculate the PSE for a dynamic series of images/parameter maps.

    Arguments:
    image_data = dynamic image series (or single image) (including all four slices), with
    shape (vox,vox,NumDyn,NumSlices).
    baseline_image = baseline (air-inhalation) image. Shape of (vox,vox,NumSlices). Set
    to 0 if not passed, which results Sarah_ICA.PSE_map function called to calculate
    the mean air-inhalation image.
    Returns:
    PSE_calc_series = PSE calculation over the dynamic image series. With
    shape (vox,vox,NumDyn,NumSlices). PSE calculation with respect to the
    mean air map - i.e. the last av_im_num images from each air cycle averaged.
    PSE_calc_series2 = PSE calculation over the dynamic image series. With
    shape (vox,vox,NumDyn,NumSlices). PSE calculation with respect to the
    last av_im_num images from the first air cycle.
      
    """
    # If baseline image is not passed (value of 0) calculate using the
    # Sarah_ICA.PSE_map function.
    if baseline_image == 0:
        PercentageChange_image, PSE_image, Difference_image, Air_map_cycle_1, Mean_Air_map, Mean_Oxy_map = Sarah_ICA.PSE_map(image_data, GasSwitch=60, av_im_num=5)
        del PercentageChange_image, PSE_image, Difference_image, Mean_Oxy_map
        baseline_image = np.array(Mean_Air_map)
        baseline_image2 = np.array(Air_map_cycle_1)

    # Need to reshape arrays for division (to allow broadcasting), and reshape back after.
    PSE_calc_series = np.multiply(np.divide((np.transpose(image_data, (2,0,1,3)) - baseline_image), baseline_image, out=np.zeros_like(np.transpose(image_data, (2,0,1,3))), where=baseline_image!=0), 100)
    PSE_calc_series = np.transpose(PSE_calc_series, (1,2,0,3))   
    PSE_calc_series2 = np.multiply(np.divide((np.transpose(image_data, (2,0,1,3)) - baseline_image2), baseline_image2, out=np.zeros_like(np.transpose(image_data, (2,0,1,3))), where=baseline_image2!=0), 100)
    PSE_calc_series2 = np.transpose(PSE_calc_series2, (1,2,0,3))   
    return PSE_calc_series, PSE_calc_series2


def calc_PSE_timeseries_wrtCycle1(image_data, GasSwitch, baseline_delay=0):
    """
    Calculate the PSE for a dynamic series of images/parameter maps.
    Calculating the dynamic PSE timeseries for each slice of different metrics.
    *** Using the mean air image from the first cycle only - this is the 
    MEAN AIR IMAGE over the ENTIRE first air cycle (all GasSwitch number
    of images averaged).***

    Arguments:
    image_data = dynamic image series (or single image) (including all four slices), with
    shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    PSE_calc_series = PSE calculation over the dynamic image series. With
    shape (vox,vox,NumDyn,NumSlices). PSE calculation with respect to 
    the mean of all of the images acquired during the first air cycle.
      
    """
    # Baseline air image - mean of all air images during the first air gas cycle.
    baseline_air_image = np.mean(image_data[:,:,baseline_delay:GasSwitch,:], axis=2)

    # Need to reshape arrays for division (to allow broadcasting), and reshape back after.
    PSE_calc_series = np.multiply(np.divide((np.transpose(image_data, (2,0,1,3)) - baseline_air_image), baseline_air_image, out=np.zeros_like(np.transpose(image_data, (2,0,1,3))), where=baseline_air_image!=0), 100)
    PSE_calc_series = np.transpose(PSE_calc_series, (1,2,0,3))    
    return PSE_calc_series


def calc_delta_timeseries_wrtCycle1(image_data, GasSwitch, baseline_delay=0):
    """
    Calculate the SI difference for a dynamic series of images/parameter maps.
    Calculating the dynamic deltaSI timeseries for each slice of different metrics.
    *** Using the mean air image from the first cycle only - this is the 
    MEAN AIR IMAGE over the ENTIRE first air cycle (all GasSwitch number
    of images averaged).***

    Arguments:
    image_data = dynamic image series (or single image) (including all four slices), with
    shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    deltaSI_calc_series = PSE calculation over the dynamic image series. With
    shape (vox,vox,NumDyn,NumSlices). PSE calculation with respect to 
    the mean of all of the images acquired during the first air cycle.
      
    """
    # Baseline air image - mean of all air images during the first air gas cycle.
    baseline_air_image = np.mean(image_data[:,:,baseline_delay:GasSwitch,:], axis=2)

    # Need to permute (np.transpose) arrays for subtraction, and permute back after.
    delta_image_series = np.transpose(image_data, (2,0,1,3)) - baseline_air_image
    delta_image_series = np.transpose(delta_image_series, (1,2,0,3))
    return delta_image_series


def calc_delta_timeseries_wrtCycle1_excludeVoxels(image_data, GasSwitch, voxels_maps_exclude, baseline_delay=0):
    """
    Calculate the SI difference for a dynamic series of images/parameter maps.
    Calculating the dynamic deltaSI timeseries for each slice of different metrics.
    *** Using the mean air image from the first cycle only - this is the 
    MEAN AIR IMAGE over the ENTIRE first air cycle (all GasSwitch number
    of images averaged).***
    **AND EXCLUDING VOXELS e.g. not fitted to**

    Arguments:
    image_data = dynamic image series (or single image) (including all four slices), with
    shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic image number at which the gases were switched.
    voxels_maps_exclude = Boolean array of voxels that the fitting is not performed on,
    in the form of the input data (representing both echoes in a single array dimension), with 
    shape (vox,vox,NumDyn,NumSlices).
    baseline_delay = dynamic to start baseline air cycle 1 calculation at - may
    be used if the first e.g. 10 dynamics involve a transient signal decay
    during the approach to the steady state.
    Returns:
    deltaSI_calc_series = PSE calculation over the dynamic image series. With
    shape (vox,vox,NumDyn,NumSlices). PSE calculation with respect to 
    the mean of all of the images acquired during the first air cycle.
      
    """
    # Baseline air image - mean of all air images during the first air gas cycle.
    baseline_air_image = np.sum(image_data[:,:,baseline_delay:GasSwitch,:], axis=2, dtype=float)
    # For mean, take into account the number of voxels present (and/or not present
    # due to lack of fitting).
    # voxels_maps_exclude = voxels_maps_exclude.astype('float64')
    voxels_fitted_image = np.sum(voxels_maps_exclude[:,:,baseline_delay:GasSwitch,:] < 1, axis=2, dtype=float)
    # Divide
    baseline_air_image = np.divide(baseline_air_image, voxels_fitted_image, out=np.zeros_like(voxels_fitted_image), where=voxels_fitted_image!=0)

    # Need to permute (np.transpose) arrays for subtraction, and permute back after.
    delta_image_series = np.transpose(image_data, (2,0,1,3)) - baseline_air_image
    delta_image_series = np.transpose(delta_image_series, (1,2,0,3))
    return delta_image_series


def apply_mask_maps(image_data, mask_data):
    """
    Apply masks to images.

    Arguments:
    image_data = dynamic image series (or single image) (including all four slices), with
    shape (vox,vox,NumDyn,NumSlices).
    mask_data = lung or cardiac masks, to be multiplied with the image series or single image data.
    With shape (vox,vox,NumSlices).
    Returns:
    masked_maps = Masked data, with shape (vox,vox,NumDyn,NumSlices).
      
    """
    # Need to reshape arrays for multiplication of masks (to allow broadcasting),
    # and reshape back after.
    image_data = np.transpose(image_data, (2,0,1,3))
    masked_maps = np.multiply(image_data, mask_data)
    masked_maps = np.transpose(masked_maps, (1,2,0,3))   
    return masked_maps


def apply_mask_maps_singleDyn(image_data, mask_data):
    """
    Apply masks to images.

    Arguments:
    image_data = single image (including all four slices), with
    shape (vox,vox,NumSlices).
    mask_data = lung or cardiac masks, to be multiplied with the image series or single image data.
    With shape (vox,vox,NumSlices).
    Returns:
    masked_maps = Masked data, with shape (vox,vox,NumSlices).
      
    """
    masked_maps = np.multiply(image_data, mask_data)
    return masked_maps


def timeCalc_fromMaps(masked_image_data, mask_data):
    """
    Calculate mean parameter value within masked time series data.
    This could be mean SI or mean PSE depending on the input data.
    **Dynamic parameter**

    Arguments:
    masked_image_data = masked dynamic image series (or single image), with
    shape (vox,vox,NumDyn,NumSlices).
    mask_data = lung or cardiac masks, with shape (vox,vox,NumSlices).
    Returns:
    mean_param = mean parameter within the mask as a function of time, having shape
    (NumDyn,).

    """
    # Mean parameter calculation - apply mean three times to collapse dimensions to time only.
    # But, have zeros present. Therefore, calculate sum instead of mean and divide by the number
    # of masked voxels within all four slices.
    mean_param = np.sum(masked_image_data, axis=0); mean_param = np.sum(mean_param, axis=0)
    mean_param = np.sum(mean_param, axis=1)

    # Divide by the number of non-zero mask voxels
    mean_param = np.divide(mean_param, np.sum(mask_data))

    return mean_param


def timeCalc_fromMaps_excludeVoxels(masked_image_data, mask_data, voxels_maps_exclude):
    """
    Calculate mean parameter value within masked time series data.
    This could be mean SI or mean PSE depending on the input data.
    **Dynamic parameter**
    **AND EXCLUDING VOXELS e.g. not fitted to**

    Arguments:
    masked_image_data = masked dynamic image series (or single image), with
    shape (vox,vox,NumDyn,NumSlices).
    mask_data = lung or cardiac masks, with shape (vox,vox,NumSlices).
    voxels_maps_exclude = Boolean array of voxels that the fitting is not performed on,
    in the form of the input data (representing both echoes in a single array dimension), with 
    shape (vox,vox,NumDyn,NumSlices).
    Returns:
    mean_param = mean parameter within the mask as a function of time, having shape
    (NumDyn,).

    """
    # Mean parameter calculation - apply mean three times to collapse dimensions to time only.
    # But, have zeros present. Therefore, calculate sum instead of mean and divide by the number
    # of masked voxels within all four slices.
    mean_param = np.sum(masked_image_data, axis=0, dtype=float); mean_param = np.sum(mean_param, axis=0)
    mean_param = np.sum(mean_param, axis=1)

    # Number of voxels present in fitting per dynamic:
    num_voxels_dynamic_series = np.sum(voxels_maps_exclude, axis=0); num_voxels_dynamic_series = np.sum(num_voxels_dynamic_series, axis=0)
    num_voxels_dynamic_series = np.sum(num_voxels_dynamic_series, axis=1)

    # Number of voxels present in mask across all slices:
    mask_nonZ = np.int32(np.rint(np.sum(mask_data)))
    # Broadcast across number of dynamics and multipply by mask_nonZ.
    mask_nonZ_array = np.multiply(np.ones((np.shape(mean_param))), mask_nonZ)
    # Subtract num_voxels_dynamic_series for voxels not fitted to.
    mean_divide_by = mask_nonZ_array - num_voxels_dynamic_series

    # Divide by the number of non-zero mask voxels
    mean_param = np.divide(mean_param, mean_divide_by, out=np.zeros_like(mean_param), where=mean_divide_by!=0)

    return mean_param


def Median_value_fromDynMaps(masked_image_data, mask_data):
    """
    Calculate median parameter value within masked time series data - calculated
    overall and per slice.
    This could be median SI or median PSE depending on the input data.
    - Need to collapse maps first.
    **Per slice**
    **Not dynamic, single value**

    Arguments:
    masked_image_data = masked dynamic image series (or single image), with
    shape (vox,vox,NumDyn,NumSlices).
    mask_data = lung or cardiac masks, with shape (vox,vox,NumSlices).
    Returns:
    median_param_OverallANDEachSlice = median parameter within the mask over all slices and
    for each slice, having shape (1+NumSlices,).

    """
    # Empty array to store median per slice.
    median_per_slice = np.zeros((np.shape(mask_data)[2]+1))

    # Calculate the median value over all slices.
    median_per_slice[0] = np.median(collapse_maps_Fast(masked_image_data, mask_data))

    # Collapse maps - 1 slice at a time - to calculate
    # the median for each slice.
    for j in range(np.shape(mask_data)[2]):
        # Require the slices to retain a dimension in the masked_image_data array and the
        # mask_data array, in order to use collapsed_slice function.
        masked_image_data_slice_structure = np.transpose(np.array([masked_image_data[:,:,:,j]]), (1,2,3,0))
        mask_data_slice_structure = np.transpose(np.array([mask_data[:,:,j]]), (1,2,0))
        collapsed_slice = collapse_maps_Fast(masked_image_data_slice_structure, mask_data_slice_structure)
        median_per_slice[j+1] = np.median(collapsed_slice)

    return median_per_slice


def timeCalc_perSlice_fromMaps_nonZSupplied(input_timeseries_maps, nonZ_values):
    """
    Calculate mean parameter value within masked time series data.
    This could be mean SI or mean PSE depending on the input data.
    **Dynamic parameter**
    Calculating **PRE SLICE**.

    Arguments:
    input_timeseries_maps = masked dynamic image series (or single image), with
    shape (vox,vox,NumDyn,NumSlices).
    nonZ_values = number of non-zero values in each slice, with shape (NumSlices,).
    Returns:
    mean_param = mean parameter within the mask as a function of time, having shape
    (NumDyn,4).

    """
    # Mean parameter calculation - apply mean three times to collapse dimensions to time only.
    # But, have zeros present. Therefore, calculate sum instead of mean and divide by the number
    # of masked voxels within all four slices.
    mean_timeseries_eachSlice = np.nansum(input_timeseries_maps, axis=0)
    mean_timeseries_eachSlice = np.nansum(mean_timeseries_eachSlice, axis=0)
    mean_timeseries_eachSlice = np.divide(mean_timeseries_eachSlice, nonZ_values)
    return mean_timeseries_eachSlice


def timeCalc_MaskedCollapsedDynamic_EachSlice__MeanMedian(input_ARRAY_masked_collapsed_EachSlice_Dynamic, size_echo_data):
    """
    Calculate for **EACH SLICE**
    # MEAN AND **MEDIAN**

    Calculate mean and median parameter value within masked time series data, separately for each slice.
    This could be mean SI or mean PSE depending on the input data.
    **Dynamic parameter**
    Calculating **PER SLICE**.

    Arguments:
    input_ARRAY_masked_collapsed_EachSlice_Dynamic = masked dynamic image series (or single image), with the masking/collapsing
    for each slice with shape [[(NonZeroVoxels{{vox*vox}},NumDyn)], [NumSlices]].
    Returns:
    mean_timeseries_EachSlice, median_timeseries_EachSlice
    with shape (NumDyn,NumSlices)
    """

    # For each slice, extract the dyn series and calculate mean etc...
    mean_timeseries_EachSlice = np.zeros([size_echo_data[2],size_echo_data[3]])
    median_timeseries_EachSlice = np.zeros([size_echo_data[2],size_echo_data[3]])
    for j in range(size_echo_data[3]):
        mean_timeseries_EachSlice[:,j] = np.mean(input_ARRAY_masked_collapsed_EachSlice_Dynamic[j], axis=0)
        median_timeseries_EachSlice[:,j] = np.median(input_ARRAY_masked_collapsed_EachSlice_Dynamic[j], axis=0)
        # mean_timeseries_EachSlice[:,j] = np.mean(PSE_timeseries_SI_lungMask_Collapsed_EachSlice[j], axis=0)
        # median_timeseries_EachSlice[:,j] = np.median(PSE_timeseries_SI_lungMask_Collapsed_EachSlice[j], axis=0)

    return mean_timeseries_EachSlice, median_timeseries_EachSlice

def timeCalc_MaskedCollapsedDynamic_ALLSlice__MeanMedian(input_masked_collapsed_EachSlice_Dynamic):
    """
    Calculate for **ALL SLICE**
    # MEAN AND **MEDIAN**

    Calculate mean and median parameter value within masked time series data, separately for each slice.
    This could be mean SI or mean PSE depending on the input data.
    **Dynamic parameter**

    Arguments:
    input_masked_collapsed_EachSlice_Dynamic = masked dynamic image series (or single image), with the masking/collapsing
    over all slices with shape (NonZeroVoxels{{vox*vox*NumSlices}},NumDyn).
    Returns:
    mean_timeseries_ALLSlice, median_timeseries_ALLSlice
    with shape (NumDyn,)
    """
    mean_timeseries_EachSlice = np.mean(input_masked_collapsed_EachSlice_Dynamic, axis=0)
    median_timeseries_EachSlice = np.median(input_masked_collapsed_EachSlice_Dynamic, axis=0)
    # mean_timeseries_EachSlice = np.mean(PSE_timeseries_SI_lungMask_Collapsed, axis=0)
    # median_timeseries_EachSlice = np.median(PSE_timeseries_SI_lungMask_Collapsed, axis=0)

    return mean_timeseries_EachSlice, median_timeseries_EachSlice


def timeCalc_fromCollapsedMaps(collapsed_image_data):
    """
    Calculate mean parameter value within collapsed masked time series data.
    This could be mean SI or mean PSE depending on the input data.
    **Dynamic parameter**

    Arguments:
    collapsed_image_data = masked dynamic image series (or single image), with
    shape (NumNonZeroInMask{{vox*vox*NumSlices}},NumDyn).
    Returns:
    mean_param = mean parameter within the mask as a function of time, having shape
    (NumDyn,).
    median_param = median parameter within the mask as a function of time, having shape
    (NumDyn,).

    """
    # Mean parameter calculation.
    mean_param = np.mean(collapsed_image_data, axis=0)
    # Median parameter calculation.
    median_param = np.median(collapsed_image_data, axis=0)

    return mean_param, median_param


def collapse_maps(masked_data, masks):
    """
    Function to 'collapse' masked data - i.e. remove the 0s that are data outside
    the mask. To be used after timeCalc_fromMaps.

    ***Instead*** use collapse_maps_Fast function later in THIS file.
    
    Arguments:
    masked_data = masked data containing 0s outside the mask that is to be 
    collapsed down. With shape (vox,vox,NumDyn,NumSlices).
    masks = image masks, with shape (vox,vox,NumSlices).
    Returns:
    collapsed_data = data within the mask only, 0s removed. With 
    shape (NonZeroVoxels{{vox*vox*NumSlices}},NumDyn).

    """    
    # Create empty array in which to store non-zero (non-masked) voxel values, having
    # size of the number of non-zero values (i.e. sum of values that equal 1, i.e. 
    # just sum the masks). Therefore, shape is (num_non_zero_values, NumDyn).
    # NumDyn can be found from the third index of masked_data ([2] in Python).
    # Number of non-zero values:
    num_mask_voxels = np.sum(masks != 0)
    collapsed_data = np.zeros((num_mask_voxels, np.shape(masked_data)[2]))

    # Also need to reshape the input data to be of the form (vox*vox*NumSlices,NumDyn).
    # This will require permuting (np.transpose) and np.reshape.
    masked_data = np.transpose(masked_data, (0,1,3,2))
    masked_data = np.reshape(masked_data, (np.shape(masked_data)[0]*np.shape(masked_data)[1]*np.shape(masked_data)[2],np.shape(masked_data)[3]))
    # Collapse down the masks as well to match above.
    masks = np.reshape(masks, (np.shape(masks)[0]*np.shape(masks)[1]*np.shape(masks)[2]))

    # Loop over all dynamic images using j. NumDyn is now the second ([1]) index in the
    # permuted and reshaped array.
    for j in range(np.shape(masked_data)[1]):
        # Loop over the voxels in all slices using k. Use counter_k to count the 
        # number of non-zero masked values so that they can be placed correctly into 
        # the data_full_inMask_ONLY array. 
        counter_k = 0
        for k in range(np.shape(masked_data)[0]):
            # Only retain data within the mask, for which the mask value is non-zero (and 
            # the data will also be non-zero due to multiplication with the masks).
            if masks[k] != 0:
                collapsed_data[counter_k,j] = masked_data[k,j]
                counter_k = counter_k + 1
    
    return collapsed_data


def collapse_maps_Fast(masked_data, masks):
    """
    Function to 'collapse' masked data - i.e. remove the 0s that are data outside
    the mask. To be used after timeCalc_fromMaps.
    ***FAST***
    Use instead of collapse_maps - function from earlier in this script.
  
    Arguments:
    masked_data = masked data containing 0s outside the mask that is to be 
    collapsed down. With shape (vox,vox,NumDyn,NumSlices).
    masks = image masks, with shape (vox,vox,NumSlices).
    Returns:
    collapsed_data = data within the mask only, 0s removed. With 
    shape (NonZeroVoxels{{vox*vox*NumSlices}},NumDyn).

    """    
    # Create empty array in which to store non-zero (non-masked) voxel values, having
    # size of the number of non-zero values (i.e. sum of values that equal 1, i.e. 
    # just sum the masks). Therefore, shape is (num_non_zero_values, NumDyn).
    # NumDyn can be found from the third index of masked_data ([2] in Python).
    # Number of non-zero values:
    num_mask_voxels = np.sum(masks != 0)
    collapsed_data = np.zeros((num_mask_voxels, np.shape(masked_data)[2]))

    # Calculate the indices/locations of voxels present in the mask. This will
    # be used later to identify/fill the maps with the collapsed data.
    mask_vals_index = np.where(masks == 1)
    # Shape of mask_vals_index is (3, NumNonZ{{vox,vox,NumSlices}}), where 3 relates to 
    # vox,vox,NumSlices coordinates.

    # Loop over the non-zero values within the mask. The loop is over the non-zero voxels, all
    # components are filled/assigned in this loop via the final index ':'.
    for j in range(np.int32(np.rint(num_mask_voxels))):
        # Identify the coordinates of the jth voxel which is present in the mask.
        coords_index = np.array((mask_vals_index[0][j], mask_vals_index[1][j], mask_vals_index[2][j]))
        # Access the voxel's value and assign to the collapsed data array.
        # But to assign, transpose then transpose back, so dyn_final at end.
        masked_data_reshape_dynFinal = np.transpose(masked_data, (0,1,3,2))
        collapsed_data[j,:] = masked_data_reshape_dynFinal[coords_index[0],coords_index[1],coords_index[2],:]

    return collapsed_data



def collapse_maps_Fast_dynamic_EachSliceArray(input_data, masks):
    """
    Function to 'collapse' masked data - i.e. remove the 0s that are data outside
    the mask.
    # # FOR EACH SLICE SEPARATELY --> returned as an array of collapsed dynamics
    # # for each slice... so can run EachSlice calculations.
    ***FAST***
    Use instead of collapse_maps - function from earlier in this script.

    Arguments:
    input_data = masked data containing 0s outside the mask that is to be 
    collapsed down. With shape (vox,vox,NumDyn,NumSlices).
    masks = image masks, with shape (vox,vox,NumSlices).
    Returns:
    collapsed_data = data within the mask only, 0s removed. With 
    shape ARRAY OUTSIDE [[(NonZeroVoxels{{vox*vox}},NumDyn)], [NumSlices]].

    """    
    num_mask_voxels_EachSlice = np.int32(np.rint(np.zeros((np.shape(masks)[2]))))
    for j in range(np.shape(masks)[2]):
        num_mask_voxels_EachSlice[j] = np.int64(np.rint(np.sum(masks[:,:,j] != 0)))

    # Create empty array to be added to/list
    collapsed_data_EachSlice = []

    for k in range(np.shape(masks)[2]):
        collapsed_data = np.zeros((num_mask_voxels_EachSlice[k], np.shape(input_data)[2]))
        #
        # Calculate the indices/locations of voxels present in the mask. This will
        # be used later to identify/fill the maps with the collapsed data.
        mask_vals_index = np.where(masks[:,:,k] == 1)
        # Loop over the non-zero values within the mask. The loop is over the non-zero voxels, all
        # components are filled/assigned in this loop via the final index ':'.
        for j in range(np.int32(np.rint(num_mask_voxels_EachSlice[k]))):
            coords_index = np.array((mask_vals_index[0][j], mask_vals_index[1][j]))
            # Access the voxel's value and assign to the collapsed data array.
            collapsed_data[j,:] = input_data[coords_index[0],coords_index[1],:,k]
        #
        collapsed_data_EachSlice.append(collapsed_data)

    return collapsed_data_EachSlice


def collapse_maps_Fast_SingleSlice(masked_data, masks):
    """
    Function to 'collapse' masked data - i.e. remove the 0s that are data outside
    the mask. To be used after timeCalc_fromMaps.
    ***FAST***
    Use instead of collapse_maps - function from earlier in this script.
  
    Arguments:
    masked_data = masked data containing 0s outside the mask that is to be 
    collapsed down. With shape (vox,vox,NumDyn).
    masks = image masks, with shape (vox,vox).
    Returns:
    collapsed_data = data within the mask only, 0s removed. With 
    shape (NonZeroVoxels{{vox*vox}},NumDyn).

    """    
    # Create empty array in which to store non-zero (non-masked) voxel values, having
    # size of the number of non-zero values (i.e. sum of values that equal 1, i.e. 
    # just sum the masks). Therefore, shape is (num_non_zero_values, NumDyn).
    # NumDyn can be found from the third index of masked_data ([2] in Python).
    # Number of non-zero values:
    num_mask_voxels = np.sum(masks != 0)
    collapsed_data = np.zeros((num_mask_voxels, np.shape(masked_data)[2]))

    # Calculate the indices/locations of voxels present in the mask. This will
    # be used later to identify/fill the maps with the collapsed data.
    mask_vals_index = np.where(masks == 1)
    # Shape of mask_vals_index is (3, NumNonZ{{vox,vox,NumSlices}}), where 3 relates to 
    # vox,vox,NumSlices coordinates.

    # Loop over the non-zero values within the mask. The loop is over the non-zero voxels, all
    # components are filled/assigned in this loop via the final index ':'.
    for j in range(np.int32(np.rint(num_mask_voxels))):
        # Identify the coordinates of the jth voxel which is present in the mask.
        coords_index = np.array((mask_vals_index[0][j], mask_vals_index[1][j]))
        collapsed_data[j,:] = masked_data[coords_index[0],coords_index[1],:]

    return collapsed_data

def reconstruct_component(S_ica_tVox, A_ica_tVox, ica_tVox, OE_component_num, size_echo_data, NumSlices, NumDyn, masks_reg_data, add_mean=1, create_maps=0):
    """
    Function to reconstruct MRI data from an ICA component.
    Now have options to uncentre etc.
    ***Not unscaling here.
        
    Arguments:
    S_ica_tVox = S_ica output from ICA.
    A_ica_tVox = A_ica output from ICA.
    ica_tVox = ica output from ICA.
    OE_component_num = component to be reconstructed [[index]].
    size_echo_data = to be used when reshaping the (masked) array, giving details
    on the size of the matrix to be created.
    NumSlices = number of image slices acquired.
    NumDyn = number of dynamic images acquired.
    masks_reg_data = image masks, with shape (vox,vox,NumSlices).
    add_mean = 0/1 for No/Yes - whether to add the mean to reverse the centring pre-processing step.
    create_maps = 0/1 for No/Yes - whether to reshape into maps.
    Returns:
    X_recon_OE = reconstructed MRI data (collapsed), with shape of
    (NumDyn, NumNonZ{{vox,vox,Slices}})
    X_recon_OE_maps = reconstruced MRI data reshaped into images, with shape 
    (vox,vox,NumDyn,NumSlices).

    """    
    # Reconstruct with only this component - set all other components to zero.
    # Previously had OE_component -1 for Python indexing for the OE_component... but instead,
    # here argument is the OE component's *INDEX*.

    # Reconstruct
    # Add mean to the dot product as the data input to ICA was centred original data
    # (i.e. made to have zero-mean), and this transformation must be reversed.
    if add_mean == 1:
        X_recon_OE = np.dot(np.transpose(np.array([S_ica_tVox[:,OE_component_num]]), (1,0)), np.array([A_ica_tVox[:,OE_component_num]])) + ica_tVox.mean_
    else:
        X_recon_OE = np.dot(np.transpose(np.array([S_ica_tVox[:,OE_component_num]]), (1,0)), np.array([A_ica_tVox[:,OE_component_num]]))

    # Reshape back
    X_recon_OE_maps = 0
    if create_maps != 0:
        # Instead of num_components, argument for reshaping is number of dynamics.
        # Also need to permute X_recon_OE for (numVox, NumOtherDim).
        # X_recon_OE_maps = Sarah_ICA.reshape_maps(size_echo_data, NumSlices, NumDyn, masks_reg_data, np.transpose(X_recon_OE, (1,0)))
        X_recon_OE_maps = reshape_maps_Fast(size_echo_data, NumSlices, NumDyn, masks_reg_data, np.transpose(X_recon_OE, (1,0)))
        # Permute (np.tranpose) so same shape as loaded data: vox,vox,NumDyn,NumSlices
        X_recon_OE_maps = np.transpose(X_recon_OE_maps, (1,2,0,3))

    return X_recon_OE, X_recon_OE_maps


def reconstruct_component_icaMean(S_ica_tVox, A_ica_tVox, ica_tVox_mean, ica_scaling, OE_component_num, size_echo_data, NumSlices, NumDyn, masks_reg_data, add_mean=1, un_scale=1, create_maps=0):
    """
    Function to reconstruct MRI data from an ICA component.
    Now have options to uncentre etc and scaling factor.
    ** Similar to above but passing mean (ica_tVox_mean) instead of ica_tVox.***
    ** And unscaling.

    Arguments:
    S_ica_tVox = S_ica output from ICA.
    A_ica_tVox = A_ica output from ICA.
    ica_tVox_mean = uncentring mean output from ICA.
    ica_scaling = scaling factor that was applied to ICA.
    OE_component_num = component to be reconstructed [[index]].
    size_echo_data = to be used when reshaping the (masked) array, giving details
    on the size of the matrix to be created.
    NumSlices = number of image slices acquired.
    NumDyn = number of dynamic images acquired.
    masks_reg_data = image masks, with shape (vox,vox,NumSlices).
    add_mean = 0/1 for No/Yes - whether to add the mean to reverse the centring pre-processing step.
    un_scale = 0/1 for No/Yes - whether to reverse the scaling of the ICA input data that
    was performed during an ICA pre-processing step.
    create_maps = 0/1 for No/Yes - whether to reshape into maps.
    Returns:
    X_recon_OE = reconstructed MRI data (collapsed), with shape of
    (NumDyn, NumNonZ{{vox,vox,Slices}})
    X_recon_OE_maps = reconstruced MRI data reshaped into images, with shape 
    (vox,vox,NumDyn,NumSlices).

    """    
    # Reconstruct with only this component - set all other components to zero.
    # Previously had OE_component -1 for Python indexing for the OE_component... but instead,
    # here argument is the OE component's *INDEX*.

    # Reconstruct
    # Add mean to the dot product as the data input to ICA was centred original data
    # (i.e. made to have zero-mean), and this transformation must be reversed.
    # Also reverse scaling if required.
    if add_mean == 1:
        X_recon_OE = np.dot(np.transpose(np.array([S_ica_tVox[:,OE_component_num]]), (1,0)), np.array([A_ica_tVox[:,OE_component_num]])) + ica_tVox_mean
    else:
        X_recon_OE = np.dot(np.transpose(np.array([S_ica_tVox[:,OE_component_num]]), (1,0)), np.array([A_ica_tVox[:,OE_component_num]]))

    if un_scale == 1:
        X_recon_OE = np.multiply(X_recon_OE, ica_scaling)

    # Reshape back
    X_recon_OE_maps = 0
    if create_maps != 0:
        X_recon_OE_maps = reshape_maps_Fast(size_echo_data, NumSlices, NumDyn, masks_reg_data, np.transpose(X_recon_OE, (1,0)))
        X_recon_OE_maps = np.transpose(X_recon_OE_maps, (1,2,0,3))

    return X_recon_OE, X_recon_OE_maps


def reconstruct_component_noMean(S_ica_tVox, A_ica_tVox, OE_component_num):
    """
    Function to reconstruct MRI data from an ICA component.
    NOT reshaping for maps.
    NOT uncentring/adding mean.
    
    Arguments:
    S_ica_tVox = S_ica output from ICA.
    A_ica_tVox = A_ica output from ICA.
    OE_component_num = component to be reconstructed [[Python index]].
    Returns:
    X_recon_OE = reconstructed MRI data (collapsed), with shape of
    (NumDyn, NumNonZ{{vox,vox,Slices}})

    """    
    # Reconstruct with only this component - set all other components to zero.
    # Previously had OE_component -1 for Python indexing for the OE_component... but instead,
    # here argument is the OE component's *INDEX*.
    # Reconstruct
    # Add mean to the dot product as the data input to ICA was centred original data
    # (i.e. made to have zero-mean), and this transformation must be reversed.
    X_recon_OE = np.dot(np.transpose(np.array([S_ica_tVox[:,OE_component_num]]), (1,0)), np.array([A_ica_tVox[:,OE_component_num]]))
    return X_recon_OE


def calc_plot_component_ordering(ordering_method, S_ica_tVox_regOnly, A_ica_tVox_regOnly, \
    freq_S_ica_tVox_regOnly, ica_tVox_regOnly, \
    size_echo_data, NumSlices, NumDyn, halfTimepoints, \
    freqPlot, masks_reg_data_cardiac, \
    num_components, data_scaling, \
    other_parameters, MRI_params):
    """
    Function to order the ICA components using a variety of different criteria/methods.

    Arguments:
    ordering_method = ordering metric technique criteria/method, string.
    S_ica_tVox_regOnly = S_ica output from ICA.
    A_ica_tVox_regOnly = A_ica output from ICA.
    freq_S_ica_tVox_regOnly = frequency spectra of the ICA components calculated from 
    S_ica_tVox_regOnly.
    ica_tVox_regOnly = ica output to use for calculating the mean during uncentring.
    size_echo_data = np.shape(echo_data_load).
    NumSlices = the number of slices acquired.
    NumDyn = the number of dynamic images acquired.
    halfTimepoints = half of the time points value to be used as an index when plotting
    frequencies (freqPlot[0:halfTimepoints]) to plot only one side of the frequency spectrum
    (i.e. the positive frequencies). freqPlot is already adjusted for this, halfTimepoints is
    needed to index the final value of the dependent variable that is being plotted
    on the y-axis.
    # # # # # # GasSwitch = dynamic at which the gases are cycled.
    # # # # # # TempRes = temporal resolution /s.
    masks_reg_data_cardiac = masks.
    num_components = number of ICA components found.
    data_scaling = pre-processing scaling factor applied to the input data before ICA.
    Required during data reconstruction.
    av_im_num = average number of images to average over from the end of each 
    oxygen cycle when calculating mean oxygen values.
    other_parameters = [NumEchoes, num_c_start, num_c_stop, ordering_method_list, num_c_use, av_im_num].
    Returns:
    S_ica_tVox_regOnly_Sorted = sorted S_ica timeseries.
    freq_S_ica_tVox_regOnly_Sorted = sorted frequency spectra (of S_ica).
    RMS_allFreq_P_relMax_argsort_indices = argsort indices - indices for the component ordering.
    RMS_allFreq_P_relMax = sorting metric value for each component.
    RMS_allFreq_P_relMax_Sorted = sorted sorting metric value for each component.

    """
    GasSwitch_here = MRI_params[3][1]
    if len(np.shape(GasSwitch_here)) > 0:
        GasSwitch_here = GasSwitch_here[0]

    TempRes = MRI_params[0]

    # Only assess positive frequencies, contained within 0:halfTimepoints of the frequency spectra.
    freq_S_ica_tVox_regOnly_P = freq_S_ica_tVox_regOnly[0:halfTimepoints]

    # 'Normalise' all components to their maximum frequency amplitude for frequency spectrum
    # analysis. Hence, RMS of frequency spectra power are relative to the maximum component's 
    # frequency amplitude. Required due to possible scaling between component time series and 
    # their maps.
    # Also, for OE component of interest, low frequency OE cycling amplitude has been observed
    # to be >>> higher frequencies (due to respiratory, cardiac etc contamination). Hence, the
    # relative amplitude of higher frequencies should be particularly low for the OE component
    # of interest compared to the other ICA components, enabling identification of the
    # OE component.
    freq_S_ica_tVox_regOnly_P_maxAmp = np.max(np.abs(freq_S_ica_tVox_regOnly_P), axis=0)
    freq_S_ica_tVox_regOnly_P_relMax = np.divide(freq_S_ica_tVox_regOnly_P, freq_S_ica_tVox_regOnly_P_maxAmp)

    # Calculate the (positive) frequencies within the frequency spectra. These can be used 
    # for taking the RMS of the frequencies > OE gas cycling frequency if required by the 
    # ordering metric method.
    positive_freq_vals = freqPlot[0:halfTimepoints]
    OE_freq = 1/(GasSwitch_here * TempRes * 2)
    # This is the lower index which is being counted as part of the gas cycling frequency.
    # The final positive_freq_vals_BelowOEFreq is being counted as the OE frequency as between frequencies.
    positive_freq_vals_BelowOEFreq = positive_freq_vals[positive_freq_vals < OE_freq]
    positive_freq_vals_lower = np.array(positive_freq_vals_BelowOEFreq)
    lower_index_OE_freq = np.shape(positive_freq_vals_BelowOEFreq)[0] - 1 # -1 as python counts from zero
    # This is the upper index which is being counted as part of the gas cycling frequency.
    positive_freq_vals_AboveOEFreq = positive_freq_vals[positive_freq_vals > OE_freq]
    positive_freq_vals_upper = np.array(positive_freq_vals_AboveOEFreq)
    # The first positive_freq_vals_AboveOEFreq is being counted as the OE frequency as between frequencies.
    upper_index_OE_freq = np.shape(positive_freq_vals)[0] - np.shape(positive_freq_vals_AboveOEFreq)[0]

    # Also normalise the frequencies relative to the mean OE frequency amplitude.
    # OE frequency ~ 0.00556 Hz.
    # Frequency array values: 0, 0.00158, 0.00317, 0.00476, 0.00635, 0.00794, 0.00952, ...
    # Therefore take the mean OE frequency amplitude as the mean value of the 
    # absolute amplitudes of the 0.00476, 0.00635 Hz frequency elements.
    # These are 4th and 5th in the array (3, 4 Python counting from zero).
    # # OE_freq_index = np.array((4, 5)) # Counting normally, convert to Python by -1
    # # OE_freq_index = OE_freq_index - 1
    OE_freq_index = np.array((lower_index_OE_freq, upper_index_OE_freq))
    # Calculate the relative frequency compared to the OE frequency amplitude.
    OE_freq_S_ica = np.array((freq_S_ica_tVox_regOnly_P[OE_freq_index[0],:], freq_S_ica_tVox_regOnly_P[OE_freq_index[1],:]))
    freq_S_ica_tVox_regOnly_P_value_relOEAmp = np.mean(np.abs(OE_freq_S_ica), axis=0)
    freq_S_ica_tVox_regOnly_P_relOEAmp = np.divide(freq_S_ica_tVox_regOnly_P, freq_S_ica_tVox_regOnly_P_value_relOEAmp)
    # # Checking below
    # # OE_freq_frequencies = np.array((positive_freq_vals[OE_freq_index[0]], positive_freq_vals[OE_freq_index[1]]))
    # OE_freq_S_ica2 = np.array((freq_S_ica_tVox_regOnly_P[3,:], freq_S_ica_tVox_regOnly_P[4,:]))
    # freq_S_ica_tVox_regOnly_P_value_relOEAmp2 = np.mean(np.abs(OE_freq_S_ica2), axis=0)
    # freq_S_ica_tVox_regOnly_P_relOEAmp2 = np.divide(freq_S_ica_tVox_regOnly_P, freq_S_ica_tVox_regOnly_P_value_relOEAmp2)


    # RMS: root(1/N * sum(squares...)) - N is the number of elements that are included in the sum of
    # squares.

    if ordering_method == "S_ica_SpearmanCorr":
        print("S_ica_SpearmanCorr")
        freqPlot, plot_time_value, halfTimepoints = Sarah_ICA.functions_load_MRI_data.generate_freq(NumDyn, TempRes)
        positive_freq_vals = freqPlot[0:halfTimepoints]
        if MRI_params[3][0] == 1:
            # Cycled3 acquisition
            GasSwitch = MRI_params[3][1]
            OE_freq = 1/(GasSwitch * TempRes * 2)
            av_im_num = other_parameters[5]
            # # Create expected OE function based on sinusoids...
            AirO2Air_GasTimePeriod = 2*TempRes*GasSwitch
            CyclingFrequency = 1/AirO2Air_GasTimePeriod
            air_amp = 1 # 
            cycling_amplitude = np.zeros((np.shape(plot_time_value)))
            cycling_amplitude = cycling_amplitude - 10  #
            # Set amplitude during baseline air
            cycling_amplitude[0:GasSwitch] = air_amp
            # Set amplitude during gas cycling
            cycling_amplitude[GasSwitch:] = np.multiply(air_amp, np.cos(np.multiply(CyclingFrequency*(2*np.pi), plot_time_value[0:-GasSwitch])))
            # But starting from zero --> e.g. -1 from all...
            cycling_amplitude = cycling_amplitude - 1
            # And norm scaling
            cycling_amplitude_norm = np.multiply(cycling_amplitude, 0.5)
            #
            general_r_values = []
            general_p_values = []
            general_NumC_values = []
            for index in range(0, np.shape(S_ica_tVox_regOnly)[1]):
                S_ica_tVox_here_numC = S_ica_tVox_regOnly[:,index]
                # # "NORMALISATION"
                average_air_baseline = np.mean(S_ica_tVox_here_numC[5:GasSwitch])
                # Shift by...
                S_ica_tVox_here_numC_shifted = S_ica_tVox_here_numC - average_air_baseline
                # Now, amplitude of oxygen
                average_oxy_amp_cycle1 = np.mean(S_ica_tVox_here_numC_shifted[2*GasSwitch-av_im_num:2*GasSwitch])
                average_oxy_amp_cycle2 = np.mean(S_ica_tVox_here_numC_shifted[4*GasSwitch-av_im_num:4*GasSwitch])
                average_oxy_amp_cycle3 = np.mean(S_ica_tVox_here_numC_shifted[6*GasSwitch-av_im_num:6*GasSwitch])
                average_oxy_amp_overall = np.mean([average_oxy_amp_cycle1, average_oxy_amp_cycle2, average_oxy_amp_cycle3])
                # For amplitude of 1 (so that reaches -1 during oxy roughly)
                S_ica_tVox_here_numC_shifted_scaled = np.multiply(S_ica_tVox_here_numC_shifted, np.abs((1/average_oxy_amp_overall)))
                #
                r,p = stats.spearmanr(cycling_amplitude_norm, S_ica_tVox_here_numC_shifted_scaled)
                general_r_values.append(r)
                general_p_values.append(p)
                general_NumC_values.append(index)
        #
        # Once finished loop, take inverse of r_values for minimum value to correspond to the
        # greatest correlation.
        # Also take absolute value - as ICA components can have +/- sign.
        general_r_values = np.abs(np.divide(1, general_r_values))
        # Set RMS_allFreq_P_relMax to general_r_values - new ordering approach here.
        RMS_allFreq_P_relMax = np.array(general_r_values)

    RMS_allFreq_P_relMax_argsort_indices = np.argsort(RMS_allFreq_P_relMax, axis=0)
    RMS_allFreq_P_relMax_argsort_indices_ALL = np.zeros(np.shape(S_ica_tVox_regOnly))
    for j in range(NumDyn):
        RMS_allFreq_P_relMax_argsort_indices_ALL[j,:] = RMS_allFreq_P_relMax_argsort_indices
    RMS_allFreq_P_relMax_argsort_indices_ALL = np.int32(RMS_allFreq_P_relMax_argsort_indices_ALL)
    S_ica_tVox_regOnly_Sorted = np.take_along_axis(S_ica_tVox_regOnly, RMS_allFreq_P_relMax_argsort_indices_ALL, axis=1)
    RMS_allFreq_P_relMax_Sorted = np.take_along_axis(RMS_allFreq_P_relMax, RMS_allFreq_P_relMax_argsort_indices, axis=0)
    freq_S_ica_tVox_regOnly_Sorted = Sarah_ICA.freq_spec(num_components, S_ica_tVox_regOnly_Sorted)
    return S_ica_tVox_regOnly_Sorted, freq_S_ica_tVox_regOnly_Sorted, RMS_allFreq_P_relMax_argsort_indices, \
        RMS_allFreq_P_relMax, RMS_allFreq_P_relMax_Sorted


def order_components(array_to_order, arg_sort_indices):
    """
    Function to order data (ICA components) according to the sorting indices.
    Require the sorting indices to be present along rows of data array to be
    sorted, hence need to copy repeated columns of the arg_sort_indices
    to match array_to_order.

    Arguments:
    array_to_order = array to be ordered, with shape (anything, num_components).
    arg_sort_indices = column array of indices to be used to sort the data, 
    with shape (num_components,).
    Returns:
    sorted_array = sorted data array, with shape (anything, num_components).
    
    """
    arg_sort_indices_ALL = np.zeros(np.shape(array_to_order))
    # Loop over first dimension of the array to be ordered (likely to be NumDyn).
    for j in range(np.shape(array_to_order)[0]):
        arg_sort_indices_ALL[j,:] = arg_sort_indices
    # After above, np.take_along_axis() can be used, but need to ensure that the arg_sort indices are integers.
    arg_sort_indices_ALL = np.int32(arg_sort_indices_ALL)
    sorted_array = np.take_along_axis(array_to_order, arg_sort_indices_ALL, axis=1)
    return sorted_array


def reshape_maps_Fast(size_echo_data, NumSlices, num_components, masks_reg_data, array_to_reshape):
    """
    Reshape array of masked values in the form of an array that can be plotted when
    mapping values.
    ****FAST****
    ---> by using indices to relocate values.

    Below for version to reshape for a dynamic series of images...
    
    Arguments:
    size_echo_data = the size of the array of each map that is to be created, found
    from np.shape(load_data).
    NumSlices = the number of slices acquired.
    num_components = the number of maps to be created from the data.
    masks_reg_data = masks originally applied to the data. Used for reshaping the 
    data back into an array form from the knowledge of which voxels were inside the
    mask. With shape (vox, vox, NumSlices)
    array_to_reshape = array of masked data to be reshaped into an array for plotting. 
    e.g. A_ica_tVox. Of the shape (num_non_zero_masked, num_components), where
    num_components is the number of maps from the array.
    Returns:
    reshape_maps_seg = reshaped maps (single map per component, one 2D map per slice), 
    having shape (num_components, vox, vox, NumSlices).
        
    """
    # Calculate the indices/locations of voxels present in the mask. This will
    # be used later to identify/fill the maps with the collapsed data.
    mask_vals_index = np.where(masks_reg_data == 1)
    # Shape of mask_vals_index is (3, NumNonZ{{vox,vox,NumSlices}}), where 3 relates to 
    # vox,vox,NumSlices coordinates.
    # Create empty array to store the mapped/relocated maps.
    reshape_maps_seg = np.zeros((size_echo_data[0], size_echo_data[1],NumSlices,num_components))

    # Loop over the non-zero values within the mask. The loop is over the non-zero voxels, all
    # components are filled/assigned in this loop via the final index ':'.
    for j in range(np.int32(np.rint(np.sum(masks_reg_data)))):
        # Identify the coordinates of the jth voxel which is present in the mask.
        coords_index = np.array((mask_vals_index[0][j], mask_vals_index[1][j], mask_vals_index[2][j]))
        # Assign the voxel's value to the correct location in the reshaped map array.
        reshape_maps_seg[coords_index[0],coords_index[1],coords_index[2],:] = array_to_reshape[j,:]

    # Permute the component maps to have shape (num_components, vox, vox, NumSlices).
    reshape_maps_seg = np.transpose(reshape_maps_seg, (3,0,1,2))

    return reshape_maps_seg


def reshape_maps_Fast_DYNAMICS(NumDyn, NumSlices, size_echo_data, masks_reg_data_cardiac, array_to_reshape):
    """
    Reshape array of masked values in the form of an array that can be plotted when
    mapping values.
    ****FAST****
    ---> by using indices to relocate values.
    
    Arguments:
    NumDyn = number of dynamic images acquired.
    NumSlices = number of slices acquired.
    size_echo_data = shape of the loaded MRI data to base the reshaped maps on.
    masks_reg_data_cardiac = masks applied to the original data to use 
    for reshaping the collapsed array.
    array_to_reshape = array to be reshaped, of shape 
    (NonZ{{vox,vox,slices,NumDyn}},1).
    Returns:
    reshape_maps_dyn = dynamic maps of the reshaped input data. Of shape
    equal to size_echo_data (vox,vox,NumDyn,NumSlices).
        
    """
    # First, reshape the reconstructed data to obtain a dynamic series 
    # of collapsed/masked data.
    array_to_reshape = np.reshape(array_to_reshape, (np.int32(np.rint(np.sum(masks_reg_data_cardiac))),NumDyn))

    # Call reshape_maps_Fast, setting num_components equal to NumDyn due to the shape of the data.
    reshape_maps_dyn = reshape_maps_Fast(size_echo_data, NumSlices, NumDyn, masks_reg_data_cardiac, array_to_reshape)

    # Need to permute (np.transpose()) the maps for the shape (vox,vox,NumDyn,NumSlices).
    reshape_maps_dyn = np.transpose(reshape_maps_dyn, (1,2,0,3))
    return reshape_maps_dyn



if __name__ == "__main__":
    # ...
    a = 1