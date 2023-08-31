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

# Functions to apply ICA.

import numpy as np
import Sarah_ICA
import functions_preproc_MRI_data

def LoadData(dir_date, dir_subj, \
    echo_num, dictionary_name, \
    MRI_scan_directory, MRI_params):
    """
    Function to load data and mask. Also to return the full size of the echo data
    (size_echo_data) which is required by later functions.

    Arguments:
    dir_date = subject scanning date.
    dir_subj = subject ID.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    dictionary_name = name and location of the mask dictionary containing the
    subject under investigation.
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    MRI_params = [TempRes, NumSlices, NumDyn, GasSwitch, time_plot_side, TE_s, NumEchoes, Vox1, Vox2]
    Returns:
    masked_data_echo1_regOnly = masked data that ha been collapsed, having shape 
    (num_voxels_within_mask_all_slices, NumDyn).
    size_echo_data = size (np.shape) of the original MRI data loaded in, prior
    to masking/collapse.

    """
    # MRI images directory and mask directory.
    dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_mask = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[4] + '/'

    # Here call load_MRI_data_raw and load the registered first echo data.
    load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw(dir_images, MRI_params[1], echo_num, MRI_params[2])
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
    return masked_data_echo1_regOnly, size_echo_data


def LoadMasks(dir_date, dir_subj, dictionary_name, echo_num, NumDyn, NumSlices, mask_load, \
    MRI_scan_directory):
    """
    Function to load masks only.

    Arguments:
    dir_date = subject scanning date.
    dir_subj = subject ID.
    dictionary_name = name and location of the mask dictionary containing the
    subject under investigation.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    NumDyn = the number of dynamic images acquired.
    NumSlices = the number of slices acquired.
    mask_load = 'cardiac' or 'lung' - which set of masks are to be loaded.
    MRI_scan_directory = details of where the MRI scan data is stored, as a list consisting of 
        [dir_base, dir_gas, dir_end, dir_image_ending, dir_mask_ending].
    Returns:
    masks_data = masks for the subject, with shape (96, 96, NumSlices).

    """
    # MRI images directory and mask directory.
    dir_images = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[3] + '/'
    dir_images_mask = MRI_scan_directory[0] + dir_date + '/' + dir_date + '_' + dir_subj + MRI_scan_directory[2] + MRI_scan_directory[1] + '/' + MRI_scan_directory[4] + '/'

    # Here call load_MRI_data_raw and load the registered first echo data.
    load_data_echo1_regOnly = Sarah_ICA.load_MRI_data_raw(dir_images, NumSlices, echo_num, NumDyn)
    size_echo_data = np.shape(load_data_echo1_regOnly)

    # First find the names of the masks to be loaded for the particular subject.
    subject = dir_date + '_' + dir_subj
    mask_names_func_lung = functions_preproc_MRI_data.load_mask_subj_names(dictionary_name, 1, subject)
    mask_names_func_cardiac = functions_preproc_MRI_data.load_mask_subj_names(dictionary_name, 0, subject)
    # Load the image masks.
    if mask_load == 0:
        # Load cardiac mask
        masks_data = Sarah_ICA.load_MRI_reg_masks(dir_images_mask, mask_names_func_cardiac, NumSlices, \
            (size_echo_data[0],size_echo_data[1],size_echo_data[3]))
    else:
        # Load lung mask
        masks_data = Sarah_ICA.load_MRI_reg_masks(dir_images_mask, mask_names_func_lung, NumSlices, \
            (size_echo_data[0],size_echo_data[1],size_echo_data[3]))
    return masks_data


def preproc_ICA_data(masked_data_singleEcho):
    """
    Function to perform just the "ICA pre-processing step". 
    Pre-process masked/collapsed data - specifically scaling of the data.

    Arguments:
    masked_data_singleEcho = masked data that ha been collapsed, having shape 
    (num_voxels_within_mask_all_slices, NumDyn).
    Returns:
    masked_data_singleEcho_PreProc = masked data that ha been collapsed *and scaled*,
    having shape (num_voxels_within_mask_all_slices, NumDyn).
    data_scaling = scaling factor.
    
    """
    # # Perform ICA...
    # Scale the masked data to the maximum SI amplitude (within the mask).
    masked_data_singleEcho_PreProc, data_scaling = Sarah_ICA.preproc_ICA(masked_data_singleEcho)
    return masked_data_singleEcho_PreProc, data_scaling


if __name__ == "__main__":
    # ...
    a = 1
