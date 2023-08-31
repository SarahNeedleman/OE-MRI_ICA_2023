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

# Functions to load MRI data for use in ICA analysis.

import nibabel as nib
import numpy as np
from sklearn.decomposition import FastICA
from scipy.fft import fft, ifft, fftfreq
import functions_testing_pythonFastICA_errorsWarnings
import functions_Calculate_stats

def load_MRI_data_raw(dir_images, NumSlices, echo_load, NumDyn, Vox1, Vox2):
    """
    Function to load in a specific echo and all dynamics of the raw MRI data.
    *All slices loaded*

    Arguments:
    dir_image = directory where the MRI data are stored for a specific subject, including whether
    the data is registered etc, and the protocol (SingleCycle or Cycled3).
    NumSlices = the number of slices acquired.
    echo_load = the echo to be loaded - 1st or 2nd echo.
    NumDyn = the number of dynamic images acquired.
    Vox1,Vox2 = the in-plane voxel dimensions.
    Returns:
    data_load_series = an array containing the MRI data with 
    shape = (vox, vox, NumDyn, NumSlices).

    # Plotting of image arrays with the form (vox, vox, NumDyn, NumSlices).
    PLOT:
    array as is, but use ax.invert_yaxis() after plt.imshow(image_array).
    i.e. the loaded matrix has matrix coordinates and is indexed using [y,x].
    ROI:
    [x1, x2, y1, y2] aka [xmin, xmax, ymin, ymax]
    Index/select from array as:
    image_array[ROI_lung[2]:ROI_lung[3]+1, ROI_lung[0]:ROI_lung[1]+1, dyn_to_plot, slice_to_plot]
    i.e. as image_array[y1:y2, x1:x2, dynamic, slice]
    #
    # E.g.
    ROI_lung = [27, 34, 43, 58]
    # Plot to check
    fig, ((ax1)) = plt.subplots(1, 1)
    im1 = ax1.imshow(array[ROI_lung[2]:ROI_lung[3]+1,ROI_lung[0]:ROI_lung[1]+1,dyn_to_plot], cmap='bwr')
    ax1.set_title("Component #1 map (A tVox)"); ax1.invert_yaxis(); fig.colorbar(im1, ax=ax1)
    plt.show()

    """
    # Load all dynamic images for the specified echo.
    # Create an empty array to store the loaded data in with the shape (vox, vox, NumDyn, NumSlices)
    data_load_series = np.zeros((Vox1,Vox2,NumDyn,NumSlices))
    # Loop over the slices and loop over the images in the dynamic series.
    # Slice loop uses k; dynamic image loop uses j.
    for k in range(NumSlices):
        # Loop over the number of slices.
        # # If fewer than 4 slices are required, these can be loaded, asssuming the
        # # slices to be loaded include the most posterior slices, i.e. the more anterior
        # # slices are not loaded due to the cardiac signals they contain.
        if NumSlices == 2:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # If NumSlices = 2, to select the most posterior slices to be loaded, + 2 to the 
                # slice index of k, i.e. (k+2).
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1+2)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + '.img')
                data_load_in = np.array(data_load_in.dataobj)
                # Shape of the loaded data is (96,96,1,1). Need to use np.squeeze to remove the 
                # additional dimensions single element dimensions (dimensions with shape equal to 1).
                data_load_in = np.squeeze(data_load_in)
                # Also need to reorientate so that the voxel coordinates follow the matrix coordinate 
                # convention (indexing as [y,x]). Hence, when plotting, ax.invert_yaxis() is required.
                data_load_in = np.transpose(data_load_in, [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 3:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # If NumSlices = 3, to select the most posterior slices to be loaded, + 1 to the 
                # slice index of k, i.e. (k+1).
                # k + 1 is used as the MRI data has slices counting from one, whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + '.img')          
                # # OR, if the slices were too posterior, instead could load the first three slices and 
                # # exclude the most posterior slice. This requires using only (k + 1).
                # data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                #     ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + '.img')            
                data_load_in = np.array(data_load_in.dataobj)
                # Shape of the loaded data is (96,96,1,1). Need to use np.squeeze to remove the 
                # additional dimensions single element dimensions (dimensions with shape equal to 1).
                data_load_in = np.squeeze(data_load_in)
                # Also need to reorientate so that the voxel coordinates follow the matrix coordinate 
                # convention (indexing as [y,x]). Hence, when plotting, ax.invert_yaxis() is required.
                data_load_in = np.transpose(data_load_in, [1,0])
                # Assign into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 4:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + '.img')
                data_load_in = np.array(data_load_in.dataobj)
                # Shape of the loaded data is (96,96,1,1). Need to use np.squeeze to remove the 
                # additional dimensions single element dimensions (dimensions with shape equal to 1).
                data_load_in = np.squeeze(data_load_in)
                # Also need to reorientate so that the voxel coordinates follow the matrix coordinate 
                # convention (indexing as [y,x]). Hence, when plotting, ax.invert_yaxis() is required.
                data_load_in = np.transpose(data_load_in, [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 6:
            # 6 slices - acquired for the single cycle data.
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + '.img')
                data_load_in = np.array(data_load_in.dataobj)
                # Shape of the loaded data is (96,96,1,1). Need to use np.squeeze to remove the 
                # additional dimensions single element dimensions (dimensions with shape equal to 1).
                data_load_in = np.squeeze(data_load_in)
                # Also need to reorientate so that the voxel coordinates follow the matrix coordinate 
                # convention (indexing as [y,x]). Hence, when plotting, ax.invert_yaxis() is required.
                data_load_in = np.transpose(data_load_in, [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
    return data_load_series


def load_MRI_data_raw_nii(dir_images, NumSlices, echo_load, NumDyn, Vox1, Vox2, \
    densityCorr):
    """
    Function to load in a specific echo and all dynamics of the raw MRI data.
    *All slices loaded*
    ** nii data **.

    Arguments:
    dir_image = directory where the MRI data are stored for a specific subject, including whether
    the data is registered etc, and the protocol (SingleCycle or Cycled3).
    NumSlices = the number of slices acquired.
    echo_load = the echo to be loaded - 1st or 2nd echo.
    NumDyn = the number of dynamic images acquired.
    Vox1,Vox2 = the in-plane voxel dimensions.
    densityCorr = 0/1 for N/Y - whether the images are density corrected.
    Returns:
    data_load_series = an array containing the MRI data with 
    shape = (vox, vox, NumDyn, NumSlices).

    # Plotting of image arrays with the form (vox, vox, NumDyn, NumSlices).
    PLOT:
    array as is, but use ax.invert_yaxis() after plt.imshow(image_array).
    i.e. the loaded matrix has matrix coordinates and is indexed using [y,x].
    ROI:
    [x1, x2, y1, y2] aka [xmin, xmax, ymin, ymax]
    Index/select from array as:
    image_array[ROI_lung[2]:ROI_lung[3]+1, ROI_lung[0]:ROI_lung[1]+1, dyn_to_plot, slice_to_plot]
    i.e. as image_array[y1:y2, x1:x2, dynamic, slice]
    #
    # E.g.
    ROI_lung = [27, 34, 43, 58]
    # Plot to check
    fig, ((ax1)) = plt.subplots(1, 1)
    im1 = ax1.imshow(array[ROI_lung[2]:ROI_lung[3]+1,ROI_lung[0]:ROI_lung[1]+1,dyn_to_plot], cmap='bwr')
    ax1.set_title("Component #1 map (A tVox)"); ax1.invert_yaxis(); fig.colorbar(im1, ax=ax1)
    plt.show()

    """
    # Load all dynamic images for the specified echo.
    # Create an empty array to store the loaded data in with the shape (vox, vox, NumDyn, NumSlices)
    data_load_series = np.zeros((Vox1,Vox2,NumDyn,NumSlices))
    #
    dir_images = dir_images + 'reg_'
    #
    # Loop over the slices and loop over the images in the dynamic series.
    # Slice loop uses k; dynamic image loop uses j.
    for k in range(NumSlices):
        # Loop over the number of slices.
        # # If fewer than 4 slices are required, these can be loaded, asssuming the
        # # slices to be loaded include the most posterior slices, i.e. the more anterior
        # # slices are not loaded due to the cardiac signals they contain.
        if NumSlices == 2:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # If NumSlices = 2, to select the most posterior slices to be loaded, + 2 to the 
                # slice index of k, i.e. (k+2).
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1+2)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 3:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # If NumSlices = 3, to select the most posterior slices to be loaded, + 1 to the 
                # slice index of k, i.e. (k+1).
                # k + 1 is used as the MRI data has slices counting from one, whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 4:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 5:
            # 5 slices.
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 6:
            # 6 slices - acquired for the single cycle data.
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
    return data_load_series


def load_MRI_data_raw_unreg_nii(dir_images, NumSlices, echo_load, NumDyn, Vox1, Vox2, \
    densityCorr):
    """
    Function to load in a specific echo and all dynamics of the raw MRI data.
    *All slices loaded*
    ** nii data **.

    Arguments:
    dir_image = directory where the MRI data are stored for a specific subject, including whether
    the data is registered etc, and the protocol (SingleCycle or Cycled3).
    NumSlices = the number of slices acquired.
    echo_load = the echo to be loaded - 1st or 2nd echo.
    NumDyn = the number of dynamic images acquired.
    Vox1,Vox2 = the in-plane voxel dimensions.
    densityCorr = 0/1 for N/Y - whether the images are density corrected.
    Returns:
    data_load_series = an array containing the MRI data with 
    shape = (vox, vox, NumDyn, NumSlices).

    # Plotting of image arrays with the form (vox, vox, NumDyn, NumSlices).
    PLOT:
    array as is, but use ax.invert_yaxis() after plt.imshow(image_array).
    i.e. the loaded matrix has matrix coordinates and is indexed using [y,x].
    ROI:
    [x1, x2, y1, y2] aka [xmin, xmax, ymin, ymax]
    Index/select from array as:
    image_array[ROI_lung[2]:ROI_lung[3]+1, ROI_lung[0]:ROI_lung[1]+1, dyn_to_plot, slice_to_plot]
    i.e. as image_array[y1:y2, x1:x2, dynamic, slice]
    #
    # E.g.
    ROI_lung = [27, 34, 43, 58]
    # Plot to check
    fig, ((ax1)) = plt.subplots(1, 1)
    im1 = ax1.imshow(array[ROI_lung[2]:ROI_lung[3]+1,ROI_lung[0]:ROI_lung[1]+1,dyn_to_plot], cmap='bwr')
    ax1.set_title("Component #1 map (A tVox)"); ax1.invert_yaxis(); fig.colorbar(im1, ax=ax1)
    plt.show()

    """
    # Load all dynamic images for the specified echo.
    # Create an empty array to store the loaded data in with the shape (vox, vox, NumDyn, NumSlices)
    data_load_series = np.zeros((Vox1,Vox2,NumDyn,NumSlices))
    #
    # dir_images = dir_images + 'reg_'
    #
    # Loop over the slices and loop over the images in the dynamic series.
    # Slice loop uses k; dynamic image loop uses j.
    for k in range(NumSlices):
        # Loop over the number of slices.
        # # If fewer than 4 slices are required, these can be loaded, asssuming the
        # # slices to be loaded include the most posterior slices, i.e. the more anterior
        # # slices are not loaded due to the cardiac signals they contain.
        if NumSlices == 2:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # If NumSlices = 2, to select the most posterior slices to be loaded, + 2 to the 
                # slice index of k, i.e. (k+2).
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1+2)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 3:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # If NumSlices = 3, to select the most posterior slices to be loaded, + 1 to the 
                # slice index of k, i.e. (k+1).
                # k + 1 is used as the MRI data has slices counting from one, whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 4:
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 5:
            # 5 slices.
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
        if NumSlices == 6:
            # 6 slices - acquired for the single cycle data.
            for j in range(NumDyn):
                # Load one slice from a single dynamic at a time.
                # k + 1 is used as the MRI data naming convention counts the slices starting from one,
                # whereas python counts from zero.
                # Similarly, j + 1 is used as the MRI data naming convention for the dynamic image number
                # starts counting from one.
                data_load_in = nib.load(dir_images + 'IM-' + ('%(number)02d' %{"number": (k+1)}) + '-' + \
                    ('%(number)02d' %{"number": echo_load}) + '-' + ('%(number)03d' %{"number": (j+1)}) + \
                    '.nii').get_fdata()
                data_load_in = np.transpose(np.squeeze(data_load_in), [1,0])
                # Assign the loaded data (one slice from a single dynamic image) into the timeseries array.
                data_load_series[:,:,j,k] = data_load_in
    return data_load_series




def load_MRI_reg_masks(dir_images_mask, mask_name_list, NumSlices, size_mask_array):
    """
    Function to load in the masks (for registered data). Manual segmentation 
    used to create the masks using ImageJ. Mask file format = analyze.
    Either LungMasks (excluding major vessels) or CardiacMasks (thoracic).
    Masks loaded take values 0 or 255, and need to be converted to binary masks (0 or 1) 
    for use in future code. 1 indicates inside the masked region of interest.

    Arguments:
    dir_image_mask = directory where the MRI masks are stored for a specific subject.
    masks_name_list = list of names of the masks for the slices for the subject.
    NumSlices = the number of slices acquired.
    size_mask_array = the size of an array containing the masks for all slices. This
    is equal to (size_echo_data[0], size_echo_data[1], size_echo_data[3]) where 
    size_echo_data = np.shape(echo_data_load).
    Returns:
    mask_reg_allSlices = an array containing all of the masks for a subject.

    """
    # Load all masks into 3D array of the same size as the images.
    # Create an empty array to store the loaded data in with the shape (vox, vox, NumSlices),
    # the shape is specified as an input argument.
    mask_reg_allSlices = np.zeros((size_mask_array))
    # Loop over the slices for the mask using the index j.
    for j in range(size_mask_array[2]):
        # Use the name of the mask from the list to create the filename of the analyze mask
        # to be loaded in.
        # No +1 is required as indexing within a list.
        mask_reg_slice = nib.load(dir_images_mask + mask_name_list[j] + '.img')
        mask_reg_slice = np.array(mask_reg_slice.dataobj)
        # Need to use np.squeeze to remove the additional dimensions single element 
        # dimensions (dimensions with shape equal to 1).
        mask_reg_slice = np.squeeze(mask_reg_slice)
        # Also need to reorientate so that the voxel coordinates follow the matrix coordinate 
        # convention (indexing as [y,x]). Hence, when plotting, ax.invert_yaxis() is required.
        mask_reg_slice = np.transpose(mask_reg_slice, [1,0])
        # Require the mask to be binary (0 or 1), but the loaded analyze (and jpeg) masks have
        # values of 255 and 0. Therefore, divide by 255 (or the mode value).
        mask_reg_slice = np.divide(mask_reg_slice, 255)
        mask_reg_allSlices[:,:,j] = mask_reg_slice
        del mask_reg_slice
    return mask_reg_allSlices



def load_MRI_reg_masks_SingleSlice(dir_images_mask, mask_name_SingleString, size_mask_array):
    """
    Function to load in the masks (for registered data). Manual segmentation 
    used to create the masks using ImageJ. Mask file format = analyze.
    Either LungMasks (excluding major vessels) or CardiacMasks (thoracic).
    Masks loaded take values 0 or 255, and need to be converted to binary masks (0 or 1) 
    for use in future code. 1 indicates inside the masked region of interest.

    Arguments:
    dir_image_mask = directory where the MRI masks are stored for a specific subject.
    mask_name_SingleString = SINGLE STRING for the name of the mask of a single slice.
    size_mask_array = the size of an array containing the masks for all slices. This
    is equal to (size_echo_data[0], size_echo_data[1]) where 
    size_echo_data = np.shape(echo_data_load).
    Returns:
    mask_reg_allSlices = an array containing all of the masks for a subject.

    """
    # Load all masks into 3D array of the same size as the images.
    # Create an empty array to store the loaded data in with the shape (vox, vox, NumSlices),
    # the shape is specified as an input argument.
    mask_reg_allSlices = np.zeros((size_mask_array))
    # Use the name of the mask from the list to create the filename of the analyze mask
    # to be loaded in.
    # No +1 is required as indexing within a list.
    mask_reg_slice = nib.load(dir_images_mask + mask_name_SingleString + '.img')
    mask_reg_slice = np.array(mask_reg_slice.dataobj)
    # Need to use np.squeeze to remove the additional dimensions single element 
    # dimensions (dimensions with shape equal to 1).
    mask_reg_slice = np.squeeze(mask_reg_slice)
    # Also need to reorientate so that the voxel coordinates follow the matrix coordinate 
    # convention (indexing as [y,x]). Hence, when plotting, ax.invert_yaxis() is required.
    mask_reg_slice = np.transpose(mask_reg_slice, [1,0])
    # Require the mask to be binary (0 or 1), but the loaded analyze (and jpeg) masks have
    # values of 255 and 0. Therefore, divide by 255 (or the mode value).
    mask_reg_slice = np.divide(mask_reg_slice, 255)
    return mask_reg_slice



def apply_masks(mask_reg_allSlices, data_full):
    """
    Function to apply the masks to registered data.
    **Using newer FAST version*

    Arguments:
    mask_reg_allSlices = the array containing the masks of all of the slices,
    which has shape (vox,vox,NumSlices).
    data_full = array of the dynamic series of images for all slices on
    which the masking is to be performed. Shape of (vox,vox,NumDyn,NumSlices).
    Returns:
    data_full_inMask_ONLY = collapsed array of the data contained within the
    masks for all slices at each time point, with a shape 
    (num_voxels_within_mask_all_slices, NumDyn).

    """
    # Apply mask to entire timeseries.
    # Mask array has shape (vox,vox,NumSlices) whereas data array has shape (vox,vox,NumDyn,NumSlices).
    # Need to permute (np.transpose) the data array to enable the multiplication of the two arrays
    # (via/using broadcasting).
    # Broadcasting requires: (additional_dim, dim1, dim2, dim3) and (dim1, dim2, dim3).
    data_full_masked = np.multiply(mask_reg_allSlices,np.transpose(data_full, (2,0,1,3)))
    # Reverse the permute (np.transpose) for the masked array shape to have the shape (Vox,Vox,NumDyn,NumSlices).
    data_full_masked = np.transpose(data_full_masked, (1,2,0,3))

    # NEW: use functions_Calculate_stats.collapse_maps_Fast
    # for rapid collapse of masked data.
    data_full_inMask_ONLY = functions_Calculate_stats.collapse_maps_Fast(data_full_masked, mask_reg_allSlices)
    return data_full_inMask_ONLY


def apply_mask_SingleSliceDyn(mask_reg_SingleSlice, data_full_SingleSlice):
    """
    Function to apply the masks to registered data.
    **Using newer FAST version*

    Arguments:
    mask_reg_SingleSlice = the array containing the mask,
    which has shape (vox,vox).
    data_full = array of the dynamic series of images  on
    which the masking is to be performed. Shape of (vox,vox,NumDyn).
    Returns:
    data_full_inMask_ONLY = collapsed array of the data contained within the
    masks at each time point, with a shape 
    (num_voxels_within_mask, NumDyn).

    """
    # Apply mask to entire timeseries.
    # Mask array has shape (vox,vox,NumSlices) whereas data array has shape (vox,vox,NumDyn,NumSlices).
    # Need to permute (np.transpose) the data array to enable the multiplication of the two arrays
    # (via/using broadcasting).
    # Broadcasting requires: (additional_dim, dim1, dim2, dim3) and (dim1, dim2, dim3).
    data_full_masked = np.multiply(mask_reg_SingleSlice,np.transpose(data_full_SingleSlice, (2,0,1)))
    # Reverse the permute (np.transpose) for the masked array shape to have the shape (Vox,Vox,NumDyn,NumSlices).
    data_full_masked = np.transpose(data_full_masked, (1,2,0))

    # NEW: use functions_Calculate_stats.collapse_maps_Fast
    # for rapid collapse of masked data.
    data_full_inMask_ONLY = functions_Calculate_stats.collapse_maps_Fast_SingleSlice(data_full_masked, mask_reg_SingleSlice)
    return data_full_inMask_ONLY


def generate_freq(NumDyn, TempRes):
    """
    Function to generate the frequencies and time points of the OE-MRI dynamic data
    for use when plotting.

    Arguments:
    NumDyn = the number of dynamic images acquired.
    TempRes = the temporal imaging resolution /s.
    Returns:
    freqPlot = frequency values for plotting along the x-axis (only for the positive
    frequencies).
    timepoints = time points for plotting along the x-axis.
    halfTimepoints = half of the time points value to be used as an index when plotting
    frequencies (freqPlot[0:halfTimepoints]) to plot only one side of the frequency spectrum
    (i.e. the positive frequencies). freqPlot is already adjusted for this, halfTimepoints is
    needed to index the final value of the dependent variable that is being plotted
    on the y-axis.

    """
    # Create array of the time points that the images were acquired at.
    timeseries = np.arange(0,NumDyn,1)
    timepoints = timeseries*TempRes
    # For frequency plotting, plot the first half of the frequency spectrum 
    # which corresponds to the positive frequencies (after FFT in python).
    # As N (#timepoints/#sample points) is even, the positive 
    # frequencies are y[0], y[1], ... y[N/2-1].
    # --> the half time point index is NumDyn / 2 -1 but +1 for python
    # indexing --> NumDyn/2. Require an integer for indexing.
    halfTimepoints = np.int16(np.rint(NumDyn/2)) # N/2 whole number

    # Use fftfreq from python to create an array of the positive and negative
    # frequencies present.
    # fftfreq(signal_size, time_step)
    freqAll = fftfreq(NumDyn, TempRes)
    # Only retain the positive frequencies as only these will be plotted.
    # i.e. retain frequencies up to [:N/2] (for N/2-1, as + 1 for python indexing).
    freqPlot = freqAll[:halfTimepoints]
    return freqPlot, timepoints, halfTimepoints


def preproc_ICA(masked_data):
    """
    Function to pre-process the masked data to be ready for ICA.
    The data array needs to have the correct orientation for temporal ICA. In
    python scikit-learn FastICA, time down rows and voxels along the columns: tVox.
    This corresponds to voxels down rows and time along the columns compared to X
    in the literature, due to the different definition used in scikit-learn FastICA.
    FastICA will perform centring of the data, hence only scaling is required as
    a pre-processing step. Scaling is performed so that the maximum 
    array value equals 1. The scaling value should be retained for use when
    reconstructing X from the independent components extracted.

    Arguments:
    masked_data = array of collapsed masked data, as output from the function
    Sarah_ICA.apply_masks. Shape is (num_non_zero_masked_all_slices,NumDyn).
    Returns:
    X_scale_tVox = scaled masked_data. Scaling is performed so that the maximum 
    array value equals 1.
    max_X_load = value used to scale the data (this is the maximum array value).
    The data scaling value should be retained for use when reconstructing X 
    from the independent components extracted.

    """
    # Need to alter the data array to be tVox form, therefore need to 
    # permute (np.transpose) the axis for the shape (NumDyn,num_non_zero_masked_all_slices).
    X_load = np.transpose(masked_data, (1,0))

    # Scale the (SI) values.
    # Scale the SI values by the maximum SI in the entire (masked) data set.
    max_X_load = np.max(X_load)
    X_scale = np.divide(X_load, max_X_load)

    # Return the scaled data with the correct orientation for 
    # temporal ICA, and the scaling factor.
    return X_scale, max_X_load


def apply_ICA(X_scale_tVox, num_components):
    """
    Apply ICA to the data - call loop_function_ICA_errors from
    functions_testing_pythonFastICA_errorsWarnings (has int __main__ etc).
    
    ICA will be iteratively applied to heightened tolerance and number of iterations
    for convergence to be reached. If convergence is not reached, an error/exception
    will be raised and the number of components should be increased.

    Arguments:
    X_scale_tVox = scaled and normalised data to apply ICA to. In the form
    tVox for temporal ICA. Shape (NumDyn,non_zero_masked_voxels). Or as known
    in scikit-learn FastICA: (n_samples, n_features)
    num_components = number of ICA components to be found.
    Returns:
    ica_tVox = ICA application.
    S_ica_tVox = estimated component time courses (i.e. sources) with the form 
    (n_samples, n_components).
    A_ica_tVox = estimated mixing matrix with the form (n_features, n_components).
    ica_tVox_mean = mean over the features (input data) of the form (n_features,). 
    The mean is calculated as it will be required when uncentring the data during
    the reconstruction of X. scikit-learn FastICA centres the data prior to the
    application of FastICA.
    iterations_used = the tolerance and maximum number of iterations used, for referring
    back to if required later. Shape of a tuple, (tolerance, max iterations).
    convergence_test = 0 for No; 1 for Yes - whether ICA converged.

    """
    # from functions_testing_pythonFastICA_errorsWarnings import *
    # ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean, iteration_loop = functions_testing_pythonFastICA_errorsWarnings(X_scale_tVox, num_components)
    ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean, iteration_loop, convergence_test  = \
        functions_testing_pythonFastICA_errorsWarnings.loop_function_ICA_errors_intmain(X_scale_tVox, num_components)
    return ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean, iteration_loop, convergence_test
    

def freq_spec(num_components, time_series):
    """
    Calculate the frequency spectrum for an input time course.
    (Multiple components input.)

    # # 2022/10/16 - array defined as dtype='complex_' to avoid excluding complex
    numbers.

    Arguments:
    num_components = number of component time series of which
    the frequency spectra are to be calculated.
    time_series = time series (multiple components) to have their
    frequency spectra calculated. Shape of S_tVox is (n_samples, n_components)
    which is (NumDyn, num_components). 
    Returns:
    freq_time_series = frequency spectra of the input time series. This
    will include positive and negative frequencies as it has shape
    (NumDyn, num_components). Hence, use halfTimepoints as an index to plot
    only the positive frequencies (freqPlot, freq_time_series[0:halfTimepoints]).

    """
    # Loop over the number of components and calculate the frequency spectra
    # separately for each component's time series.
    # The frequency calculation uses a 1D Fourier transform (fft).
    freq_time_series = np.zeros((np.shape(time_series)), dtype = 'complex_')
    for j in range(num_components):
        freq_time_series[:,j] = fft(time_series[:,j])
    return freq_time_series


def reshape_maps(size_echo_data, NumSlices, num_components, masks_reg_data, array_to_reshape):
    """
    Reshape array of masked values in the form of an array that can be plotted when
    mapping values.
    
    **INSTEAD** use reshape_maps_Fast from functions_Calculate_stats.

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
    # Create empty array which will become the component maps.
    reshape_maps_seg = np.zeros((size_echo_data[0], size_echo_data[1],NumSlices,num_components))
    # Reshape to be able to loop over all voxels (in all slices) for each component.
    reshape_maps_seg = np.reshape(reshape_maps_seg, (size_echo_data[0]*size_echo_data[1]*size_echo_data[3],num_components))
    # Loop over image voxels (full) and look for ones in mask to fill.
    # Use a counter, counter_c, to identify the lung voxel to be filled.
    for c in range(num_components):
        # Reset the counter for each component loop.
        counter_c = 0
        for j in range(size_echo_data[0]*size_echo_data[1]*NumSlices):
            if np.reshape(masks_reg_data, (np.shape(masks_reg_data)[0]*np.shape(masks_reg_data)[1]*np.shape(masks_reg_data)[2]))[j] == 1:
                reshape_maps_seg[j,c] = array_to_reshape[counter_c,c]
                counter_c = counter_c + 1
    # Now need to reshape back and permute (np.transpose) for plotting the
    # component maps which have the component in the first index.
    reshape_maps_seg = np.reshape(reshape_maps_seg, (size_echo_data[0], size_echo_data[1], size_echo_data[3], num_components))
    reshape_maps_seg = np.transpose(reshape_maps_seg, (3,0,1,2))
    return reshape_maps_seg


def PSE_map(data_map, GasSwitch, av_im_num):
    """
    Calculate air-O2 PSE maps (SI_oxy - SI_air) / SI_air.
    Average air and O2 maps over 5 images.

    Arguments:
    data_map = image/parameter map time series, with shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic number at which the gases are switched/cycled.
    av_im_num = number of images to average over to create the 
    baseline (air) and plateau (oxy) images.
    Returns:
    PercentageChange_image = percentage change image, relative enhancement ratio,
    equal to Oxy_mean / Air_mean. With shape (vox,vox,NumSlices).    
    PSE_image = PSE image, percentage signal enhancement (difference) image,
    equal to ([Oxy_mean - Air_mean] / Air_mean)*100. With shape (vox,vox,NumSlices).
    Difference_image = difference (subtraction) image, Oxy_mean - Air_mean.
    Air_map_cycle_1 = mean air image during the first cycle (vox,vox,NumSlices).
    Mean_Air_map = mean air image over all three cycles (vox,vox,NumSlices).
    Mean_Oxy_map = mean air image over all three cycles (vox,vox,NumSlices).
    
    """
    # Calculate the mean baseline and plateau images - for each cycle (three mean images for each gas).
    Air_map_cycle_1 = np.mean(data_map[:,:,GasSwitch-av_im_num:GasSwitch,:], axis=2)
    Air_map_cycle_2 = np.mean(data_map[:,:,GasSwitch*3-av_im_num:GasSwitch*3,:], axis=2)
    Air_map_cycle_3 = np.mean(data_map[:,:,GasSwitch*5-av_im_num:GasSwitch*5,:], axis=2)
    Oxy_map_cycle_1 = np.mean(data_map[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:], axis=2)
    Oxy_map_cycle_2 = np.mean(data_map[:,:,GasSwitch*4-av_im_num:GasSwitch*4,:], axis=2)
    Oxy_map_cycle_3 = np.mean(data_map[:,:,GasSwitch*6-av_im_num:GasSwitch*6,:], axis=2)

    # Mean of cycles
    Mean_Air_map = np.divide(Air_map_cycle_1 + Air_map_cycle_2 + Air_map_cycle_3, 3)
    Mean_Oxy_map = np.divide(Oxy_map_cycle_1 + Oxy_map_cycle_2 + Oxy_map_cycle_3, 3)

    # Calculate PSE
    PSE_image = np.multiply(np.divide((Mean_Oxy_map - Mean_Air_map), Mean_Air_map, out=np.zeros_like(Mean_Oxy_map - Mean_Air_map), where=Mean_Air_map!=0), 100)
    # Calculate PercentageChange_image
    PercentageChange_image = np.divide(Mean_Oxy_map, Mean_Air_map, out=np.zeros_like(Mean_Oxy_map - Mean_Air_map), where=Mean_Air_map!=0)
    # Calculate Difference_image
    Difference_image = np.array(Mean_Oxy_map - Mean_Air_map)

    return PercentageChange_image, PSE_image, Difference_image, Air_map_cycle_1, Mean_Air_map, Mean_Oxy_map


def PSE_map_old(masks_reg_data, load_data, GasSwitch, av_im_num):
    """
    Calculate air-O2 PSE maps (SI_oxy - SI_air) / SI_air.
    Average air and O2 maps over 5 images.

    Arguments:
    masks_reg_data = the masks of the registered data, with 
    shape (vox,vox,NumSlices).
    load_data = dynamic image series, with shape (vox,vox,NumDyn,NumSlices).
    GasSwitch = dynamic number at which the gases are switched/cycled.
    av_im_num = number of images to average over to create the 
    baseline (air) and plateau (oxy) images.
    Returns:
    PercentageChange_image = percentage change image, shape (vox,vox,NumSlices).
    PSE_image = PSE image, shape (vox,vox,NumSlices).

    """
    # Calculate the mean baseline and plateau images.
    mean_baseline = np.mean(load_data[:,:,GasSwitch-av_im_num:GasSwitch,:], axis=2)
    mean_oxy = np.mean(load_data[:,:,GasSwitch*2-av_im_num:GasSwitch*2,:], axis=2)
    # Multiply by the masks to create masked difference images.
    mean_baseline = np.multiply(mean_baseline, masks_reg_data)
    mean_oxy = np.multiply(mean_oxy, masks_reg_data)
    # Calculate the PSE and percentage change, taking possible zeros into account.
    # np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    PercentageChange_image = np.multiply(np.divide(mean_oxy, mean_baseline, out=np.zeros_like(mean_oxy - mean_baseline), where=mean_baseline!=0), 100)
    PSE_image = np.multiply(np.divide((mean_oxy - mean_baseline), mean_baseline, out=np.zeros_like(mean_oxy - mean_baseline), where=mean_baseline!=0), 100)
    return PercentageChange_image, PSE_image


def ICA_component_recon(OE_component_number, S_ica_tVox, A_ica_tVox):
    """
    Reconstruct MRI data using a specific ICA component. Simple reconstruction,
    not reversing the centring or scaling.

    Arguments:
    OE_component_number = component number of the OE component, will -1 to this
    number for use as an index.
    S_ica_tVox = array of S_ica_tVox inluding all components.
    A_ica_tVox = array of A_ica_tVox inluding all components.
    Returns:
    X_recon_OEonly = reconstructed OE component only.

    """
    # # Component's contribution to the time course from a particular voxel
    # Is S . A^T
    # Where S and A have all components that are not of interest set to zero
    S_ica_tVox_OEonly = S_ica_tVox[:,OE_component_number-1] # -1 as counting from zero
    A_ica_tVox_OEonly = A_ica_tVox[:,OE_component_number-1]

    # But S and A OEonly are (#,) need second dimension to be 1 to be able to dot product.
    S_ica_tVox_OEonly = np.reshape(S_ica_tVox_OEonly, ((np.shape(S_ica_tVox_OEonly)[0],1)))
    A_ica_tVox_OEonly = np.reshape(A_ica_tVox_OEonly, ((np.shape(A_ica_tVox_OEonly)[0],1)))
    X_recon_OEonly = np.dot(S_ica_tVox_OEonly, (A_ica_tVox_OEonly.T))
    return X_recon_OEonly



if __name__ == "__main__":
    # ...
    a = 1