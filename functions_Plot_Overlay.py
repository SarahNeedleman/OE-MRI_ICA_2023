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

# Functions to plot ICA components/PSE/parameter maps overlaid on SI images

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.colors as mplcolors

def plot_Map_Overlay_onSI_Scaling(data_full, map_plotting, dir_date, dir_subj, \
    NumSlices, echo_num, NumC, metric_chosen, \
    param_plotting, loaded_masks, \
    save_plots, showplot, dpi_save, \
    saving_details, which_plots, \
    clims_multi_lower_list, clims_multi_upper_list, \
    clims_lower_list, clims_upper_list, \
    RunNum_use):
    """
    To plot parameter maps (within e.g. lung mask or cardiac mask - loaded_masks) as an
    overlay on the SI image.
    **WITH** scaling of the colourmap and colourbar and/or setting of the upper and 
    lower limits of the colourmap and colourbar.

    Arguments:
    data_full = registered dynamic MRI SI data, with shape (vox,vox,NumSlices,NumDyn).
    map_plotting = map to be plotted as an overlay, with shape (vox,vox,NumSlices).
    dir_date = subject scanning date.
    dir_subj = subject ID.
    NumSlices = number of MRI slices acquired.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    NumC = number of components to go in *title*.
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    param_plotting = string describing what is being plotted, to form part of the
    figure name when saved --> identify if plotting:
    PSE (SI and/or recon OE) overlay, and/or the OE component.
    # use in list plotting_param_name_list = ['Recon_OE_PSE', 'SI_PSE', 'OEc'] for 
    the reconstructed OE component // MRI SI PSE maps, and the OE component map.
    loaded_masks = masks (cardiac or lung) for the subject, with shape (vox,vox,NumSlices).
    save_plots = 0 if No; 1 if Yes - whether to save plots.
    showplot = 0 if No; 1 if Yes - whether to show plots.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    which_plots = [plot_PSE_timeseries, plot_SI_timeseries, plot_PSE_map, plot_OE_ICA, plot_ICA_components, \
    plot_all_component_TimeFreq, plot_all_component_FreqScaled, plot_SI_images, plot_PSE_overlay, \
    plot_OE_component_overlay, plot_clims_set, plot_clims_multi] - 0/1 in list for Yes/No
    to plot some of the plots contained within this function.
    #
    clims_multi_lower_list, clims_multi_upper_list = a list of values to multiply
    the upper/lower limits by (for use when plotting PSE). The list values will be 
    looped over. e.g. [0.8, 0.5] and [0.25, 0.2] = clims_multi_lower_list, clims_multi_upper_list.
    - If list is empty, [], these will not be plotted.
    #
    clims_lower_list, clims_upper_list = a list of values to set/threshold the colourmaps
    and colourbars (for use when plotting PSE). The list values will be 
    looped over. e.g. [30, 25, 20] and [5, 5, 5] = clims_lower_list, clims_upper_list.
    - If list is empty, [], these will not be plotted.
    #
    Returns:
    None - plots created/saved as required.

    """
    # Base image - take first dynamic image during air-breathing.
    base_air = data_full[:,:,0,:]

    # BWR colourmap
    cm = plt.get_cmap('bwr')

    # # ---------------------
    # # 1) MRI SI images are currently greyscale single channel/single number per voxel with 
    # increasing intensity/whiteness).
    # Need to convert to float32 before RGB conversion.

    # Create empty array to store the 3-channel RGB maps of the form (shape_voxels, RGB (3), NumSlices).
    base_air_RGB = np.zeros((np.shape(base_air)[0],np.shape(base_air)[1], 3, np.shape(base_air)[2]))
    # Rescale the images so max image intensity is 1, and convert to float32 before conversion to RGB.
    base_air_scale_factor = np.max(np.abs(base_air))
    base_air_convert = np.float32(np.divide(base_air, base_air_scale_factor))

    # Use cv2...
    # cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.cvtColor converts an image from one colour space to another.
    # cv2.COLOUR_GRAY2RGB converts from grey (image intensity) to RGB colour image.
    # So the input single channel greyscale image is converted to a multichannel (3 channel)
    # RGB image.
    for j in range(NumSlices):
        base_air_RGB[:,:,:,j] = cv2.cvtColor(base_air_convert[:,:,j], cv2.COLOR_GRAY2RGB)

    # # Also mask the air image - using an inverted mask.
    # Invert cardiac mask by -1 and then take abs
    inverted_mask_reg_allSlices = np.abs(loaded_masks - 1)
    base_air_RGB_masked = np.multiply(np.transpose(base_air_RGB, (2,0,1,3)), inverted_mask_reg_allSlices)
    # Permute (transpose) back
    base_air_RGB_masked = np.transpose(base_air_RGB_masked, (1,2,0,3))

    # # ---------------------
    # # 2) Convert map_plotting to RGB using the bwr colourmap scale of interest for plotting.
    # SI RGB values run from 0.0 to 1.0 as image intensities are positive and were scaled.
    # *However*, the OE component map values may be negative.
    # Need positive values for when adding/RGB/bwr plotting.
    # Therefore, shift the OE component map values to all be positive. Shift by the maximum
    # *absolute* array value to ensure cbar plots about the background (white).
    # Also need to scale the shifted array to have a maximum value of 1.0.
    min_before_shift = np.max([-np.min(map_plotting), np.max(map_plotting)])
    map_plotting_shift = np.zeros((np.shape(map_plotting)[0], np.shape(map_plotting)[1]))
    map_plotting_shift = np.abs(min_before_shift) + map_plotting
    # Scaling (for maximum of 1.0).
    OE_map_scale_factor = np.max(np.abs(map_plotting_shift))
    map_plotting_shift_scale = np.divide(map_plotting_shift, OE_map_scale_factor)
    # ^^ As cmap (used later) will map *normalised* data values to RGBA colours.


    # FOR SCALING:
    # 1. Alter the **colourmap** to have a specific plot range.
    # 2. Alter the **colourbar** to have bwr scaling as desired, so that it does not
    # need to be symmetrical, but still maintains 0 as a central value (white).


    # # CLIMS_MULTI
    # Plot colourbar limits set, multiplied, or both versions.
    if which_plots[11] == 1:
        for j in range(len(clims_multi_lower_list)):
            clims_multi_lower = clims_multi_lower_list[j]
            clims_multi_upper = clims_multi_upper_list[j]
            # Calculate the lower and upper threshold values using
            # the multipliers.
            clims_value_lower = np.multiply(clims_multi_lower, np.min(map_plotting))
            clims_value_upper = np.multiply(clims_multi_upper, np.max(map_plotting))

            # Set corner values to the upper and lower limits.
            map_plotting_array = np.array(map_plotting)
            map_plotting_array[0,0] = clims_value_lower
            map_plotting_array[0,1] = clims_value_upper

            # Replace values beyond the threshold with the threshold values.
            # Boolean array of where greater than or less than, to use for setting the
            # map values to the threshold limit values.
            array_lower = map_plotting_array < clims_value_lower
            array_upper = map_plotting_array > clims_value_upper
            # Set values to upper/lower as appropriate
            map_plotting_clims_multi = np.array(map_plotting_array)
            map_plotting_clims_multi[np.multiply(array_lower,1) == 1] = clims_value_lower
            map_plotting_clims_multi[np.multiply(array_upper,1) == 1] = clims_value_upper

            # Shift and scale the new array that has been thresholded.
            min_before_shift_V2 = np.max([-np.min(map_plotting_clims_multi), np.max(map_plotting_clims_multi)])
            map_plotting_shift_V2 = np.abs(min_before_shift_V2) + map_plotting_clims_multi
            OE_map_scale_factor_V2 = np.max(np.abs(map_plotting_shift_V2))
            map_plotting_shift_scale_V2 = np.divide(map_plotting_shift_V2, OE_map_scale_factor_V2)



            # # # 1. Alter the **colourmap** to have a specific plot range.
            # use vcenter to define centre and the upper and lower limits of the colourbar.
            # Use TwoSlopeNorm for this.
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.TwoSlopeNorm.html#matplotlib.colors.TwoSlopeNorm
            # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html

            # The central (vcenter) value of the *plotted* colormap can be found from 
            # the shifted AND scaled single-channel image (before it is transformed
            # to a multi-channel RGBA image).
            # The minimum and maximum are 0.0 and 1.0 due to the scaling for PSE,
            # and close to for the OE component maps.
            # The central value was originally zero, but the image was shifted and
            # then scaled. Therefore, the new central value is the shift value that has
            # been scaled (i.e. divided by the scaling factor).
            zero_shiftANDscale_V2 = np.divide(min_before_shift_V2, OE_map_scale_factor_V2)
            # Use TwoSlopeNorm to create the colourmap scaling for the image over these limits.
            if zero_shiftANDscale_V2 == 1:
                # BUT if maximum array value = 0, zero_shiftANDscale_V2 = vmax = 1. If this is true...
                # e.g. set vmax to 1.0000001.
                offset_scaleANDshift_V2 = mplcolors.TwoSlopeNorm(vmin=0, vcenter=zero_shiftANDscale_V2, vmax=1.00000000001)
            else:
                offset_scaleANDshift_V2 = mplcolors.TwoSlopeNorm(vmin=0, vcenter=zero_shiftANDscale_V2, vmax=1)
            
            # Generate the map/transform the shifted&scaled map to the new colourmap range.
            colored_image_newMethod_V2 = offset_scaleANDshift_V2(map_plotting_shift_scale_V2)

            # The resulting image is single-channel, need to use cm to create
            # the RGBA image.
            colored_image_V2 = cm(colored_image_newMethod_V2)
            colored_image_V2 = np.transpose(colored_image_V2, (0,1,3,2))
            # Mask and add to the SI RGB image.
            colored_image_V2_masked = np.multiply(np.transpose(colored_image_V2, (2,0,1,3)), loaded_masks)
            colored_image_V2_masked = np.transpose(colored_image_V2_masked, (1,2,0,3))
            # Add the two masked images to generate the overlay (bwr colored_image exluding alpha, hence :3).
            RGB_map_overlay_newMethod_V2 = base_air_RGB_masked + colored_image_V2_masked[:,:,:3,:]



            # # # 2. Alter the **colourbar** to have bwr scaling as desired, so that it does not
            # # # need to be symmetrical, but still maintains 0 as a central value (white).
            # Use TwoSlopeNorm again, but this time for creating the colourbar scale that will
            # be included in the plot. This is the colourbar relating to the original image,
            # which can be edited without altering the plotted range of the colourmap.
            # Then use in imshow with norm=
            if clims_value_upper == 0:
                # e.g. set vmax to 0.0000001.
                offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=clims_value_lower, vcenter=0, vmax=0.0000000001)
            else:
                offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=clims_value_lower, vcenter=0, vmax=clims_value_upper)



            # # # Perform plotting
            # Repeat plotting of first figure, sometimes doesn't work properly.
            fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
            plt.close(fig1)

            # subplots_num_row = 2 # Assume. These are the number of *rows* of subplots, i.e. assume two rows of subplots.
            # subplots_num_column = np.divmod(NumSlices, 2); subplots_num_column = subplots_num_column[0]
            subplots_num_column = 2 # Assume. These are the number of subplots *per* column - assume 2 subplots per column, i.e. 2 rows.
            subplots_num_row = np.int32(np.ceil(np.divide(NumSlices, 2))) #; subplots_num_row = subplots_num_row[0]


            # Single figure
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
                ims[slice_plot] = axs[slice_plot].imshow(RGB_map_overlay_newMethod_V2[:,:,:,slice_plot], 'bwr', norm=offset_scaleANDshift_V2_new_cbar)
                axs[slice_plot].invert_yaxis(); fig1.colorbar(ims[slice_plot], ax=axs[slice_plot])
                # ims[slice_plot].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[slice_plot,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[slice_plot,:,:,slice_plot])))
                #
            # Save figure if required.
            if saving_details[0] == 1:
                # As V2 has a restriction on the number of components, include G21 in Figure saving name.
                if RunNum_use > 1:
                    fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + \
                        metric_chosen[0] + '_' + str(NumC) + '_Overlay_' + \
                        param_plotting + '_' + '_ClimsMult_' + str(clims_multi_lower) + '_' + str(clims_multi_upper) + '.png', dpi=dpi_save, bbox_inches='tight')
                else:
                    fig1.savefig(saving_details[1] + saving_details[2] + '_' + \
                        metric_chosen[0] + '_' + str(NumC) + '_Overlay_' + \
                        param_plotting + '_' + '_ClimsMult_' + str(clims_multi_lower) + '_' + str(clims_multi_upper) + '.png', dpi=dpi_save, bbox_inches='tight')


            if showplot == 0:
                plt.close('all')
            else:
                plt.show()


    # # CLIMS_UPPER and LOWER
    if which_plots[10] == 1:
        # if which_plots[11] == 1:
        # # I.e. clims_set
        # # CLIMS_UPPER and LOWER
        # if len(clims_lower_list) > 0:
        for j in range(len(clims_lower_list)):
            clims_lower = clims_lower_list[j]
            clims_upper = clims_upper_list[j]

            # Set corner values to the upper and lower limits.
            map_plotting_array = np.array(map_plotting)
            map_plotting_array[0,0] = -clims_lower
            map_plotting_array[0,1] = clims_upper


            # Use the supplied values to set the upper and lower thresholds.
            array_lower = map_plotting_array < -clims_lower
            array_upper = map_plotting_array > clims_upper
            # Replace values beyond the threshold with the threshold values.
            # Boolean array of where greater than or less than, to use for setting the
            # map values to the threshold limit values.
            # Set values to upper/lower as appropriate
            map_plotting_clims_multi = np.array(map_plotting_array)
            # map_plotting_clims_multi[np.multiply(array_lower,1) == 1] = clims_value_lower
            # map_plotting_clims_multi[np.multiply(array_upper,1) == 1] = clims_value_upper
            map_plotting_clims_multi[np.multiply(array_lower,1) == 1] = -clims_lower
            map_plotting_clims_multi[np.multiply(array_upper,1) == 1] = clims_upper

            # Shift and scale the new array that has been thresholded.
            min_before_shift_V2 = np.max([-np.min(map_plotting_clims_multi), np.max(map_plotting_clims_multi)])
            map_plotting_shift_V2 = np.abs(min_before_shift_V2) + map_plotting_clims_multi
            OE_map_scale_factor_V2 = np.max(np.abs(map_plotting_shift_V2))
            map_plotting_shift_scale_V2 = np.divide(map_plotting_shift_V2, OE_map_scale_factor_V2)



            # # # 1. Alter the **colourmap** to have a specific plot range.
            # use vcenter to define centre and the upper and lower limits of the colourbar.
            # Use TwoSlopeNorm for this.
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.TwoSlopeNorm.html#matplotlib.colors.TwoSlopeNorm
            # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html

            # The central (vcenter) value of the *plotted* colormap can be found from 
            # the shifted AND scaled single-channel image (before it is transformed
            # to a multi-channel RGBA image).
            # The minimum and maximum are 0.0 and 1.0 due to the scaling for PSE,
            # and close to for the OE component maps.
            # The central value was originally zero, but the image was shifted and
            # then scaled. Therefore, the new central value is the shift value that has
            # been scaled (i.e. divided by the scaling factor).
            zero_shiftANDscale_V2 = np.divide(min_before_shift_V2, OE_map_scale_factor_V2)
            # Use TwoSlopeNorm to create the colourmap scaling for the image over these limits.
            if zero_shiftANDscale_V2 == 1:
                # BUT if maximum array value = 0, zero_shiftANDscale_V2 = vmax = 1. If this is true...
                # e.g. set vmax to 1.0000001.
                offset_scaleANDshift_V2 = mplcolors.TwoSlopeNorm(vmin=0, vcenter=zero_shiftANDscale_V2, vmax=1.00000000001)
            else:
                offset_scaleANDshift_V2 = mplcolors.TwoSlopeNorm(vmin=0, vcenter=zero_shiftANDscale_V2, vmax=1)

            # Generate the map/transform the shifted&scaled map to the new colourmap range.
            colored_image_newMethod_V2 = offset_scaleANDshift_V2(map_plotting_shift_scale_V2)

            # The resulting image is single-channel, need to use cm to create
            # the RGBA image.
            colored_image_V2 = cm(colored_image_newMethod_V2)
            colored_image_V2 = np.transpose(colored_image_V2, (0,1,3,2))
            # Mask and add to the SI RGB image.
            colored_image_V2_masked = np.multiply(np.transpose(colored_image_V2, (2,0,1,3)), loaded_masks)
            colored_image_V2_masked = np.transpose(colored_image_V2_masked, (1,2,0,3))
            # Add the two masked images to generate the overlay (bwr colored_image exluding alpha, hence :3).
            RGB_map_overlay_newMethod_V2 = base_air_RGB_masked + colored_image_V2_masked[:,:,:3,:]



            # # # 2. Alter the **colourbar** to have bwr scaling as desired, so that it does not
            # # # need to be symmetrical, but still maintains 0 as a central value (white).
            # Use TwoSlopeNorm again, but this time for creating the colourbar scale that will
            # be included in the plot. This is the colourbar relating to the original image,
            # which can be edited without altering the plotted range of the colourmap.
            # Then use in imshow with norm=
            if clims_upper == 0:
                # e.g. set vmax to 0.0000001.
                offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=-clims_lower, vcenter=0, vmax=0.0000000001)
            else:
                offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=-clims_lower, vcenter=0, vmax=clims_upper)



            # # # Perform plotting
            # Repeat plotting of first figure, sometimes doesn't work properly.
            fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
            plt.close(fig1)

            # subplots_num_row = 2 # Assume. These are the number of *rows* of subplots, i.e. assume two rows of subplots.
            # subplots_num_column = np.divmod(NumSlices, 2); subplots_num_column = subplots_num_column[0]
            subplots_num_column = 2 # Assume. These are the number of subplots *per* column - assume 2 subplots per column, i.e. 2 rows.
            subplots_num_row = np.int32(np.ceil(np.divide(NumSlices, 2))) #; subplots_num_row = subplots_num_row[0]


            # Single figure
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
                ims[slice_plot] = axs[slice_plot].imshow(RGB_map_overlay_newMethod_V2[:,:,:,slice_plot], 'bwr', norm=offset_scaleANDshift_V2_new_cbar)
                axs[slice_plot].invert_yaxis(); fig1.colorbar(ims[slice_plot], ax=axs[slice_plot])
                # ims[slice_plot].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[slice_plot,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[slice_plot,:,:,slice_plot])))
                #



            # Save figure if required.
            if saving_details[0] == 1:
                # As V2 has a restriction on the number of components, include G21 in Figure saving name.
                if RunNum_use > 1:
                    fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + \
                        metric_chosen[0] + '_' + str(NumC) + '_Overlay_' + \
                        param_plotting + '_' + '_ClimsSet_' + str(clims_lower) + '_' + str(clims_upper) + '.png', dpi=dpi_save, bbox_inches='tight')
                else:
                    fig1.savefig(saving_details[1] + saving_details[2] + '_' + \
                        metric_chosen[0] + '_' + str(NumC) + '_Overlay_' + \
                        param_plotting + '_' + '_ClimsSet_' + str(clims_lower) + '_' + str(clims_upper) + '.png', dpi=dpi_save, bbox_inches='tight')

            if showplot == 0:
                plt.close('all')
            else:
                plt.show()

    return



# Separate function for plotting OE//ICA components overlay - as require symmetrical colourbar.
def plot_Map_Overlay_onSI_Scaling_ComponentOnly(data_full, map_plotting, dir_date, dir_subj, \
    NumSlices, echo_num, NumC, metric_chosen, \
    param_plotting, loaded_masks, \
    save_plots, showplot, dpi_save, \
    saving_details, \
    clims_multi_componentOverlay_list, \
    RunNum_use):
    """
    To plot parameter maps (within e.g. lung mask or cardiac mask - loaded_masks) as an
    overlay on the SI image.
    ***FOR COMPONENT OVERLAY ONLY --> only use scaling of the colourmaps/colourbar
    and use a symmetrical colourbar scale for the ICA components.

    Arguments:
    data_full = registered dynamic MRI SI data, with shape (vox,vox,NumSlices,NumDyn).
    map_plotting = map to be plotted as an overlay, with shape (vox,vox,NumSlices).
    dir_date = subject scanning date.
    dir_subj = subject ID.
    NumSlices = number of MRI slices acquired.
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    NumC = number of components to go in *title*.
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    param_plotting = string describing what is being plotted, to form part of the
    figure name when saved --> identify if plotting:
    PSE (SI and/or recon OE) overlay, and/or the OE component.
    # use in list plotting_param_name_list = ['Recon_OE_PSE', 'SI_PSE', 'OEc'] for 
    the reconstructed OE component // MRI SI PSE maps, and the OE component map.
    loaded_masks = masks (cardiac or lung) for the subject, with shape (vox,vox,NumSlices).
    save_plots = 0 if No; 1 if Yes - whether to save plots.
    showplot = 0 if No; 1 if Yes - whether to show plots.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    #
    clims_multi_componentOverlay_list = a list of values to multiply *both* the upper/lower 
    limits by (for use when plotting the OE//ICA components). The list values will be 
    looped over. e.g. [0.7, 0.4].
    - If list is empty, [], these will not be plotted.
    Returns:
    None - plots created/saved as required.

    """
    # Base image - take first dynamic image during air-breathing.
    base_air = data_full[:,:,0,:]

    # BWR colourmap
    cm = plt.get_cmap('bwr')

    # # ---------------------
    # # 1) MRI SI images are currently greyscale single channel/single number per voxel with 
    # increasing intensity/whiteness).
    # Need to convert to float32 before RGB conversion.

    # Create empty array to store the 3-channel RGB maps of the form (shape_voxels, RGB (3), NumSlices).
    base_air_RGB = np.zeros((np.shape(base_air)[0],np.shape(base_air)[1], 3, np.shape(base_air)[2]))
    # Rescale the images so max image intensity is 1, and convert to float32 before conversion to RGB.
    base_air_scale_factor = np.max(np.abs(base_air))
    base_air_convert = np.float32(np.divide(base_air, base_air_scale_factor))

    # Use cv2...
    # cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.cvtColor converts an image from one colour space to another.
    # cv2.COLOUR_GRAY2RGB converts from grey (image intensity) to RGB colour image.
    # So the input single channel greyscale image is converted to a multichannel (3 channel)
    # RGB image.
    for j in range(NumSlices):
        base_air_RGB[:,:,:,j] = cv2.cvtColor(base_air_convert[:,:,j], cv2.COLOR_GRAY2RGB)

    # # Also mask the air image - using an inverted mask.
    # Invert cardiac mask by -1 and then take abs
    inverted_mask_reg_allSlices = np.abs(loaded_masks - 1)
    base_air_RGB_masked = np.multiply(np.transpose(base_air_RGB, (2,0,1,3)), inverted_mask_reg_allSlices)
    # Permute (transpose) back
    base_air_RGB_masked = np.transpose(base_air_RGB_masked, (1,2,0,3))

    # # ---------------------
    # # 2) Convert map_plotting to RGB using the bwr colourmap scale of interest for plotting.
    # SI RGB values run from 0.0 to 1.0 as image intensities are positive and were scaled.
    # *However*, the OE component map values may be negative.
    # Need positive values for when adding/RGB/bwr plotting.
    # Therefore, shift the OE component map values to all be positive. Shift by the maximum
    # *absolute* array value to ensure cbar plots about the background (white).
    # Also need to scale the shifted array to have a maximum value of 1.0.
    min_before_shift = np.max([-np.min(map_plotting), np.max(map_plotting)])
    map_plotting_shift = np.zeros((np.shape(map_plotting)[0], np.shape(map_plotting)[1]))
    map_plotting_shift = np.abs(min_before_shift) + map_plotting
    # Scaling (for maximum of 1.0).
    OE_map_scale_factor = np.max(np.abs(map_plotting_shift))
    map_plotting_shift_scale = np.divide(map_plotting_shift, OE_map_scale_factor)
    # ^^ As cmap (used later) will map *normalised* data values to RGBA colours.

    # FOR SCALING:
    # 1. Alter the **colourmap** to have a specific plot range.
    # 2. Alter the **colourbar** to have bwr scaling as desired, so that it does not
    # need to be symmetrical, but still maintains 0 as a central value (white).

    # # CLIMS_MULTI
    if len(clims_multi_componentOverlay_list) > 0:
        for j in range(len(clims_multi_componentOverlay_list)):
            clims_multi_lower = clims_multi_componentOverlay_list[j]
            clims_multi_upper = np.float32(clims_multi_lower)

            # Calculate the lower and upper threshold values using
            # the multipliers.
            clims_value_lower = -np.multiply(clims_multi_lower, np.max(np.abs(map_plotting)))
            clims_value_upper = np.multiply(clims_multi_upper, np.max(np.abs(map_plotting)))

            # Set corner values to the upper and lower limits.
            map_plotting_array = np.array(map_plotting)
            map_plotting_array[0,0] = clims_value_lower
            map_plotting_array[0,1] = clims_value_upper

            # Replace values beyond the threshold with the threshold values.
            # Boolean array of where greater than or less than, to use for setting the
            # map values to the threshold limit values.
            array_lower = map_plotting_array < clims_value_lower
            array_upper = map_plotting_array > clims_value_upper
            # Set values to upper/lower as appropriate
            map_plotting_clims_multi = np.array(map_plotting_array)
            map_plotting_clims_multi[np.multiply(array_lower,1) == 1] = clims_value_lower
            map_plotting_clims_multi[np.multiply(array_upper,1) == 1] = clims_value_upper

            # Shift and scale the new array that has been thresholded.
            min_before_shift_V2 = np.max([-np.min(map_plotting_clims_multi), np.max(map_plotting_clims_multi)])
            map_plotting_shift_V2 = np.abs(min_before_shift_V2) + map_plotting_clims_multi
            OE_map_scale_factor_V2 = np.max(np.abs(map_plotting_shift_V2))
            map_plotting_shift_scale_V2 = np.divide(map_plotting_shift_V2, OE_map_scale_factor_V2)



            # # # 1. Alter the **colourmap** to have a specific plot range.
            # use vcenter to define centre and the upper and lower limits of the colourbar.
            # Use TwoSlopeNorm for this.
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.TwoSlopeNorm.html#matplotlib.colors.TwoSlopeNorm
            # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html

            # The central (vcenter) value of the *plotted* colormap can be found from 
            # the shifted AND scaled single-channel image (before it is transformed
            # to a multi-channel RGBA image).
            # The minimum and maximum are 0.0 and 1.0 due to the scaling for PSE,
            # and close to for the OE component maps.
            # The central value was originally zero, but the image was shifted and
            # then scaled. Therefore, the new central value is the shift value that has
            # been scaled (i.e. divided by the scaling factor).
            zero_shiftANDscale_V2 = np.divide(min_before_shift_V2, OE_map_scale_factor_V2)
            # Use TwoSlopeNorm to create the colourmap scaling for the image over these limits.
            if zero_shiftANDscale_V2 == 1:
                # BUT if maximum array value = 0, zero_shiftANDscale_V2 = vmax = 1. If this is true...
                # e.g. set vmax to 1.0000001.
                offset_scaleANDshift_V2 = mplcolors.TwoSlopeNorm(vmin=0, vcenter=zero_shiftANDscale_V2, vmax=1.00000000001)
            else:
                offset_scaleANDshift_V2 = mplcolors.TwoSlopeNorm(vmin=0, vcenter=zero_shiftANDscale_V2, vmax=1)

            # Generate the map/transform the shifted&scaled map to the new colourmap range.
            colored_image_newMethod_V2 = offset_scaleANDshift_V2(map_plotting_shift_scale_V2)

            # The resulting image is single-channel, need to use cm to create
            # the RGBA image.
            colored_image_V2 = cm(colored_image_newMethod_V2)
            colored_image_V2 = np.transpose(colored_image_V2, (0,1,3,2))
            # Mask and add to the SI RGB image.
            colored_image_V2_masked = np.multiply(np.transpose(colored_image_V2, (2,0,1,3)), loaded_masks)
            colored_image_V2_masked = np.transpose(colored_image_V2_masked, (1,2,0,3))
            # Add the two masked images to generate the overlay (bwr colored_image exluding alpha, hence :3).
            RGB_map_overlay_newMethod_V2 = base_air_RGB_masked + colored_image_V2_masked[:,:,:3,:]



            # # # 2. Alter the **colourbar** to have bwr scaling as desired, so that it does not
            # # # need to be symmetrical, but still maintains 0 as a central value (white).
            # Use TwoSlopeNorm again, but this time for creating the colourbar scale that will
            # be included in the plot. This is the colourbar relating to the original image,
            # which can be edited without altering the plotted range of the colourmap.
            # Then use in imshow with norm=
            # offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=clims_value_lower, vcenter=0, vmax=clims_value_upper)
            if clims_value_upper == 0:
                # e.g. set vmax to 0.0000001.
                offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=clims_value_lower, vcenter=0, vmax=0.0000000001)
            else:
                offset_scaleANDshift_V2_new_cbar = mplcolors.TwoSlopeNorm(vmin=clims_value_lower, vcenter=0, vmax=clims_value_upper)




            # # # Perform plotting
            # Repeat plotting of first figure, sometimes doesn't work properly.
            fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
            plt.close(fig1)

            # subplots_num_row = 2 # Assume. These are the number of *rows* of subplots, i.e. assume two rows of subplots.
            # subplots_num_column = np.divmod(NumSlices, 2); subplots_num_column = subplots_num_column[0]
            subplots_num_column = 2 # Assume. These are the number of subplots *per* column - assume 2 subplots per column, i.e. 2 rows.
            subplots_num_row = np.int32(np.ceil(np.divide(NumSlices, 2))) #; subplots_num_row = subplots_num_row[0]



            # Single figure
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
                ims[slice_plot] = axs[slice_plot].imshow(RGB_map_overlay_newMethod_V2[:,:,:,slice_plot], 'bwr', norm=offset_scaleANDshift_V2_new_cbar)
                axs[slice_plot].invert_yaxis(); fig1.colorbar(ims[slice_plot], ax=axs[slice_plot])
                # ims[slice_plot].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[slice_plot,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[slice_plot,:,:,slice_plot])))
                #


            # Save figure if required.
            if saving_details[0] == 1:
                # As V2 has a restriction on the number of components, include G21 in Figure saving name.
                if RunNum_use > 1:
                    fig1.savefig(saving_details[1] + saving_details[2] + '_Run' + str(RunNum_use) + '_' + \
                        metric_chosen[0] +  '_' + NumC + '_Overlay_' + \
                        param_plotting + '_' + '_ClimsMult_' + str(clims_multi_lower) + '.png', dpi=dpi_save, bbox_inches='tight')
                else:
                    fig1.savefig(saving_details[1] + saving_details[2] + '_' + \
                        metric_chosen[0] +  '_' + NumC + '_Overlay_' + \
                        param_plotting + '_' + '_ClimsMult_' + str(clims_multi_lower) + '.png', dpi=dpi_save, bbox_inches='tight')

            if showplot == 0:
                plt.close('all')
            else:
                plt.show()


    return













