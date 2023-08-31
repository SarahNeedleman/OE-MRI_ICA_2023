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

# Functions to apply ICA using scikit-learn.

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings
import Sarah_ICA
import functions_Calculate_stats
import functions_PlottingICAetc_testing_Subplots
import functions_LoopSave_ICA_HDF5

def run_ICA_loop(X_scale_tVox, num_components, max_iter_set, tol_set):
    """
    Function to perform the ICA calculation using scikit-learn FastICA
    and return the ICA component timeseries, component maps (mixing matrix) 
    and the centring (for zero-mean input to ICA) performed by ICA.

    Arguments:
    X_scale_tVox = scaled and normalised data to apply ICA to. In the form
    tVox for temporal ICA. Shape (NumDyn,non_zero_masked_voxels). Or as known
    in scikit-learn FastICA: (n_samples, n_features)
    num_components = number of ICA components to be found.
    max_iter_set = maximum number of iterations to be performed by FastICA, if
    max_iter_set is reached before the ICA algorithm converges, a warning will
    be produced by scikit-learn FastICA.
    tol_set = tolerance of the convergence to be reached by FastICA, if
    tol_set is not reached before the ICA algorithm converges, a warning will
    be produced by scikit-learn FastICA.
    Returns:
    ica_tVox = ICA application.
    S_ica_tVox = estimated component time courses (i.e. sources) with the form 
    (n_samples, n_components).
    A_ica_tVox = estimated mixing matrix with the form (n_features, n_components).
    ica_tVox_mean = mean over the features (input data) of the form (n_features,). 
    The mean is calculated as it will be required when uncentring the data during
    the reconstruction of X. scikit-learn FastICA centres the data prior to the
    application of FastICA.
    
    """
    ica_tVox = FastICA(n_components=num_components, whiten=True, max_iter=max_iter_set, tol=tol_set)
    # Calculate the component time series.
    S_ica_tVox = ica_tVox.fit(X_scale_tVox).transform(X_scale_tVox)
    # Estimate the mixing matrix (component maps).
    A_ica_tVox = ica_tVox.mixing_
    # Calculate the mean value of the input measured data X. For use
    # later when uncentring the data during data reconstruction.
    ica_tVox_mean = ica_tVox.mean_           
    return ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean


def loop_function_ICA_errors_intmain(X_scale_tVox, num_components):
    """
    Function to enable ICA to be performed with increasing levels of 
    tolerance and number of iterations. If the ICA algorithm does not 
    converge, an error does *NOT* stop the function (allowing it to 
    be used in a loop). Instead, details of convergence, or lack of, are
    returned to the function call to identify the converged cases of ICA.

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
    iteration_loop = iteration settings in the form (tol_set, max_iter_set).
    tol_set is the tolerance to be reached by the FastICA algorithm for convergence
    to occur. max_iter_set is the maximum number of iterations to be performed by the 
    FastICA algorithm, otherwise convergence is not reached. In the case that convergence
    is not reached, iteration_loop will equal the iteration settings of the final 
    attempted ICA run.
    converged_test = 0 if No; 1 if Yes - whether the ICA algorithm has converged. 
    This relates to whether a warning was thrown by FastICA for when convergence does
    not occur.

    """
    # Set convergence test to zero, and the desired outputs to None. If the ICA 
    # algorithm reaches convergence, these parameters will be reassigned to their
    # converged/('best') estimated ICA values.
    converged_test = 0
    ica_tVox = None; S_ica_tVox = None; A_ica_tVox = None; ica_tVox_mean = None
    # Loop over increasing convergence levels to attempt to reach convergence of the 
    # FastICA algorithm. Set warnings to temporarily avoid terminating the loop, to 
    # that the increasing convergence levels/iterations settings can be applied.
    with warnings.catch_warnings(record=True) as w:
        # Iteration settings - these have been chosen from experiments with
        # ICA applied to OE-MRI SI MRI images and from scikit-learn FastICA.
        iteration_settings_list = [
            (1e-09, 200),
            (1e-09, 500),
            (1e-09, 5000),
            (1e-10, 5000),
            (1e-09, 50000),
            (1e-10, 50000),
            (1e-11, 50000)
            ]
        iteration_settings = []
        iteration_settings.extend(iteration_settings_list)
        iteration_settings.extend(iteration_settings_list)
        iteration_settings.extend(iteration_settings_list)
        iteration_settings.extend(iteration_settings_list)
        iteration_settings.extend(iteration_settings_list)
        warnings.simplefilter("error") # error for exceptions to be raised.
        
        # Loop over the iteration settings - only continue with the iterations 
        # if an error/warning is raised (i.e. convergence has not been reached).
        # Otherwise, convergence has been reached and the components have been 
        # estimated by ICA and so the loop can be broken out of.
        for i in range(np.shape(iteration_settings)[0]):
            iteration_loop = iteration_settings[i]
            try:
                # # And within this... RuntimeWarning from ICA...
                try:
                    ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean = \
                        run_ICA_loop(X_scale_tVox, num_components, iteration_loop[1], iteration_loop[0])
                except RuntimeWarning:
                    try:
                        ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean = \
                            run_ICA_loop(X_scale_tVox, num_components, iteration_loop[1], iteration_loop[0])
                    except RuntimeWarning:
                        ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean = \
                            run_ICA_loop(X_scale_tVox, num_components, iteration_loop[1], iteration_loop[0])
            # except SomeSpecificException:
            except ConvergenceWarning:
                # If a warning is thrown, continue with the loop as convergence
                # has not been reached. The next iteration of the loop will use
                # heightened iteration settings to try to reach convergence.
                continue

            # If convergence warning has not been raised, convergence has been reached and so 
            # the iteration loop can be broken out of and converged_test set to 1 to signify
            # convergence.
            converged_test = 1
            break

    return ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean, iteration_loop, converged_test


def loopICA_HDF5save_plot_order(num_components_list, ordering_method_list, X_scale_tVox, size_echo_data, NumSlices, \
    masks_reg_data, halfTimepoints, freqPlot, plot_time_value, \
    NumDyn, TempRes, GasSwitch, data_scaling, \
    hdf5_loc, subject, ICA_applied, mask_applied, echo_num, \
    saving_details, showplot, plot_cmap=0, plot_all=0):
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
    Returns:
    RunNum, RunNum_2 = RunNumber of the ICA analysis for the particular component.
    May be used for plotting saving names and saving components/details/metrics
    to the HDF5 file.
    Also used to check later.
    
    """
    RunNum = 0; RunNum_2 = 0

    # Loop over all components in the components list supplied. For reach loop,
    # apply ICA to the input data, and optionally plot and save the results.
    for k in num_components_list:
        # Perform ICA.
        ica_tVox, S_ica_tVox, A_ica_tVox, ica_tVox_mean, \
            iteration_loop, converged_test = Sarah_ICA.apply_ICA(X_scale_tVox, k)
        
        # Save original components and details of whether the ICA algorithm converged.
        RunNum = functions_LoopSave_ICA_HDF5.hdf5_save_ICA(hdf5_loc, subject, k, mask_applied, ICA_applied, \
            converged_test, iteration_loop, echo_num, S_ica_tVox, A_ica_tVox, ica_tVox_mean, data_scaling, \
            metrics_name=0, argsort_indices=0, sorting_metrics=0)

        # Plot S_ica timeseries and frequency spectra, and A_ica component maps 
        # **ONLY** if ICA converges.
        if converged_test == 1:
            #  Calculate the frequency spectra of the ICA components (spectra of S_ica).
            freq_S_ica_tVox = Sarah_ICA.freq_spec(k, S_ica_tVox)

            # If plotting, plot_all != 0 (i.e. plot_all == 1).
            if plot_all != 0:

            # If plot_cmap != 0 (i.e. plot_cmap == 1), plot 'original' component maps.
                if plot_cmap != 0:
                    # Need to reshape first ('remove the mask'), as A_ica_tVox is collapsed data.
                    maskmap_A_ica_tVox = functions_Calculate_stats.reshape_maps_Fast(size_echo_data, NumSlices, k, \
                        masks_reg_data, A_ica_tVox)
                    functions_PlottingICAetc_testing_Subplots.plot_maps_components(maskmap_A_ica_tVox, k, saving_details, showplot, cmap='bwr', \
                        subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, clims_multi=0.75, RunNum=RunNum)

                # Plot 'original' component time series and frequency spectra:
                # Time series
                title_add = "time series"; x_axis_label = "time /s"; saving_add = "timeseries"
                further_details = [title_add, x_axis_label, saving_add]
                functions_PlottingICAetc_testing_Subplots.plot_lineplot(plot_time_value, S_ica_tVox, k, saving_details, showplot, further_details, \
                    halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=RunNum)
                # Frequency spectra
                title_add = "frequency spectrum"; x_axis_label = "frequency /Hz"; saving_add = "freq"
                further_details = [title_add, x_axis_label, saving_add]
                functions_PlottingICAetc_testing_Subplots.plot_lineplot(freqPlot, freq_S_ica_tVox, k, saving_details, showplot, further_details, \
                    halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=RunNum)


        # Calculate component ordering and metric values.
        # Save the ordering information to the HDF5 file and plot the ordered
        # components, if desired.
        # ONLY IF THE ICA ALGORITHM CONVERGED:
        if converged_test == 1:
            # Calculate component ordering/metric values.
            # Loop over the different ordering methods.
            for j in ordering_method_list:
                S_ica_tVox_Sorted, freq_S_ica_tVox_Sorted, RMS_allFreq_P_relMax_argsort_indices, \
                    RMS_allFreq_P_relMax, RMS_allFreq_P_relMax_Sorted = \
                    functions_Calculate_stats.calc_plot_component_ordering(j, S_ica_tVox, A_ica_tVox, \
                        freq_S_ica_tVox, ica_tVox, size_echo_data, NumSlices, NumDyn, halfTimepoints, \
                        freqPlot, GasSwitch, TempRes, masks_reg_data, k, \
                        data_scaling)
                # Save components and ordering information.
                RunNum_2 = functions_LoopSave_ICA_HDF5.hdf5_save_ICA_metricsOnly(hdf5_loc, subject, k, mask_applied, ICA_applied, \
                    echo_num, RunNumber=RunNum, metrics_name=j, argsort_indices=np.array(RMS_allFreq_P_relMax_argsort_indices), \
                    sorting_metrics=np.array(RMS_allFreq_P_relMax))

                if plot_all != 0:
                    # Plot component ordering for the metric under investigation (specifically, plotting
                    # the component time series and frequency spectra).
                    # Additional (optional) arguments regarding plotting metrics values etc...
                    metric_title = 1 # 1 = Yes, 0 = No - whether to include details of a metric in the plot title.
                    saving_add_2 = "ordered" # IF ordering_used == 1 etc
                    # And pass metric value - could be one list with all of these
                    metric_value = np.array([RMS_allFreq_P_relMax])
                    # metric_value = np.array(RMS_allFreq_P_relMax)
                    arg_sort_indices = np.array(RMS_allFreq_P_relMax_argsort_indices)
                    ordered_already = 0 # Not ordered within this loop/function.
                    metrics_plotting = [metric_title, saving_add_2, metric_value, j, ordered_already, arg_sort_indices]
                    metrics_plotting = 0

                    # Time series
                    title_add = "time series"; x_axis_label = "time /s"; saving_add = "timeseries"
                    further_details = [title_add, x_axis_label, saving_add]
                    functions_PlottingICAetc_testing_Subplots.plot_lineplot(plot_time_value, S_ica_tVox, k, saving_details, showplot, further_details, \
                        halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=metrics_plotting, RunNum=RunNum)
                    # Frequency spectra
                    title_add = "frequency spectrum"; x_axis_label = "frequency /Hz"; saving_add = "freq"
                    further_details = [title_add, x_axis_label, saving_add]
                    functions_PlottingICAetc_testing_Subplots.plot_lineplot(freqPlot, freq_S_ica_tVox, k, saving_details, showplot, further_details, \
                        halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=metrics_plotting, RunNum=RunNum)

        print(str(k)) # Print num_component in loop.

    # Nothing to return - saved and plotted components and ordering if required.
    return RunNum, RunNum_2



if __name__ == "__main__":
    # ...
    a = 1