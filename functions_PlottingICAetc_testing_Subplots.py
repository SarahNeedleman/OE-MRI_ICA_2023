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

# Functions for plotting ICA/MRI parameters/outputs.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import functions_Calculate_stats

def plot_lineplot(plot_x_axis, plot_y_axis, num_components, saving_details, showplot, further_details, \
    halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=0):
    """
    Plot and save (for any number of time series/frequency spectra).

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = time series/frequency spectra amplitudes to be plotted on the y-axis. 
    Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name should include "ordered" [1] as the components plotted will
    be ordered by the metric value, with the metric value [2] to be included in the subplot titles.
    Metric value [3] describes the name of the ordering method, to be included in the name of the
    figure saved. Metric value [4] details whether the components are ordered = 0 No, 1 Yes.
    If No, call functions_Calculate_stats.order_components to order the components and metric 
    values before plotting. Metrics plotting [5] is the arg_sort indices array which contains
    details for ordering the components.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    if metrics_plotting != 0:
        if metrics_plotting[4] == 0:
            # # Need to order the components to be plotted and their metric values.
            plot_y_axis = functions_Calculate_stats.order_components(plot_y_axis, metrics_plotting[5])
            metrics_plotting[2] = functions_Calculate_stats.order_components(metrics_plotting[2], metrics_plotting[5])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))

    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            if metrics_plotting == 0:
                axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))) + " " + further_details[0])
            else:
                axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))) + " " + further_details[0] + ", {:.4f}".format(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))]))
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            if metrics_plotting == 0:
                if RunNum != 0:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + '_' + \
                        str(RunNum) + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + \
                    '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else:
                if RunNum != 0:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + '_' + \
                        str(RunNum) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + '_' + \
                    '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')


    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}

        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            # an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                if metrics_plotting == 0:
                    an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))) + " " + further_details[0])
                else:
                    print(metrics_plotting[2])
                    print(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))])
                    an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))) + " " + further_details[0] + ", {:.4f}".format(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))]))

                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()

        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')

        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)

        # Save figure if required.
        if saving_details[0] == 1:
            if metrics_plotting == 0:
                if RunNum != 0:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                        + '_' + str(RunNum) + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                    + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else:
                if RunNum != 0:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                        + '_' + str(RunNum) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                    + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_lineplot_shortTitle(plot_x_axis, plot_y_axis, num_components, saving_details, showplot, further_details, \
    halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=0):
    """
    Plot and save (for any number of time series/frequency spectra).
    'Short title' - just C#.

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = time series/frequency spectra amplitudes to be plotted on the y-axis. 
    Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name should include "ordered" [1] as the components plotted will
    be ordered by the metric value, with the metric value [2] to be included in the subplot titles.
    Metric value [3] describes the name of the ordering method, to be included in the name of the
    figure saved. Metric value [4] details whether the components are ordered = 0 No, 1 Yes.
    If No, call functions_Calculate_stats.order_components to order the components and metric 
    values before plotting. Metrics plotting [5] is the arg_sort indices array which contains
    details for ordering the components.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    if metrics_plotting != 0:
        if metrics_plotting[4] == 0:
            # # Need to order the components to be plotted and their metric values.
            plot_y_axis = functions_Calculate_stats.order_components(plot_y_axis, metrics_plotting[5])
            metrics_plotting[2] = functions_Calculate_stats.order_components(metrics_plotting[2], metrics_plotting[5])
            # metrics_plotting[2] = np.squeeze(functions_Calculate_stats.order_components(metrics_plotting[2], metrics_plotting[5]))
            # print(metrics_plotting[2])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))

    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            if metrics_plotting == 0:
                axs[axes_c].set_title("C#" + str(np.int32(np.rint(axes_c+1))) + "" + further_details[0])
            else:
                axs[axes_c].set_title("C#" + str(np.int32(np.rint(axes_c+1))) + "" + further_details[0] + ", {:.4f}".format(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))]))
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            if metrics_plotting == 0:
                if RunNum != 0:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + '_' + \
                        str(RunNum) + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + \
                    '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else:
                if RunNum != 0:
                    figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + '_' + \
                        str(RunNum) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] + '_' + \
                    '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')


    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}

        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            # an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                if metrics_plotting == 0:
                    an_axs_sub[j].set_title("C#" + str(np.int32(np.rint(axes_c+1))) + "" + further_details[0])
                else:
                    an_axs_sub[j].set_title("C#" + str(np.int32(np.rint(axes_c+1))) + "" + further_details[0] + ", {:.4f}".format(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))]))

                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()

        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')

        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)

        # Save figure if required.
        if saving_details[0] == 1:
            if metrics_plotting == 0:
                if RunNum != 0:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                        + '_' + str(RunNum) + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                    + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else:
                if RunNum != 0:
                    figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                        + '_' + str(RunNum) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
                else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + further_details[2] \
                    + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



def plot_lineplot_metricPlot(plot_x_axis, plot_y_axis, num_components, saving_details, showplot, further_details, \
    halfTimepoints, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=0):
    """
    Plot and save (for any number of time series/frequency spectra).
    ((metricPlot - including metric names etc...))

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = time series/frequency spectra amplitudes to be plotted on the y-axis. 
    Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name (e.g. min_RMS) [1] as the components plotted will
    be ordered by the metric value, with the metric names [2], metric values [3] and 
    component numbers [4] to be included in the subplot titles. 
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]

    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            axs[axes_c].set_title("Component " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
                ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum != 0:
                figs[idx].savefig(saving_details[1] + saving_details[2] + 'metrics_' + further_details[2] + '_' + str(RunNum) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else: figs[idx].savefig(saving_details[1] + saving_details[2] + 'metrics_' + further_details[2] + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')

    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}

        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                an_axs_sub[j].set_title("Component " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
                    ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()

        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')

        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)

        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum != 0:
                figs_sub.savefig(saving_details[1] + saving_details[2] + 'metrics_' + further_details[2] + '_' + str(RunNum) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else: figs_sub.savefig(saving_details[1] + saving_details[2] + 'metrics_' + further_details[2] + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_lineplot_metricPlot_investigation(plot_x_axis, plot_y_axis, num_components, saving_details, showplot, further_details, \
    halfTimepoints, spec_Num, subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=0):
    """
    Plot and save (for any number of time series/frequency spectra).
    ((metricPlot - including metric names etc...))

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = time series/frequency spectra amplitudes to be plotted on the y-axis. 
    Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name (e.g. min_RMS) [1] as the components plotted will
    be ordered by the metric value, with the metric names [2], metric values [3] and 
    component numbers [4] to be included in the subplot titles. 
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]

    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            axs[axes_c].set_title("Component " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
                ", " + metrics_plotting[2] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum != 0:
                figs[idx].savefig(saving_details[1] + saving_details[2] + '_metrics_' + metrics_plotting[2] + '_' + further_details[2] + 'Extra_' + 'specNumLowest' + str(spec_Num) + '_' + str(RunNum) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else: figs[idx].savefig(saving_details[1] + saving_details[2] + 'metrics_' + metrics_plotting[2] + '_' + further_details[2] + 'Extra_' + 'specNumLowest' + str(spec_Num) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')

    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}

        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                an_axs_sub[j].set_title("Component " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
                    ", " + metrics_plotting[2] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()

        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')

        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)


        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum != 0:
                figs_sub.savefig(saving_details[1] + saving_details[2] + 'metrics_' + metrics_plotting[2] + '_' + further_details[2] + 'Extra_' + 'specNumLowest' + str(spec_Num) + '_' + str(RunNum) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')
            else: figs_sub.savefig(saving_details[1] + saving_details[2] + 'metrics_' + metrics_plotting[2] + '_' + further_details[2] + 'Extra_' + 'specNumLowest' + str(spec_Num) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_maps_components(map_data, num_components, saving_details, showplot, cmap='bwr', \
    subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, clims_multi=0.75, RunNum=0):
    """
    Plot and save (for any number of maps). Plot a set of maps for each slice.

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of four elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name should include "ordered" [1] as the components plotted will
    be ordered by the metric value, with the metric value [2] to be included in the subplot titles.
    Metric value [3] describes the name of the ordering method, to be included in the name of the
    figure saved. Metric value [4] details whether the components are ordered = 0 No, 1 Yes.
    If No, call functions_Calculate_stats.order_components to order the components and metric 
    values before plotting.
    clims_multi = multiplier of the colour plot scale, default = 0.75. It will multiply the
    colour bar scale which is calculated separately *for each component in each slice*.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))

    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[3]):
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # Create array of figure identifiers to loop over.
        separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        # Loop over number of separate figures
        for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
            figs[idx]=plt.figure(figsize=(19.20,10.80))
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            for j in range(subplots_num_in_figure):
                # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
                axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
                ims[axes_c] = axs[axes_c].imshow(map_data[axes_c,:,:,slice_plot], cmap)
                axs[axes_c].invert_yaxis(); figs[idx].colorbar(ims[axes_c], ax=axs[axes_c])
                ims[axes_c].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])))
                #
                if metrics_plotting == 0:
                    axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                else:
                    axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))) + ", {:.4f}".format(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))]))
                # Increase the axis counter.
                axes_c = axes_c + 1

            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                        + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(idx+1))) +'.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot)\
                        + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')


        # Next plot the final partially filled figure. Only do this if there are
        # remainder subplots, i.e. subplots_num[1] != 0
        if subplots_num[1] != 0:
            figs_sub = plt.figure(figsize=(19.20,10.80))
            # Again, create a dictionary of axes (axs) identifiers to plot the axes
            # of the final figure. And for the images, an_ims_sub.
            an_axs_sub = {}
            an_ims_sub = {}

            # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
            # remainder axes or not) to maintain the correct scaling/sizes of the axes.
            for j in range(subplots_num_in_figure):
                # an_axs_sub[j] relates to each axis.
                # Add an axis for all subplots whether they are to be plotted or not.
                an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
                if j < subplots_num[1]:
                    # If the subplot axis is one of the remainder axes, plot.
                    # an_axs_sub[j] relates to each axis. ims[axes_c] relates to each image.
                    an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
                    an_ims_sub[j] = an_axs_sub[j].imshow(map_data[axes_c,:,:,slice_plot], cmap)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_ims_sub[j].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])), vmax=clims_multi * np.max(np.abs(map_data[axes_c,:,:,slice_plot])))
                    if metrics_plotting == 0:
                        an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                    else:
                        an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))) + ", {:.4f}".format(metrics_plotting[2][np.int32(np.rint(axes_c+1-1))]))
                else:
                    # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                    # to maintain the axes scaling/sizes.
                    # Maybe plot as zeros so square shape
                    an_ims_sub[j] = an_axs_sub[j].imshow(np.multiply(map_data[0,:,:,slice_plot],0), cmap, vmin=0, vmax=0)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_axs_sub[j].set_title(" ")
                    an_axs_sub[j].set_aspect('equal')
                axes_c = axes_c + 1

            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                                + '_' + str(RunNum) + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                            + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    # else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) \
                        + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return




def plot_maps_components_PSE_plot_MultiClims(map_data, num_components, saving_details, showplot, \
    title_extra, save_name_add, metric_args_plot, \
    cmap='bwr', subplots_num_in_figure=9, subplots_num_row=3, \
    metrics_plotting=0, clims_multi=0.75, RunNum=0, dpi_val=300):
    """
    Plot and save (for any number of maps). Plot a set of maps for each slice.
    Plot PSE - bwr.
    Clims multi - as different components (reconstructed PSE) plotted.
    

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of four elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name should include "ordered" [1] as the components plotted will
    be ordered by the metric value, with the metric value [2] to be included in the subplot titles.
    Metric value [3] describes the name of the ordering method, to be included in the name of the
    figure saved. Metric value [4] details whether the components are ordered = 0 No, 1 Yes.
    If No, call functions_Calculate_stats.order_components to order the components and metric 
    values before plotting.
    clims_multi = multiplier of the colour plot scale, default = 0.75. It will multiply the
    colour bar scale which is calculated separately *for each component in each slice*.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))


    # BWR overlay...
    # Colourmap scaling - maybe should be clims_multi as different for each component.
    cmap = cm.bwr
    # Calculate for each component (all slices) recon PSE.
    min_components = np.min(np.min(np.min(map_data, axis=0), axis=0), axis=0)
    max_components = np.max(np.max(np.max(map_data, axis=0), axis=0), axis=0)

    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[2]):
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # Create array of figure identifiers to loop over.
        separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        # Loop over number of separate figures
        for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
            figs[idx]=plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            for j in range(subplots_num_in_figure):
                # Previously... but include in loop instead, when plot each component.
                min_max_plotted = [min_components[axes_c], max_components[axes_c]]
                min_max_abs = np.max(np.abs(min_max_plotted))
                norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

                # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
                axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
                ims[axes_c] = axs[axes_c].imshow(map_data[:,:,slice_plot,axes_c], norm=norm_unC, cmap=cmap)
                axs[axes_c].invert_yaxis(); figs[idx].colorbar(ims[axes_c], ax=axs[axes_c])
                ims[axes_c].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[:,:,slice_plot,axes_c])), vmax=clims_multi * np.max(np.abs(map_data[:,:,slice_plot,axes_c])))
                axs[axes_c].set_xticks([])
                axs[axes_c].set_yticks([])
                #
                if metrics_plotting == 0:
                    axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                else:
                    axs[axes_c].set_title(title_extra[0] + " = {:.0f}".format(metric_args_plot[1][axes_c]) + \
                        ", " + title_extra[1] + " = {:.0f}".format(metric_args_plot[2][axes_c]))
                
                # Increase the axis counter.
                axes_c = axes_c + 1

            figs[idx].tight_layout()
            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: 
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_' + metric_args_plot[0] + \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot)\
                            + metric_args_plot[0] \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')


        # Next plot the final partially filled figure. Only do this if there are
        # remainder subplots, i.e. subplots_num[1] != 0
        if subplots_num[1] != 0:
            figs_sub = plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # Again, create a dictionary of axes (axs) identifiers to plot the axes
            # of the final figure. And for the images, an_ims_sub.
            an_axs_sub = {}
            an_ims_sub = {}

            # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
            # remainder axes or not) to maintain the correct scaling/sizes of the axes.
            for j in range(subplots_num_in_figure):
                # an_axs_sub[j] relates to each axis.
                # Add an axis for all subplots whether they are to be plotted or not.
                an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
                if j < subplots_num[1]:
                    # Previously... but include in loop instead, when plot each component.
                    min_max_plotted = [min_components[axes_c], max_components[axes_c]]
                    min_max_abs = np.max(np.abs(min_max_plotted))
                    norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

                    # If the subplot axis is one of the remainder axes, plot.
                    # an_axs_sub[j] relates to each axis. ims[axes_c] relates to each image.
                    an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
                    # an_ims_sub[j] = an_axs_sub[j].imshow(map_data[:,:,slice_plot,axes_c], cmap)
                    an_ims_sub[j] = an_axs_sub[j].imshow(map_data[:,:,slice_plot,axes_c], norm=norm_unC, cmap=cmap)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_ims_sub[j].set_clim(vmin=-clims_multi * np.max(np.abs(map_data[:,:,slice_plot,axes_c])), vmax=clims_multi * np.max(np.abs(map_data[:,:,slice_plot,axes_c])))
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                    if metrics_plotting == 0:
                        an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                    else:
                        an_axs_sub[j].set_title(title_extra[0] + " = {:.0f}".format(metric_args_plot[1][axes_c]) + \
                            ", " + title_extra[1] + " = {:.0f}".format(metric_args_plot[2][axes_c]))
                else:
                    # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                    # to maintain the axes scaling/sizes.
                    # Maybe plot as zeros so square shape
                    an_ims_sub[j] = an_axs_sub[j].imshow(np.multiply(map_data[:,:,slice_plot,0],0), cmap, vmin=-1, vmax=1)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[0], ax=an_axs_sub[j])
                    an_axs_sub[j].set_title(" ")
                    an_axs_sub[j].set_aspect('equal')
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                axes_c = axes_c + 1

            # # Finally, hide all axes that are not the remainder axes.
            figs_sub.tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                                + '_' + str(RunNum) + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_' + metric_args_plot[0] \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                        + metric_args_plot[0] \
                        + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_maps_components_PSE_plot_SetClims(map_data, num_components, saving_details, showplot, \
    title_extra, save_name_add, metric_args_plot, \
    clims_overlay, clims_multi=0.8, \
    cmap='bwr', \
    subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=0, dpi_val=300):
    """
    Plot and save (for any number of maps). Plot a set of maps for each slice.
    Plot PSE - bwr.
    Clims multi - as different components (reconstructed PSE) plotted.
    

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of four elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name should include "ordered" [1] as the components plotted will
    be ordered by the metric value, with the metric value [2] to be included in the subplot titles.
    Metric value [3] describes the name of the ordering method, to be included in the name of the
    figure saved. Metric value [4] details whether the components are ordered = 0 No, 1 Yes.
    If No, call functions_Calculate_stats.order_components to order the components and metric 
    values before plotting.
    clims_multi = multiplier of the colour plot scale, default = 0.75. It will multiply the
    colour bar scale which is calculated separately *for each component in each slice*.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))


    # BWR overlay...
    # Colourmap scaling - maybe should be clims_multi as different for each component.
    cmap = cm.bwr
    # Calculate for each component (all slices) recon PSE.
    min_components = np.min(np.min(np.min(map_data, axis=0), axis=0), axis=0)
    max_components = np.max(np.max(np.max(map_data, axis=0), axis=0), axis=0)

    # # Previously... but include in loop instead, when plot each component.
    # min_max_plotted = [np.min(map_data[0,:,:,:]), np.max(map_data[0,:,:,:])]
    # min_max_abs = np.max(np.abs(min_max_plotted))
    # norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)
        
    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[2]):
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # Create array of figure identifiers to loop over.
        separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        # Loop over number of separate figures
        for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
            figs[idx]=plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            for j in range(subplots_num_in_figure):
                # Previously... but include in loop instead, when plot each component.
                min_max_plotted = [min_components[axes_c], max_components[axes_c]]
                min_max_abs = np.max(np.abs(min_max_plotted))
                norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

                # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
                axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
                ims[axes_c] = axs[axes_c].imshow(map_data[:,:,slice_plot,axes_c], norm=norm_unC, cmap=cmap)
                axs[axes_c].invert_yaxis(); figs[idx].colorbar(ims[axes_c], ax=axs[axes_c])
                ims[axes_c].set_clim(vmin=-clims_overlay[2], vmax=clims_overlay[3])
                axs[axes_c].set_xticks([])
                axs[axes_c].set_yticks([])
                #
                if metrics_plotting == 0:
                    axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                else:
                    axs[axes_c].set_title(title_extra[0] + " = {:.0f}".format(metric_args_plot[1][axes_c]) + \
                        ", " + title_extra[1] + " = {:.0f}".format(metric_args_plot[2][axes_c]))# + \

                # Increase the axis counter.
                axes_c = axes_c + 1

            # plt.show()
            figs[idx].tight_layout()
            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: 
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) \
                            + metric_args_plot[0] + \
                            '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot)\
                            + metric_args_plot[0] \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    


        # Next plot the final partially filled figure. Only do this if there are
        # remainder subplots, i.e. subplots_num[1] != 0
        if subplots_num[1] != 0:
            figs_sub = plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # Again, create a dictionary of axes (axs) identifiers to plot the axes
            # of the final figure. And for the images, an_ims_sub.
            an_axs_sub = {}
            an_ims_sub = {}

            # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
            # remainder axes or not) to maintain the correct scaling/sizes of the axes.
            for j in range(subplots_num_in_figure):
                # an_axs_sub[j] relates to each axis.
                # Add an axis for all subplots whether they are to be plotted or not.
                an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
                if j < subplots_num[1]:
                    # Previously... but include in loop instead, when plot each component.
                    min_max_plotted = [min_components[axes_c], max_components[axes_c]]
                    min_max_abs = np.max(np.abs(min_max_plotted))
                    norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

                    # If the subplot axis is one of the remainder axes, plot.
                    # an_axs_sub[j] relates to each axis. ims[axes_c] relates to each image.
                    an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
                    # an_ims_sub[j] = an_axs_sub[j].imshow(map_data[:,:,slice_plot,axes_c], cmap)
                    an_ims_sub[j] = an_axs_sub[j].imshow(map_data[:,:,slice_plot,axes_c], norm=norm_unC, cmap=cmap)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_ims_sub[j].set_clim(vmin=-clims_overlay[2], vmax=clims_overlay[3])
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                    if metrics_plotting == 0:
                        an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                    else:
                        an_axs_sub[j].set_title(title_extra[0] + " = {:.0f}".format(metric_args_plot[1][axes_c]) + \
                            ", " + title_extra[1] + " = {:.0f}".format(metric_args_plot[2][axes_c]))# + \


                else:
                    # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                    # to maintain the axes scaling/sizes.
                    # Maybe plot as zeros so square shape
                    an_ims_sub[j] = an_axs_sub[j].imshow(np.multiply(map_data[:,:,slice_plot,0],0), cmap, vmin=-1, vmax=1)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[0], ax=an_axs_sub[j])
                    an_axs_sub[j].set_title(" ")
                    an_axs_sub[j].set_aspect('equal')
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                axes_c = axes_c + 1

            figs_sub.tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_' + metric_args_plot[0] \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')                    
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + metric_args_plot[0] \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



def plot_maps_components_PSE_plot_SetClims_MetricValue(map_data, num_components, saving_details, showplot, \
    title_extra, save_name_add, metric_args_plot, \
    clims_overlay, clims_multi=0.8, \
    cmap='bwr', \
    subplots_num_in_figure=9, subplots_num_row=3, metrics_plotting=0, RunNum=0, dpi_val=300):
    """
    Plot and save (for any number of maps). Plot a set of maps for each slice.
    Plot PSE - bwr.
    Clims multi - as different components (reconstructed PSE) plotted.
    

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of subplots per row, default = 3 (for 3 x 3 subplots
    per figure).
    metrics_plotting = list of four elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name should include "ordered" [1] as the components plotted will
    be ordered by the metric value, with the metric value [2] to be included in the subplot titles.
    Metric value [3] describes the name of the ordering method, to be included in the name of the
    figure saved. Metric value [4] details whether the components are ordered = 0 No, 1 Yes.
    If No, call functions_Calculate_stats.order_components to order the components and metric 
    values before plotting.
    clims_multi = multiplier of the colour plot scale, default = 0.75. It will multiply the
    colour bar scale which is calculated separately *for each component in each slice*.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))


    # BWR overlay...
    # Colourmap scaling - maybe should be clims_multi as different for each component.
    cmap = cm.bwr
    # Calculate for each component (all slices) recon PSE.
    min_components = np.min(np.min(np.min(map_data, axis=0), axis=0), axis=0)
    max_components = np.max(np.max(np.max(map_data, axis=0), axis=0), axis=0)

    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[2]):
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # Create array of figure identifiers to loop over.
        separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        # Loop over number of separate figures
        for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
            figs[idx]=plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # figs[idx] relates to each figure.
            # Loop over the separate figures and plot the subplots.
            for j in range(subplots_num_in_figure):
                min_max_plotted = [min_components[axes_c], max_components[axes_c]]
                min_max_abs = np.max(np.abs(min_max_plotted))
                norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

                # axs[axes_c] relates to each axis. ims[axes_c] relates to each image.
                axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
                ims[axes_c] = axs[axes_c].imshow(map_data[:,:,slice_plot,axes_c], norm=norm_unC, cmap=cmap)
                axs[axes_c].invert_yaxis(); figs[idx].colorbar(ims[axes_c], ax=axs[axes_c])
                ims[axes_c].set_clim(vmin=-clims_overlay[2], vmax=clims_overlay[3])
                axs[axes_c].set_xticks([])
                axs[axes_c].set_yticks([])
                #
                if metrics_plotting == 0:
                    axs[axes_c].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                else:
                    axs[axes_c].set_title("{:.0f}".format(metric_args_plot[1][axes_c]) + " " + title_extra[0] + \
                        ", " + "metric = {:.3f}".format(metric_args_plot[3][axes_c]))

                # Increase the axis counter.
                axes_c = axes_c + 1


            figs[idx].tight_layout()
            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: 
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) \
                            + metric_args_plot[0] + \
                            '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + 'metric.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs[idx].savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot)\
                            + metric_args_plot[0] \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(idx+1))) \
                            + 'metric.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    


        # Next plot the final partially filled figure. Only do this if there are
        # remainder subplots, i.e. subplots_num[1] != 0
        if subplots_num[1] != 0:
            figs_sub = plt.figure(figsize=(19.20,10.80))
            # plt.rcParams["figure.figsize"] = (19.20,10.80)
            # Again, create a dictionary of axes (axs) identifiers to plot the axes
            # of the final figure. And for the images, an_ims_sub.
            an_axs_sub = {}
            an_ims_sub = {}

            # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
            # remainder axes or not) to maintain the correct scaling/sizes of the axes.
            for j in range(subplots_num_in_figure):
                # an_axs_sub[j] relates to each axis.
                # Add an axis for all subplots whether they are to be plotted or not.
                an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
                if j < subplots_num[1]:
                    # Previously... but include in loop instead, when plot each component.
                    min_max_plotted = [min_components[axes_c], max_components[axes_c]]
                    min_max_abs = np.max(np.abs(min_max_plotted))
                    norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

                    # If the subplot axis is one of the remainder axes, plot.
                    # an_axs_sub[j] relates to each axis. ims[axes_c] relates to each image.
                    an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_row, j+1)
                    an_ims_sub[j] = an_axs_sub[j].imshow(map_data[:,:,slice_plot,axes_c], norm=norm_unC, cmap=cmap)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[j], ax=an_axs_sub[j])
                    an_ims_sub[j].set_clim(vmin=-clims_overlay[2], vmax=clims_overlay[3])
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                    if metrics_plotting == 0:
                        an_axs_sub[j].set_title("Component number " + str(np.int32(np.rint(axes_c+1))))
                    else:
                        an_axs_sub[j].set_title("{:.0f}".format(metric_args_plot[1][axes_c]) + " " + title_extra[0] + \
                            ", " + "metric = {:.3f}".format(metric_args_plot[3][axes_c]))


                else:
                    # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                    # to maintain the axes scaling/sizes.
                    # Maybe plot as zeros so square shape
                    an_ims_sub[j] = an_axs_sub[j].imshow(np.multiply(map_data[:,:,slice_plot,0],0), cmap, vmin=-1, vmax=1)
                    an_axs_sub[j].invert_yaxis(); figs_sub.colorbar(an_ims_sub[0], ax=an_axs_sub[j])
                    an_axs_sub[j].set_title(" ")
                    an_axs_sub[j].set_aspect('equal')
                    an_axs_sub[j].set_xticks([])
                    an_axs_sub[j].set_yticks([])
                axes_c = axes_c + 1

            figs_sub.tight_layout()

            # Save figure if required.
            if saving_details[0] == 1:
                if metrics_plotting == 0:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + '.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                else:
                    if RunNum != 0:
                        figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + '_' + str(RunNum) + '_' + metric_args_plot[0] \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + 'metric.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')                    
                    # else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + str(num_components) + 'c_' + '_Maps_Slice' + str(slice_plot) + '_' + metrics_plotting[1] +  '_' + metrics_plotting[3] + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
                    else: figs_sub.savefig(saving_details[1] + saving_details[2] + '_' + save_name_add + '_Maps_Slice' + str(slice_plot) \
                            + metric_args_plot[0] \
                            + '_ClimsSet_' + str(clims_overlay[2]) + '_' + str(clims_overlay[3]) \
                            + '__PSE_components_variety' + '_Plot' + str(np.int32(np.rint(subplots_num[0]+1))) \
                            + 'metric.png', dpi=dpi_val, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_maps_components_metrics_plot(map_data, SI_data, saving_details, showplot, num_plot_metric, map_type, dyn_plot=0, \
    cmap='bwr', metrics_plotting=0, clims_multi=0.75, RunNum=0):
    """
    Plot and save (for any number of maps). Plot a set of maps for each slice.
    ((metricPlot - including metric names etc...))

    Arguments:
    map_data = map data to be plotted, shape of (num_components, vox, vox, NumSlices).
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name (e.g. min_RMS) [1] as the components plotted will
    be ordered by the metric value, with the metric names [2], metric values [3] and 
    component numbers [4] to be included in the subplot titles. 
    clims_multi = multiplier of the colour plot scale, default = 0.75.
    RunNum = ICA run number, as for saved components. Default = 0 for if not saved and 
    will not be included in the plot saving information.
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num_in_figure = 8
    subplots_num_column = 4
    subplots_num_row = 2


    # Loop over all slices:
    for slice_plot in range(np.shape(map_data)[3]):
        # Calculate clims for each slice separately
        SI_data_slice = SI_data[:,:,:,slice_plot]
        clims_all = [np.min(SI_data_slice[SI_data_slice != 0]), \
            clims_multi*np.max(SI_data_slice[SI_data_slice != 0])]
        # slice_plot = slice to be plotted in the loop.
        # First loop and plot the full figure subplots. Create a dictionary 
        # of figure identifiers (figs) and axes (axs) to plot the figures/axes
        # in a loop. Also, image identifier (ims).
        figs={}
        axs={}
        ims={}
        # axes_c = a counter for the axis number to be used when plotting the 
        # axis titles.
        axes_c = 0

        fig1=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.


        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 1)
        ims[axes_c] = axs[axes_c].imshow(map_data[:,:,dyn_plot,slice_plot,axes_c], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("C " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
            ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 2)
        ims[axes_c] = axs[axes_c].imshow(map_data[:,:,dyn_plot,slice_plot,axes_c], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("C " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
            ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 3)
        ims[axes_c] = axs[axes_c].imshow(map_data[:,:,dyn_plot,slice_plot,axes_c], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("C " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
            ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 5)
        ims[axes_c] = axs[axes_c].imshow(map_data[:,:,dyn_plot,slice_plot,axes_c], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("C " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
            ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 6)
        ims[axes_c] = axs[axes_c].imshow(map_data[:,:,dyn_plot,slice_plot,axes_c], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("C " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
            ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 7)
        ims[axes_c] = axs[axes_c].imshow(map_data[:,:,dyn_plot,slice_plot,axes_c], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("C " + str(np.int32(np.rint(metrics_plotting[4][axes_c]))) + \
            ", " + metrics_plotting[2][axes_c] + " = {:.4f}".format(metrics_plotting[3][axes_c]))
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        # # FINAL 2 SI...
        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 4)
        ims[axes_c] = axs[axes_c].imshow(SI_data[:,:,dyn_plot,slice_plot], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("SI original data masked")
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1

        axs[axes_c] = fig1.add_subplot(subplots_num_row, subplots_num_column, 8)
        ims[axes_c] = axs[axes_c].imshow(SI_data[:,:,dyn_plot,slice_plot], cmap)
        axs[axes_c].invert_yaxis(); fig1.colorbar(ims[axes_c], ax=axs[axes_c])
        ims[axes_c].set_clim(vmin=clims_all[0], vmax=clims_all[1])
        axs[axes_c].set_title("SI original data masked")
        # figs[idx].tight_layout()
        # Increase the axis counter.
        axes_c = axes_c + 1


        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum != 0:
                fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + str(RunNum) + '_' + map_type + '_' + 'Maps_Slice' + str(slice_plot) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(num_plot_metric))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
            else: fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + map_type + '_' + 'Maps_Slice' + str(slice_plot) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(num_plot_metric))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')
    
    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_maps_recon_BWR(recon_OE_maps, NumSlices, map_type, \
    saving_details, showplot, metrics_plotting, num_plot_metric, \
    clims_multi=0.75, dyn_plot=120):
    """
    Plot recon maps with BWR colour bar. E.g. for not unMean and not scaled.
    For different metrics.

    Arguments:
    recon_OE_maps = reconstructed OE maps of shape (vox,vox,NumDyn,NumSlices,len(number of metrics to plot))).
    NumSlices = number of slcies acquired.
    map_type = whether SI or reconstructed etc...
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name (e.g. min_RMS) [1] as the components plotted will
    be ordered by the metric value, with the metric names [2], metric values [3] and 
    component numbers [4] to be included in the subplot titles. 
    num_plot_metric = was using to identify the plot, but instead take this into account with
    'G21' (> 21 components used?) and saving details etc and metrics_plotting.
    clims_multi = colourbar limits multiplier on maximum value that is plotted.
    dyn_plot = dynamic timepoint to plot. Default of 120 = end of first oxygen cycle.
    Returns:
    None, plot and saves if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]

    # Calculate colourbar scaling for each metric and for the specific dyn_plot timepoint.
    # (Over all slices).
    reshaped_recon_OE_maps = np.reshape(recon_OE_maps[:,:,dyn_plot,:,:], (np.shape(recon_OE_maps)[0]*np.shape(recon_OE_maps)[1]*np.shape(recon_OE_maps)[3], np.shape(recon_OE_maps)[4]))
    min_max_EachMetric = np.array([np.min(reshaped_recon_OE_maps, axis=0), np.max(reshaped_recon_OE_maps, axis=0)])
    min_max_EachMetric = np.max(np.abs(min_max_EachMetric))

    # Loop over each slice
    for j in range(NumSlices):
        fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3); plt.rcParams["figure.figsize"] = (19.20,10.80)
        #
        im1 = ax1.imshow(recon_OE_maps[:,:,dyn_plot,j,0], cmap="bwr")
        ax1.set_title("C " + str(np.int32(np.rint(metrics_plotting[4][0]))) + \
            ", " + metrics_plotting[2][0] + " = {:.4f}".format(metrics_plotting[3][0]))
        ax1.invert_yaxis(); fig1.colorbar(im1, ax=ax1); im1.set_clim(vmin=-min_max_EachMetric*clims_multi, vmax=min_max_EachMetric*clims_multi)
        #
        im2 = ax2.imshow(recon_OE_maps[:,:,dyn_plot,j,1], cmap="bwr")
        ax2.set_title("C " + str(np.int32(np.rint(metrics_plotting[4][1]))) + \
            ", " + metrics_plotting[2][1] + " = {:.4f}".format(metrics_plotting[3][1]))
        ax2.invert_yaxis(); fig1.colorbar(im2, ax=ax2); im2.set_clim(vmin=-min_max_EachMetric*clims_multi, vmax=min_max_EachMetric*clims_multi)
        #
        im3 = ax3.imshow(recon_OE_maps[:,:,dyn_plot,j,2], cmap="bwr")
        ax3.set_title("C " + str(np.int32(np.rint(metrics_plotting[4][2]))) + \
            ", " + metrics_plotting[2][2] + " = {:.4f}".format(metrics_plotting[3][2]))
        ax3.invert_yaxis(); fig1.colorbar(im3, ax=ax3); im3.set_clim(vmin=-min_max_EachMetric*clims_multi, vmax=min_max_EachMetric*clims_multi)
        #
        im4 = ax4.imshow(recon_OE_maps[:,:,dyn_plot,j,3], cmap="bwr")
        ax4.set_title("C " + str(np.int32(np.rint(metrics_plotting[4][3]))) + \
            ", " + metrics_plotting[2][3] + " = {:.4f}".format(metrics_plotting[3][3]))
        ax4.invert_yaxis(); fig1.colorbar(im4, ax=ax4); im4.set_clim(vmin=-min_max_EachMetric*clims_multi, vmax=min_max_EachMetric*clims_multi)
        #
        im5 = ax5.imshow(recon_OE_maps[:,:,dyn_plot,j,4], cmap="bwr")
        ax5.set_title("C " + str(np.int32(np.rint(metrics_plotting[4][4]))) + \
            ", " + metrics_plotting[2][4] + " = {:.4f}".format(metrics_plotting[3][4]))
        ax5.invert_yaxis(); fig1.colorbar(im5, ax=ax5); im5.set_clim(vmin=-min_max_EachMetric*clims_multi, vmax=min_max_EachMetric*clims_multi)
        #
        im6 = ax6.imshow(recon_OE_maps[:,:,dyn_plot,j,5], cmap="bwr")
        ax6.set_title("C " + str(np.int32(np.rint(metrics_plotting[4][5]))) + \
            ", " + metrics_plotting[2][5] + " = {:.4f}".format(metrics_plotting[3][5]))
        ax6.invert_yaxis(); fig1.colorbar(im6, ax=ax6); im6.set_clim(vmin=-min_max_EachMetric*clims_multi, vmax=min_max_EachMetric*clims_multi)
        fig1.tight_layout()

        if saving_details[0] == 1:
            fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + map_type + '_Dyn' + str(np.int32(np.rint(dyn_plot))) + '_' + 'Maps_Slice' + str(np.int32(np.rint(j+1))) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(num_plot_metric))) + '.png', dpi=300, bbox_inches='tight')#, bbox_inches=0) # #, bbox_inches='tight'), bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def PSE_maps_plot_metricsSI(PSE_image_SIdata, PSE_image_metricsdata, \
    stats_eachSlice_PSE_SI_LungMask, stats_eachSlice_PSE_Recon_unMean_unSc_LungMask, \
    NumSlices, map_type, \
    saving_details, showplot, metrics_plotting, num_plot_metric, \
    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25]):
    """
    Plot PSE maps for the metrics and compare directly to the SI PSE maps.
    Hence, the PSE metrics maps are of the reconstructed unMean and unScaled data.
    PSE map colourbars can be unequal from the diverging BWR scale, having zero at the 
    centre. This is controlled by vlims_multi and vlims_multi2.

    Arguments:
    PSE_image_SIdata = PSE map of the input registered SI MRI data, with shape
    of (vox, vox, NumSlices).
    PSE_image_metricsdata = PSE maps of the reconstructed metrics, with shape
    of (vox, vox, NumSlices, number of metrics being investigated)
    stats_eachSlice_PSE_SI_LungMask = stats per slice of the SI PSE image. With shape
    (NumSlices, stats).
    stats_eachSlice_PSE_Recon_unMean_unSc_LungMask = stats per slice of the SI PSE image for
    all metrics. With shape (NumSlices, stats, different metrics).
    NumSlices = number of slices acquired.
    map_type = whether SI or reconstructed etc...
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name (e.g. min_RMS) [1] as the components plotted will
    be ordered by the metric value, with the metric names [2], metric values [3] and 
    component numbers [4] to be included in the subplot titles. 
    num_plot_metric = was using to identify the plot, but instead take this into account with
    'G21' (> 21 components used?) and saving details etc and metrics_plotting.
    clims_multi = colourbar limits multiplier on maximum value that is plotted.
    vlims_multi = multiplier of the colourbar *range* plotted, multiplying the lower limit
    of the colourbar. Default value = 0.8.
    vlims_multi2 = (possible array, each will be plotted and saved according to the 
    index into the array) multiplier of the colourbar *range* plotted, multiplying the upper limit
    of the colourbar. Default value = [0.5, 0.25].
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]

    # Plot PSE image maps.
    # Colourmap scaling - diverging colourmap but 'skewed' - centre about value but without
    # requiring the same +/- colourbar scales to be plotted.
    # Using the bwr diverging colourmap - set cmap to this colourmap.
    import matplotlib.cm as cm
    cmap = cm.bwr
    import matplotlib.colors as colors

    for k in range(len(vlims_multi2)):
        # Calculate the maximum absolute PSE value over all PSE data from the SI image
        # over (all voxels and slices).
        min_max_plotted = [np.min(PSE_image_SIdata[:,:,:]), np.max(PSE_image_SIdata[:,:,:])]
        min_max_abs = np.max(np.abs(min_max_plotted))
        # Set the BWR diverging colourmap scale to have limits of + and - clims_multi*max(abs(value)),
        # and to be centred about zero.
        norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)
        # The above set the colourbar scale. The colourbar range plotted can be altered by setting the colourbar
        # limits for each image, using im.set_clim(vmin=plottingMin, vmax=plottingMax).
        # The lower colourbar limit is multiplied by vlims_multi (likely to be slightly < clims_multi - as 
        # the max(abs(value)) was calculated over all slices.
        # The upper colourbar limit is multiplied by vlims_multi2 - the kth value. Multiple versions can be used
        # to optimise the plots and maximise diverging colours.

        for j in range(NumSlices):
            # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
            # Also include the metric, component number used and metric value.
            fig1, ((ax1, ax2, ax3, ax7), (ax4, ax5, ax6, ax8)) = plt.subplots(2, 4); plt.rcParams["figure.figsize"] = (19.20,10.80)
            #
            im1 = ax1.imshow(PSE_image_metricsdata[:,:,j,0], norm=norm_unC, cmap=cmap)
            ax1.set_title("Mean air-oxy PSE in lung mask slice = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[j,0,0]) \
                + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][0]))) + \
                ", " + metrics_plotting[2][0] + " = {:.4f}".format(metrics_plotting[3][0]))
            ax1.invert_yaxis(); fig1.colorbar(im1, ax=ax1); im1.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im2 = ax2.imshow(PSE_image_metricsdata[:,:,j,1], norm=norm_unC, cmap=cmap)
            ax2.set_title("Mean air-oxy PSE in lung mask slice = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[j,0,1]) \
                + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][1]))) + \
                ", " + metrics_plotting[2][1] + " = {:.4f}".format(metrics_plotting[3][1]))
            ax2.invert_yaxis(); fig1.colorbar(im2, ax=ax2); im2.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im3 = ax3.imshow(PSE_image_metricsdata[:,:,j,2], norm=norm_unC, cmap=cmap)
            ax3.set_title("Mean air-oxy PSE in lung mask slice = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[j,0,2]) \
                + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][2]))) + \
                ", " + metrics_plotting[2][2] + " = {:.4f}".format(metrics_plotting[3][2]))
            ax3.invert_yaxis(); fig1.colorbar(im3, ax=ax3); im3.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im4 = ax4.imshow(PSE_image_metricsdata[:,:,j,3], norm=norm_unC, cmap=cmap)
            ax4.set_title("Mean air-oxy PSE in lung mask slice = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[j,0,3]) \
                + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][3]))) + \
                ", " + metrics_plotting[2][3] + " = {:.4f}".format(metrics_plotting[3][3]))
            ax4.invert_yaxis(); fig1.colorbar(im4, ax=ax4); im4.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im5 = ax5.imshow(PSE_image_metricsdata[:,:,j,4], norm=norm_unC, cmap=cmap)
            ax5.set_title("Mean air-oxy PSE in lung mask slice = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[j,0,4]) \
                + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][4]))) + \
                ", " + metrics_plotting[2][4] + " = {:.4f}".format(metrics_plotting[3][4]))
            ax5.invert_yaxis(); fig1.colorbar(im5, ax=ax5); im5.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im6 = ax6.imshow(PSE_image_metricsdata[:,:,j,5], norm=norm_unC, cmap=cmap)
            ax6.set_title("Mean air-oxy PSE in lung mask slice = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[j,0,5]) \
                + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][5]))) + \
                ", " + metrics_plotting[2][5] + " = {:.4f}".format(metrics_plotting[3][5]))
            ax6.invert_yaxis(); fig1.colorbar(im6, ax=ax6); im6.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im7 = ax7.imshow(PSE_image_SIdata[:,:,j], norm=norm_unC, cmap=cmap)
            ax7.set_title("Mean air-oxy PSE in SI lung mask = {:.2f}".format(stats_eachSlice_PSE_SI_LungMask[j,0]))
            ax7.invert_yaxis(); fig1.colorbar(im7, ax=ax7); im7.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            #
            im8 = ax8.imshow(PSE_image_SIdata[:,:,j], norm=norm_unC, cmap=cmap)
            ax8.set_title("Mean air-oxy PSE in SI lung mask = {:.2f}".format(stats_eachSlice_PSE_SI_LungMask[j,0]))
            ax8.invert_yaxis(); fig1.colorbar(im8, ax=ax8); im8.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            
            fig1.tight_layout()

            if saving_details[0] == 1:
                fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + map_type + '_PSE' + '_' + 'Maps_Slice' + str(np.int32(np.rint(j+1))) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(num_plot_metric))) + 'v'+ str(k) + '.png', dpi=600, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_ValueMeanSlice_time(slice_to_plot, plot_time_value, \
    PSE_timeseries_SI_lungMask_MeanEachSlice, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice, \
    stats_eachSlice_PSE_SI_LungMask, stats_eachSlice_PSE_Recon_unMean_unSc_LungMask, \
    mask_type, saving_details, map_type, num_plot_metric, showplot, metrics_plotting):
    """
    Plot mean PSE within a slice over time.

    Arguments:
    slice_to_plot = mean PSE within the specified slice will be plotted.
    plot_time_value = x-values of time to be plotted.
    PSE_timeseries_SI_lungMask_MeanEachSlice = PSE values over time for each slice (mean in
    each slice) for the SI MRI registered data. With shape (NumDyn, NumSlices).
    PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice = PSE values over 
    time for each slice (mean in each slice) for each metric's reconstructed 
    MRI data (unMean and unScaled). With shape (NumDyn, NumSlices, number of metrics).
    stats_eachSlice_PSE_SI_LungMask = stats per slice of the SI PSE image. With shape
    (NumSlices, stats).
    stats_eachSlice_PSE_Recon_unMean_unSc_LungMask = stats per slice of the SI PSE image for
    all metrics. With shape (NumSlices, stats, different metrics).
    mask_type = lung or cardiac, for titles/saving.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    map_type = whether SI or reconstructed etc...
    num_plot_metric = was using to identify the plot, but instead take this into account with
    'G21' (> 21 components used?) and saving details etc and metrics_plotting.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metrics_plotting = list of six elements, or 0 (default) describing details of the metric
    values, or if 0, not to include. Whether to plot the metric is either from != 0 value, or
    from [0] == 1. The saving name (e.g. min_RMS) [1] as the components plotted will
    be ordered by the metric value, with the metric names [2], metric values [3] and 
    component numbers [4] to be included in the subplot titles. 
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If restriction on number of components, include G21 in Figure saving name.
    if saving_details[3][0] == 1:
        saving_end_detail = saving_details[3][1] + 'G21'
    else: saving_end_detail = saving_details[3][1]

    # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
    # Also include the metric, component number used and metric value.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    #
    im1 = ax1.plot(plot_time_value, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,slice_to_plot-1, 0])
    ax1.set_title("Mean air-oxy PSE in lung mask = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[slice_to_plot-1,0,0]) \
        + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][0]))) + \
        ", " + metrics_plotting[2][0] + " = {:.4f}".format(metrics_plotting[3][0]))
    ax1.set_xlabel('Time /s'); ax1.set_ylabel('PSE %')
    #
    im2 = ax2.plot(plot_time_value, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,slice_to_plot-1, 1])
    ax2.set_title("Mean air-oxy PSE in lung mask = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[slice_to_plot-1,0,1]) \
        + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][1]))) + \
        ", " + metrics_plotting[2][1] + " = {:.4f}".format(metrics_plotting[3][1]))
    ax2.set_xlabel('Time /s'); ax2.set_ylabel('PSE %')
    #
    im3 = ax3.plot(plot_time_value, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,slice_to_plot-1, 2])
    ax3.set_title("Mean air-oxy PSE in lung mask = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[slice_to_plot-1,0,2]) \
        + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][2]))) + \
        ", " + metrics_plotting[2][2] + " = {:.4f}".format(metrics_plotting[3][2]))
    ax3.set_xlabel('Time /s'); ax3.set_ylabel('PSE %')
    #
    im4 = ax4.plot(plot_time_value, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,slice_to_plot-1, 3])
    ax4.set_title("Mean air-oxy PSE in lung mask = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[slice_to_plot-1,0,3]) \
        + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][3]))) + \
        ", " + metrics_plotting[2][3] + " = {:.4f}".format(metrics_plotting[3][3]))
    ax4.set_xlabel('Time /s'); ax4.set_ylabel('PSE %')
    #
    im5 = ax5.plot(plot_time_value, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,slice_to_plot-1, 4])
    ax5.set_title("Mean air-oxy PSE in lung mask = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[slice_to_plot-1,0,4]) \
        + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][4]))) + \
        ", " + metrics_plotting[2][4] + " = {:.4f}".format(metrics_plotting[3][4]))
    ax5.set_xlabel('Time /s'); ax5.set_ylabel('PSE %')
    #
    im6 = ax6.plot(plot_time_value, PSE_timeseries_X_recon_OE_unMean_unSc_maps_lungMask_MeanEachSlice[:,slice_to_plot-1, 5])
    ax6.set_title("Mean air-oxy PSE in lung mask = {:.2f}".format(stats_eachSlice_PSE_Recon_unMean_unSc_LungMask[slice_to_plot-1,0,5]) \
        + "\n" + "C " + str(np.int32(np.rint(metrics_plotting[4][5]))) + \
        ", " + metrics_plotting[2][5] + " = {:.4f}".format(metrics_plotting[3][5]))
    ax6.set_xlabel('Time /s'); ax6.set_ylabel('PSE %')
    #
    im7 = ax7.plot(plot_time_value, PSE_timeseries_SI_lungMask_MeanEachSlice[:,slice_to_plot-1])
    ax7.set_title("Mean air-oxy PSE in SI lung mask = {:.2f}".format(stats_eachSlice_PSE_SI_LungMask[slice_to_plot-1,0]))
    ax7.set_xlabel('Time /s'); ax7.set_ylabel('PSE %')
    #
    im8 = ax8.plot(plot_time_value, PSE_timeseries_SI_lungMask_MeanEachSlice[:,slice_to_plot-1])
    ax8.set_title("Mean air-oxy PSE in SI lung mask = {:.2f}".format(stats_eachSlice_PSE_SI_LungMask[slice_to_plot-1,0]))
    ax8.set_xlabel('Time /s'); ax8.set_ylabel('PSE %')
    
    fig1.tight_layout()

    if saving_details[0] == 1:
        fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + map_type + '_PSEtimeseries' + '_' + mask_type + '_' + 'Maps_Slice' + str(np.int32(np.rint(slice_to_plot))) + '_' + saving_end_detail + '_Plot' + str(np.int32(np.rint(num_plot_metric))) + '.png', dpi=600, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_singleMetric_Component_TimeFreq(halfTimepoints, plot_x_axis_Time, plot_x_axis_Freq, \
    plot_y_axis_Time, plot_y_axis_Freq, plot_y_axis_SI, \
    saving_details, showplot, metric_chosen, sorted_array_metric_sorted):
    """
    Plot timeseries and frequency spectrum for a single component (metric),
    assuming g21.

    Arguments:
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    plot_x_axis_Time = x-axis values of time, shape of (NumDyn,).
    plot_x_axis_Freq = x-axis values of frequency (positive values only), 
    shape of (NumDyn/2,).
    plot_y_axis_Time = time series amplitudes to be plotted on the y-axis. 
    Shape of (NumDyn,).
    plot_y_axis_Freq = frequency spectra amplitudes to be plotted on the y-axis. 
    Shape of (NumDyn,) - need to plot only the positive frequencies by using
    halfTimepoints.
    plot_y_axis_SI = mean SI timeseries in each slice. Shape of (NumDyn, NumSlices).
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    sorted_array_metric_sorted = array of the component numbers and metric value
    for the metric under investigation, sorted by the metric value. For use
    when describing the metric in the plot titles. Shape of 
    (number of converged ICA runs, 2, 1); with the first index running over
    the different numbers of components for which ICA reached convergence;
    the second index containing  the component numbers 
    sorted_array_metric_sorted[:,0,:] and their corresponding metric values 
    sorted_array_metric_sorted[:,1,:]; the third index previously ran over the
    different metrics under investigation.
    Returns:
    None - just saves figures and plots.
    
    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)
    
    # Plot timeseries and frequency spectrum
    fig1, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1); plt.rcParams["figure.figsize"] = (19.20,10.80)
    
    # For frequency spectrum, need to truncate plot to positive frequencies only.
    plot_y_axis_Freq = np.array(plot_y_axis_Freq[:halfTimepoints])
    ax1.plot(plot_x_axis_Time, plot_y_axis_SI[:,3-1]); ax1.set_xlabel("Time /s")
    ax2.plot(plot_x_axis_Time, plot_y_axis_Time); ax2.set_xlabel("Time /s")
    ax3.plot(plot_x_axis_Freq, plot_y_axis_Freq); ax3.set_xlabel("Frequency /Hz")
    # Axis title to include:
    # Component number = np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))
    # Metric value = sorted_array_metric_sorted[0,1,0]
    # Metric name = metric_chosen[0]
    ax1.set_title("Time series of registered SI MRI data")
    ax2.set_title("Time series of component " + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + \
        ", " + metric_chosen[0] + " = {:.4f}".format(sorted_array_metric_sorted[0,1,0]))
    ax3.set_title("Frequency spectra of component " + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + \
        ", " + metric_chosen[0] + " = {:.4f}".format(sorted_array_metric_sorted[0,1,0]))

    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        fig1.savefig(saving_details[1] + saving_details[2] + '_' + 'TimeFreq' + '___' + metric_chosen[0] + 'G21' + '.png', dpi=600, bbox_inches=0)

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_singleMetric_PSEmaps_SIandRecon(PSE_image_SI_all, PSE_image_unMean_unSc_all, \
    stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung, \
    NumSlices, saving_details, showplot, \
    metric_chosen, sorted_array_metric_sorted, \
    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25]):
    """
    Plot reconstructed and SI PSE maps for each slice
    for each oxygen cycle and the mean oxygen cycle PSE.
    Assuming g21, single metric of interest.

    1) Plot each slice for recon and SI PSE for the mean oxygen
    calculation.
    2) Plot each slice separately for recon and SI PSE for each cycle.
    
    Arguments:
    PSE_image_SI_all = PSE image calculated for the MRI registered input data for 
    a specific echo. Calculated for each slice for each oxygen cycle and the 
    three cycles (V2). Shape of (4 for overall and the three cycles, vox, vox, NumSlices).
    PSE_image_unMean_unSc_all = PSE image calculated for the MRI registered input data for 
    a specific echo. Calculated for each slice for each oxygen cycle and the 
    three cycles (V2). Shape of (4 for overall and the three cycles, vox, vox, NumSlices).
    #
    stats_eachSlice_PSE_SI_all_lung = PSE stats per slice for each oxygen cycle and
    over all oxygen cycles. Stats calculated according to V2. With shape (4 for overall 
    and the three cycles, NumSlices, len(stats_PSE_V2)).
    stats_eachSlice_PSE_recon_all_lung = PSE stats per slice for each oxygen cycle and
    over all oxygen cycles. Stats calculated according to V2. With shape (4 for overall 
    and the three cycles, NumSlices, len(stats_PSE_V2)).
    #
    NumSlices = number of slices acquired.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    sorted_array_metric_sorted = array of the component numbers and metric value
    for the metric under investigation, sorted by the metric value. For use
    when describing the metric in the plot titles. Shape of 
    (number of converged ICA runs, 2, 1); with the first index running over
    the different numbers of components for which ICA reached convergence;
    the second index containing  the component numbers 
    sorted_array_metric_sorted[:,0,:] and their corresponding metric values 
    sorted_array_metric_sorted[:,1,:]; the third index previously ran over the
    different metrics under investigation.
    clims_multi = colourbar limits multiplier on maximum value that is plotted.
    vlims_multi = multiplier of the colourbar *range* plotted, multiplying the lower limit
    of the colourbar. Default value = 0.8.
    vlims_multi2 = (possible array, each will be plotted and saved according to the 
    index into the array) multiplier of the colourbar *range* plotted, multiplying the upper limit
    of the colourbar. Default value = [0.5, 0.25].
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # Colourmap scaling
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    cmap = cm.bwr

    # # # 1) Plot each slice for recon and SI PSE for the mean oxygen 
    # calculation.
    for k in range(len(vlims_multi2)):
        min_max_plotted = [np.min(PSE_image_SI_all[0,:,:,:]), np.max(PSE_image_SI_all[0,:,:,:])]
        min_max_abs = np.max(np.abs(min_max_plotted))
        norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

        # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
        # Also include the metric, component number used and metric value.
        fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4); plt.rcParams["figure.figsize"] = (19.20,10.80)
    
        im1 = ax1.imshow(PSE_image_SI_all[0,:,:,0], norm=norm_unC, cmap=cmap)
        im2 = ax2.imshow(PSE_image_SI_all[0,:,:,1], norm=norm_unC, cmap=cmap)
        im3 = ax3.imshow(PSE_image_SI_all[0,:,:,2], norm=norm_unC, cmap=cmap)
        im4 = ax4.imshow(PSE_image_SI_all[0,:,:,3], norm=norm_unC, cmap=cmap)
        im5 = ax5.imshow(PSE_image_unMean_unSc_all[0,:,:,0], norm=norm_unC, cmap=cmap)
        im6 = ax6.imshow(PSE_image_unMean_unSc_all[0,:,:,1], norm=norm_unC, cmap=cmap)
        im7 = ax7.imshow(PSE_image_unMean_unSc_all[0,:,:,2], norm=norm_unC, cmap=cmap)
        im8 = ax8.imshow(PSE_image_unMean_unSc_all[0,:,:,3], norm=norm_unC, cmap=cmap)

        # Invert plot in y-direction, colourbars, colourbar scaling/limits:
        ax1.invert_yaxis(); fig1.colorbar(im1, ax=ax1); im1.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax2.invert_yaxis(); fig1.colorbar(im2, ax=ax2); im2.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax3.invert_yaxis(); fig1.colorbar(im3, ax=ax3); im3.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax4.invert_yaxis(); fig1.colorbar(im4, ax=ax4); im4.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax5.invert_yaxis(); fig1.colorbar(im5, ax=ax5); im5.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax6.invert_yaxis(); fig1.colorbar(im6, ax=ax6); im6.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax7.invert_yaxis(); fig1.colorbar(im7, ax=ax7); im7.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
        ax8.invert_yaxis(); fig1.colorbar(im8, ax=ax8); im8.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])

        # Axis titles:
        # Include *median* PSE - these are in stats_eachSlice_PSE_SI//recon_all_lung[0 for overall,##slices,1]
        ax1.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,0,1]))
        ax2.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,1,1]))
        ax3.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,2,1]))
        ax4.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,3,1]))
        ax5.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,0,1]))
        ax6.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,1,1]))
        ax7.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,2,1]))
        ax8.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,3,1]))

        fig1.tight_layout()

        if saving_details[0] == 1:
            fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + 'PSE___' + \
            metric_chosen[0] + "G21" + "_C" + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + '_Plot' + str(k) + '.png', dpi=300, bbox_inches='tight')

    # # # 2) Plot each slice separately for recon and SI PSE for the mean oxygen 
    # calculation *AND* each oxygen cycle.
    # Loop over each slice:
    for j in range(NumSlices):
        for k in range(len(vlims_multi2)):
            min_max_plotted = [np.min(PSE_image_SI_all[0,:,:,:]), np.max(PSE_image_SI_all[0,:,:,:])]
            min_max_abs = np.max(np.abs(min_max_plotted))
            norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

            # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
            # Also include the metric, component number used and metric value.
            fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4); plt.rcParams["figure.figsize"] = (19.20,10.80)

            im1 = ax1.imshow(PSE_image_SI_all[0,:,:,j], norm=norm_unC, cmap=cmap)
            im2 = ax2.imshow(PSE_image_SI_all[1,:,:,j], norm=norm_unC, cmap=cmap)
            im3 = ax3.imshow(PSE_image_SI_all[2,:,:,j], norm=norm_unC, cmap=cmap)
            im4 = ax4.imshow(PSE_image_SI_all[3,:,:,j], norm=norm_unC, cmap=cmap)
            im5 = ax5.imshow(PSE_image_unMean_unSc_all[0,:,:,j], norm=norm_unC, cmap=cmap)
            im6 = ax6.imshow(PSE_image_unMean_unSc_all[1,:,:,j], norm=norm_unC, cmap=cmap)
            im7 = ax7.imshow(PSE_image_unMean_unSc_all[2,:,:,j], norm=norm_unC, cmap=cmap)
            im8 = ax8.imshow(PSE_image_unMean_unSc_all[3,:,:,j], norm=norm_unC, cmap=cmap)

            # Invert plot in y-direction, colourbars, colourbar scaling/limits:
            ax1.invert_yaxis(); fig1.colorbar(im1, ax=ax1); im1.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax2.invert_yaxis(); fig1.colorbar(im2, ax=ax2); im2.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax3.invert_yaxis(); fig1.colorbar(im3, ax=ax3); im3.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax4.invert_yaxis(); fig1.colorbar(im4, ax=ax4); im4.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax5.invert_yaxis(); fig1.colorbar(im5, ax=ax5); im5.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax6.invert_yaxis(); fig1.colorbar(im6, ax=ax6); im6.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax7.invert_yaxis(); fig1.colorbar(im7, ax=ax7); im7.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])
            ax8.invert_yaxis(); fig1.colorbar(im8, ax=ax8); im8.set_clim(vmin=min_max_plotted[0]*vlims_multi, vmax=min_max_plotted[1]*vlims_multi2[k])

            # Axis titles:
            # Include *median* PSE - these are in stats_eachSlice_PSE_SI//recon_all_lung[0,1,2,3 - ##PSe V2 type,##slices,1]
            ax1.set_title("Overall median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,j,1]))
            ax2.set_title("Cycle 1 median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[1,j,1]))
            ax3.set_title("Cycle 2 median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[2,j,1]))
            ax4.set_title("Cycle 3 median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[3,j,1]))
            ax5.set_title("Overall median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,j,1]))
            ax6.set_title("Cycle 1 median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[1,j,1]))
            ax7.set_title("Cycle 2 median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[2,j,1]))
            ax8.set_title("Cycle 3 median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[3,j,1]))

            fig1.tight_layout()

            if saving_details[0] == 1:
                fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + 'PSE___' + \
                metric_chosen[0] + "G21" + "_C" + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + \
                "_Cycles_Slice" + str(np.int32(np.rint(j+1))) + '_Plot' + str(k) + '.png', dpi=600, bbox_inches='tight')


    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



def plot_singleMetric_PSEtimeseries(plot_time_value, echo_num, \
    PSE_timeseries_SI_lungMask_MeanEachSlice, \
    PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice, \
    stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung, \
    saving_details, showplot, metric_chosen, sorted_array_metric_sorted):
    """
    Function to plot the PSE timeseries (mean in each slice within the lung masks)
    for the SI and recon PSE.
    *Median* PSE values are given in the plot titles (as calculated using the
    overall oxygen last av_im_num maps).

    Arguments:
    plot_time_value = x-axis values for the dynamic timepoints, shape of (NumDyn,).
    echo_num = the echo number of the data to be loaded and ICA to be performed on.
    PSE_timeseries_SI_lungMask_MeanEachSlice = timeseries of the mean PSE value within
    the lung masks for each slice of the SI PSE. Shape (NumDyn,).
    PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice = timeseries of the mean PSE 
    value within the lung masks for each slice of the recon PSE (unMean and unScaled).
    Shape (NumDyn,).
    stats_eachSlice_PSE_SI_all_lung = PSE stats per slice for each oxygen cycle and
    over all oxygen cycles. Stats calculated according to V2. With shape (4 for overall 
    and the three cycles, NumSlices, len(stats_PSE_V2)).
    stats_eachSlice_PSE_recon_all_lung = PSE stats per slice for each oxygen cycle and
    over all oxygen cycles. Stats calculated according to V2. With shape (4 for overall 
    and the three cycles, NumSlices, len(stats_PSE_V2)).
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    sorted_array_metric_sorted = array of the component numbers and metric value
    for the metric under investigation, sorted by the metric value. For use
    when describing the metric in the plot titles. Shape of 
    (number of converged ICA runs, 2, 1); with the first index running over
    the different numbers of components for which ICA reached convergence;
    the second index containing  the component numbers 
    sorted_array_metric_sorted[:,0,:] and their corresponding metric values 
    sorted_array_metric_sorted[:,1,:]; the third index previously ran over the
    different metrics under investigation.
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
    plt.close(fig1)

    # Plot timeseries of PSE as 4x2 subplots.
    fig1, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(4, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)

    ax1.plot(plot_time_value, PSE_timeseries_SI_lungMask_MeanEachSlice[:,0]); ax1.set_xlabel("Time /s") 
    ax2.plot(plot_time_value, PSE_timeseries_SI_lungMask_MeanEachSlice[:,1]); ax2.set_xlabel("Time /s")
    ax3.plot(plot_time_value, PSE_timeseries_SI_lungMask_MeanEachSlice[:,2]); ax3.set_xlabel("Time /s")
    ax4.plot(plot_time_value, PSE_timeseries_SI_lungMask_MeanEachSlice[:,3]); ax4.set_xlabel("Time /s")
    ax5.plot(plot_time_value, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice[:,0]); ax5.set_xlabel("Time /s")
    ax6.plot(plot_time_value, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice[:,1]); ax6.set_xlabel("Time /s")
    ax7.plot(plot_time_value, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice[:,2]); ax7.set_xlabel("Time /s")
    ax8.plot(plot_time_value, PSE_timeseries_recon_unMean_unSc_lungMask_MeanEachSlice[:,3]); ax8.set_xlabel("Time /s")
    ax1.set_title("Mean SI PSE in slice 1, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,0,1]))
    ax2.set_title("Mean SI PSE in slice 2, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,1,1]))
    ax3.set_title("Mean SI PSE in slice 3, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,2,1]))
    ax4.set_title("Mean SI PSE in slice 4, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,3,1]))
    ax5.set_title("Mean Recon PSE in slice 1, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,0,1]))
    ax6.set_title("Mean Recon PSE in slice 2, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,1,1]))
    ax7.set_title("Mean Recon PSE in slice 3, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,2,1]))
    ax8.set_title("Mean Recon PSE in slice 4, echo " + str(np.int32(np.rint(echo_num))) \
        + ". Median = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,3,1]))

    fig1.tight_layout()

    # Save figure if required.
    if saving_details[0] == 1:
        # As V2 has a restriction on the number of components, include G21 in Figure saving name.
        fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + 'PSEtimeseries___' + \
            metric_chosen[0] + "G21" + "_C" + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + \
            "_PSEtimeseries" + '.png', dpi=600, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return    



def plot_singleMetric_PSEmaps_SIandRecon_setCBAR(PSE_image_SI_all, PSE_image_unMean_unSc_all, \
    stats_eachSlice_PSE_SI_all_lung, stats_eachSlice_PSE_recon_all_lung, \
    NumSlices, saving_details, showplot, \
    metric_chosen, sorted_array_metric_sorted, \
    clim_lower, clim_upper, \
    clims_multi=0.8, vlims_multi=0.8, vlims_multi2=[0.5, 0.25]):
    """
    Plot reconstructed and SI PSE maps for each slice
    for the mean oxygen cycle PSE.
    Assuming g21, single metric of interest.
    Colourbar limits plotted - **common to all subjects** as specified
    by clim_lower and clim_upper.

    Plot each slice for recon and SI PSE for the mean oxygen
    calculation.
    

    Arguments:
    PSE_image_SI_all = PSE image calculated for the MRI registered input data for 
    a specific echo. Calculated for each slice for each oxygen cycle and the 
    three cycles (V2). Shape of (4 for overall and the three cycles, vox, vox, NumSlices).
    PSE_image_unMean_unSc_all = PSE image calculated for the MRI registered input data for 
    a specific echo. Calculated for each slice for each oxygen cycle and the 
    three cycles (V2). Shape of (4 for overall and the three cycles, vox, vox, NumSlices).
    #
    stats_eachSlice_PSE_SI_all_lung = PSE stats per slice for each oxygen cycle and
    over all oxygen cycles. Stats calculated according to V2. With shape (4 for overall 
    and the three cycles, NumSlices, len(stats_PSE_V2)).
    stats_eachSlice_PSE_recon_all_lung = PSE stats per slice for each oxygen cycle and
    over all oxygen cycles. Stats calculated according to V2. With shape (4 for overall 
    and the three cycles, NumSlices, len(stats_PSE_V2)).
    #
    NumSlices = number of slices acquired.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). [0] gives a value
    regarding whether to save the plot = 0 for not saving or 1 to save; [1] gives the location
    of where the figures are to be saved; [2] gives the name of the parameter that is being plotted
    /that ICA was applied to; [3] gives the metrics_methods - info about the metrics used and 
    if any restrictions were placed on the number of components (e.g. > 21, 'G21').
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    metric_chosen = the metric under investigation (SINGLE METRIC ONLY), in the
    form of a list with only one element.
    sorted_array_metric_sorted = array of the component numbers and metric value
    for the metric under investigation, sorted by the metric value. For use
    when describing the metric in the plot titles. Shape of 
    (number of converged ICA runs, 2, 1); with the first index running over
    the different numbers of components for which ICA reached convergence;
    the second index containing  the component numbers 
    sorted_array_metric_sorted[:,0,:] and their corresponding metric values 
    sorted_array_metric_sorted[:,1,:]; the third index previously ran over the
    different metrics under investigation.
    clim_lower = lower limit of the PSE map plots.
    clim_upper = upper limit of the PSE map plots.
    # # FOR PSE MAP PLOTTING, e.g.
    # clim_lower = [35, 30, 25, 20] # Negative of this
    # clim_upper = [5]
    clims_multi = colourbar limits multiplier on maximum value that is plotted.
    vlims_multi = multiplier of the colourbar *range* plotted, multiplying the lower limit
    of the colourbar. Default value = 0.8.
    vlims_multi2 = (possible array, each will be plotted and saved according to the 
    index into the array) multiplier of the colourbar *range* plotted, multiplying the upper limit
    of the colourbar. Default value = [0.5, 0.25].
    Returns:
    None, figure plotted and saved if required.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)

    # Colourmap scaling
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    cmap = cm.bwr


    # # # 1) Plot each slice for recon and SI PSE for the mean oxygen 
    # calculation.
    for k in range(len(clim_lower)):
        min_max_plotted = [np.min(PSE_image_SI_all[0,:,:,:]), np.max(PSE_image_SI_all[0,:,:,:])]
        min_max_abs = np.max(np.abs(min_max_plotted))
        norm_unC = colors.TwoSlopeNorm(vmin=-min_max_abs*clims_multi, vcenter=0, vmax=min_max_abs*clims_multi)

        # In the title, include the mean PSE value (calculated from the map) for the slice being plotted.
        # Also include the metric, component number used and metric value.
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)
        fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2); plt.rcParams["figure.figsize"] = (19.20,10.80)

        im1 = ax1.imshow(PSE_image_SI_all[0,:,:,0], norm=norm_unC, cmap=cmap)
        im2 = ax2.imshow(PSE_image_SI_all[0,:,:,1], norm=norm_unC, cmap=cmap)
        im3 = ax3.imshow(PSE_image_SI_all[0,:,:,2], norm=norm_unC, cmap=cmap)
        im4 = ax4.imshow(PSE_image_SI_all[0,:,:,3], norm=norm_unC, cmap=cmap)
        im5 = ax5.imshow(PSE_image_unMean_unSc_all[0,:,:,0], norm=norm_unC, cmap=cmap)
        im6 = ax6.imshow(PSE_image_unMean_unSc_all[0,:,:,1], norm=norm_unC, cmap=cmap)
        im7 = ax7.imshow(PSE_image_unMean_unSc_all[0,:,:,2], norm=norm_unC, cmap=cmap)
        im8 = ax8.imshow(PSE_image_unMean_unSc_all[0,:,:,3], norm=norm_unC, cmap=cmap)

        # Invert plot in y-direction, colourbars, colourbar scaling/limits:
        ax1.invert_yaxis(); fig1.colorbar(im1, ax=ax1); im1.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax2.invert_yaxis(); fig1.colorbar(im2, ax=ax2); im2.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax3.invert_yaxis(); fig1.colorbar(im3, ax=ax3); im3.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax4.invert_yaxis(); fig1.colorbar(im4, ax=ax4); im4.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax5.invert_yaxis(); fig2.colorbar(im5, ax=ax5); im5.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax6.invert_yaxis(); fig2.colorbar(im6, ax=ax6); im6.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax7.invert_yaxis(); fig2.colorbar(im7, ax=ax7); im7.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])
        ax8.invert_yaxis(); fig2.colorbar(im8, ax=ax8); im8.set_clim(vmin=-clim_lower[k], vmax=clim_upper[0])

        # Axis titles:
        # Include *median* PSE - these are in stats_eachSlice_PSE_SI//recon_all_lung[0 for overall,##slices,1]
        ax1.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,0,1]))
        ax2.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,1,1]))
        ax3.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,2,1]))
        ax4.set_title("Median SI PSE = {:.1f}".format(stats_eachSlice_PSE_SI_all_lung[0,3,1]))
        ax5.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,0,1]))
        ax6.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,1,1]))
        ax7.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,2,1]))
        ax8.set_title("Median Recon PSE = {:.1f}".format(stats_eachSlice_PSE_recon_all_lung[0,3,1]))

        fig1.tight_layout()
        fig2.tight_layout()

        if saving_details[0] == 1:
            fig1.savefig(saving_details[1] + saving_details[2] + 'metrics_' + 'PSE___' + \
            metric_chosen[0] + "G21" + "_C" + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + '_Plot' + str(k) + '_ClimsSet_' + str(clim_lower[k]) + '_' + str(clim_upper[0]) + '_SI.png', dpi=600, bbox_inches='tight')
            fig2.savefig(saving_details[1] + saving_details[2] + 'metrics_' + 'PSE___' + \
            metric_chosen[0] + "G21" + "_C" + str(np.int32(np.rint(sorted_array_metric_sorted[0,0,0]))) + '_Plot' + str(k) + '_ClimsSet_' + str(clim_lower[k]) + '_' + str(clim_upper[0]) + '_Recon.png', dpi=600, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return

def plot_lineplot_RMSordered_metric(plot_x_axis, plot_y_axis, num_components, saving_details, \
    showplot, further_details, halfTimepoints, \
    metric_args_plot, dpi_save, RunNum_use, \
    subplots_num_in_figure=9, subplots_num_row=3):
    """
    Plot and save ordered timeseries/frequency spectra (for any number of 
    component subplots). The RMS metric value of the ordered components
    will be included in the subplot titles.

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = *ordered* time series/frequency spectra amplitudes to be plotted
    on the y-axis. Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    metric_args_plot = a list of [metric_chosen, metric_values_ordered]. metric_chosen
    is the name of the frequency RMS ordering method used. metric_values_ordered contains
    the metric values (ordered) for all components (ordered) being plotted.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            axs[axes_c].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]))
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            #
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            # SAVE METRIC NAME IN FIGURE SAVING NAME, but values in title.
            if RunNum_use > 1:
                figs[idx].savefig(saving_details[1] + '_Run' + str(RunNum_use) + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')
            else:
                figs[idx].savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')


    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}
        
        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                an_axs_sub[j].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                    ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]))
                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()
        
        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')
        
        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)
        
        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum_use > 1:
                figs_sub.savefig(saving_details[1] + '_Run' + str(RunNum_use) + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')
            else:
                figs_sub.savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



def plot_lineplot_RMSordered_metric_CA_ordering_2(plot_x_axis, plot_y_axis, num_components, saving_details, \
    showplot, further_details, halfTimepoints, \
    metric_args_plot, dpi_save, \
    title_addition, \
    subplots_num_in_figure=9, subplots_num_row=3):
    """
    Plot and save ordered timeseries/frequency spectra (for any number of 
    component subplots). The RMS metric value of the ordered components
    will be included in the subplot titles.
    ALSO include the **original** component ordering number in the title.
    - Or whatever is also passed, and additional title to indicate what this number is ("title_addition").
    Assume integers.
    The extra to be plotted is in metric_args_plot [2].

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = *ordered* time series/frequency spectra amplitudes to be plotted
    on the y-axis. Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    metric_args_plot = a list of [metric_chosen, metric_values_ordered, RMS_allFreq_P_relMax_argsort_indices]. 
    metric_chosen is the name of the frequency RMS ordering method used. metric_values_ordered
    contains the metric values (ordered) for all components (ordered) being plotted. 
    RMS_allFreq_P_relMax_argsort_indices contains details of the original component analysis ordering
    output, with the index of the CA output given, but ordered by the Spearman correlation coefficient.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            axs[axes_c].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]) + \
                ", " + "(#{:.0f}".format(title_addition[axes_c]) + ")")
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            # SAVE METRIC NAME IN FIGURE SAVING NAME, but values in title.
            figs[idx].savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                'Plot' + str(np.int32(np.rint(idx+1))) + '__OrigOrder.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')


    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}
        
        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                an_axs_sub[j].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                    ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]) + \
                    ", " + "(#{:.0f}".format(title_addition[axes_c]) + ")")
                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()
        
        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')
        
        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)
        
        # Save figure if required.
        if saving_details[0] == 1:
            figs_sub.savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '__OrigOrder.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_lineplot_RMSordered_metric_CA_ordering(plot_x_axis, plot_y_axis, num_components, saving_details, \
    showplot, further_details, halfTimepoints, \
    metric_args_plot, dpi_save, \
    title_addition, \
    subplots_num_in_figure=9, subplots_num_row=3):
    """
    Plot and save ordered timeseries/frequency spectra (for any number of 
    component subplots). The RMS metric value of the ordered components
    will be included in the subplot titles.
    ALSO include the **original** component ordering number in the title.
    - Or whatever is also passed, and additional title to indicate what this number is ("title_addition").
    Assume integers.
    The extra to be plotted is in metric_args_plot [2].

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = *ordered* time series/frequency spectra amplitudes to be plotted
    on the y-axis. Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    metric_args_plot = a list of [metric_chosen, metric_values_ordered, RMS_allFreq_P_relMax_argsort_indices]. 
    metric_chosen is the name of the frequency RMS ordering method used. metric_values_ordered
    contains the metric values (ordered) for all components (ordered) being plotted. 
    RMS_allFreq_P_relMax_argsort_indices contains details of the original component analysis ordering
    output, with the index of the CA output given, but ordered by the Spearman correlation coefficient.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            axs[axes_c].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]) + \
                ", " + title_addition + " = {:.0f}".format(metric_args_plot[2][axes_c]))
            axs[axes_c].set_xlabel(further_details[1])
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            # SAVE METRIC NAME IN FIGURE SAVING NAME, but values in title.
            figs[idx].savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                'Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')


    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}
        
        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                an_axs_sub[j].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                    ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]) + \
                    ", " + title_addition + " = {:.0f}".format(metric_args_plot[2][axes_c]))
                an_axs_sub[j].set_xlabel(further_details[1])
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()
        
        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')
        
        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)
        
        # Save figure if required.
        if saving_details[0] == 1:
            figs_sub.savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return


def plot_lineplot_RMSordered_metric_CA_ordering_PSE(plot_x_axis, plot_y_axis, num_components, saving_details, \
    showplot, further_details, halfTimepoints, \
    metric_args_plot, dpi_save, \
    title_addition, \
    subplots_num_in_figure, subplots_num_row):
    """
    Plot and save ordered timeseries/frequency spectra (for any number of 
    component subplots). The RMS metric value of the ordered components
    will be included in the subplot titles.
    ALSO include the **original** component ordering number in the title.
    - Or whatever is also passed, and additional title to indicate what this number is ("title_addition").
    Assume integers.
    The extra to be plotted is in metric_args_plot [2].

    PSE - include y-axis title.

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = *ordered* time series/frequency spectra amplitudes to be plotted
    on the y-axis. Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    metric_args_plot = a list of [metric_chosen, metric_values_ordered, RMS_allFreq_P_relMax_argsort_indices]. 
    metric_chosen is the name of the frequency RMS ordering method used. metric_values_ordered
    contains the metric values (ordered) for all components (ordered) being plotted. 
    RMS_allFreq_P_relMax_argsort_indices contains details of the original component analysis ordering
    output, with the index of the CA output given, but ordered by the Spearman correlation coefficient.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.int32(np.linspace(1, subplots_num[0], subplots_num[0]))
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0
    # print(subplots_num_in_figure, subplots_num_row, num_components)
    # print(subplots_num, subplots_num_column)
    # print(separateFigures_nums)

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        for j in range(subplots_num_in_figure):
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            axs[axes_c].set_title(title_addition[0] + " = {:.0f}".format(metric_args_plot[1][axes_c]) + \
                ", " + title_addition[1] + " = {:.0f}".format(metric_args_plot[2][axes_c]) + \
                ", " + "metric = {:.3f}".format(metric_args_plot[3][axes_c]))
            axs[axes_c].set_xlabel(further_details[1])
            axs[axes_c].set_ylabel("PSE (%)")
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1


        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            # SAVE METRIC NAME IN FIGURE SAVING NAME, but values in title.
            figs[idx].savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                'Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')


    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}
        
        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                an_axs_sub[j].set_title(title_addition[0] + " = {:.0f}".format(metric_args_plot[1][axes_c]) + \
                    ", " + title_addition[1] + " = {:.0f}".format(metric_args_plot[2][axes_c]) + \
                    ", " + "metric = {:.3f}".format(metric_args_plot[3][axes_c]))
                an_axs_sub[j].set_xlabel(further_details[1])
                an_axs_sub[j].set_ylabel("PSE (%)")
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()
        
        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')
        
        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)
        
        # Save figure if required.
        if saving_details[0] == 1:
            figs_sub.savefig(saving_details[1] + '__' + metric_args_plot[0] + '_' + further_details[2] + '_' + str(subplots_num_row) + \
                'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



def plot_lineplot_multiple_title_option_YaxisLims(\
    plot_x_axis, plot_y_axis, num_components, saving_details, \
    showplot, further_details, halfTimepoints, \
    metric_args_plot, dpi_save, RunNum_use, \
    PSE_yaxis_range, further_details_y_axis, group_plot_details, echo_num, \
    subplots_num_in_figure=9, subplots_num_row=3):
    """
    Plot and save ordered timeseries/frequency spectra (for any number of 
    component subplots).
    # # PSE timeseries - and can supply/set PSE y-axis range, e.g. for 
    plotting PSE recon ICA OE component from multiple subjects.
    Title - can easily change, but currently set to the NumC... or not include perhaps.
    #((The RMS metric value of the ordered components will be included in the subplot titles.))

    Arguments:
    plot_x_axis = x-axis values, shape of (NumDyn,).
    plot_y_axis = *ordered* time series/frequency spectra amplitudes to be plotted
    on the y-axis. Shape of (NumDyn,num_components).
    num_components = number of components.
    saving_details = details to use for naming the figures that are being saved, and
    whether to save the figure. This is a list of the shape (3,). The first
    value is 0 for not saving or 1 to save; the second value is the location
    of where the figures are to be saved; the third value is the name of the 
    parameter that being plotted/that ICA was applied to.
    showplot = 0 (no) or 1 (yes) for whether to show the plot using plt.show().
    further_details = a list of three elements giving additions to the title [0], x-axis
    label [1] and figure saving name [2]. e.g. ["time series", "time /s", "timeseries"].
    Or for frequencies, ["frequency spectra", "frequency /Hz", "freq"]
    halfTimepoints = half timepoints used to plot only the positive frequencies.
    metric_args_plot = a list of [metric_chosen, metric_values_ordered]. metric_chosen
    is the name of the frequency RMS ordering method used. metric_values_ordered contains
    the metric values (ordered) for all components (ordered) being plotted.
    dpi_save = the dpi setting for image saving (e.g. 300 or 600).
    PSE_yaxis_range = [plot_PSE_yaxis_limited, PSE_yaxis_lower, PSE_yaxis_upper]
    further_details_y_axis = [0, "PSE (%)", "PSE"] --> [0, Y-axis title, what is being plotted for use in saving name].
    group_plot_details = 'NonS_HVRV' e.g. or 'CS_HV' - To identify the group of subjects being plotted for use in saving name.
    subplots_num_in_figure = number of subplots per figure, default = 9.
    subplots_num_row = number of *ROWS* of subplots, default = 3 (for 3 x 3 subplots
    per figure).
    # Previous default = 9 subplots in total; 3 per row; 3 per column.
    # But when plotting all components...
    # - 16 subplots in total; 4 per row; 4 per column. (_4Plot)
    # = 4 rows and 4 columns.
    # - 20 subplots in total; 4 per row; 5 per column. (_5Plot)
    # = 5 rows and 4 columns.
    # (as a list and loop over in main function call)
    # --> subplots_num_in_figure_list = [16, 20]
    # --> subplots_num_row_list = [4, 5]
    # ==> subplots_num_row = the *number of rows of subplots*, i.e.
    # the number of subplots per column.
    # subplots_num_in_figure; subplots_num_row
    Returns:
    None - just saves figures and plots.

    """
    # Repeat plotting of first figure, sometimes doesn't work properly.
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2); plt.rcParams["figure.figsize"] = (19.20,10.80); plt.close(fig1)
    
    # If plotting frequencies, truncate to plot positive frequencies only.
    if further_details[2] == "freq":
        plot_y_axis = np.array(plot_y_axis[:halfTimepoints,:])

    # Calculate the number of 'full' figures and the number of remaining
    # subplots in the additional 'half-full' figure.
    # The number of full figures is the quotient and the number of remaining
    # subplots are given by the remainder using np.divmod().
    subplots_num = np.divmod(num_components, subplots_num_in_figure)
    subplots_num_column = np.int32(np.rint(np.divide(subplots_num_in_figure, subplots_num_row)))
    # First loop and plot the full figure subplots. Create a dictionary 
    # of figure identifiers (figs) and axes (axs) to plot the figures/axes
    # in a loop.
    figs={}
    axs={}
    # Create array of figure identifiers to loop over.
    separateFigures_nums = np.linspace(1, subplots_num[0], subplots_num[0])
    # axes_c = a counter for the axis number to be used when plotting the 
    # axis titles.
    axes_c = 0

    add_saving_name = []
    if PSE_yaxis_range[0] == 1:
        # If setting y-axis limits, include in saving name
        add_saving_name.append(str(PSE_yaxis_range[1]) + 'to' + str(PSE_yaxis_range[2]))

    # Loop over number of separate figures
    for idx,separateFigures_nums_number in enumerate(separateFigures_nums):
        # print(axes_c)
        figs[idx]=plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # figs[idx] relates to each figure.
        # Loop over the separate figures and plot the subplots.
        for j in range(subplots_num_in_figure):
            # axs[axes_c] relates to each axis.
            axs[axes_c] = figs[idx].add_subplot(subplots_num_row, subplots_num_column, j+1)
            axs[axes_c].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
            # axs[axes_c].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
            #     ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]))
            axs[axes_c].set_xlabel(further_details[1])
            axs[axes_c].set_ylabel(further_details_y_axis[1])
            if PSE_yaxis_range[0] == 1:
                # Set y-axis limits
                axs[axes_c].set_ylim((PSE_yaxis_range[1], PSE_yaxis_range[2]))
            figs[idx].tight_layout()
            # Increase the axis counter.
            axes_c = axes_c + 1
            
        figs[idx].set_figheight(10.80)
        figs[idx].set_figwidth(19.20)
        # Save figure if required.
        if saving_details[0] == 1:
            # SAVE METRIC NAME IN FIGURE SAVING NAME, but values in title.
            if RunNum_use > 1:
                figs[idx].savefig(saving_details[1] + '_Run' + str(RunNum_use) + 'Echo' + str(echo_num) + '__' + metric_args_plot[0] + '_' + \
                    further_details_y_axis[2] + '_' + further_details[2] + '_' + add_saving_name[0] + '__' + group_plot_details + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')
            else:
                figs[idx].savefig(saving_details[1] + 'Echo' + str(echo_num) + '__' + metric_args_plot[0] + '_' + \
                    further_details_y_axis[2] + '_' + further_details[2] + '_' + add_saving_name[0] + '__' + group_plot_details + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(idx+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    # Next plot the final partially filled figure. Only do this if there are
    # remainder subplots, i.e. subplots_num[1] != 0
    if subplots_num[1] != 0:
        figs_sub = plt.figure(figsize=(19.20,10.80))
        # plt.rcParams["figure.figsize"] = (19.20,10.80)
        # Again, create a dictionary of axes (axs) identifiers to plot the axes
        # of the final figure.
        an_axs_sub = {}
        
        # Loop over the number of subplots in the figure (whether they are to be plotted, i.e.
        # remainder axes or not) to maintain the correct scaling/sizes of the axes.
        for j in range(subplots_num_in_figure):
            # print(axes_c, an_axs_sub)
            # an_axs_sub[j] relates to each axis.
            # Add an axis for all subplots whether they are to be plotted or not.
            an_axs_sub[j] = figs_sub.add_subplot(subplots_num_row, subplots_num_column, j+1)
            if j < subplots_num[1]:
                # If the subplot axis is one of the remainder axes, plot.
                an_axs_sub[j].plot(plot_x_axis, plot_y_axis[:,np.int32(np.rint(axes_c))])
                # an_axs_sub[j].set_title("Component " + str(np.int32(np.rint(axes_c+1))) + \
                #     ", " + "metric = {:.3f}".format(metric_args_plot[1][axes_c]))
                an_axs_sub[j].set_xlabel(further_details[1])
                an_axs_sub[j].set_ylabel(further_details_y_axis[1])
                if PSE_yaxis_range[0] == 1:
                    # Set y-axis limits
                    an_axs_sub[j].set_ylim((PSE_yaxis_range[1], PSE_yaxis_range[2]))
            else:
                # If the subplot axis is one of the remainder axes, set the title to be empty (" ")
                # to maintain the axes scaling/sizes.
                an_axs_sub[j].set_title(" ")
            axes_c = axes_c + 1
        figs_sub.tight_layout()
        
        # Finally, hide all axes that are not the remainder axes.
        for j in range(subplots_num_in_figure):
            if j >= subplots_num[1]:
                an_axs_sub[j].axis('off')
        
        figs_sub.set_figheight(10.80)
        figs_sub.set_figwidth(19.20)
        
        # Save figure if required.
        if saving_details[0] == 1:
            if RunNum_use > 1:
                figs_sub.savefig(saving_details[1] + '_Run' + str(RunNum_use) + 'Echo' + str(echo_num) + '__' + metric_args_plot[0] + '_' + \
                    further_details_y_axis[2] + '_' + further_details[2] + '_' + add_saving_name[0] + '__' + group_plot_details + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')
            else:
                figs_sub.savefig(saving_details[1] + 'Echo' + str(echo_num) + '__' + metric_args_plot[0] + '_' + \
                    further_details_y_axis[2] + '_' + further_details[2] + '_' + add_saving_name[0] + '__' + group_plot_details + '_' + str(subplots_num_row) + \
                    'Plot' + str(np.int32(np.rint(subplots_num[0]+1))) + '.png', dpi=dpi_save, bbox_inches=0) # #, bbox_inches='tight')

    if showplot == 0:
        plt.close('all')
    else:
        plt.show()

    return



if __name__ == "__main__":
    # ...
    a = 1