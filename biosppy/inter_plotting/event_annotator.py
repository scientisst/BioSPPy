# -*- coding: utf-8 -*-
"""
biosppy.inter_plotting.ecg
-------------------

This module provides an interactive UI for manual annotation of time stamps on biosignals.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
import os
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from tkinter import messagebox, filedialog
from matplotlib.backend_bases import MouseButton
from biosppy import tools as st, storage
from biosppy.storage import store_txt
from .config import list_functions


def rescale_signals(input_signal: np.ndarray, new_min, new_max):
    # first normalize from 0 to 1
    input_signal = (input_signal - np.min(input_signal)) / (np.max(input_signal) - np.min(input_signal))

    # then normalize to the [new_min, new_max] range
    input_signal = input_signal * (new_max - new_min) + new_min

    return input_signal


plot_colors = list(mcolors.TABLEAU_COLORS.values())


def milliseconds_to_samples(time_milliseconds: int, sampling_rate: float):
    return int(time_milliseconds * (int(sampling_rate) / 1000))


class event_annotator:
    """Opens an editor of event annotations of the input biosignal modality."""
    global list_functions

    def __init__(self, input_raw_signal, mdata, window_size=6.0, window_stride=1.5, moving_avg_wind_sz=700,
                 templates_dir=None):
        """Initializes the event annotator.

        Parameters
        ----------
        input_raw_signal : np.ndarray
            Input raw signal.
        mdata : dict
            Metadata as provided by `storage` utility functions.
        window_size : float, optional
            Size of the moving window used for navigation in seconds.
        window_stride : float, optional
            Sride of the moving window used for navigation in seconds.
        moving_avg_wind_sz : int, optional
            Window size of the moving average filter used to obtain the filtered signals in milliseconds.
        """

        # If no directory is provided, it remains None -> when saving / loading templates, default dir will open
        self.templates_dir = templates_dir

        # Extracting metadata
        self.sampling_rate = mdata['sampling_rate']
        self.labels = mdata['labels']

        # Sub-signals or channels of the same signal modality
        self.nr_sub_signals = len(self.labels)

        self.raw_signal = input_raw_signal

        self.moving_window_size = window_size
        self.window_shift = window_stride

        self.time_arr = np.arange(0, self.raw_signal.shape[0] / self.sampling_rate, 1 / self.sampling_rate)

        self.nr_samples = self.raw_signal.shape[0]

        # tkinter figures and plots
        self.root = Tk()
        self.root.resizable(False, False)

        self.figure = plt.Figure(figsize=(8, 4), dpi=100)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylabel("Amplitude [-]")
        self.ax.set_xlabel("Time [s]")

        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, columnspan=1, sticky='w', padx=10)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_click_ax)

        self.toolbarFrame = Frame(master=self.root)
        self.toolbarFrame.grid(row=1, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbarFrame)

        self.main_plots = {}

        # it's easier to expand one dim when single-channel than to check the dimensions for every loop in which signals are plotted.
        if self.raw_signal.ndim == 1:
            self.raw_signal = np.expand_dims(self.raw_signal, axis=-1)

        print(self.raw_signal.shape)
        print(self.time_arr.shape)

        for label, i in zip(self.labels, range(self.nr_sub_signals)):
            # in next version we need to have this structure different, for example to have 1 raw and 1 filt per channel

            # filter signal with a standard moving average with window=1s
            tmp_filt_signal, _ = st.smoother(self.raw_signal[:, i],
                                             size=milliseconds_to_samples(moving_avg_wind_sz,
                                                                          self.sampling_rate))  # 500 ms

            tmp_filt_signal = rescale_signals(tmp_filt_signal, np.min(self.raw_signal[:, i]),
                                              np.max(self.raw_signal[:, i]))

            self.main_plots['{}_filt'.format(label)] = [None, tmp_filt_signal]
            self.main_plots['{}_raw'.format(label)] = [
                self.ax.plot(self.time_arr, self.raw_signal[:, i], linewidth=0.8, color=plot_colors[i]),
                self.raw_signal[:, i]]

        # Saving x- and y-lims of original signals
        self.original_xlims = self.ax.get_xlim()
        self.original_ylims = self.ax.get_ylim()

        # Dictionary to store templates (annotated events)
        self.template_pack = {}

        # Setting plot title for one or multiple signal modalities
        if self.nr_sub_signals == 1:
            self.ax.set_title('Signal: {}'.format(self.labels[0]))
        else:
            signal_substr = ', '.join(self.labels)
            self.ax.set_title('Signals: ' + signal_substr)

        # Controllable On / Off variables
        self.var_edit_plots = IntVar()  # enable / disable template editing
        self.var_toggle_Ctrl = IntVar()  # Fit ylims to amplitude within displayed window / show original ylims
        self.var_view_filtered_signal = IntVar()  # Enable / disable view overlapped filtered signal
        self.var_zoomed_in = False  # Zoom-in / Zoom-out

        # Variable to store last Zoom-in configuration, so that when zooming-in again it returns back to previous
        # zoomed-in view
        self.zoomed_in_lims = [0, self.moving_window_size]

        # Check Button to allow editing templates
        self.template_checkbox = Checkbutton(self.root, text='Edit Templates', variable=self.var_edit_plots, onvalue=1,
                                             offvalue=0,
                                             command=self.template_checkbox_on_click)

        self.template_checkbox.grid(row=0, column=1)

        self.template_ind = 0
        self.moving_window_ind = 0

        # Button to save templates (as they are currently shown)
        self.saving_button = Button(self.root, text='Save templates', width=25, command=self.save_templates_file)
        self.saving_button.grid(row=0, column=2)

        # Button to optionally view filtered signal
        self.overlap_raw = Checkbutton(self.root, text='View filtered signal', variable=self.var_view_filtered_signal,
                                       onvalue=1,
                                       offvalue=0,
                                       command=self.view_filtered_signals)
        self.overlap_raw.grid(row=1, column=1)

        # Drop menu for computing or loading templates
        self.template_opt_menu = Menubutton(self.root, text="Add templates", relief='raised')
        self.template_opt_menu.grid(row=1, column=2, sticky=W)
        self.template_opt_menu.menu = Menu(self.template_opt_menu, tearoff=0)
        self.template_opt_menu["menu"] = self.template_opt_menu.menu
        self.template_opt_menu.update()

        for k, v in list_functions.items():

            if k != 'Load Templates':
                sub_menu = Menu(self.template_opt_menu.menu)

                self.template_opt_menu.menu.add_cascade(label=k, menu=sub_menu)

                for sub_k, sub_v in v.items():
                    sub_menu.add_command(label=sub_k,
                                         command=lambda option="{},{}".format(k, sub_k): self.update_templates_options(option))

            else:
                self.template_opt_menu.menu.add_command(label=k,
                                                        command=lambda option='Load Templates': self.update_templates_options(option))

        self.last_save = True
        self.moving_onset = None

        self.root.bind_all('<Key>', self.template_navigator)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def view_filtered_signals(self):
        """Enables or disables the display of the filtered signal(s)."""

        # if user toggled "view filtered signals"
        if self.var_view_filtered_signal.get() == 1:

            for label, i in zip(self.labels, range(self.nr_sub_signals)):
                tmp_ax, = self.ax.plot(self.time_arr,
                                       self.main_plots['{}_filt'.format(label)][1], linewidth=0.5, alpha=0.6,
                                       color=plot_colors[i])
                self.main_plots['{}_filt'.format(label)].insert(0, tmp_ax)

        else:
            for label, i in zip(self.labels, range(self.nr_sub_signals)):
                # Note: this .remove() function is from matplotlib!
                self.main_plots['{}_filt'.format(label)][0].remove()
                del self.main_plots['{}_filt'.format(label)][0]

        self.canvas_plot.draw()

    def update_templates_options(self, option):
        """Dropdown menu for template options (load or compute templates). Currently only supporting loading."""

        # start by removing all existing templates
        for template in list(self.template_pack.keys()):
            try:
                self.template_pack[template].remove()
                del self.template_pack[template]

            except:
                pass

        # then apply signal processing as requested by user
        if option == "load templates":
            files = [('Text Document', '*.txt'),
                     ('Comma-Separated Values file', '*.csv')]

            # If more than one sub-signal, guess the name of the first as default
            init_filename_guess = self.labels[0] + "_templates.txt"

            templates_path = filedialog.askopenfile(filetypes=files, defaultextension=files,
                                                    title="Loading ACC templates file",
                                                    initialdir=self.templates_dir,
                                                    initialfile=init_filename_guess)

            try:
                loaded_templates, _ = storage.load_txt(templates_path.name)

                # adding loaded templates
                for template in loaded_templates:
                    template_temp = self.ax.vlines(template, self.original_ylims[0], self.original_ylims[1],
                                                   colors='#FF6B66')
                    self.template_pack[template] = template_temp
            except:
                pass

        else:
            chosen_modality = option.split(",")[0]
            chosen_algorithm = option.split(",")[1]

            preprocess = list_functions[chosen_modality][chosen_algorithm]["preprocess"]
            function = list_functions[chosen_modality][chosen_algorithm]["function"]
            template_key = list_functions[chosen_modality][chosen_algorithm]["template_key"]

            # only computing for one of the sub-signals (the first one)
            if function is None:
                input_extraction_signal = self.raw_signal[:, 0]
            else:
                input_extraction_signal = preprocess(self.raw_signal[:, 0], sampling_rate=self.sampling_rate)
                input_extraction_signal = input_extraction_signal['filtered']

            templates = function(input_extraction_signal, sampling_rate=self.sampling_rate)[template_key]

            # print(templates)

            # adding loaded templates
            for template in templates:
                print(template / self.sampling_rate)

                template_temp = self.ax.vlines(template / self.sampling_rate, self.original_ylims[0],
                                               self.original_ylims[1],
                                               colors='#FF6B66')

                self.template_pack[template] = template_temp

        # finally re-draw everything back
        self.canvas_plot.draw()

    def save_templates_file(self):
        """Saves the currently edited templates."""

        files = [('Text Document', '*.txt'),
                 ('Comma-Separated Values file', '*.csv')]

        # If more than one sub-signal, guess the name of the first as default
        init_filename_guess = self.labels[0] + "_templates.txt"

        saving_path_main = filedialog.asksaveasfile(filetypes=files, defaultextension=files,
                                                    title="Saving ACC Template file", initialdir=self.templates_dir,
                                                    initialfile=init_filename_guess)

        if saving_path_main is not None:
            print("Saving templates...")

            # unpacking keys (x-values of the annotations) of the templates pack
            templates = list(self.template_pack.keys())
            templates = np.asarray([int(round(x)) for x in templates])

            # the order of the templates is the order they were added in the UI, thus we need to sort them before saving
            templates = np.sort(templates)

            store_txt(saving_path_main.name, templates, sampling_rate=self.sampling_rate, resolution=None, date=None,
                      labels=["templates"], precision=6)

            print("...done")
            self.last_save = True

    def on_closing(self):
        """If there are unsaved template changes, prompts the user if he wants to quit the GUI."""

        if not self.last_save:

            if messagebox.askokcancel("Quit", "Do you want to quit without saving?"):
                self.root.destroy()
            else:
                print("canceled...")

        else:
            self.root.destroy()

    def template_navigator(self, event):
        """Navigates the signal based on a moving window and pressing Right and Left arrow keys."""

        # Moving to the right (right arrow) unless the left most limit + shift surpasses the length of signal
        if event.keysym == 'Right':

            if self.zoomed_in_lims[1] + self.window_shift <= self.time_arr[-1]:

                # d * self.moving_window_size) < len(self.acc_signal_filt):
                self.zoomed_in_lims[0] += self.window_shift
                self.zoomed_in_lims[1] += self.window_shift

                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                self.canvas_plot.draw()

            else:
                self.zoomed_in_lims[1] = self.time_arr[-1]
                self.zoomed_in_lims[0] = self.zoomed_in_lims[1] - self.moving_window_size
                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

                self.canvas_plot.draw()

        # Moving to the left (left arrow)
        elif event.keysym == 'Left':

            if self.zoomed_in_lims[0] - self.window_shift >= 0:

                self.zoomed_in_lims[0] -= self.window_shift
                self.zoomed_in_lims[1] -= self.window_shift

                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                self.canvas_plot.draw()

            else:

                self.zoomed_in_lims[0] = 0
                self.zoomed_in_lims[1] = self.zoomed_in_lims[0] + self.moving_window_size
                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

                self.canvas_plot.draw()

        # Zooming out to original xlims or to previous zoomed in lims
        elif event.keysym == 'Shift_L':

            # if the window was already zoomed in, it should zoom out to original xlims
            if self.var_zoomed_in:

                self.ax.set_xlim(self.original_xlims[0], self.original_xlims[1])
                self.canvas_plot.draw()

                self.var_zoomed_in = False

            elif not self.var_zoomed_in:

                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                self.canvas_plot.draw()
                self.var_zoomed_in = True

        # if Left Control is pressed, the signals should be normalized in amplitude, within the displayed window
        elif event.keysym == 'Control_L':

            if self.var_toggle_Ctrl.get() == 0:

                # Setting baseline min/max values is needed because if there are no filtered signals, these values
                # won't interfer with the min/max computation on the raw signals
                # values for min set at unreasonably high number (1000)
                min_max_within_window = 1000 * np.ones((2 * self.nr_sub_signals, 2))

                # values for max set at unreasonably high number (-1000)
                min_max_within_window[:, 1] = -1 * min_max_within_window[:, 1]

                for label, i in zip(self.labels, range(self.nr_sub_signals)):
                    # let's find the maximum and minimum values within the current window

                    # if the "view filtered signal" is toggled, means that filtered signals are currently displayed,
                    # therefore the axes exist
                    if self.var_view_filtered_signal.get() == 1:
                        # mins are in 2nd index = 0
                        min_max_within_window[i, 0] = np.min(self.main_plots['{}_filt'.format(label)][1][
                                                             int(self.ax.get_xlim()[0]):int(self.ax.get_xlim()[1])])

                        min_max_within_window[i, 1] = np.max(self.main_plots['{}_filt'.format(label)][1][
                                                             int(self.ax.get_xlim()[0]):int(self.ax.get_xlim()[1])])

                    # anyways, we always compute for raw signals
                    min_max_within_window[self.nr_sub_signals + i, 0] = np.min(
                        self.main_plots['{}_raw'.format(label)][1][
                        int(self.ax.get_xlim()[0]):int(self.ax.get_xlim()[1])])

                    min_max_within_window[self.nr_sub_signals + i, 1] = np.max(
                        self.main_plots['{}_raw'.format(label)][1][
                        int(self.ax.get_xlim()[0]):int(self.ax.get_xlim()[1])])

                min_in_window = np.min(min_max_within_window[:, 0])
                max_in_window = np.max(min_max_within_window[:, 1])

                self.ax.set_ylim(min_in_window, max_in_window)

                self.var_toggle_Ctrl.set(1)
                self.canvas_plot.draw()

            else:
                self.ax.set_ylim(self.original_ylims[0], self.original_ylims[1])

                self.var_toggle_Ctrl.set(0)
                self.canvas_plot.draw()

        elif event.keysym == 'Escape':
            self.on_closing()

    def template_checkbox_on_click(self):
        """Logs the template editing checkbox."""

    def on_click_ax(self, event):
        """Adds a template at the clicked location within the plotting area (Left mouse click) or deletes templates that are close to
        the clicked coordinates (Right mouse click). """

        if event.inaxes is not None and self.var_edit_plots.get() == 1 and not event.dblclick:

            if event.button == MouseButton.RIGHT:

                templates_in_samples_temp = np.asarray(list(self.template_pack.keys())) / self.sampling_rate

                closest_template = templates_in_samples_temp[np.argmin(np.abs(templates_in_samples_temp - event.xdata))]

                print(closest_template)

                # using a "detection" window of 0.2s (=200 ms)
                if closest_template - 0.2 < event.xdata < closest_template + 0.2:
                    self.template_pack[round(closest_template * self.sampling_rate, 3)].remove()
                    del self.template_pack[round(closest_template * self.sampling_rate, 3)]

                    self.last_save = False

            elif event.button == MouseButton.LEFT:

                self.template_pack[int(event.xdata * 1000)] = self.ax.vlines(event.xdata, self.original_ylims[0],
                                                                             self.original_ylims[1], colors='#FF6B66')

                self.last_save = False

            self.canvas_plot.draw()
