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
import threading
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from biosppy import tools as st, storage
from biosppy.storage import store_txt
from .config import list_functions, UI_intention, UI_intention_list, plot_colors


def rescale_signals(input_signal: np.ndarray, new_min, new_max):
    # first normalize from 0 to 1
    input_signal = (input_signal - np.min(input_signal)) / (np.max(input_signal) - np.min(input_signal))

    # then normalize to the [new_min, new_max] range
    input_signal = input_signal * (new_max - new_min) + new_min

    return input_signal


def milliseconds_to_samples(time_milliseconds: int, sampling_rate: float):
    return int(time_milliseconds * (int(sampling_rate) / 1000))


class event_annotator:
    """Opens an editor of event annotations of the input biosignal modality."""
    global list_functions

    def __init__(self, input_raw_signal, mdata, window_size=6.0, window_stride=1.5, moving_avg_wind_sz=700,
                 annotations_dir=None):
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

        # If no directory is provided, it remains None -> when saving / loading annotations, default dir will open
        self.annotations_dir = annotations_dir

        # Extracting metadata
        self.sampling_rate = mdata['sampling_rate']
        self.labels = mdata['labels']

        # assuming modality of the first signal
        self.assumed_modality = self.labels[0]

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
        self.canvas_plot.get_tk_widget().grid(row=1, column=0, columnspan=1, sticky='w', padx=10)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_click_ax)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_key_press)

        self.toolbarFrame = Frame(master=self.root)
        self.toolbarFrame.grid(row=2, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbarFrame)

        self.main_plots = {}

        # it's easier to expand one dim when single-channel than to check the dimensions for every loop in which signals are plotted.
        if self.raw_signal.ndim == 1:
            self.raw_signal = np.expand_dims(self.raw_signal, axis=-1)

        for label, i in zip(self.labels, range(self.nr_sub_signals)):

            # in next version we need to have this structure different, for example to have 1 raw and 1 filt per channel

            try:

                key_for_first_processing = list(list_functions[self.assumed_modality].keys())[0]

                preprocessing_function = list_functions[self.assumed_modality][key_for_first_processing]["preprocess"]
                tmp_filt_signal = preprocessing_function(self.raw_signal[:, i], sampling_rate=self.sampling_rate)

                tmp_filt_signal = np.squeeze(tmp_filt_signal)

            except:
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

        # Dictionary to store annotated events
        self.annotation_pack = {}

        # Setting plot title for one or multiple signal modalities
        if self.nr_sub_signals == 1:
            self.ax.set_title('Signal: {}'.format(self.labels[0]))
        else:
            signal_substr = ', '.join(self.labels)
            self.ax.set_title('Signals: ' + signal_substr)

        # Controllable On / Off variables
        self.var_edit_plots = IntVar()  # enable / disable annotation editing
        self.var_edit_plots.set(0)  # edit is disabled at start so that the user won't add

        self.var_toggle_Ctrl = IntVar()  # Fit ylims to amplitude within displayed window / show original ylims
        self.var_view_filtered_signal = IntVar()  # Enable / disable view overlapped filtered signal
        self.var_zoomed_in = False  # Zoom-in / Zoom-out
        self.previous_rmv_annotation_intention = False  # False = Not within closest range, True = within range

        # Variable to store last Zoom-in configuration, so that when zooming-in again it returns back to previous
        # zoomed-in view
        self.zoomed_in_lims = [0, self.moving_window_size]

        # Add keyword image
        # Create an object of tkinter ImageTk
        img = ImageTk.PhotoImage(Image.open(os.path.join(os.path.dirname(sys.argv[0]), "biosppy", "inter_plotting",
                                                         "biosppy_layout_labeller_small.png")))

        # Create a Label Widget to display the text or Image
        self.image_label = Label(self.root, image=img)
        self.image_label.grid(row=4, column=0)

        self.f1 = Frame(self.root)
        self.f1.grid(row=0, column=0, sticky=W)

        # Check Button to allow editing annotations (left here as an option)
        self.annotation_checkbox = Checkbutton(self.f1, text='Edit Annotations', variable=self.var_edit_plots,
                                               onvalue=1,
                                               offvalue=0,
                                               command=self.annotation_checkbox_on_click)

        self.annotation_ind = 0
        self.moving_window_ind = 0

        # Drop menu for "file" section
        self.file_menu = Menubutton(self.f1, text="File", relief='raised')

        self.file_menu.menu = Menu(self.file_menu, tearoff=0)
        self.file_menu["menu"] = self.file_menu.menu
        self.file_menu.update()

        file_menu_options = ["Open", "Save As..."]

        for menu_option in file_menu_options:
            self.file_menu.menu.add_command(label=menu_option,
                                            command=lambda option=menu_option: self.file_options(option))

        # Drop menu for "file" section
        self.edit_menu = Menubutton(self.f1, text="Edit", relief='raised')

        self.edit_menu.menu = Menu(self.edit_menu, tearoff=0)
        self.edit_menu["menu"] = self.edit_menu.menu
        self.edit_menu.update()

        # self.edit_menu_options = {"Edit Annotations": ["Lock Annotations", "Unlock Annotations"], "Reset Annotations": ["Reset Annotations"]}
        self.edit_menu_options = {"Reset Annotations": ["Reset Annotations"]}

        for menu_option, menu_values in self.edit_menu_options.items():
            self.edit_menu.menu.add_command(label=menu_values[0],
                                            command=lambda option=menu_option: self.edit_options(option))

        # Button to optionally view filtered signal
        self.overlap_raw = Checkbutton(self.f1, text='View filtered signal', variable=self.var_view_filtered_signal,
                                       onvalue=1,
                                       offvalue=0,
                                       command=self.view_filtered_signals)

        # Drop menu for computing or loading annotations
        self.annotation_opt_menu = Menubutton(self.f1, text="Compute annotations", relief='raised')
        self.annotation_opt_menu.menu = Menu(self.annotation_opt_menu, tearoff=0)
        self.annotation_opt_menu["menu"] = self.annotation_opt_menu.menu
        self.annotation_opt_menu.update()

        # packing
        self.file_menu.pack(side="left")
        self.edit_menu.pack(side="left")
        self.annotation_opt_menu.pack(side="left")
        self.annotation_checkbox.pack(side="left")
        self.overlap_raw.pack(side="left")

        self.pressed_key_display = Label(self.root, text=" ", font=('Helvetica', 14, 'bold'))
        self.pressed_key_display.grid(row=3, column=0)

        self.pressed_key_display.update()

        for k, v in list_functions.items():

            sub_menu = Menu(self.annotation_opt_menu.menu)

            self.annotation_opt_menu.menu.add_cascade(label=k, menu=sub_menu)

            for sub_k, sub_v in v.items():
                sub_menu.add_command(label=sub_k,
                                     command=lambda
                                         option="{},{}".format(k, sub_k): self.update_annotations_options(
                                         option))

        self.last_save = True
        self.moving_onset = None

        self.root.bind("<Key>", self.on_key_press)

        self.root.bind_all('<Key>', self.annotation_navigator)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def hide_text(self):
        self.pressed_key_display.config(text=" ")
        self.pressed_key_display.update()

    def on_key_press(self, event):

        triggered_result = UI_intention(event, var_edit_plots=self.var_edit_plots, var_toggle_Ctrl=self.var_toggle_Ctrl,
                                        var_zoomed_in=self.var_zoomed_in,
                                        window_in_border=self.moving_window_is_in_border(),
                                        closest_event=self.previous_rmv_annotation_intention)

        triggered_intention = triggered_result['triggered_intention']
        triggered_action = triggered_result['triggered_action']

        if triggered_intention is not None:
            display_text = UI_intention_list[triggered_intention][triggered_action]

            self.pressed_key_display.config(text=display_text)

            t = threading.Timer(2, self.hide_text)
            t.start()

    def view_filtered_signals(self):
        """Enables or disables the display of the filtered signal(s)."""

        # if user toggled "view filtered signals"
        if self.var_view_filtered_signal.get() == 1:

            for label, i in zip(self.labels, range(self.nr_sub_signals)):

                if self.nr_sub_signals == 1:
                    color_tmp = 'black'
                else:
                    color_tmp = plot_colors[i]

                tmp_ax, = self.ax.plot(self.time_arr,
                                       self.main_plots['{}_filt'.format(label)][1], linewidth=0.8, alpha=1.0,
                                       color=color_tmp)

                self.main_plots['{}_filt'.format(label)][0] = tmp_ax

        else:
            for label, i in zip(self.labels, range(self.nr_sub_signals)):
                # Note: this .remove() function is from matplotlib!
                self.main_plots['{}_filt'.format(label)][0].remove()
                del self.main_plots['{}_filt'.format(label)][0]
                self.main_plots['{}_filt'.format(label)].insert(0, None)

        self.canvas_plot.draw()

    def reset_annotations(self):

        # Remove all existing annotations
        for annotation in list(self.annotation_pack.keys()):
            try:
                self.annotation_pack[annotation].remove()
                del self.annotation_pack[annotation]

            except:
                pass

        # finally re-draw everything back
        self.canvas_plot.draw()

    def edit_options(self, option):
        """Dropdown menu for Edit options. Currently only supporting loading."""

        # if option == "Edit Annotations":
        #
        #     # If currently Un-locked (label=lock), lock and put label to "Unlock"
        #     if self.var_edit_plots.get() == 1:
        #
        #         self.var_edit_plots.set(0)
        #
        #         # delete contents first
        #         self.edit_menu.menu.delete(0, 'end')
        #
        #         self.edit_menu.menu.add_command(label="Unlock Annotations", command=lambda option="Edit Annotations": self.edit_options(option))
        #         self.edit_menu.menu.add_command(label="Reset Annotations", command=lambda option="Reset Annotations": self.edit_options(option))
        #
        #     # If currently locked (label=Unlock), unlock and put label to "lock"
        #     else:
        #         # delete contents first
        #         self.edit_menu.menu.delete(0, 'end')
        #
        #         self.var_edit_plots.set(1)
        #
        #         self.edit_menu.menu.add_command(label="Lock Annotations", command=lambda option="Edit Annotations": self.edit_options(option))
        #         self.edit_menu.menu.add_command(label="Reset Annotations", command=lambda option="Reset Annotations": self.edit_options(option))

        if option == "Reset Annotations":
            self.reset_annotations()

    def file_options(self, option):
        """Dropdown menu for File options. Currently only supporting loading."""

        files = [('Text Document', '*.txt'),
                 ('Comma-Separated Values file', '*.csv')]

        if option == "Open":

            # If more than one sub-signal, guess the name of the first as default
            init_filename_guess = self.labels[0] + "_annotations.txt"

            annotations_path = filedialog.askopenfile(filetypes=files, defaultextension=files,
                                                      title="Loading Annotations file",
                                                      initialdir=self.annotations_dir,
                                                      initialfile=init_filename_guess)

            try:
                print("Loading annotations...")

                loaded_annotations, _ = storage.load_txt(annotations_path.name)
                print("...done")

                # we only reset the annotations and load new ones if the loaded annotations are not empty
                # if len(loaded_annotations) != 0:

                self.reset_annotations()

                # adding loaded annotations
                for annotation in loaded_annotations:
                    annotation_temp = self.ax.vlines(annotation / self.sampling_rate, self.original_ylims[0],
                                                     self.original_ylims[1], colors='#FF6B66')
                    self.annotation_pack[annotation] = annotation_temp


            except:
                pass


        elif option == "Save As...":

            # If more than one sub-signal, guess the name of the first as default
            init_filename_guess = self.labels[0] + "_annotations.txt"

            saving_path_main = filedialog.asksaveasfile(filetypes=files, defaultextension=files,
                                                        title="Saving Annotations file",
                                                        initialdir=self.annotations_dir,
                                                        initialfile=init_filename_guess)

            if saving_path_main is not None:
                print("Saving annotations...")

                # unpacking keys (x-values of the annotations) of the annotations pack
                annotations = list(self.annotation_pack.keys())
                annotations = np.asarray([int(round(x)) for x in annotations])

                # the order of the annotations is the order they were added in the UI, thus we need to sort them before saving
                annotations = np.sort(annotations)

                store_txt(saving_path_main.name, annotations, sampling_rate=self.sampling_rate, resolution=None,
                          date=None,
                          labels=["annotations"], precision=6)

                print("...done")
                self.last_save = True

        self.canvas_plot.draw()

    def update_annotations_options(self, option):
        """Dropdown menu for annotation options (load or compute annotations). Currently only supporting loading."""

        # start by removing all existing annotations
        for annotation in list(self.annotation_pack.keys()):
            try:
                self.annotation_pack[annotation].remove()
                del self.annotation_pack[annotation]

            except:
                pass

        # then apply signal processing as requested by user
        if option == "load Annotations":
            files = [('Text Document', '*.txt'),
                     ('Comma-Separated Values file', '*.csv')]

            # If more than one sub-signal, guess the name of the first as default
            init_filename_guess = self.labels[0] + "_annotations.txt"

            annotations_path = filedialog.askopenfile(filetypes=files, defaultextension=files,
                                                      title="Loading Annotations file",
                                                      initialdir=self.annotations_dir,
                                                      initialfile=init_filename_guess)

            try:
                loaded_annotations, _ = storage.load_txt(annotations_path.name)

                # adding loaded annotations
                for annotation in loaded_annotations:
                    annotation_temp = self.ax.vlines(annotation, self.original_ylims[0], self.original_ylims[1],
                                                     colors='#FF6B66')
                    self.annotation_pack[annotation] = annotation_temp
            except:
                pass

        else:
            chosen_modality = option.split(",")[0]
            chosen_algorithm = option.split(",")[1]

            preprocess = list_functions[chosen_modality][chosen_algorithm]["preprocess"]
            function = list_functions[chosen_modality][chosen_algorithm]["function"]
            annotation_key = list_functions[chosen_modality][chosen_algorithm]["template_key"]

            # only computing for one of the sub-signals (the first one)
            if function is None:
                input_extraction_signal = self.raw_signal[:, 0]
            else:
                input_extraction_signal = preprocess(self.raw_signal[:, 0], sampling_rate=self.sampling_rate)
                input_extraction_signal = input_extraction_signal['filtered']

            annotations = function(input_extraction_signal, sampling_rate=self.sampling_rate)[annotation_key]

            # adding loaded annotations
            for annotation in annotations:
                annotation_temp = self.ax.vlines(annotation / self.sampling_rate, self.original_ylims[0],
                                                 self.original_ylims[1],
                                                 colors='#FF6B66')

                self.annotation_pack[annotation] = annotation_temp

        # finally re-draw everything back
        self.canvas_plot.draw()

    def save_annotations_file(self):
        """Saves the currently edited annotations."""

        files = [('Text Document', '*.txt'),
                 ('Comma-Separated Values file', '*.csv')]

        # If more than one sub-signal, guess the name of the first as default
        init_filename_guess = self.labels[0] + "_annotations.txt"

        saving_path_main = filedialog.asksaveasfile(filetypes=files, defaultextension=files,
                                                    title="Saving Annotations file", initialdir=self.annotations_dir,
                                                    initialfile=init_filename_guess)

        if saving_path_main is not None:
            print("Saving annotations...")

            # unpacking keys (x-values of the annotations) of the annotations pack
            annotations = list(self.annotation_pack.keys())
            annotations = np.asarray([int(round(x)) for x in annotations])

            # the order of the annotations is the order they were added in the UI, thus we need to sort them before saving
            annotations = np.sort(annotations)

            store_txt(saving_path_main.name, annotations, sampling_rate=self.sampling_rate, resolution=None, date=None,
                      labels=["annotations"], precision=6)

            print("...done")
            self.last_save = True

    def on_closing(self):
        """If there are unsaved annotation changes, prompts the user if he wants to quit the GUI."""

        if not self.last_save:

            if messagebox.askokcancel("Quit", "Do you want to quit without saving?"):
                self.root.destroy()
            else:
                print("canceled...")

        else:
            self.root.destroy()

    def moving_window_is_in_border(self):
        """Return whether the current moving window is already placed at the border of the signal."""

        if self.zoomed_in_lims[1] == self.time_arr[-1] or self.zoomed_in_lims[0] == self.time_arr[0]:
            return True
        else:
            return False

    def annotation_navigator(self, event):
        """Navigates the signal based on a moving window and pressing Right and Left arrow keys."""

        # Moving to the right (right arrow) unless the left most limit + shift surpasses the length of signal
        if UI_intention(event)['triggered_intention'] == "move_right":

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
        elif UI_intention(event)['triggered_intention'] == "move_left":

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
        elif UI_intention(event)['triggered_intention'] == "adjust_in_time":

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
        elif UI_intention(event)['triggered_intention'] == "adjust_in_amplitude":

            # if currently zoomed-out, then zoom-in
            if self.var_toggle_Ctrl.get() == 0:

                # Setting baseline min/max values is needed because if there are no filtered signals, these values
                # won't interfer with the min/max computation on the raw signals
                # values for min set at unreasonably high number (1000)
                min_max_within_window = 1000000 * np.ones((2 * self.nr_sub_signals, 2))

                # values for max set at unreasonably high number (-1000)
                min_max_within_window[:, 1] = -1 * min_max_within_window[:, 1]

                for label, i in zip(self.labels, range(self.nr_sub_signals)):
                    # let's find the maximum and minimum values within the current window

                    # if the "view filtered signal" is toggled, means that filtered signals are currently displayed,
                    # therefore the axes exist

                    # min and max indices of x-axis within current window to get min and max in amplitude
                    idx_i = max(int(self.sampling_rate * self.ax.get_xlim()[0]), 0)
                    idx_f = min(int(self.sampling_rate * self.ax.get_xlim()[1]), self.nr_samples)

                    if self.var_view_filtered_signal.get() == 1:
                        # mins are in 2nd index = 0
                        min_max_within_window[i, 0] = np.min(self.main_plots['{}_filt'.format(label)][1][idx_i:idx_f])

                        min_max_within_window[i, 1] = np.max(self.main_plots['{}_filt'.format(label)][1][idx_i:idx_f])

                    # anyways, we always compute for raw signals
                    min_max_within_window[self.nr_sub_signals + i, 0] = np.min(
                        self.main_plots['{}_raw'.format(label)][1][idx_i:idx_f])

                    min_max_within_window[self.nr_sub_signals + i, 1] = np.max(
                        self.main_plots['{}_raw'.format(label)][1][idx_i:idx_f])

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

    def annotation_checkbox_on_click(self):
        """Logs the annotation editing checkbox."""

    def on_click_ax(self, event):
        """Adds a annotation at the clicked location within the plotting area (Left mouse click) or deletes annotation that are close to
        the clicked coordinates (Right mouse click). """

        if UI_intention(event)['triggered_intention'] == "rmv_annotation" and self.var_edit_plots.get() == 1:

            closest_annotation = self.closest_nearby_event(event)

            # using a "detection" window of 0.2s (=200 ms)
            if closest_annotation is not None:
                self.annotation_pack[round(closest_annotation * self.sampling_rate, 3)].remove()
                del self.annotation_pack[round(closest_annotation * self.sampling_rate, 3)]

                self.last_save = False

        elif UI_intention(event)['triggered_intention'] == "add_annotation" and self.var_edit_plots.get() == 1:

            self.annotation_pack[int(event.xdata * self.sampling_rate)] = self.ax.vlines(event.xdata,
                                                                                         self.original_ylims[0],
                                                                                         self.original_ylims[1],
                                                                                         colors='#FF6B66')

            self.last_save = False

        self.canvas_plot.draw()

    def closest_nearby_event(self, event):

        closest_annotation = None

        try:
            annotations_in_samples_temp = np.asarray(list(self.annotation_pack.keys())) / self.sampling_rate
            closest_annotation = annotations_in_samples_temp[
                np.argmin(np.abs(annotations_in_samples_temp - event.xdata))]

            nearby_condition = (closest_annotation - 0.2 < event.xdata < closest_annotation + 0.2)

            if not nearby_condition:
                closest_annotation = None
                self.previous_rmv_annotation_intention = False

            else:
                self.previous_rmv_annotation_intention = True

        # if it doesn't work, it's because tmeplate_pakc is empty
        except:
            pass

        return closest_annotation
