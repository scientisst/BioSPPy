# -*- coding: utf-8 -*-
"""
biosppy.inter_plotting.ecg
-------------------

This module provides an interactive UI for manual annotation of time stamps on biosignals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from matplotlib.backend_bases import MouseButton
from biosppy import tools as st, storage
from biosppy.storage import store_txt

class event_annotator:
    """Opens an editor of event annotations of the input biosignal modality."""

    def __init__(self, input_raw_signal, mdata, window_size, window_shift, path_to_signal=None):

        # Extracting metadata
        self.sampling_rate = mdata['sampling_rate']
        self.resolution = mdata['resolution']
        self.labels = mdata['labels']

        # Sub-signals or channels of the same signal modality
        self.nr_sub_signals = len(self.labels)

        self.path_to_signal = path_to_signal
        self.raw_signal = input_raw_signal

        # window_size is in seconds. Nr samples in window = seconds x freq (nr samples / second)
        self.moving_window_size = window_size * self.sampling_rate
        self.window_shift = window_shift * self.sampling_rate # overlap of 2.5 seconds to confirm labels on the previous window

        self.nr_samples = self.raw_signal.shape[0]

        # processing data
        # filt_param = 500  # moving average with 500ms
        # self.filt_signal, _ = st.smoother(input_raw_signal, size=filt_param)

        # tkinter figures and plots
        self.root = Tk()
        self.root.resizable(False, False)

        self.figure = plt.Figure(figsize=(9, 2.4), dpi=100)
        self.figure.canvas.callbacks.connect('button_press_event', self.on_click_ax1)

        self.ax = self.figure.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, columnspan=1, sticky='w', padx=10)

        self.toolbarFrame = Frame(master=self.root)
        self.toolbarFrame.grid(row=1, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbarFrame)

        # Dictionary containing visible (axes) or disabled (None) plots

        # self.main_plots = {'raw': None, 'filt': None}
        # self.main_plots['raw'] = self.ax.plot(np.arange(0, self.raw_signal.shape[0], 1),
        #                                       self.raw_signal, linewidth=0.5)

        self.main_plots = {}
        for label, i in zip(self.labels, range(self.nr_sub_signals)):
            # in next version we need to have this structure different, for example to have 1 raw and 1 filt per channel
            self.main_plots['{}_filt'.format(label)] = None
            self.main_plots['{}_raw'.format(label)] = [self.ax.plot(np.arange(0, self.raw_signal.shape[0], 1), self.raw_signal[:, i], linewidth=0.5),
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
            signal_substr = ' '.join(self.labels)
            self.ax.set_title('Signals: ' + signal_substr)

        # Controllable On / Off variables
        self.var_edit_plots = IntVar() # enable / disable template editing
        self.var_toggle_Ctrl = IntVar() # Fit ylims to amplitude within displayed window / show original ylims
        self.var_view_filtered_signal = IntVar() # Enable / disable view overlapped filtered signal
        self.var_zoomed_in = False # Zoom-in / Zoom-out

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
        self.overlap_raw = Checkbutton(self.root, text='View filtered signal', variable=self.var_view_filtered_signal, onvalue=1,
                                       offvalue=0,
                                       command=self.overlap_raw_on_click)
        self.overlap_raw.grid(row=1, column=1)

        # Drop menu for computing or loading templates
        self.template_opt_menu = Menubutton(self.root, text="Add templates", relief='raised')
        self.template_opt_menu.grid(row=1, column=2, sticky=W)
        self.template_opt_menu.menu = Menu(self.template_opt_menu, tearoff=0)
        self.template_opt_menu["menu"] = self.template_opt_menu.menu
        self.template_opt_menu.update()

        self.template_options = {'load templates': False}
        for k, v in self.template_options.items():
            self.template_opt_menu.menu.add_command(label=k,
                                                    command=lambda option=k: self.update_templates_options(option))

        self.last_save = True
        self.moving_onset = None

        self.root.bind_all('<Key>', self.template_navigator)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def overlap_raw_on_click(self):

        if self.var_view_filtered_signal.get() == 1:  # 1 previously

            self.filt_signal = (self.filt_signal - np.min(self.filt_signal)) / (
                    np.max(self.filt_signal) - np.min(self.filt_signal))

            self.filt_signal = self.filt_signal * (
                    np.max(self.raw_signal) - np.min(self.raw_signal)) + np.min(
                self.raw_signal)

            self.main_plots['filt'], = self.ax.plot(np.arange(0, self.raw_signal.shape[0], 1), self.filt_signal,
                                                    linewidth=0.5, color='#ff7f0e', alpha=0.6)

        else:
            self.main_plots['filt'].remove()
            del self.main_plots['filt']

        self.canvas_plot.draw()

    def clear_all_plots(self):

        # clearing main ppg plots
        for ax in self.ax.get_lines():
            try:
                ax.remove()
            except:
                pass

        # clearing templates
        for onset in list(self.template_pack.keys()):
            try:
                self.template_pack[onset].remove()
                del self.template_pack[onset]

            except:
                pass

        self.canvas_plot.draw()

    def update_templates_options(self, option):

        for onset in list(self.template_pack.keys()):
            try:
                self.template_pack[onset].remove()
                del self.template_pack[onset]

            except:
                pass

        if option == "load templates":
            files = [('Text Document', '*.txt'),
                     ('Comma-Separated Values file', '*.csv')]

            # If more than one sub-signal, guess the name of the first as default
            init_filename_guess = self.labels[0] + "_templates.txt"

            templates_path = filedialog.askopenfile(filetypes=files, defaultextension=files,
                                                             title="Loading ACC templates file",
                                                             initialdir=self.path_to_signal,
                                                             initialfile=init_filename_guess)

            loaded_templates, _ = storage.load_txt(templates_path.name)

            # adding loaded templates
            for onset in loaded_templates:
                template_temp = self.ax.vlines(onset, self.original_ylims[0], self.original_ylims[1],
                                               colors='#FF6B66')
                self.template_pack[onset] = template_temp

        self.canvas_plot.draw()

    def save_templates_file(self):
        """Saves the current state of edited templates."""
        # TODO: SORT THE TEMPLATE INDICES BEFORE SAVING THE DATAFRAME
        files = [('Text Document', '*.txt'),
                 ('Comma-Separated Values file', '*.csv')]

        # If more than one sub-signal, guess the name of the first as default
        init_filename_guess = self.labels[0] + "_templates.txt"

        saving_path_main = filedialog.asksaveasfile(filetypes=files, defaultextension=files,
                                                    title="Saving ACC Template file", initialdir=self.path_to_signal,
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
        print(event.keysym)
        print("zoomed_in_before")
        print(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
        print(self.zoomed_in_lims[1] + self.window_shift)

        # Moving to the right (right arrow) unless the left most limit + shift surpasses the length of signal
        if event.keysym == 'Right':

            if self.zoomed_in_lims[1] + self.window_shift <= self.nr_samples:

                print("righttttt")
                # d * self.moving_window_size) < len(self.acc_signal_filt):
                self.zoomed_in_lims[0] += self.window_shift
                self.zoomed_in_lims[1] += self.window_shift

                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                self.canvas_plot.draw()

            else:
                self.zoomed_in_lims[1] = self.nr_samples - 1
                self.zoomed_in_lims[0] = self.zoomed_in_lims[1] - self.moving_window_size
                self.ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

                self.canvas_plot.draw()

        # Moving to the left (left arrow)
        elif event.keysym == 'Left':

            if self.zoomed_in_lims[0] - self.window_shift >= 0:
                print("lefttttt")

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

                segm_1 = self.filt_signal[int(self.ax.get_xlim()[0]):int(self.ax.get_xlim()[1])]

                print((np.min(segm_1), np.max(segm_1)))

                self.ax.set_ylim(np.min(segm_1), np.max(segm_1))

                self.var_toggle_Ctrl.set(1)
                self.canvas_plot.draw()

            else:
                self.ax.set_ylim(self.original_ylims[0], self.original_ylims[1])

                self.var_toggle_Ctrl.set(0)
                self.canvas_plot.draw()

        elif event.keysym == 'Escape':
            self.on_closing()

        print("zoomed_in_after")
        print(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

    def template_checkbox_on_click(self):
        """Logs the template editing checkbox."""

        print(self.var_edit_plots.get())

    def on_click_ax1(self, event):
        """Adds templates at clicked location in first plot (Left mouse click) or deletes templates that are close to
        the clicked coordinates (Right mouse click). """

        print(type(event.button))

        if event.inaxes is not None and self.var_edit_plots.get() == 1 and not event.dblclick:

            if event.button == MouseButton.RIGHT:

                onsets_temp = list(self.template_pack.keys())
                closest_onset = onsets_temp[np.argmin(np.abs(np.asarray(onsets_temp) - event.xdata))]

                if closest_onset - 200 < event.xdata < closest_onset + 200:
                    self.template_pack[closest_onset].remove()
                    del self.template_pack[closest_onset]

                    self.last_save = False

            elif event.button == MouseButton.LEFT:
                self.template_pack[event.xdata] = self.ax.vlines(event.xdata, self.original_ylims[0],
                                                                 self.original_ylims[1], colors='#FF6B66')

                self.last_save = False

            self.canvas_plot.draw()

    # def on_click_ax2(self, event):
    #     """Adds templates at clicked location in first plot (Left mouse click) or deletes templates that are close to
    #     the clicked coordinates (Right mouse click). """
    #
    #     if event.inaxes is not None and self.var1.get() == 1:
    #
    #         # print("this")
    #
    #         if event.dblclick and event.button == MouseButton.LEFT and self.moving_onset is None:
    #             print('left')
    #
    #             onsets_temp = list(self.template_pack_2.keys())
    #             closest_onset = onsets_temp[np.argmin(np.abs(np.asarray(onsets_temp) - event.xdata))]
    #
    #             if closest_onset - 200 < event.xdata < closest_onset + 200:
    #
    #                 try:
    #                     self.template_pack_1[closest_onset].remove()
    #                     del self.template_pack_1[closest_onset]
    #
    #                     self.template_pack_1[closest_onset] = self.ax1.vlines(closest_onset, self.or_plot_1_lims[0],
    #                                                                           self.or_plot_1_lims[1], colors='#8CFF66')
    #                 except:
    #                     pass
    #
    #                 self.moving_onset = closest_onset
    #
    #                 # self.last_save = False
    #             self.last_save = False
    #
    #         if not event.dblclick and event.button == MouseButton.LEFT and self.moving_onset is not None:
    #
    #             # un-highlighting selected ppg peak
    #             try:
    #                 self.template_pack_1[self.moving_onset].remove()
    #                 del self.template_pack_1[self.moving_onset]
    #
    #                 self.template_pack_1[self.moving_onset] = self.ax1.vlines(self.moving_onset, self.or_plot_1_lims[0],
    #                                                                           self.or_plot_1_lims[1], colors='#FF6B66')
    #             except:
    #                 pass
    #
    #             self.moving_onset = None
    #             self.last_save = False
    #
    #         elif not event.dblclick and event.button == MouseButton.RIGHT and self.moving_onset is None:
    #
    #             onsets_temp = list(self.template_pack_2.keys())
    #             closest_onset = onsets_temp[np.argmin(np.abs(np.asarray(onsets_temp) - event.xdata))]
    #
    #             if closest_onset - 200 < event.xdata < closest_onset + 200:
    #                 try:
    #                     self.template_pack_1[closest_onset].remove()
    #                     del self.template_pack_1[closest_onset]
    #                 except:
    #                     pass
    #
    #                 self.last_save = False
    #
    #         self.canvas_plot_1.draw()
