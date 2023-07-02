# -*- coding: utf-8 -*-
"""
biosppy.inter_plotting.ecg
-------------------

This module provides configuration functions and variables for the Biosignal Annotator.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

from matplotlib.backend_bases import MouseButton
import matplotlib.colors as mcolors
from ..signals import eda
from ..signals import ecg
from ..signals import abp
from ..signals import emg
from ..signals import pcg
from ..signals import ppg


def UI_intention(event, var_edit_plots=None, var_toggle_Ctrl=None, var_zoomed_in=None, window_in_border=None,
                 closest_event=False):
    triggered_intention = None
    triggered_action = None

    if hasattr(event, 'inaxes'):

        if event.inaxes is not None and not event.dblclick:  # and var_edit_plots.get() == 1

            if event.button == MouseButton.RIGHT:
                triggered_intention = "rmv_annotation"

                if var_edit_plots is not None:

                    if var_edit_plots.get() == 1 and closest_event:
                        triggered_action = "rmv_annotation"

                    elif var_edit_plots.get() == 1 and not closest_event:
                        triggered_action = "rmv_annotation_not_close"

                # else the action remain None

            elif event.button == MouseButton.LEFT:
                triggered_intention = "add_annotation"

                if var_edit_plots is not None:

                    if var_edit_plots.get() == 1:
                        triggered_action = "add_annotation"

    elif hasattr(event, 'keysym'):

        # Moving to the Right (right arrow)
        if event.keysym == 'Right':
            triggered_intention = "move_right"

            if window_in_border is not None:
                if not window_in_border:
                    triggered_action = "move_right"

                # else, window doesn't move to the right (stays the same)

        # Moving to the left (left arrow)
        elif event.keysym == 'Left':
            triggered_intention = "move_left"

            if window_in_border is not None:
                if not window_in_border:
                    triggered_action = "move_left"
                # else, window doesn't move to the left (stays the same)

        # Zooming out to original xlims or to previous zoomed in lims
        elif event.keysym == 'Shift_L':
            triggered_intention = "adjust_in_time"

            if var_zoomed_in is not None:
                # if the window was not zoomed in, it should zoom in
                if not var_zoomed_in:
                    triggered_action = "zooming_in"

                # else, it should zoom out
                else:
                    triggered_action = "zooming_out"

        # if Left Control is pressed
        elif event.keysym == 'Control_L':

            triggered_intention = "adjust_in_amplitude"

            if var_toggle_Ctrl is not None:

                # if the window was not zoomed in, it should zoom in
                if var_toggle_Ctrl.get() == 0:
                    triggered_action = "zooming_in"

                # else, it should zoom out
                else:
                    triggered_action = "zooming_out"

        elif event.keysym == 'Escape':
            triggered_intention = "quit_ui"

    return_dict = {'triggered_intention': triggered_intention, 'triggered_action': triggered_action}
    return return_dict


UI_intention_list = {

    "adjust_in_amplitude": {None: "Error", "zooming_in": "Amplitude Zoom-in", "zooming_out": "Amplitude Zoom-out"},

    "adjust_in_time": {None: "Error", "zooming_in": "Time zoom-in", "zooming_out": "Time zoom-out"},

    "move_right": {None: "Cannot move further to the right (signal ends there).", "move_right": "Moving right"},
    "move_left": {None: "Cannot move further to the left (signal ends there).", "move_left": "Moving left"},

    "add_annotation": {None: "Click on \'Edit Annotations\' checkbox", "add_annotation": "Adding Annotation"},
    "rmv_annotation": {None: "Click on \'Edit Annotations\' checkbox", "rmv_annotation": "Removing Annotation",
                       "rmv_annotation_not_close": "Click closer to the annotation."},

}

list_functions = {'EDA': {
    'Basic SCR Extractor - Onsets': {'preprocess': eda.preprocess_eda, 'function': eda.basic_scr,
                                     'template_key': 'onsets'},
    'Basic SCR Extractor - Peaks': {'preprocess': eda.preprocess_eda, 'function': eda.basic_scr,
                                    'template_key': 'onsets'},
    'KBK SCR Extractor - Onsets': {'preprocess': eda.preprocess_eda, 'function': eda.kbk_scr,
                                   'template_key': 'peaks'},
    'KBK SCR Extractor - Peaks': {'preprocess': eda.preprocess_eda, 'function': eda.kbk_scr,
                                  'template_key': 'peaks'}},
    'ECG': {'R-peak Hamilton Segmenter': {'preprocess': ecg.preprocess_ecg,
                                          'function': ecg.hamilton_segmenter, 'template_key': 'rpeaks'},
            'R-peak SSF Segmenter': {'preprocess': ecg.preprocess_ecg, 'function': ecg.ssf_segmenter,
                                     'template_key': 'rpeaks'},
            'R-peak Christov Segmenter': {'preprocess': ecg.preprocess_ecg,
                                          'function': ecg.christov_segmenter, 'template_key': 'rpeaks'},
            'R-peak Engzee Segmenter': {'preprocess': ecg.preprocess_ecg,
                                        'function': ecg.engzee_segmenter, 'template_key': 'rpeaks'},
            'R-peak Gamboa Segmenter': {'preprocess': ecg.preprocess_ecg,
                                        'function': ecg.gamboa_segmenter, 'template_key': 'rpeaks'},
            'R-peak ASI Segmenter': {'preprocess': ecg.preprocess_ecg, 'function': ecg.ASI_segmenter,
                                     'template_key': 'rpeaks'}},
    'ABP': {'Onset Extractor': {'preprocess': abp.preprocess_abp, 'function': abp.find_onsets_zong2003,
                                'template_key': 'onsets'}},
    'EMG': {'Basic Onset Finder': {'preprocess': emg.preprocess_emg, 'function': emg.find_onsets,
                                   'template_key': 'onsets'}},

    # 'Hodges Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.hodges_bui_onset_detector, 'template_key': 'onsets'},
    #     'Bonato Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.bonato_onset_detector, 'template_key': 'onsets'},
    #     'Lidierth Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.lidierth_onset_detector, 'template_key': 'onsets'},
    #     'Abbink Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.abbink_onset_detector, 'template_key': 'onsets'},
    #     'Solnik Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.solnik_onset_detector, 'template_key': 'onsets'},
    #     'Silva Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.silva_onset_detector, 'template_key': 'onsets'},
    #     'londral_onset_detector': {'preprocess': emg.preprocess_emg,'function': emg.londral_onset_detector, 'template_key': 'onsets'}},

    'PCG': {
        'Basic Peak Finger': {'preprocess': None, 'function': pcg.find_peaks, 'template_key': 'peaks'}},
    'PPG': {'Elgendi Onset Finder': {'preprocess': ppg.preprocess_ppg,
                                     'function': ppg.find_onsets_elgendi2013, 'template_key': 'onsets'},
            'Kavsaoglu Onset Finder': {'preprocess': ppg.preprocess_ppg,
                                       'function': ppg.find_onsets_kavsaoglu2016,
                                       'template_key': 'onsets'}}
}

plot_colors = list(mcolors.TABLEAU_COLORS.values())
