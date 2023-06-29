from ..signals import eda
from ..signals import ecg
from ..signals import abp
from ..signals import emg
from ..signals import pcg
from ..signals import ppg

list_functions = {'Load Templates': {},
                  'EDA / GSR': {'Basic SCR Extractor - Onsets': {'preprocess': eda.preprocess_eda, 'function': eda.basic_scr, 'template_key': 'onsets'},
                                'Basic SCR Extractor - Peaks': {'preprocess': eda.preprocess_eda, 'function': eda.basic_scr, 'template_key': 'onsets'},
                                'KBK SCR Extractor - Onsets': {'preprocess': eda.preprocess_eda, 'function': eda.kbk_scr, 'template_key': 'peaks'},
                                'KBK SCR Extractor - Peaks': {'preprocess': eda.preprocess_eda, 'function': eda.kbk_scr, 'template_key': 'peaks'}},
                  'ECG': {'R-peak Hamilton Segmenter': {'preprocess': ecg.preprocess_ecg, 'function': ecg.hamilton_segmenter, 'template_key': 'rpeaks'},
                          'R-peak SSF Segmenter': {'preprocess': ecg.preprocess_ecg, 'function': ecg.ssf_segmenter, 'template_key': 'rpeaks'},
                          'R-peak Christov Segmenter': {'preprocess': ecg.preprocess_ecg, 'function': ecg.christov_segmenter, 'template_key': 'rpeaks'},
                          'R-peak Engzee Segmenter': {'preprocess': ecg.preprocess_ecg,'function': ecg.engzee_segmenter, 'template_key': 'rpeaks'},
                          'R-peak Gamboa Segmenter': {'preprocess': ecg.preprocess_ecg,'function': ecg.gamboa_segmenter, 'template_key': 'rpeaks'},
                          'R-peak ASI Segmenter': {'preprocess': ecg.preprocess_ecg,'function': ecg.ASI_segmenter, 'template_key': 'rpeaks'}},
                  'ABP': {'Onset Extractor': {'preprocess': abp.preprocess_abp,'function': abp.find_onsets_zong2003, 'template_key': 'onsets'}},
                  'EMG': {'Basic Onset Finder': {'preprocess': emg.preprocess_emg, 'function': emg.find_onsets, 'template_key': 'onsets'}},

                      # 'Hodges Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.hodges_bui_onset_detector, 'template_key': 'onsets'},
                      #     'Bonato Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.bonato_onset_detector, 'template_key': 'onsets'},
                      #     'Lidierth Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.lidierth_onset_detector, 'template_key': 'onsets'},
                      #     'Abbink Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.abbink_onset_detector, 'template_key': 'onsets'},
                      #     'Solnik Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.solnik_onset_detector, 'template_key': 'onsets'},
                      #     'Silva Onset Finder': {'preprocess': emg.preprocess_emg,'function': emg.silva_onset_detector, 'template_key': 'onsets'},
                      #     'londral_onset_detector': {'preprocess': emg.preprocess_emg,'function': emg.londral_onset_detector, 'template_key': 'onsets'}},

                  'PCG': {'Basic Peak Finger': {'preprocess': None,'function': pcg.find_peaks, 'template_key': 'peaks'}},
                  'PPG': {'Elgendi Onset Finder': {'preprocess': ppg.preprocess_ppg,'function': ppg.find_onsets_elgendi2013, 'template_key': 'onsets'},
                          'Kavsaoglu Onset Finder': {'preprocess': ppg.preprocess_ppg,'function': ppg.find_onsets_kavsaoglu2016,
                                                     'template_key': 'onsets'}}
                  }