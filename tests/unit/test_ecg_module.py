import unittest

from numpy import array, loadtxt, allclose
from numpy import median
from pandas import DataFrame

import sys
sys.path.append('C:\\Users\\Mariana\\PycharmProjects\\BioSPPy')
from biosppy.signals import ecg, tools
from biosppy import utils

class ECGModule(unittest.TestCase):

    ecg: array
    fs: float
    show = bool

    @classmethod
    def setUpClass(cls) -> None:
        # example of ecg
        cls.ecg_signal = loadtxt('C:\\Users\\Mariana\\PycharmProjects\\BioSPPy\\examples\\ecg.txt')
        # example of empty
        cls.no_signal = array([])

        cls.fs = 1000.0
        cls.ecg_keys = ['ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate']

        cls.rpeaks = [286, 1206, 2161, 3191, 4212, 5190, 6203, 7233, 8201, 9160,
                      10158, 11200, 12161, 13142, 14165]
        cls.ecg_pandas = DataFrame({'ecg': cls.ecg_signal})
        cls.hr = 60
        show = False

    def test_ecg_formats(self):
        
        self.assertRaises(TypeError, ecg.ecg, None) ## no argument was given
        self.assertRaises(ValueError, ecg.ecg, self.no_signal) ## empty argument
        self.assertRaises(ValueError, ecg.ecg, self.ecg_pandas)  ## argument in pandas format instead of array/list 
        self.assertRaises(ValueError, ecg.ecg, self.ecg_signal, sampling_rate=10, show=False) ## good argument but bad sampling frequency

        ecg_info = ecg.ecg(self.ecg_signal, show=False) ## good argument
        self.assertIsInstance(ecg_info, utils.ReturnTuple) ## confirm if output is ReturnTuple
        self.assertEquals(ecg_info.keys(), self.ecg_keys) ## confirm output keys
        
        


    def aux_test_RPeak_detector(self, str_detector, **args):
        """
        Auxiliary method to test a given R peak detector/ segmenter
        """
        # 1. Get Rpeaks from an Rpeak detector
        rpeaks = eval('ecg.' + str_detector)(self.ecg_signal, **args)['rpeaks']
        # 2. Correct Rpeaks to R position
        rpeaks = ecg.correct_rpeaks(self.ecg_signal, rpeaks)['rpeaks']
        # 2. Compare new rpeaks with those previously defined
        self.assertTrue(allclose(rpeaks, self.rpeaks))
    

    def test_hamilton_segmenter(self):
        """
        Test hamilton segmenter
        """
        self.aux_test_RPeak_detector(str_detector='hamilton_segmenter')
    
    
    def test_christov_segmenter(self):
        """
        Test christov segmenter
        """
        self.aux_test_RPeak_detector(str_detector='christov_segmenter')



if __name__ == '__main__':
    unittest.main()