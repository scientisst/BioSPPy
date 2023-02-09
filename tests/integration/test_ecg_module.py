import unittest

from numpy import median, array, loadtxt, allclose
import sys
sys.path.append('C:\\Users\\Mariana\\PycharmProjects\\BioSPPy')
from biosppy.signals import ecg


class ECGModule(unittest.TestCase):

    ecg: array
    fs: float
    show = bool

    @classmethod
    def setUpClass(cls) -> None:
        cls.ecg_signal = loadtxt('C:\\Users\\Mariana\\PycharmProjects\\BioSPPy\\examples\\ecg.txt')
        cls.fs = 1000.0

        cls.rpeaks = [286, 1206, 2161, 3191, 4212, 5190, 6203, 7233, 8201, 9160,
                      10158, 11200, 12161, 13142, 14165]
        
        cls.hr = 60
        cls.templates = []
        cls.filtered = []
        show = False

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


    def test_ecg(self):

        ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr = ecg.ecg(self.ecg_signal, sampling_rate=self.fs, show=False)
        self.assertAlmostEqual(int(median(hr)), self.hr)



if __name__ == '__main__':
    unittest.main()