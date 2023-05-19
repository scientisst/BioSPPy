import unittest

import numpy as np
from pandas import DataFrame

from biosppy.signals import ecg, tools
from biosppy import utils


class ECGModule(unittest.TestCase):

	ecg: np.array
	fs: float
	show = bool

	@classmethod
	def setUpClass(cls) -> None:

		# example of ecg
		cls.ecg_signal = np.loadtxt('../../examples/ecg.txt')
		# example of empty
		cls.no_signal = np.array([])
		cls.complex_signal = cls.ecg_signal.astype(complex)

		cls.fs = 1000.0

		cls.ecg_keys = ['ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate']

		cls.r_peaks = [286, 1206, 2161, 3191, 4212, 5190, 6203, 7233, 8201, 9160, 10158, 11200, 12161, 13142, 14165]
		cls.t_waves = [543, 1459, 2412, 3446, 4466, 5440, 6458, 7489, 8450, 9413, 10412, 11454, 12412, 13395, 14419]
		cls.t_starts = [464, 1376, 2330, 3369, 4395, 5366, 6377, 7408,  8380, 9336, 10340, 11371, 12340, 13321, 14352]
		cls.t_ends = [607, 1523, 2479, 3512, 4538, 5513, 6521, 7556, 8529, 9474, 10476, 11522, 12474, 13461, 14477]

		cls.ecg_pandas = DataFrame({'ecg': cls.ecg_signal})
		cls.hr = 60

	def test_ecg_datatypes_type_error(self):
		types = [None, self.ecg_pandas, utils.ReturnTuple]
		for datatype in types:
			with self.subTest(datatype=type(datatype)):
				self.assertRaises(TypeError, ecg.ecg, datatype)

	def test_ecg_arrays_value_error(self):
		signals = [self.no_signal]
		# , self.complex_signal
		for signal in signals:
			with self.subTest(numpy_array=signal):
				self.assertRaises(ValueError, ecg.ecg, self.no_signal)

	def test_ecg_incorrect_sampling_frequencies_value_error(self):
		fs = [10, -1000, 0, np.inf, -np.inf]
		for sampling_frequency in fs:
			with self.subTest(sampling_frequency=sampling_frequency):
				self.assertRaises(ValueError, ecg.ecg, self.ecg_signal, sampling_rate=sampling_frequency, show=False, interactive=False)

	def test_ecg_output_is_ReturnTuple(self):

		ecg_info = ecg.ecg(self.ecg_signal, show=False, sampling_rate=1000)  # good argument
		self.assertIsInstance(ecg_info, utils.ReturnTuple)

	def test_ecg_correct_keys(self):

		ecg_info = ecg.ecg(self.ecg_signal, show=False)  # good argument
		self.assertEqual(ecg_info.keys(), self.ecg_keys)  # confirm output keys

	def assertSequenceAlmostEqual(self, list1, list2):

		self.assertEqual(len(list1), len(list2))
		for i in range(len(list1)):
			self.assertAlmostEqual(list1[i], list2[i])

	def aux_test_RPeak_detector(self, str_detector, **args):
		"""
		Auxiliary method to test a given R peak detector/ segmenter
		"""
		# 1. Get Rpeaks from an Rpeak detector
		r_peaks = eval('ecg.' + str_detector)(self.ecg_signal, **args)['rpeaks']
		# 2. Correct Rpeaks to R position
		r_peaks = ecg.correct_rpeaks(self.ecg_signal, r_peaks)['rpeaks']
		# 2. Compare new rpeaks with those previously defined
		self.assertSequenceAlmostEqual(r_peaks, self.r_peaks)

	def aux_test_Rpeak_detector_output_is_tuple(self, str_detector, **args):
		"""
		Auxiliary method to test if output of R peak detectors is ReturnTuple
		"""
		r_peaks = eval('ecg.' + str_detector)(self.ecg_signal, **args)
		self.assertIsInstance(r_peaks, utils.ReturnTuple)

	def aux_test_Rpeak_detector_is_array(self, str_detector, **args):
		"""
		Auxiliary method to test if ReturnTuple of R peak detectors contains field "r_peaks" with array
		"""
		r_peaks = eval('ecg.' + str_detector)(self.ecg_signal, **args)["rpeaks"]
		self.assertIsInstance(r_peaks, np.ndarray)

	def test_hamilton_segmenter_r_peaks(self):
		"""
		Test hamilton segmenter
		"""
		self.aux_test_RPeak_detector(str_detector='hamilton_segmenter')

	def test_hamilton_segmenter_output_is_tuple(self):

		self.aux_test_Rpeak_detector_output_is_tuple(str_detector='hamilton_segmenter')

	def test_hamilton_segmenter_is_array(self):

		self.aux_test_Rpeak_detector_is_array(str_detector='hamilton_segmenter')

	def test_christov_segmenter_r_peaks(self):
		"""
		Test christov segmenter
		"""
		self.aux_test_RPeak_detector(str_detector='christov_segmenter')

	def test_christov_segmenter_output_is_tuple(self):

		self.aux_test_Rpeak_detector_output_is_tuple(str_detector='christov_segmenter')

	def test_christov_segmenter_is_array(self):

		self.aux_test_Rpeak_detector_is_array(str_detector='christov_segmenter')

	def test_ssf_segmenter_r_peaks(self):
		"""
		Test ssf segmenter
		"""
		self.aux_test_RPeak_detector(str_detector='ssf_segmenter')

	def test_ssf_segmenter_output_is_tuple(self):

		self.aux_test_Rpeak_detector_output_is_tuple(str_detector='ssf_segmenter')

	def test_ssf_segmenter_is_array(self):

		self.aux_test_Rpeak_detector_is_array(str_detector='ssf_segmenter')

	def test_engzee_segmenter_r_peaks(self):
		"""
		Test engzee segmenter
		"""
		self.aux_test_RPeak_detector(str_detector='engzee_segmenter')

	def test_engzee_segmenter_output_is_tuple(self):

		self.aux_test_Rpeak_detector_output_is_tuple(str_detector='engzee_segmenter')

	def test_engzee_segmenter_is_array(self):

		self.aux_test_Rpeak_detector_is_array(str_detector='engzee_segmenter')

	def test_gamboa_segmenter_r_peaks(self):
		"""
		Test gamboa segmenter
		"""
		self.aux_test_RPeak_detector(str_detector='gamboa_segmenter')

	def test_gamboa_segmenter_output_is_tuple(self):

		self.aux_test_Rpeak_detector_output_is_tuple(str_detector='gamboa_segmenter')

	def test_gamboa_segmenter_is_array(self):

		self.aux_test_Rpeak_detector_is_array(str_detector='gamboa_segmenter')

	def test_ASI_segmenter_r_peaks(self):
		"""
		Test ASI segmenter
		"""
		self.aux_test_RPeak_detector(str_detector='ASI_segmenter')

	def test_ASI_segmenter_output_is_tuple(self):

		self.aux_test_Rpeak_detector_output_is_tuple(str_detector='ASI_segmenter')

	def test_ASI_segmenter_is_array(self):

		self.aux_test_Rpeak_detector_is_array(str_detector='ASI_segmenter')

	def test_T_wave_detector_t_positions(self):
		"""
		Test T-wave detector
		"""
		ts = ecg.getTPositions(ecg.ecg(self.ecg_signal, show=False, interactive=False))
		self.assertSequenceAlmostEqual(ts["T_positions"], self.t_waves)

	def test_T_wave_detector_t_starts(self):

		ts = ecg.getTPositions(ecg.ecg(self.ecg_signal, show=False, interactive=False))
		self.assertSequenceAlmostEqual(ts["T_start_positions"], self.t_starts)

	def test_T_wave_detector_t_ends(self):

		ts = ecg.getTPositions(ecg.ecg(self.ecg_signal, show=False, interactive=False))
		self.assertSequenceAlmostEqual(ts["T_end_positions"], self.t_ends)


if __name__ == '__main__':
	unittest.main()