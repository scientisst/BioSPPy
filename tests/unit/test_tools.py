import unittest

import numpy as np
from pandas import DataFrame

from biosppy.signals import ecg, tools
from biosppy import utils


class ToolsModule(unittest.TestCase):

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

	def test__norm_freq_incorrect_sampling_frequencies_value_error(self):
		fs = [10, -1000, 0, np.inf, -np.inf]
		for sampling_frequency in fs:
			with self.subTest(sampling_frequency=sampling_frequency):
				self.assertRaises(ValueError, tools._norm_freq, frequency=1000, sampling_rate=sampling_frequency)

	def test__norm_freq_incorrect_sampling_frequencies_type_error(self):
		types = [None, -1000, 0, np.inf, -np.inf]
		for sampling_frequency in fs:
			with self.subTest(sampling_frequency=sampling_frequency):
				self.assertRaises(ValueError, tools._norm_freq, frequency=1000, sampling_rate=sampling_frequency)

	def test__norm_freq_incorrect_frequencies_value_error(self):
		fs = [5000, -1000, np.inf, -np.inf, [], [0, np.inf, -1000]]
		for frequency in fs:
			with self.subTest(frequency=frequency):
				self.assertRaises(ValueError, tools._norm_freq, frequency=frequency, sampling_rate=1000.0)
				
	def test__norm_freq_incorrect_frequencies_type_error(self):
		fs = [5000, -1000, np.inf, -np.inf, [], [0, np.inf, -1000]]
		for frequency in fs:
			with self.subTest(frequency=frequency):
				self.assertRaises(ValueError, tools._norm_freq, frequency=frequency, sampling_rate=1000.0)


if __name__ == '__main__':
	unittest.main()