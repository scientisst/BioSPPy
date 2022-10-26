import numpy as np
from ..features.time_freq import get_DWT


def getData(LEN=100, SR=100):
    const0 = np.zeros(LEN)
    const1 = np.ones(LEN)
    constNeg = -1 * np.ones(LEN)

    f = 5
    x = np.arange(0, LEN/SR, 1/SR)
    sine = np.sin(2 * np.pi * f * x)
    np.random.seed(0)
    sine = sine + np.random.normal(0, 0.5, LEN)
    lin = np.arange(LEN)
    sine = +2*np.sin(2 * np.pi * 10 * x)
    return const0, const1, constNeg, lin, sine


def test(LEN=1000, SR=100):

    const0, const1, constNeg, lin, sine = getData(LEN, SR)
    
    const0_ca, const0_cd  = get_DWT(const0)
    const1_ca, const1_cd = get_DWT(const1)
    constNeg_ca, constNeg_cd = get_DWT(constNeg)
    lin_ca, lin_cd = get_DWT(lin)
    sine_a, sine_d = get_DWT(sine)
    
#    print(const0_ca, const0_cd)
   # print(const1_ca, const1_cd)
   # print(constNeg_ca, constNeg_cd)
   # print(lin_ca, lin_cd)
   # print(sine_a, sine_d)
   # 
   # ## dwt 
   # np.testing.assert_almost_equal(const0_ca,  err_msg="const0 ca", decimal=2)
   # np.testing.assert_almost_equal(const0_cd, err_msg="const0 cd", decimal=2)
   # 
   # np.testing.assert_almost_equal(const1_ca,  err_msg="const1 ca", decimal=2)
   # np.testing.assert_almost_equal(const1_cd, err_msg="const1 cd", decimal=2)
   # 
   # np.testing.assert_almost_equal(constNeg_ca,  err_msg="constNeg ca", decimal=2)
   # np.testing.assert_almost_equal(constNeg_cd, err_msg="constNeg cd", decimal=2)
   # 
   # np.testing.assert_almost_equal(lin_ca,  err_msg="lin ca", decimal=2)
   # np.testing.assert_almost_equal(lin_cd, err_msg="lin cd", decimal=2)

   # np.testing.assert_almost_equal(sine_ca,  err_msg="sine ca", decimal=2)
   # np.testing.assert_almost_equal(sine_cd, err_msg="sine cd", decimal=2)


