import unittest

import numpy as np
import time
from chainer.cuda import cupy as cp
from utils import cupyutils


class TestCupyutils(unittest.TestCase):

    def _arrays_with_module(self, xp, arr_len, arr_step, n_arr):
        return [xp.arange(0, arr_len, arr_step) for a in range(n_arr)]

    def test_meshgrid_performance(self):
        arr_len = 1000
        arr_step = 0.5
        n_arr = 50  # Should be an even number

        # NumPy, CPU
        arrs = self._arrays_with_module(np, arr_len, arr_step, n_arr)
        start_time = time.time()

        meshed_np = []
        for i in range(0, len(arrs), 2):
            meshed = np.meshgrid(arrs[i], arrs[i+1])
            self.assertTrue(cp.get_array_module(meshed, np))
            meshed_np.append(meshed)

        running_time = time.time() - start_time
        print('Time with NumPy: {} s'.format(round(running_time, 2)))

        # CuPy, GPU
        arrs = self._arrays_with_module(cp, arr_len, arr_step, n_arr)
        start_time = time.time()

        meshed_cp = []
        for i in range(0, len(arrs), 2):
            meshed = cupyutils.meshgrid(arrs[i], arrs[i+1])
            self.assertTrue(cp.get_array_module(meshed, cp))
            meshed_cp.append(meshed)

        running_time = time.time() - start_time
        print('Time with CuPy: {} s'.format(round(running_time, 2)))

    def test_meshgrid_acc(self):
        arr_len = 10
        arr_step = 0.2
        n_arr = 6  # Should be an even number

        # NumPy, CPU
        arrs = self._arrays_with_module(np, arr_len, arr_step, n_arr)

        meshed_np = []
        for i in range(0, len(arrs), 2):
            meshed = np.meshgrid(arrs[i], arrs[i+1])
            self.assertTrue(cp.get_array_module(meshed, np))
            meshed_np.append(meshed)

        # Cuda, GPU
        arrs = self._arrays_with_module(cp, arr_len, arr_step, n_arr)

        meshed_cp = []
        for i in range(0, len(arrs), 2):
            meshed = cupyutils.meshgrid(arrs[i], arrs[i+1])
            self.assertTrue(cp.get_array_module(meshed, cp))
            meshed_cp.append(meshed)

        print('Comparing NumPy and CuPy values')
        self.assertEqual(len(meshed_np), len(meshed_cp))  # n_arr / 2
        for m_cp, m_np in zip(meshed_cp, meshed_np):
            self.assertEqual(len(m_cp), len(m_np))  # 2
            for cs, ns in zip(m_cp[0], m_np[0]):
                self.assertEqual(cs.shape, ns.shape)
                for c, n in zip(cs, ns):
                    self.assertEqual(c, n)
            for cs, ns in zip(m_cp[1], m_np[1]):
                self.assertEqual(cs.shape, ns.shape)
                for c, n in zip(cs, ns):
                    self.assertEqual(c, n)


if __name__ == '__main__':
    unittest.main()
