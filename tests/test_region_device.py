"""Device argument regression checks: mocked CUDA calls, no model or tensor execution."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from dinoexp.region_ab_prepare import cuda_device


class RegionDeviceTests(unittest.TestCase):
    def setUp(self):
        mocks = ExitStack()
        self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', self.matmul_tf32)
        self.addCleanup(setattr, torch.backends.cudnn, 'allow_tf32', self.cudnn_tf32)
        self.addCleanup(mocks.close)
        self.available = mocks.enter_context(patch('torch.cuda.is_available', return_value=True))
        self.current = mocks.enter_context(patch('torch.cuda.current_device', return_value=1))
        self.select = mocks.enter_context(patch('torch.cuda.set_device', side_effect=self.require_index))
        mocks.enter_context(patch('torch.use_deterministic_algorithms'))

    @staticmethod
    def require_index(device):
        if device.index is None:
            raise ValueError('Expected a torch.device with a specified index or an integer, but got:cuda')

    def test_bare_cuda_selects_current_logical_device(self):
        result = cuda_device('cuda')
        self.assertEqual(result, torch.device('cuda:1'))
        self.current.assert_called_once_with()
        self.select.assert_called_once_with(torch.device('cuda:1'))

    def test_explicit_index_is_preserved(self):
        result = cuda_device('cuda:0')
        self.assertEqual(result, torch.device('cuda:0'))
        self.current.assert_not_called()
        self.select.assert_called_once_with(torch.device('cuda:0'))

    def test_unavailable_cuda_never_falls_back(self):
        self.available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
            cuda_device('cuda')
        self.current.assert_not_called()
        self.select.assert_not_called()

    def test_cpu_is_rejected_without_cuda_calls(self):
        with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
            cuda_device('cpu')
        self.available.assert_not_called()
        self.current.assert_not_called()
        self.select.assert_not_called()


if __name__ == '__main__':
    unittest.main()
