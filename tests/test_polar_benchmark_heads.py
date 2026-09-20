import numpy as np
import pytest
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from hac.polar_benchmark_heads import CudaKernelSVC, cuda_rbf_kernel, fit_group_calibrated_rbf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity needs GPU")
def test_cuda_rbf_kernel_and_svc_match_reference():
    rng = np.random.default_rng(64)
    x = rng.normal(size=(60, 12)).astype(np.float32)
    y = np.tile(np.arange(3), 20)
    actual = cuda_rbf_kernel(x, x, 1 / 12, batch_size=13)
    expected = rbf_kernel(x.astype(float), gamma=1 / 12)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    model = CudaKernelSVC().fit(x, y)
    reference = SVC(C=10, kernel="rbf", gamma=1 / 12).fit(x, y)
    np.testing.assert_array_equal(model.predict(x), reference.predict(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA calibration needs GPU")
def test_grouped_calibration_generic_nine_classes():
    rng = np.random.default_rng(75)
    y = np.tile(np.arange(9), 15)
    x = np.eye(9)[y] + rng.normal(scale=0.1, size=(len(y), 9))
    groups = np.asarray([str(i) for i in range(len(y))])
    model = fit_group_calibrated_rbf(x, y, groups, folds=3)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (len(y), 9)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
