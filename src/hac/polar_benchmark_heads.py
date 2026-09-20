"""Bounded CUDA heads and hybrid CUDA-kernel / libsvm reference classification.

The RBF kernel is computed on CUDA; libsvm optimization and sigmoid calibration
remain CPU operations. This is never described as a fully GPU-native SVM.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


def cuda_rbf_kernel(
    left: np.ndarray, right: np.ndarray, gamma: float, *, batch_size: int = 512
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    if gamma <= 0 or batch_size < 1:
        raise ValueError("Invalid kernel settings")
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("RBF inputs must have equal feature dimensions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite kernel inputs")
    # Highest disables TF32 for this numerical reference; feature AMP is separate.
    torch.set_float32_matmul_precision("highest")
    output = np.empty((len(left), len(right)), dtype=np.float64)
    with torch.inference_mode():
        reference = torch.as_tensor(right, device="cuda")
        reference_norm = reference.square().sum(dim=1)[None, :]
        for start in range(0, len(left), batch_size):
            batch = torch.as_tensor(left[start : start + batch_size], device="cuda")
            distances = (
                batch.square().sum(dim=1, keepdim=True) + reference_norm - 2 * (batch @ reference.T)
            ).clamp_min_(0)
            output[start : start + len(batch)] = torch.exp(-float(gamma) * distances).cpu().numpy()
    return output


class CudaKernelSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C: float = 10.0, gamma: float | None = None, seed: int = 42):
        self.C = C
        self.gamma = gamma
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.reference_ = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = self.reference_.shape[1]
        self.gamma_ = 1.0 / self.n_features_in_ if self.gamma is None else float(self.gamma)
        kernel = cuda_rbf_kernel(self.reference_, self.reference_, self.gamma_)
        np.fill_diagonal(kernel, 1.0)
        self.estimator_ = SVC(
            C=self.C,
            kernel="precomputed",
            probability=False,
            random_state=self.seed,
            cache_size=512,
        )
        self.estimator_.fit(kernel, y)
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(cuda_rbf_kernel(X, self.reference_, self.gamma_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(cuda_rbf_kernel(X, self.reference_, self.gamma_))


def fit_group_calibrated_rbf(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, seed: int = 42, folds: int = 5
):
    splits = list(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            features, labels, groups
        )
    )
    classes = set(np.unique(labels))
    for train, calibration in splits:
        if set(groups[train]) & set(groups[calibration]):
            raise ValueError("Calibration source-group leakage")
        if set(labels[train]) != classes or set(labels[calibration]) != classes:
            raise ValueError("A calibration fold is missing classes")
    # Scaling and SVM fitting occur independently inside each calibration training fold.
    estimator = make_pipeline(StandardScaler(), CudaKernelSVC(seed=seed))
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method="sigmoid", cv=splits, n_jobs=1, ensemble=True
    )
    calibrated.fit(features, labels)
    return calibrated
