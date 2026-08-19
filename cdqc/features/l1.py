"""L1 — 이미지 전역 피쳐 (측정 무관, OOD 탐지).

FFT 전역 샤프니스는 넣지 않는다 — TEM에서 구조 변화와 초점 변화가 분리되지
않는다 (spec §3.4). hist_emd는 코호트 템플릿이 필요하므로 여기서는 256-bin
정규화 히스토그램만 저장하고, EMD는 calibrate/run에서 계산한다.
"""

from __future__ import annotations

import cv2
import numpy as np

_LAP_NORM = np.sqrt(20.0)   # [[0,1,0],[1,-4,1],[0,1,0]]의 L2 norm — 단위 잡음 응답 std
_TILE_GRID = 4              # tile_energy_cv용 타일 분할 (4×4)


def _noise_sigma(img_f: np.ndarray) -> float:
    """Laplacian 응답 MAD × 1.4826 (커널 norm으로 나눠 gray-level σ 스케일)."""
    lap = cv2.Laplacian(img_f, cv2.CV_64F)
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(1.4826 * mad / _LAP_NORM)


def _grad_mag(img_f: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def _struct_energy(grad: np.ndarray, noise: float) -> float:
    return float(np.percentile(grad, 90) / max(noise, 1e-9))


def l1_image_features(img: np.ndarray) -> dict:
    """이미지 하나의 L1 피쳐 + 정규화 히스토그램(hist, EMD용)."""
    img_f = img.astype(np.float64)
    noise = _noise_sigma(img_f)
    grad = _grad_mag(img_f)
    p1, p99 = np.percentile(img_f, [1, 99])

    h, w = img.shape
    th, tw = h // _TILE_GRID, w // _TILE_GRID
    energies = []
    for r in range(_TILE_GRID):
        for c in range(_TILE_GRID):
            tile = img_f[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            tile_noise = _noise_sigma(tile)
            energies.append(_struct_energy(grad[r * th:(r + 1) * th,
                                                c * tw:(c + 1) * tw], tile_noise))
    energies = np.asarray(energies)
    mean_e = energies.mean()
    tile_cv = float(energies.std() / max(mean_e, 1e-9))

    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()

    return {
        "noise_sigma": noise,
        "sat_lo": float(np.mean(img == 0)),
        "sat_hi": float(np.mean(img == 255)),
        "dyn_range": float(p99 - p1),
        "struct_energy": _struct_energy(grad, noise),
        "tile_energy_cv": tile_cv,
        "hist": hist,   # list column으로 저장 — EMD는 calibrate/run에서
    }


def hist_emd(hist: np.ndarray, template: np.ndarray) -> float:
    """1D EMD = 누적분포 차의 절대합 (bin 단위)."""
    return float(np.abs(np.cumsum(hist - template)).sum())
