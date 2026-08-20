"""마스크 기반 피쳐 — DL segmentation 결과(이진 마스크)와 좌표/이미지의 정합.

마스크는 이진 ndarray (bool 또는 uint8 0/비0), 이미지와 같은 shape.
라벨맵 → 클래스별 이진 분리(컬러/라벨 PNG 해석)는 사용자 몫이다.

- CD 레벨 (mask_l3_features): 보고 좌표 ↔ 마스크 경계 거리(mdist), 경계
  위치의 이미지 그래디언트 증거(mgrad), 세그먼트 중점의 마스크 내부 여부.
- 이미지 레벨 (mask_image_features): DL이 이미지 전체에서 헛것을 그렸는지
  (mask_grad_agree), 성분/구멍/경계 거칠기 등 마스크 형태 자체의 이상.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_erosion, binary_fill_holes,
                           distance_transform_edt, label, map_coordinates)

from ..sampling import estimate_local_sigma
from .l1 import _grad_mag, _noise_sigma

_EIGHT = np.ones((3, 3), dtype=bool)   # 8-이웃 연결
_MIN_COMPONENT_PX = 16                 # boundary_rough 계산에 포함할 최소 성분 크기
                                       # (잡티가 성분별 중앙값을 지배하지 않게 —
                                       #  잡티 자체는 mask_n_components가 잡는다)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """마스크 내부 경계 픽셀 (mask & ~erode). bool (H, W)."""
    m = mask.astype(bool)
    return m & ~binary_erosion(m, structure=_EIGHT, border_value=0)


# ---------------------------------------------------------------- CD 레벨

def mask_l3_features(mask: np.ndarray, img: np.ndarray, S: np.ndarray,
                     E: np.ndarray, px_nm: float, cfg: dict) -> dict[str, np.ndarray]:
    """한 시퀀스의 CD별 마스크 피쳐. 반환: 이름 → (n,) 배열.

    - mdist_s/e: 보고 좌표에서 가장 가까운 마스크 경계까지 거리 (nm).
      경계 distance transform을 좌표 위치에서 bilinear 샘플.
    - mgrad_s/e: 가장 가까운 경계점 위치의 **이미지** 그래디언트 크기 / 국소
      노이즈 σ — DL 경계가 실제 명암 전이 위에 있는지.
    - minside: 세그먼트 중점(플래토 중앙)이 마스크 내부인가 (bool).
    """
    n = len(S)
    m = mask.astype(bool)
    bnd = boundary_pixels(m)
    out: dict[str, np.ndarray] = {
        "mdist_s": np.full(n, np.nan), "mdist_e": np.full(n, np.nan),
        "mgrad_s": np.full(n, np.nan), "mgrad_e": np.full(n, np.nan),
        "minside": np.zeros(n, dtype=bool),
    }

    h, w = m.shape
    mid = (S + E) / 2.0
    mi = np.clip(np.round(mid[:, 1]).astype(int), 0, h - 1)
    mj = np.clip(np.round(mid[:, 0]).astype(int), 0, w - 1)
    out["minside"] = m[mi, mj]

    if not bnd.any():
        return out

    # 경계까지 거리(px) + 가장 가까운 경계 픽셀 인덱스
    dt, ind = distance_transform_edt(~bnd, return_indices=True)
    grad = _grad_mag(img.astype(np.float64))
    patch_px = int(cfg["sampling"]["noise_patch_px"])

    for tag, P in (("s", S), ("e", E)):
        d = map_coordinates(dt, [P[:, 1], P[:, 0]], order=1, mode="nearest")
        out[f"mdist_{tag}"] = d * px_nm
        yi = np.clip(np.round(P[:, 1]).astype(int), 0, h - 1)
        xi = np.clip(np.round(P[:, 0]).astype(int), 0, w - 1)
        by, bx = ind[0, yi, xi], ind[1, yi, xi]
        g = grad[by, bx]
        sig = np.array([estimate_local_sigma(img, float(bx[k]), float(by[k]),
                                             patch_px) for k in range(n)])
        out[f"mgrad_{tag}"] = g / np.maximum(sig, 1e-9)
    return out


# ---------------------------------------------------------------- 이미지 레벨

def mask_image_features(mask: np.ndarray, img: np.ndarray) -> dict:
    """마스크 전체의 이미지 레벨 피쳐 (좌표 무관). registry level="lm".

    코호트 축 주의: 마스크는 보통 (이미지 × 클래스)당 하나 — L1(이미지당
    하나)과 코호트 축이 다를 수 있다. 코호트 분리는 사용자 몫.
    """
    m = mask.astype(bool)
    imgf = img.astype(np.float64)
    bnd = boundary_pixels(m)

    # DL 경계가 실제 명암 전이 위에 있는지의 단일 지표
    if bnd.any():
        grad = _grad_mag(imgf)
        agree = float(np.median(grad[bnd]) / max(_noise_sigma(imgf), 1e-9))
    else:
        agree = np.nan

    lbl, n_comp = label(m, structure=_EIGHT)

    if m.any():
        filled = binary_fill_holes(m)
        hole_frac = float((filled & ~m).sum() / max(int(filled.sum()), 1))
    else:
        hole_frac = np.nan

    roughs = []
    for c in range(1, n_comp + 1):
        comp = lbl == c
        area = int(comp.sum())
        if area < _MIN_COMPONENT_PX:
            continue
        perim = int(boundary_pixels(comp).sum())
        roughs.append(perim / max(2.0 * np.sqrt(np.pi * area), 1e-9))
    rough = float(np.median(roughs)) if roughs else np.nan

    return {
        "mask_grad_agree": agree,
        "mask_n_components": float(n_comp),
        "mask_hole_frac": hole_frac,
        "mask_boundary_rough": rough,
        "mask_area_frac": float(m.mean()),
    }
