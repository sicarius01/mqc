"""리본 프로파일 샘플러와 국소 노이즈 추정.

세그먼트 축을 따라 S−margin ~ E+margin을 subpixel 샘플링하고, 노이즈 억제를
위해 국소 엣지 접선 방향으로 ±ribbon_half_w px 오프셋한 줄들을 평균한다
(spec §3.2). 이미지의 모든 CD 샘플 좌표를 한 번에 모아 map_coordinates 호출을
최소화한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, map_coordinates

DS = 0.5  # 프로파일 샘플 간격 (px)

# Immerkær 고주파 커널: 단위 백색잡음에 대한 응답 std = 6
_IMMERKAER = np.array([[1.0, -2.0, 1.0],
                      [-2.0, 4.0, -2.0],
                      [1.0, -2.0, 1.0]])
_IMMERKAER_NORM = 6.0


@dataclass
class Profile:
    """한 CD의 리본 평균 프로파일. s=0이 S, s=L이 E."""
    s: np.ndarray       # 샘플 위치 (px)
    p: np.ndarray       # 리본 평균 강도
    L: float            # 세그먼트 길이 (px)
    sigma: float        # 국소 노이즈 σ (gray level)
    u: np.ndarray       # 세그먼트 축 단위벡터 (S→E)
    tangent: np.ndarray  # 리본 오프셋 방향 (국소 엣지 접선)


def effective_margin(L: float, cfg: dict) -> float:
    """프로파일/탐색의 바깥쪽 여유.

    margin_px는 하한이고, 탐색 창 절반(W/2)까지는 바깥도 봐야 창 크기만큼의
    엣지 점프를 delta로 추적할 수 있다. evidence.py의 탐색 범위와 동일해야 한다.
    """
    W = float(np.clip(cfg["win_frac"] * L, cfg["win_min_px"], cfg["win_max_px"]))
    return max(float(cfg["margin_px"]), W / 2 + 2.0)


def estimate_local_sigma(img: np.ndarray, cx: float, cy: float,
                         patch_px: int) -> float:
    """세그먼트 주변 패치의 고주파 잔차 MAD × 1.4826 → 노이즈 σ (gray level).

    전역 σ는 TEM 두께 편차 때문에 무의미하므로 쓰지 않는다 (spec §3.2).
    """
    h, w = img.shape
    half = patch_px // 2
    r0 = int(np.clip(round(cy) - half, 0, max(0, h - patch_px)))
    c0 = int(np.clip(round(cx) - half, 0, max(0, w - patch_px)))
    patch = img[r0:r0 + patch_px, c0:c0 + patch_px].astype(np.float64)
    if patch.size < 9:
        return np.nan
    resp = convolve(patch, _IMMERKAER, mode="reflect")
    mad = np.median(np.abs(resp - np.median(resp)))
    return float(1.4826 * mad / _IMMERKAER_NORM)


def sample_ribbon_profiles(img: np.ndarray, S: np.ndarray, E: np.ndarray,
                           tangents: np.ndarray, sampling_cfg: dict) -> list[Profile]:
    """이미지 내 CD들의 리본 프로파일을 배치로 샘플링.

    S, E: (n, 2) 내부 좌표 (x=col, y=row). tangents: (n, 2) 리본 오프셋 방향.
    """
    half_w = int(sampling_cfg["ribbon_half_w"])
    patch_px = int(sampling_cfg["noise_patch_px"])
    offsets = np.arange(-half_w, half_w + 1, dtype=np.float64)

    imgf = img.astype(np.float64)
    seg = E - S
    lengths = np.linalg.norm(seg, axis=1)

    # 샘플 좌표를 전부 모아 한 번에 보간
    all_pts: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    units: list[np.ndarray] = []
    for i in range(len(S)):
        L = lengths[i]
        u = seg[i] / L if L > 0 else np.array([1.0, 0.0])
        units.append(u)
        margin = effective_margin(L, sampling_cfg)
        s_grid = np.arange(-margin, L + margin + DS, DS)
        grids.append(s_grid)
        # (n_offsets, n_s, 2)
        pts = (S[i][None, None, :]
               + u[None, None, :] * s_grid[None, :, None]
               + tangents[i][None, None, :] * offsets[:, None, None])
        all_pts.append(pts.reshape(-1, 2))

    flat = np.concatenate(all_pts, axis=0)
    # map_coordinates는 (row, col) 순서
    vals = map_coordinates(imgf, [flat[:, 1], flat[:, 0]], order=1, mode="nearest")

    profiles: list[Profile] = []
    pos = 0
    for i in range(len(S)):
        n_s = len(grids[i])
        n_pts = len(offsets) * n_s
        block = vals[pos:pos + n_pts].reshape(len(offsets), n_s)
        pos += n_pts
        p = block.mean(axis=0)
        mid = (S[i] + E[i]) / 2
        sigma = estimate_local_sigma(img, mid[0], mid[1], patch_px)
        # 리본 평균은 노이즈를 1/sqrt(n_offsets)로 줄인다 — CNR은 프로파일 기준
        sigma_prof = sigma / np.sqrt(len(offsets))
        profiles.append(Profile(s=grids[i], p=p, L=float(lengths[i]),
                                sigma=sigma_prof, u=units[i], tangent=tangents[i]))
    return profiles
