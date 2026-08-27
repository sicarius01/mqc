"""마스크 기반 피쳐 — DL segmentation 결과와 좌표/이미지의 정합.

두 경로가 있고 **라벨맵 경로가 권장 경로**다:

- **라벨맵 경로 (mask_maps + boundary_features)** — 라벨맵(단일 채널 int
  배열)의 *모든 라벨 전이의 합집합*을 경계로 삼는다. 클래스를 고를 필요가
  없어 S와 E가 서로 다른 계면에 있어도 측정되고, 거리맵을 **이미지당 1회**만
  계산한다. `bdist_ratio`는 "허공에 그은 선"(층 안에 떠 있는 CD)을 잡는다 —
  cnr로는 정상과 구분되지 않던 실패다.
- **이진 마스크 경로 (mask_l3_features / mask_image_features)** — 클래스
  하나만 보는 초기 방식. 마스크는 이진 ndarray(bool 또는 uint8 0/비0),
  이미지와 같은 shape. 클래스 선택이 필요하고, CD의 S/E가 다른 계면에 있으면
  한 클래스로는 측정할 수 없다. 호환을 위해 유지한다.

이미지 레벨 (mask_image_features): DL이 이미지 전체에서 헛것을 그렸는지
(mask_grad_agree), 성분/구멍/경계 거칠기 등 마스크 형태 자체의 이상.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import (binary_erosion, binary_fill_holes,
                           distance_transform_edt, gaussian_filter, label,
                           map_coordinates)

from ..sampling import DS, estimate_local_sigma
from .l1 import _grad_mag, _noise_sigma

_EIGHT = np.ones((3, 3), dtype=bool)   # 8-이웃 연결
_MIN_COMPONENT_PX = 16                 # boundary_rough 계산에 포함할 최소 성분 크기
                                       # (잡티가 성분별 중앙값을 지배하지 않게 —
                                       #  잡티 자체는 mask_n_components가 잡는다)
_SIGMA_TILE_PX = 32     # 국소 노이즈 σ 맵의 타일 크기
_SIGMA_FLOOR = 0.5      # gray level. uint8 양자화 수준 — 노이즈 없는 합성
                        # 이미지에서 σ→0으로 값이 폭주하는 것을 막는다
_AGREE_CAP = 1e3        # mask_grad_agree 상한 (log-z 안정화)
_BDIST_MID_FLOOR_PX = 0.5   # bdist_ratio 분모 하한 (경계 픽셀 양자화 수준)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """마스크 내부 경계 픽셀 (mask & ~erode). bool (H, W).

    **이미지 프레임은 경계가 아니다** (erosion의 border_value=1). 화면 밖으로
    이어지는 구조가 잘린 자리는 세그멘테이션이 그은 경계가 아니라 크롭
    아티팩트다. 프레임을 경계로 세면 이미지 끝까지 뻗은 마스크(TEM 밴드가
    보통 그렇다)에서 프레임 픽셀이 경계 표본을 지배해 `mask_grad_agree`의
    중앙값이 평탄부에 눌러앉고, `mdist`도 프레임까지의 거리를 잰다.
    """
    m = mask.astype(bool)
    return m & ~binary_erosion(m, structure=_EIGHT, border_value=1)


def local_sigma_map(img_f: np.ndarray, tile_px: int = _SIGMA_TILE_PX) -> np.ndarray:
    """타일별 노이즈 σ를 원본 크기로 펼친 맵 (gray level, 하한 _SIGMA_FLOOR).

    전역 σ는 TEM 두께 편차 때문에 구조 대비에 물린다 — 구조가 강한 이미지에서는
    σ가 부풀어 경계가 진짜 전이 위에 있어도 비가 작게 나오고, 균일한 이미지에서는
    σ가 0에 붙어 값이 폭주한다. mgrad가 국소 σ를 쓰기 때문에 잘 작동하는 것과
    같은 이유로 여기도 국소 σ를 쓴다 (spec §3.5).
    """
    h, w = img_f.shape
    ny = max(1, h // tile_px)
    nx = max(1, w // tile_px)
    coarse = np.empty((ny, nx), dtype=np.float64)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    for i in range(ny):
        for j in range(nx):
            coarse[i, j] = _noise_sigma(img_f[ys[i]:ys[i + 1], xs[j]:xs[j + 1]])
    coarse = np.nan_to_num(coarse, nan=0.0, posinf=0.0, neginf=0.0)
    if (ny, nx) == (h, w):
        full = coarse
    else:
        full = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(full, _SIGMA_FLOOR)


# ------------------------------------------------- 라벨맵 경로 (권장)

def mask_maps(labelmap: np.ndarray, img: np.ndarray | None = None,
              blur_px: float = 1.0) -> dict:
    """라벨맵 → 경계/거리 맵. **이미지당 1회만** 계산해서 재사용한다.

    경계 = 4-이웃 라벨 전이의 **합집합**, 전이 양쪽 픽셀을 모두 표시한다
    (참 계면은 두 픽셀 중심 사이에 있으므로 어느 쪽을 고르든 ±0.5px 양자화가
    남는다 — 양쪽 표시가 "경계 위 좌표 → 거리 0"을 보장한다). 배열 밖은 비교
    대상이 아니므로 **이미지 프레임은 경계가 되지 않는다** — `boundary_pixels`가
    밟았던 함정이 여기엔 없다.

    클래스를 고르지 않으므로:
    - S와 E가 서로 다른 계면 위에 있어도 둘 다 측정된다
    - 카테고리 × 클래스마다 distance transform을 돌리지 않는다 (이미지당 1회)

    img를 주면 그래디언트/국소 σ 맵도 함께 만들어 `bgrad_s/e`를 낼 수 있다.

    반환: {"labels", "boundary"(bool), "dist_px", "shape", "grad_snr"(또는 None)}.
    """
    lab = np.asarray(labelmap)
    if lab.ndim != 2:
        raise ValueError(f"labelmap must be 2-D, got ndim={lab.ndim}")
    b = np.zeros(lab.shape, dtype=bool)
    dv = lab[:-1, :] != lab[1:, :]
    dh = lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= dv
    b[1:, :] |= dv
    b[:, :-1] |= dh
    b[:, 1:] |= dh
    dist = (distance_transform_edt(~b) if b.any()
            else np.full(lab.shape, np.inf, dtype=np.float64))

    grad_snr = None
    if img is not None:
        if img.shape != lab.shape:
            raise ValueError(f"img{img.shape} vs labelmap{lab.shape}")
        gi = gaussian_filter(img.astype(np.float64), float(blur_px))
        gy, gx = np.gradient(gi)
        # 국소 σ로 나눠 무차원화 — 이미지 간·세션 간 비교가 가능해진다
        # (원본 스크립트는 raw gray-level 그래디언트였다)
        grad_snr = np.hypot(gx, gy) / local_sigma_map(img.astype(np.float64))

    return {"labels": lab, "boundary": b, "dist_px": dist, "shape": lab.shape,
            "grad_snr": grad_snr}


def boundary_features(maps: dict, S: np.ndarray, E: np.ndarray,
                      px_nm: float, cfg: dict | None = None
                      ) -> dict[str, np.ndarray]:
    """한 시퀀스의 CD별 라벨 경계 정합 피쳐. 반환: 이름 → (n,) 배열.

    - `bdist_s/e`: 보고 좌표 ↔ 가장 가까운 라벨 전이 거리 (nm). 정상 계면
      측정이면 ≈0.
    - `bdist_mid`: 세그먼트 중점 ↔ 라벨 전이 거리 (nm). 중점이 층 안에 있으면
      층 반폭 정도가 되므로 `bdist_ratio`의 분모(스케일)로 쓴다.
    - `bdist_ratio` = max(bdist_s, bdist_e) / bdist_mid — **배율에 무관**.
      정상 CD(경계를 가로지름) ≈ 0, 허공에 그은 선(층에 떠 있음) ≈ 1.
    - `label_runs`: 라벨 런 개수. **끝점에서 label_inset_px 안쪽 구간만** 센다
      — 끝점은 계면 위에 있어 라벨이 양쪽을 오가므로, 끝까지 세면 정상 CD의
      런 수가 측정마다 흔들린다.
    - `label_s/e/mid`: 각 지점의 라벨값 (이 CD가 어느 층을 재는가).
    - `bgrad_s/e`: 보고 좌표 **그 자리**의 이미지 그래디언트/국소 σ.
      `mask_maps`에 img를 준 경우에만, 아니면 NaN. `cnr`(프로파일 기반)과
      독립적인 증거라 둘이 어긋나면 그 자체가 정보다.
    """
    cfg = cfg or {}
    inset = float(cfg.get("label_inset_px", 2.0))
    dist, labels, gsnr = maps["dist_px"], maps["labels"], maps.get("grad_snr")
    S = np.asarray(S, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    n = len(S)
    mid = (S + E) / 2.0
    seg = E - S
    L = np.linalg.norm(seg, axis=1)
    u = seg / np.where(L > 0, L, 1.0)[:, None]

    def sample(a, P: np.ndarray, order: int = 1) -> np.ndarray:
        if a is None:
            return np.full(len(P), np.nan)
        return map_coordinates(a, [P[:, 1], P[:, 0]], order=order,
                               mode="nearest")

    def sample_dist(P: np.ndarray) -> np.ndarray:
        if not np.isfinite(dist).any():
            return np.full(len(P), np.nan)
        return sample(dist, P) * float(px_nm)

    b_s, b_e, b_m = sample_dist(S), sample_dist(E), sample_dist(mid)
    floor_nm = _BDIST_MID_FLOOR_PX * float(px_nm)
    ratio = np.maximum(b_s, b_e) / np.maximum(b_m, floor_nm)

    labf = labels.astype(np.float64)
    runs = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(L[i]) or L[i] <= 0:
            continue
        # 끝점은 계면 위 → 라벨이 오간다. 안쪽으로 inset만큼 물러나서 센다
        pad = inset if L[i] > 4 * inset else 0.0
        t = np.arange(pad, L[i] - pad + DS, DS)
        if len(t) < 2:
            t = np.array([L[i] / 2.0])
        pts = S[i][None, :] + u[i][None, :] * t[:, None]
        v = sample(labels, pts, order=0)
        runs[i] = 1.0 + float(np.count_nonzero(v[1:] != v[:-1]))

    return {"bdist_s": b_s, "bdist_e": b_e, "bdist_mid": b_m,
            "bdist_ratio": ratio, "label_runs": runs,
            "bgrad_s": sample(gsnr, S), "bgrad_e": sample(gsnr, E),
            "label_s": np.round(sample(labf, S + u * inset)),
            "label_e": np.round(sample(labf, E - u * inset)),
            "label_mid": np.round(sample(labf, mid))}


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

    # DL 경계가 실제 명암 전이 위에 있는지의 단일 지표.
    # 분모는 **국소** σ — 전역 σ는 구조 대비에 물려 판별력이 사라진다
    # (마스크를 0/5/20px 밀어도 값이 거의 안 움직였던 원인). 픽셀별 비의
    # 중앙값이라 mgrad와 같은 스케일로 읽힌다.
    if bnd.any():
        grad = _grad_mag(imgf)
        sig = local_sigma_map(imgf)
        agree = float(np.median(np.clip(grad[bnd] / sig[bnd], 0.0, _AGREE_CAP)))
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
