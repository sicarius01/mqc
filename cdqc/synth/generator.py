"""합성 TEM 이미지/측정 생성기 — 개발(사외) 검증 전용 자산 (spec §7).

라이브러리 사용자(사내)와 무관하다. 파일 I/O 없이 메모리 내에서
(records DataFrame, {image_id: 이미지}) 를 만들어 selftest/pytest가 쓴다.
눈 확인용 저장은 save_dataset() 옵션.

이미지 모델:
    img = background(gradient) + Σ bands(erf edge profile) + fringe(옵션)
          + noise(Poisson-like) → clip → uint8

- 밴드(층) k의 좌/우 경계가 엣지. 카테고리 k = 그 밴드의 (좌, 우) 엣지 쌍.
- Ground truth 좌표는 내부 컨벤션 (x=col, y=row, zero-origin).
- "레시피 출력" = 참 좌표 + 지터. 실패는 inject.py가 주입.
- value = **참 길이 × px_nm** (주입 후에도 유지 → value_mismatch 경로 검증),
  단위는 행마다 Å/nm 교대 (행별 단위 변환 경로 검증).

합성 PASS는 "기계적으로 맞다"이지 실제 TEM 실패를 잡는다는 증명이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.special import erf

from . import inject

RECIPE_ID = "synthR"
SQRT2 = np.sqrt(2.0)


def _default_inject() -> dict:
    return {
        "defocus": [0, 1, 2, 4],                 # rise 배수 (0 = 주입 없음)
        "low_contrast": [1.0, 0.5, 0.25],
        "noise_up": [1.0, 2.0, 4.0],
        "edge_jump_nm": [0, 1, 2, 5],            # 최대값은 탐색 창(W/2) 안
        "systematic_bias_nm": [0, 1, 3],
        "oblique_deg": [0, 10, 20],
        "missing_frac": [0, 0.1, 0.3],
        "double_edge_offset_px": [0, 4, 8],
        "plateau_defect": [False, True],
        "saturation": [1.0, 2.0, 4.0],
        "partial_damage": [0, 0.25],
        "mask_shift_px": [0, 2, 4],              # 마스크만 평행이동 (이미지 정상)
        "mask_ragged_p": [0, 0.15, 0.3],         # 경계 픽셀 플립 확률
        "rotated_frame_deg": [0, 10, 20],        # 시퀀스 전체 회전 배치 + CD 수 변경
    }


@dataclass
class SynthParams:
    """합성 생성 파라미터 (개발 전용 — 구 config [synthetic]+[selftest] 일부)."""
    n_images: int = 30                 # 베이스라인(정상) 이미지 수
    image_size: tuple[int, int] = (512, 512)
    n_categories: int = 3
    cds_per_category: int = 20
    px_nm: float = 0.5
    base_contrast: float = 60.0
    noise_sigma: float = 6.0
    edge_rise_px: float = 1.5
    coord_jitter_px: float = 0.3       # 정상 레시피 출력 지터 (DL 컨투어 현실치)
    fringe: bool = False
    n_images_per_case: int = 4         # (실패, 강도) 조합당 이미지 수
    seed: int = 42
    inject: dict = field(default_factory=_default_inject)


@dataclass
class Band:
    """수직 밴드 하나. 좌/우 엣지는 기울기+사인 굴곡을 가진 매끄러운 궤적."""
    cx: float
    w: float
    tilt_l: float
    tilt_r: float
    amp: float
    lam: float
    phase: float

    def x_left(self, y: np.ndarray, H: int) -> np.ndarray:
        return (self.cx - self.w / 2 + self.tilt_l * (y - H / 2)
                + self.amp * np.sin(2 * np.pi * y / self.lam + self.phase))

    def x_right(self, y: np.ndarray, H: int) -> np.ndarray:
        return (self.cx + self.w / 2 + self.tilt_r * (y - H / 2)
                + self.amp * np.sin(2 * np.pi * y / self.lam + self.phase))


@dataclass
class Scene:
    H: int
    W: int
    px_nm: float
    contrast: float
    rise: float
    noise_sigma: float
    fringe: bool
    bands: list[Band] = field(default_factory=list)


def make_scene(sp: SynthParams, rng: np.random.Generator) -> Scene:
    H, W = int(sp.image_size[0]), int(sp.image_size[1])
    n = sp.n_categories
    scene = Scene(H=H, W=W, px_nm=sp.px_nm, contrast=sp.base_contrast,
                  rise=sp.edge_rise_px, noise_sigma=sp.noise_sigma,
                  fringe=sp.fringe)
    centers = np.linspace(W / (n + 1), W * n / (n + 1), n)
    for cx in centers:
        base_tilt = rng.uniform(-0.03, 0.03)
        scene.bands.append(Band(
            cx=float(cx + rng.uniform(-8, 8)),
            w=float(np.clip(rng.normal(56, 3.0), 40, min(70, W / (n + 1) - 10))),
            tilt_l=base_tilt + rng.uniform(-0.003, 0.003),
            tilt_r=base_tilt + rng.uniform(-0.003, 0.003),
            amp=float(rng.uniform(0.5, 1.5)),
            lam=float(rng.uniform(250, 400)),
            phase=float(rng.uniform(0, 2 * np.pi)),
        ))
    return scene


def render(scene: Scene, mods: inject.ImageMods, rng: np.random.Generator) -> np.ndarray:
    """장면 + 주입 모드 → uint8 이미지."""
    H, W = scene.H, scene.W
    y = np.arange(H, dtype=np.float64)
    X = np.arange(W, dtype=np.float64)[None, :]
    img = 80.0 + 10.0 * (X / W) + 5.0 * (y[:, None] / H)

    s = scene.rise * mods.rise_mult
    contrast = scene.contrast * mods.contrast_mult
    for band in scene.bands:
        xl = band.x_left(y, H)[:, None]
        xr = band.x_right(y, H)[:, None]
        img += contrast * 0.5 * (erf((X - xl) / (s * SQRT2))
                                 - erf((X - xr) / (s * SQRT2)))
        if scene.fringe and mods.fringe_gain > 0:
            for xe in (xl, xr):
                d = np.abs(X - xe)
                img += (mods.fringe_gain * contrast
                        * np.exp(-d / 4.0) * np.sin(2 * np.pi * d / 3.5))
        if mods.double_edge_offset > 0:
            # 밴드 안쪽(플래토)에 약한 고스트 라인 → margin/npk 반응
            xg = xl + mods.double_edge_offset
            img += 0.5 * contrast * np.exp(-((X - xg) ** 2) / (2 * 1.2 ** 2))

    for bx, by, amp in mods.plateau_blobs:
        d2 = (X - bx) ** 2 + (y[:, None] - by) ** 2
        img += amp * np.exp(-d2 / (2 * 3.0 ** 2))

    if mods.sat_mult != 1.0:
        img = (img - 128.0) * mods.sat_mult + 128.0

    sigma = scene.noise_sigma * mods.noise_mult
    img += rng.normal(0, 1, img.shape) * sigma * np.sqrt(np.clip(img, 1, 255) / 128.0)

    if mods.damage_tiles:
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        for (r0, r1, c0, c1) in mods.damage_tiles:
            img[r0:r1, c0:c1] = (blurred[r0:r1, c0:c1]
                                 + rng.normal(0, 35.0, (r1 - r0, c1 - c0)))

    return np.clip(img, 0, 255).astype(np.uint8)


def truth_records(scene: Scene, sp: SynthParams) -> list[dict]:
    """카테고리별 참 (sx, sy, ex, ey) + 보고 측정값(value, 단위 혼합)."""
    n_cd = sp.cds_per_category
    ys = np.linspace(40, scene.H - 40, n_cd)
    rows = []
    for k, band in enumerate(scene.bands):
        xl = band.x_left(ys, scene.H)
        xr = band.x_right(ys, scene.H)
        cat = chr(ord("A") + k)
        for i in range(n_cd):
            length_nm = float(np.hypot(xr[i] - xl[i], 0.0)) * scene.px_nm
            angstrom = (len(rows) % 2 == 0)
            rows.append({"category_id": cat, "cd_index": i,
                         "sx": float(xl[i]), "sy": float(ys[i]),
                         "ex": float(xr[i]), "ey": float(ys[i]),
                         "value": length_nm * 10 if angstrom else length_nm,
                         "unit": "Å" if angstrom else "nm",
                         "band": k})
    return rows


def truth_mask(scene: Scene, band_idx: int) -> np.ndarray:
    """밴드 정의에서 직접 만든 참 이진 마스크 (카테고리 하나 분량)."""
    y = np.arange(scene.H, dtype=np.float64)
    X = np.arange(scene.W, dtype=np.float64)[None, :]
    band = scene.bands[band_idx]
    xl = band.x_left(y, scene.H)[:, None]
    xr = band.x_right(y, scene.H)[:, None]
    return (X >= xl) & (X <= xr)


def _apply_mask_mods(mask: np.ndarray, mods: inject.ImageMods,
                     rng: np.random.Generator) -> np.ndarray:
    m = mask.copy()
    d = int(round(mods.mask_shift_px))
    if d > 0:                                   # +x 평행이동 (이미지는 정상)
        shifted = np.zeros_like(m)
        shifted[:, d:] = m[:, :-d]
        m = shifted
    if mods.mask_ragged_p > 0:
        from scipy.ndimage import binary_dilation
        from ..features.mask import boundary_pixels
        p = mods.mask_ragged_p
        # 1) 경계 안쪽 1px 플립 → 경계 거칠기 (notch — 성분 수엔 영향 없음)
        inner = boundary_pixels(m)
        m = m ^ (inner & (rng.random(m.shape) < p))
        # 2) 바깥 거리-4 링에 저밀도 고립 잡티 → 성분 수. 링이 마스크에서
        #    떨어져 있어 병합이 없고, q<1/9 영역이라 개수가 p에 단조 증가
        d4 = binary_dilation(mask, iterations=4)
        d3 = binary_dilation(mask, iterations=3)
        ring = d4 & ~d3
        m = m | (ring & (rng.random(m.shape) < p / 3.0))
    return m


def generate_dataset(sp: SynthParams | None = None
                     ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    """전체 케이스(베이스라인 + 실패×강도) 생성 — 메모리 내.

    반환: (records, images, masks)
    - records: 행=CD. 컬럼 image_id, category_id, cd_index, sx..ey, value,
      unit + truth(injected_failure, injected_strength, sev_rank, affected)
    - images: {image_id: np.ndarray[uint8]}
    - masks: {(image_id, category_id): bool ndarray} — DL 마스크 대역
    """
    sp = sp or SynthParams()
    cases = inject.build_cases(sp)
    all_rows: list[dict] = []
    images: dict[str, np.ndarray] = {}
    masks: dict[tuple[str, str], np.ndarray] = {}
    for ci, case in enumerate(cases):
        for ii in range(case.n_images):
            rng = np.random.default_rng([sp.seed, ci, ii])
            scene = make_scene(sp, rng)
            image_id = f"{case.failure}_{case.sev_rank}_{ii:02d}"
            mods, rec_rows = inject.apply_case(case, scene,
                                               truth_records(scene, sp), sp, rng)
            img = render(scene, mods, rng)
            if mods.damage_tiles:
                for r in rec_rows:
                    mx, my = (r["sx"] + r["ex"]) / 2, (r["sy"] + r["ey"]) / 2
                    for (r0, r1, c0, c1) in mods.damage_tiles:
                        if r0 <= my < r1 and c0 <= mx < c1:
                            r["affected"] = 1
            images[image_id] = img
            for k in range(sp.n_categories):
                cat = chr(ord("A") + k)
                masks[(image_id, cat)] = _apply_mask_mods(
                    truth_mask(scene, k), mods, rng)
            for r in rec_rows:
                r.pop("band", None)
                r.update({"recipe_id": RECIPE_ID, "image_id": image_id,
                          "injected_failure": case.failure,
                          "injected_strength": str(case.strength),
                          "sev_rank": case.sev_rank})
                all_rows.append(r)

    cols = ["recipe_id", "image_id", "category_id", "cd_index",
            "sx", "sy", "ex", "ey", "value", "unit",
            "injected_failure", "injected_strength", "sev_rank", "affected"]
    records = pd.DataFrame(all_rows)[cols]
    return records, images, masks


def save_dataset(records: pd.DataFrame, images: dict[str, np.ndarray],
                 out_dir: str | Path, masks: dict | None = None) -> None:
    """눈 확인용 저장 (PNG + CSV). 검증 경로에는 불필요."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for image_id, img in images.items():
        cv2.imwrite(str(out / "images" / f"{image_id}.png"), img)
    if masks:
        (out / "masks").mkdir(parents=True, exist_ok=True)
        for (image_id, cat), m in masks.items():
            cv2.imwrite(str(out / "masks" / f"{image_id}_{cat}.png"),
                        m.astype(np.uint8) * 255)
    records.to_csv(out / "records.csv", index=False)
