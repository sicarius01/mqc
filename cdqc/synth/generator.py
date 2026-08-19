"""합성 TEM 이미지/측정 생성기 — 자가검증의 핵심 자산 (spec §7).

이미지 모델:
    img = background(gradient) + Σ bands(erf edge profile) + fringe(옵션)
          + noise(Poisson-like) → clip → uint8

- 밴드(층) k의 좌/우 경계가 엣지. 카테고리 k = 그 밴드의 (좌, 우) 엣지 쌍.
- Ground truth 좌표는 내부 컨벤션 (x=col, y=row, zero-origin).
- "레시피 출력" = 참 좌표 + 소량 지터. 실패는 inject.py가 주입.

합성 PASS는 "기계적으로 맞다"이지 실제 TEM 실패를 잡는다는 증명이 아니다
(spec §1). 유효성은 실데이터 라벨로만 증명한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from scipy.special import erf

from ..config import Config
from . import inject

RECIPE_ID = "synthR"
COORD_JITTER_PX = 0.05   # 정상 레시피 출력의 좌표 지터
SQRT2 = np.sqrt(2.0)


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


def make_scene(cfg: Config, rng: np.random.Generator) -> Scene:
    syn = cfg["synthetic"]
    H, W = int(syn["image_size"][0]), int(syn["image_size"][1])
    n = int(syn["n_categories"])
    scene = Scene(H=H, W=W, px_nm=float(syn["px_nm"]),
                  contrast=float(syn["base_contrast"]),
                  rise=float(syn["edge_rise_px"]),
                  noise_sigma=float(syn["noise_sigma"]),
                  fringe=bool(syn["fringe"]))
    centers = np.linspace(W / (n + 1), W * n / (n + 1), n)
    for cx in centers:
        base_tilt = rng.uniform(-0.03, 0.03)
        # 폭 변동은 작게 유지 — cd_nm 코호트 산포가 크면 oblique류의
        # 과대 CD가 z에 묻힌다 (실데이터에선 코호트 자체가 이 산포를 정의)
        scene.bands.append(Band(
            cx=float(cx + rng.uniform(-8, 8)),
            w=float(np.clip(rng.normal(56, 0.5), 40, min(70, W / (n + 1) - 10))),
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


def truth_records(scene: Scene, cfg: Config) -> list[dict]:
    """카테고리별 참 (sx, sy, ex, ey). cd_index는 y 오름차순."""
    syn = cfg["synthetic"]
    n_cd = int(syn["cds_per_category"])
    ys = np.linspace(40, scene.H - 40, n_cd)
    rows = []
    for k, band in enumerate(scene.bands):
        xl = band.x_left(ys, scene.H)
        xr = band.x_right(ys, scene.H)
        cat = chr(ord("A") + k)
        for i in range(n_cd):
            rows.append({"category_id": cat, "cd_index": i,
                         "sx": float(xl[i]), "sy": float(ys[i]),
                         "ex": float(xr[i]), "ey": float(ys[i]),
                         "band": k})
    return rows


def generate_dataset(cfg: Config, out_root: Path) -> pl.DataFrame:
    """전체 케이스(베이스라인 + 실패×강도)의 이미지와 레코드를 생성.

    out_root/images/*.png 와 out_root/records.csv 를 쓴다.
    레코드에는 truth 컬럼(injected_failure, injected_strength, sev_rank,
    affected)이 포함된다 — selftest가 조인해서 쓴다.
    """
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed()
    cases = inject.build_cases(cfg)

    all_rows: list[dict] = []
    for ci, case in enumerate(cases):
        for ii in range(case.n_images):
            rng = np.random.default_rng([seed, ci, ii])
            scene = make_scene(cfg, rng)
            image_id = f"{case.failure}_{case.sev_rank}_{ii:02d}"
            mods, rec_rows = inject.apply_case(case, scene, truth_records(scene, cfg),
                                               cfg, rng)
            img = render(scene, mods, rng)
            # 부분 손상 타일 안의 CD 표시 (렌더 후 확정되는 유일한 affected)
            if mods.damage_tiles:
                for r in rec_rows:
                    mx, my = (r["sx"] + r["ex"]) / 2, (r["sy"] + r["ey"]) / 2
                    for (r0, r1, c0, c1) in mods.damage_tiles:
                        if r0 <= my < r1 and c0 <= mx < c1:
                            r["affected"] = 1
            cv2.imwrite(str(img_dir / f"{image_id}.png"), img)
            for r in rec_rows:
                r.pop("band", None)
                r.update({
                    "recipe_id": RECIPE_ID,
                    "image_id": image_id,
                    "image_path": f"images/{image_id}.png",
                    "px_nm": scene.px_nm,
                    "injected_failure": case.failure,
                    "injected_strength": str(case.strength),
                    "sev_rank": case.sev_rank,
                })
                all_rows.append(r)

    df = pl.DataFrame(all_rows)
    std = ["recipe_id", "image_id", "image_path", "category_id", "cd_index",
           "sx", "sy", "ex", "ey", "px_nm",
           "injected_failure", "injected_strength", "sev_rank", "affected"]
    df = df.select(std).sort(["injected_failure", "sev_rank", "image_id",
                              "category_id", "cd_index"])
    df.write_csv(out_root / "records.csv")
    return df
