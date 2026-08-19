"""추출(extract)과 실행(run) 오케스트레이션.

extract: 레코드+이미지 → L3/L2/L1 피쳐 parquet 캐시. 코호트/임계값과 무관한
순수 함수 (이미지 I/O는 이미지당 딱 한 번). run: 캐시 → 정규화 → 판정.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from . import io
from .config import Config
from .errors import CdqcError, warn
from .features.l1 import hist_emd, l1_image_features
from .features.l2 import l2_static_features
from .features.l3 import l3_sequence_features
from .normalize import CohortStats
from .scoring import add_l2_runtime, score_images, score_l2, score_l3

CACHE_FILES = ("features_l3.parquet", "features_l2.parquet", "features_l1.parquet")


def cache_paths(cfg: Config) -> tuple[Path, Path, Path]:
    d = cfg.path("cache_dir")
    return tuple(d / f for f in CACHE_FILES)  # type: ignore[return-value]


def extract_features(cfg: Config, records: pl.DataFrame | None = None,
                     force: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """피쳐 추출 (캐시 있으면 스킵, --force로 재추출). 반환 (l3, l2, l1)."""
    p3, p2, p1 = cache_paths(cfg)
    if not force and all(p.exists() for p in (p3, p2, p1)):
        return pl.read_parquet(p3), pl.read_parquet(p2), pl.read_parquet(p1)

    if records is None:
        records = io.load_records(cfg)
    conv = io.get_convention(cfg)

    l3_parts: list[pl.DataFrame] = []
    l1_rows: list[dict] = []
    for (recipe_id, image_id), img_part in records.partition_by(
            ["recipe_id", "image_id"], as_dict=True).items():
        path = io.resolve_image_path(cfg, img_part[0, "image_path"])
        img = io.load_image(path)

        row = {"recipe_id": recipe_id, "image_id": image_id}
        feats = l1_image_features(img)
        row.update({k: (list(v) if k == "hist" else v) for k, v in feats.items()})
        l1_rows.append(row)

        for (cat,), seq in img_part.partition_by(["category_id"],
                                                 as_dict=True).items():
            seq = seq.sort("cd_index")
            sx, sy = io.transform_coords(seq["sx"].to_numpy(), seq["sy"].to_numpy(),
                                         conv, img.shape)
            ex, ey = io.transform_coords(seq["ex"].to_numpy(), seq["ey"].to_numpy(),
                                         conv, img.shape)
            io.check_coords_in_bounds(np.concatenate([sx, ex]),
                                      np.concatenate([sy, ey]), img.shape,
                                      f"image={image_id} cat={cat}")
            S = np.stack([sx, sy], axis=1)
            E = np.stack([ex, ey], axis=1)
            px_nm = float(seq[0, "px_nm"])
            feats3 = l3_sequence_features(img, S, E, px_nm, cfg)
            base = {
                "recipe_id": np.repeat(recipe_id, len(seq)),
                "image_id": np.repeat(image_id, len(seq)),
                "image_path": np.repeat(img_part[0, "image_path"], len(seq)),
                "category_id": np.repeat(cat, len(seq)),
                "cd_index": seq["cd_index"].to_numpy(),
                "px_nm": np.repeat(px_nm, len(seq)),
                # 내부 좌표 보존 — 오버레이/국소화용 (internal 전용 출력에만 씀)
                "ix_s": sx, "iy_s": sy, "ix_e": ex, "iy_e": ey,
            }
            base.update(feats3)
            l3_parts.append(pl.DataFrame(base))

    if not l3_parts:
        raise CdqcError("E-DATA-08")
    l3 = pl.concat(l3_parts).sort(["recipe_id", "image_id", "category_id", "cd_index"])
    l2 = l2_static_features(l3)
    l1 = pl.DataFrame(l1_rows).sort(["recipe_id", "image_id"])

    d = cfg.path("cache_dir")
    d.mkdir(parents=True, exist_ok=True)
    l3.write_parquet(p3)
    l2.write_parquet(p2)
    l1.write_parquet(p1)
    (d / "extract_meta.json").write_text(json.dumps({
        "config_hash": cfg.config_hash, "n_cd": l3.height, "n_images": l1.height,
    }), encoding="utf-8")
    return l3, l2, l1


def add_hist_emd(l1: pl.DataFrame, cs: CohortStats) -> pl.DataFrame:
    """L1 프레임에 hist_emd 컬럼 추가 (코호트 히스토그램 템플릿 대비)."""
    vals = []
    for row in l1.iter_rows(named=True):
        t = cs.template_for(row["recipe_id"])
        vals.append(hist_emd(np.asarray(row["hist"]), t) if t is not None else np.nan)
    return l1.with_columns(pl.Series("hist_emd", vals))


def resolve_calibration(cfg: Config, l3: pl.DataFrame, l2s: pl.DataFrame,
                        l1: pl.DataFrame) -> tuple[CohortStats, dict]:
    """calibrated.toml이 있으면 그걸, 없으면 즉석 캘리브레이션 (경고)."""
    if cfg.calibrated.get("cohort_stats"):
        cs = CohortStats.from_dict(cfg.calibrated)
        thr = {name: cfg.threshold(name) for name in ("t_soft", "t_image", "t_seq")}
        return cs, thr
    warn("W-CAL-01")
    from .calibrate import calibrate_frames
    cs, auto_thr = calibrate_frames(cfg, l3, l2s, l1, write=False)
    thr = {}
    for name in ("t_soft", "t_image", "t_seq"):
        v = cfg["thresholds"][name]
        thr[name] = float(v) if v != "auto" else auto_thr[name]
    return cs, thr


def run_pipeline(cfg: Config, force: bool = False) -> dict:
    """extract(캐시) → 정규화 → 판정. 리포트 출력은 호출부(__main__) 담당."""
    l3, l2s, l1 = extract_features(cfg, force=force)
    cs, thr = resolve_calibration(cfg, l3, l2s, l1)

    l1e = add_hist_emd(l1, cs)
    l3z = cs.apply(l3, "l3", cfg)
    l3s = score_l3(l3z, cfg, thr["t_soft"])
    l2full = add_l2_runtime(l3s, l2s, cfg)
    l2z = cs.apply(l2full, "l2", cfg)
    l2sc = score_l2(l2z, cfg, thr["t_seq"])
    l1z = cs.apply(l1e, "l1", cfg)
    imgs = score_images(l1z, l2sc, cfg, thr["t_image"])

    l3s = l3s.sort(["recipe_id", "image_id", "category_id", "cd_index"])
    return {"l3": l3s, "l2": l2sc.sort(["recipe_id", "image_id", "category_id"]),
            "l1": imgs.sort(["recipe_id", "image_id"]),
            "thresholds": thr, "cohort_stats": cs}
