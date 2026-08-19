"""L2 — 카테고리 시퀀스 피쳐 (image × category).

여기서는 임계값과 무관한 부분만 계산한다 (extract는 순수 함수여야 하므로).
플래그에 의존하는 frac_flagged / max_run / impact_nm은 run 시점에
scoring.py가 채운다.
"""

from __future__ import annotations

import numpy as np
import polars as pl

GROUP_KEY = ["recipe_id", "image_id", "category_id"]


def _mad(name: str) -> pl.Expr:
    return (pl.col(name) - pl.col(name).median()).abs().median()


def _rms(name: str) -> pl.Expr:
    return (pl.col(name).pow(2).mean()).sqrt()


def l2_static_features(l3: pl.DataFrame) -> pl.DataFrame:
    """L3 프레임 → (recipe, image, category)별 임계값-독립 L2 피쳐."""
    return (
        l3.group_by(GROUP_KEY)
        .agg(
            pl.len().alias("n_cd"),
            pl.col("cd_nm").median().alias("cd_median"),
            _mad("cd_nm").alias("cd_mad"),
            pl.col("delta_s").median().alias("delta_median_s"),
            pl.col("delta_e").median().alias("delta_median_e"),
            _rms("s_resid").alias("traj_rms_s"),
            _rms("e_resid").alias("traj_rms_e"),
            pl.col("px_nm").first().alias("px_nm"),
        )
        .sort(GROUP_KEY)
        .with_columns(pl.col("n_cd").cast(pl.Float64))
    )
