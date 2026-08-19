"""판정 — CD 레벨 소프트 플래그와 이미지 레벨 게이트 (G0/G1/G2), 사유 코드.

CD 레벨은 느슨(고재현율, t_soft), 이미지 레벨은 엄격(집계 증거) — 두 층의
임계 철학이 다르다 (spec §4.3). impact_nm은 물리 단위라 스펙 공차와 직접
비교한다.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .features.l2 import GROUP_KEY
from .features.registry import BY_NAME, enabled_names, features_of

# 이미지 레벨 사유 코드
R_IMAGE_BAD = "IMAGE_BAD"
R_STAT_CONTAMINATED = "STAT_CONTAMINATED"
R_SYSTEMATIC_SHIFT = "SYSTEMATIC_SHIFT"
R_LOCAL = "IMAGE_BAD_LOCAL"       # 플래그 뭉침 → 국소 이미지 손상
R_SPORADIC = "LOCKON_SPORADIC"    # 플래그 산발 → 락온 실패


def _z_cols(df: pl.DataFrame, level: str, cfg, only_g2: bool = False,
            only_g0: bool = False) -> list[str]:
    names = enabled_names(cfg, level)
    out = []
    for n in names:
        f = BY_NAME[n]
        if only_g2 and not f.g2:
            continue
        if only_g0 and not f.g0:
            continue
        if f"z_{n}" in df.columns:
            out.append(f"z_{n}")
    return out


def score_l3(l3z: pl.DataFrame, cfg, t_soft: float) -> pl.DataFrame:
    """CD별 (flag, reason_code, top_feature, z) 추가."""
    cols = _z_cols(l3z, "l3", cfg)
    Z = np.column_stack([l3z[c].to_numpy() for c in cols]) if cols else np.empty((l3z.height, 0))
    if Z.shape[1] == 0:
        return l3z.with_columns(pl.lit(False).alias("flag"),
                                pl.lit("").alias("top_feature"),
                                pl.lit(np.nan).alias("top_z"),
                                pl.lit("").alias("reason_code"))
    Zf = np.where(np.isfinite(Z), Z, -np.inf)
    top_idx = np.argmax(Zf, axis=1)
    top_z = Zf[np.arange(len(Zf)), top_idx]
    all_nan = ~np.isfinite(Z).any(axis=1)
    top_z = np.where(all_nan, np.nan, top_z)
    flag = np.where(all_nan, False, top_z > t_soft)
    names = np.array([c[2:] for c in cols])
    top_feature = np.where(flag, names[top_idx], "")
    reasons = np.array([BY_NAME[n].reason for n in names])
    reason = np.where(flag, reasons[top_idx], "")
    return l3z.with_columns(
        pl.Series("flag", flag.astype(bool)),
        pl.Series("top_feature", top_feature),
        pl.Series("top_z", top_z),
        pl.Series("reason_code", reason),
    )


def _impact_stat(x: np.ndarray, cfg) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    mode = cfg["thresholds"]["impact"]["stat"]
    if mode == "mean":
        return float(np.mean(x))
    if mode == "trimmed_mean":
        f = float(cfg["thresholds"]["impact"]["trimmed_frac"])
        lo, hi = np.quantile(x, [f, 1 - f])
        inner = x[(x >= lo) & (x <= hi)]
        return float(np.mean(inner)) if len(inner) else float(np.mean(x))
    return float(np.median(x))


def _max_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def add_l2_runtime(l3_scored: pl.DataFrame, l2_static: pl.DataFrame,
                   cfg) -> pl.DataFrame:
    """플래그 의존 L2 피쳐 (frac_flagged, max_run, impact_nm)를 채운다."""
    rows = []
    for keyvals, part in l3_scored.sort(GROUP_KEY + ["cd_index"]) \
                                  .partition_by(GROUP_KEY, as_dict=True).items():
        flags = part["flag"].to_numpy().astype(bool)
        cd = part["cd_nm"].to_numpy()
        stat_all = _impact_stat(cd, cfg)
        clean = cd[~flags]
        stat_clean = _impact_stat(clean, cfg) if len(clean) else np.nan
        if flags.any():
            impact = abs(stat_all - stat_clean) if len(clean) else np.inf
        else:
            impact = 0.0
        rows.append({
            "recipe_id": keyvals[0], "image_id": keyvals[1], "category_id": keyvals[2],
            "frac_flagged": float(flags.mean()),
            "max_run": int(_max_run(flags)),
            "impact_nm": float(impact),
        })
    runtime = pl.DataFrame(rows)
    return l2_static.join(runtime, on=GROUP_KEY, how="left")


def score_l2(l2z: pl.DataFrame, cfg, t_seq: float) -> pl.DataFrame:
    """카테고리(시퀀스)별 G1/G2 판정."""
    g2_cols = _z_cols(l2z, "l2", cfg, only_g2=True)
    enable_g1 = bool(cfg["gates"]["enable_g1"])
    enable_g2 = bool(cfg["gates"]["enable_g2"])
    run_min = int(cfg["gates"]["run_cluster_min"])

    tol = np.array([float(cfg["thresholds"]["tolerance_nm"].get(
        c, cfg["thresholds"]["tolerance_nm"]["default"])) for c in l2z["category_id"]])
    impact = l2z["impact_nm"].to_numpy()
    g1_hit = enable_g1 & np.where(np.isfinite(impact), impact > tol, impact == np.inf)

    if g2_cols:
        Z = np.column_stack([l2z[c].to_numpy() for c in g2_cols])
        zmax = np.max(np.where(np.isfinite(Z), Z, -np.inf), axis=1, initial=-np.inf)
        g2_hit = enable_g2 & (zmax > t_seq)
    else:
        g2_hit = np.zeros(l2z.height, dtype=bool)

    max_run = l2z["max_run"].to_numpy()
    pattern = np.where(max_run >= run_min, R_LOCAL, R_SPORADIC)
    return l2z.with_columns(
        pl.Series("g1_hit", g1_hit.astype(bool)),
        pl.Series("g2_hit", np.asarray(g2_hit, dtype=bool)),
        pl.Series("flag_pattern", pattern),
        pl.Series("tolerance_nm", tol),
    )


def score_images(l1z: pl.DataFrame, l2_scored: pl.DataFrame, cfg,
                 t_image: float) -> pl.DataFrame:
    """이미지 레벨 최종 판정 — 세 갈래 OR (spec §4.3)."""
    g0_cols = _z_cols(l1z, "l1", cfg, only_g0=True)
    enable_g0 = bool(cfg["gates"]["enable_g0"])
    if g0_cols and enable_g0:
        Z = np.column_stack([l1z[c].to_numpy() for c in g0_cols])
        zmax = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1, initial=-np.inf)
        g0_hit = zmax > t_image
    else:
        g0_hit = np.zeros(l1z.height, dtype=bool)
    l1s = l1z.with_columns(pl.Series("g0_hit", g0_hit.astype(bool)))

    per_img = (
        l2_scored.group_by(["recipe_id", "image_id"])
        .agg(
            pl.col("g1_hit").any().alias("g1_hit"),
            pl.col("g2_hit").any().alias("g2_hit"),
            (pl.col("g1_hit") | pl.col("g2_hit")).sum().alias("n_bad_categories"),
            pl.col("flag_pattern").filter(pl.col("g1_hit")).first().alias("g1_pattern"),
        )
    )
    out = l1s.join(per_img, on=["recipe_id", "image_id"], how="left").with_columns(
        pl.col("g1_hit").fill_null(False),
        pl.col("g2_hit").fill_null(False),
        pl.col("n_bad_categories").fill_null(0),
    )

    def reasons(row) -> str:
        r = []
        if row["g0_hit"]:
            r.append(R_IMAGE_BAD)
        if row["g1_hit"]:
            pat = row["g1_pattern"] or R_SPORADIC
            r.append(f"{R_STAT_CONTAMINATED}({pat})")
        if row["g2_hit"]:
            r.append(R_SYSTEMATIC_SHIFT)
        return "+".join(r)

    reason_col = [reasons(row) for row in out.iter_rows(named=True)]
    return out.with_columns(
        pl.Series("image_reasons", reason_col),
        (pl.col("g0_hit") | pl.col("g1_hit") | pl.col("g2_hit")).alias("image_fail"),
    )
