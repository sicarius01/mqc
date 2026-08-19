import numpy as np
import polars as pl
import pytest

from cdqc.normalize import CohortStats, robust_stats


def test_robust_stats_trim_resists_contamination():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    x[:30] += 50.0
    med, mad = robust_stats(x, trim_frac=0.05)
    assert abs(med) < 0.2
    assert 0.5 < mad < 1.0


def _l3_frame(n=300, recipe="r1", cat="A", seed=1):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "recipe_id": [recipe] * n,
        "image_id": [f"i{k % 10}" for k in range(n)],
        "category_id": [cat] * n,
        "cd_index": list(range(n)),
        "delta_s": rng.normal(0, 0.1, n),     # both
        "cnr_s": rng.normal(10, 1, n),        # low
        "rise_s": rng.lognormal(0, 0.2, n),   # high + log
        "pol_s": np.ones(n),                  # match
        "edge_valid_s": [True] * n,           # bool
    })


def test_directed_z_signs(cfg):
    df = _l3_frame()
    cs = CohortStats()
    cs.compute_level(df, "l3", cfg)
    # 나쁜 방향의 행 추가: delta 크게, cnr 낮게, pol 뒤집힘, edge_valid False
    bad = df[:1].with_columns(
        pl.lit(1.0).alias("delta_s"), pl.lit(2.0).alias("cnr_s"),
        pl.lit(-1.0).alias("pol_s"), pl.lit(False).alias("edge_valid_s"))
    out = cs.apply(pl.concat([df, bad]), "l3", cfg)
    row = out.row(-1, named=True)
    assert row["z_delta_s"] > 5          # both → |z| 큼
    assert row["z_cnr_s"] > 5            # low → 낮을수록 z 큼
    assert row["z_pol_s"] == pytest.approx(4.0)
    assert row["z_edge_valid_s"] == pytest.approx(4.0)
    assert "zs_delta_s" in out.columns   # both는 부호 z도 보존
    good = out.row(0, named=True)
    assert abs(good["z_pol_s"]) < 1e-9


def test_fallback_levels(cfg):
    cfg2 = cfg.with_overrides({"cohort": {"min_cohort_n": 100}})
    big = _l3_frame(n=300, cat="A")
    small = _l3_frame(n=20, cat="B", seed=2)
    df = pl.concat([big, small])
    cs = CohortStats()
    cs.compute_level(df, "l3", cfg2)
    out = cs.apply(df, "l3", cfg2)
    fb_a = out.filter(pl.col("category_id") == "A")["cohort_fallback_level"]
    fb_b = out.filter(pl.col("category_id") == "B")["cohort_fallback_level"]
    assert fb_a.max() == 0
    assert fb_b.min() >= 1               # 표본 부족 → 상위 티어


def test_mad_floor_prevents_blowup(cfg):
    n = 250
    df = _l3_frame(n=n).with_columns(pl.lit(1.0).alias("npk_s"))  # MAD=0 이산
    cs = CohortStats()
    cs.compute_level(df, "l3", cfg)
    bad = df[:1].with_columns(pl.lit(3.0).alias("npk_s"))
    out = cs.apply(pl.concat([df, bad]), "l3", cfg)
    z = out.row(-1, named=True)["z_npk_s"]
    assert z == pytest.approx((3.0 - 1.0) / 0.5)   # floor 0.5


def test_serialization_roundtrip(cfg):
    df = _l3_frame()
    cs = CohortStats()
    cs.compute_level(df, "l3", cfg)
    from cdqc.config import dump_toml
    import tomllib
    back = CohortStats.from_dict(tomllib.loads(dump_toml(cs.to_dict())))
    out1 = cs.apply(df, "l3", cfg)["z_delta_s"].to_numpy()
    out2 = back.apply(df, "l3", cfg)["z_delta_s"].to_numpy()
    assert np.allclose(out1, out2, equal_nan=True)
