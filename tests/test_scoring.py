import numpy as np
import polars as pl
import pytest

from cdqc.scoring import (_max_run, add_l2_runtime, score_images, score_l2,
                          score_l3)


def test_max_run():
    assert _max_run(np.array([0, 1, 1, 1, 0, 1], dtype=bool)) == 3
    assert _max_run(np.zeros(5, dtype=bool)) == 0


def _l3z(n=10):
    return pl.DataFrame({
        "recipe_id": ["r"] * n, "image_id": ["img"] * n,
        "category_id": ["A"] * n, "cd_index": list(range(n)),
        "cd_nm": [30.0] * n,
        "z_delta_s": [0.5] * n,
        "z_dstep_s": [0.5] * n,
    })


def test_score_l3_flag_and_reason(cfg):
    df = _l3z()
    df[5, "z_delta_s"] = 6.0
    out = score_l3(df, cfg, t_soft=3.0)
    row = out.row(5, named=True)
    assert row["flag"] and row["top_feature"] == "delta_s"
    assert row["reason_code"] == "POSITION_MISMATCH"
    assert out["flag"].sum() == 1


def test_impact_and_g1(cfg):
    df = _l3z(n=10)
    # CD 5개를 크게 왜곡 + 플래그
    for i in range(5):
        df[i, "cd_nm"] = 40.0
        df[i, "z_delta_s"] = 8.0
    scored = score_l3(df, cfg, t_soft=3.0)
    l2s = pl.DataFrame({"recipe_id": ["r"], "image_id": ["img"],
                        "category_id": ["A"], "n_cd": [10.0],
                        "cd_median": [32.0], "cd_mad": [1.0],
                        "delta_median_s": [0.0], "delta_median_e": [0.0],
                        "traj_rms_s": [0.1], "traj_rms_e": [0.1],
                        "px_nm": [0.5]})
    l2 = add_l2_runtime(scored, l2s, cfg)
    row = l2.row(0, named=True)
    assert row["frac_flagged"] == pytest.approx(0.5)
    assert row["max_run"] == 5
    # 전체 median(35) vs 클린 median(30) → impact 5nm > tol 1nm
    assert row["impact_nm"] == pytest.approx(5.0)
    l2sc = score_l2(l2.with_columns(pl.lit(0.0).alias("z_n_cd"),
                                    pl.lit(0.0).alias("z_cd_median"),
                                    pl.lit(0.0).alias("z_delta_median_s"),
                                    pl.lit(0.0).alias("z_delta_median_e")),
                    cfg, t_seq=3.0)
    assert l2sc.row(0, named=True)["g1_hit"]
    assert l2sc.row(0, named=True)["flag_pattern"] == "IMAGE_BAD_LOCAL"


def test_g2_and_image_verdict(cfg):
    l2z = pl.DataFrame({
        "recipe_id": ["r"], "image_id": ["img"], "category_id": ["A"],
        "impact_nm": [0.0], "max_run": [0],
        "z_n_cd": [0.0], "z_cd_median": [0.0],
        "z_delta_median_s": [9.0], "z_delta_median_e": [0.0],
    })
    l2sc = score_l2(l2z, cfg, t_seq=3.0)
    assert l2sc.row(0, named=True)["g2_hit"]

    l1z = pl.DataFrame({
        "recipe_id": ["r"], "image_id": ["img"],
        "z_noise_sigma": [1.0], "z_sat_lo": [0.0], "z_sat_hi": [0.0],
        "z_dyn_range": [0.0], "z_hist_emd": [0.0],
        "z_struct_energy": [0.0], "z_tile_energy_cv": [0.0],
    })
    imgs = score_images(l1z, l2sc, cfg, t_image=3.0)
    row = imgs.row(0, named=True)
    assert row["image_fail"] and not row["g0_hit"] and row["g2_hit"]
    assert "SYSTEMATIC_SHIFT" in row["image_reasons"]
    assert row["n_bad_categories"] == 1


def test_g0_disabled_by_gate_config(cfg):
    cfg2 = cfg.with_overrides({"gates": {"enable_g0": False}})
    l2sc = score_l2(pl.DataFrame({
        "recipe_id": ["r"], "image_id": ["img"], "category_id": ["A"],
        "impact_nm": [0.0], "max_run": [0], "z_n_cd": [0.0],
        "z_cd_median": [0.0], "z_delta_median_s": [0.0],
        "z_delta_median_e": [0.0]}), cfg2, t_seq=3.0)
    l1z = pl.DataFrame({
        "recipe_id": ["r"], "image_id": ["img"], "z_noise_sigma": [99.0],
        "z_sat_lo": [0.0], "z_sat_hi": [0.0], "z_dyn_range": [0.0],
        "z_hist_emd": [0.0], "z_struct_energy": [0.0],
        "z_tile_energy_cv": [0.0]})
    imgs = score_images(l1z, l2sc, cfg2, t_image=3.0)
    assert not imgs.row(0, named=True)["image_fail"]
