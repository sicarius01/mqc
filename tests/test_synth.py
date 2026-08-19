import numpy as np
import polars as pl
import pytest

from cdqc.synth.generator import generate_dataset


@pytest.fixture
def tiny_cfg(cfg):
    return cfg.with_overrides({
        "synthetic": {"n_images": 2, "image_size": [256, 256],
                      "cds_per_category": 10, "n_categories": 2,
                      "inject": {"edge_jump_nm": [0, 2],
                                 "missing_frac": [0, 0.2]}},
        "selftest": {"n_images_per_case": 1},
    })


def test_generate_deterministic(tiny_cfg, tmp_path):
    df1 = generate_dataset(tiny_cfg, tmp_path / "a")
    df2 = generate_dataset(tiny_cfg, tmp_path / "b")
    assert df1.equals(df2)


def test_edge_jump_moves_only_affected_sx(tiny_cfg, tmp_path):
    df = generate_dataset(tiny_cfg, tmp_path / "s")
    base = df.filter(pl.col("injected_failure") == "edge_jump",
                     pl.col("sev_rank") == 0)
    jump = df.filter(pl.col("injected_failure") == "edge_jump",
                     pl.col("sev_rank") == 1)
    aff = jump.filter(pl.col("affected") == 1)
    assert aff.height > 0
    # 2nm / 0.5nmpx = 4px 이동 (지터 ±0.2px)
    j = aff.join(base.filter(pl.col("affected") == 1),
                 on=["category_id", "cd_index"], suffix="_b")
    if j.height:
        d = (j["sx"] - j["sx_b"]).to_numpy()
        assert np.all(d > 3.0)


def test_missing_drops_rows(tiny_cfg, tmp_path):
    df = generate_dataset(tiny_cfg, tmp_path / "m")
    n0 = df.filter(pl.col("injected_failure") == "missing",
                   pl.col("sev_rank") == 0).height
    n1 = df.filter(pl.col("injected_failure") == "missing",
                   pl.col("sev_rank") == 1).height
    assert n1 < n0


def test_images_written(tiny_cfg, tmp_path):
    df = generate_dataset(tiny_cfg, tmp_path / "img")
    import cv2
    for p in (tmp_path / "img" / "images").glob("*.png"):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert im is not None and im.shape == (256, 256)
        break
    # config 오버라이드는 병합 의미론 — inject 기본 키들도 함께 생성된다
    assert {"none", "edge_jump", "missing"} <= set(df["injected_failure"].unique())
