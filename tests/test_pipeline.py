import numpy as np
import polars as pl
import pytest

from cdqc import io
from cdqc.config import load_config
from cdqc.pipeline import extract_features
from cdqc.synth.generator import generate_dataset


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """작은 합성 데이터셋 + 그걸 가리키는 Config (역산·단위 경로 포함)."""
    tmp = tmp_path_factory.mktemp("pipe")
    root = tmp / "synth"
    cfg = load_config(None, root=str(tmp)).with_overrides({
        "synthetic": {"n_images": 2, "image_size": [256, 256],
                      "cds_per_category": 8, "n_categories": 2},
        "selftest": {"n_images_per_case": 1},
        "paths": {"data_dir": str(root), "image_dir": str(root),
                  "cache_dir": str(tmp / "cache")},
        "data": {"coords": {"convention": "xy"}, "columns": {"unit": "unit"}},
        "cohort": {"min_cohort_n": 10},
    })
    generate_dataset(cfg, root)
    return cfg


def test_px_nm_inversion_on_synth(tiny):
    """지터 있는 합성에서도 이미지별 역산 px_nm이 참값(0.5) 근처.

    정상 이미지 기준 — systematic_bias처럼 모든 행이 균일하게 밀린 이미지는
    역산이 편향을 흡수한다 (본질적 성질. 탐지는 delta_median/G2 몫).
    """
    rec = io.load_records(tiny)
    base = rec.filter(pl.col("image_id").str.starts_with("none_"))
    px = base.group_by("image_id").agg(pl.col("px_nm").first())["px_nm"].to_numpy()
    assert np.all(np.abs(px - 0.5) < 0.01)


def test_value_mismatch_reacts_to_tampering(tiny):
    rec = io.load_records(tiny)
    l3, _, _ = extract_features(tiny, records=rec, force=True)
    base = l3.filter(pl.col("image_id").str.starts_with("none_"))
    # 정상: 보고값과 기하 길이가 지터 수준(≈0.2nm)에서 일치
    assert np.nanmedian(np.abs(base["value_mismatch_nm"].to_numpy())) < 0.5

    img = base[0, "image_id"]
    tampered = rec.with_columns(
        pl.when((pl.col("image_id") == img) & (pl.col("category_id") == "A")
                & (pl.col("cd_index") == 3))
        .then(pl.col("ex") + 8.0).otherwise(pl.col("ex")).alias("ex"))
    l3t, _, _ = extract_features(tiny, records=tampered, force=True)
    row = l3t.filter((pl.col("image_id") == img) & (pl.col("category_id") == "A")
                     & (pl.col("cd_index") == 3))
    # 8px × 0.5nm = 4nm 어긋남 → mismatch가 크게 반응
    assert abs(row[0, "value_mismatch_nm"]) > 2.0
