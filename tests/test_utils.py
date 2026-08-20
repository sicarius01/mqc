import numpy as np
import pytest

import cdqc


def test_transform_identity():
    conv = {"convention": "xy", "origin": "zero", "y_flip": False, "scale": 1.0}
    x, y = cdqc.transform_coords(np.array([10.0]), np.array([20.0]), conv, (100, 200))
    assert x[0] == 10.0 and y[0] == 20.0


def test_transform_rowcol_origin_flip():
    conv = {"convention": "rowcol", "origin": "one", "y_flip": True, "scale": 1.0}
    # 원본 (a,b) = (row=21, col=11), origin 1 → row=20, col=10 → y_flip(H=100) → y=79
    x, y = cdqc.transform_coords(np.array([21.0]), np.array([11.0]), conv, (100, 200))
    assert x[0] == 10.0
    assert y[0] == 79.0


def test_transform_scale():
    conv = {"convention": "xy", "origin": "zero", "y_flip": False, "scale": 2.0}
    x, y = cdqc.transform_coords(np.array([5.0]), np.array([7.0]), conv, (100, 100))
    assert x[0] == 10.0 and y[0] == 14.0


def test_normalize_unit_variants():
    # Å: U+00C5, U+212B, NFD(A+U+030A), ASCII, 이름
    for s in ["Å", "Å", "Å", "A", "a", "Angstrom", " ang "]:
        assert cdqc.normalize_unit(s) == "angstrom", repr(s)
    for s in ["nm", "NM", "nanometer", "Nanometre"]:
        assert cdqc.normalize_unit(s) == "nm"
    assert cdqc.normalize_unit("furlong") is None
    assert cdqc.normalize_unit("") is None


def test_to_nm_rowwise_and_scalar():
    v = cdqc.to_nm([250.0, 25.0], ["Å", "nm"])
    assert np.allclose(v, [25.0, 25.0])
    v2 = cdqc.to_nm(np.array([250.0]), "Å")   # ANGSTROM SIGN 스칼라
    assert v2[0] == pytest.approx(25.0)


def test_to_nm_unknown_unit_aborts():
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.to_nm([1.0, 2.0], ["nm", "furlong"])
    assert e.value.code == "E-ARG-03"
    assert "furlong" in str(e.value)


def test_infer_px_nm_exact():
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 50, (10, 2))
    E = S + rng.uniform(20, 60, (10, 2))
    value_nm = np.linalg.norm(E - S, axis=1) * 0.73
    assert cdqc.infer_px_nm(S, E, value_nm) == pytest.approx(0.73, abs=1e-9)


def test_infer_px_nm_no_valid_rows():
    S = np.zeros((3, 2))
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.infer_px_nm(S, S, np.full(3, np.nan))
    assert e.value.code == "E-ARG-04"


def test_ratio_cv_detects_inconsistency():
    rng = np.random.default_rng(1)
    S = rng.uniform(0, 50, (20, 2))
    E = S + rng.uniform(30, 60, (20, 2))
    good = np.linalg.norm(E - S, axis=1) * 0.5
    assert cdqc.ratio_cv(S, E, good) < 1e-9
    bad = good * rng.normal(1.0, 0.05, 20)   # 좌표와 값이 따로 노는 경우
    assert cdqc.ratio_cv(S, E, bad) > 0.01


def test_convention_scores_picks_truth(params):
    """합성 이미지 + 참 좌표 → xy/zero/no-flip이 1등이어야 한다."""
    from cdqc.synth.generator import SynthParams, generate_dataset
    sp = SynthParams(n_images=3, image_size=(256, 256), cds_per_category=8,
                     n_categories=2, n_images_per_case=0, inject={})
    records, images = generate_dataset(sp)
    items = []
    for iid, g in records.groupby("image_id"):
        items.append((images[iid], g[["sx", "sy"]].to_numpy(),
                      g[["ex", "ey"]].to_numpy()))
    table = cdqc.convention_scores(items)
    best = table.iloc[0]
    assert (best["convention"], best["origin"], best["y_flip"]) == ("xy", "zero", False)
    assert best["median_dist_px"] < 0.75
    assert table.iloc[1]["median_dist_px"] / max(best["median_dist_px"], 0.1) >= 2.0
