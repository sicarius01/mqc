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


def test_to_uint8_passthrough_and_conversions():
    u8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert cdqc.to_uint8(u8) is u8                        # 복사 없음

    u16 = (np.arange(256, dtype=np.uint16) * 257).reshape(16, 16)
    shifted = cdqc.to_uint8(u16, method="shift")
    assert shifted.dtype == np.uint8
    assert shifted[0, 1] == 1 and shifted[-1, -1] == 255  # 상위 8비트

    f = np.linspace(100.0, 900.0, 256).reshape(16, 16)
    st = cdqc.to_uint8(f)                                 # percentile 스트레치
    assert st.dtype == np.uint8
    assert st.min() == 0 and st.max() == 255

    with pytest.raises(cdqc.CdqcError):
        cdqc.to_uint8(u8.astype(np.float64), method="shift")


def test_convention_scores_picks_truth(params):
    """합성 이미지 + 참 좌표 → xy/zero/no-flip이 1등이어야 한다."""
    from cdqc.synth.generator import SynthParams, generate_dataset
    sp = SynthParams(n_images=3, image_size=(256, 256), cds_per_category=8,
                     n_categories=2, n_images_per_case=0, inject={})
    records, images, _ = generate_dataset(sp)
    items = []
    for iid, g in records.groupby("image_id"):
        items.append((images[iid], g[["sx", "sy"]].to_numpy(),
                      g[["ex", "ey"]].to_numpy()))
    table = cdqc.convention_scores(items)
    best = table.iloc[0]
    assert (best["convention"], best["origin"], best["y_flip"]) == ("xy", "zero", False)
    assert best["median_dist_px"] < 0.75
    assert table.iloc[1]["median_dist_px"] / max(best["median_dist_px"], 0.1) >= 2.0


def _label_bgr(h=60, w=60):
    import cv2
    lab = np.full((h, w), 10, dtype=np.uint8)
    lab[:, 20:41] = 50
    return cv2.cvtColor(lab, cv2.COLOR_GRAY2BGR), lab


def test_close_annotation_restores_labels_under_color_line():
    """컬러 주석이 덮은 라벨을 closing으로 복구 — 원본과 일치해야 한다."""
    bgr, lab = _label_bgr()
    bgr = bgr.copy()
    bgr[:, 29:32] = (0, 0, 255)          # 3px 빨간 CD 선이 라벨을 덮음
    out = cdqc.close_annotation(bgr)
    assert np.array_equal(out[..., 1], lab)
    assert out.shape == bgr.shape and out.dtype == bgr.dtype


def test_close_annotation_leaves_clean_labelmap_alone():
    bgr, lab = _label_bgr()
    assert np.array_equal(cdqc.close_annotation(bgr)[..., 1], lab)


def test_close_annotation_fixes_downstream_boundary_features():
    """주석 오염은 라벨 전이를 가짜로 만든다 — 복구하면 원래 경계로 돌아온다."""
    bgr, lab = _label_bgr()
    dirty = bgr.copy()
    dirty[:, 29:32] = (0, 0, 255)
    n_clean = cdqc.mask_maps(lab)["boundary"].sum()
    n_dirty = cdqc.mask_maps(dirty[..., 1])["boundary"].sum()
    n_fixed = cdqc.mask_maps(cdqc.close_annotation(dirty)[..., 1])["boundary"].sum()
    assert n_dirty > n_clean                 # 가짜 전이가 생겼다
    assert n_fixed == n_clean


def test_close_annotation_rejects_non_bgr():
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.close_annotation(np.zeros((10, 10), dtype=np.uint8))
    assert e.value.code == "E-ARG-02"
