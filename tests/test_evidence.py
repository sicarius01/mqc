import numpy as np
import pytest
from scipy.special import erf

import cdqc


def make_band_image(h=200, w=200, x_left=80.0, x_right=130.0,
                    contrast=60.0, rise=1.5, noise=3.0, seed=0):
    """수직 밴드 하나짜리 합성 이미지 (테스트 전용 최소 모델)."""
    rng = np.random.default_rng(seed)
    X = np.arange(w, dtype=float)[None, :]
    img = 90.0 + contrast * 0.5 * (erf((X - x_left) / (rise * np.sqrt(2)))
                                   - erf((X - x_right) / (rise * np.sqrt(2))))
    img = np.broadcast_to(img, (h, w)).copy()
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def seq_coords(n=10, x_left=80.0, x_right=130.0, h=200):
    ys = np.linspace(30, h - 30, n)
    S = np.stack([np.full(n, x_left), ys], axis=1)
    E = np.stack([np.full(n, x_right), ys], axis=1)
    return S, E


def test_evidence_on_clean_edge(params):
    img = make_band_image()
    S, E = seq_coords()
    f = cdqc.extract_l3(img, S, E, px_nm=0.5, params=params)
    # 보고 좌표가 참 엣지 위 → delta ~ 0
    assert np.nanmedian(np.abs(f["delta_s"])) < 0.15
    assert np.nanmedian(np.abs(f["delta_e"])) < 0.15
    # 선명한 엣지 → cnr 큼, 극성: S는 상승(+), E는 하강(-)
    assert np.nanmedian(f["cnr_s"]) > 5
    assert (f["pol_s"][np.isfinite(f["pol_s"])] > 0).all()
    assert (f["pol_e"][np.isfinite(f["pol_e"])] < 0).all()
    assert f["edge_valid_s"].all()
    # rise ~ 1.5px * 2.563 * 0.5nm ≈ 1.9nm
    assert 1.0 < np.nanmedian(f["rise_s"]) < 3.5
    assert np.nanmedian(f["plateau_cv"]) < 0.1


def test_delta_tracks_reported_shift(params):
    img = make_band_image()
    S, E = seq_coords()
    S[:, 0] += 4.0   # 보고 S를 밴드 안쪽(+x)으로 4px 이동 → 참 엣지는 s=-4
    f = cdqc.extract_l3(img, S, E, px_nm=0.5, params=params)
    assert np.nanmedian(f["delta_s"]) < -1.5   # 부호: -4px * 0.5nm = -2nm
    assert abs(np.nanmedian(f["delta_e"])) < 0.2  # E는 영향 없음


def test_rise_grows_with_blur(params):
    S, E = seq_coords()
    f1 = cdqc.extract_l3(make_band_image(rise=1.5), S, E, 0.5, params=params)
    f2 = cdqc.extract_l3(make_band_image(rise=5.0), S, E, 0.5, params=params)
    assert np.nanmedian(f2["rise_s"]) > 2 * np.nanmedian(f1["rise_s"])


def test_cnr_drops_with_contrast(params):
    S, E = seq_coords()
    f1 = cdqc.extract_l3(make_band_image(contrast=60), S, E, 0.5, params=params)
    f2 = cdqc.extract_l3(make_band_image(contrast=15), S, E, 0.5, params=params)
    assert np.nanmedian(f2["cnr_s"]) < 0.5 * np.nanmedian(f1["cnr_s"])


def test_geometry_only_without_image(params):
    S, E = seq_coords()
    f = cdqc.extract_l3(None, S, E, 0.5, params=params)
    assert f["delta_s"].isna().all()
    assert np.isfinite(f["cd_nm"]).all()
    assert abs(f["cd_nm"].iloc[0] - 25.0) < 1e-6   # 50px * 0.5nm


def test_value_mismatch_column(params):
    img = make_band_image()
    S, E = seq_coords()
    value_nm = np.full(len(S), 25.0)          # 참값과 일치
    f = cdqc.extract_l3(img, S, E, 0.5, value_nm=value_nm, params=params)
    assert np.abs(f["value_mismatch_nm"]).max() < 1e-9
    value_nm2 = value_nm.copy()
    value_nm2[3] += 4.0                        # 보고값만 4nm 어긋남
    f2 = cdqc.extract_l3(img, S, E, 0.5, value_nm=value_nm2, params=params)
    assert abs(f2["value_mismatch_nm"].iloc[3] - 4.0) < 1e-9


def test_input_validation(params):
    S, E = seq_coords()
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.extract_l3(np.zeros((10, 10), dtype=np.float64), S, E, 0.5)
    assert e.value.code == "E-ARG-02"
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.extract_l3(None, S[:, :1], E[:, :1], 0.5)
    assert e.value.code == "E-ARG-01"
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.extract_l3(None, S, E, px_nm=0.0)
    assert e.value.code == "E-ARG-06"
