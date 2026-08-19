import numpy as np
import pytest
from scipy.special import erf

from cdqc.features.l3 import l3_sequence_features


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


@pytest.fixture
def base_cfg(cfg):
    return cfg


def test_evidence_on_clean_edge(base_cfg):
    img = make_band_image()
    S, E = seq_coords()
    f = l3_sequence_features(img, S, E, px_nm=0.5, cfg=base_cfg)
    # 보고 좌표가 참 엣지 위 → delta ~ 0
    assert np.nanmedian(np.abs(f["delta_s"])) < 0.15
    assert np.nanmedian(np.abs(f["delta_e"])) < 0.15
    # 선명한 엣지 → cnr 큼, 극성: S는 상승(+), E는 하강(-)
    assert np.nanmedian(f["cnr_s"]) > 5
    assert np.all(f["pol_s"][np.isfinite(f["pol_s"])] > 0)
    assert np.all(f["pol_e"][np.isfinite(f["pol_e"])] < 0)
    assert np.all(f["edge_valid_s"])
    # rise ~ 1.5px * 2.563 * 0.5nm ≈ 1.9nm
    assert 1.0 < np.nanmedian(f["rise_s"]) < 3.5
    assert np.nanmedian(f["plateau_cv"]) < 0.1


def test_delta_tracks_reported_shift(base_cfg):
    img = make_band_image()
    S, E = seq_coords()
    S[:, 0] += 4.0   # 보고 S를 밴드 안쪽(+x)으로 4px 이동 → 참 엣지는 s=-4
    f = l3_sequence_features(img, S, E, px_nm=0.5, cfg=base_cfg)
    assert np.nanmedian(f["delta_s"]) < -1.5   # 부호: -4px * 0.5nm = -2nm
    assert abs(np.nanmedian(f["delta_e"])) < 0.2  # E는 영향 없음


def test_rise_grows_with_blur(base_cfg):
    sharp = make_band_image(rise=1.5)
    blurred = make_band_image(rise=5.0)
    S, E = seq_coords()
    f1 = l3_sequence_features(sharp, S, E, 0.5, base_cfg)
    f2 = l3_sequence_features(blurred, S, E, 0.5, base_cfg)
    assert np.nanmedian(f2["rise_s"]) > 2 * np.nanmedian(f1["rise_s"])


def test_cnr_drops_with_contrast(base_cfg):
    hi = make_band_image(contrast=60)
    lo = make_band_image(contrast=15)
    S, E = seq_coords()
    f1 = l3_sequence_features(hi, S, E, 0.5, base_cfg)
    f2 = l3_sequence_features(lo, S, E, 0.5, base_cfg)
    assert np.nanmedian(f2["cnr_s"]) < 0.5 * np.nanmedian(f1["cnr_s"])


def test_geometry_only_without_image(base_cfg):
    S, E = seq_coords()
    f = l3_sequence_features(None, S, E, 0.5, base_cfg)
    assert np.all(np.isnan(f["delta_s"]))
    assert np.isfinite(f["cd_nm"]).all()
    assert abs(f["cd_nm"][0] - 25.0) < 1e-6   # 50px * 0.5nm
