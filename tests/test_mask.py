import numpy as np
import pytest

import cdqc
from tests.test_evidence import make_band_image, seq_coords


def rect_mask(h=100, w=100, x0=30, x1=60, y0=10, y1=90):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1 + 1] = True
    return m


def test_mdist_hand_computed(params):
    """알려진 사각형 마스크에서 mdist 손계산 일치."""
    img = np.full((100, 100), 128, dtype=np.uint8)
    m = rect_mask()                      # 좌경계 x=30, 우경계 x=60
    S = np.array([[30.0, 50.0], [27.0, 50.0], [35.0, 50.0]])
    E = np.array([[60.0, 50.0], [60.0, 50.0], [60.0, 50.0]])
    f = cdqc.extract_mask_l3(m, img, S, E, px_nm=0.5, params=params)
    assert f["mdist_s"].iloc[0] == pytest.approx(0.0, abs=1e-9)   # 경계 위
    assert f["mdist_s"].iloc[1] == pytest.approx(3.0 * 0.5)       # 3px 밖
    assert f["mdist_s"].iloc[2] == pytest.approx(5.0 * 0.5)       # 5px 안 (31~60 내부 경계 기준... 35-30=5)
    assert f["mdist_e"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert f["minside"].all()            # 중점 (45,50) 등은 전부 내부


def test_minside_false_outside(params):
    img = np.full((100, 100), 128, dtype=np.uint8)
    m = rect_mask()
    S = np.array([[70.0, 50.0]])
    E = np.array([[90.0, 50.0]])         # 세그먼트가 마스크 밖
    f = cdqc.extract_mask_l3(m, img, S, E, 0.5, params=params)
    assert not f["minside"].iloc[0]


def test_mgrad_high_on_real_edge_low_on_plateau(params):
    """마스크 경계가 실제 명암 전이 위면 mgrad 큼, 플래토 위로 밀리면 작음."""
    img = make_band_image()              # 엣지 x=80, 130
    S, E = seq_coords()
    m_good = np.zeros(img.shape, dtype=bool)
    m_good[:, 80:131] = True             # 경계가 실제 엣지 위
    m_bad = np.zeros(img.shape, dtype=bool)
    m_bad[:, 90:121] = True              # 경계가 플래토 위
    fg = cdqc.extract_mask_l3(m_good, img, S, E, 0.5, params=params)
    fb = cdqc.extract_mask_l3(m_bad, img, S, E, 0.5, params=params)
    assert np.nanmedian(fg["mgrad_s"]) > 3 * np.nanmedian(fb["mgrad_s"])
    # 좌표는 그대로인데 마스크가 10px 밀렸으니 mdist ≈ 10px*0.5nm
    assert np.nanmedian(fb["mdist_s"]) == pytest.approx(5.0, abs=0.6)


def test_mask_shape_mismatch_rejected(params):
    img = np.full((100, 100), 128, dtype=np.uint8)
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.extract_mask_l3(np.zeros((50, 50), dtype=bool), img,
                             np.zeros((3, 2)), np.ones((3, 2)), 0.5)
    assert e.value.code == "E-ARG-07"


def test_mask_image_features_shapes(params):
    img = np.full((100, 100), 128, dtype=np.uint8)
    m = rect_mask()
    f = cdqc.extract_mask_image(m, img, params)
    assert f["mask_n_components"] == 1.0
    assert f["mask_hole_frac"] == 0.0
    assert f["mask_area_frac"] == pytest.approx(m.mean())
    assert f["mask_boundary_rough"] > 1.0       # 직사각형 > 원

    m2 = m.copy()
    m2[40:45, 40:45] = False                    # 내부 구멍
    m2[5, 5] = True                             # 고립 잡티
    f2 = cdqc.extract_mask_image(m2, img, params)
    assert f2["mask_n_components"] == 2.0
    assert f2["mask_hole_frac"] > 0.0


def test_empty_mask_degrades_gracefully(params):
    img = np.full((100, 100), 128, dtype=np.uint8)
    m = np.zeros((100, 100), dtype=bool)
    f = cdqc.extract_mask_l3(m, img, np.array([[10.0, 10.0]]),
                             np.array([[20.0, 10.0]]), 0.5, params=params)
    assert np.isnan(f["mdist_s"].iloc[0])
    assert not f["minside"].iloc[0]
    fi = cdqc.extract_mask_image(m, img, params)
    assert fi["mask_n_components"] == 0.0
