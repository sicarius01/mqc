import numpy as np
import pytest
from scipy.special import erf

import cdqc
from cdqc.features.mask import boundary_pixels


def rect_labelmap(h=100, w=100, x0=30, x1=60, bg=10, fg=50):
    """세로 띠 하나짜리 라벨맵. 라벨 전이는 열 29|30과 60|61 사이."""
    lab = np.full((h, w), bg, dtype=np.int32)
    lab[:, x0:x1 + 1] = fg
    return lab


def step_image(h=160, w=160, x_edge=80.0, rise=2.0, contrast=60.0,
               noise=3.0, seed=0):
    """수직 계단 엣지 하나 (erf 프로파일). noise=0이면 완전 무잡음."""
    X = np.arange(w, dtype=float)[None, :]
    prof = 90.0 + contrast * 0.5 * (1 + erf((X - x_edge) / (rise * np.sqrt(2))))
    img = np.broadcast_to(prof, (h, w)).astype(np.float64)
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0, noise, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- mask_maps

def test_mask_maps_boundary_is_union_of_all_transitions():
    lab = np.full((20, 20), 10, dtype=np.int32)
    lab[:, 5:10] = 30
    lab[:, 10:15] = 50                       # 30|50 전이도 경계여야 한다
    m = cdqc.mask_maps(lab)
    b = m["boundary"]
    assert b[10, 4] and b[10, 5]              # 10|30 전이 양쪽
    assert b[10, 9] and b[10, 10]             # 30|50 전이 양쪽 (클래스 선택 없음)
    assert b[10, 14] and b[10, 15]            # 50|10 전이 양쪽
    assert not b[10, 7] and not b[10, 12]     # 층 내부는 경계 아님
    assert not b[0, 0]                        # 이미지 프레임은 전이가 아님


def test_mask_maps_rejects_non_2d():
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.mask_maps(np.zeros((4, 4, 3), dtype=np.int32))
    assert e.value.code == "E-ARG-02"


# ------------------------------------------------------- boundary_features

def test_bdist_hand_computed(params):
    """알려진 사각형 라벨맵에서 bdist 손계산 일치.

    경계 픽셀은 {29, 30, 60, 61}열. 거리는 가장 가까운 경계 픽셀 중심까지의
    유클리드 거리 × px_nm (참 계면은 픽셀 중심 사이라 ±0.5px 양자화가 남는다).
    """
    maps = cdqc.mask_maps(rect_labelmap())
    S = np.array([[30.0, 50.0], [27.0, 50.0], [45.0, 50.0]])
    E = np.array([[60.0, 50.0], [60.0, 50.0], [46.0, 50.0]])
    f = cdqc.boundary_features(maps, S, E, px_nm=0.5, params=params)

    assert f["bdist_s"].iloc[0] == pytest.approx(0.0)        # 경계 픽셀 위
    assert f["bdist_s"].iloc[1] == pytest.approx(2.0 * 0.5)  # 27 → 29열까지 2px
    assert f["bdist_s"].iloc[2] == pytest.approx(15.0 * 0.5)  # 45 → 30열까지 15px
    assert f["bdist_e"].iloc[0] == pytest.approx(0.0)
    # 중점 (45, 50)은 층 한가운데 → 30열/60열 중 가까운 쪽 15px
    assert f["bdist_mid"].iloc[0] == pytest.approx(15.0 * 0.5)


def test_bdist_ratio_separates_interface_from_void_line(params):
    """정상 CD(계면 가로지름) ≈ 0 vs 허공에 그은 선(층 안에 떠 있음) ≈ 1.

    허공선은 층을 가로지르지 않고 층 **안을 따라** 그어진 선이라 양 끝과
    중점이 경계에서 같은 거리에 있다 → 비가 1. 층을 가로지르는 정상 CD는
    양 끝이 경계 위(0)고 중점만 층 한가운데라 비가 0에 붙는다.
    """
    maps = cdqc.mask_maps(rect_labelmap())
    S = np.array([[30.0, 50.0], [45.0, 30.0]])
    E = np.array([[60.0, 50.0], [45.0, 70.0]])   # 두 번째는 층 안을 세로로
    f = cdqc.boundary_features(maps, S, E, 0.5, params=params)
    assert f["bdist_ratio"].iloc[0] < 0.05
    assert f["bdist_ratio"].iloc[1] == pytest.approx(1.0, abs=0.05)
    # 배율에 무관 (비율이므로 px_nm을 바꿔도 같다)
    g = cdqc.boundary_features(maps, S, E, 1.7, params=params)
    assert np.allclose(f["bdist_ratio"], g["bdist_ratio"])


def test_label_runs_counts_crossings(params):
    maps = cdqc.mask_maps(rect_labelmap())
    S = np.array([[20.0, 50.0], [40.0, 50.0]])
    E = np.array([[70.0, 50.0], [50.0, 50.0]])   # 첫 CD는 층을 통과 (10-50-10)
    f = cdqc.boundary_features(maps, S, E, 0.5, params=params)
    assert f["label_runs"].iloc[0] == 3
    assert f["label_runs"].iloc[1] == 1          # 층 안에만 있음


def test_bdist_measures_both_ends_on_different_interfaces(params):
    """S와 E가 서로 다른 계면 위여도 둘 다 측정된다 (클래스 선택이 없으므로)."""
    lab = np.full((60, 120), 10, dtype=np.int32)
    lab[:, 20:50] = 30
    lab[:, 50:80] = 50
    maps = cdqc.mask_maps(lab)
    S = np.array([[20.0, 30.0]])        # 10|30 계면
    E = np.array([[80.0, 30.0]])        # 50|10 계면
    f = cdqc.boundary_features(maps, S, E, 0.5, params=params)
    assert f["bdist_s"].iloc[0] == pytest.approx(0.0)
    assert f["bdist_e"].iloc[0] == pytest.approx(0.0)


def test_boundary_features_validates_inputs(params):
    maps = cdqc.mask_maps(rect_labelmap())
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.boundary_features({"nope": 1}, np.zeros((1, 2)), np.ones((1, 2)), 0.5)
    assert e.value.code == "E-ARG-02"
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.boundary_features(maps, np.zeros((1, 2)), np.ones((1, 2)), 0.0)
    assert e.value.code == "E-ARG-06"


def test_extract_l2_carries_bdist_medians(params):
    import pandas as pd
    l3 = pd.DataFrame({"cd_nm": [30.0] * 4,
                       "bdist_s": [0.0, 0.1, 0.2, 8.0],
                       "bdist_e": [0.0, 0.0, 0.0, 0.0]})
    l2 = cdqc.extract_l2(l3, None, params)
    assert l2["bdist_median_s"].iloc[0] == pytest.approx(0.15)
    assert l2["bdist_median_e"].iloc[0] == pytest.approx(0.0)


# ------------------------------------------------- mask_grad_agree 회귀 (§E)

def test_boundary_pixels_excludes_image_frame():
    m = np.zeros((20, 20), dtype=bool)
    m[:, 10:] = True                     # 오른쪽 절반 — 위/아래/오른쪽이 프레임에 닿음
    b = boundary_pixels(m)
    assert b[:, 10].all()                # 실제 경계 열은 전부 경계
    assert not b[0, 15] and not b[19, 15]   # 프레임에 닿은 자리는 경계 아님
    assert not b[10, 19]


def test_mask_grad_agree_drops_when_mask_leaves_the_edge():
    """마스크를 0/5/20px 밀면 값이 단조 감소하고 0↔5 비가 3배 이상.

    이전에는 분모가 전역 σ라 구조 대비에 물려 0.57/0.38/0.41로 거의 안
    움직였다 (mgrad는 국소 σ를 써서 20배 갈렸다).
    """
    img = step_image()
    vals = []
    for d in (0, 5, 20):
        m = np.zeros(img.shape, dtype=bool)
        m[:, 80 + d:] = True
        vals.append(cdqc.extract_mask_image(m, img)["mask_grad_agree"])
    assert vals[0] > vals[1] > vals[2]
    assert vals[0] / vals[1] >= 3.0


def test_mask_grad_agree_finite_on_noiseless_image():
    """노이즈 없는 이미지에서 σ→0으로 값이 폭주하면 안 된다 (구 코드 1e11)."""
    img = step_image(noise=0.0)
    m = np.zeros(img.shape, dtype=bool)
    m[:, 80:] = True
    v = cdqc.extract_mask_image(m, img)["mask_grad_agree"]
    assert np.isfinite(v) and v < 1e3
