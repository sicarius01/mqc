import numpy as np
import pytest

from cdqc import geometry as geo


def _line_traj(n=20, slope=0.1):
    t = np.arange(n, dtype=float)
    return np.stack([100 + slope * t, 10 * t], axis=1)


def test_unit_tangents_direction():
    pts = _line_traj()
    t = geo.unit_tangents(pts)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0)
    assert np.all(t[:, 1] > 0.9)  # 주로 +y 방향


def test_normal_residual_straight_line_zero():
    pts = _line_traj()
    r = geo.normal_residual(pts, window=9, method="hampel", min_len=5)
    assert np.nanmax(np.abs(r)) < 1e-6


def test_normal_residual_detects_jump():
    pts = _line_traj()
    pts[10, 0] += 4.0  # 법선(x) 방향 점프
    r = geo.normal_residual(pts, window=9, method="hampel", min_len=5)
    assert abs(r[10]) > 3.0
    others = np.delete(r, 10)
    assert np.nanmedian(np.abs(others)) < 0.5


def test_local_residual_short_seq_nan():
    r = geo.local_residual_1d(np.arange(3, dtype=float), 9, "hampel", 5)
    assert np.all(np.isnan(r))


def test_step_normal_isolates_jump():
    pts = _line_traj()
    pts[7, 0] += 5.0                 # 법선(x) 방향 점프
    d = geo.step_normal(pts)
    assert d[7] > 4.0                # 점프 지점: 양쪽 차분 다 법선 성분 큼
    assert d[6] < 0.5 and d[8] < 0.5  # 이웃: min이 점프를 고립


def test_step_normal_ignores_pitch_change():
    # 진행 방향(피치) 변동은 법선 성분이 아니므로 반응하면 안 됨
    y = np.array([0, 10, 20, 35, 45, 55, 70, 80, 90, 100], dtype=float)
    pts = np.stack([np.full(len(y), 50.0), y], axis=1)
    d = geo.step_normal(pts)
    assert np.nanmax(d) < 0.1


def test_curvature_flags_kink():
    pts = _line_traj()
    pts[5, 0] += 3.0
    c = geo.curvature3(pts)
    assert np.isnan(c[0]) and np.isnan(c[-1])
    assert c[5] > 3.0


def test_obliquity_perpendicular_zero():
    seg = np.tile([1.0, 0.0], (10, 1))       # 수평 세그먼트
    tangent = np.tile([0.0, 1.0], (10, 1))   # 수직 엣지
    ob = geo.obliquity_deg(seg, tangent)
    assert np.allclose(ob, 0.0, atol=1e-9)


def test_obliquity_detects_rotated_subset():
    seg = np.tile([1.0, 0.0], (11, 1))
    seg[5] = [np.cos(np.radians(20)), np.sin(np.radians(20))]
    tangent = np.tile([0.0, 1.0], (11, 1))
    ob = geo.obliquity_deg(seg, tangent)
    assert abs(ob[5] - 20.0) < 0.5
    assert np.median(np.delete(ob, 5)) < 0.5


def test_circular_median_wraps_at_90():
    # 단순 median이면 (89-89+88)/3 ≈ 29로 깨지는 케이스 — ±90° 랩 경계
    m = geo.circular_median_deg180(np.array([89.0, -89.0, 88.0]))
    dev = (m - 89.0 + 90.0) % 180.0 - 90.0      # 89°와의 랩 거리
    assert abs(dev) < 2.0
    spread = geo.circular_mad_deg180(np.array([89.0, -89.0, 88.0]))
    assert spread < 2.0                          # 실제로는 서로 2° 안


def test_circular_median_plain_case():
    assert geo.circular_median_deg180(np.array([10.0, 12.0, 14.0])) == \
        pytest.approx(12.0, abs=1e-6)
    assert geo.circular_mad_deg180(np.array([10.0, 12.0, 14.0])) == \
        pytest.approx(2.0, abs=1e-6)


def test_theil_sen_robust_linear():
    v = np.arange(20, dtype=float) * 2.0
    v[9] += 10.0
    r = geo.local_residual_1d(v, 9, "robust_linear", 5)
    assert abs(r[9] - 10.0) < 1.0
    assert np.nanmedian(np.abs(np.delete(r, 9))) < 0.5


def test_circular_residual_at_wrap_boundary():
    """±90° 랩 경계 — 89.9°와 -89.9°는 0.2° 차이지 179.8° 차이가 아니다."""
    a = np.array([89.9, -89.9, 89.5, -89.7])
    r = geo.circular_residual_deg180(a)
    assert np.abs(r).max() < 1.0
    assert np.all(np.abs(r) <= 90.0)


def test_circular_residual_mixes_zero_and_179():
    """0°와 179.9°는 같은 방향이다 (180° 주기) — 잔차가 0.1° 수준."""
    a = np.array([0.0, 179.9, 0.1, 179.8, 0.0])
    r = geo.circular_residual_deg180(a)
    assert np.abs(r).max() < 0.5


def test_circular_residual_isolates_a_rotated_member():
    a = np.array([10.0, 11.0, 10.5, 40.0, 10.2, 10.8, 11.1])
    r = geo.circular_residual_deg180(a)
    assert abs(r[3] - 29.4) < 1.0
    assert np.max(np.abs(np.delete(r, 3))) < 1.5


def test_circular_residual_zero_when_whole_sequence_rotates():
    """시퀀스가 통째로 돌면 중심도 같이 돈다 → 잔차 0.

    총체적 회전은 L2 angle_median이 코호트 대비로 잡을 실패지, CD 레벨
    잔차가 잡을 실패가 아니다.
    """
    a = np.array([10.0, 11.0, 9.5, 10.5, 10.2])
    r0 = geo.circular_residual_deg180(a)
    r1 = geo.circular_residual_deg180(a + 35.0)
    # 원형 중앙값은 atan2(median sin, median cos)이라 회전 등가성이 수치
    # 오차 수준(1e-5 deg)까지만 성립한다 — 물리적으로는 0
    assert np.allclose(r0, r1, atol=1e-3)


def test_circular_residual_explicit_center():
    r = geo.circular_residual_deg180(np.array([5.0]), center=0.0)
    assert r[0] == pytest.approx(5.0)
    assert np.isnan(geo.circular_residual_deg180(np.array([np.nan, np.nan]))).all()
