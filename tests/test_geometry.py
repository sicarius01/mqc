import numpy as np

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


def test_step_magnitude_isolates_jump():
    pts = _line_traj()               # 기본 스텝 ≈ 10 (y 간격)
    pts[7, 0] += 5.0
    d = geo.step_magnitude(pts)
    assert d[7] > 11.0               # 점프 지점: 양쪽 다 커짐 → min도 큼
    assert abs(d[6] - 10.0) < 0.2    # 이웃: 한쪽만 큼 → min은 기본 스텝
    assert abs(d[8] - 10.0) < 0.2


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


def test_theil_sen_robust_linear():
    v = np.arange(20, dtype=float) * 2.0
    v[9] += 10.0
    r = geo.local_residual_1d(v, 9, "robust_linear", 5)
    assert abs(r[9] - 10.0) < 1.0
    assert np.nanmedian(np.abs(np.delete(r, 9))) < 0.5
