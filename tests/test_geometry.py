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


def test_theil_sen_robust_linear():
    v = np.arange(20, dtype=float) * 2.0
    v[9] += 10.0
    r = geo.local_residual_1d(v, 9, "robust_linear", 5)
    assert abs(r[9] - 10.0) < 1.0
    assert np.nanmedian(np.abs(np.delete(r, 9))) < 0.5
