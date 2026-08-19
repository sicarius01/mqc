"""접선 추정, 국소 프레임, 시퀀스 잔차, 곡률, obliquity.

한 시퀀스 = 한 (image, category)의 CD들을 cd_index 순으로 늘어놓은 것.
S/E 궤적은 DL 컨투어의 샘플이므로 매끄러워야 하고 (spec §2.2), 여기의 잔차는
전부 **국소 추세** 대비다 — 전역 중앙값을 쓰면 실제 테이퍼에서 전부 플래그된다.

이 모듈은 px 단위로만 계산한다. nm 변환은 호출부(features/l3.py)가 한다.
"""

from __future__ import annotations

import numpy as np


def unit_tangents(pts: np.ndarray) -> np.ndarray:
    """궤적 점들의 단위 접선. tangent(i) = normalize(P[i+1] − P[i−1]), 끝은 단측.

    pts: (n, 2). n==1이면 (1,0)을 반환 (방향 정의 불가 폴백).
    """
    n = len(pts)
    if n == 1:
        return np.array([[1.0, 0.0]])
    d = np.empty_like(pts, dtype=np.float64)
    d[1:-1] = pts[2:] - pts[:-2]
    d[0] = pts[1] - pts[0]
    d[-1] = pts[-1] - pts[-2]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return d / norm


def _window_slice(i: int, n: int, half: int) -> slice:
    return slice(max(0, i - half), min(n, i + half + 1))


def _local_fit_1d(v: np.ndarray, window: int, method: str) -> np.ndarray:
    """1차원 시퀀스의 국소 추세 적합값.

    hampel: 창 내 중앙값. robust_linear: 창 내 Theil–Sen 직선의 중심 적합값.
    """
    n = len(v)
    half = window // 2
    fit = np.empty(n)
    for i in range(n):
        sl = _window_slice(i, n, half)
        w = v[sl]
        if method == "hampel" or len(w) < 3:
            fit[i] = np.nanmedian(w)
        else:
            t = np.arange(sl.start, sl.stop, dtype=np.float64)
            ok = ~np.isnan(w)
            t, w2 = t[ok], w[ok]
            if len(w2) < 3:
                fit[i] = np.nanmedian(w)
                continue
            # Theil–Sen: 기울기 = 쌍별 기울기의 중앙값
            dt = t[:, None] - t[None, :]
            dv = w2[:, None] - w2[None, :]
            iu = np.triu_indices(len(t), k=1)
            slope = np.median(dv[iu] / dt[iu])
            intercept = np.median(w2 - slope * t)
            fit[i] = slope * i + intercept
    return fit


def local_residual_1d(v: np.ndarray, window: int, method: str,
                      min_len: int) -> np.ndarray:
    """스칼라 시퀀스의 국소 추세 대비 잔차. n < min_len이면 전부 NaN."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < min_len:
        return np.full(len(v), np.nan)
    return v - _local_fit_1d(v, window, method)


def normal_residual(pts: np.ndarray, window: int, method: str,
                    min_len: int) -> np.ndarray:
    """궤적 점의 국소 적합 대비 **법선 방향** 잔차 (px, 부호 있음).

    좌표별로 국소 추세를 적합하고, 잔차 벡터를 접선의 법선에 사영한다.
    """
    n = len(pts)
    if n < min_len:
        return np.full(n, np.nan)
    fx = _local_fit_1d(pts[:, 0], window, method)
    fy = _local_fit_1d(pts[:, 1], window, method)
    res = pts - np.stack([fx, fy], axis=1)
    t = unit_tangents(pts)
    normal = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return np.einsum("ij,ij->i", res, normal)


def _robust_tangents(pts: np.ndarray, win: int = 7) -> np.ndarray:
    """점프에 강건한 단위 접선 — 3점 접선의 성분별 롤링 메디안 후 재정규화.

    점 하나의 점프는 중앙차분 접선 3개(i−1, i, i+1)를 오염시키므로, 창 7의
    메디안이면 오염 표가 소수라 접선이 흔들리지 않는다.
    """
    t = unit_tangents(pts)
    n = len(t)
    if n < 3:
        return t
    half = win // 2
    sm = np.empty_like(t)
    for i in range(n):
        sl = _window_slice(i, n, half)
        sm[i, 0] = np.median(t[sl, 0])
        sm[i, 1] = np.median(t[sl, 1])
    norm = np.linalg.norm(sm, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return sm / norm


def step_normal(pts: np.ndarray) -> np.ndarray:
    """이웃 간 1차 차분의 **법선 성분** 크기 (px).

    유클리드 차분 크기는 측정 피치(슬라이딩 간격)를 재는 것이라 피치 변동이
    전부 플래그된다 — 점프는 궤적 진행 방향이 아니라 법선 방향 이탈이므로
    차분 벡터를 궤적 법선에 사영한다. 법선은 강건 접선(_robust_tangents)에서
    얻는다 — 점프 자신이 접선을 기울여 사영을 감쇠시키는 것을 막는다.

    내부 점은 앞뒤 차분 중 **작은 쪽** — 점프한 점 자신은 양쪽 다 크고,
    그 이웃은 한쪽만 크므로 min이 점프 지점을 고립시킨다. 끝점은 단측.
    """
    n = len(pts)
    if n < 2:
        return np.full(n, np.nan)
    d = np.diff(pts, axis=0)                          # (n-1, 2)
    t = _robust_tangents(pts)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    dn = np.abs(np.einsum("ij,ij->i", d, 0.5 * (nrm[:-1] + nrm[1:])))
    out = np.empty(n)
    out[0] = dn[0]
    out[-1] = dn[-1]
    if n > 2:
        out[1:-1] = np.minimum(dn[:-1], dn[1:])
    return out


def curvature3(pts: np.ndarray) -> np.ndarray:
    """3점 국소 곡률 대리: 2차 차분 벡터의 크기 (px). 끝점은 NaN."""
    n = len(pts)
    out = np.full(n, np.nan)
    if n >= 3:
        second = pts[:-2] - 2 * pts[1:-1] + pts[2:]
        out[1:-1] = np.linalg.norm(second, axis=1)
    return out


def mean_edge_tangent(tan_s: np.ndarray, tan_e: np.ndarray) -> np.ndarray:
    """S/E 접선의 평균 방향 (부호 정렬 후). 리본 오프셋 방향으로 쓴다."""
    sign = np.sign(np.einsum("ij,ij->i", tan_s, tan_e))
    sign[sign == 0] = 1.0
    m = tan_s + tan_e * sign[:, None]
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return m / norm


def obliquity_deg(seg_dir: np.ndarray, edge_tangent: np.ndarray) -> np.ndarray:
    """세그먼트 방향과 엣지 접선이 이루는 각(0–90°)의 시퀀스 중앙값 대비 편차.

    정상이면 세그먼트 ⊥ 엣지 (≈90°). 시퀀스 중앙값을 기준으로 잡아
    레시피 고유의 기울어짐은 상쇄한다.
    """
    dot = np.abs(np.einsum("ij,ij->i", seg_dir, edge_tangent))
    dot = np.clip(dot, 0.0, 1.0)
    theta = np.degrees(np.arcsin(dot))  # 접선과의 각이 아니라 수직에서 벗어난 각
    med = np.nanmedian(theta)
    return np.abs(theta - med)


def segment_angle_deg(seg: np.ndarray) -> np.ndarray:
    """세그먼트 절대 각도 (deg, [-180, 180))."""
    return np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
