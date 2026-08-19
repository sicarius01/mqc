"""프로파일 → 이미지 증거 피쳐.

세그먼트 축 프로파일에는 S에서 스텝, 중간 플래토, E에서 스텝이 있어야 정상이다
(spec §2.2). 여기서는 각 엔드포인트 주변 탐색 창에서 그래디언트 피크를 찾아
delta / cnr / rise / margin / npk / overshoot / 극성을 계산한다.

좌표 계약: 프로파일 s축에서 s=0이 보고된 S, s=L이 보고된 E. delta는 두
엔드포인트 모두 **+u(S→E) 방향으로 부호**를 갖는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .sampling import DS, Profile, effective_margin

_EPS = 1e-9
_MARGIN_CAP = 1e3   # 2등 피크가 사실상 없을 때 margin 상한 (log-z 안정화)


@dataclass
class EndpointEvidence:
    delta_nm: float
    cnr: float
    rise_nm: float
    margin: float
    npk: float
    overshoot: float
    pol: float          # ±1, +u 방향 스텝 부호. 판정 불가면 0
    edge_valid: bool
    s_peak: float       # 프로파일 좌표 (px). plateau_cv/오버레이용


def _parabolic_refine(y: np.ndarray, i: int) -> float:
    """피크 인덱스 i 주변 3점 포물선 보간 오프셋 (샘플 단위, [-0.5, 0.5])."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    denom = y[i - 1] - 2 * y[i] + y[i + 1]
    if abs(denom) < _EPS:
        return 0.0
    return float(np.clip(0.5 * (y[i - 1] - y[i + 1]) / denom, -0.5, 0.5))


def _crossing_nearest(s: np.ndarray, q: np.ndarray, thr: float,
                      s_ref: float) -> float:
    """q가 thr를 가로지르는 위치 중 s_ref에 가장 가까운 것 (선형 보간). 없으면 NaN."""
    d = q - thr
    sign_change = np.where(d[:-1] * d[1:] <= 0)[0]
    sign_change = sign_change[np.abs(d[sign_change]) + np.abs(d[sign_change + 1]) > _EPS]
    if len(sign_change) == 0:
        return np.nan
    xs = []
    for i in sign_change:
        t = d[i] / (d[i] - d[i + 1])
        xs.append(s[i] + t * (s[i + 1] - s[i]))
    xs = np.asarray(xs)
    return float(xs[np.argmin(np.abs(xs - s_ref))])


def _halfmax_centroid(s: np.ndarray, ga: np.ndarray, ipk: int) -> float:
    """피크 주변 반치폭(>= 0.5·peak) 연속 구간의 가중 센트로이드 (px).

    argmax(+포물선)는 노이즈가 강한 쪽(밝은 쪽)으로 계통 편향된다 — 특히
    defocus로 피크가 넓고 낮을 때. 대칭 구간 센트로이드는 노이즈가 0평균으로
    상쇄되어 편향이 훨씬 작다. 구간이 너무 짧으면 포물선 보간으로 폴백.
    """
    g1 = ga[ipk]
    thr = 0.5 * g1
    a = ipk
    while a > 0 and ga[a - 1] >= thr:
        a -= 1
    b = ipk
    while b < len(ga) - 1 and ga[b + 1] >= thr:
        b += 1
    if b - a + 1 < 3:
        return s[ipk] + _parabolic_refine(ga, ipk) * (s[1] - s[0] if len(s) > 1 else 1.0)
    w = ga[a:b + 1] - thr
    return float(np.sum(w * s[a:b + 1]) / np.sum(w))


def _local_maxima(y: np.ndarray) -> np.ndarray:
    """단순 국소 최대 인덱스 (양쪽보다 크거나, 왼쪽보다 크고 오른쪽과 같음)."""
    if len(y) < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def analyze_endpoint(prof: Profile, g: np.ndarray, expected_s: float,
                     inward: float, px_nm: float, cfg: dict) -> EndpointEvidence:
    """한 엔드포인트의 증거 피쳐. inward = +1(S) / -1(E): 플래토 쪽 방향."""
    s, p = prof.s, prof.p
    nan = EndpointEvidence(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           0.0, False, np.nan)

    W = float(np.clip(cfg["win_frac"] * prof.L, cfg["win_min_px"], cfg["win_max_px"]))
    half = W / 2
    out_lim = min(half, effective_margin(prof.L, cfg))
    in_lim = min(half, prof.L / 2)
    lo = expected_s - (out_lim if inward > 0 else in_lim)
    hi = expected_s + (in_lim if inward > 0 else out_lim)
    idx = np.where((s >= lo) & (s <= hi))[0]
    if len(idx) < 5:
        return nan

    ga = np.abs(g[idx])
    ipk_local = int(np.argmax(ga))
    ipk = idx[ipk_local]
    g1 = float(ga[ipk_local])
    edge_valid = 0 < ipk_local < len(idx) - 1
    s_peak = _halfmax_centroid(s[idx], ga, ipk_local)
    pol = float(np.sign(g[ipk])) if g1 > _EPS else 0.0
    delta_nm = (s_peak - expected_s) * px_nm

    # margin: 1등 주변 ±peak_suppress 샘플 억제 후 2등 피크
    suppress = int(cfg["peak_suppress"])
    rest = ga.copy()
    a, b = max(0, ipk_local - suppress), min(len(rest), ipk_local + suppress + 1)
    rest[a:b] = 0.0
    g2 = float(rest.max()) if len(rest) else 0.0
    margin = float(min(g1 / max(g2, g1 * 1e-3, _EPS), _MARGIN_CAP))

    # npk: 유의 피크 개수 (g > npk_ratio * peak)
    maxima = _local_maxima(ga)
    npk = float(np.sum(ga[maxima] > cfg["npk_ratio"] * g1)) if len(maxima) else 1.0
    npk = max(npk, 1.0)

    # 레벨 밴드: 피크에서 guard 이상 떨어진 안/밖 구간의 중앙값
    guard = max(2.0 * float(cfg["grad_sigma_px"]), 1.5)
    s_rel = (s - s_peak) * inward          # 양수 = 플래토(안) 쪽
    in_band = (s_rel >= guard) & (s_rel <= in_lim + (expected_s - s_peak) * inward)
    out_band = (s_rel <= -guard) & (s_rel >= -(out_lim - (expected_s - s_peak) * inward))
    if in_band.sum() < 2 or out_band.sum() < 2:
        return EndpointEvidence(delta_nm, np.nan, np.nan, margin, npk, np.nan,
                                pol, edge_valid, s_peak)
    p_in = float(np.median(p[in_band]))
    p_out = float(np.median(p[out_band]))
    step_h = abs(p_in - p_out)
    cnr = step_h / prof.sigma if (prof.sigma and prof.sigma > 0) else np.nan

    # rise: 10–90% 상승폭. q = 0(밖) → 1(안) 정규화 후 피크 근처 교차점
    rise_nm = np.nan
    if step_h > _EPS:
        near = np.abs(s - s_peak) <= max(4 * guard, 5.0)
        if near.sum() >= 4:
            q = (p[near] - p_out) / (p_in - p_out)
            s10 = _crossing_nearest(s[near], q, 0.1, s_peak)
            s90 = _crossing_nearest(s[near], q, 0.9, s_peak)
            if not (np.isnan(s10) or np.isnan(s90)):
                rise_nm = abs(s90 - s10) * px_nm

    # overshoot: 스텝 양쪽 near 밴드에서 자기 쪽 점근선을 넘는 진폭 / 스텝 높이
    overshoot = np.nan
    if step_h > _EPS:
        near_in = (s_rel >= guard) & (s_rel <= 3 * guard)
        near_out = (s_rel <= -guard) & (s_rel >= -3 * guard)
        if near_in.sum() >= 2 and near_out.sum() >= 2:
            hi_side_in = p_in > p_out
            os_in = (np.max(p[near_in]) - p_in) if hi_side_in else (p_in - np.min(p[near_in]))
            os_out = (p_out - np.min(p[near_out])) if hi_side_in else (np.max(p[near_out]) - p_out)
            overshoot = max(0.0, float(os_in), float(os_out)) / step_h

    return EndpointEvidence(delta_nm, cnr, rise_nm, margin, npk, overshoot,
                            pol, edge_valid, s_peak)


def plateau_cv(prof: Profile, s_peak_s: float, s_peak_e: float,
               grad_sigma_px: float) -> float:
    """S–E 사이 플래토의 robust 변동계수. 엣지 영향권(3×guard) 제외."""
    guard = max(2.0 * grad_sigma_px, 1.5)
    a = (np.nan_to_num(s_peak_s, nan=0.0)) + 3 * guard
    b = (np.nan_to_num(s_peak_e, nan=prof.L)) - 3 * guard
    band = (prof.s >= a) & (prof.s <= b)
    if band.sum() < 5:
        return np.nan
    v = prof.p[band]
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return float(1.4826 * mad / max(abs(med), 1e-6))


def evidence_features(prof: Profile, px_nm: float, cfg: dict) -> dict:
    """한 CD의 이미지 증거 피쳐 전체를 dict로."""
    sigma_samples = float(cfg["grad_sigma_px"]) / DS
    g = gaussian_filter1d(prof.p, sigma=sigma_samples, order=1) / DS

    ev_s = analyze_endpoint(prof, g, 0.0, +1.0, px_nm, cfg)
    ev_e = analyze_endpoint(prof, g, prof.L, -1.0, px_nm, cfg)
    pcv = plateau_cv(prof, ev_s.s_peak, ev_e.s_peak, float(cfg["grad_sigma_px"]))

    return {
        "delta_s": ev_s.delta_nm, "delta_e": ev_e.delta_nm,
        "cnr_s": ev_s.cnr, "cnr_e": ev_e.cnr,
        "rise_s": ev_s.rise_nm, "rise_e": ev_e.rise_nm,
        "margin_s": ev_s.margin, "margin_e": ev_e.margin,
        "npk_s": ev_s.npk, "npk_e": ev_e.npk,
        "overshoot_s": ev_s.overshoot, "overshoot_e": ev_e.overshoot,
        "plateau_cv": pcv,
        "pol_s": ev_s.pol, "pol_e": ev_e.pol,
        "edge_valid_s": ev_s.edge_valid, "edge_valid_e": ev_e.edge_valid,
    }
