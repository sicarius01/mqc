"""L3 — 개별 CD 피쳐 (순수 함수, 코호트 무관).

한 시퀀스(image × category, cd_index 순)에 대해 기하/시퀀스 잔차와 이미지
증거 피쳐를 계산한다. 입력 좌표는 io.transform_coords를 거친 내부 컨벤션
(x=col, y=row, zero-origin)이어야 한다.

마지막에 sanitize()가 비율 피쳐의 분모 폭발을 막는다 — **카테고리 예외가
아니라 피쳐 자체의 일반 규칙**이다 (spec §3.2).
"""

from __future__ import annotations

import numpy as np

from .. import geometry as geo
from ..evidence import evidence_features
from ..sampling import sample_ribbon_profiles


def l3_sequence_features(img: np.ndarray | None, S: np.ndarray, E: np.ndarray,
                         px_nm: float, cfg) -> dict[str, np.ndarray]:
    """한 시퀀스의 L3 피쳐 전체. img=None이면 기하 피쳐만 (이미지 증거는 NaN).

    S, E: (n, 2) 내부 좌표, cd_index 오름차순 정렬 상태.
    반환: 피쳐 이름 → (n,) 배열.
    """
    n = len(S)
    seq_cfg = cfg["sequence"]
    window = int(seq_cfg["local_window"])
    method = str(seq_cfg["method"])
    min_len = int(seq_cfg["min_seq_len"])

    seg = E - S
    cd_px = np.linalg.norm(seg, axis=1)
    cd_nm = cd_px * px_nm
    u = seg / np.where(cd_px > 0, cd_px, 1.0)[:, None]

    tan_s = geo.unit_tangents(S)
    tan_e = geo.unit_tangents(E)
    ribbon_t = geo.mean_edge_tangent(tan_s, tan_e)

    out: dict[str, np.ndarray] = {
        "cd_nm": cd_nm,
        "cd_resid": geo.local_residual_1d(cd_nm, window, method, min_len),
        "s_resid": geo.normal_residual(S, window, method, min_len) * px_nm,
        "e_resid": geo.normal_residual(E, window, method, min_len) * px_nm,
        "dstep_s": geo.step_normal(S) * px_nm,
        "dstep_e": geo.step_normal(E) * px_nm,
        "obliquity": geo.obliquity_deg(u, ribbon_t),
        "curv_s": geo.curvature3(S) * px_nm,
        "curv_e": geo.curvature3(E) * px_nm,
        "angle": geo.segment_angle_deg(seg),
    }
    out["angle_resid_seq"] = geo.circular_residual_deg180(out["angle"])

    ev_names = ["delta_s", "delta_e", "delta_B_s", "delta_B_e",
                "delta_scatter_s", "delta_scatter_e",
                "cnr_s", "cnr_e", "rise_s", "rise_e",
                "margin_s", "margin_e", "npk_s", "npk_e",
                "overshoot_s", "overshoot_e", "plateau_cv", "pol_s", "pol_e"]
    if img is None:
        for name in ev_names:
            out[name] = np.full(n, np.nan)
        out["edge_valid_s"] = np.zeros(n, dtype=bool)
        out["edge_valid_e"] = np.zeros(n, dtype=bool)
        return out

    profiles = sample_ribbon_profiles(img, S, E, ribbon_t, cfg["sampling"])
    rows = [evidence_features(p, px_nm, cfg["sampling"]) for p in profiles]
    for name in ev_names:
        out[name] = np.array([r[name] for r in rows], dtype=np.float64)
    out["edge_valid_s"] = np.array([r["edge_valid_s"] for r in rows], dtype=bool)
    out["edge_valid_e"] = np.array([r["edge_valid_e"] for r in rows], dtype=bool)
    return sanitize(out, px_nm, cfg.get("sanitize", {}))


def sanitize(feats: dict[str, np.ndarray], px_nm: float,
             cfg: dict) -> dict[str, np.ndarray]:
    """비율 피쳐의 분모 폭발을 막는 **일반 규칙** (제자리 수정 후 같은 dict 반환).

    카테고리 예외 목록이 아니다 (spec §11-1). 스텝이 노이즈 수준이면
    `overshoot = 오버슈트 진폭 / 스텝 높이`는 0으로 나눈 값이고,
    1px의 절반도 안 되는 `rise`는 물리적으로 측정 불가다 — 어느 카테고리든
    같은 조건에서 같은 이유로 무의미하므로 NaN(= 미측정)으로 만든다.

    두 규칙 모두 임계는 Params.min_cnr_for_ratio / min_rise_px.
    """
    min_cnr = float(cfg.get("min_cnr_for_ratio", 0.0))
    min_rise_nm = float(cfg.get("min_rise_px", 0.0)) * float(px_nm)
    for tag in ("s", "e"):
        cnr = feats.get(f"cnr_{tag}")
        rise = feats.get(f"rise_{tag}")
        over = feats.get(f"overshoot_{tag}")
        # rise < min_rise_px: 엣지 폭이 subpixel — 상승폭을 잰 게 아니라
        # 샘플 격자를 잰 것이다
        if rise is not None and min_rise_nm > 0:
            rise[np.isfinite(rise) & (rise < min_rise_nm)] = np.nan
        if over is None:
            continue
        bad = np.zeros(len(over), dtype=bool)
        if cnr is not None and min_cnr > 0:
            bad |= ~np.isfinite(cnr) | (cnr < min_cnr)     # 스텝이 노이즈 수준
        if rise is not None:
            bad |= ~np.isfinite(rise)                       # 스텝을 못 쟀음
        over[bad] = np.nan
    return feats
