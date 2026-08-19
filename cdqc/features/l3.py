"""L3 — 개별 CD 피쳐 (순수 함수, 코호트 무관).

한 시퀀스(image × category, cd_index 순)에 대해 기하/시퀀스 잔차와 이미지
증거 피쳐를 계산한다. 입력 좌표는 io.transform_coords를 거친 내부 컨벤션
(x=col, y=row, zero-origin)이어야 한다.
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

    ev_names = ["delta_s", "delta_e", "cnr_s", "cnr_e", "rise_s", "rise_e",
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
    return out
