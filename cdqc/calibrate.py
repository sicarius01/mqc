"""캘리브레이션 — 코호트 통계와 auto 임계값 계산 → calibrated.toml.

calibrated.toml은 이 모듈만 쓴다. 손으로 편집하지 않으며 config.toml은
건드리지 않는다 (spec §6.5). frac_flagged처럼 플래그에 의존하는 L2 피쳐의
통계는 자체 t_soft로 플래그를 만든 뒤 2단계로 계산한다.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl

from .config import Config, dump_toml
from .errors import CdqcError
from .normalize import CohortStats
from .scoring import _z_cols, add_l2_runtime, score_l3


def _max_directed(df: pl.DataFrame, level: str, cfg, **kw) -> np.ndarray:
    cols = _z_cols(df, level, cfg, **kw)
    if not cols:
        return np.full(df.height, np.nan)
    Z = np.column_stack([df[c].to_numpy() for c in cols])
    Zf = np.where(np.isfinite(Z), Z, -np.inf)
    out = Zf.max(axis=1)
    out[~np.isfinite(Z).any(axis=1)] = np.nan
    return out


def _quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise CdqcError("E-CAL-02", "no finite scores for threshold quantile")
    return float(np.quantile(x, q))


def calibrate_frames(cfg: Config, l3: pl.DataFrame, l2s: pl.DataFrame,
                     l1: pl.DataFrame, write: bool = True
                     ) -> tuple[CohortStats, dict]:
    """주어진 피쳐 프레임으로 코호트 통계 + auto 임계값 계산."""
    from .pipeline import add_hist_emd  # 순환 임포트 방지 (함수 내 지연)

    auto = cfg["thresholds"]["auto"]
    cs = CohortStats()

    cs.compute_hist_templates(l1)
    l1e = add_hist_emd(l1, cs)

    cs.compute_level(l3, "l3", cfg)
    l3z = cs.apply(l3, "l3", cfg)
    t_soft = _quantile(_max_directed(l3z, "l3", cfg), float(auto["soft_quantile"]))

    l3sc = score_l3(l3z, cfg, t_soft)
    l2full = add_l2_runtime(l3sc, l2s, cfg)
    cs.compute_level(l2full, "l2", cfg)
    l2z = cs.apply(l2full, "l2", cfg)
    t_seq = _quantile(_max_directed(l2z, "l2", cfg, only_g2=True),
                      float(auto["seq_quantile"]))

    cs.compute_level(l1e, "l1", cfg)
    l1z = cs.apply(l1e, "l1", cfg)
    t_image = _quantile(_max_directed(l1z, "l1", cfg, only_g0=True),
                        float(auto["image_quantile"]))

    thresholds = {"t_soft": round(t_soft, 6), "t_image": round(t_image, 6),
                  "t_seq": round(t_seq, 6)}

    if write:
        doc = {
            "meta": {
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "config_hash": cfg.config_hash,
                "n_cd": l3.height,
                "n_images": l1.height,
            },
            "thresholds": thresholds,
        }
        doc.update(cs.to_dict())
        path = cfg.path("calibrated")
        path.write_text(dump_toml(doc), encoding="utf-8")
    return cs, thresholds


def run_calibrate(cfg: Config, force: bool = False) -> dict:
    """CLI 진입점: 캐시된 피쳐로 calibrated.toml 작성."""
    from .pipeline import extract_features
    l3, l2s, l1 = extract_features(cfg, force=force)
    _, thresholds = calibrate_frames(cfg, l3, l2s, l1, write=True)
    return thresholds
