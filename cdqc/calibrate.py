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
                     l1: pl.DataFrame, write: bool = True,
                     baseline_note: str = "no"
                     ) -> tuple[CohortStats, dict]:
    """주어진 피쳐 프레임으로 코호트 통계 + auto 임계값 계산.

    주의: 통계·auto 임계값은 "데이터셋 대부분이 정상" 가정에 기댄다 (트림은
    5% 수준 오염만 방어). 불량률이 높은 데이터로 캘리브레이션하면 분위 기반
    임계값이 그대로 올라간다 — 그런 경우 --baseline으로 정상 이미지 목록을
    지정할 것.
    """
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
                "baseline_filtered": baseline_note,
            },
            "thresholds": thresholds,
        }
        doc.update(cs.to_dict())
        path = cfg.path("calibrated")
        path.write_text(dump_toml(doc), encoding="utf-8")
    return cs, thresholds


def load_baseline_ids(path) -> set[str]:
    """--baseline 목록 파일: 한 줄에 image_id 하나, '#' 주석 허용."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise CdqcError("E-CAL-03", str(p))
    ids = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    if not ids:
        raise CdqcError("E-CAL-03", f"{p}: 빈 목록")
    return ids


def run_calibrate(cfg: Config, force: bool = False,
                  baseline: str | None = None) -> dict:
    """CLI 진입점: 캐시된 피쳐로 calibrated.toml 작성.

    baseline이 주어지면 그 image_id들만으로 통계/임계값을 계산한다
    (오염된 데이터셋에서 auto 분위수가 올라가는 것을 방어).
    """
    from .pipeline import extract_features
    l3, l2s, l1 = extract_features(cfg, force=force)
    note = "no"
    if baseline is not None:
        ids = load_baseline_ids(baseline)
        l3 = l3.filter(pl.col("image_id").is_in(list(ids)))
        l2s = l2s.filter(pl.col("image_id").is_in(list(ids)))
        l1 = l1.filter(pl.col("image_id").is_in(list(ids)))
        if l1.height == 0:
            raise CdqcError("E-CAL-03", "목록과 매칭되는 이미지가 0개")
        note = f"yes ({l1.height} images)"
    _, thresholds = calibrate_frames(cfg, l3, l2s, l1, write=True,
                                     baseline_note=note)
    return thresholds
