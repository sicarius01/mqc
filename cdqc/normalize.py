"""정규화 — 코호트 robust 통계, 트림, 폴백, 방향 있는 z.

두 층의 정규화 중 코호트 층을 담당한다 (시퀀스 내 잔차는 geometry/l3에서 이미
피쳐로 계산됨). robust z = (x − median) / max(1.4826·MAD, mad_floor).
heavy-tail 피쳐는 log 후 z. 오염 방어로 1차 z 상위 trim_frac 제거 후 1회 재계산.

폴백 체인: (recipe, category) → (recipe,) → 전역. 코호트 행수 < min_cohort_n이면
다음 티어로. 폴백 여부는 cohort_fallback_level(0/1/2)로 기록.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .errors import warn
from .features.registry import BY_NAME, Z_ON_BAD, features_of, worse_when

LOG_EPS = 1e-3          # log 변환 오프셋: log(x + LOG_EPS)
GLOBAL_KEY = "__all__"
TIERS = ("full", "recipe", "global")


def robust_stats(x: np.ndarray, trim_frac: float) -> tuple[float, float]:
    """트림 1회 적용한 (median, MAD). 유효 표본 없으면 (nan, nan)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if trim_frac > 0 and len(x) >= 10:
        scale = max(1.4826 * mad, 1e-12)
        z = np.abs(x - med) / scale
        keep = z <= np.quantile(z, 1 - trim_frac)
        if keep.sum() >= 3:
            xk = x[keep]
            med = float(np.median(xk))
            mad = float(np.median(np.abs(xk - med)))
    return (med, mad)


def _mode_pm1(x: np.ndarray) -> float:
    """±1 범주형의 최빈값. 표본 없으면 0 (불일치 판정 안 함)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return 0.0
    return 1.0 if np.sum(x > 0) >= np.sum(x < 0) else -1.0


def key_cols_for(level: str, cfg) -> list[str]:
    return list(cfg["cohort"]["key_l1" if level == "l1" else "key_l3"])


def _tier_cols(level: str, cfg) -> dict[str, list[str]]:
    full = key_cols_for(level, cfg)
    tiers = {"full": full}
    tiers["recipe"] = ["recipe_id"] if full != ["recipe_id"] else full
    tiers["global"] = []
    return tiers


def _key_str(vals: tuple) -> str:
    return "|".join(str(v) for v in vals) if vals else GLOBAL_KEY


def _log_names(cfg) -> set[str]:
    return set(cfg["cohort"]["log_features"]["names"])


def _mad_floor(cfg, name: str) -> float:
    return float(cfg["cohort"]["mad_floor"].get(name, cfg["cohort"]["mad_floor_default"]))


def transform_value(x: np.ndarray, name: str, cfg) -> np.ndarray:
    """z 계산 전 변환 (log 피쳐면 log(x + eps))."""
    if name in _log_names(cfg):
        return np.log(np.clip(np.asarray(x, dtype=np.float64), 0, None) + LOG_EPS)
    return np.asarray(x, dtype=np.float64)


class CohortStats:
    """코호트 통계 + 극성 최빈값 + 히스토그램 템플릿. calibrated.toml과 왕복."""

    def __init__(self):
        # stats[level][tier][key_str][feat] = [med, mad]; ["__n__"] = 행수
        self.stats: dict = {}
        self.modes: dict = {}
        self.hist_templates: dict = {}   # recipe_id(또는 GLOBAL_KEY) → [256 floats]

    # ------------------------------------------------------------- 계산

    def compute_level(self, df: pl.DataFrame, level: str, cfg) -> None:
        trim = float(cfg["cohort"]["trim_frac"])
        znames = [f.name for f in features_of(level, "z") if f.name in df.columns]
        mnames = [f.name for f in features_of(level, "match") if f.name in df.columns]
        lstats = self.stats.setdefault(level, {})
        lmodes = self.modes.setdefault(level, {})
        for tier, cols in _tier_cols(level, cfg).items():
            tstats = lstats.setdefault(tier, {})
            tmodes = lmodes.setdefault(tier, {})
            groups = df.partition_by(cols, as_dict=True) if cols else {(): df}
            for keyvals, part in groups.items():
                ks = _key_str(keyvals)
                entry = tstats.setdefault(ks, {})
                entry["__n__"] = part.height
                for name in znames:
                    x = transform_value(part[name].to_numpy(), name, cfg)
                    med, mad = robust_stats(x, trim)
                    entry[name] = [med, mad]
                if mnames:
                    mentry = tmodes.setdefault(ks, {})
                    for name in mnames:
                        mentry[name] = _mode_pm1(part[name].to_numpy())

    def compute_hist_templates(self, l1: pl.DataFrame) -> None:
        """recipe별 + 전역 중앙값 히스토그램 (재정규화)."""
        def template(part: pl.DataFrame) -> list[float]:
            h = np.stack([np.asarray(v) for v in part["hist"].to_list()])
            t = np.median(h, axis=0)
            s = t.sum()
            return list(t / s if s > 0 else t)

        for (recipe,), part in l1.partition_by(["recipe_id"], as_dict=True).items():
            self.hist_templates[str(recipe)] = template(part)
        self.hist_templates[GLOBAL_KEY] = template(l1)

    def template_for(self, recipe_id: str) -> np.ndarray | None:
        t = self.hist_templates.get(str(recipe_id)) or self.hist_templates.get(GLOBAL_KEY)
        return np.asarray(t) if t is not None else None

    # ------------------------------------------------------------- 적용

    def _lookup(self, level: str, keyvals_by_tier: dict[str, tuple],
                min_n: int) -> tuple[dict, dict, int]:
        """폴백 체인 따라 (stats entry, modes entry, fallback_level) 선택."""
        lstats = self.stats.get(level, {})
        lmodes = self.modes.get(level, {})
        chosen = None
        for fb, tier in enumerate(TIERS):
            ks = _key_str(keyvals_by_tier[tier])
            entry = lstats.get(tier, {}).get(ks)
            if entry is None:
                continue
            if chosen is None:
                chosen = (entry, lmodes.get(tier, {}).get(ks, {}), fb)
            if entry["__n__"] >= min_n:
                return entry, lmodes.get(tier, {}).get(ks, {}), fb
        if chosen is not None:   # 전 티어 표본 부족 → 마지막으로 존재하는 것
            lastfb = len(TIERS) - 1
            ks = _key_str(keyvals_by_tier[TIERS[lastfb]])
            entry = lstats.get(TIERS[lastfb], {}).get(ks)
            if entry is not None:
                return entry, lmodes.get(TIERS[lastfb], {}).get(ks, {}), lastfb
            return chosen
        return {}, {}, len(TIERS) - 1

    def apply(self, df: pl.DataFrame, level: str, cfg) -> pl.DataFrame:
        """방향 있는 z 컬럼(z_*)과 cohort_fallback_level을 추가."""
        min_n = int(cfg["cohort"]["min_cohort_n"])
        tiers = _tier_cols(level, cfg)
        full_cols = tiers["full"]
        znames = [f.name for f in features_of(level, "z") if f.name in df.columns]
        bnames = [f.name for f in features_of(level, "bool") if f.name in df.columns]
        mnames = [f.name for f in features_of(level, "match") if f.name in df.columns]

        df = df.with_row_index("__ridx__")
        parts = []
        n_fallback = 0
        for keyvals, part in df.partition_by(full_cols, as_dict=True).items():
            keymap = {t: tuple(part[0, c] for c in cols) if cols else ()
                      for t, cols in tiers.items()}
            keymap["full"] = keyvals
            entry, modes, fb = self._lookup(level, keymap, min_n)
            if fb > 0:
                n_fallback += 1
            new_cols = [pl.lit(fb).cast(pl.Int8).alias("cohort_fallback_level")]
            for name in znames:
                med, mad = entry.get(name, (np.nan, np.nan))
                x = transform_value(part[name].to_numpy(), name, cfg)
                scale = max(1.4826 * (mad if np.isfinite(mad) else np.nan),
                            _mad_floor(cfg, name))
                z = (x - med) / scale
                ww = worse_when(cfg, name)
                dz = np.abs(z) if ww == "both" else (-z if ww == "low" else z)
                new_cols.append(pl.Series(f"z_{name}", dz))
                if ww == "both":
                    # 부호 보존 z — 계통 편향 분석(selftest 특이도, 리포트)용
                    new_cols.append(pl.Series(f"zs_{name}", z))
            for name in bnames:
                ok = part[name].to_numpy().astype(bool)
                new_cols.append(pl.Series(f"z_{name}", np.where(ok, 0.0, Z_ON_BAD)))
            for name in mnames:
                mode = modes.get(name, 0.0)
                v = part[name].to_numpy()
                bad = (mode != 0.0) & np.isfinite(v) & (v != 0) & (v != mode)
                new_cols.append(pl.Series(f"z_{name}", np.where(bad, Z_ON_BAD, 0.0)))
            parts.append(part.with_columns(new_cols))

        if n_fallback:
            warn("W-COH-01", f"level={level}, cohorts={n_fallback}")
        # 입력 행 순서 보존
        return pl.concat(parts).sort("__ridx__").drop("__ridx__")

    # ------------------------------------------------------------- 직렬화

    def to_dict(self) -> dict:
        return {"cohort_stats": self.stats, "modes": self.modes,
                "hist_templates": self.hist_templates}

    @classmethod
    def from_dict(cls, d: dict) -> "CohortStats":
        cs = cls()
        cs.stats = d.get("cohort_stats", {})
        cs.modes = d.get("modes", {})
        cs.hist_templates = d.get("hist_templates", {})
        return cs
