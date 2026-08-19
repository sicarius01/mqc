"""TOML 설정 로드/병합/해시.

우선순위 (spec §5.1):
    CLI 플래그(--set/--root)  >  config.toml 명시값  >  calibrated.toml  >  내장 기본값

- 임계값에 문자열 "auto"를 쓰면 calibrated.toml 값을 따른다.
- calibrated.toml은 calibrate가 쓰는 파일이며 손으로 편집하지 않는다.
- 모든 출력에 config 해시와 calibrated 해시를 스탬프한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from .errors import CdqcError

# ---------------------------------------------------------------- 내장 기본값

DEFAULTS: dict[str, Any] = {
    "project": {"root": ".", "name": "cdqc", "seed": 42},
    "paths": {
        "data_dir": "data",
        "image_dir": "images",
        "output_dir": "out",
        "cache_dir": "out/cache",
        "calibrated": "calibrated.toml",
        "labels": "labels.csv",
    },
    "data": {
        "format": "csv",
        "image_ext": ".png",
        "value_unit": "nm",              # 단위 컬럼이 없을 때의 폴백 (nm | angstrom)
        "cd_index_source": "row_order",  # row_order | column
        "default_recipe_id": "R1",       # recipe_id 컬럼이 없을 때 채우는 상수
        "columns": {
            "recipe_id": "recipe_id",    # 선택 — 없으면 default_recipe_id
            "image_id": "image_id",
            "image_path": "image_path",
            "category_id": "category_id",
            "cd_index": "cd_index",      # cd_index_source="column"일 때만 사용
            "sx": "sx", "sy": "sy", "ex": "ex", "ey": "ey",
            "value": "value",            # 보고 측정값 (필수)
            "unit": "",                  # 단위 컬럼. "" = 없음 → value_unit 폴백
            "px_nm": "px_nm",            # 선택 — 있으면 우선, 없으면 value로 역산
        },
        "coords": {"convention": "auto", "origin": "zero", "y_flip": False, "scale": 1.0},
    },
    "sampling": {
        "ribbon_half_w": 2,
        "win_frac": 0.4,
        "win_min_px": 8,
        "win_max_px": 64,
        "margin_px": 6,
        "grad_sigma_px": 1.0,
        "noise_patch_px": 15,
        "peak_suppress": 6,
        "npk_ratio": 0.3,
    },
    "sequence": {"local_window": 9, "method": "robust_linear", "min_seq_len": 5},
    "cohort": {
        "key_l3": ["recipe_id", "category_id"],
        "key_l1": ["recipe_id"],
        "min_cohort_n": 200,
        "trim_frac": 0.05,
        "mad_floor_default": 1e-6,
        "mad_floor": {"npk_s": 0.5, "npk_e": 0.5, "n_cd": 0.5,
                      "sat_lo": 0.002, "sat_hi": 0.002, "hist_emd": 0.5,
                      "frac_flagged": 0.05, "overshoot_s": 0.02,
                      "overshoot_e": 0.02, "max_run": 0.5,
                      "noise_sigma": 0.05, "dyn_range": 1.0,
                      "delta_median_s": 0.05, "delta_median_e": 0.05,
                      "value_mismatch_nm": 0.02},
        "log_features": {
            "names": ["rise_s", "rise_e", "margin_s", "margin_e",
                      "dstep_s", "dstep_e", "cd_mad"],
        },
    },
    "features": {
        "enabled_l3": "all",
        "enabled_l2": "all",
        "enabled_l1": "all",
        "direction": {},
    },
    "thresholds": {
        "t_soft": "auto",
        "t_image": "auto",
        "t_seq": "auto",
        "auto": {"soft_quantile": 0.90, "image_quantile": 0.99, "seq_quantile": 0.99},
        "tolerance_nm": {"default": 1.0},
        "impact": {"stat": "median", "trimmed_frac": 0.1},
    },
    "gates": {"enable_g0": True, "enable_g1": True, "enable_g2": True, "run_cluster_min": 3},
    "report": {
        "html": True,
        "overlay_max_images": 50,
        "overlay_flagged_only": True,
        "summary_dir": "summary",
        "internal_dir": "internal",
        "summary": {
            "feature_stats": True, "correlation": True, "gate_counts": True,
            "convention": True, "selftest": True, "version": True,
        },
    },
    "synthetic": {
        "n_images": 30,
        "image_size": [512, 512],
        "n_categories": 3,
        "cds_per_category": 20,
        "px_nm": 0.5,
        "base_contrast": 60,
        "noise_sigma": 6.0,
        "edge_rise_px": 1.5,
        "coord_jitter_px": 0.3,
        "fringe": False,
        "inject": {
            "defocus": [0, 1, 2, 4],
            "low_contrast": [1.0, 0.5, 0.25],
            "noise_up": [1.0, 2.0, 4.0],
            "edge_jump_nm": [0, 1, 2, 5],
            "systematic_bias_nm": [0, 1, 3],
            "oblique_deg": [0, 10, 20],
            "missing_frac": [0, 0.1, 0.3],
            "double_edge_offset_px": [0, 4, 8],
            "plateau_defect": [False, True],
            "saturation": [1.0, 2.0, 4.0],
            "partial_damage": [0, 0.25],
        },
    },
    "selftest": {
        "n_images_per_case": 4,
        "z_pass": 3.0,
        "z_specificity": 1.5,
        "mono_slack": 0.5,
    },
    "evaluate": {"fpr": 0.05},
    "logging": {"level": "INFO", "file": "out/cdqc.log", "error_codes": True},
}

# 임의 키를 허용하는 섹션 (카테고리명/피쳐명이 키가 됨)
_WILDCARD_SECTIONS = {
    ("thresholds", "tolerance_nm"),
    ("features", "direction"),
    ("cohort", "mad_floor"),
}


# ---------------------------------------------------------------- 유틸

def _deep_merge(base: dict, overlay: dict) -> dict:
    """overlay를 base 위에 재귀 병합한 새 dict를 반환."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_unknown_keys(user: dict, schema: dict, path: tuple = ()) -> None:
    for k, v in user.items():
        if tuple(path) in _WILDCARD_SECTIONS:
            continue
        if k not in schema:
            raise CdqcError("E-CONF-02", f"{'.'.join(path + (k,))}")
        if isinstance(v, dict) and isinstance(schema[k], dict):
            _check_unknown_keys(v, schema[k], path + (k,))


def parse_set_overrides(pairs: list[str]) -> dict:
    """--set thresholds.t_soft=3.0 형태를 중첩 dict로. 값은 TOML 문법으로 해석."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise CdqcError("E-CONF-04", pair)
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise CdqcError("E-CONF-04", pair)
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # 따옴표 없는 문자열
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def dict_hash(d: dict) -> str:
    """정렬된 JSON 직렬화의 sha256 앞 12자리."""
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def dump_toml(d: dict) -> str:
    """중첩 dict → TOML 문자열. calibrated.toml 기록용 최소 구현."""
    lines: list[str] = []

    def fmt(v: Any) -> str:
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            v = v.item()          # numpy 스칼라 → 파이썬 스칼라 (repr 오염 방지)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return repr(v)
        if isinstance(v, float):
            if v != v:            # NaN
                return "nan"
            return repr(float(v))
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        raise TypeError(f"cannot serialize {type(v)} to TOML")

    def quote_key(k: str) -> str:
        bare = k and all(c.isalnum() or c in "_-" for c in k)
        return k if bare else json.dumps(k, ensure_ascii=False)

    def walk(d: dict, prefix: tuple = ()) -> None:
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if prefix and (scalars or not tables):
            lines.append(f"[{'.'.join(quote_key(p) for p in prefix)}]")
        for k, v in scalars.items():
            lines.append(f"{quote_key(k)} = {fmt(v)}")
        if scalars:
            lines.append("")
        for k, v in tables.items():
            walk(v, prefix + (k,))

    walk(d)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- Config

class Config:
    """병합 완료된 설정 + 경로 해석 + 해시.

    cfg["sampling"]["win_frac"] 처럼 dict로 접근하거나 cfg.get("sampling.win_frac").
    """

    def __init__(self, merged: dict, calibrated: dict, config_path: Path | None,
                 root_override: str | None = None):
        self.data = merged
        self.calibrated = calibrated  # calibrated.toml 전체 (cohort stats 포함)
        self.config_path = config_path
        base = config_path.parent if config_path else Path.cwd()
        root = Path(root_override or merged["project"]["root"])
        self.root = root if root.is_absolute() else (base / root).resolve()
        self.config_hash = dict_hash(merged)
        self.calibrated_hash = dict_hash(calibrated) if calibrated else "none"

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def path(self, key: str) -> Path:
        """[paths] 항목을 root 기준 절대경로로."""
        p = Path(self.data["paths"][key])
        return p if p.is_absolute() else self.root / p

    def threshold(self, name: str) -> float:
        """t_soft/t_image/t_seq 해석. "auto"면 calibrated에서 가져온다."""
        v = self.data["thresholds"][name]
        if v == "auto":
            cal = self.calibrated.get("thresholds", {})
            if name not in cal:
                raise CdqcError("E-CONF-05", f"thresholds.{name}")
            return float(cal[name])
        return float(v)

    def tolerance_nm(self, category_id: str) -> float:
        tol = self.data["thresholds"]["tolerance_nm"]
        return float(tol.get(category_id, tol["default"]))

    def seed(self) -> int:
        return int(self.data["project"]["seed"])

    def with_overrides(self, overrides: dict) -> "Config":
        """일부 키만 바꾼 새 Config (selftest 샌드박스, --sweep 용)."""
        merged = _deep_merge(self.data, overrides)
        cfg = Config(merged, self.calibrated, self.config_path, str(self.root))
        return cfg


def load_config(config_path: str | Path | None = None,
                root: str | None = None,
                set_overrides: list[str] | None = None) -> Config:
    """우선순위대로 병합된 Config 생성."""
    user: dict = {}
    cpath: Path | None = None
    if config_path is not None:
        cpath = Path(config_path)
        if not cpath.exists():
            raise CdqcError("E-CONF-01", str(cpath))
        with open(cpath, "rb") as f:
            user = tomllib.load(f)
        _check_unknown_keys(user, DEFAULTS)
    merged = _deep_merge(DEFAULTS, user)
    if set_overrides:
        merged = _deep_merge(merged, parse_set_overrides(set_overrides))

    # calibrated.toml 로드 (있으면). 병합이 아니라 별도 보관 — "auto" 해석에만 쓰임.
    tmp = Config(merged, {}, cpath, root)
    calibrated: dict = {}
    cal_path = tmp.path("calibrated")
    if cal_path.exists():
        try:
            with open(cal_path, "rb") as f:
                calibrated = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise CdqcError("E-CAL-01", f"{cal_path}: {e}") from e

    return Config(merged, calibrated, cpath, root)
