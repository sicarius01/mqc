"""cdqc 공개 API — 순수 연산 함수 모음.

계약 (cdqc_change_02 §3.5):
- 좌표: (n,2) float ndarray, (x=col, y=row), zero-origin, 이미지 픽셀 단위.
  시퀀스는 **측정 순서대로 정렬**해서 넣는다.
- 이미지: np.ndarray[uint8], shape (H, W).
- 길이 단위는 함수 경계에서 항상 nm.
- DataFrame 입출력은 pandas.
- 파일 I/O 없음, 전역 상태 없음. 코호트 분리·임계값·판정 기준은 사용자 몫.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import CdqcError
from .features.l1 import hist_emd, l1_image_features
from .features.l3 import l3_sequence_features
from .features.registry import BY_NAME, REGISTRY, Z_ON_BAD, features_of
from .params import Params

LOG_EPS = 1e-3   # log 피쳐 변환 오프셋: log(x + LOG_EPS)


# ================================================================ 검증

def _check_image(img) -> None:
    if img is None:
        return
    if not (isinstance(img, np.ndarray) and img.ndim == 2 and img.dtype == np.uint8):
        raise CdqcError("E-ARG-02", f"got {type(img).__name__}"
                        + (f" dtype={img.dtype} ndim={img.ndim}"
                           if isinstance(img, np.ndarray) else ""))


def _check_points(S, E) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 2 or S.shape != E.shape:
        raise CdqcError("E-ARG-01", f"S{S.shape} vs E{E.shape}, (n,2) 필요")
    if len(S) == 0:
        raise CdqcError("E-ARG-04", "빈 시퀀스")
    return S, E


def _worse_when(params: Params, name: str) -> str:
    ov = params.direction_overrides.get(name)
    return ov if ov else BY_NAME[name].worse_when


# ================================================================ 피쳐 추출

def extract_l3(img: np.ndarray | None, S, E, px_nm: float,
               value_nm=None, params: Params | None = None) -> pd.DataFrame:
    """한 시퀀스(이미지 × CD 카테고리)의 CD별 피쳐.

    img=None이면 좌표 기하 피쳐만 (이미지 증거는 NaN). value_nm을 주면
    value_mismatch_nm(보고값 ↔ 좌표 기하 길이 차) 컬럼이 추가된다.
    행 순서는 입력 순서 그대로 (입력이 곧 측정 순서라는 전제).
    """
    params = params or Params()
    _check_image(img)
    S, E = _check_points(S, E)
    if not (np.isfinite(px_nm) and px_nm > 0):
        raise CdqcError("E-ARG-06", f"px_nm={px_nm}")

    feats = l3_sequence_features(img, S, E, float(px_nm), params.view())
    if value_nm is not None:
        v = np.asarray(value_nm, dtype=np.float64)
        if v.shape != (len(S),):
            raise CdqcError("E-ARG-01", f"value_nm shape {v.shape}, ({len(S)},) 필요")
        feats["value_mismatch_nm"] = v - feats["cd_nm"]

    order = [f.name for f in features_of("l3") if f.name in feats]
    return pd.DataFrame({name: feats[name] for name in order})


def extract_l2(l3_df: pd.DataFrame, group_cols: list[str] | None = None,
               params: Params | None = None) -> pd.DataFrame:
    """시퀀스(그룹)별 요약 피쳐. 임계값과 무관한 부분만.

    group_cols=None이면 l3_df 전체를 한 시퀀스로 취급한다.
    플래그 의존 값(frac_flagged/max_run/impact)은 사용자가 플래그를 정한 뒤
    impact_nm()/max_run()으로 직접 계산한다.
    """
    def agg(g: pd.DataFrame) -> dict:
        cd = g["cd_nm"].to_numpy(dtype=np.float64)
        med = np.nanmedian(cd) if np.isfinite(cd).any() else np.nan
        out = {
            "n_cd": int(len(g)),
            "cd_median": med,
            "cd_mad": (np.nanmedian(np.abs(cd - med))
                       if np.isfinite(cd).any() else np.nan),
        }
        for c, name in (("delta_s", "delta_median_s"), ("delta_e", "delta_median_e")):
            x = g[c].to_numpy(dtype=np.float64) if c in g else np.array([np.nan])
            out[name] = np.nanmedian(x) if np.isfinite(x).any() else np.nan
        for c, name in (("s_resid", "traj_rms_s"), ("e_resid", "traj_rms_e")):
            x = g[c].to_numpy(dtype=np.float64) if c in g else np.array([np.nan])
            out[name] = (float(np.sqrt(np.nanmean(x ** 2)))
                         if np.isfinite(x).any() else np.nan)
        return out

    if group_cols is None:
        return pd.DataFrame([agg(l3_df)])
    rows = []
    for key, g in l3_df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **agg(g)})
    return pd.DataFrame(rows)


def extract_l1(img: np.ndarray, params: Params | None = None) -> dict:
    """이미지 전역 피쳐 (측정 무관, OOD 탐지).

    반환 dict의 "hist"는 256-bin 정규화 히스토그램(np.ndarray) — 코호트
    템플릿(예: 정상 이미지 hist들의 중앙값)과 hist_emd()로 비교한다.
    """
    _check_image(img)
    if img is None:
        raise CdqcError("E-ARG-02", "None")
    return l1_image_features(img)


# ================================================================ 통계·정규화

def robust_stats(x, trim_frac: float = 0.05) -> tuple[float, float]:
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


def _transform(x: np.ndarray, name: str, params: Params) -> np.ndarray:
    if name in params.log_features:
        return np.log(np.clip(np.asarray(x, dtype=np.float64), 0, None) + LOG_EPS)
    return np.asarray(x, dtype=np.float64)


def _mode_pm1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return 0.0
    return 1.0 if np.sum(x > 0) >= np.sum(x < 0) else -1.0


def cohort_stats(df: pd.DataFrame, feature_cols: list[str] | None = None,
                 params: Params | None = None) -> dict:
    """받은 행 전체를 **한 코호트**로 보고 피쳐별 robust 통계 계산.

    코호트를 어떻게 자를지(레시피×카테고리 등)는 호출 전에 사용자가 정한다.
    반환은 JSON 직렬화 가능한 평범한 dict — 저장/로드는 사용자 자유:
        {"n": 행수,
         "features": {이름: {"median":…, "mad":…}},   # z 계열 (log 변환 후 통계)
         "modes":    {이름: ±1.0}}                     # 극성 등 범주형 최빈값
    """
    params = params or Params()
    znames = [f.name for f in REGISTRY if f.kind == "z"]
    mnames = [f.name for f in REGISTRY if f.kind == "match"]
    if feature_cols is not None:
        unknown = set(feature_cols) - {f.name for f in REGISTRY}
        if unknown:
            raise CdqcError("E-ARG-05", f"{sorted(unknown)}")
        znames = [n for n in znames if n in feature_cols]
        mnames = [n for n in mnames if n in feature_cols]

    out: dict = {"n": int(len(df)), "features": {}, "modes": {}}
    for name in znames:
        if name not in df.columns:
            continue
        med, mad = robust_stats(_transform(df[name].to_numpy(), name, params),
                                params.trim_frac)
        out["features"][name] = {"median": med, "mad": mad}
    for name in mnames:
        if name not in df.columns:
            continue
        out["modes"][name] = _mode_pm1(df[name].to_numpy())
    return out


def apply_z(df: pd.DataFrame, stats: dict,
            params: Params | None = None) -> pd.DataFrame:
    """directed z 컬럼(z_*) 추가. worse_when="both" 피쳐는 부호 z(zs_*)도.

    - z 계열: (x − median) / max(1.4826·MAD, mad_floor), log 피쳐는 변환 후.
      방향: low → −z, high → +z, both → |z| (zs_에 부호 보존).
    - bool 계열(edge_valid_*): False면 고정 z=Z_ON_BAD.
    - match 계열(pol_*): stats["modes"] 최빈값과 불일치면 고정 z=Z_ON_BAD.
    원본 df는 수정하지 않는다 (복사본 반환).
    """
    params = params or Params()
    out = df.copy()
    for name, st in stats.get("features", {}).items():
        if name not in df.columns:
            continue
        x = _transform(df[name].to_numpy(), name, params)
        mad = st["mad"]
        scale = max(1.4826 * (mad if np.isfinite(mad) else np.nan),
                    params.mad_floor(name))
        z = (x - st["median"]) / scale
        ww = _worse_when(params, name)
        out[f"z_{name}"] = np.abs(z) if ww == "both" else (-z if ww == "low" else z)
        if ww == "both":
            out[f"zs_{name}"] = z
    for f in REGISTRY:
        if f.kind == "bool" and f.name in df.columns:
            ok = df[f.name].to_numpy().astype(bool)
            out[f"z_{f.name}"] = np.where(ok, 0.0, Z_ON_BAD)
        elif f.kind == "match" and f.name in df.columns:
            mode = stats.get("modes", {}).get(f.name, 0.0)
            v = df[f.name].to_numpy(dtype=np.float64)
            bad = (mode != 0.0) & np.isfinite(v) & (v != 0) & (v != mode)
            out[f"z_{f.name}"] = np.where(bad, Z_ON_BAD, 0.0)
    return out


def threshold_from_quantile(values, q: float) -> float:
    """분위수 기반 임계값 계산 유틸 (기준을 '정하는' 건 사용자).

    예: t_soft = threshold_from_quantile(top_feature(z_normal)["top_z"], 0.90)
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise CdqcError("E-ARG-04", "유효한 값이 없음")
    return float(np.quantile(v, q))


# ================================================================ 판정 연산 헬퍼

def top_feature(z_df: pd.DataFrame, feature_cols: list[str] | None = None,
                params: Params | None = None) -> pd.DataFrame:
    """행별 최대 directed z와 그 피쳐/사유 코드.

    feature_cols 기본값: registry 기본 활성 피쳐 중 z_<이름> 컬럼이 있는 것
    (curv_s/e는 기본 제외 — 명시하면 포함 가능).
    반환: DataFrame(top_z, top_feature, reason_code), 인덱스는 z_df와 동일.
    """
    if feature_cols is None:
        names = [f.name for f in REGISTRY if f.enabled_default
                 and f"z_{f.name}" in z_df.columns]
    else:
        unknown = set(feature_cols) - {f.name for f in REGISTRY}
        if unknown:
            raise CdqcError("E-ARG-05", f"{sorted(unknown)}")
        names = [n for n in feature_cols if f"z_{n}" in z_df.columns]
    if not names:
        raise CdqcError("E-ARG-04", "z_ 컬럼이 없음 — apply_z 먼저")

    Z = np.column_stack([z_df[f"z_{n}"].to_numpy(dtype=np.float64) for n in names])
    Zf = np.where(np.isfinite(Z), Z, -np.inf)
    idx = np.argmax(Zf, axis=1)
    top_z = Zf[np.arange(len(Zf)), idx]
    all_nan = ~np.isfinite(Z).any(axis=1)
    arr = np.array(names)
    reasons = np.array([BY_NAME[n].reason for n in names])
    return pd.DataFrame({
        "top_z": np.where(all_nan, np.nan, top_z),
        "top_feature": np.where(all_nan, "", arr[idx]),
        "reason_code": np.where(all_nan, "", reasons[idx]),
    }, index=z_df.index)


def _stat(x: np.ndarray, stat: str, trimmed_frac: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(x))
    if stat == "trimmed_mean":
        lo, hi = np.quantile(x, [trimmed_frac, 1 - trimmed_frac])
        inner = x[(x >= lo) & (x <= hi)]
        return float(np.mean(inner)) if len(inner) else float(np.mean(x))
    if stat != "median":
        raise CdqcError("E-ARG-06", f"stat={stat!r}")
    return float(np.median(x))


def impact_nm(cd_nm, flags, stat: str = "median",
              trimmed_frac: float = 0.1) -> float:
    """플래그 CD 제외 시 보고 통계량 변화 (nm) — 스펙 공차와 직접 비교.

    플래그가 없으면 0, 전부 플래그면 inf (깨끗한 통계량이 없음).
    """
    cd = np.asarray(cd_nm, dtype=np.float64)
    fl = np.asarray(flags, dtype=bool)
    if cd.shape != fl.shape:
        raise CdqcError("E-ARG-01", f"cd_nm{cd.shape} vs flags{fl.shape}")
    if not fl.any():
        return 0.0
    clean = cd[~fl]
    if not np.isfinite(clean).any():
        return float("inf")
    return float(abs(_stat(cd, stat, trimmed_frac)
                     - _stat(clean, stat, trimmed_frac)))


def max_run(flags) -> int:
    """연속 플래그 최대 길이 — 뭉침(국소 손상) vs 산발(락온 실패) 구분용."""
    best = cur = 0
    for f in np.asarray(flags, dtype=bool):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best
