"""코호트별 robust z와 z 집계 — `cohort_stats`/`apply_z` 위의 편의 층.

`cohort_stats` + `apply_z`는 **한 코호트**를 다루는 원시 연산이다. 실전에서는
카테고리마다 코호트를 나눠 돌리고, 카테고리 안에서 MAD가 죽는 경우를 막고,
피쳐 여럿의 z를 한 점수로 모으는 일이 반복된다 — 그 반복을 여기 담는다.

세 가지가 얹힌다:

1. **카테고리별 통계** — 중심과 스케일을 그룹마다 따로 낸다. 카테고리 고유
   성질(예: cnr이 원래 1.5인 카테고리)이 자동으로 흡수된다. "원래 1.5인데
   지금도 1.5"는 이상이 아니다. **카테고리 예외 목록을 만들지 않는 이유가
   이것이다** (spec §11-1).
2. **pooled MAD 폴백** — 그룹 MAD가 죽으면(이산 피쳐, 0에 몰린 피쳐) 분모가
   0에 붙어 z가 폭발한다. 전체 pooled MAD의 `pooled_mad_frac`을 분모 하한으로
   깐다. **피쳐를 빼서 풀 문제가 아니라 분모 하한으로 풀 문제다** (spec §4.2).
3. **z 상한** — 최종 방어선 `Z_CAP`. 포화값이므로 **z 크기를 심각도로 읽으면
   안 된다** (spec §4.4).

집계는 다중비교 방어다: 피쳐가 수십 개면 하나쯤은 늘 튄다. `z_top2`/`z_top3`는
"k개 피쳐가 동시에 t를 넘었는가"를 묻는다. **어느 것을 판정에 쓸지는 정하지
않는다 — 전부 계산해서 컬럼으로 남긴다** (spec §4.4).

코호트 축(group_cols)은 여전히 사용자가 정한다 — 라이브러리는 이름으로만 동작.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .api import apply_z, cohort_stats
from .errors import CdqcError
from .features.registry import BY_NAME, REGISTRY
from .geometry import circular_residual_deg180
from .params import Params

DEFAULT_TOP_K = (1, 2, 3)
AGG_HOWS = ("kth", "mean")   # k번째로 큰 값 / 상위 k개 평균 — 둘 다 계산한다


def _agg_name(k: int, how: str = "kth") -> str:
    return "z_max" if k == 1 else f"z_top{k}_{how}"


def agg_columns(ks: tuple[int, ...] = DEFAULT_TOP_K) -> list[str]:
    """aggregate_z가 만드는 집계 컬럼 이름 (n_z_valid 제외)."""
    out = []
    for k in ks:
        if k == 1:
            out.append("z_max")
        else:
            out += [_agg_name(k, h) for h in AGG_HOWS]
    return out


def _z_feature_names(z_df: pd.DataFrame,
                     feature_cols: list[str] | None) -> list[str]:
    """집계에 넣을 피쳐 이름 — 기본은 registry 기본 활성 + z_ 컬럼 존재."""
    if feature_cols is None:
        return [f.name for f in REGISTRY
                if f.enabled_default and f"z_{f.name}" in z_df.columns]
    unknown = set(feature_cols) - {f.name for f in REGISTRY}
    if unknown:
        raise CdqcError("E-ARG-05", f"{sorted(unknown)}")
    return [n for n in feature_cols if f"z_{n}" in z_df.columns]


# ================================================================ 코호트 z

def floor_stats(stats: dict, pooled: dict,
                params: Params | None = None) -> dict:
    """코호트 통계의 MAD에 pooled 하한을 깐다 (cohort_stats 반환형 그대로).

    그룹 MAD가 죽으면(이산 피쳐, 값이 0에 몰린 피쳐) 분모가 0에 붙어 z가
    폭발한다. **피쳐를 빼지 않고** 분모 하한으로 막는다 (spec §4.2). 최종
    분모는 네 하한의 최댓값이다:

        max(1.4826·MAD_그룹,
            pooled_mad_frac · 1.4826·MAD_pooled,   ← 코호트 전체의 산포
            rel_scale_floor · |중앙값|,             ← 중심이 큰 피쳐 방어
            abs_scale_floor)                        ← 절대 바닥

    `rel_scale_floor`가 없으면 `cd_nm`처럼 값이 수십 nm인 피쳐가 MAD만 작을 때
    폭발한다. (log 피쳐는 |중앙값|이 로그 공간 값이라 상대 하한이 대략
    "몇 % 변화"로 읽힌다.) `mad_floors[피쳐]`는 apply_z가 마지막에 한 번 더
    적용하므로 여기서는 다루지 않는다.

    원형 피쳐(angle_median)는 각도 원점이 임의이므로 |중앙값| 상대 하한을
    적용하지 않는다. 원형 MAD와 pooled/절대/피쳐별 하한은 그대로 적용한다.

    cohort_z가 내부에서 쓰는 것과 같은 연산이다 — 고정 통계를 저장했다가
    나중에 apply_z로 쓰는 경우(캘리브레이션 재사용, injection_test)에도 같은
    방어를 받도록 공개한다.
    """
    params = params or Params()
    pooled_mad = {n: st["mad"] for n, st in pooled.get("features", {}).items()}
    out = {"n": stats.get("n", 0), "modes": dict(stats.get("modes", {})),
           "features": {}}
    for name, st in stats.get("features", {}).items():
        pm = pooled_mad.get(name, np.nan)
        med, mad = st["median"], st["mad"]
        # Angular zero is arbitrary: |median| cannot set a circular scale.
        relative_center = (abs(med) if np.isfinite(med)
                           and BY_NAME[name].period is None else 0.0)
        scale_floor = max(
            params.pooled_mad_frac * 1.4826 * (pm if np.isfinite(pm) else 0.0),
            params.rel_scale_floor * relative_center,
            params.abs_scale_floor)
        mad_floor = scale_floor / 1.4826
        mad = mad_floor if not np.isfinite(mad) else max(float(mad), mad_floor)
        out["features"][name] = {"median": med, "mad": float(mad)}
    return out


def directed_z_cols(df: pd.DataFrame, exclude=()) -> list[str]:
    """directed z 컬럼 이름 목록 (z_*, zs_*). exclude에 든 이름은 뺀다."""
    ex = set(exclude)
    return [c for c in df.columns
            if c not in ex and (c.startswith("z_") or c.startswith("zs_"))]


def clip_directed_z(df: pd.DataFrame, cols, cap: float) -> pd.DataFrame:
    """directed z 컬럼을 ±cap으로 자른다 (제자리 수정 후 같은 df 반환).

    MAD가 0에 붙는 경우의 최종 방어선. **포화값이므로 z 크기를 심각도로
    읽으면 안 된다** — 탐지 여부만 의미가 있다.
    """
    for c in cols:
        if c in df.columns:
            df[c] = np.clip(df[c].to_numpy(dtype=np.float64), -cap, cap)
    return df


def cohort_z(df: pd.DataFrame, group_cols: list[str] | None = None,
             base_mask=None, feature_cols: list[str] | None = None,
             params: Params | None = None, return_stats: bool = False):
    """그룹별 코호트 통계로 directed z를 붙인다 (원본 df는 수정하지 않음).

    group_cols: 코호트를 자르는 축 (예: ["category_id"], ["recipe_id",
        "category_id"]). None이면 전체를 한 코호트로.
    base_mask: 통계를 낼 행 (bool 배열/Series). None이면 df 전체.
        **정상으로 확인된 행만 넣는 것을 권장** — 불량률이 높으면 통계가
        오염된다 (trim_frac이 방어하지만 한계가 있다).
    feature_cols: None이면 registry의 z/match 피쳐 중 df에 있는 것 전부.

    분모의 하한은 `floor_stats` 참조. 기저 통계가 없는 그룹(base 행 0개)은
    pooled 통계를 쓴다.

    return_stats=True면 `(z_df, {그룹키: 하한이 적용된 stats})`를 준다 —
    같은 기준으로 나중에 `apply_z`/`injection_test`를 돌리려면 이 통계를
    저장해야 한다 (주입 케이스로 통계를 다시 내면 주입이 흡수된다).
    """
    params = params or Params()
    if group_cols is not None:
        missing = [name for name in group_cols if name not in df.columns]
        if not group_cols or missing:
            raise CdqcError("E-ARG-06", f"group_cols must name existing columns; missing={missing}")
        null_counts = df[group_cols].isna().sum()
        null_counts = {name: int(count) for name, count in null_counts.items() if count}
        if null_counts:
            raise CdqcError("E-ARG-06", f"cohort group keys contain missing values: {null_counts}")
    base = df if base_mask is None else df[np.asarray(base_mask, dtype=bool)]
    if len(base) == 0:
        raise CdqcError("E-ARG-04", "base_mask가 고른 행이 0개")

    pooled = cohort_stats(base, feature_cols, params)

    before = set(df.columns)
    used: dict = {}
    if group_cols is None:
        used[None] = floor_stats(pooled, pooled, params)
        out = apply_z(df, used[None], params)
    else:
        base_g = {k: g for k, g in base.groupby(group_cols, sort=False)}
        # 위치 인덱스로 잘랐다 붙인다 — 입력 인덱스에 중복이 있어도 행 순서가
        # 보존된다 (여러 시퀀스를 ignore_index 없이 concat한 프레임이 흔하다)
        work = df.reset_index(drop=True)
        parts = []
        for key, g in work.groupby(group_cols, sort=False):
            b = base_g.get(key)
            st = floor_stats(pooled if b is None or len(b) == 0
                             else cohort_stats(b, feature_cols, params),
                             pooled, params)
            used[key] = st
            parts.append(apply_z(g, st, params))
        out = pd.concat(parts).sort_index()
        out.index = df.index

    if "angle" in df.columns and "angle_resid_cohort" not in df.columns:
        out["angle_resid_cohort"] = _cohort_angle_resid(df, group_cols)
    out = clip_directed_z(out, directed_z_cols(out, exclude=before),
                          float(params.z_cap))
    return (out, used) if return_stats else out


def _cohort_angle_resid(df: pd.DataFrame,
                        group_cols: list[str] | None) -> np.ndarray:
    """코호트(그룹) 원형 중앙값 대비 각도 잔차 — 시퀀스 전체 회전도 잡힌다.

    `angle_resid_seq`(extract_l3가 내는 시퀀스 중심 잔차)와 짝이다. 시퀀스가
    통째로 돌면 seq 쪽은 0이 되고 cohort 쪽만 반응한다 — 어느 층에서 잡는 게
    나은지는 데이터가 쌓여야 알므로 둘 다 남긴다.
    """
    a = df["angle"].to_numpy(dtype=np.float64)
    out = np.full(len(df), np.nan)
    if group_cols is None:
        return circular_residual_deg180(a)
    pos = np.arange(len(df))
    for _, idx in df.reset_index(drop=True).groupby(group_cols, sort=False).indices.items():
        out[pos[idx]] = circular_residual_deg180(a[idx])
    return out


# ================================================================ 집계

def aggregate_z(z_df: pd.DataFrame, feature_cols: list[str] | None = None,
                ks: tuple[int, ...] = DEFAULT_TOP_K,
                params: Params | None = None) -> pd.DataFrame:
    """행별 z 집계 — `z_max`, `z_top{k}_kth`, `z_top{k}_mean`, `n_z_valid`.

    다중비교 방어의 두 정의를 **둘 다** 계산해 남긴다 (어느 것을 판정에 쓸지는
    판정 단계에서 정한다):

    - `z_top{k}_kth` = **k번째로 큰** directed z. 임계 t를 넘었다는 것은 곧
      "k개 이상의 피쳐가 동시에 t를 넘었다"는 뜻이다. 피쳐 하나가 튄 것만으로는
      안 걸리는 대신, **반응 피쳐가 k개 미만인 실패는 놓친다** (예: CD 하나를
      자기 중점 기준으로 회전시키면 obliquity와 angle_resid_seq 둘만 반응해
      k=3에서 사라진다).
    - `z_top{k}_mean` = 상위 k개의 평균. 놓치는 실패는 적지만, 하나가 상한까지
      포화하면 평균이 여전히 크므로 다중비교 방어가 온전하지 않다.

    `z_max`는 k=1이라 두 정의가 같고 `top_feature`의 top_z와도 같다.

    **NaN 주의**: np.sort는 NaN을 맨 뒤(최댓값 자리)로 보낸다. −inf로 치환하지
    않으면 상위 k개가 전부 NaN이 되어 결과가 NaN이 된다 — CD 1개짜리
    카테고리(관계 피쳐가 전부 NaN)에서 실제로 났던 버그. 유효 피쳐 수를
    `n_z_valid`로 같이 기록하니 z 값을 볼 때 함께 봐야 한다.

    반환 인덱스는 z_df와 동일.
    """
    names = _z_feature_names(z_df, feature_cols)
    if not names:
        raise CdqcError("E-ARG-04", "z_ 컬럼이 없음 — apply_z/cohort_z 먼저")
    Z = np.column_stack([z_df[f"z_{n}"].to_numpy(dtype=np.float64) for n in names])
    valid = np.isfinite(Z)
    n_valid = valid.sum(axis=1)
    Zf = np.where(valid, Z, -np.inf)
    srt = np.sort(Zf, axis=1)[:, ::-1]          # 내림차순
    out = {"n_z_valid": n_valid.astype(int)}
    for k in ks:
        if k < 1:
            raise CdqcError("E-ARG-06", f"k={k}")
        kk = min(k, srt.shape[1])
        kth = np.where(n_valid >= k, srt[:, kk - 1], np.nan)
        out[_agg_name(k, "kth")] = np.where(np.isfinite(kth), kth, np.nan)
        if k > 1:
            mean = np.where(n_valid >= k, srt[:, :kk].mean(axis=1), np.nan)
            out[_agg_name(k, "mean")] = np.where(np.isfinite(mean), mean, np.nan)
    return pd.DataFrame(out, index=z_df.index)


# ================================================================ 절대 임계

def mode_flags(df: pd.DataFrame, cols: list[str],
               group_cols: list[str] | None = None,
               base_mask=None) -> pd.DataFrame:
    """그룹 최빈값과 다르면 플래그 — `flag_mode_<col>` + `n_mode_flags`.

    `n_cd` 같은 **본질적으로 이산이고 규칙적인** 값에는 z가 맞지 않는다.
    카테고리 안에서 늘 같은 값이면 MAD가 0이라 분모가 `mad_floors`가 되고,
    그러면 "CD 1개 누락 = z 2.0"처럼 **하한이 그대로 답이 된다** — z가 정보를
    담고 있지 않다는 뜻이다. 최빈값 불일치는 임계도 필요 없고 해석도 명확하다.

    base_mask: 최빈값을 낼 행 (정상으로 확인된 행). None이면 df 전체.
    """
    if not cols:
        raise CdqcError("E-ARG-04", "cols가 비어 있음")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise CdqcError("E-ARG-01", f"컬럼 없음: {sorted(missing)}")
    base = df if base_mask is None else df[np.asarray(base_mask, dtype=bool)]
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if group_cols is None:
            m = base[c].mode()
            modes = pd.Series(m.iloc[0] if len(m) else np.nan, index=df.index)
        else:
            table = base.groupby(group_cols, sort=False)[c].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan)
            keys = (df[group_cols[0]] if len(group_cols) == 1
                    else pd.MultiIndex.from_frame(df[group_cols]))
            modes = pd.Series(table.reindex(keys).to_numpy(), index=df.index)
        out[f"flag_mode_{c}"] = (df[c].to_numpy() != modes.to_numpy()) & \
            modes.notna().to_numpy()
    out["n_mode_flags"] = out.to_numpy(dtype=bool).sum(axis=1).astype(int)
    return out


def abs_flags(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """물리 단위 절대 임계 플래그 — `flag_<피쳐>` + `n_abs_flags`.

    z가 성립하지 않거나(값이 0에 몰려 MAD가 구조적으로 죽는) 물리 단위 임계가
    해석하기 쉬운 피쳐용이다 (예: `bdist_s` 1.0nm, `obliquity` 5°). 그런
    피쳐도 **z는 z대로 계산해서 기록**하고, 어느 쪽을 쓸지는 판정 단계에서
    정한다.

    **임계값에 기본값을 두지 않는다** — 공정 스펙 근거가 있는 값은 사용자만
    안다 (spec §4.4). 비교 방향은 registry의 worse_when을 따른다:
    high → x > t, low → x < t, both → |x| > t.
    """
    if not thresholds:
        raise CdqcError("E-ARG-04", "thresholds가 비어 있음 — 임계값은 사용자 입력")
    unknown = set(thresholds) - {f.name for f in REGISTRY}
    if unknown:
        raise CdqcError("E-ARG-05", f"{sorted(unknown)}")
    missing = [n for n in thresholds if n not in df.columns]
    if missing:
        raise CdqcError("E-ARG-01", f"컬럼 없음: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    for name, t in thresholds.items():
        x = df[name].to_numpy(dtype=np.float64)
        ww = BY_NAME[name].worse_when
        if ww == "low":
            hit = x < float(t)
        elif ww == "both":
            hit = np.abs(x) > float(t)
        else:                                    # high 및 방향 미지정
            hit = x > float(t)
        out[f"flag_{name}"] = np.where(np.isfinite(x), hit, False)
    out["n_abs_flags"] = out.to_numpy(dtype=bool).sum(axis=1).astype(int)
    return out
