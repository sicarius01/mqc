"""검증·진단 연산 — 주입 테스트, 궤적 점검, 카테고리 요약.

**이 모듈의 어떤 함수도 피쳐를 빼거나 무력화하지 않는다.** 계산해서 기록만
한다 (spec §11-2). 한 데이터 세트에서 "이 피쳐는 변동이 없어 보인다"가 나와도,
레시피와 구조가 수십 가지인데 그중 하나를 보고 피쳐의 유효성을 단정할 수 없다
— 다른 레시피에서는 좌표가 이미지에서 독립적으로 탐색될 수 있고, 그러면 같은
피쳐가 핵심 탐지기가 된다.

- `inject_coords` / `injection_test`: 정상 좌표에 **알려진 크기**의 오류를
  주입하고, 코호트 통계는 고정한 채 z가 얼마나 오르는지 잰다. 합성 이미지가
  아니라 **실제 이미지**에 주입할 수 있는 형태다 (extract 콜러블을 받는다).
- `trajectory_check`: 좌표 궤적이 일렬인지, 측정 순서와 진행 방향이 맞는지.
- `category_summary`: 카테고리별 진단 지표 표.

**z 크기를 심각도로 읽으면 안 된다**: 이동량이 커지면 오히려 탐색 창을 벗어나
delta가 원래 엣지를 놓쳐 z가 떨어진다. 탐지 여부만 판단할 것.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .api import apply_z, extract_l2, extract_l3, top_feature
from .errors import CdqcError
from .normalize import (DEFAULT_TOP_K, agg_columns, aggregate_z,
                        clip_directed_z, directed_z_cols)
from .params import Params

INJECTIONS = ("shift_one", "rotate_one", "rotate_frame", "swap",
              "all_shift", "drop_one")


# ================================================================ 좌표 주입

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def inject_coords(S, E, kind: str, magnitude: float = 0.0,
                  index: int | None = None, center=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """좌표에 알려진 실패를 주입. 반환 (S2, E2, affected) — 순수 함수.

    `affected`는 **출력 행 기준** bool 배열이다. `drop_one`은 행 자체가
    사라지므로 전부 False — 그게 요점이다. 없어진 CD는 z를 낼 행이 없어
    CD 레벨로는 원리적으로 볼 수 없고, L2 `n_cd`만 잡는다.

    | kind | 내용 | magnitude 단위 |
    |---|---|---|
    | `shift_one` | CD 하나의 S를 세그먼트 축(+u) 방향으로 이동 | px |
    | `rotate_one` | CD 하나를 **자기 중점** 기준 회전 (개별 각도 오류) | deg |
    | `rotate_frame` | **시퀀스 전체**를 `center` 기준 회전 (기준 각도 오설정) | deg |
    | `swap` | 이웃 CD 둘의 E를 맞바꿈 (대응 오류) | (무시) |
    | `all_shift` | **시퀀스 전체**를 축 방향으로 평행이동 | px |
    | `drop_one` | CD 하나 삭제 | (무시) |

    **회전 두 종류는 서로 다른 실패이고 반응 피쳐가 다르다.** `rotate_one`은
    끝점이 궤적 **진행 방향**으로 움직여서 법선 잔차(`s_resid`)가 원리적으로
    못 본다 — `obliquity`와 `angle_resid_seq` 둘만 반응하는 2피쳐 신호라
    `z_top3_kth`에서는 사라진다. `rotate_frame`은 모든 CD가 다른 자리로 옮겨져
    `delta`·`s_resid`·`bdist`가 전부 반응한다.

    `all_shift`는 모든 CD가 똑같이 밀리므로 이웃 대비 잔차
    (`s_resid`/`dstep`/`cd_resid`)가 전부 0이 된다 — 시퀀스 내 잔차의
    구조적 맹점이고, 그래서 L2 요약을 코호트와 비교해야 잡힌다.

    index 기본값은 시퀀스 중앙, `center` 기본값은 전체 좌표의 무게중심
    (둘 다 결정론적 — 난수를 쓰지 않는다). 이미지 중심을 기준으로 돌리려면
    `center=(W/2, H/2)`를 넘긴다.
    """
    S = np.asarray(S, dtype=np.float64).copy()
    E = np.asarray(E, dtype=np.float64).copy()
    if S.shape != E.shape or S.ndim != 2 or S.shape[1] != 2:
        raise CdqcError("E-ARG-01", f"S{S.shape} vs E{E.shape}, (n,2) 필요")
    n = len(S)
    if n == 0:
        raise CdqcError("E-ARG-04", "빈 시퀀스")
    if kind not in INJECTIONS:
        raise CdqcError("E-ARG-06", f"kind={kind!r} — {INJECTIONS} 중 하나")
    i = n // 2 if index is None else int(index)
    if not (0 <= i < n):
        raise CdqcError("E-ARG-06", f"index={i}, n={n}")

    u = _unit(E - S)
    aff = np.zeros(n, dtype=bool)

    if kind == "shift_one":
        S[i] = S[i] + u[i] * float(magnitude)
        aff[i] = True
    elif kind in ("rotate_one", "rotate_frame"):
        th = np.radians(float(magnitude))
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        if kind == "rotate_one":
            ctr = (S[i] + E[i]) / 2.0
            S[i] = ctr + R @ (S[i] - ctr)
            E[i] = ctr + R @ (E[i] - ctr)
            aff[i] = True
        else:
            ctr = (np.asarray(center, dtype=np.float64) if center is not None
                   else np.concatenate([S, E]).mean(axis=0))
            S = (S - ctr) @ R.T + ctr
            E = (E - ctr) @ R.T + ctr
            aff[:] = True
    elif kind == "swap":
        if n < 2:
            raise CdqcError("E-ARG-04", "swap은 CD 2개 이상 필요")
        j = i + 1 if i + 1 < n else i - 1
        E[[i, j]] = E[[j, i]]
        aff[i] = aff[j] = True
    elif kind == "all_shift":
        S = S + _unit(u.mean(axis=0)) * float(magnitude)
        E = E + _unit(u.mean(axis=0)) * float(magnitude)
        aff[:] = True
    elif kind == "drop_one":
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        S, E, aff = S[keep], E[keep], aff[keep]
    return S, E, aff


# ================================================================ 주입 테스트

def _top_name(top: pd.DataFrame) -> str:
    """top_feature 결과에서 top_z가 가장 큰 행의 피쳐 이름 (없으면 빈 문자열)."""
    if not len(top):
        return ""
    z = top["top_z"].to_numpy(dtype=np.float64)
    if not np.isfinite(z).any():
        return ""
    return str(top["top_feature"].iloc[int(np.nanargmax(z))])



def injection_test(S, E, extract, l3_stats: dict,
                   cases: list[tuple[str, float]] | None = None,
                   l2_stats: dict | None = None,
                   feature_cols: list[str] | None = None,
                   ks: tuple[int, ...] = DEFAULT_TOP_K,
                   primary: str = "z_top3_kth",
                   params: Params | None = None) -> pd.DataFrame:
    """정상 시퀀스에 실패를 주입하고 L3/L2 집계 z가 얼마나 오르는지 잰다.

    extract: `extract(S, E) -> l3 DataFrame` 콜러블. 이미지·px_nm·Params를
        클로저로 잡아 **사용자가** 만든다 — 그래서 합성이든 실제 이미지든 같은
        코드로 돌고, cdqc가 파일을 만지지 않는다는 계약도 유지된다.
    l3_stats / l2_stats: `cohort_stats`로 미리 낸 **고정** 코호트 통계.
        주입된 시퀀스로 통계를 다시 내면 주입이 흡수되어 아무것도 안 보인다.
        l2_stats가 None이면 L2 컬럼은 생략된다.
    cases: [(kind, magnitude), ...]. None이면 INJECTIONS의 대표 강도.
    primary: `l3_rise`/`l2_rise`를 계산할 기준 집계 컬럼.

    반환 (한 행 = 한 케이스, **첫 행 kind="none"이 기준선**):
        kind, magnitude, n_cd, n_affected, l3_n_z_valid,
        l3_<집계> …, l3_top_feature, l3_rise,
        l2_<집계> …, l2_top_feature, l2_rise
    집계는 `aggregate_z`가 내는 것 전부 (`z_max`, `z_top{k}_kth`,
    `z_top{k}_mean`).

    `*_top_feature`는 **무엇이 걸렸는지** 알려준다 — 집계 z만 보면 주입이
    의도한 경로로 잡혔는지 알 수 없다. `*_rise`는 기준선 대비 상승분이며,
    **같은 행 안에서 짝지어 계산된다**: 케이스별 값을 따로 집계한 뒤 빼면
    (원본 스크립트가 `pivot_table(aggfunc="median")`으로 그렇게 했다)
    서로 다른 행의 중앙값을 빼게 되어 내부 모순이 생긴다.

    L3 집계는 affected 행의 최댓값이다. affected가 없으면(`drop_one`) 전체
    행의 최댓값 — 그 경우 값이 안 오르는 것 자체가 결과다.
    """
    params = params or Params()
    if cases is None:
        cases = [("shift_one", 5.0), ("rotate_one", 2.0),
                 ("rotate_frame", 1.0), ("swap", 0.0),
                 ("all_shift", 5.0), ("drop_one", 0.0)]
    agg_cols = agg_columns(ks)
    if primary not in agg_cols:
        raise CdqcError("E-ARG-06", f"primary={primary!r} — {agg_cols}")

    cap = float(params.z_cap)

    def _z(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
        """cohort_z와 같은 방어(z 상한)를 받은 directed z."""
        before = set(df.columns)
        out = apply_z(df, stats, params)
        return clip_directed_z(out, directed_z_cols(out, exclude=before), cap)

    def measure(S2, E2, aff) -> dict:
        l3 = extract(S2, E2)
        z3 = _z(l3, l3_stats)
        a3 = aggregate_z(z3, feature_cols, ks, params)
        sel = a3[aff] if np.any(aff) else a3
        row: dict = {"n_cd": int(len(l3)), "n_affected": int(np.sum(aff))}
        for col in agg_cols:
            v = sel[col].to_numpy(dtype=np.float64) if len(sel) else np.array([])
            row[f"l3_{col}"] = float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
        nv = a3["n_z_valid"].to_numpy()
        row["l3_n_z_valid"] = int(np.median(nv)) if len(nv) else 0
        row["l3_top_feature"] = _top_name(
            top_feature(z3[aff] if np.any(aff) else z3, feature_cols))
        if l2_stats is not None:
            z2 = _z(extract_l2(l3, None, params), l2_stats)
            a2 = aggregate_z(z2, feature_cols, ks, params)
            for col in agg_cols:
                row[f"l2_{col}"] = float(a2[col].iloc[0])
            row["l2_top_feature"] = _top_name(top_feature(z2, feature_cols))
        return row

    S0 = np.asarray(S, dtype=np.float64)
    E0 = np.asarray(E, dtype=np.float64)
    rows = [{"kind": "none", "magnitude": 0.0,
             **measure(S0, E0, np.zeros(len(S0), dtype=bool))}]
    for kind, mag in cases:
        S2, E2, aff = inject_coords(S0, E0, kind, mag)
        rows.append({"kind": kind, "magnitude": float(mag),
                     **measure(S2, E2, aff)})
    out = pd.DataFrame(rows)
    # rise는 같은 행 안에서 기준선을 빼서 만든다 (케이스별로 따로 집계한 뒤
    # 빼면 서로 다른 행의 중앙값을 빼게 된다 — 원본 pivot 버그)
    for lvl in ("l3", "l2"):
        col = f"{lvl}_{primary}"
        if col in out.columns:
            out[f"{lvl}_rise"] = out[col] - out[col].iloc[0]
    return out


# ================================================================ σ 자동 선택

def select_grad_sigma(img: np.ndarray, S, E, px_nm: float,
                      sigmas: tuple[float, ...] | None = None,
                      tol_px: float = 0.25, min_valid: float = 0.8,
                      params: Params | None = None) -> dict:
    """delta가 σ에 흔들리지 않기 시작하는 **최소** grad_sigma_px를 고른다.

    σ가 작으면 노이즈 피크를 잡고, 크면 엣지를 뭉갠다. 어느 쪽인지는 이미지
    조건마다 다르므로 사람이 정할 게 아니다 (spec §11-3): σ를 훑으면서 **이웃
    σ끼리 답이 일치하기 시작하는 지점**을 찾는다.

    - 일치도 = 이웃 σ 간 delta 차이의 **p90** (px). 중앙값은 너무 관대하다 —
      절반이 맞으면 통과해버린다.
    - `min_valid` 미만으로만 유효 delta가 나오는 σ는 후보에서 뺀다.
    - 어느 σ도 `tol_px` 아래로 못 내려가면 일치도가 가장 좋은 σ를 준다.

    반환: {"sigma", "agreement"(σ별 p90 차이), "valid_frac", "sigmas",
           "converged"(tol 이하를 만났는가)}.
    **값을 그대로 쓰지 말고 `Params(grad_sigma_px=...)`에 넣는 것은 사용자가
    한다** — 코호트마다 다른 σ를 쓰면 delta 통계가 섞인다는 점에 주의.

    비용: σ 개수만큼 extract_l3를 돈다. 캘리브레이션 단계에서 한 번 쓰는 용도다.
    """
    import dataclasses

    params = params or Params()
    sigmas = tuple(sigmas) if sigmas else tuple(params.delta_sigma_sweep)
    if len(sigmas) < 2:
        raise CdqcError("E-ARG-06", "sigmas는 2개 이상")

    D, V = [], []
    for sg in sigmas:
        f = extract_l3(img, S, E, px_nm,
                       params=dataclasses.replace(params, grad_sigma_px=sg))
        v = f["delta_s"].to_numpy(dtype=np.float64) / float(px_nm)
        D.append(v)
        V.append(float(np.isfinite(v).mean()))
    M, V = np.vstack(D), np.array(V)

    ok = [i for i in range(len(sigmas)) if V[i] >= min_valid]
    agree = np.full(len(sigmas), np.nan)
    for a, b in zip(ok, ok[1:]):
        d = np.abs(M[a] - M[b])
        if np.isfinite(d).any():
            agree[a] = float(np.nanpercentile(d, 90))
    chosen, converged = None, False
    for i in ok[:-1]:
        if np.isfinite(agree[i]) and agree[i] < tol_px:
            chosen, converged = sigmas[i], True
            break
    if chosen is None and np.isfinite(agree).any():
        chosen = sigmas[int(np.nanargmin(agree))]
    return {"sigma": chosen, "agreement": agree, "valid_frac": V,
            "sigmas": sigmas, "converged": converged}


# ================================================================ 궤적 진단

def trajectory_check(pts, order=None) -> dict:
    """좌표 궤적의 형태 진단 — **기록용**. 판정도, 피쳐 무력화도 하지 않는다.

    - `linearity`: PCA 1축이 설명하는 분산 비율. 1이면 완전 일렬.
    - `monotonic`: 측정 순서와 주축 위 위치의 순위 상관 (Spearman).
      ±1이면 순서대로 한 방향 진행. 0 근처면 순서가 뒤섞였거나 좌표가 애초에
      궤적을 이루지 않는다.
    - `spread_px`: 주축 방향 표준편차 (궤적 길이 스케일).
    - `resid_rms_px`: 주축 직선 대비 잔차 RMS.

    유효 점이 3개 미만이면 전부 NaN.
    """
    P = np.asarray(pts, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise CdqcError("E-ARG-01", f"pts{P.shape}, (n,2) 필요")
    ok = np.isfinite(P).all(axis=1)
    P = P[ok]
    nan = {"linearity": np.nan, "monotonic": np.nan, "spread_px": np.nan,
           "resid_rms_px": np.nan, "n_pts": int(len(P))}
    if len(P) < 3:
        return nan

    c = P - P.mean(axis=0)
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    var = sv ** 2
    total = float(var.sum())
    if total <= 0:
        return nan
    t = c @ vt[0]                       # 주축 위 좌표
    o = (np.arange(len(P), dtype=np.float64) if order is None
         else np.asarray(order, dtype=np.float64)[ok])

    from scipy.stats import rankdata
    ro, rt = rankdata(o), rankdata(t)
    mono = (np.nan if ro.std() == 0 or rt.std() == 0
            else float(np.corrcoef(ro, rt)[0, 1]))

    return {"linearity": float(var[0] / total),
            "monotonic": mono,
            "spread_px": float(np.std(t)),
            "resid_rms_px": float(np.sqrt(np.mean((c @ vt[1]) ** 2))),
            "n_pts": int(len(P))}


# ================================================================ 카테고리 요약

# 진단용 중앙값 컬럼: 출력 이름 → l3 컬럼
_MED_COLS = {"cnr_med_s": "cnr_s", "cnr_med_e": "cnr_e",
             "rise_med_s": "rise_s", "rise_med_e": "rise_e",
             "bdist_ratio_med": "bdist_ratio",
             "bdist_med_s": "bdist_s", "bdist_med_e": "bdist_e",
             "bgrad_med_s": "bgrad_s", "bgrad_med_e": "bgrad_e",
             "label_runs_med": "label_runs",
             "delta_med_s": "delta_s", "delta_med_e": "delta_e",
             "delta_B_med_s": "delta_B_s", "delta_B_med_e": "delta_B_e",
             "delta_scatter_med_s": "delta_scatter_s",
             "delta_scatter_med_e": "delta_scatter_e"}
_RESID_COLS = ("s_resid", "e_resid", "cd_resid")   # px 환산 MAD로 낸다


def category_summary(l3_df: pd.DataFrame, group_cols: list[str],
                     agg_df: pd.DataFrame | None = None,
                     traj_by: str | None = "image_id") -> pd.DataFrame:
    """카테고리별 진단 지표 표 — **계산해서 기록만 한다**.

    어느 피쳐가 어느 조건에서 일하는지는 여러 레시피·구조를 돌려 누적한 뒤에
    판단할 문제다. 여기 값이 이상하다고 해서 피쳐를 빼거나 NaN 처리하는 자동
    규칙을 만들지 않는다 (spec §11-2). 경고를 출력하는 건 자유지만 계산 결과를
    바꾸지 않는다.

    컬럼: `n_cd`, `n_img`, `cd_per_img`, 궤적 진단
    (`linearity`/`monotonic`/`traj_resid_rms_px`), 잔차 산포
    (`s_resid_mad_px`/`e_resid_mad_px`/`cd_resid_mad_px` — **px 환산**이라
    배율에 무관), 피쳐 중앙값(`cnr_med_s/e`, `rise_med_s/e`,
    `bdist_ratio_med`, `bdist_med_s/e`, `bgrad_med_s/e`, `label_runs_med`,
    `delta_med_s/e`, `delta_B_med_s/e`, `delta_scatter_med_s/e`), 그리고
    agg_df를 주면 `n_z_valid_med`/`z_top3_kth_p99`/`z_top3_mean_p99`.

    **궤적 진단은 트래젝토리 하나씩 계산해서 중앙값을 낸다.** `traj_by`(기본
    "image_id")가 group_cols에 없으면 그 컬럼으로 한 번 더 쪼갠다 — 여러
    이미지의 좌표를 한 덩어리로 PCA에 넣으면 궤적이 겹쳐 linearity가
    무의미해진다. `monotonic`은 진행 방향의 부호가 임의라 **절댓값**의
    중앙값을 낸다.

    S 궤적을 쓰려면 extract_l3의 `s_x`/`s_y` 캐리어가 필요하다. 없으면
    `mid_x`/`mid_y`로 대체하고, 그것도 없으면 NaN.

    agg_df: `aggregate_z` 결과 (l3_df와 같은 인덱스·길이).
    """
    if agg_df is not None and len(agg_df) != len(l3_df):
        raise CdqcError("E-ARG-01", f"agg_df {len(agg_df)} vs l3_df {len(l3_df)}")
    df = l3_df if agg_df is None else l3_df.join(
        agg_df[[c for c in agg_df.columns if c not in l3_df.columns]])

    def med(g: pd.DataFrame, col: str) -> float:
        if col not in g:
            return np.nan
        x = g[col].to_numpy(dtype=np.float64)
        return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan

    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n_img = int(g[traj_by].nunique()) if traj_by in g.columns else 1
        row: dict = {**dict(zip(group_cols, key)), "n_cd": int(len(g)),
                     "n_img": n_img, "cd_per_img": len(g) / max(n_img, 1)}
        row.update(_traj_metrics(g, group_cols, traj_by))
        # 잔차는 nm이라 배율에 딸린다 — px로 환산해야 데이터 세트 간 비교 가능
        px = (float(g["px_nm"].iloc[0]) if "px_nm" in g and len(g) else np.nan)
        for c in _RESID_COLS:
            row[f"{c}_mad_px"] = np.nan
            if c in g and np.isfinite(px) and px > 0:
                r = g[c].to_numpy(dtype=np.float64)
                r = r[np.isfinite(r)]
                if len(r):
                    row[f"{c}_mad_px"] = float(
                        np.median(np.abs(r - np.median(r))) / px)
        for out_col, src in _MED_COLS.items():
            row[out_col] = med(g, src)
        row["n_z_valid_med"] = med(g, "n_z_valid")
        for c in ("z_top3_kth", "z_top3_mean"):
            v = (g[c].to_numpy(dtype=np.float64) if c in g
                 else np.array([], dtype=np.float64))
            v = v[np.isfinite(v)]
            row[f"{c}_p99"] = float(np.quantile(v, 0.99)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _traj_metrics(g: pd.DataFrame, group_cols: list[str],
                  traj_by: str | None) -> dict:
    """그룹 안의 트래젝토리별 진단을 중앙값으로 모은다."""
    if {"s_x", "s_y"} <= set(g.columns):
        cols = ["s_x", "s_y"]
    elif {"mid_x", "mid_y"} <= set(g.columns):
        cols = ["mid_x", "mid_y"]
    else:
        return {"linearity": np.nan, "monotonic": np.nan,
                "traj_resid_rms_px": np.nan}

    if traj_by and traj_by in g.columns and traj_by not in group_cols:
        parts = [gi for _, gi in g.groupby(traj_by, sort=False)]
    else:
        parts = [g]
    lin, mono, rms = [], [], []
    for gi in parts:
        tc = trajectory_check(gi[cols].to_numpy())
        if np.isfinite(tc["linearity"]):
            lin.append(tc["linearity"])
            rms.append(tc["resid_rms_px"])
        if np.isfinite(tc["monotonic"]):
            mono.append(abs(tc["monotonic"]))   # 진행 방향 부호는 임의
    return {"linearity": float(np.median(lin)) if lin else np.nan,
            "monotonic": float(np.median(mono)) if mono else np.nan,
            "traj_resid_rms_px": float(np.median(rms)) if rms else np.nan}
