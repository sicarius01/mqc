"""피쳐별 카테고리 히스토그램 — 눈으로 분포를 확인하는 용도 (사용자 코드 계층).

궤적 진단(`trajectory_check`)과 카테고리 요약(`category_summary`)은 cdqc로
승격됐다. 여기 남은 것은 **그리기**뿐이다.

사용:
    from feature_hist import plot_feature_hists, print_diagnostics

    plot_feature_hists(l3)                      # 원본 + z 피쳐 전부
    plot_feature_hists(l3, kind="z")            # z 피쳐만
    plot_feature_hists(l3, cols=["delta_s"])    # 지정한 것만
    print_diagnostics(l3)                       # cdqc.category_summary 출력

출력: out/feature_hist/<피쳐명>.png
"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_batch import GROUP, cdqc                              # noqa: E402

# 좌표·식별자·캐리어 컬럼은 분포를 봐야 의미가 없다
SKIP = {"image_id", "category", "alias", "cd_index", "px_nm", "sigma_used",
        "judged", "n_z_valid", "s_x", "s_y", "e_x", "e_y", "mid_x", "mid_y",
        "label_s", "label_e", "label_mid"}


def _pick_cols(feat, kind):
    num = feat.select_dtypes(include=[np.number]).columns
    cols = [c for c in num if c not in SKIP]
    if kind == "z":
        return [c for c in cols if c.startswith(("z_", "zs_"))]
    if kind == "raw":
        return [c for c in cols if not c.startswith(("z_", "zs_"))]
    return cols


def plot_feature_hists(feat, outdir="out/feature_hist", cols=None, kind="all",
                       bins=40, clip=(0.005, 0.995), cat_col="alias",
                       ncol=4, share_x=True, verbose=True):
    """피쳐 하나당 PNG 한 장. 카테고리별 서브플롯으로 분포를 나란히 본다.

    clip : 공통 x 범위를 정하는 분위 (극단값이 축을 끌고 가는 것 방지)
    빨간 세로선 = 카테고리 중앙값, 회색 점선 = 전체 중앙값
    제목의 (N↔) = 축 밖으로 밀린 값의 개수
    """
    os.makedirs(outdir, exist_ok=True)
    if cat_col not in feat.columns:
        cat_col = GROUP[-1]
    cols = cols or _pick_cols(feat, kind)
    cats = sorted(feat[cat_col].dropna().unique())
    nrow = int(np.ceil(len(cats) / ncol))

    made = 0
    for col in cols:
        v_all = feat[col].to_numpy(float)
        v_all = v_all[np.isfinite(v_all)]
        if len(v_all) < 5:
            continue

        if share_x:
            lo, hi = np.quantile(v_all, clip)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(v_all)), float(np.max(v_all)) + 1e-9
            pad = 0.03 * (hi - lo)
            edges = np.linspace(lo - pad, hi + pad, bins + 1)
            xr = (edges[0], edges[-1])
        else:
            edges, xr = bins, None

        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow),
                                 squeeze=False)
        gmed = float(np.median(v_all))
        for i, cat in enumerate(cats):
            ax = axes[i // ncol][i % ncol]
            v = feat.loc[feat[cat_col] == cat, col].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
                ax.set_title(f"{cat}  (n=0)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            n_out = int((v < xr[0]).sum() + (v > xr[1]).sum()) if share_x else 0
            ax.hist(np.clip(v, xr[0], xr[1]) if share_x else v,
                    bins=edges, color="steelblue", edgecolor="none")
            ax.axvline(np.median(v), color="crimson", lw=1)
            ax.axvline(gmed, color="gray", lw=0.8, ls=":")
            if xr:
                ax.set_xlim(*xr)
            ttl = f"{cat}  n={len(v)}  med={np.median(v):.3g}"
            if n_out:
                ttl += f"  ({n_out}↔)"
            ax.set_title(ttl, fontsize=9)
            ax.tick_params(labelsize=7)

        for j in range(len(cats), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")

        fig.suptitle(f"{col}   (전체 n={len(v_all)}, 중앙값 {gmed:.4g}, "
                     f"빨강=카테고리 중앙값 / 회색점선=전체 중앙값)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in col)
        fig.savefig(f"{outdir}/{safe}.png", dpi=100)
        plt.close(fig)
        made += 1
        if verbose:
            print(f"  {outdir}/{safe}.png")

    print(f"\n{made}개 저장 → {outdir}/")
    return made


def print_diagnostics(feat, group_cols=None, cat_col="alias"):
    """cdqc.category_summary를 표로 출력 — **기록일 뿐 판정이 아니다.**

    관계 피쳐(s_resid 등)는 "측정 순서상 이웃한 CD 는 공간적으로도 이웃"을
    전제한다. 구조가 2D로 배열돼 있으면 그 전제가 깨지는데, 그래도 **오경보는
    나지 않는다** — 잔차가 전부 크면 카테고리 중앙값과 MAD 가 같이 커져 z 가
    0 근처가 되기 때문이다. 대신 탐지력을 조용히 잃는다. `linearity`와
    `monotonic`은 어느 카테고리에서 그런지 **알아두기 위한** 지표다.

    여기서 임계를 걸어 "성립/깨짐"을 판정하지 않는다: 한 데이터 세트에서 정한
    0.98 같은 값을 코드에 박으면 그게 곧 부채가 된다 (spec §11-2).
    """
    group_cols = group_cols or GROUP
    cols = [c for c in cdqc.agg_columns() + ["n_z_valid"] if c in feat.columns]
    agg = feat[cols] if cols else None
    summ = cdqc.category_summary(feat, group_cols, agg)
    if cat_col in feat.columns:
        m = feat.drop_duplicates(group_cols).set_index(group_cols)[cat_col]
        summ[cat_col] = summ.set_index(group_cols).index.map(m)
    show = [c for c in [cat_col, "n_cd", "cd_per_img", "linearity", "monotonic",
                        "s_resid_mad_px", "cd_resid_mad_px", "cnr_med_s",
                        "bdist_ratio_med", "label_runs_med", "n_z_valid_med"]
            if c in summ.columns]
    print("=" * 100)
    print("카테고리 진단 — 관계 피쳐의 슬라이딩 가정이 성립하는지 등 (기록용)")
    print("=" * 100)
    print(summ[show].to_string(index=False,
                               float_format=lambda v: f"{v:8.3f}"))
    print("\nlinearity  : S 점들이 한 직선 위에 있는 정도 (1 = 완전 일렬)")
    print("monotonic  : 측정 순서와 공간 위치의 순위 상관 (1 = 순서대로 이동)")
    print("*_mad_px   : 잔차의 카테고리 내 산포 (px 환산 — 배율 무관)")
    return summ


if __name__ == "__main__":
    feat = pd.read_csv("out/features_all.csv")
    plot_feature_hists(feat)
    print_diagnostics(feat)
