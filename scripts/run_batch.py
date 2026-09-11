"""TEM CD 측정 품질 — 배치 피쳐 추출 + 카테고리별 코호트 z + 주입 민감도 테스트.

**이 파일은 라이브러리가 아니라 사용자 코드 계층이다.** cdqc는 파일을 읽지
않으므로 여기서 읽고, 컬럼을 정리하고, 좌표를 변환해서 넘긴다. 계산은 전부
cdqc가 한다 — 이 스크립트에 피쳐 계산이나 정규화 로직을 다시 쓰지 말 것
(0.3까지는 그랬고, 그래서 검증된 기능을 저장소 밖에서 잃을 뻔했다).

cdqc 위치는 아래 순서로 자동 탐색한다:
  1) 환경변수 CDQC_PATH
  2) 이미 설치된 cdqc (pip install -e .)
  3) 이 파일 기준 상위 디렉토리들에서 cdqc/ 폴더 탐색

사용:
    from run_batch import run_batch, injection_test
    DATASETS = [(dm3, tif, seg_png, xlsx), ...]
    out = run_batch(DATASETS, exclude=["가이드라인카테고리", ...])
    inj = injection_test(DATASETS[0], out["l3_stats"], out["l2_stats"])

**판정하지 않는다.** z와 플래그를 계산해서 CSV로 내보낼 뿐이고, 임계값은
호출자가 정한다 (spec §4.4).
"""
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")   # cv2 import 전에 설정

import sys                                                   # noqa: E402
import warnings                                              # noqa: E402
from pathlib import Path                                     # noqa: E402

import cv2                                                   # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

warnings.filterwarnings("ignore")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass


def _import_cdqc():
    """어느 디렉토리에서 실행하든 cdqc를 찾아 import."""
    p = os.environ.get("CDQC_PATH")
    if p and (Path(p) / "cdqc").is_dir():
        sys.path.insert(0, str(Path(p).resolve()))
    try:
        import cdqc as _c
        return _c
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for base in [here, *here.parents]:
        for cand in (base, base / "mqc"):
            if (cand / "cdqc" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                import cdqc as _c
                return _c
    raise ImportError(
        "cdqc를 찾을 수 없음. 환경변수 CDQC_PATH에 mqc 저장소 경로를 지정하거나 "
        "저장소에서 `pip install -e .` 실행할 것.")


cdqc = _import_cdqc()
from cdqc import func as nasca_io                              # noqa: E402

try:
    import ncempy.io.dm as dm
except ImportError:
    dm = None


# ------------------------------------------------------------------ 설정
CAT_COL = "MetrologyActivityName"
VAL_COL, UNIT_COL = "Measurement", "MeasurementUnit"
GROUP = ["category"]          # 코호트 축 — 레시피가 섞이면 ["recipe", "category"]

# 물리 단위 절대 임계. **공정 스펙 근거가 없는 경험값이다** — 여기 두는 이유는
# 사용자 코드 계층이라 고쳐도 라이브러리에 영향이 없기 때문이고, cdqc 쪽에
# 기본값을 두지 않는 이유도 같다 (spec §4.4). 스펙이 생기면 이 값을 바꾼다.
ABS_THRESHOLD = {
    "bdist_s": 1.0,          # nm — 좌표가 라벨 경계에서 이만큼 떨어지면 이상
    "bdist_e": 1.0,          # nm
    "obliquity": 5.0,        # deg — 세그먼트가 엣지 접선과 이만큼 어긋남
    "angle_resid_seq": 5.0,  # deg — 시퀀스 중앙 방향에서 벗어난 각
}

# 판정에서 뺄 카테고리를 자동으로 정하지 않는다 (spec §11-1). exclude 인자로만.
MIN_INSIDE_FRAC = 0.5        # 좌표가 이미지 안에 있는 비율이 이 미만이면 건너뜀


# ------------------------------------------------------------------ 로딩
def read_px_nm(dm3_path):
    """dm3 → (nm/px, dm3 이미지 크기). **픽셀 크기는 dm3에서만 읽을 수 있다.**"""
    if dm is None:
        raise ImportError("ncempy 필요: pip install ncempy")
    d = dm.dmReader(str(dm3_path))
    unit = str(np.atleast_1d(d["pixelUnit"])[0])
    scale = float(np.atleast_1d(d["pixelSize"])[0])
    return float(cdqc.to_nm([scale], unit)[0]), np.asarray(d["data"]).shape[:2]


def load_one(dm3, tif, seg_png, xlsx):
    """4개 경로 → (img, labelmap, df, px_nm)."""
    px_nm, dm3_hw = read_px_nm(dm3)

    img = cv2.imread(str(tif), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(tif)
    if img.ndim == 3:
        img = img[..., 0]
    img = cdqc.to_uint8(img)
    if dm3_hw != img.shape:                     # tif 리사이즈 보정
        px_nm *= dm3_hw[1] / img.shape[1]

    labelmap = None
    if seg_png:
        s = cv2.imread(str(seg_png), cv2.IMREAD_COLOR)
        if s is not None:
            # CD가 컬러 선으로 얹혀 라벨을 덮은 것을 복구 → 단일 채널 라벨맵
            labelmap = cdqc.close_annotation(s)[..., 1]
            if labelmap.shape != img.shape:
                labelmap = None                 # 해상도 불일치 → 라벨 경로 포기

    return img, labelmap, nasca_io.read_nasca_csv(xlsx, visible=False, header=True), px_nm


def meters_to_px(P_m, px_nm, W, H):
    """중심 원점·미터·y위쪽양수 → 내부 컨벤션 픽셀 (x=col, y=row).

    y 부호 반전이 핵심이다. 검증은 `cdqc.convention_scores()`로 후보 8개를
    비교하거나, 정상 카테고리에서 `bdist_s`가 0.000이 나오는지로 한다.
    """
    m_per_px = px_nm * 1e-9
    return np.column_stack([P_m[:, 0] / m_per_px + W / 2.0,
                            -P_m[:, 1] / m_per_px + H / 2.0])


# ------------------------------------------------------------------ 이미지 1장
def process_one(paths, image_id=None, params=None, auto_sigma=False,
                verbose=True):
    """(dm3, tif, seg, xlsx) → CD 레벨 피쳐 DataFrame.

    auto_sigma=True면 카테고리마다 `cdqc.select_grad_sigma`로 grad_sigma_px를
    고른다 (σ 개수만큼 extract_l3를 더 돈다). **주의**: 코호트마다 다른 σ를
    쓰면 delta 통계가 섞이므로, 캘리브레이션 때 한 번 보고 고정하는 편이 낫다.
    """
    dm3, tif, seg_png, xlsx = paths
    iid = image_id or Path(tif).stem
    p0 = params or cdqc.Params()
    img, labelmap, df, px_nm = load_one(dm3, tif, seg_png, xlsx)
    H, W = img.shape
    if CAT_COL not in df.columns:
        raise ValueError(f"{CAT_COL} 컬럼 없음: {list(df.columns)[:12]}")

    # 라벨 경계 맵은 **이미지당 1회**. img를 같이 넘겨 bgrad까지 받는다
    maps = cdqc.mask_maps(labelmap, img) if labelmap is not None else None

    rows = []
    for cat, g in df.groupby(CAT_COL, sort=False):
        S = meters_to_px(g[["LineBeginX", "LineBeginY"]].to_numpy(float), px_nm, W, H)
        E = meters_to_px(g[["LineEndX", "LineEndY"]].to_numpy(float), px_nm, W, H)
        inside = float(np.mean((S[:, 0] >= 0) & (S[:, 0] < W)
                               & (S[:, 1] >= 0) & (S[:, 1] < H)))
        if len(S) < 1 or inside < MIN_INSIDE_FRAC:
            if verbose:
                print(f"    {cat}: CD {len(S)}개, 안 {inside:.0%} — 건너뜀")
            continue

        value_nm = None
        if VAL_COL in g.columns and UNIT_COL in g.columns:
            value_nm = cdqc.to_nm(g[VAL_COL].to_numpy(float),
                                  g[UNIT_COL].to_numpy())

        p = p0
        sigma = p0.grad_sigma_px
        if auto_sigma:
            sel = cdqc.select_grad_sigma(img, S, E, px_nm, params=p0)
            if sel["sigma"]:
                sigma = sel["sigma"]
                p = _with_sigma(p0, sigma)

        f = cdqc.extract_l3(img, S, E, px_nm, value_nm=value_nm, params=p)
        if maps is not None:
            f = f.join(cdqc.boundary_features(maps, S, E, px_nm, params=p))

        f.insert(0, "image_id", iid)
        f.insert(1, "category", str(cat))
        f.insert(2, "cd_index", np.arange(len(f)))
        f["sigma_used"] = sigma
        rows.append(f)

    if not rows:
        raise ValueError(f"{iid}: 유효한 카테고리 없음")
    return pd.concat(rows, ignore_index=True)


def _with_sigma(params, sigma):
    import dataclasses
    return dataclasses.replace(params, grad_sigma_px=float(sigma))


def _alias(i):
    s = ""
    while True:
        s = chr(65 + i % 26) + s
        i = i // 26 - 1
        if i < 0:
            return s


# ------------------------------------------------------------------ 배치
def run_batch(datasets, outdir="out", exclude=None, params=None,
              auto_sigma=False, verbose=True):
    """datasets: [(dm3, tif, seg_png, xlsx), ...]

    exclude: 코호트 통계에서 뺄 카테고리명 (가이드라인/길이전용 등).
             제외해도 피쳐와 z는 계산되고 judged=False로 표시만 된다 —
             **한 데이터 세트의 관찰로 피쳐를 빼지 않는다** (spec §11-2).

    반환 dict: l3, l2, cat(카테고리 요약), l3_stats, l2_stats.
    l3_stats/l2_stats는 주입 테스트에 그대로 넘길 **고정 코호트 통계**다.
    """
    p = params or cdqc.Params()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames, fails = [], []
    for i, paths in enumerate(datasets, 1):
        iid = Path(paths[1]).stem
        try:
            f = process_one(paths, params=p, auto_sigma=auto_sigma,
                            verbose=verbose)
            frames.append(f)
            if verbose:
                print(f"[{i}/{len(datasets)}] {iid}  CD {len(f)}  "
                      f"카테고리 {f['category'].nunique()}  "
                      f"px_nm {f['px_nm'].iloc[0]:.4f}")
        except Exception as e:
            fails.append((iid, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(datasets)}] {iid}  실패 — {type(e).__name__}: {e}")

    if not frames:
        raise SystemExit("전부 실패")
    l3 = pd.concat(frames, ignore_index=True)
    l3["judged"] = ~l3["category"].isin(exclude or [])

    # ---- 정규화: 코호트 축은 여기서 정한다 (라이브러리는 이름으로만 동작)
    z3, l3_stats = cdqc.cohort_z(l3, GROUP, base_mask=l3["judged"], params=p,
                                 return_stats=True)
    agg3 = cdqc.aggregate_z(z3)
    l3 = pd.concat([z3, agg3], axis=1)

    l2 = cdqc.extract_l2(l3, ["image_id"] + GROUP, p)
    z2, l2_stats = cdqc.cohort_z(l2, GROUP, params=p, return_stats=True)
    l2 = pd.concat([z2, cdqc.aggregate_z(z2)], axis=1)

    # ---- 플래그: 임계는 사용자(이 파일)가 정한다. 라이브러리는 계산만
    thr = {k: v for k, v in ABS_THRESHOLD.items() if k in l3.columns}
    if thr:
        l3 = pd.concat([l3, cdqc.abs_flags(l3, thr)], axis=1)
    # n_cd는 z가 아니라 최빈값 불일치로 본다 — 카테고리 안에서 늘 같은 값이라
    # MAD가 0이고, 그러면 mad_floor가 민감도를 결정해버린다
    l2 = pd.concat([l2, cdqc.mode_flags(l2, ["n_cd"], GROUP)], axis=1)

    cat = cdqc.category_summary(l3, GROUP, agg3, traj_by="image_id")

    # ---- 익명화 (사외로 나갈 수 있는 형태로)
    cats = sorted(l3["category"].unique())
    alias = {c: _alias(i) for i, c in enumerate(cats)}
    for d in (l3, l2, cat):
        d["alias"] = d["category"].map(alias)

    l3.to_csv(outdir / "features_all.csv", index=False, encoding="utf-8-sig")
    l2.to_csv(outdir / "sequence_all.csv", index=False, encoding="utf-8-sig")
    cat.to_csv(outdir / "category_summary.csv", index=False, encoding="utf-8-sig")

    if verbose:
        _report(l3, l2, cat, cats, alias, fails)
    print(f"\n→ {outdir}/features_all.csv, sequence_all.csv, category_summary.csv")
    return {"l3": l3, "l2": l2, "cat": cat,
            "l3_stats": l3_stats, "l2_stats": l2_stats, "alias": alias}


def _report(l3, l2, cat, cats, alias, fails):
    """무차원 요약만 출력 — 사외로 내보낼 수 있는 형태 (spec §9)."""
    fmt = lambda v: f"{v:8.3f}"                                   # noqa: E731
    print("\n" + "=" * 100)
    print("카테고리 익명화 매핑")
    print("=" * 100)
    for c in cats:
        print(f"  {alias[c]:>4s} = {c}")

    print("\n" + "=" * 100)
    print(f"이미지 {l3['image_id'].nunique()}장, CD {len(l3)}개, "
          f"카테고리 {len(cats)}개, 판정대상 {int(l3['judged'].sum())}개, "
          f"실패 {len(fails)}건")
    print("=" * 100)
    show = [c for c in ["n_cd", "n_img", "cd_per_img", "linearity", "monotonic",
                        "s_resid_mad_px", "n_z_valid_med",
                        "delta_B_med_s", "delta_scatter_med_s",
                        "cnr_med_s", "cnr_med_e", "bdist_ratio_med",
                        "label_runs_med", "z_top3_kth_p99", "z_top3_mean_p99"]
            if c in cat.columns]
    print("\n[익명]")
    print(cat[["alias"] + show].to_string(index=False, float_format=fmt))
    print("\n※ 위 표는 진단용 기록이다. 이 값으로 피쳐를 빼거나 무력화하는 "
          "규칙을 만들지 않는다 (spec §11-2).")

    j = l3[l3["judged"]]
    if len(j):
        print("\n" + "=" * 100)
        print("판정 대상 z 분위 — 전부 정상이라면 이게 곧 오경보 기준선")
        print("=" * 100)
        cols = [c for c in cdqc.agg_columns() if c in j.columns]
        print(f"{'':14s}" + "".join(f"{q:>10}" for q in
                                    ["50%", "90%", "99%", "99.9%", "max"]))
        for c in cols:
            qs = j[c].quantile([.5, .9, .99, .999, 1.0]).to_numpy()
            print(f"{c:14s}" + "".join(f"{v:10.2f}" for v in qs))
        print("\n※ z_top{k}_kth = k번째로 큰 z (k개가 동시에 넘어야 걸림), "
              "z_top{k}_mean = 상위 k개 평균. 둘은 놓치는 실패가 다르다.")

        top = cdqc.top_feature(j)
        print("\n최대 z 피쳐 분포")
        print(top["top_feature"].value_counts().head(12).to_string())

    if "n_abs_flags" in l3.columns:
        print(f"\n절대 임계 플래그: {int(l3['n_abs_flags'].gt(0).sum())}개 CD "
              f"(임계 {ABS_THRESHOLD} — 공정 스펙 근거 없는 경험값)")
    if "n_mode_flags" in l2.columns:
        print(f"CD 개수 최빈값 불일치 시퀀스: {int(l2['n_mode_flags'].gt(0).sum())}개")

    if fails:
        print("\n실패 목록")
        for iid, msg in fails:
            print(f"  {iid}: {msg}")


# ------------------------------------------------------------------ 주입 테스트
def injection_test(paths, l3_stats, l2_stats=None, categories=None,
                   params=None, cases=None, primary="z_top3_kth", verbose=True):
    """실데이터에 알려진 오류를 주입해 탐지 하한을 측정.

    l3_stats / l2_stats: `run_batch`가 돌려준 **고정** 코호트 통계
    (`{(카테고리,): stats}`). 주입된 시퀀스로 통계를 다시 내면 주입이 흡수된다.

    연산은 전부 `cdqc.injection_test`가 한다 — 여기서는 파일을 읽고 카테고리별로
    호출할 뿐이다. **z 크기를 심각도로 읽지 말 것**: 이동량이 커지면 오히려
    탐색 창을 벗어나 z가 떨어진다. 탐지 여부만 본다.
    """
    p = params or cdqc.Params()
    dm3, tif, seg_png, xlsx = paths
    img, labelmap, df, px_nm = load_one(dm3, tif, seg_png, xlsx)
    H, W = img.shape
    maps = cdqc.mask_maps(labelmap, img) if labelmap is not None else None
    if cases is None:
        cases = [("shift_one", a) for a in (1, 2, 3, 5, 10, 20)] + \
                [("rotate_one", a) for a in (0.5, 1, 2, 5)] + \
                [("rotate_frame", a) for a in (0.2, 0.5, 1, 2)] + \
                [("all_shift", a) for a in (1, 3, 5)] + \
                [("swap", 0.0), ("drop_one", 0.0)]

    def extract(S, E):
        f = cdqc.extract_l3(img, S, E, px_nm, params=p)
        return f.join(cdqc.boundary_features(maps, S, E, px_nm, params=p)) \
            if maps is not None else f

    out = []
    for cat in (categories or list(df[CAT_COL].unique())):
        g = df[df[CAT_COL] == str(cat)]
        st3 = l3_stats.get((str(cat),))
        if len(g) < 3 or st3 is None:
            continue
        S = meters_to_px(g[["LineBeginX", "LineBeginY"]].to_numpy(float),
                         px_nm, W, H)
        E = meters_to_px(g[["LineEndX", "LineEndY"]].to_numpy(float),
                         px_nm, W, H)
        tab = cdqc.injection_test(
            S, E, extract, st3, cases=cases,
            l2_stats=(l2_stats or {}).get((str(cat),)),
            primary=primary, params=p)
        out.append(tab.assign(category=str(cat)))

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if verbose and len(res):
        print("=" * 100)
        print(f"주입 민감도 (기준 집계={primary}) — rise가 확실히 오르는 amount "
              f"= 탐지 하한")
        print("=" * 100)
        # rise는 각 행 안에서 기준선을 뺀 값이다. 여기서 중앙값을 내도
        # z와 rise가 서로 다른 행에서 오지 않는다
        show = ["l3_rise", f"l3_{primary}", "l3_top_feature"]
        if "l2_rise" in res.columns:
            show += ["l2_rise", "l2_top_feature"]
        piv = (res.groupby(["kind", "magnitude"])
               .agg({c: ("median" if res[c].dtype.kind == "f"
                         else (lambda x: x.mode().iloc[0] if len(x.mode()) else ""))
                     for c in show}))
        print(piv.to_string(float_format=lambda v: f"{v:9.2f}"))
    return res


if __name__ == "__main__":
    print(__doc__)
    print(f"cdqc {cdqc.__version__}: {Path(cdqc.__file__).parent}")
