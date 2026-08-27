"""README의 사용 예제 골격을 합성 데이터로 그대로 돌린다 (문서 회귀 방지).

selftest가 피쳐의 **감도**를 보는 반면, 여기는 문서에 적힌 **호출 순서**가
실제로 이어지는지만 본다: 세그멘테이션 복구 → 라벨맵 → 경계 피쳐 → 코호트 z
→ 집계 → 절대 임계 → 롤업 → 진단 → 주입 테스트.
"""
import cv2
import numpy as np
import pandas as pd
import pytest

import cdqc
from cdqc.synth.generator import SynthParams, generate_dataset


@pytest.fixture(scope="module")
def dataset():
    sp = SynthParams(n_images=6, image_size=(256, 256), cds_per_category=12,
                     n_categories=3, n_images_per_case=0, inject={})
    return sp, generate_dataset(sp)


def _labelmaps(images, masks):
    """세그멘테이션 PNG를 흉내: 라벨맵 위에 CD 컬러 선 → close_annotation 복구."""
    cats = sorted({c for (_, c) in masks})
    out = {}
    for iid, img in images.items():
        lab = np.full(img.shape, 10, dtype=np.uint8)
        for k, cat in enumerate(cats):
            lab[masks[(iid, cat)]] = 30 + 20 * k
        bgr = cv2.cvtColor(lab, cv2.COLOR_GRAY2BGR)
        bgr[::37, :] = (0, 0, 255)
        out[iid] = cdqc.close_annotation(bgr)[..., 1]
    return out


def test_readme_pipeline_runs_end_to_end(dataset, params):
    p = params
    _, (records, images, masks) = dataset
    df = records.copy()
    df["value_nm"] = cdqc.to_nm(df["value"].to_numpy(), df["unit"].to_numpy())
    maps = {iid: cdqc.mask_maps(lab)
            for iid, lab in _labelmaps(images, masks).items()}

    parts = []
    for (iid, cat), g in df.groupby(["image_id", "category_id"], sort=False):
        S = g[["sx", "sy"]].to_numpy()
        E = g[["ex", "ey"]].to_numpy()
        px = cdqc.infer_px_nm(S, E, g["value_nm"].to_numpy())
        f = cdqc.extract_l3(images[iid], S, E, px,
                            value_nm=g["value_nm"].to_numpy(), params=p)
        fb = cdqc.boundary_features(maps[iid], S, E, px, params=p)
        parts.append(pd.concat([f, fb], axis=1)
                     .assign(image_id=iid, category_id=cat))
    l3 = pd.concat(parts, ignore_index=True)
    l2 = cdqc.extract_l2(l3, ["image_id", "category_id"], p)
    l1 = pd.DataFrame([{"image_id": iid, **cdqc.extract_l1(img, p)}
                       for iid, img in images.items()])
    assert len(l3) == len(df) and len(l2) == 6 * 3 and len(l1) == 6

    is_normal = np.ones(len(l3), dtype=bool)
    z = cdqc.cohort_z(l3, ["category_id"], base_mask=is_normal, params=p)
    agg = cdqc.aggregate_z(z)
    assert np.isfinite(agg["z_top3_kth"]).all()
    assert (agg["z_max"] <= p.z_cap + 1e-9).all()
    # 정상 데이터라 대부분의 CD는 계면 위 — bdist가 실제로 계산됐는지
    assert np.nanmedian(l3["bdist_ratio"]) < 0.2
    assert (agg["n_z_valid"] > 15).all()

    t = cdqc.threshold_from_quantile(agg.loc[is_normal, "z_top3_kth"], 0.99)
    abs_f = cdqc.abs_flags(l3, {"bdist_s": 1.0, "obliquity": 5.0})
    mode_f = cdqc.mode_flags(l2, ["n_cd"], ["category_id"])
    flag = (agg["z_top3_kth"] > t) | (abs_f["n_abs_flags"] > 0)
    assert (mode_f["n_mode_flags"] == 0).all()   # 정상이면 CD 개수가 다 같다
    assert 0 < flag.sum() < 0.2 * len(flag)      # 정상 데이터 = 소수만 플래그

    roll = cdqc.flag_rollup(l3, flag, ["image_id", "category_id"])
    assert roll["n_cd"].sum() == len(l3)
    assert cdqc.impact_nm(l3["cd_nm"].to_numpy(), flag.to_numpy()) < 1.0

    summ = cdqc.category_summary(l3, ["image_id", "category_id"], agg)
    assert len(summ) == len(l2)
    assert summ["linearity"].min() > 0.9         # 합성 궤적은 거의 일렬
    assert np.isfinite(summ["z_top3_kth_p99"]).all()


def test_readme_injection_flow(dataset, params):
    """README의 injection_test 호출 형태 — 콜러블 하나로 실데이터에도 그대로."""
    p = params
    _, (records, images, _) = dataset
    df = records.copy()
    df["value_nm"] = cdqc.to_nm(df["value"].to_numpy(), df["unit"].to_numpy())

    # 코호트는 **카테고리별**로 자른다 (spec §4.2) — 카테고리를 섞으면 고유
    # 성질(길이 산포 등)이 잡음이 되어 주입 신호를 덮는다
    l3s, l2s = [], []
    for (iid, cat), g in df.groupby(["image_id", "category_id"], sort=False):
        if cat != "A":
            continue
        S, E = g[["sx", "sy"]].to_numpy(), g[["ex", "ey"]].to_numpy()
        px = cdqc.infer_px_nm(S, E, g["value_nm"].to_numpy())
        f = cdqc.extract_l3(images[iid], S, E, px, params=p)
        l3s.append(f)
        l2s.append(cdqc.extract_l2(f, None, p))
    st3 = cdqc.cohort_stats(pd.concat(l3s, ignore_index=True), params=p)
    st2 = cdqc.cohort_stats(pd.concat(l2s, ignore_index=True), params=p)

    iid = df["image_id"].iloc[0]
    g = df[(df.image_id == iid) & (df.category_id == "A")]
    S, E = g[["sx", "sy"]].to_numpy(), g[["ex", "ey"]].to_numpy()
    px = cdqc.infer_px_nm(S, E, g["value_nm"].to_numpy())
    tab = cdqc.injection_test(
        S, E, lambda s, e: cdqc.extract_l3(images[iid], s, e, px, params=p),
        st3, cases=[("shift_one", 5.0), ("all_shift", 5.0), ("drop_one", 0.0)],
        l2_stats=st2, params=p)

    rows = {r["kind"]: r for _, r in tab.iterrows()}
    assert rows["shift_one"]["l3_rise"] > 2.0
    assert rows["all_shift"]["l2_top_feature"].startswith("delta_median")
    # 누락은 n_cd가 잡는다. 다만 **어느 피쳐가 top이 되는지**는 코호트 잡음에
    # 달렸으므로(여기선 6장뿐이라 cd_mad가 이미 3σ) 집계가 오르는지로 본다 —
    # n_cd z가 정확히 1/mad_floor인 것은 test_validate.py가 고정한다
    assert rows["drop_one"]["l2_z_top3_kth"] > rows["none"]["l2_z_top3_kth"] + 1.0
    # 누락은 L3에서 원리적으로 안 보인다 — 남은 행이 전부 정상이므로
    assert rows["drop_one"]["l3_rise"] == pytest.approx(0.0)
