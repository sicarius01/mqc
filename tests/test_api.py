import numpy as np
import pandas as pd
import pytest

import cdqc


def _l3_frame(n=300, cat="A", seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "category_id": [cat] * n,
        "delta_s": rng.normal(0, 0.1, n),     # both
        "cnr_s": rng.normal(10, 1, n),        # low
        "rise_s": rng.lognormal(0, 0.2, n),   # high + log
        "pol_s": np.ones(n),                  # match
        "edge_valid_s": [True] * n,           # bool
    })


def test_directed_z_signs(params):
    df = _l3_frame()
    st = cdqc.cohort_stats(df, params=params)
    bad = pd.DataFrame([{"category_id": "A", "delta_s": 1.0, "cnr_s": 2.0,
                         "rise_s": 1.0, "pol_s": -1.0, "edge_valid_s": False}])
    z = cdqc.apply_z(bad, st, params)
    row = z.iloc[0]
    assert row["z_delta_s"] > 5            # both → |z| 큼
    assert row["z_cnr_s"] > 5              # low → 낮을수록 z 큼
    assert row["z_pol_s"] == pytest.approx(cdqc.Z_ON_BAD)
    assert row["z_edge_valid_s"] == pytest.approx(cdqc.Z_ON_BAD)
    assert "zs_delta_s" in z.columns       # both는 부호 z 보존
    assert row["zs_delta_s"] > 5           # 부호 유지 (+ 방향)

    good = cdqc.apply_z(df.head(1), st, params).iloc[0]
    assert abs(good["z_pol_s"]) < 1e-9


def test_stats_are_plain_dict(params):
    st = cdqc.cohort_stats(_l3_frame(), params=params)
    import json
    blob = json.dumps(st)                  # 저장/로드는 사용자 자유
    st2 = json.loads(blob)
    z1 = cdqc.apply_z(_l3_frame(seed=2), st, params)
    z2 = cdqc.apply_z(_l3_frame(seed=2), st2, params)
    assert np.allclose(z1["z_delta_s"], z2["z_delta_s"], equal_nan=True)


def test_mad_floor_prevents_blowup(params):
    df = _l3_frame().assign(npk_s=1.0)     # MAD=0 이산 피쳐
    st = cdqc.cohort_stats(df, params=params)
    bad = df.head(1).assign(npk_s=3.0)
    z = cdqc.apply_z(bad, st, params)
    assert z.iloc[0]["z_npk_s"] == pytest.approx((3.0 - 1.0) / 0.5)


def test_robust_stats_trim():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    x[:30] += 50.0
    med, mad = cdqc.robust_stats(x, trim_frac=0.05)
    assert abs(med) < 0.2
    assert 0.5 < mad < 1.0


def test_top_feature_and_reason(params):
    z = pd.DataFrame({"z_delta_s": [0.5, 6.0], "z_dstep_s": [0.5, 1.0]})
    top = cdqc.top_feature(z)
    assert top["top_feature"].tolist() == ["delta_s", "delta_s"]
    assert top.iloc[1]["top_z"] == 6.0
    assert top.iloc[1]["reason_code"] == "POSITION_MISMATCH"


def test_top_feature_excludes_curv_by_default():
    z = pd.DataFrame({"z_curv_s": [9.0], "z_delta_s": [1.0]})
    top = cdqc.top_feature(z)
    assert top.iloc[0]["top_feature"] == "delta_s"      # curv 기본 제외
    top2 = cdqc.top_feature(z, feature_cols=["curv_s", "delta_s"])
    assert top2.iloc[0]["top_feature"] == "curv_s"      # 명시하면 포함


def test_impact_and_max_run():
    cd = np.array([30.0] * 5 + [40.0] * 5)
    flags = np.array([False] * 5 + [True] * 5)
    # 전체 median(35) vs 클린 median(30) → 5nm
    assert cdqc.impact_nm(cd, flags) == pytest.approx(5.0)
    assert cdqc.impact_nm(cd, np.zeros(10, dtype=bool)) == 0.0
    assert cdqc.impact_nm(cd, np.ones(10, dtype=bool)) == float("inf")
    assert cdqc.max_run(np.array([0, 1, 1, 1, 0, 1], dtype=bool)) == 3


def test_threshold_from_quantile():
    v = np.arange(100, dtype=float)
    assert cdqc.threshold_from_quantile(v, 0.9) == pytest.approx(89.1)
    with pytest.raises(cdqc.CdqcError):
        cdqc.threshold_from_quantile([np.nan], 0.9)


def test_extract_l2_groups(params):
    l3 = pd.DataFrame({
        "image_id": ["i1"] * 4 + ["i2"] * 3,
        "category_id": ["A"] * 7,
        "cd_nm": [30.0, 31.0, 32.0, 33.0, 40.0, 41.0, 42.0],
        "delta_s": [0.1] * 7, "delta_e": [-0.1] * 7,
        "s_resid": [0.2] * 7, "e_resid": [0.0] * 7,
    })
    l2 = cdqc.extract_l2(l3, ["image_id", "category_id"], params)
    assert l2["n_cd"].tolist() == [4, 3]
    assert l2["cd_median"].tolist() == [31.5, 41.0]
    assert l2.iloc[0]["traj_rms_s"] == pytest.approx(0.2)
    one = cdqc.extract_l2(l3, None, params)          # 전체 = 한 시퀀스
    assert one.iloc[0]["n_cd"] == 7


def test_extract_l2_wholesale_failure_columns(params):
    """angle_median(원형)/angle_spread/pitch_median/span_nm — 변경 #03 §2."""
    n = 5
    l3 = pd.DataFrame({
        "cd_nm": [30.0] * n,
        "angle": [10.0, 12.0, 11.0, 13.0, 12.0],
        "mid_x": np.full(n, 50.0),
        "mid_y": 10.0 * np.arange(n, dtype=float),   # 피치 10px
        "px_nm": [0.5] * n,
    })
    l2 = cdqc.extract_l2(l3, None, params)
    r = l2.iloc[0]
    assert r["angle_median"] == pytest.approx(12.0, abs=0.5)
    assert r["angle_spread"] == pytest.approx(1.0, abs=0.5)
    assert r["pitch_median"] == pytest.approx(5.0)       # 10px * 0.5nm
    assert r["span_nm"] == pytest.approx(20.0)           # 40px * 0.5nm


def test_extract_l3_carries_midpoints(params):
    S = np.array([[0.0, 0.0], [0.0, 10.0]])
    E = np.array([[10.0, 0.0], [10.0, 10.0]])
    f = cdqc.extract_l3(None, S, E, 0.5, params=params)
    assert f["mid_x"].tolist() == [5.0, 5.0]
    assert f["px_nm"].iloc[0] == 0.5


def test_recall_at_fpr_hand_computed():
    good = np.arange(100, dtype=float)          # 0..99 → 95분위 = 94.05
    bad = np.array([50.0, 96.0, 98.0, np.nan])
    r = cdqc.recall_at_fpr(bad, good, fpr=0.05)
    assert r["threshold"] == pytest.approx(94.05)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["fpr_actual"] == pytest.approx(0.05)
    assert r["n_nan"] == 1 and r["n_bad"] == 3


def test_localization():
    assert cdqc.localization_hit([7, 3, 5], {3, 9}, k=2)
    assert not cdqc.localization_hit([7, 3, 5], {9}, k=3)
    rate = cdqc.localization_rate([[1], [2], [3]], [{1}, {5}, {3}], k=1)
    assert rate == pytest.approx(2 / 3)


def test_ablation_table(params):
    rng = np.random.default_rng(0)
    n = 200
    z = pd.DataFrame({
        "z_delta_s": rng.normal(0, 1, n),
        "z_cnr_s": rng.normal(0, 1, n),
    })
    labels = np.zeros(n, dtype=bool)
    labels[:20] = True
    z.loc[z.index[:20], "z_delta_s"] += 6.0      # delta만 신호
    tab = cdqc.ablation_table(z, {"delta만": ["delta_s"],
                                  "cnr만": ["cnr_s"],
                                  "둘 다": ["delta_s", "cnr_s"]},
                              labels, fpr=0.05, params=params)
    r = {row["set"]: row["recall"] for _, row in tab.iterrows()}
    assert r["delta만"] > 0.9
    assert r["cnr만"] < 0.3
    assert r["둘 다"] > 0.8
