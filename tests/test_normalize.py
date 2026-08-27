import numpy as np
import pandas as pd
import pytest

import cdqc


def _frame(n=40, seed=0):
    """카테고리 A/B/C는 정상 산포, D는 카테고리 내 변동이 전혀 없음(MAD=0)."""
    rng = np.random.default_rng(seed)
    cats, vals = [], []
    for i, c in enumerate("ABC"):
        cats += [c] * n
        vals.append(rng.normal(10 + i, 1.0, n))
    cats += ["D"] * n
    vals.append(np.full(n, 3.0))
    return pd.DataFrame({"category_id": cats, "cnr_s": np.concatenate(vals)})


def test_cohort_z_mad_zero_does_not_blow_up(params):
    """카테고리 MAD=0에서 z 폭발 금지 — 피쳐를 빼서가 아니라 분모 하한으로."""
    df = _frame()
    df.loc[df.index[-1], "cnr_s"] = 3.6        # D에서 1개만 벗어남
    raw = cdqc.apply_z(df.tail(1),
                       cdqc.cohort_stats(df[df.category_id == "D"], params=params),
                       params)
    # cnr_s는 worse_when="low"라 directed z는 부호가 뒤집힌다 — 크기로 본다
    assert abs(raw["z_cnr_s"].iloc[0]) > 1e5   # 폴백 없으면 이렇게 터진다

    z = cdqc.cohort_z(df, ["category_id"], params=params)
    assert abs(z["z_cnr_s"].iloc[-1]) < 5      # pooled MAD 폴백이 받아냄
    d = z[z.category_id == "D"]["z_cnr_s"].to_numpy()[:-1]
    assert np.allclose(d, 0.0)                 # 나머지는 중앙값 그대로


def test_cohort_z_caps_and_is_per_category(params):
    df = _frame()
    df.loc[0, "cnr_s"] = -1e6                  # A에 극단값
    z = cdqc.cohort_z(df, ["category_id"], params=params)
    assert z["z_cnr_s"].iloc[0] == pytest.approx(params.z_cap)
    # 카테고리마다 중심이 따로 → 중앙값 근처는 전부 0 근처
    med = z.groupby("category_id")["z_cnr_s"].median().abs()
    assert (med < 0.5).all()


def test_cohort_z_base_mask_excludes_bad_rows(params):
    """통계는 정상 행으로만 — 불량이 섞이면 중심이 끌려간다."""
    df = _frame()
    bad = np.zeros(len(df), dtype=bool)
    bad[:15] = True                            # A의 15개가 증거 없는 불량
    df.loc[df.index[:15], "cnr_s"] = 0.5
    z = cdqc.cohort_z(df, ["category_id"], base_mask=~bad, params=params)
    assert (z["z_cnr_s"].iloc[:15] > 10).all()


def test_aggregate_z_kth_and_mean(params):
    """두 정의를 다 낸다 — k번째 값(다중비교 방어)과 상위 k 평균."""
    z = pd.DataFrame({"z_delta_s": [10.0, 1.0], "z_delta_e": [8.0, 1.0],
                      "z_cnr_s": [6.0, 1.0], "z_cnr_e": [1.0, 1.0]})
    a = cdqc.aggregate_z(z)
    assert a["z_max"].tolist() == [10.0, 1.0]
    assert a["z_top2_kth"].tolist() == [8.0, 1.0]
    assert a["z_top3_kth"].tolist() == [6.0, 1.0]
    assert a["z_top2_mean"].tolist() == [9.0, 1.0]     # (10+8)/2
    assert a["z_top3_mean"].tolist() == [8.0, 1.0]     # (10+8+6)/3
    assert a["n_z_valid"].tolist() == [4, 4]
    # z_max는 top_feature의 top_z와 같아야 한다 (같은 기본 피쳐 셋)
    assert a["z_max"].tolist() == cdqc.top_feature(z)["top_z"].tolist()
    assert set(cdqc.agg_columns()) == set(a.columns) - {"n_z_valid"}


def test_aggregate_z_mean_is_fooled_by_one_saturated_feature(params):
    """상위 k 평균은 피쳐 하나가 상한까지 튀면 여전히 크다 — k번째 값은 아니다."""
    z = pd.DataFrame({"z_delta_s": [30.0], "z_delta_e": [0.2],
                      "z_cnr_s": [0.1], "z_cnr_e": [0.0]})
    a = cdqc.aggregate_z(z)
    assert a["z_top3_mean"].iloc[0] > 10.0
    assert a["z_top3_kth"].iloc[0] < 0.5


def test_aggregate_z_nan_does_not_win_sort(params):
    """np.sort는 NaN을 맨 뒤(최댓값 자리)로 보낸다 — −inf 치환 확인."""
    z = pd.DataFrame({"z_delta_s": [np.nan, 5.0], "z_delta_e": [np.nan, np.nan],
                      "z_cnr_s": [2.0, 3.0], "z_cnr_e": [1.0, np.nan]})
    a = cdqc.aggregate_z(z)
    assert a["z_max"].tolist() == [2.0, 5.0]
    assert a["z_top2_kth"].tolist() == [1.0, 3.0]
    assert a["z_top2_mean"].tolist() == [1.5, 4.0]
    assert np.isnan(a["z_top3_kth"].iloc[0])   # 유효 피쳐가 2개뿐
    assert np.isnan(a["z_top3_mean"].iloc[0])
    assert a["n_z_valid"].tolist() == [2, 2]


def test_aggregate_z_all_nan_row(params):
    """CD 1개짜리 카테고리 — 관계 피쳐가 전부 NaN이어도 NaN이 최댓값이 되면 안 됨."""
    z = pd.DataFrame({"z_delta_s": [np.nan], "z_cd_resid": [np.nan]})
    a = cdqc.aggregate_z(z)
    assert np.isnan(a["z_max"].iloc[0]) and a["n_z_valid"].iloc[0] == 0


def test_abs_flags_directions_and_count():
    df = pd.DataFrame({"bdist_s": [0.0, 2.0],       # high
                       "cnr_s": [12.0, 0.3],        # low
                       "angle_resid_seq": [1.0, -9.0]})  # both
    f = cdqc.abs_flags(df, {"bdist_s": 1.0, "cnr_s": 1.0,
                            "angle_resid_seq": 5.0})
    assert f["flag_bdist_s"].tolist() == [False, True]
    assert f["flag_cnr_s"].tolist() == [False, True]
    assert f["flag_angle_resid_seq"].tolist() == [False, True]
    assert f["n_abs_flags"].tolist() == [0, 3]


def test_abs_flags_requires_explicit_thresholds():
    """임계값에 기본값을 두지 않는다 — 공정 스펙 근거는 사용자만 안다."""
    df = pd.DataFrame({"bdist_s": [0.0]})
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.abs_flags(df, {})
    assert e.value.code == "E-ARG-04"
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.abs_flags(df, {"nonexistent_feature": 1.0})
    assert e.value.code == "E-ARG-05"


def test_abs_flags_nan_is_not_a_flag():
    f = cdqc.abs_flags(pd.DataFrame({"bdist_s": [np.nan]}), {"bdist_s": 1.0})
    assert not f["flag_bdist_s"].iloc[0]


def test_flag_rollup(params):
    l3 = pd.DataFrame({"image_id": ["i1"] * 6 + ["i2"] * 4,
                       "cd_nm": [30.0] * 10})
    flags = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
    r = cdqc.flag_rollup(l3, flags, ["image_id"])
    assert r["n_flagged"].tolist() == [3, 2]
    assert r["frac_flagged"].tolist() == [0.5, 0.5]
    assert r["max_run"].tolist() == [3, 1]      # 뭉침 vs 산발


def test_cohort_z_preserves_row_order_with_duplicate_index(params):
    """시퀀스별 프레임을 ignore_index 없이 concat하면 인덱스가 중복된다."""
    df = _frame(n=12)
    df.index = list(range(12)) * 4                 # 중복 인덱스
    df = df.assign(mark=np.arange(len(df)))
    z = cdqc.cohort_z(df, ["category_id"], params=params)
    assert z["mark"].tolist() == df["mark"].tolist()   # 행 순서 보존
    assert z.index.tolist() == df.index.tolist()
    assert np.allclose(z["cnr_s"], df["cnr_s"])
