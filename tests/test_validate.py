import numpy as np
import pandas as pd
import pytest

import cdqc
from tests.test_evidence import make_band_image

PX_NM = 0.5
X_L, X_R, H, N = 80.0, 130.0, 260, 16


def _coords(seed=0, jitter=0.3):
    """정상 레시피 출력 — 참 엣지 위 좌표 + DL 컨투어 지터."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(30, H - 30, N)
    S = np.stack([np.full(N, X_L), ys], axis=1)
    E = np.stack([np.full(N, X_R), ys], axis=1)
    return (S + rng.normal(0, jitter, S.shape),
            E + rng.normal(0, jitter, E.shape))


@pytest.fixture(scope="module")
def cohort():
    """정상 시퀀스 12개로 L3/L2 코호트 통계 — 주입 채점의 고정 기준."""
    p = cdqc.Params()
    l3s, l2s = [], []
    for seed in range(12):
        img = make_band_image(h=H, w=200, x_left=X_L, x_right=X_R, seed=seed)
        S, E = _coords(seed=100 + seed)
        f = cdqc.extract_l3(img, S, E, PX_NM, params=p)
        l3s.append(f)
        l2s.append(cdqc.extract_l2(f, None, p))
    l3 = pd.concat(l3s, ignore_index=True)
    l2 = pd.concat(l2s, ignore_index=True)
    img = make_band_image(h=H, w=200, x_left=X_L, x_right=X_R, seed=99)
    S, E = _coords(seed=99)
    return {"p": p, "img": img, "S": S, "E": E,
            "l3_stats": cdqc.cohort_stats(l3, params=p),
            "l2_stats": cdqc.cohort_stats(l2, params=p)}


# ---------------------------------------------------------------- 좌표 주입

def test_inject_coords_shapes_and_affected():
    S, E = _coords()
    for kind in cdqc.INJECTIONS:
        S2, E2, aff = cdqc.inject_coords(S, E, kind, 3.0)
        assert S2.shape == E2.shape and len(aff) == len(S2)
        assert S2.shape[1] == 2
    assert cdqc.inject_coords(S, E, "drop_one")[0].shape[0] == N - 1
    assert not cdqc.inject_coords(S, E, "drop_one")[2].any()   # 행이 없다
    assert cdqc.inject_coords(S, E, "shift_one", 5.0)[2].sum() == 1
    assert cdqc.inject_coords(S, E, "swap")[2].sum() == 2
    assert cdqc.inject_coords(S, E, "all_shift", 5.0)[2].all()


def test_inject_coords_geometry():
    S = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    E = np.array([[10.0, 0.0], [10.0, 10.0], [10.0, 20.0]])
    S2, _, _ = cdqc.inject_coords(S, E, "shift_one", 4.0, index=1)
    assert S2[1].tolist() == [4.0, 10.0]        # +u(=+x) 방향 4px
    S3, E3, _ = cdqc.inject_coords(S, E, "all_shift", 2.0)
    assert np.allclose(S3 - S, [2.0, 0.0]) and np.allclose(E3 - E, [2.0, 0.0])
    _, E4, _ = cdqc.inject_coords(S, E, "rotate_one", 90.0, index=1)
    assert E4[1] == pytest.approx([5.0, 15.0])  # 중점 (5,10) 기준 90° 회전
    _, E5, _ = cdqc.inject_coords(S, E, "swap", index=0)
    assert E5[0].tolist() == E[1].tolist() and E5[1].tolist() == E[0].tolist()


def test_inject_coords_rejects_bad_args():
    S, E = _coords()
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.inject_coords(S, E, "nope")
    assert e.value.code == "E-ARG-06"
    with pytest.raises(cdqc.CdqcError) as e:
        cdqc.inject_coords(S, E, "shift_one", 1.0, index=999)
    assert e.value.code == "E-ARG-06"


# ------------------------------------------- L2 완료 기준: all_shift / drop_one

def test_injection_test_l2_catches_what_l3_cannot(cohort):
    """**L2의 완료 기준** — 시퀀스 전체 이동과 CD 누락이 L2에서 탐지된다.

    둘 다 CD 레벨로는 원리적으로 불가능하다: 통째로 밀리면 이웃 대비 잔차가
    전부 0이 되고, 없어진 CD는 z를 낼 행 자체가 없다.
    """
    c = cohort

    def extract(S, E):
        return cdqc.extract_l3(c["img"], S, E, PX_NM, params=c["p"])

    tab = cdqc.injection_test(c["S"], c["E"], extract, c["l3_stats"],
                              cases=[("all_shift", 5.0), ("drop_one", 0.0)],
                              l2_stats=c["l2_stats"], params=c["p"])
    base = tab[tab.kind == "none"].iloc[0]
    shift = tab[tab.kind == "all_shift"].iloc[0]
    drop = tab[tab.kind == "drop_one"].iloc[0]

    # 시퀀스 전체 이동: 양 끝 delta_median이 같이 움직여 z_top2까지 뜬다
    assert shift["l2_z_top2_kth"] > base["l2_z_top2_kth"] + 10.0
    assert shift["l2_rise"] > 1.0
    assert shift["l2_top_feature"].startswith("delta_median")

    # CD 누락: L3는 남은 행이 전부 정상이라 기준선과 **완전히 같다**
    assert drop["n_cd"] == base["n_cd"] - 1
    assert drop["l3_z_top3_kth"] == pytest.approx(base["l3_z_top3_kth"])
    assert drop["l3_z_max"] == pytest.approx(base["l3_z_max"])
    assert drop["l3_rise"] == pytest.approx(0.0)
    # L2에서는 n_cd가 상위 3개 안으로 올라온다
    assert drop["l2_z_top3_kth"] > base["l2_z_top3_kth"] + 1.0


def test_drop_one_signal_is_n_cd_scaled_by_its_mad_floor(cohort):
    """누락 1개 = |z_n_cd| 2.0 — n_cd MAD가 0이라 mad_floor(0.5)가 분모다.

    레시피가 규칙적이면 카테고리 안에서 n_cd는 늘 같은 값이라 MAD가 0이다.
    민감도는 전적으로 Params.mad_floors["n_cd"]가 정한다 (기본 0.5 → 1개
    누락이 2σ). 더 엄히 보려면 사용자가 이 하한을 낮춘다.
    """
    c = cohort
    l3 = cdqc.extract_l3(c["img"], c["S"], c["E"], PX_NM, params=c["p"])
    S2, E2, _ = cdqc.inject_coords(c["S"], c["E"], "drop_one")
    l3d = cdqc.extract_l3(c["img"], S2, E2, PX_NM, params=c["p"])
    z0 = cdqc.apply_z(cdqc.extract_l2(l3, None, c["p"]), c["l2_stats"], c["p"])
    z1 = cdqc.apply_z(cdqc.extract_l2(l3d, None, c["p"]), c["l2_stats"], c["p"])
    assert z0["z_n_cd"].iloc[0] == pytest.approx(0.0)
    assert z1["z_n_cd"].iloc[0] == pytest.approx(1.0 / c["p"].mad_floor("n_cd"))


def test_all_shift_is_invisible_to_sequence_residuals(cohort):
    """시퀀스 내 잔차의 구조적 맹점 — 통째로 밀리면 이웃 대비 잔차가 0이다."""
    c = cohort
    S2, E2, _ = cdqc.inject_coords(c["S"], c["E"], "all_shift", 5.0)
    z0 = cdqc.apply_z(cdqc.extract_l3(c["img"], c["S"], c["E"], PX_NM,
                                      params=c["p"]), c["l3_stats"], c["p"])
    z1 = cdqc.apply_z(cdqc.extract_l3(c["img"], S2, E2, PX_NM, params=c["p"]),
                      c["l3_stats"], c["p"])
    for col in ("z_s_resid", "z_e_resid", "z_dstep_s", "z_cd_resid"):
        assert abs(np.nanmedian(z1[col]) - np.nanmedian(z0[col])) < 1.0, col
    # delta는 (잔차와 달리) 반응한다 — 두 층이 서로 다른 실패를 본다는 증거
    assert np.nanmedian(z1["z_delta_s"]) > 3.0


def test_injection_test_detects_local_failures(cohort):
    """국소 실패는 CD 레벨에서 잡힌다 — 다만 **몇 개 피쳐가 반응하는지**가 다르다.

    좌표 이동은 잔차·delta 계열이 함께 반응해 z_top3까지 오르지만, 중점 기준
    회전은 끝점이 궤적 **진행 방향**으로 움직여서 법선 잔차(s_resid)가 원리적
    으로 못 본다 — 반응하는 것은 obliquity와 angle_resid 둘뿐이라 신호가
    z_top2에서 멈춘다. 다중비교 방어(z_topk)가 무엇을 대가로 치르는지 보여준다.
    """
    c = cohort

    def extract(S, E):
        return cdqc.extract_l3(c["img"], S, E, PX_NM, params=c["p"])

    tab = cdqc.injection_test(c["S"], c["E"], extract, c["l3_stats"],
                              cases=[("shift_one", 5.0), ("rotate_one", 3.0),
                                     ("rotate_frame", 1.0), ("swap", 0.0)],
                              params=c["p"])
    base = tab.iloc[0]
    rows = {r["kind"]: r for _, r in tab.iterrows()}

    assert rows["shift_one"]["l3_z_top3_kth"] > base["l3_z_top3_kth"] + 5.0
    assert rows["swap"]["l3_z_top3_kth"] > base["l3_z_top3_kth"] + 3.0
    # rotate_one은 2피쳐 신호 — top2까지만 오르고 top3에서는 사라진다
    assert rows["rotate_one"]["l3_z_top2_kth"] > base["l3_z_top2_kth"] + 3.0
    assert rows["rotate_one"]["l3_top_feature"] in ("angle_resid_seq",
                                                    "obliquity")
    # rotate_frame은 모든 CD가 다른 자리로 옮겨져 여러 피쳐가 함께 반응한다
    assert rows["rotate_frame"]["l3_z_top3_kth"] > base["l3_z_top3_kth"] + 3.0
    assert rows["rotate_frame"]["n_affected"] == rows["rotate_frame"]["n_cd"]


def test_injection_test_without_l2_stats(cohort):
    c = cohort
    tab = cdqc.injection_test(
        c["S"], c["E"],
        lambda S, E: cdqc.extract_l3(c["img"], S, E, PX_NM, params=c["p"]),
        c["l3_stats"], cases=[("shift_one", 3.0)], params=c["p"])
    assert "l2_z_max" not in tab.columns
    assert list(tab["kind"]) == ["none", "shift_one"]


# ---------------------------------------------------------------- 궤적 진단

def test_trajectory_check_straight_and_ordered():
    pts = np.stack([np.full(20, 50.0), np.linspace(0, 190, 20)], axis=1)
    t = cdqc.trajectory_check(pts)
    assert t["linearity"] == pytest.approx(1.0, abs=1e-9)
    assert t["monotonic"] == pytest.approx(1.0)
    assert t["resid_rms_px"] == pytest.approx(0.0, abs=1e-9)


def test_trajectory_check_scrambled_order():
    ys = np.linspace(0, 190, 20)
    pts = np.stack([np.full(20, 50.0), ys], axis=1)
    rng = np.random.default_rng(0)
    order = rng.permutation(20)
    t = cdqc.trajectory_check(pts, order=order)
    assert t["linearity"] == pytest.approx(1.0, abs=1e-9)   # 여전히 일렬
    assert abs(t["monotonic"]) < 0.6                        # 순서만 뒤죽박죽


def test_trajectory_check_cloud_and_short():
    rng = np.random.default_rng(1)
    t = cdqc.trajectory_check(rng.normal(0, 1, (60, 2)))
    assert t["linearity"] < 0.8              # 방향성 없는 구름
    short = cdqc.trajectory_check(np.zeros((2, 2)))
    assert np.isnan(short["linearity"]) and short["n_pts"] == 2


# ---------------------------------------------------------------- 카테고리 요약

def test_category_summary_records_without_changing_anything(cohort):
    c = cohort
    l3 = cdqc.extract_l3(c["img"], c["S"], c["E"], PX_NM, params=c["p"])
    l3["category_id"] = "A"
    before = l3.copy()
    agg = cdqc.aggregate_z(cdqc.apply_z(l3, c["l3_stats"], c["p"]))
    summ = cdqc.category_summary(l3, ["category_id"], agg)

    assert len(summ) == 1
    r = summ.iloc[0]
    assert r["n_cd"] == N
    assert r["linearity"] > 0.99            # S 좌표가 거의 일직선
    assert r["monotonic"] == pytest.approx(1.0, abs=0.05)
    assert 0 < r["s_resid_mad_px"] < 2.0    # px 단위 (배율 무관)
    assert r["cnr_med_s"] > 3
    assert r["n_img"] == 1 and r["cd_per_img"] == N
    assert np.isfinite(r["n_z_valid_med"])
    assert np.isfinite(r["z_top3_kth_p99"]) and np.isfinite(r["z_top3_mean_p99"])
    # 진단은 기록만 한다 — 입력 프레임을 건드리지 않는다
    pd.testing.assert_frame_equal(l3, before)


def test_category_summary_without_aggregates(params):
    l3 = pd.DataFrame({"category_id": ["A", "A", "B"],
                       "cd_nm": [30.0, 31.0, 40.0],
                       "s_resid": [0.1, -0.1, 0.0], "px_nm": [0.5] * 3})
    summ = cdqc.category_summary(l3, ["category_id"])
    assert summ["n_cd"].tolist() == [2, 1]
    assert np.isnan(summ["linearity"]).all()   # s_x/s_y 캐리어가 없으면 NaN
    assert np.isnan(summ["z_top3_kth_p99"]).all()
