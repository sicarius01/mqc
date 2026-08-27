"""scripts/run_batch.py — 사용자 코드 계층이 새 cdqc API로 끝까지 도는지.

파일 I/O만 가짜로 갈아끼우고(`load_one`) 나머지 경로는 실제로 돌린다:
좌표 변환 → extract_l3 → boundary_features → cohort_z → aggregate_z →
abs_flags/mode_flags → category_summary → injection_test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cdqc
from cdqc.synth.generator import SynthParams, generate_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_batch as rb                                          # noqa: E402

PX_NM = 0.5


def _to_meters(P_px, px_nm, W, H):
    """meters_to_px의 역변환 — 합성 픽셀 좌표를 xlsx 형식(미터)으로."""
    m = px_nm * 1e-9
    return np.column_stack([(P_px[:, 0] - W / 2.0) * m,
                            -(P_px[:, 1] - H / 2.0) * m])


@pytest.fixture(scope="module")
def fake_inputs():
    """{image_id: (img, labelmap, xlsx형 DataFrame, px_nm)}."""
    sp = SynthParams(n_images=6, image_size=(256, 256), cds_per_category=10,
                     n_categories=3, n_images_per_case=0, inject={},
                     px_nm=PX_NM)
    records, images, masks = generate_dataset(sp)
    cats = sorted({c for (_, c) in masks})
    out = {}
    for iid, img in images.items():
        H, W = img.shape
        lab = np.full(img.shape, 10, dtype=np.uint8)
        for k, cat in enumerate(cats):
            lab[masks[(iid, cat)]] = 30 + 20 * k
        g = records[records.image_id == iid]
        S = _to_meters(g[["sx", "sy"]].to_numpy(float), PX_NM, W, H)
        E = _to_meters(g[["ex", "ey"]].to_numpy(float), PX_NM, W, H)
        df = pd.DataFrame({
            rb.CAT_COL: g["category_id"].to_numpy(),
            "LineBeginX": S[:, 0], "LineBeginY": S[:, 1],
            "LineEndX": E[:, 0], "LineEndY": E[:, 1],
            rb.VAL_COL: g["value"].to_numpy(),
            rb.UNIT_COL: g["unit"].to_numpy(),
        })
        out[iid] = (img, lab, df, PX_NM)
    return out


@pytest.fixture
def patched(monkeypatch, fake_inputs):
    def fake_load_one(dm3, tif, seg_png, xlsx):
        return fake_inputs[Path(tif).stem]

    monkeypatch.setattr(rb, "load_one", fake_load_one)
    return [(f"{iid}.dm3", f"{iid}.tif", f"{iid}.png", f"{iid}.xlsx")
            for iid in fake_inputs]


def test_meters_to_px_roundtrip():
    P = np.array([[10.0, 20.0], [2000.0, 1500.0]])
    m = _to_meters(P, 0.55, 2048, 2048)
    back = rb.meters_to_px(m, 0.55, 2048, 2048)
    assert np.allclose(back, P)


def test_run_batch_end_to_end(patched, tmp_path):
    out = rb.run_batch(patched, outdir=tmp_path, verbose=False)
    l3, l2, cat = out["l3"], out["l2"], out["cat"]

    assert len(l3) == 6 * 3 * 10 and l2.shape[0] == 6 * 3
    assert cat.shape[0] == 3                       # 카테고리 3개로 롤업
    for f in ("features_all.csv", "sequence_all.csv", "category_summary.csv"):
        assert (tmp_path / f).exists()

    # 라벨 경계 경로가 실제로 붙었는지 (bdist / bgrad / label_*)
    for c in ("bdist_s", "bdist_ratio", "label_runs", "bgrad_s", "label_mid"):
        assert c in l3.columns, c
    assert np.nanmedian(l3["bdist_ratio"]) < 0.2   # 정상 CD는 계면 위

    # 정규화·집계가 전부 붙었는지
    for c in cdqc.agg_columns():
        assert c in l3.columns and c in l2.columns, c
    assert np.isfinite(l3["z_top3_kth"]).all()
    assert (l3["z_max"].abs() <= cdqc.Params().z_cap + 1e-9).all()
    assert "angle_resid_cohort" in l3.columns      # cohort_z가 붙인다

    # 플래그
    assert l3["n_abs_flags"].sum() < 0.1 * len(l3)   # 정상 데이터
    assert (l2["n_mode_flags"] == 0).all()           # CD 개수가 다 같다

    # 고정 코호트 통계가 주입 테스트에 쓸 수 있는 형태로 나오는지
    assert set(out["l3_stats"]) == {("A",), ("B",), ("C",)}
    assert "delta_s" in out["l3_stats"][("A",)]["features"]


def test_run_batch_exclude_marks_but_keeps_features(patched, tmp_path):
    """제외해도 피쳐와 z는 계산된다 — 한 세트의 관찰로 피쳐를 빼지 않는다."""
    out = rb.run_batch(patched, outdir=tmp_path, exclude=["C"], verbose=False)
    l3 = out["l3"]
    c = l3[l3["category"] == "C"]
    assert len(c) > 0 and not c["judged"].any()
    assert np.isfinite(c["z_top3_kth"]).any()      # 여전히 계산돼 있다


def test_run_batch_injection_test(patched, tmp_path):
    out = rb.run_batch(patched, outdir=tmp_path, verbose=False)
    res = rb.injection_test(patched[0], out["l3_stats"], out["l2_stats"],
                            categories=["A"],
                            cases=[("shift_one", 5.0), ("all_shift", 5.0),
                                   ("drop_one", 0.0)],
                            verbose=False)
    rows = {r["kind"]: r for _, r in res.iterrows()}
    assert set(rows) == {"none", "shift_one", "all_shift", "drop_one"}
    assert rows["shift_one"]["l3_rise"] > 2.0
    # 시퀀스 전체 이동은 L2가, CD 누락은 L3가 원리적으로 못 본다
    assert rows["all_shift"]["l2_rise"] > rows["none"]["l2_rise"]
    assert rows["drop_one"]["l3_rise"] == pytest.approx(0.0)
    assert rows["drop_one"]["n_cd"] == rows["none"]["n_cd"] - 1
