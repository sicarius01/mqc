import numpy as np
import pytest

from cdqc.synth.generator import SynthParams, generate_dataset


@pytest.fixture(scope="module")
def tiny():
    sp = SynthParams(n_images=2, image_size=(256, 256), cds_per_category=10,
                     n_categories=2, n_images_per_case=1,
                     inject={"edge_jump_nm": [0, 2], "missing_frac": [0, 0.2]})
    return sp, generate_dataset(sp)


def test_generate_deterministic(tiny):
    sp, (rec1, imgs1, masks1) = tiny
    rec2, imgs2, masks2 = generate_dataset(sp)
    assert rec1.equals(rec2)
    assert all(np.array_equal(imgs1[k], imgs2[k]) for k in imgs1)
    assert all(np.array_equal(masks1[k], masks2[k]) for k in masks1)


def test_edge_jump_moves_only_affected_sx(tiny):
    _, (df, _, _) = tiny
    base = df[(df.injected_failure == "edge_jump") & (df.sev_rank == 0)]
    jump = df[(df.injected_failure == "edge_jump") & (df.sev_rank == 1)]
    aff = jump[jump.affected == 1]
    assert len(aff) > 0
    j = aff.merge(base[base.affected == 1], on=["category_id", "cd_index"],
                  suffixes=("", "_b"))
    if len(j):
        d = (j["sx"] - j["sx_b"]).to_numpy()
        assert np.all(d > 2.5)   # 2nm / 0.5nmpx = 4px 이동 (지터 ±0.3px×2)


def test_missing_drops_rows(tiny):
    _, (df, _, _) = tiny
    n0 = len(df[(df.injected_failure == "missing") & (df.sev_rank == 0)])
    n1 = len(df[(df.injected_failure == "missing") & (df.sev_rank == 1)])
    assert n1 < n0


def test_masks_match_bands(tiny):
    """참 마스크가 밴드(카테고리) 정의와 일치 — 좌표가 경계 근처."""
    _, (df, imgs, masks) = tiny
    base = df[df.injected_failure == "none"]
    for (iid, cat), g in base.groupby(["image_id", "category_id"]):
        m = masks[(iid, cat)]
        assert m.shape == imgs[iid].shape and m.dtype == bool
        row = g.iloc[0]
        mid = (int(round((row.sy + row.ey) / 2)),
               int(round((row.sx + row.ex) / 2)))
        assert m[mid]                     # 세그먼트 중점은 마스크 내부


def test_value_is_truth_with_mixed_units(tiny):
    """value = 참 길이 × px_nm — 단위 Å/nm 혼합, 지터·주입과 무관하게 참값."""
    _, (df, imgs, _) = tiny
    base = df[df.injected_failure == "none"]
    assert set(base["unit"].unique()) == {"Å", "nm"}
    nm = np.where(base["unit"] == "Å", base["value"] / 10, base["value"])
    cd_px = np.hypot(base["ex"] - base["sx"], base["ey"] - base["sy"])
    # 좌표는 지터 포함, value는 참값 → 비는 0.5 근처에서 지터 폭만큼만 흔들림
    assert np.all(np.abs(nm / cd_px - 0.5) < 0.03)
    for img in imgs.values():
        assert img.dtype == np.uint8 and img.shape == (256, 256)
