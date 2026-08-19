import numpy as np
import pytest

from cdqc import io
from cdqc.config import load_config
from cdqc.errors import CdqcError


def test_transform_identity():
    conv = {"convention": "xy", "origin": "zero", "y_flip": False, "scale": 1.0}
    x, y = io.transform_coords(np.array([10.0]), np.array([20.0]), conv, (100, 200))
    assert x[0] == 10.0 and y[0] == 20.0


def test_transform_rowcol_origin_flip():
    conv = {"convention": "rowcol", "origin": "one", "y_flip": True, "scale": 1.0}
    # 원본 (a,b) = (row=21, col=11), origin 1 → row=20, col=10 → y_flip(H=100) → y=79
    x, y = io.transform_coords(np.array([21.0]), np.array([11.0]), conv, (100, 200))
    assert x[0] == 10.0
    assert y[0] == 79.0


def test_transform_scale():
    conv = {"convention": "xy", "origin": "zero", "y_flip": False, "scale": 2.0}
    x, y = io.transform_coords(np.array([5.0]), np.array([7.0]), conv, (100, 100))
    assert x[0] == 10.0 and y[0] == 14.0


def test_normalize_unit_variants():
    # Å: U+00C5, U+212B, NFD(A+U+030A), ASCII, 이름
    for s in ["Å", "Å", "Å", "A", "a", "Angstrom", " ang "]:
        assert io.normalize_unit(s) == "angstrom", repr(s)
    for s in ["nm", "NM", "nanometer", "Nanometre"]:
        assert io.normalize_unit(s) == "nm"
    assert io.normalize_unit("furlong") is None
    assert io.normalize_unit("") is None


# ---------------------------------------------------------------- 로더

def _write_records(tmp_path, rows, name="records.csv"):
    import pandas as pd
    df = pd.DataFrame(rows)
    p = tmp_path / "data" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def _row(**kw):
    base = dict(image_id="i1", image_path="i1.png", category_id="A",
                sx=1.0, sy=2.0, ex=3.0, ey=4.0)
    base.update(kw)
    if "value" not in kw:   # 명시 안 하면 px_nm=0.5 기준 참값
        base["value"] = float(np.hypot(base["ex"] - base["sx"],
                                       base["ey"] - base["sy"])) * 0.5
    return base


def test_load_minimal_schema(tmp_path):
    """필수 8컬럼만으로 로드 — recipe_id/cd_index/px_nm 자동 생성."""
    _write_records(tmp_path, [_row(sy=2.0), _row(sy=12.0)])
    cfg = load_config(None, root=str(tmp_path))
    df = io.load_records(cfg)
    assert df.height == 2
    assert df["recipe_id"].unique().to_list() == ["R1"]
    assert df["cd_index"].to_list() == [0, 1]
    assert np.allclose(df["px_nm"].to_numpy(), 0.5, atol=1e-9)
    assert np.allclose(df["value_nm"].to_numpy(), df["value"].to_numpy())


def test_px_nm_inversion_exact(tmp_path):
    """역산 px_nm = 이미지별 value/기하 비 중앙값 — 참값과 1e-6 내 일치."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        sx, sy = rng.uniform(0, 50, 2)
        ex, ey = sx + rng.uniform(20, 60), sy + rng.uniform(-3, 3)
        rows.append(_row(sx=sx, sy=sy, ex=ex, ey=ey,
                         value=float(np.hypot(ex - sx, ey - sy)) * 0.73))
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path))
    df = io.load_records(cfg)
    assert np.all(np.abs(df["px_nm"].to_numpy() - 0.73) < 1e-6)


def test_unit_column_angstrom_variants(tmp_path):
    rows = [
        _row(sy=2.0, value=14.142135623730951, unit="Å"),   # Å = nm*10
        _row(sy=12.0, value=14.142135623730951, unit="Å"),  # ANGSTROM SIGN
        _row(sy=22.0, value=1.4142135623730951, unit="nm"),
    ]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path),
                      set_overrides=["data.columns.unit=unit"])
    df = io.load_records(cfg)
    assert np.allclose(df["value_nm"].to_numpy(), 1.4142135623730951)


def test_unknown_unit_aborts_with_list(tmp_path):
    rows = [_row(sy=2.0, unit="nm"), _row(sy=12.0, unit="furlong")]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path),
                      set_overrides=["data.columns.unit=unit"])
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-08"
    assert "furlong" in str(e.value)


def test_value_unit_fallback_angstrom(tmp_path):
    _write_records(tmp_path, [_row(value=14.142135623730951)])
    cfg = load_config(None, root=str(tmp_path),
                      set_overrides=["data.value_unit=angstrom"])
    df = io.load_records(cfg)
    assert abs(df[0, "value_nm"] - 1.4142135623730951) < 1e-12
    assert abs(df[0, "px_nm"] - 0.5) < 1e-9


def test_row_order_cd_index_multifile_interleaved(tmp_path):
    """여러 파일(파일명 정렬 순) + 섞인 카테고리에서 row_order cd_index."""
    f1 = [_row(category_id="A", sy=2.0), _row(category_id="B", sy=2.0),
          _row(category_id="A", sy=12.0), _row(category_id="B", sy=12.0)]
    f2 = [_row(category_id="A", sy=22.0), _row(category_id="B", sy=22.0)]
    _write_records(tmp_path, f1, name="01.csv")
    _write_records(tmp_path, f2, name="02.csv")
    cfg = load_config(None, root=str(tmp_path))
    df = io.load_records(cfg).sort(["category_id", "cd_index"])
    a = df.filter(df["category_id"] == "A")
    assert a["cd_index"].to_list() == [0, 1, 2]
    assert a["sy"].to_list() == [2.0, 12.0, 22.0]   # 파일 순 → 행 순
    b = df.filter(df["category_id"] == "B")
    assert b["cd_index"].to_list() == [0, 1, 2]


def test_cd_index_column_mode_duplicate_rejected(tmp_path):
    rows = [_row(cd_index=0, sy=2.0), _row(cd_index=0, sy=12.0)]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path),
                      set_overrides=["data.cd_index_source=column"])
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-06"


def test_px_nm_column_priority_and_consistency(tmp_path):
    """px_nm 컬럼이 있으면 역산 대신 우선 사용 + 이미지 내 일관성 검증."""
    rows = [_row(px_nm=0.9, sy=2.0, value=100.0), _row(px_nm=0.9, sy=12.0)]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path))
    df = io.load_records(cfg)
    assert df["px_nm"].to_list() == [0.9, 0.9]   # value로 역산하지 않음

    rows = [_row(px_nm=0.9, sy=2.0), _row(px_nm=0.7, sy=12.0)]
    _write_records(tmp_path, rows)
    with pytest.raises(CdqcError) as e:
        io.load_records(load_config(None, root=str(tmp_path)))
    assert e.value.code == "E-DATA-07"


def test_missing_value_column_rejected(tmp_path):
    rows = [_row()]
    for r in rows:
        del r["value"]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-02"


def test_recipe_id_column_used_when_present(tmp_path):
    _write_records(tmp_path, [_row(recipe_id="RCP7")])
    cfg = load_config(None, root=str(tmp_path))
    df = io.load_records(cfg)
    assert df["recipe_id"].to_list() == ["RCP7"]


def test_column_mapping(tmp_path):
    rows = [dict(IID="i1", PATH="i1.png", CAT="A",
                 SX=1.0, SY=2.0, EX=3.0, EY=4.0, VAL=1.4142135623730951)]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path), set_overrides=[
        "data.columns.image_id=IID", "data.columns.image_path=PATH",
        "data.columns.category_id=CAT", "data.columns.sx=SX",
        "data.columns.sy=SY", "data.columns.ex=EX", "data.columns.ey=EY",
        "data.columns.value=VAL"])
    df = io.load_records(cfg)
    assert df["image_id"][0] == "i1"
    assert abs(df[0, "px_nm"] - 0.5) < 1e-9


def test_zero_length_segments_only_rejected(tmp_path):
    _write_records(tmp_path, [_row(ex=1.0, ey=2.0, value=5.0)])  # 길이 0
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-10"


def test_convention_auto_requires_doctor(tmp_path):
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.get_convention(cfg)
    assert e.value.code == "E-CONV-02"
