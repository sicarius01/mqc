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


def _write_records(tmp_path, rows, columns=None):
    import pandas as pd
    df = pd.DataFrame(rows)
    p = tmp_path / "data" / "records.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def _row(**kw):
    base = dict(recipe_id="r1", image_id="i1", image_path="i1.png",
                category_id="A", cd_index=0, sx=1.0, sy=2.0, ex=3.0, ey=4.0,
                px_nm=0.5)
    base.update(kw)
    return base


def test_load_records_and_mapping(tmp_path):
    _write_records(tmp_path, [_row(cd_index=0), _row(cd_index=1)])
    cfg = load_config(None, root=str(tmp_path))
    df = io.load_records(cfg)
    assert df.height == 2
    assert df["cd_index"].dtype.is_integer()


def test_duplicate_cd_index_rejected(tmp_path):
    _write_records(tmp_path, [_row(), _row()])
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-06"


def test_px_nm_inconsistency_rejected(tmp_path):
    _write_records(tmp_path, [_row(cd_index=0), _row(cd_index=1, px_nm=0.7)])
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-07"


def test_missing_column_rejected(tmp_path):
    rows = [_row()]
    for r in rows:
        del r["px_nm"]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.load_records(cfg)
    assert e.value.code == "E-DATA-02"


def test_column_mapping(tmp_path):
    rows = [dict(RID="r1", IID="i1", PATH="i1.png", CAT="A", IDX=0,
                 SX=1.0, SY=2.0, EX=3.0, EY=4.0, PX=0.5)]
    _write_records(tmp_path, rows)
    cfg = load_config(None, root=str(tmp_path), set_overrides=[
        "data.columns.recipe_id=RID", "data.columns.image_id=IID",
        "data.columns.image_path=PATH", "data.columns.category_id=CAT",
        "data.columns.cd_index=IDX", "data.columns.sx=SX",
        "data.columns.sy=SY", "data.columns.ex=EX", "data.columns.ey=EY",
        "data.columns.px_nm=PX"])
    df = io.load_records(cfg)
    assert df["recipe_id"][0] == "r1"
    assert df["px_nm"][0] == 0.5


def test_convention_auto_requires_doctor(tmp_path):
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        io.get_convention(cfg)
    assert e.value.code == "E-CONV-02"
