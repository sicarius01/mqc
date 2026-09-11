"""Directory → real image files → NASCA reader → analysis → GUI integration.

Only external Excel COM and DM3 decoder boundaries are replaced. The XLSX files
are deliberately non-ZIP placeholders: this verifies the original path reaches
Excel instead of silently succeeding through a generic workbook reader. This
does not certify a real corporate Excel/DRM installation or real DM3 decoding.
"""
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from cdqc_workbench import discovery as ds


APP = Path(__file__).resolve().parents[1] / "cdqc_workbench" / "app.py"


@pytest.fixture
def four_file_flow(tmp_path, monkeypatch):
    root = tmp_path / "사내 측정 폴더"
    headers = ("LineBeginX", "LineBeginY", "LineEndX", "LineEndY",
               "MetrologyActivityName", "Measurement", "MeasurementUnit")
    measurements = tuple(((-20) * .5e-9, -(y-32) * .5e-9, 5 * .5e-9,
                          -(y-32) * .5e-9, category, 12.5e-9, "m")
                         for category in ("A", "B") for y in range(15, 45, 5))
    xlsx_paths = []
    for directory in (root / "lotA", root / "lotB" / "nested"):
        (directory / "result").mkdir(parents=True)
        image = np.zeros((64, 80), np.uint16)
        image[:, 20:45] = 2000
        labels = np.zeros((64, 80), np.uint8)
        labels[:, 20:45] = 1
        for path, data in ((directory / "abc.tif", image),
                           (directory / "result" / "abc.png", labels)):
            ok, encoded = cv2.imencode(path.suffix, data)
            assert ok
            encoded.tofile(str(path))
        (directory / "abc.dm3").write_bytes(b"DM3 decoder boundary fixture")
        xlsx = directory / "result" / "abc_Result.xlsx"
        xlsx.write_bytes(b"Original protected-workbook placeholder, not ZIP")
        xlsx_paths.append(str(xlsx.resolve()))

    events, failed_paths = [], set()
    pythoncom = ModuleType("pythoncom")
    pythoncom.COINIT_APARTMENTTHREADED = 2
    pythoncom.CoInitializeEx = lambda flag: events.append(("initialize", flag))
    pythoncom.CoUninitialize = lambda: events.append(("uninitialize",))
    client = ModuleType("win32com.client")
    win32com = ModuleType("win32com")
    win32com.client = client

    class Workbook:
        def Worksheets(self, index):
            assert index == 1
            used = SimpleNamespace(Row=3, Column=1, Rows=SimpleNamespace(Count=13),
                Columns=SimpleNamespace(Count=7), Value2=(headers,) + measurements)
            return SimpleNamespace(Name="NASCA measurements", UsedRange=used)

        def Close(self, **kwargs):
            assert kwargs == {"SaveChanges": False}
            events.append(("close",))

    class Excel:
        def __init__(self):
            self.Workbooks = self

        def Open(self, **kwargs):
            path = kwargs["Filename"]
            events.append(("open", path))
            assert Path(path).is_absolute() and Path(path).is_file()
            assert kwargs["ReadOnly"] is True
            if path in failed_paths:
                raise RuntimeError("Simulated company Excel open failure")
            return Workbook()

        def Quit(self):
            events.append(("quit",))

    def dispatch(name):
        assert name == "Excel.Application"
        events.append(("dispatch", name))
        return Excel()

    client.DispatchEx = dispatch
    for name, module in (("pythoncom", pythoncom), ("win32com", win32com), ("win32com.client", client)):
        monkeypatch.setitem(sys.modules, name, module)
    dm = pytest.importorskip("ncempy.io.dm")

    def read_dm(path):
        events.append(("dm", path))
        assert Path(path).is_file()
        return dict(data=np.zeros((64, 80)), pixelSize=[.5, .5], pixelUnit=["nm", "nm"])

    monkeypatch.setattr(dm, "dmReader", read_dm)
    monkeypatch.setattr(ds, "load_config", lambda: ds.default_config())
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.text_input(key="root_path").set_value(str(root)).run()
    next(b for b in app.button if b.label == "탐색 후 전체 분석").click().run()
    return app, events, failed_paths, xlsx_paths


def test_nested_four_files_through_nasca_com_reader_and_gui(four_file_flow):
    app, events, _, paths = four_file_flow
    assert not list(app.exception)
    assert not list(app.error)
    result = app.session_state["analysis"]
    assert [d.image_id for d in result.datasets] == ["lotA/abc", "lotB/nested/abc"]
    assert len(result.l3) == 24
    assert result.l3.groupby(["image_id", "category"]).size().tolist() == [6, 6, 6, 6]
    assert [e[1] for e in events if e[0] == "open"] == paths
    assert len([e for e in events if e[0] == "dm"]) == 2
    assert len([e for e in events if e[0] == "quit"]) == 2
    for data in result.datasets:
        assert data.records.source_row.tolist() == list(range(4, 16))
        np.testing.assert_allclose(data.records.sx, 20)
        np.testing.assert_allclose(data.records.ex, 45)
        np.testing.assert_allclose(data.records.value_nm, 12.5)
        assert data.labelmap.shape == data.img.shape == (64, 80)
        assert data.px_nm == .5
    app.selectbox(key="selected_image").select("lotB/nested/abc").run()
    app.selectbox(key="selected_category").select("B").run()
    app.selectbox(key="selected_cd").select(19).run()
    assert not list(app.exception)
    assert not list(app.error)
    assert app.selectbox(key="selected_cd").value == 19
    row = result.l3.loc[19]
    assert (row.image_id, row.category, row.source_row) == ("lotB/nested/abc", "B", 11)
    assert app.session_state["run_count"] == 1
    assert len(app.get("plotly_chart")) >= 5
    assert len([e for e in events if e[0] == "open"]) == 2  # Filters do not reload.


def test_excel_partial_and_total_failure_keep_traceable_result(four_file_flow):
    app, _, failures, paths = four_file_flow
    failures.add(paths[1])
    app.button(key="analyze_run").click().run()
    assert not list(app.exception)
    assert app.session_state["run_count"] == 2
    assert [d.image_id for d in app.session_state["analysis"].datasets] == ["lotA/abc"]
    assert len(app.session_state["load_issues"]) == 1
    assert "lotB/nested/abc" == app.session_state["load_issues"][0]["image_id"]
    snapshot = {key: deepcopy(app.session_state[key]) for key in
                ("run_manifest", "run_config", "run_scan_config", "load_issues", "run_count")}
    run_id = app.session_state["analysis"].run_id
    failures.update(paths)
    app.button(key="analyze_run").click().run()
    assert not list(app.exception)
    assert app.session_state["analysis"].run_id == run_id
    for key, value in snapshot.items():
        assert app.session_state[key] == value, key
    failed = app.session_state["failed_attempt"]
    assert len(failed["manifest"]) == 2
    assert len(failed["load_issues"]) == 2
    assert all("Simulated company Excel open failure" in item["message"] for item in failed["load_issues"])
    assert any("실패한 최근 시도" in item.value for item in app.warning)
