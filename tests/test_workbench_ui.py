"""Exercise complete local UI interactions and diagnostic figure fidelity."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from cdqc_workbench import backend as be
from cdqc_workbench import discovery as ds
from cdqc_workbench.plots import image_overlay, profile_plot


APP = Path(__file__).resolve().parents[1] / "cdqc_workbench" / "app.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(ds, "load_config", lambda: ds.default_config())
    return AppTest.from_file(str(APP), default_timeout=60).run()


def clean(app):
    assert not list(app.exception), [e.message for e in app.exception]
    assert not list(app.error), [e.value for e in app.error]


def test_demo_filters_drilldown_keep_analysis_snapshot(app):
    clean(app)
    app.button(key="demo_run").click().run()
    clean(app)
    run_id = app.session_state["analysis"].run_id
    assert app.session_state["run_count"] == 1
    assert len(app.tabs) >= 7
    app.text_input(key="overview_query").set_value("normal").run()
    app.selectbox(key="selected_image").select("shift_04").run()
    app.selectbox(key="selected_cd").select(48).run()
    clean(app)
    assert app.session_state["analysis"].run_id == run_id
    assert app.session_state["run_count"] == 1
    assert app.selectbox(key="selected_cd").value == 48
    app.button(key="analyze_run").click().run()
    clean(app)
    assert app.session_state["run_count"] == 2
    assert app.session_state["analysis"].run_id != run_id


def test_invalid_params_preserve_previous_success(app):
    app.button(key="demo_run").click().run()
    previous = app.session_state["analysis"].run_id
    app.text_area(key="params_json").set_value('{"win_min_px":100,"win_max_px":1}').run()
    app.button(key="analyze_run").click().run()
    assert not list(app.exception)
    assert any("win_min_px" in error.value for error in app.error)
    assert app.session_state["analysis"].run_id == previous
    assert app.session_state["run_count"] == 1


def test_selected_normal_baseline_and_category_to_image(app):
    app.button(key="demo_run").click().run()
    radio = next(r for r in app.radio if r.label == "비교 기준")
    radio.set_value("선택한 정상 이미지").run()
    selection = next(m for m in app.multiselect if m.label == "정상 기준 이미지 ID")
    selection.set_value(["normal_01", "normal_02", "normal_03"]).run()
    app.button(key="analyze_run").click().run()
    clean(app)
    assert app.session_state["analysis"].baseline_ids == ["normal_01", "normal_02", "normal_03"]
    app.selectbox(key="drill_image").select("local_05").run()
    next(b for b in app.button if b.label == "D2에 선택 적용").click().run()
    clean(app)
    assert app.selectbox(key="selected_image").value == "local_05"


def test_rule_change_requires_fresh_scan_but_coordinates_do_not(app, monkeypatch):
    config = ds.default_config()
    config["root"] = "example"
    matches = pd.DataFrame([dict(enabled=True, image_id="abc", image_path="abc.tif", table_path="abc.csv",
                                 seg_path="abc.png", dm3_path="abc.dm3", px_nm=None, status="ready", issues="")])
    app.session_state["matches"] = matches
    app.session_state["scanned_config"] = config
    app.text_input(key="root_path").set_value("different").run()
    clean(app)
    assert app.button(key="analyze_run").disabled
    app.text_input(key="root_path").set_value("example").run()
    clean(app)
    assert not app.button(key="analyze_run").disabled
    next(n for n in app.number_input if n.label.startswith("좌표 X 보정")).set_value(1.0).run()
    clean(app)
    assert not app.button(key="analyze_run").disabled


def test_pixel_focus_uses_native_roi_and_profile_has_windows_and_peaks():
    dataset = be.demo_datasets()[0]
    dataset.img = np.zeros((2400, 2600), np.uint8)
    dataset.labelmap = None
    row = pd.Series(dict(s_x=1200.25, s_y=1100.5, e_x=1250.25, e_y=1100.5, px_nm=.5,
                         delta_s=1., delta_e=-.5))
    rows = pd.DataFrame([row])
    focused = image_overlay(dataset, rows, row, focus=True)
    assert np.diff(np.asarray(focused.data[0].x)).min() == 1
    assert np.asarray(focused.data[0].z).shape[0] < 200
    assert 1200 in focused.data[0].x
    overview = image_overlay(dataset, rows, row, focus=False)
    assert np.diff(np.asarray(overview.data[0].x)).min() > 1
    profile = pd.DataFrame(dict(s_px=[-10, 0, 50, 60], intensity=[0, 1, 1, 0], gradient=[0, 1, -1, 0]))
    profile.attrs.update(length_px=50, s_endpoint_px=0., e_endpoint_px=50., window_half_px=10.)
    figure = profile_plot(profile, row)
    assert any(s.type == "rect" and s.x0 == -10 and s.x1 == 10 for s in figure.layout.shapes)
    assert any(s.type == "line" and s.x0 == 2 for s in figure.layout.shapes)
    assert any(s.type == "line" and s.x0 == 49 for s in figure.layout.shapes)


def test_invalid_coordinate_overlay_keeps_usable_axes():
    dataset = be.demo_datasets()[0]
    row = pd.Series(dict(s_x=np.nan, s_y=10., e_x=30., e_y=10.))
    figure = image_overlay(dataset, pd.DataFrame([row]), row, focus=True)
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.layout.xaxis.range is None


def test_empty_new_scan_disables_previous_data_fallback(app, tmp_path):
    app.button(key="demo_run").click().run()
    previous = app.session_state["analysis"].run_id
    app.text_input(key="root_path").set_value(str(tmp_path)).run()
    next(b for b in app.button if b.label == "탐색 · 연결 미리보기").click().run()
    clean(app)
    assert app.session_state["matches"].empty
    assert app.button(key="analyze_run").disabled
    assert app.session_state["analysis"].run_id == previous
    assert app.session_state["run_count"] == 1


def test_imported_configuration_applies_params_before_widgets(app):
    from cdqc_workbench.app import validated_import
    cfg = ds.default_config()
    cfg["params"] = {"grad_sigma_px": 1.75}
    app.session_state["pending_import_config"] = validated_import(cfg)
    app.run()
    clean(app)
    import json
    assert json.loads(app.text_area(key="params_json").value)["grad_sigma_px"] == 1.75
    app.button(key="demo_run").click().run()
    clean(app)
    assert app.session_state["analysis"].params.grad_sigma_px == 1.75


def test_selection_positions_map_to_original_row_after_filtering():
    from cdqc_workbench.app import selected_source_index
    candidates = pd.DataFrame({"value": [8., 2., 5.]}, index=[81, 12, 40]).sort_values("value")
    assert selected_source_index(candidates, [0]) == 12
    assert selected_source_index(candidates, [2]) == 81
    assert selected_source_index(candidates, [9]) is None


def test_table_value_click_shows_inline_roi_and_links_exact_cd(app):
    app.button(key="demo_run").click().run()
    analysis = app.session_state["analysis"]
    key = f"value_rows_z_max_{hash(tuple(analysis.l3.index))}"
    app.session_state[key] = {"selection": {"rows": [7], "columns": [], "cells": []}}
    app.session_state["distribution_selection_kind"] = "table"
    app.run()
    clean(app)
    assert app.button(key="distribution_apply")
    assert len(app.get("plotly_chart")) >= 6
    app.button(key="distribution_apply").click().run()
    clean(app)
    assert app.selectbox(key="selected_cd").value == analysis.l3.index[7]


def test_missing_role_cells_mean_empty_suffix_and_same_directory():
    from cdqc_workbench.app import role_config_from_frame
    data = pd.DataFrame([dict(role="table", directory=None, extension="xlsx", suffix=np.nan)])
    assert role_config_from_frame(data) == {"table": {"directory": None, "extension": "xlsx", "suffix": ""}}
