from pathlib import Path

import pytest

from cdqc_workbench.discovery import default_config, load_config, save_config, scan_directory, validate_config


def _touch(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def _config(root, **settings):
    return {**default_config(), "root": str(root), "match_mode": "stem", **settings}


def test_nested_stem_matching_unicode_and_required_roles(tmp_path):
    for directory, suffix in [("images/a", "tif"), ("tables/b", "xlsx"),
                              ("masks", "png"), ("raw/x/y", "dm3")]:
        _touch(tmp_path, f"{directory}/측정01.{suffix}")
    row = scan_directory(_config(tmp_path)).iloc[0]
    assert row.image_id == "측정01" and row.status == "ready" and row.enabled
    assert len(row.match_details) == 4
    assert {item["key"] for item in row.match_details} == {"측정01"}
    assert Path(row.dm3_path).name == "측정01.dm3"


def test_duplicates_are_never_silently_selected(tmp_path):
    for relative in ["a/one.tif", "b/one.tif", "one.csv", "one.png", "one.dm3"]:
        _touch(tmp_path, relative)
    row = scan_directory(_config(tmp_path)).iloc[0]
    assert row.image_path == "" and len(row.image_candidates) == 2
    assert not row.enabled and "Ambiguous image" in row.issues


def test_missing_roles_optional_and_manual_scale(tmp_path):
    _touch(tmp_path, "one.tiff")
    _touch(tmp_path, "one.csv")
    row = scan_directory(_config(tmp_path)).iloc[0]
    assert row.status == "error" and "required dm3" in row.issues
    row = scan_directory(_config(tmp_path, require_seg=False, require_dm3=False, px_nm=0.4)).iloc[0]
    assert row.status == "ready" and row.enabled and row.px_nm == 0.4
    assert "optional seg" in row.issues and "optional dm3" in row.issues


def test_role_regex_with_relative_paths_and_key_errors(tmp_path):
    for relative in ["images/run01_image.tif", "tables/run01_table.csv", "seg/run01_mask.png",
                     "raw/run01_source.dm3", "images/unmatched.tif"]:
        _touch(tmp_path, relative)
    config = _config(tmp_path, match_mode="regex", key_source="relative_path",
                     key_regex=r"/(?P<key>run\d+)_", role_key_regex={"dm3": r"(?P<key>run\d+)_source"})
    frame = scan_directory(config)
    valid = frame[frame.status == "ready"].iloc[0]
    assert valid.image_id == "run01"
    assert frame.status.tolist().count("key_error") == 1
    assert "images/unmatched.tif" in frame[frame.status == "key_error"].iloc[0].issues
    assert any(item["source"] == "images/run01_image.tif" for item in valid.match_details)


def test_folder_matching_and_nonrecursive_scan(tmp_path):
    for relative in ["sample/source.tif", "sample/table.csv", "sample/labels.png", "sample/raw.dm3"]:
        _touch(tmp_path, relative)
    frame = scan_directory(_config(tmp_path, match_mode="folder"))
    assert frame.iloc[0].image_id == "sample" and frame.iloc[0].enabled
    assert scan_directory(_config(tmp_path, recursive=False)).empty


def test_ignores_outputs_hidden_dirs_temp_files_and_user_globs(tmp_path):
    for relative in ["run.tif", "run.csv", "run.png", "run.dm3", "out/run.csv", ".venv/run.csv",
                     ".secret/run.csv", "logs/generated.csv", "~$run.xlsx"]:
        _touch(tmp_path, relative)
    frame = scan_directory(_config(tmp_path, ignored_patterns=["logs/**"]))
    assert len(frame) == 1 and frame.iloc[0].status == "ready"
    assert frame.attrs["files_scanned"] == 4


def test_globs_allow_role_subfolders_and_case_insensitive_extensions(tmp_path):
    for relative in ["x/images/run.TIF", "tables/run.CSV", "run.PNG", "run.DM3", "x/skip.tif"]:
        _touch(tmp_path, relative)
    frame = scan_directory(_config(tmp_path, patterns={"image": "**/images/*.tif"}))
    assert len(frame) == 1 and frame.iloc[0].enabled
    assert frame.attrs["unmatched_files"] == ["x/skip.tif"]


def test_role_overlap_rejected(tmp_path):
    for relative in ["run.png", "run.csv", "run.dm3"]:
        _touch(tmp_path, relative)
    row = scan_directory(_config(tmp_path, patterns={"image": "*.png"})).iloc[0]
    assert not row.enabled and "multiple roles" in row.issues


def test_config_roundtrip_preserves_options_without_writing_elsewhere(tmp_path):
    path = tmp_path / "settings" / "rules.json"
    config = _config(tmp_path, options={"coords": {"swap_xy": True}, "baseline_ids": ["측정01"]})
    assert save_config(config, path) == path
    assert load_config(path) == config
    assert list(path.parent.iterdir()) == [path]
    assert load_config(tmp_path / "absent.json") == default_config()


@pytest.mark.parametrize("settings,match", [
    ({"key_regex": "("}, "invalid regular expression"),
    ({"key_regex": "(.+)"}, "named key group"),
    ({"match_mode": "regex"}, "needs key_regex"),
    ({"px_nm": -1}, "finite positive"),
    ({"version": 2}, "Unsupported"),
    ({"recursive": "yes"}, "true or false"),
])
def test_invalid_config_diagnostics(settings, match):
    with pytest.raises(ValueError, match=match):
        validate_config(settings)


def test_missing_root_empty_scan_and_invalid_json(tmp_path):
    with pytest.raises(ValueError, match="Select a root"):
        scan_directory(default_config())
    with pytest.raises(ValueError, match="does not exist"):
        scan_directory(_config(tmp_path / "missing"))
    empty = scan_directory(_config(tmp_path))
    assert empty.empty and "image_path" in empty.columns
    path = tmp_path / "bad.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load"):
        load_config(path)


def _layout_files(root, base="", name="abc"):
    prefix = f"{base}/" if base else ""
    for relative in [f"{name}.dm3", f"{name}.tif", f"result/{name}.png", f"result/{name}_Result.xlsx"]:
        _touch(root, prefix + relative)


@pytest.mark.parametrize("recursive", [True, False])
def test_default_layout_matches_measurement_and_result_directory(tmp_path, recursive):
    _layout_files(tmp_path)
    config = {**default_config(), "root": str(tmp_path), "recursive": recursive}
    row = scan_directory(config).iloc[0]
    assert row.image_id == "abc" and row.status == "ready" and row.enabled
    assert Path(row.table_path) == tmp_path / "result" / "abc_Result.xlsx"
    assert Path(row.seg_path) == tmp_path / "result" / "abc.png"
    table_detail = next(item for item in row.match_details if item["role"] == "table")
    assert table_detail["measurement_base"] == "."
    assert table_detail["stripped_stem"] == "abc"
    assert table_detail["suffix"] == "_Result"
    assert config["options"]["seg_restore"] is True


def test_layout_preserves_nested_measurement_base_identity(tmp_path):
    _layout_files(tmp_path, "lotA")
    _layout_files(tmp_path, "lotB/deeper")
    frame = scan_directory({**default_config(), "root": str(tmp_path)})
    assert frame.image_id.tolist() == ["lotA/abc", "lotB/deeper/abc"]
    assert frame.enabled.all()
    assert all(len(paths) == 1 for paths in frame.table_candidates)


def test_layout_nonrecursive_ignores_nested_measurement_bases(tmp_path):
    _layout_files(tmp_path)
    _layout_files(tmp_path, "lotA")
    frame = scan_directory({**default_config(), "root": str(tmp_path), "recursive": False})
    assert frame.image_id.tolist() == ["abc"] and frame.enabled.all()


def test_layout_none_and_empty_directory_mean_measurement_base(tmp_path):
    config = default_config()
    config["root"] = str(tmp_path)
    config["roles"]["seg"]["directory"] = None
    config["roles"]["table"]["directory"] = ""
    for relative in ["abc.dm3", "abc.tif", "abc.png", "abc_Result.xlsx"]:
        _touch(tmp_path, relative)
    frame = scan_directory(config)
    assert frame.image_id.tolist() == ["abc"] and frame.enabled.all()


def test_layout_literal_suffixes_case_and_multisegment_directory(tmp_path):
    config = {**default_config(), "root": str(tmp_path), "recursive": False}
    config["roles"] = {
        "image": {"directory": None, "extension": ".tif", "suffix": "_image"},
        "dm3": {"directory": "raw", "extension": ".dm3", "suffix": "(raw)"},
        "seg": {"directory": "result/labels", "extension": ".png", "suffix": "[mask]+"},
        "table": {"directory": "result/tables", "extension": ".xlsx", "suffix": "_Result"},
    }
    for relative in ["Abc_IMAGE.TIF", "raw/abc(RAW).DM3", "RESULT/LABELS/ABC[mask]+.PNG",
                     "result/tables/aBc_result.XLSX"]:
        _touch(tmp_path, relative)
    frame = scan_directory(config)
    assert len(frame) == 1 and frame.iloc[0].enabled
    assert frame.iloc[0].image_id.casefold() == "abc"


def test_layout_wrong_directory_or_suffix_does_not_match(tmp_path):
    for relative in ["abc.tif", "abc.dm3", "notresult/abc.png", "result/abc_Result_extra.xlsx"]:
        _touch(tmp_path, relative)
    frame = scan_directory({**default_config(), "root": str(tmp_path)})
    assert len(frame) == 1 and not frame.iloc[0].enabled
    assert "Missing required seg" in frame.iloc[0].issues
    assert "Missing required table" in frame.iloc[0].issues
    details = frame.attrs["unmatched_details"]
    assert any("directory" in item["reason"] for item in details)
    assert any("literal suffix" in item["reason"] for item in details)


@pytest.mark.parametrize("directory", ["../result", "/absolute", "C:\\mts\\result", "a/../b", "\\\\server\\share"])
def test_layout_rejects_absolute_and_parent_directories(directory):
    with pytest.raises(ValueError, match="must be relative"):
        validate_config({"roles": {"seg": {"directory": directory}}})
