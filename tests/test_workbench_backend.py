"""Real file and numerical contract checks for the local diagnostic workbench."""
from dataclasses import asdict, replace
import json

import cv2
import numpy as np
import pandas as pd
import pytest

import cdqc
from cdqc.evidence import gradient_of
from cdqc.geometry import mean_edge_tangent, unit_tangents
from cdqc.sampling import sample_ribbon_profiles
from cdqc_workbench.backend import (analyze, demo_datasets, export_stats,
    feature_diagnostics, import_stats, load_dataset, profile_table, read_table,
    validate_params)


def files(tmp_path, records=None):
    folder = tmp_path / '한글 데이터'
    folder.mkdir(exist_ok=True)
    image = np.zeros((64, 80), np.uint16)
    image[:, 20:45] = 2000
    image_path = folder / '실제.tif'
    ok, encoded = cv2.imencode('.tif', image)
    assert ok
    encoded.tofile(str(image_path))
    if records is None:
        records = pd.DataFrame(dict(sx=[20]*6, sy=np.arange(6)*5+15,
                                    ex=[45]*6, ey=np.arange(6)*5+15, category='폭', value_nm=12.5))
    table_path = folder / '측정.csv'
    records.to_csv(table_path, index=False, encoding='cp949')
    return image_path, table_path


def test_unicode_real_tiff_csv_and_missing_seg(tmp_path):
    image, table = files(tmp_path)
    data = load_dataset(image, table, px_nm=.5)
    assert data.img.dtype == np.uint8
    assert data.records.source_row.tolist() == [2, 3, 4, 5, 6, 7]
    assert data.records.analysis_valid.all()
    assert data.records.category.unique().tolist() == ['폭']
    assert {'NO_SEGMENTATION', 'UINT8_CONVERSION'} <= {i['code'] for i in data.issues}
    result = analyze([data])
    np.testing.assert_allclose(result.l3.cd_nm, 12.5)


def test_read_xlsx_uses_nasca_com_original_path(tmp_path, monkeypatch):
    from cdqc import func
    from unittest.mock import Mock
    original = pd.DataFrame({'category': ['한글'], 'sx': [12]})
    path = tmp_path / '측정.xlsx'
    reader = Mock(return_value=original)
    monkeypatch.setattr(func, 'read_nasca_csv', reader)
    monkeypatch.setattr(type(path), 'read_bytes', Mock(side_effect=AssertionError('no direct workbook bytes')))
    monkeypatch.setattr(pd, 'read_excel', Mock(side_effect=AssertionError('no openpyxl fallback')))
    pd.testing.assert_frame_equal(read_table(path), original)
    reader.assert_called_once_with(path, visible=False, header=True)
    with pytest.raises(ValueError, match='원본 파일 경로'):
        read_table(b'protected data', '측정.xlsx')


def test_excel_source_row_and_reader_diagnostics(tmp_path, monkeypatch):
    from cdqc import func
    image, csv = files(tmp_path)
    table = pd.read_csv(csv, encoding='cp949')
    table.attrs.update(source_data_start_row=5, sheet_name='NASCA', table_reader='Excel COM')
    monkeypatch.setattr(func, 'read_nasca_csv', lambda *args, **kwargs: table)
    result = load_dataset(image, tmp_path / '측정.xlsx', px_nm=.5)
    assert result.records.source_row.tolist() == [5, 6, 7, 8, 9, 10]
    assert result.metadata['table_reader'] == 'Excel COM'
    assert result.metadata['table_source']['sheet_name'] == 'NASCA'


def test_csv_bytes_still_supported():
    result = read_table('category,sx\n폭,12\n'.encode('cp949'), '측정.csv')
    pd.testing.assert_frame_equal(result, pd.DataFrame({'category': ['폭'], 'sx': [12]}))


def test_duplicate_excel_headers_report_ambiguous_mapping(tmp_path, monkeypatch):
    from cdqc import func
    image, csv = files(tmp_path)
    table = pd.read_csv(csv, encoding='cp949')
    table.columns = ['sx', 'sy', 'ex', 'ey', 'category', 'sx']
    monkeypatch.setattr(func, 'read_nasca_csv', lambda *args, **kwargs: table)
    with pytest.raises(ValueError, match='중복 컬럼명'):
        load_dataset(image, tmp_path / '측정.xlsx', px_nm=.5)


def test_empty_unnamed_excel_formatting_columns_do_not_block_data(tmp_path, monkeypatch):
    from cdqc import func
    image, csv = files(tmp_path)
    table = pd.read_csv(csv, encoding='cp949')
    columns = list(table.columns)
    for name in ('blank_a', 'blank_b', 'blank_c', 'blank_d'):
        table[name] = None
    table.columns = columns + [None, None, '', '  ']
    table.attrs.update(source_data_start_row=5, sheet_name='NASCA')
    monkeypatch.setattr(func, 'read_nasca_csv', lambda *args, **kwargs: table)
    result = load_dataset(image, tmp_path / '측정.xlsx', px_nm=.5)
    assert result.records.analysis_valid.all()
    assert result.records.source_row.tolist() == [5, 6, 7, 8, 9, 10]
    assert not any(name in result.records for name in ('None', 'nan', '', '  '))
    # The raw reader frame remains unchanged for source-data diagnostics.
    assert len(table.columns) == len(columns) + 4


def test_populated_unnamed_excel_columns_are_preserved(tmp_path, monkeypatch):
    from cdqc import func
    image, csv = files(tmp_path)
    table = pd.read_csv(csv, encoding='cp949')
    table[None] = ['raw extra'] * len(table)
    table['empty_but_named'] = None
    monkeypatch.setattr(func, 'read_nasca_csv', lambda *args, **kwargs: table)
    result = load_dataset(image, tmp_path / '측정.xlsx', px_nm=.5)
    assert result.records[str(table.columns[6])].tolist() == ['raw extra'] * len(table)
    assert 'empty_but_named' in result.records


def test_old_meters_columns_and_pixel_origin(tmp_path):
    old = pd.DataFrame({'LineBeginX': [-10e-9]*6, 'LineBeginY': [0.]*6,
        'LineEndX': [0.]*6, 'LineEndY': [0.]*6, 'MetrologyActivityName': ['A']*6,
        'Measurement': [10e-9]*6, 'MeasurementUnit': ['m']*6})
    image, table = files(tmp_path, old)
    data = load_dataset(image, table, px_nm=.5)
    np.testing.assert_allclose(data.records.sx, 20)
    np.testing.assert_allclose(data.records.sy, 32)
    np.testing.assert_allclose(data.records.value_nm, 10)
    alternative = load_dataset(image, table, px_nm=.5,
        options={'origin': 'center_pixel', 'offset_x': -1, 'offset_y': 1})
    np.testing.assert_allclose(alternative.records.sx, 18.5)
    np.testing.assert_allclose(alternative.records.sy, 32.5)


def test_invalid_coordinates_categories_retained(tmp_path):
    rec = pd.DataFrame(dict(sx=[20, np.nan, 20, 20, 20, 20], sy=[20]*6,
        ex=[45, 45, 999, 20, 45, 45], ey=[20]*6,
        category=['A', 'A', 'A', 'A', None, 'A']))
    image, table = files(tmp_path, rec)
    data = load_dataset(image, table, px_nm=.5)
    assert len(data.records) == 6
    assert data.records.analysis_valid.sum() == 2
    result = analyze([data])
    assert len(result.l3) == 6
    excluded = result.l3[~result.l3.analysis_valid]
    assert excluded.z_max.isna().all()
    assert (excluded.n_z_valid == 0).all()
    assert result.l3.loc[result.l3.source_row == 4, 'e_x'].iloc[0] == 999
    assert (result.l3.px_nm == .5).all()
    assert {'NONFINITE_COORD', 'E_OUTSIDE', 'ZERO_LENGTH', 'MISSING_CATEGORY'} <= {i['code'] for i in result.issues}


def test_seg_shape_requires_explicit_resize(tmp_path):
    image, table = files(tmp_path)
    path = tmp_path / 'seg.png'
    cv2.imwrite(str(path), np.zeros((32, 40), np.uint8))
    with pytest.raises(ValueError, match='최근접'):
        load_dataset(image, table, seg_path=path, px_nm=.5)
    data = load_dataset(image, table, seg_path=path, px_nm=.5, options={'seg_resize': True})
    assert data.labelmap.shape == data.img.shape
    assert 'SEG_RESIZED' in {i['code'] for i in data.issues}


def test_manual_scale_retains_all_four_file_metadata(tmp_path):
    image, table = files(tmp_path)
    seg = tmp_path / 'seg.png'
    cv2.imwrite(str(seg), np.zeros((64, 80), np.uint8))
    dm3 = tmp_path / '원본.dm3'
    data = load_dataset(image, table, seg_path=seg, dm3_path=dm3, px_nm=.5)
    assert data.metadata['image_file'] == str(image)
    assert data.metadata['table_file'] == str(table)
    assert data.metadata['seg_file'] == str(seg)
    assert data.metadata['dm3_file'] == str(dm3)
    assert data.metadata['scale_source'] == 'manual nm/px'
    assert 'MANUAL_SCALE_OVERRIDE' in {i['code'] for i in data.issues}


@pytest.fixture(scope='module')
def calibrated():
    return analyze(demo_datasets(), baseline_ids=['normal_01', 'normal_02', 'normal_03'])


def test_demo_frozen_roundtrip_does_not_refit(calibrated):
    encoded = export_stats(calibrated)
    assert 'NaN' not in encoded and 'Infinity' not in encoded
    json.loads(encoded)
    restored = analyze(demo_datasets(), frozen_stats=import_stats(encoded))
    zcols = [c for c in calibrated.l3 if c.startswith('z_')]
    pd.testing.assert_frame_equal(calibrated.l3[zcols], restored.l3[zcols])
    shifted_only = analyze([demo_datasets()[3]], frozen_stats=import_stats(encoded))
    original = calibrated.l3[calibrated.l3.image_id == 'shift_04'][zcols].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, shifted_only.l3[zcols])
    assert shifted_only.stats['l3']['demo_CD']['n'] == 48


def test_missing_baseline_never_borrows_test_category(calibrated):
    data = demo_datasets()[0]
    data.records['category'] = 'UNSEEN'
    result = analyze([data], frozen_stats=import_stats(export_stats(calibrated)))
    assert not result.l3.baseline_available.any()
    assert result.l3.z_max.isna().all()
    assert (result.l3.n_z_valid == 0).all()
    assert 'MISSING_BASELINE' in {i['code'] for i in result.issues}


def test_live_baseline_missing_category_never_pooled():
    data = demo_datasets()[:2]
    data[1].records['category'] = 'TEST_ONLY'
    result = analyze(data, baseline_ids=[data[0].image_id])
    rows = result.l3[result.l3.category == 'TEST_ONLY']
    assert rows.z_max.isna().all()
    assert not rows.baseline_available.any()


def test_baseline_identity_params_and_duplicates(calibrated):
    with pytest.raises(ValueError, match='파라미터'):
        analyze(demo_datasets(), params=replace(cdqc.Params(), grad_sigma_px=2),
                frozen_stats=import_stats(export_stats(calibrated)))
    data = demo_datasets()
    with pytest.raises(ValueError, match='중복'):
        analyze([data[0], data[0]])
    with pytest.raises(ValueError, match='정상 기준'):
        analyze(data, baseline_ids=[])


def test_profile_exact_core_sampling_and_gradient():
    dataset = demo_datasets()[4]
    params = cdqc.Params()
    records = dataset.records
    s, e = records[['sx', 'sy']].to_numpy(), records[['ex', 'ey']].to_numpy()
    tangents = mean_edge_tangent(unit_tangents(s), unit_tangents(e))
    profile = sample_ribbon_profiles(dataset.img, s, e, tangents, params.view()['sampling'])[7]
    shown = profile_table(dataset, 'demo_CD', 7, params)
    np.testing.assert_array_equal(shown.intensity, profile.p)
    np.testing.assert_array_equal(shown.gradient, gradient_of(profile, params.grad_sigma_px))
    assert shown.attrs['e_endpoint_px'] == profile.L


def test_diagnostics_reference_matches_stats(calibrated):
    shown = feature_diagnostics(calibrated, 0).set_index('feature')
    ref = calibrated.stats['l3']['demo_CD']['features']['cd_nm']
    assert shown.loc['cd_nm', 'reference_median'] == ref['median']
    assert shown.loc['cd_nm', 'reference_n'] == 48
    assert shown.loc['cd_nm', 'raw'] == calibrated.l3.loc[0, 'cd_nm']


@pytest.mark.parametrize('change', [{'grad_sigma_px': -1}, {'local_window': 4},
    {'trim_frac': .5}, {'delta_sigma_sweep': []}, {'z_cap': float('nan')},
    {'ribbon_half_w': 2.5}, {'method': 'invalid'}, {'win_min_px': 1000}])
def test_invalid_params(change):
    with pytest.raises(ValueError):
        validate_params(change)


def test_json_params_roundtrip():
    params = validate_params(json.loads(json.dumps(asdict(cdqc.Params()))))
    assert params == cdqc.Params()


def test_all_invalid_rows_keep_overlay_columns(tmp_path):
    rec = pd.DataFrame(dict(sx=[np.nan], sy=[2], ex=[5], ey=[2], category=['A']))
    image, table = files(tmp_path, rec)
    result = analyze([load_dataset(image, table, px_nm=.5)])
    assert {'s_x', 's_y', 'e_x', 'e_y', 'px_nm', 'z_max', 'n_z_valid'} <= set(result.l3)
    assert result.l3.z_max.isna().all()
    assert result.l3.px_nm.iloc[0] == .5


def test_dm_scale_axes_and_resize(monkeypatch):
    dm = pytest.importorskip('ncempy.io.dm')
    from cdqc_workbench.backend import _dm_scale
    monkeypatch.setattr(dm, 'dmReader', lambda _: dict(data=np.zeros((100, 200)),
        pixelSize=[.25, .25], pixelUnit=['nm', 'nm']))
    scale, meta = _dm_scale('dummy.dm3', (50, 100))
    assert scale == .5
    assert meta['dm_dataset_index'] == 0
    with pytest.raises(ValueError, match='비등방'):
        _dm_scale('dummy.dm3', (50, 200))


def test_floor_diagnostics_preserve_original_stats(calibrated):
    shown = feature_diagnostics(calibrated, 0).set_index('feature')
    assert shown.loc['cd_nm', 'reference_valid_n'] == 48
    st = calibrated.stats['l3']['demo_CD']
    assert shown.loc['cd_nm', 'raw_reference_mad'] == st['raw_features']['cd_nm']['mad']
    assert shown.loc['cd_nm', 'reference_mad'] >= shown.loc['cd_nm', 'raw_reference_mad']
