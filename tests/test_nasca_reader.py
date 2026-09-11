"""Contract tests for Excel COM; these do not claim an installed Excel run."""
import builtins
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from cdqc.func import read_nasca_csv


@pytest.fixture
def fake_excel(monkeypatch, tmp_path):
    path = tmp_path / '한글 원본_Result.xlsx'
    # Intentionally not a ZIP/XLSX: Excel is the only reader of protected input.
    path.write_bytes(b'protected workbook opened by Excel only')
    used = SimpleNamespace(Row=4, Column=2, Rows=SimpleNamespace(Count=3),
        Columns=SimpleNamespace(Count=2), Value2=(('category', 'value'), ('폭', 2.5), ('두께', None)))
    worksheet = SimpleNamespace(Name='NASCA 결과', UsedRange=used)
    workbook = SimpleNamespace(Worksheets=Mock(return_value=worksheet), Close=Mock())
    excel = SimpleNamespace(Workbooks=SimpleNamespace(Open=Mock(return_value=workbook)), Quit=Mock())
    pythoncom = ModuleType('pythoncom')
    pythoncom.COINIT_APARTMENTTHREADED = 2
    pythoncom.CoInitializeEx = Mock()
    pythoncom.CoUninitialize = Mock()
    client = ModuleType('win32com.client')
    client.DispatchEx = Mock(return_value=excel)
    win32com = ModuleType('win32com')
    win32com.client = client
    for name, module in [('pythoncom', pythoncom), ('win32com', win32com), ('win32com.client', client)]:
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(path=path, used=used, worksheet=worksheet, workbook=workbook,
                           excel=excel, pythoncom=pythoncom, client=client)


def test_original_path_hidden_private_excel_and_metadata(fake_excel, monkeypatch):
    f = fake_excel
    def no_raw_read(*args, **kwargs):
        pytest.fail('Workbook must be opened by Excel from the original path')
    monkeypatch.setattr(type(f.path), 'read_bytes', no_raw_read)
    monkeypatch.setattr(pd, 'read_excel', no_raw_read)
    frame = read_nasca_csv(f.path)
    pd.testing.assert_frame_equal(frame, pd.DataFrame({'category': ['폭', '두께'], 'value': [2.5, None]}))
    f.client.DispatchEx.assert_called_once_with('Excel.Application')
    assert f.excel.Visible is False and f.excel.DisplayAlerts is False
    assert f.excel.AskToUpdateLinks is False and f.excel.EnableEvents is False
    assert f.excel.AutomationSecurity == 3
    f.excel.Workbooks.Open.assert_called_once_with(Filename=str(f.path.absolute()),
        UpdateLinks=0, ReadOnly=True, IgnoreReadOnlyRecommended=True, AddToMru=False, Notify=False)
    f.workbook.Worksheets.assert_called_once_with(1)
    assert frame.attrs['sheet_name'] == 'NASCA 결과'
    assert frame.attrs['source_data_start_row'] == 5
    assert frame.attrs['used_range_start_column'] == 2
    assert frame.attrs['source_path'] == str(f.path.absolute())
    f.workbook.Close.assert_called_once_with(SaveChanges=False)
    f.excel.Quit.assert_called_once_with()
    f.pythoncom.CoInitializeEx.assert_called_once_with(2)
    f.pythoncom.CoUninitialize.assert_called_once_with()


@pytest.mark.parametrize(('value', 'shape', 'header', 'expected'), [
    (None, (1, 1), True, pd.DataFrame()),
    (((None, None), (None, None)), (2, 2), True, pd.DataFrame()),
    ('single', (1, 1), False, pd.DataFrame([['single']])),
    ('header', (1, 1), True, pd.DataFrame(columns=['header'])),
    ((('name', 'value'),), (1, 2), True, pd.DataFrame(columns=['name', 'value'])),
    ((('name',), ('폭',), ('두께',)), (3, 1), True, pd.DataFrame({'name': ['폭', '두께']})),
    ((('x', 'y'), (1, 2)), (2, 2), False, pd.DataFrame([['x', 'y'], [1, 2]])),
    ((('x', 'y'), (None, None), (1, 2)), (3, 2), True, pd.DataFrame({'x': [None, 1], 'y': [None, 2]})),
    (('x', 'y'), (1, 2), False, pd.DataFrame([['x', 'y']])),
    (('x', 3), (2, 1), False, pd.DataFrame([['x'], [3]])),
])
def test_value_shapes_and_header_no_lost_rows(fake_excel, value, shape, header, expected):
    f = fake_excel
    f.used.Value2 = value
    f.used.Rows.Count, f.used.Columns.Count = shape
    result = read_nasca_csv(f.path, visible=True, header=header)
    pd.testing.assert_frame_equal(result, expected)
    assert f.excel.Visible is True
    assert result.attrs['source_data_start_row'] == 4 + int(header)


@pytest.mark.parametrize('stage', ['initialize', 'dispatch', 'open', 'read', 'dataframe'])
def test_failure_always_cleans_only_owned_excel(fake_excel, monkeypatch, stage):
    f = fake_excel
    fault = RuntimeError('intentional ' + stage)
    if stage == 'initialize':
        f.pythoncom.CoInitializeEx.side_effect = fault
    elif stage == 'dispatch':
        f.client.DispatchEx.side_effect = fault
    elif stage == 'open':
        f.excel.Workbooks.Open.side_effect = fault
    elif stage == 'read':
        f.workbook.Worksheets.side_effect = fault
    else:
        monkeypatch.setattr(pd, 'DataFrame', Mock(side_effect=fault))
    with pytest.raises(RuntimeError, match='Excel COM') as caught:
        read_nasca_csv(f.path)
    assert caught.value.__cause__ is fault
    assert f.pythoncom.CoUninitialize.call_count == int(stage != 'initialize')
    assert f.excel.Quit.call_count == int(stage not in ('initialize', 'dispatch'))
    assert f.workbook.Close.call_count == int(stage in ('read', 'dataframe'))


@pytest.mark.parametrize('action', ['close', 'quit'])
def test_cleanup_failure_still_uninitializes_and_reports(fake_excel, action):
    f = fake_excel
    target = f.workbook.Close if action == 'close' else f.excel.Quit
    target.side_effect = RuntimeError('cleanup fault')
    with pytest.raises(RuntimeError, match='정리 실패'):
        read_nasca_csv(f.path)
    f.excel.Quit.assert_called_once()
    f.pythoncom.CoUninitialize.assert_called_once()


def test_read_failure_not_masked_by_cleanup_failure(fake_excel):
    f = fake_excel
    f.workbook.Worksheets.side_effect = ValueError('read fault')
    f.workbook.Close.side_effect = RuntimeError('close fault')
    with pytest.raises(RuntimeError, match='read fault') as caught:
        read_nasca_csv(f.path)
    assert 'close fault' in str(caught.value.__cause__.__notes__)
    f.excel.Quit.assert_called_once()
    f.pythoncom.CoUninitialize.assert_called_once()


def test_missing_file_never_starts_excel(fake_excel):
    with pytest.raises(FileNotFoundError):
        read_nasca_csv(fake_excel.path.with_name('missing.xlsx'))
    fake_excel.client.DispatchEx.assert_not_called()
    fake_excel.pythoncom.CoInitializeEx.assert_not_called()


def test_missing_pywin32_gives_install_action(fake_excel, monkeypatch):
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'pythoncom':
            raise ImportError('no pythoncom')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(RuntimeError, match='pip install pywin32'):
        read_nasca_csv(fake_excel.path)
    fake_excel.client.DispatchEx.assert_not_called()


def test_worker_thread_initializes_and_releases_com(fake_excel):
    import threading
    thread_events = []
    fake_excel.pythoncom.CoInitializeEx.side_effect = lambda _: thread_events.append(('init', threading.get_ident()))
    fake_excel.pythoncom.CoUninitialize.side_effect = lambda: thread_events.append(('uninit', threading.get_ident()))
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(read_nasca_csv, fake_excel.path).result(timeout=10)
    assert len(result) == 2
    assert thread_events[0][0] == 'init' and thread_events[1][0] == 'uninit'
    assert thread_events[0][1] == thread_events[1][1] != threading.get_ident()


def test_legacy_batch_uses_same_original_path_com_reader(fake_excel, monkeypatch):
    from scripts import run_batch
    from cdqc import func
    expected = pd.DataFrame({'category': ['폭']})
    reader = Mock(return_value=expected)
    monkeypatch.setattr(func, 'read_nasca_csv', reader)
    monkeypatch.setattr(run_batch, 'read_px_nm', lambda _: (.5, (10, 10)))
    monkeypatch.setattr(run_batch.cv2, 'imread', lambda *args: np.zeros((10, 10), np.uint8))
    _, _, actual, scale = run_batch.load_one('image.dm3', 'image.tif', None, fake_excel.path)
    assert actual is expected and scale == .5
    reader.assert_called_once_with(fake_excel.path, visible=False, header=True)


def test_real_excel_worker_roundtrip_when_installed(tmp_path):
    """A real COM run, explicitly skipped when the host lacks desktop Excel."""
    if sys.platform != 'win32':
        pytest.skip('Real Excel COM integration requires Windows and desktop Excel')
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'Excel.Application\CLSID'):
            pass
    except FileNotFoundError:
        pytest.skip('Desktop Excel is not installed/registered; fake COM tests are separate')
    pythoncom = pytest.importorskip('pythoncom', reason='Real Excel integration requires pywin32')
    client = pytest.importorskip('win32com.client')
    path = tmp_path / '실제 Excel 검증.xlsx'
    app = workbook = sheet = None
    pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
    try:
        app = client.DispatchEx('Excel.Application')
        app.Visible = False
        app.DisplayAlerts = False
        workbook = app.Workbooks.Add()
        sheet = workbook.Worksheets(1)
        sheet.Name = 'NASCA'
        sheet.Range('B4:C6').Value2 = (('category', 'value'), ('폭', 2.5), ('두께', 7))
        workbook.SaveAs(str(path), FileFormat=51)
        workbook.Close(SaveChanges=False)
        sheet = workbook = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(read_nasca_csv, path).result(timeout=60)
        pd.testing.assert_frame_equal(result, pd.DataFrame({'category': ['폭', '두께'], 'value': [2.5, 7]}))
        assert result.attrs['sheet_name'] == 'NASCA'
        assert result.attrs['source_data_start_row'] == 5
        # Reader shutdown must leave even another private Excel session alive.
        assert app.Workbooks.Count == 0
    finally:
        sheet = None
        try:
            if workbook is not None:
                workbook.Close(SaveChanges=False)
        finally:
            workbook = None
            try:
                if app is not None:
                    app.Quit()
            finally:
                app = None
                pythoncom.CoUninitialize()
