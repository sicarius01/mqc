"""NASCA measurement-table input through a private Microsoft Excel instance.

Excel must open the original file: copying its bytes to a temporary workbook or
using a ZIP/XML reader does not work for some company-protected NASCA exports.
The optional Windows dependency is imported only when the reader is called.
"""
from pathlib import Path

import pandas as pd


def _range_rows(value, row_count, column_count):
    """Normalize the scalar or rectangular SAFEARRAY returned by Excel."""
    if value is None:
        return []
    if not isinstance(value, (tuple, list)):
        return [[value]]
    if not value:
        return []
    if isinstance(value[0], (tuple, list)):
        return [list(row) for row in value]
    if column_count == 1:
        return [[cell] for cell in value]
    if row_count == 1:
        return [list(value)]
    raise ValueError('Excel UsedRange returned a non-rectangular value array.')


def read_nasca_csv(csv_path, visible=False, header=True):
    """Read the first worksheet via Excel COM and return a pandas DataFrame.

    The historical name is retained for callers; ``csv_path`` can be an XLSX
    path. ``header=True`` uses the first UsedRange row as column names, while
    ``False`` keeps every row. Empty cells and intervening blank rows are kept.
    Value2 reads evaluated cell values (Excel dates remain numeric serials).

    ``DataFrame.attrs`` records the sheet, used range and actual Excel row of
    the first data record, so downstream diagnostics can locate source cells.
    Every call owns its COM initialization and DispatchEx application. Existing
    interactive Excel sessions are never reused, closed or terminated.
    """
    path = Path(csv_path).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f'측정 파일이 없습니다: {path}')
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            'Excel COM 읽기에는 Windows, 설치된 Microsoft Excel 및 pywin32가 필요합니다. '
            '현재 Python 환경에서 python -m pip install pywin32를 실행하세요.'
        ) from exc

    excel = workbook = worksheet = used_range = None
    initialized = False
    failure = None
    cleanup_errors = []
    stage = 'COM 초기화'
    try:
        # Streamlit executes application code on a worker thread.
        # Unlike CoInitialize, CoInitializeEx reports RPC_E_CHANGED_MODE, so we
        # never unbalance an apartment that this call did not initialize.
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
        initialized = True
        stage = 'Microsoft Excel 실행 (설치/COM 등록 확인)'
        excel = win32com.client.DispatchEx('Excel.Application')
        excel.Visible = bool(visible)
        excel.DisplayAlerts = False
        excel.AskToUpdateLinks = False
        excel.EnableEvents = False
        excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        stage = '원본 파일 열기 (Excel 열기 권한/보안 모듈 확인)'
        workbook = excel.Workbooks.Open(
            Filename=str(path), UpdateLinks=0, ReadOnly=True,
            IgnoreReadOnlyRecommended=True, AddToMru=False, Notify=False,
        )
        stage = '첫 워크시트 UsedRange 읽기'
        worksheet = workbook.Worksheets(1)
        used_range = worksheet.UsedRange
        start_row, start_column = int(used_range.Row), int(used_range.Column)
        row_count = int(used_range.Rows.Count)
        column_count = int(used_range.Columns.Count)
        sheet_name = str(worksheet.Name)
        rows = _range_rows(used_range.Value2, row_count, column_count)
        if rows and all(cell is None for row in rows for cell in row):
            rows = []
        stage = 'DataFrame 변환'
        frame = (pd.DataFrame(rows[1:], columns=rows[0])
                 if header and rows else pd.DataFrame(rows))
        frame.attrs.update(
            table_reader='cdqc.func.read_nasca_csv / Excel COM DispatchEx',
            source_path=str(path), sheet_name=sheet_name,
            used_range_start_row=start_row, used_range_start_column=start_column,
            used_range_row_count=row_count, used_range_column_count=column_count,
            header=bool(header), source_data_start_row=start_row + int(bool(header)),
            read_options={'visible': bool(visible), 'read_only': True,
                          'update_links': 0, 'worksheet_index': 1, 'value_property': 'Value2'},
        )
    except BaseException as exc:
        failure = exc
        if not isinstance(exc, Exception):
            raise
        raise RuntimeError(f'Excel COM {stage} 실패: {path}\n{exc}') from exc
    finally:
        # Release child proxies before the apartment is uninitialized.
        used_range = worksheet = None
        if workbook is not None:
            try:
                workbook.Close(SaveChanges=False)
            except Exception as exc:
                cleanup_errors.append(f'Workbook.Close: {exc}')
        workbook = None
        if excel is not None:
            try:
                excel.Quit()
            except Exception as exc:
                cleanup_errors.append(f'Excel.Quit: {exc}')
        excel = None
        if initialized:
            try:
                pythoncom.CoUninitialize()
            except Exception as exc:
                cleanup_errors.append(f'COM 해제: {exc}')
        if cleanup_errors:
            message = 'Excel COM 정리 실패: ' + '; '.join(cleanup_errors)
            if failure is not None:
                failure.add_note(message)
            else:
                raise RuntimeError(message)
    return frame
